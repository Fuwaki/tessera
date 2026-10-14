//! Graph IR → planner primitive graph (`PlanInput`) lowering.
//!
//! The planner deliberately consumes a typed, lowered primitive graph
//! ([`crate::planner::PlanInput`]) and never parses a manifest. This module is
//! the one adapter between the two, and it is the only place that knows how a
//! [`crate::graph::Graph`] maps onto planner tensors and operators.
//!
//! Mapping rules (each one explicit, none inferred):
//!
//! * Every graph point becomes one activation tensor per *version*. The graph
//!   updates points in place (`add`, `upsample_nearest2x`), so a new tensor id
//!   is allocated for every operator output and the point's "current" tensor is
//!   updated. Tensor names are `p{point}.v{version}`.
//! * Every convolution layer gets a weight tensor named
//!   `weights:{layer}` (see [`weight_tensor_name`]) and, when it has a non-zero
//!   int32 bias, a bias tensor named `bias:{layer}`. A bias vector that is
//!   exactly zero is dropped, because adding zero is the identity; anything
//!   else is preserved so the planner and the RTL backend see it and can
//!   reject it rather than silently miscompile.
//! * The manifest requantization vector is serialized once, here, into the
//!   fixed-point `(M0, shift)` form the hardware model uses
//!   ([`crate::mlir::quant::serialize_requant`]). Per-output-channel
//!   multipliers are carried verbatim into `Requant::per_channel_multipliers`
//!   so the layout artifact preserves them.
//! * `relu` graph ops carry no state change (the fused activation is already on
//!   the convolution layer) and are skipped; the report records that.
//! * `add` is lowered to a two-input elementwise op whose requantization factor
//!   is `scale(source_point) / scale(point)`, read from the graph's point
//!   scales rather than from the op's `scale` field (which the manifest
//!   contract defines as the *destination* point scale).
//! * `upsample_nearest2x` is lowered with scale 2 and a doubled output shape.
//! * `quantize_u8_output` is lowered to an elementwise op whose activation is
//!   the unsigned output clamp `[0, output_qmax]`.

use crate::graph::{
    Graph, GraphError, GraphOp, Layer, LayerKind, Padding as GraphPadding, PointRef,
};
use crate::layout::{
    Activation, ActivationKind, ElemType, OpKind, Requant, Rounding, TensorId, TensorShape,
};
use crate::mlir::quant::{ClampSpec, QuantError, serialize_requant};
use crate::planner::{
    ConvDetail, ElementwiseDetail, OpDetail, Padding as PlanPadding, PlanInput, PrimitiveOp,
    TensorDesc, UpsampleDetail,
};
use std::collections::BTreeMap;
use std::fmt;

/// Name of the planner tensor that carries a layer's weights.
///
/// [`crate::weights::GraphWeightResolver`] resolves weight bytes by this exact
/// name, so both sides share this function instead of duplicating a convention.
pub fn weight_tensor_name(layer: &str) -> String {
    format!("weights:{layer}")
}

/// Name of the planner tensor that carries a layer's int32 bias.
pub fn bias_tensor_name(layer: &str) -> String {
    format!("bias:{layer}")
}

/// Concrete input activation shape, in channels/height/width order.
///
/// The manifest carries no spatial extents, so shape propagation needs one
/// concrete input tensor. Batch is fixed at 1 by the v1 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputShape {
    /// Input channels.
    pub channels: u32,
    /// Input height.
    pub height: u32,
    /// Input width.
    pub width: u32,
}

impl InputShape {
    /// Build a validated input shape.
    pub fn new(channels: u32, height: u32, width: u32) -> Result<Self, LowerError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(LowerError::Invalid {
                detail: format!(
                    "input shape must have non-zero channels/height/width, got \
                     {channels}x{height}x{width}"
                ),
            });
        }
        Ok(InputShape {
            channels,
            height,
            width,
        })
    }

    /// Parse `N,H,W,C` (NHWC), as used by the MLIR layer, requiring `N == 1`.
    pub fn parse_nhwc(text: &str) -> Result<Self, LowerError> {
        let parts: Vec<&str> = text.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(LowerError::Invalid {
                detail: format!(
                    "input shape {text:?} must have four comma-separated components N,H,W,C"
                ),
            });
        }
        let parse = |name: &str, raw: &str| -> Result<u32, LowerError> {
            raw.parse::<u32>().map_err(|_| LowerError::Invalid {
                detail: format!("input shape {name}={raw:?} is not a non-negative integer"),
            })
        };
        let n = parse("N", parts[0])?;
        if n != 1 {
            return Err(LowerError::Invalid {
                detail: format!("v1 supports batch 1 only, got N={n}"),
            });
        }
        InputShape::new(
            parse("C", parts[3])?,
            parse("H", parts[1])?,
            parse("W", parts[2])?,
        )
    }

    /// The same shape in the `[n, h, w, c]` order the MLIR emitter uses.
    pub fn to_nhwc(self) -> [u64; 4] {
        [
            1,
            u64::from(self.height),
            u64::from(self.width),
            u64::from(self.channels),
        ]
    }
}

/// Lowering failure: either the graph itself is unusable at this input shape,
/// or a quantization contract cannot be serialized.
#[derive(Debug)]
pub enum LowerError {
    /// Shape propagation or an input-shape parse failure.
    Shape { detail: String },
    /// The manifest requantization vector cannot be serialized.
    Quant { layer: String, source: QuantError },
    /// The graph contains a construct this adapter cannot lower.
    Invalid { detail: String },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::Shape { detail } => write!(f, "lower: {detail}"),
            LowerError::Quant { layer, source } => {
                write!(f, "lower: layer {layer}: requantization: {source}")
            }
            LowerError::Invalid { detail } => write!(f, "lower: {detail}"),
        }
    }
}

impl std::error::Error for LowerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LowerError::Quant { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<GraphError> for LowerError {
    fn from(error: GraphError) -> Self {
        LowerError::Shape {
            detail: error.to_string(),
        }
    }
}

/// Decisions the adapter made while lowering, so nothing is silent.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LowerReport {
    /// Human-readable, deterministic notes about the lowering.
    pub notes: Vec<String>,
    /// Layer name → weight tensor name, for inspection and weight resolution.
    pub weight_tensors: BTreeMap<String, String>,
}

/// Lower a validated graph into the planner's primitive graph.
pub fn lower_graph(
    graph: &Graph,
    input: InputShape,
) -> Result<(PlanInput, LowerReport), LowerError> {
    let shape = crate::graph::TensorShape::try_new(input.channels, input.height, input.width)
        .map_err(|error| LowerError::Shape {
            detail: error.to_string(),
        })?;
    // The graph's shape report is the graph's own propagation, used here as an
    // independent cross-check of this module's per-version shape arithmetic.
    // It is *not* used to size operator outputs: the graph updates points in
    // place (`add`, `upsample_nearest2x`), so a point has one shape per version
    // and only the op that writes it knows which version is live.
    let report = graph.infer_shapes(&shape)?;

    let mut builder = Builder::new(graph, input);
    for op in graph.ops() {
        builder.lower_op(op)?;
    }
    let (plan, report_out) = builder.finish()?;
    let computed_image = report_out.image_shape;
    let graph_image = report.image;
    if computed_image
        != (
            graph_image.channels.get(),
            graph_image.height.get(),
            graph_image.width.get(),
        )
    {
        return Err(LowerError::Shape {
            detail: format!(
                "lowered image shape {}x{}x{} does not match the graph's own propagation {}x{}x{}",
                computed_image.0,
                computed_image.1,
                computed_image.2,
                graph_image.channels.get(),
                graph_image.height.get(),
                graph_image.width.get()
            ),
        });
    }
    Ok((plan, report_out.report))
}

struct Builder<'a> {
    graph: &'a Graph,
    tensors: Vec<TensorDesc>,
    /// Current tensor id of each point.
    current: Vec<Option<TensorId>>,
    /// Number of tensors already written for each point.
    versions: Vec<u32>,
    /// Tensor holding the unsigned image, once a head layer has produced it.
    image: Option<TensorId>,
    /// Shape of the unsigned image, for the cross-check against the graph.
    image_shape: Option<(u32, u32, u32)>,
    ops: Vec<PrimitiveOp>,
    report: LowerReport,
}

/// Result of lowering, with the image shape for cross-checking.
struct Lowered {
    report: LowerReport,
    image_shape: (u32, u32, u32),
}

impl<'a> Builder<'a> {
    fn new(graph: &'a Graph, input: InputShape) -> Self {
        let mut builder = Builder {
            graph,
            tensors: Vec::new(),
            current: vec![None; graph.point_count()],
            versions: vec![0; graph.point_count()],
            image: None,
            image_shape: None,
            ops: Vec::new(),
            report: LowerReport::default(),
        };
        let input_point = graph.input_point().index();
        let tensor = builder.push_tensor(
            format!("p{input_point}.v0"),
            TensorShape::new(1, input.channels, input.height, input.width),
            ElemType::I8,
        );
        builder.current[input_point] = Some(tensor);
        builder
    }

    fn push_tensor(&mut self, name: String, shape: TensorShape, elem: ElemType) -> TensorId {
        let id = TensorId::new(self.tensors.len() as u32);
        self.tensors.push(TensorDesc {
            id,
            name,
            shape,
            elem,
        });
        id
    }

    fn point_tensor(
        &self,
        point: crate::graph::PointId,
        what: &str,
    ) -> Result<TensorId, LowerError> {
        self.current
            .get(point.index())
            .copied()
            .flatten()
            .ok_or_else(|| LowerError::Invalid {
                detail: format!(
                    "{what}: point {} has no value yet; the graph is not topologically ordered",
                    point.get()
                ),
            })
    }

    fn layer(&self, name: &str) -> Result<&'a Layer, LowerError> {
        self.graph
            .layer_named(name)
            .ok_or_else(|| LowerError::Invalid {
                detail: format!("graph references unknown layer {name:?}"),
            })
    }

    fn lower_op(&mut self, op: &GraphOp) -> Result<(), LowerError> {
        match op {
            GraphOp::Conv {
                layer,
                in_point,
                out_point,
            } => self.lower_conv(self.layer(layer.as_str())?, *in_point, *out_point),
            GraphOp::Relu { layer } => {
                self.report.notes.push(format!(
                    "layer {}: `relu` graph op skipped; the fused activation is carried by the \
                     convolution op's activation field",
                    layer
                ));
                Ok(())
            }
            GraphOp::Add {
                point,
                source_point,
                ..
            } => self.lower_add(*point, *source_point),
            GraphOp::UpsampleNearest2x { point } => self.lower_upsample(*point),
            GraphOp::QuantizeU8Output { .. } => self.lower_quantize(),
        }
    }

    fn lower_conv(
        &mut self,
        layer: &'a Layer,
        in_point: crate::graph::PointId,
        out_point: PointRef,
    ) -> Result<(), LowerError> {
        let name = layer.name.as_str().to_string();
        let input = self.point_tensor(in_point, "conv")?;
        let in_shape = self.tensors[input.get() as usize].shape;
        let (out_h, out_w) = conv_out_spatial(in_shape.h, in_shape.w, layer)?;
        let out_shape = TensorShape::new(1, layer.out_channels.get(), out_h, out_w);

        let kind = match layer.kind {
            LayerKind::Conv2d => OpKind::Conv2d,
            LayerKind::DepthwiseConv2d => OpKind::DepthwiseConv2d,
            LayerKind::PointwiseConv2d => OpKind::PointwiseConv2d,
        };
        let weight_elem = match layer.weight_bits.get() {
            8 => ElemType::I8,
            4 => ElemType::I4,
            bits => {
                return Err(LowerError::Invalid {
                    detail: format!(
                        "layer {name}: weight width {bits} is outside the v1 set {{4, 8}}"
                    ),
                });
            }
        };
        // The planner reads a convolution's kernel from the weight tensor's
        // spatial extents, so the tensor is `(n, c, kh, kw)`: `n` is the output
        // channel count, `c` is one channel per group (`1` for depthwise).
        let in_per_group = layer.in_channels.get() / layer.groups.get().max(1);
        let channels_per_group = if kind == OpKind::DepthwiseConv2d {
            1
        } else {
            in_per_group
        };
        let weight_shape = TensorShape {
            n: layer.out_channels.get(),
            c: channels_per_group,
            h: layer.kernel.h.get(),
            w: layer.kernel.w.get(),
        };
        let weight_name = weight_tensor_name(&name);
        let weights = self.push_tensor(weight_name.clone(), weight_shape, weight_elem);
        self.report.weight_tensors.insert(name.clone(), weight_name);

        let bias = if layer.bias_int32.iter().any(|value| *value != 0) {
            Some(self.push_tensor(
                bias_tensor_name(&name),
                // The planner models a bias as a per-output-channel vector:
                // `n == out_channels`, then three unit extents.
                TensorShape::new(layer.out_channels.get(), 1, 1, 1),
                ElemType::I32,
            ))
        } else {
            self.report.notes.push(format!(
                "layer {name}: int32 bias is all zero and was dropped (exact identity)"
            ));
            None
        };

        let activation = activation_of(self.graph, layer)?;
        let requant = requant_of(self.graph, layer, &name)?;
        let padding = padding_of(layer, &name)?;

        let output = self.write_point(out_point, out_shape, output_elem(layer));
        if layer.out_unsigned {
            // An unsigned head produces the image whether it writes the sentinel
            // or a real point; the graph IR allows both, and the last unsigned
            // layer wins, exactly as the graph's own shape propagation decides.
            self.image = Some(output);
            self.image_shape = Some((out_shape.c, out_shape.h, out_shape.w));
        }
        self.ops.push(PrimitiveOp {
            name: name.clone(),
            kind,
            inputs: vec![input],
            outputs: vec![output],
            detail: OpDetail::Conv(ConvDetail {
                weights,
                bias,
                groups: layer.groups.get(),
                stride: [layer.stride.h.get(), layer.stride.w.get()],
                padding,
                requant,
                activation,
            }),
        });
        Ok(())
    }

    fn lower_add(
        &mut self,
        point: crate::graph::PointId,
        source_point: crate::graph::PointId,
    ) -> Result<(), LowerError> {
        let name = format!("add.{}", point.get());
        let destination = self.point_tensor(point, "add")?;
        let source = self.point_tensor(source_point, "add")?;
        let shape = self.tensors[destination.get() as usize].shape;
        let source_shape = self.tensors[source.get() as usize].shape;
        if source_shape != shape {
            return Err(LowerError::Invalid {
                detail: format!(
                    "add on point {}: operand shapes differ ({:?} vs {:?})",
                    point.get(),
                    shape,
                    source_shape
                ),
            });
        }
        let dst_scale = self
            .graph
            .point_scale(point)
            .ok_or_else(|| LowerError::Invalid {
                detail: format!("add: point {} has no scale", point.get()),
            })?;
        let src_scale =
            self.graph
                .point_scale(source_point)
                .ok_or_else(|| LowerError::Invalid {
                    detail: format!("add: source point {} has no scale", source_point.get()),
                })?;
        let factor = src_scale / dst_scale;
        let act_qmax = self.graph.act_qmax();
        let (spec, error) = serialize_requant(&[factor], 1, ClampSpec::symmetric_branch(act_qmax))
            .map_err(|source| LowerError::Quant {
                layer: name.clone(),
                source,
            })?;
        let requant = Requant {
            m0: i64::from(spec.multiplier[0]),
            shift: spec.shift as i32,
            rounding: Rounding::HalfUp,
            saturate_lo: spec.clamp.lower as i32,
            saturate_hi: spec.clamp.upper as i32,
            per_channel_multipliers: Vec::new(),
            multiplier_error_ppm: Some(ppm(error)),
        };
        self.report.notes.push(format!(
            "add on point {}: branch requant factor {src_scale}/{dst_scale} serialized to \
             M0={} shift={} (worst relative error {error:.3e})",
            point.get(),
            spec.multiplier[0],
            spec.shift
        ));
        let output = self.write_point(PointRef::Point(point), shape, ElemType::I8);
        self.ops.push(PrimitiveOp {
            name,
            kind: OpKind::Add,
            inputs: vec![destination, source],
            outputs: vec![output],
            detail: OpDetail::Elementwise(ElementwiseDetail {
                requant: Some(requant),
                activation: Activation::identity(-(act_qmax as i32), act_qmax as i32),
            }),
        });
        Ok(())
    }

    fn lower_upsample(&mut self, point: crate::graph::PointId) -> Result<(), LowerError> {
        let name = format!("upsample_nearest2x.{}", point.get());
        let input = self.point_tensor(point, "upsample")?;
        let in_shape = self.tensors[input.get() as usize].shape;
        let out_shape = TensorShape {
            n: 1,
            c: in_shape.c,
            h: in_shape
                .h
                .checked_mul(2)
                .ok_or_else(|| LowerError::Invalid {
                    detail: format!("upsample on point {}: height overflow", point.get()),
                })?,
            w: in_shape
                .w
                .checked_mul(2)
                .ok_or_else(|| LowerError::Invalid {
                    detail: format!("upsample on point {}: width overflow", point.get()),
                })?,
        };
        let output = self.write_point(PointRef::Point(point), out_shape, ElemType::I8);
        self.ops.push(PrimitiveOp {
            name,
            kind: OpKind::NearestUpsample2x,
            inputs: vec![input],
            outputs: vec![output],
            detail: OpDetail::Upsample(UpsampleDetail { scale: 2 }),
        });
        Ok(())
    }

    fn lower_quantize(&mut self) -> Result<(), LowerError> {
        let image = self.image.ok_or_else(|| LowerError::Invalid {
            detail: "quantize_u8_output: no operator produced the image".to_string(),
        })?;
        let shape = self.tensors[image.get() as usize].shape;
        let output = self.push_tensor("output.u8".to_string(), shape, ElemType::U8);
        self.ops.push(PrimitiveOp {
            name: "quantize_u8_output".to_string(),
            kind: OpKind::QuantizeOutput,
            inputs: vec![image],
            outputs: vec![output],
            detail: OpDetail::Elementwise(ElementwiseDetail {
                requant: None,
                activation: Activation {
                    kind: ActivationKind::UnsignedClamp,
                    clamp_lo: 0,
                    clamp_hi: self.graph.output_qmax() as i32,
                },
            }),
        });
        Ok(())
    }

    /// Allocate the next tensor for a point (or for the image sentinel) and make
    /// it the point's current value.
    fn write_point(&mut self, target: PointRef, shape: TensorShape, elem: ElemType) -> TensorId {
        match target {
            PointRef::Point(id) => {
                let version = self.versions[id.index()];
                self.versions[id.index()] += 1;
                let tensor = self.push_tensor(format!("p{}.v{version}", id.get()), shape, elem);
                self.current[id.index()] = Some(tensor);
                tensor
            }
            PointRef::ImageSentinel => {
                let tensor = self.push_tensor("image.v0".to_string(), shape, elem);
                self.image = Some(tensor);
                self.image_shape = Some((shape.c, shape.h, shape.w));
                tensor
            }
        }
    }

    fn finish(self) -> Result<(PlanInput, Lowered), LowerError> {
        let output = self
            .ops
            .last()
            .and_then(|op| op.outputs.first().copied())
            .ok_or_else(|| LowerError::Invalid {
                detail: "graph lowered to no operators".to_string(),
            })?;
        let input = self.current[0].ok_or_else(|| LowerError::Invalid {
            detail: "graph has no input tensor".to_string(),
        })?;
        let name = if self.graph.arch().trim().is_empty() {
            "tessera.graph".to_string()
        } else {
            self.graph.arch().to_string()
        };
        let plan = PlanInput {
            name,
            tensors: self.tensors,
            ops: self.ops,
            inputs: vec![input],
            outputs: vec![output],
        };
        plan.validate().map_err(|detail| LowerError::Invalid {
            detail: format!("lowered primitive graph is invalid: {detail}"),
        })?;
        let image_shape = self.image_shape.ok_or_else(|| LowerError::Invalid {
            detail: "the graph produced no unsigned image".to_string(),
        })?;
        Ok((
            plan,
            Lowered {
                report: self.report,
                image_shape,
            },
        ))
    }
}

fn output_elem(layer: &Layer) -> ElemType {
    if layer.out_unsigned {
        ElemType::U8
    } else {
        ElemType::I8
    }
}

/// Activation contract of a convolution layer.
fn activation_of(graph: &Graph, layer: &Layer) -> Result<Activation, LowerError> {
    let act_qmax = graph.act_qmax();
    if act_qmax <= 0.0 {
        return Err(LowerError::Invalid {
            detail: format!("graph act_qmax {act_qmax} must be positive"),
        });
    }
    if layer.out_unsigned {
        if layer.out_relu {
            return Err(LowerError::Invalid {
                detail: format!(
                    "layer {} is marked both relu and unsigned, which has no representation",
                    layer.name
                ),
            });
        }
        let output_qmax = graph.output_qmax();
        if output_qmax <= 0.0 {
            return Err(LowerError::Invalid {
                detail: format!("graph output_qmax {output_qmax} must be positive"),
            });
        }
        Ok(Activation {
            kind: ActivationKind::UnsignedClamp,
            clamp_lo: 0,
            clamp_hi: output_qmax as i32,
        })
    } else if layer.out_relu {
        Ok(Activation {
            kind: ActivationKind::Relu,
            clamp_lo: 0,
            clamp_hi: act_qmax as i32,
        })
    } else {
        Ok(Activation {
            kind: ActivationKind::None,
            clamp_lo: -(act_qmax as i32),
            clamp_hi: act_qmax as i32,
        })
    }
}

/// Serialize the layer's per-output-channel requantization into fixed point.
fn requant_of(graph: &Graph, layer: &Layer, name: &str) -> Result<Requant, LowerError> {
    let activation = activation_of(graph, layer)?;
    let clamp = match activation.kind {
        ActivationKind::UnsignedClamp => ClampSpec::unsigned_output(graph.output_qmax()),
        ActivationKind::Relu | ActivationKind::Relu6 => ClampSpec::relu(graph.act_qmax()),
        ActivationKind::None => ClampSpec::symmetric(graph.act_qmax()),
    };
    let values = layer.requant.values();
    let channels = layer.out_channels.get();
    if values.len() != channels as usize {
        return Err(LowerError::Invalid {
            detail: format!(
                "layer {name}: requant vector has {} entries, expected {channels}",
                values.len()
            ),
        });
    }
    let (spec, error) =
        serialize_requant(values, channels, clamp).map_err(|source| LowerError::Quant {
            layer: name.to_string(),
            source,
        })?;
    let m0 = spec
        .multiplier
        .iter()
        .copied()
        .max()
        .ok_or_else(|| LowerError::Invalid {
            detail: format!("layer {name}: requant serialization produced no multipliers"),
        })?;
    Ok(Requant {
        m0: i64::from(m0),
        shift: spec.shift as i32,
        rounding: Rounding::HalfUp,
        saturate_lo: spec.clamp.lower as i32,
        saturate_hi: spec.clamp.upper as i32,
        per_channel_multipliers: spec.multiplier.iter().map(|m| i64::from(*m)).collect(),
        multiplier_error_ppm: Some(ppm(error)),
    })
}

/// Spatial extent of a convolution output.
///
/// Mirrors the graph IR's own rule (`(input + 2*pad - kernel) / stride + 1`) so
/// this module and the graph agree on every shape; `PlanInput::validate`
/// re-derives the planner's view of it after lowering.
fn conv_out_spatial(in_h: u32, in_w: u32, layer: &Layer) -> Result<(u32, u32), LowerError> {
    let name = layer.name.as_str();
    let pad_h = layer.padding.amount().map(|a| a.get()).unwrap_or(0);
    let pad_w = pad_h;
    let axis =
        |input: u32, pad: u32, kernel: u32, stride: u32, what: &str| -> Result<u32, LowerError> {
            let padded = input
                .checked_add(2 * pad)
                .ok_or_else(|| LowerError::Invalid {
                    detail: format!("layer {name}: {what} padding overflows"),
                })?;
            if padded < kernel {
                return Err(LowerError::Invalid {
                    detail: format!(
                        "layer {name}: {what} kernel {kernel} exceeds the padded input {padded}"
                    ),
                });
            }
            Ok((padded - kernel) / stride + 1)
        };
    Ok((
        axis(
            in_h,
            pad_h,
            layer.kernel.h.get(),
            layer.stride.h.get(),
            "height",
        )?,
        axis(
            in_w,
            pad_w,
            layer.kernel.w.get(),
            layer.stride.w.get(),
            "width",
        )?,
    ))
}

/// Map the graph's padding to the planner's padding contract.
///
/// Reflection padding and the planner's `Reflect`/`Same` mode are mutually
/// shape-equivalent (both keep `ceil(input/stride)` output extent), and the
/// planner re-derives the amount from the kernel and stride, so the graph's
/// declared amount is checked against the re-derived shape by
/// [`crate::planner::PlanInput::validate`] after lowering.
fn padding_of(layer: &Layer, _name: &str) -> Result<PlanPadding, LowerError> {
    Ok(match layer.padding {
        GraphPadding::None => PlanPadding::Zero {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        },
        GraphPadding::Reflect(_) => PlanPadding::Reflect,
        GraphPadding::Zero(amount) => PlanPadding::Zero {
            top: amount.get(),
            bottom: amount.get(),
            left: amount.get(),
            right: amount.get(),
        },
    })
}

/// Worst-case relative error, in parts per million, saturating at `u32::MAX`.
fn ppm(relative: f64) -> u32 {
    let value = (relative * 1_000_000.0).ceil();
    if !value.is_finite() || value < 0.0 {
        0
    } else if value > u32::MAX as f64 {
        u32::MAX
    } else {
        value as u32
    }
}
