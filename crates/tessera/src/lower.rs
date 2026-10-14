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
//!   contract defines as the *destination* point scale). The reference requantizes
//!   and clamps only the branch and leaves the sum unclamped
//!   (`fpga_st_train/src/intref.rs:257-268`), so an `add` output carries the
//!   *sum* of its operands' code intervals — `[-254, 254]` on a symmetric
//!   `qmax = 127` model — as [`ElemType::I9`] in a 16-bit word. It is never
//!   narrowed to `i8`, and every produced tensor records its exact interval.
//! * `upsample_nearest2x` is lowered with scale 2 and a doubled output shape.
//! * `quantize_u8_output` is lowered to an elementwise op whose activation is
//!   the unsigned output clamp `[0, output_qmax]`.

use crate::graph::{
    Graph, GraphError, GraphOp, Layer, LayerKind, Padding as GraphPadding, PointRef,
};
use crate::layout::{
    Activation, ActivationKind, CodeRange, ElemType, OpKind, Requant, Rounding, TensorId,
    TensorShape,
};
use crate::mlir::quant::{ClampSpec, QuantError, serialize_requant};
use crate::mlir::{
    ArgumentRole, LayerBinding, MlirArtifact, ModuleAnalysis, OperatorKind,
    PaddingSpec as MlirPadding, WeightPacking,
};
use crate::planner::{
    ConvDetail, ElementwiseDetail, OpDetail, Padding as PlanPadding, PlanInput, PrimitiveOp,
    TensorDesc, UpsampleDetail,
};
use std::collections::BTreeMap;
use std::fmt;

/// Name of a non-layer operator step: the point it writes when that is
/// unambiguous, with an execution-order suffix when a model writes the same point
/// more than once.
///
/// Both lowering paths must agree on this, because the full-graph backend checks a
/// plan's step names against its own graph lowering. The manifest graph writes
/// points in place, and the real `ink_dw` model adds the same point twice (a
/// residual add, then a skip add after an upsample), so a point-only name is not
/// unique; an operator-index name is unique but is not derivable from the MLIR
/// module alone. Counting occurrences per point, in execution order, is derivable
/// from both.
fn elementwise_name(prefix: &str, point: Option<u32>, occurrence: u32) -> String {
    match (point, occurrence) {
        (Some(point), 1) => format!("{prefix}.{point}"),
        (Some(point), nth) => format!("{prefix}.{point}.{nth}"),
        (None, nth) => format!("{prefix}.op{nth}"),
    }
}

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
    /// The MLIR artifact is unusable or its provenance does not hold.
    Artifact { detail: String },
    /// The optimized IR does not contain the structure the bindings promise.
    Analysis { detail: String },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::Shape { detail } => write!(f, "lower: {detail}"),
            LowerError::Quant { layer, source } => {
                write!(f, "lower: layer {layer}: requantization: {source}")
            }
            LowerError::Invalid { detail } => write!(f, "lower: {detail}"),
            LowerError::Artifact { detail } => write!(f, "lower: MLIR artifact: {detail}"),
            LowerError::Analysis { detail } => {
                write!(f, "lower: optimized module: {detail}")
            }
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
    /// How many `add` / `upsample_nearest2x` steps have already written a point,
    /// so a repeated write is named `add.<point>.<n>` in both lowering paths.
    elementwise_writes: std::collections::BTreeMap<(u8, u32), u32>,
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
            elementwise_writes: std::collections::BTreeMap::new(),
            image: None,
            image_shape: None,
            ops: Vec::new(),
            report: LowerReport::default(),
        };
        let input_point = graph.input_point().index();
        // `intref.rs:195-197` quantizes the image onto point 0 with the symmetric
        // clamp, so the input tensor carries exactly that interval.
        let qmax = graph.act_qmax() as i64;
        let tensor = builder.push_tensor_coded(
            format!("p{input_point}.v0"),
            TensorShape::new(1, input.channels, input.height, input.width),
            ElemType::I8,
            Some(CodeRange::new(-qmax, qmax)),
        );
        builder.current[input_point] = Some(tensor);
        builder
    }

    fn push_tensor(&mut self, name: String, shape: TensorShape, elem: ElemType) -> TensorId {
        self.push_tensor_coded(name, shape, elem, None)
    }

    /// Push a tensor with the exact interval of codes it carries.
    fn push_tensor_coded(
        &mut self,
        name: String,
        shape: TensorShape,
        elem: ElemType,
        code_range: Option<CodeRange>,
    ) -> TensorId {
        let id = TensorId::new(self.tensors.len() as u32);
        self.tensors.push(TensorDesc {
            id,
            name,
            shape,
            elem,
            code_range,
        });
        id
    }

    /// The interval a tensor carries, if the lowering bounded it.
    fn code_range(&self, id: TensorId) -> Option<CodeRange> {
        self.tensors
            .get(id.get() as usize)
            .and_then(|tensor| tensor.code_range)
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

    /// Next occurrence index for a non-layer write to `point`, in execution order.
    fn next_elementwise_write(&mut self, kind: u8, point: u32) -> u32 {
        let counter = self.elementwise_writes.entry((kind, point)).or_insert(0);
        *counter += 1;
        *counter
    }

    fn layer(&self, name: &str) -> Result<&'a Layer, LowerError> {
        self.graph
            .layer_named(name)
            .ok_or_else(|| LowerError::Invalid {
                detail: format!("graph references unknown layer {name:?}"),
            })
    }

    /// Lower one graph op, in execution order.
    ///
    /// Non-layer steps are named by the point they write plus an occurrence suffix
    /// ([`elementwise_name`]), because a graph may add or upsample the *same* point
    /// more than once and the MLIR-derived plan must arrive at the same names.
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

        let (elem, range) = elem_for_range(
            i64::from(activation.clamp_lo),
            i64::from(activation.clamp_hi),
            layer.out_unsigned,
        );
        let output = self.write_point(out_point, out_shape, elem, Some(range));
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
        let occurrence = self.next_elementwise_write(b'a', point.get());
        let name = elementwise_name("add", Some(point.get()), occurrence);
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
            rounding: Rounding::HalfAwayFromZero,
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
        // The sum of the two operand intervals: the branch is requantized and
        // clamped by the add itself, the trunk carries whatever its producer
        // carries, and the reference does not clamp the sum. The branch interval is
        // the branch tensor's squeezed by the clamp: the emitted datapath clamps it
        // to `[lower, upper]`, and a factor at most one cannot move a code outside
        // the interval it came from.
        let branch_source = self
            .code_range(source)
            .unwrap_or_else(|| CodeRange::new(-(act_qmax as i64), act_qmax as i64));
        let branch_range = if factor <= 1.0 {
            CodeRange::new(
                branch_source.lo.max(spec.clamp.lower),
                branch_source.hi.min(spec.clamp.upper),
            )
        } else {
            CodeRange::new(spec.clamp.lower, spec.clamp.upper)
        };
        let trunk_range = self
            .code_range(destination)
            .unwrap_or_else(|| CodeRange::new(-(act_qmax as i64), act_qmax as i64));
        let sum_range = trunk_range.sum(branch_range);
        let (elem, range) = elem_for_range(sum_range.lo, sum_range.hi, false);
        let output = self.write_point(PointRef::Point(point), shape, elem, Some(range));
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
        let occurrence = self.next_elementwise_write(b'u', point.get());
        let name = elementwise_name("upsample_nearest2x", Some(point.get()), occurrence);
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
        // Duplication changes no code.
        let (elem, range) = (
            self.tensors[input.get() as usize].elem,
            self.code_range(input),
        );
        let output = self.write_point(PointRef::Point(point), out_shape, elem, range);
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
        let output = self.push_tensor_coded(
            "output.u8".to_string(),
            shape,
            ElemType::U8,
            Some(CodeRange::new(0, self.graph.output_qmax() as i64)),
        );
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
    fn write_point(
        &mut self,
        target: PointRef,
        shape: TensorShape,
        elem: ElemType,
        code_range: Option<CodeRange>,
    ) -> TensorId {
        match target {
            PointRef::Point(id) => {
                let version = self.versions[id.index()];
                self.versions[id.index()] += 1;
                let tensor = self.push_tensor_coded(
                    format!("p{}.v{version}", id.get()),
                    shape,
                    elem,
                    code_range,
                );
                self.current[id.index()] = Some(tensor);
                tensor
            }
            PointRef::ImageSentinel => {
                let tensor =
                    self.push_tensor_coded("image.v0".to_string(), shape, elem, code_range);
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
        rounding: Rounding::HalfAwayFromZero,
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

/// Element type and interval for a tensor that carries `[lo, hi]`.
///
/// The interval decides the arithmetic width (`ElemType::I9` is a nine-bit value
/// in a sixteen-bit word) and is recorded next to the element type, so no stage
/// has to infer a datapath width from a container again.
pub fn elem_for_range(lo: i64, hi: i64, unsigned: bool) -> (ElemType, CodeRange) {
    let range = CodeRange::new(lo, hi);
    let elem = if unsigned && lo >= 0 && hi > i8::MAX as i64 && hi <= u8::MAX as i64 {
        ElemType::U8
    } else if range.fits_i8() {
        ElemType::I8
    } else if range.signed_bits() <= 9 {
        ElemType::I9
    } else {
        ElemType::I16
    };
    (elem, range)
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

// ---------------------------------------------------------------------------
// Optimized MLIR artifact -> planner primitive graph
// ---------------------------------------------------------------------------

/// Lower an *optimized* MLIR artifact into the planner's primitive graph.
///
/// This is the planning path that actually consumes the optimizer's output. The
/// module it reads is the bufferized, loop-lowered, fused text the pass pipeline
/// produced, and every operator it emits is checked against the structured
/// analysis of that text before the plan is returned:
///
/// * an operator implemented as IR structure (`conv`, `add`, `upsample`) must
///   still carry its operations in the optimized module, and a convolution must
///   appear in a loop nest, so a pass that changed or dropped its work is
///   detected instead of being reported as a statistic;
/// * a padded convolution must have materialized its padded buffer, with the
///   exact padded shape this lowering derives from the layer's padding rule and
///   the input shape;
/// * tensor shapes, element types and code intervals come from the artifact's
///   tensor table, which the emitter wrote from the same walk that produced the
///   module text — nothing is recovered from generated arithmetic.
///
/// Graph-stage artifacts are rejected: [`crate::mlir::optimize`] has to have run.
pub fn lower_mlir_artifact(
    artifact: &MlirArtifact,
) -> Result<(PlanInput, LowerReport), LowerError> {
    artifact
        .validate_header()
        .map_err(|error| LowerError::Artifact {
            detail: error.to_string(),
        })?;
    artifact.check(None).map_err(|error| LowerError::Artifact {
        detail: error.to_string(),
    })?;
    let analysis = artifact.analyzed().map_err(|error| LowerError::Artifact {
        detail: error.to_string(),
    })?;

    let mut builder = MlirBuilder::new(artifact, analysis)?;
    for operator in &artifact.operators {
        builder.lower_operator(operator)?;
    }
    builder.finish()
}

/// Builder that turns persisted MLIR bindings plus the analysis of the optimized
/// module into a [`PlanInput`].
struct MlirBuilder<'a> {
    artifact: &'a MlirArtifact,
    analysis: &'a ModuleAnalysis,
    tensors: Vec<TensorDesc>,
    ids: BTreeMap<String, TensorId>,
    ops: Vec<PrimitiveOp>,
    report: LowerReport,
    /// Occurrence counters for repeated non-layer writes, keyed like the graph
    /// path's, so both paths name a step identically.
    elementwise_writes: std::collections::BTreeMap<(u8, u32), u32>,
    input: Option<TensorId>,
    output: Option<TensorId>,
}

impl<'a> MlirBuilder<'a> {
    fn new(artifact: &'a MlirArtifact, analysis: &'a ModuleAnalysis) -> Result<Self, LowerError> {
        let mut builder = MlirBuilder {
            artifact,
            analysis,
            tensors: Vec::new(),
            ids: BTreeMap::new(),
            ops: Vec::new(),
            report: LowerReport::default(),
            elementwise_writes: std::collections::BTreeMap::new(),
            input: None,
            output: None,
        };
        let input = artifact
            .tensor("input")
            .ok_or_else(|| LowerError::Artifact {
                detail: "the artifact names no `input` activation".to_string(),
            })?;
        for operator in &artifact.operators {
            if operator.kind == OperatorKind::QuantizeU8 {
                continue;
            }
            for produced in &operator.produces {
                if let Some(tensor) = artifact.tensor(produced) {
                    builder.define(tensor)?;
                }
            }
        }
        let input_id = builder.define(input)?;
        builder.input = Some(input_id);
        builder.report.notes.push(format!(
            "lowered from the optimized module ({} operations, {} loops, {} buffers, module hash {})",
            analysis.counts.operations,
            analysis.counts.loops,
            analysis.counts.buffers,
            crate::mlir::content_hash(&artifact.module)
        ));
        Ok(builder)
    }

    /// Register one activation tensor from the artifact's tensor table.
    fn define(&mut self, tensor: &crate::mlir::TensorBinding) -> Result<TensorId, LowerError> {
        if let Some(existing) = self.ids.get(&tensor.ssa_name) {
            return Ok(*existing);
        }
        let range = tensor.code_range.map(plan_range);
        let shape = plan_shape(&tensor.shape)?;
        // The element type follows from the transport the module chose and the
        // exact interval: a widened value transported in `i16` is a nine-bit code
        // while it fits, an interval beyond a signed byte in an `i8` container is
        // the unsigned image head (the emitter's documented convention), and
        // anything else in a byte is a signed code.
        let elem = match (tensor.element, range) {
            (crate::mlir::ElementType::I8, Some(range))
                if range.lo >= 0 && range.hi > i8::MAX as i64 =>
            {
                ElemType::U8
            }
            (crate::mlir::ElementType::I8, Some(range)) if range.fits_i8() => ElemType::I8,
            (crate::mlir::ElementType::I16, Some(range)) if range.signed_bits() <= 9 => {
                ElemType::I9
            }
            (crate::mlir::ElementType::I16, _) => ElemType::I16,
            _ => ElemType::I8,
        };
        let id = TensorId::new(self.tensors.len() as u32);
        self.tensors.push(TensorDesc {
            id,
            name: tensor.ssa_name.clone(),
            shape,
            elem,
            code_range: range,
        });
        self.ids.insert(tensor.ssa_name.clone(), id);
        Ok(id)
    }

    fn tensor(&self, ssa: &str) -> Result<TensorId, LowerError> {
        self.ids
            .get(ssa)
            .copied()
            .ok_or_else(|| LowerError::Artifact {
                detail: format!(
                    "the artifact references activation {ssa:?}, which it does not describe"
                ),
            })
    }

    fn lower_operator(
        &mut self,
        operator: &crate::mlir::OperatorBinding,
    ) -> Result<(), LowerError> {
        self.require_ir_work(operator)?;
        match operator.kind {
            OperatorKind::Conv2d
            | OperatorKind::DepthwiseConv2d
            | OperatorKind::PointwiseConv2d => self.lower_conv(operator),
            OperatorKind::Relu => {
                self.report.notes.push(format!(
                    "operator {}: `relu` is a scheduling marker; the saturation is carried by the \
                     convolution epilogue",
                    operator.index
                ));
                if self.analysis.tag_count(&operator.tag) != 0 {
                    return Err(LowerError::Analysis {
                        detail: format!(
                            "the relu marker {} unexpectedly has {} operations in the optimized module",
                            operator.tag,
                            self.analysis.tag_count(&operator.tag)
                        ),
                    });
                }
                Ok(())
            }
            OperatorKind::Add => self.lower_add(operator),
            OperatorKind::UpsampleNearest2x => self.lower_upsample(operator),
            OperatorKind::QuantizeU8 => self.lower_quantize(operator),
        }
    }

    /// Every operator that the emitter lowers to real IR must still be there.
    fn require_ir_work(&self, operator: &crate::mlir::OperatorBinding) -> Result<(), LowerError> {
        let structural = matches!(
            operator.kind,
            OperatorKind::Conv2d
                | OperatorKind::DepthwiseConv2d
                | OperatorKind::PointwiseConv2d
                | OperatorKind::Add
                | OperatorKind::UpsampleNearest2x
        );
        if structural && self.analysis.tag_count(&operator.tag) == 0 {
            return Err(LowerError::Analysis {
                detail: format!(
                    "operator {} ({:?}, tag {}) has no operations in the optimized module; the \
                     primitive lowering would not be driven by the optimized IR",
                    operator.index, operator.kind, operator.tag
                ),
            });
        }
        Ok(())
    }

    fn lower_conv(&mut self, operator: &crate::mlir::OperatorBinding) -> Result<(), LowerError> {
        let layer_name = operator.layer.clone().ok_or_else(|| LowerError::Artifact {
            detail: format!("convolution operator {} names no layer", operator.index),
        })?;
        let layer = self
            .artifact
            .layer(&layer_name)
            .ok_or_else(|| LowerError::Artifact {
                detail: format!("the artifact has no binding for layer {layer_name:?}"),
            })?;
        let input = self.tensor(&operator.consumes[0])?;
        let output = self.tensor(&operator.produces[0])?;
        let kind = match operator.kind {
            OperatorKind::Conv2d => OpKind::Conv2d,
            OperatorKind::DepthwiseConv2d => OpKind::DepthwiseConv2d,
            OperatorKind::PointwiseConv2d => OpKind::PointwiseConv2d,
            other => {
                return Err(LowerError::Artifact {
                    detail: format!(
                        "operator {} is not a convolution ({other:?})",
                        operator.index
                    ),
                });
            }
        };

        // The planner reads a convolution's kernel from the weight tensor's
        // spatial extents, so the tensor is `(n, c, kh, kw)` in the storage
        // layout the artifact recorded for the filter argument.
        let filter = self
            .artifact
            .argument(ArgumentRole::Filter, Some(&layer_name))
            .ok_or_else(|| LowerError::Artifact {
                detail: format!("layer {layer_name:?} has no filter argument"),
            })?;
        let weight_elem = match filter.weight.as_ref().map(|weight| weight.packing) {
            Some(WeightPacking::I8x1) => ElemType::I8,
            Some(WeightPacking::I4x2) => ElemType::I4,
            None => ElemType::I8,
        };
        let weight_shape = match (filter.shape.as_slice(), &filter.weight) {
            ([kh, kw, channels], Some(weight))
                if weight.mlir_layout == crate::mlir::MlirFilterLayout::Hwc =>
            {
                TensorShape::new(narrow(*channels)?, 1, narrow(*kh)?, narrow(*kw)?)
            }
            ([out_channels, kh, kw, in_channels], _) => TensorShape::new(
                narrow(*out_channels)?,
                narrow(*in_channels)?,
                narrow(*kh)?,
                narrow(*kw)?,
            ),
            (shape, _) => {
                return Err(LowerError::Artifact {
                    detail: format!(
                        "layer {layer_name:?}: filter argument has shape {shape:?}, which is neither \
                         HWC nor FHWC"
                    ),
                });
            }
        };
        let weights = self.push_tensor(weight_tensor_name(&layer_name), weight_shape, weight_elem);
        let bias = if layer.bias.requires_tensor() {
            Some(self.push_tensor(
                bias_tensor_name(&layer_name),
                TensorShape::new(layer.quantization.multiplier.len() as u32, 1, 1, 1),
                ElemType::I32,
            ))
        } else {
            self.report.notes.push(format!(
                "layer {layer_name}: int32 bias is all zero and was dropped (exact identity)"
            ));
            None
        };

        let requant = requant_of_mlir(&layer.quantization, &layer_name)?;
        let activation = activation_of_mlir(&layer.quantization);
        let padding = padding_of_mlir(layer.padding);
        let (buffer_bytes, nest_ordinal, nest_depth, fused, others) =
            self.verify_conv(operator, layer, input)?;
        self.report.notes.push(format!(
            "layer {layer_name}: {} operations in the optimized module, nest {nest_ordinal} at              depth {nest_depth} ({}{others} other operator(s)), {buffer_bytes} bytes of buffers",
            self.analysis.tag_count(&operator.tag),
            if fused { "fused with " } else { "alone, " },
        ));

        self.ops.push(PrimitiveOp {
            name: layer_name,
            kind,
            inputs: vec![input],
            outputs: vec![output],
            detail: OpDetail::Conv(ConvDetail {
                weights,
                bias,
                groups: layer.groups,
                stride: layer.stride,
                padding,
                requant,
                activation,
            }),
        });
        Ok(())
    }

    /// Cross-check a convolution against the optimized IR it must have produced.
    fn verify_conv(
        &self,
        operator: &crate::mlir::OperatorBinding,
        layer: &LayerBinding,
        input: TensorId,
    ) -> Result<(u64, usize, u32, bool, usize), LowerError> {
        let nest = self
            .analysis
            .nest_of_operator(operator.index)
            .ok_or_else(|| LowerError::Analysis {
                detail: format!(
                    "layer {}: no loop nest in the optimized module performs operator {} ({})",
                    layer.layer, operator.index, operator.tag
                ),
            })?;
        let nest_ordinal = nest.ordinal;
        let nest_depth = nest.depth;
        let fused = nest.fused;
        let others = nest
            .operator_indices
            .iter()
            .filter(|index| **index != operator.index)
            .count();
        // The padded buffer is the optimizer's own materialization of the padding
        // rule; if it does not exist with the expected shape, the plan and the
        // optimized IR disagree.
        let amount = layer.padding.amount() as u64;
        let input_desc =
            self.tensors
                .get(input.get() as usize)
                .ok_or_else(|| LowerError::Artifact {
                    detail: "convolution input tensor is missing from the plan".to_string(),
                })?;
        let mut bytes = 0u64;
        if amount > 0 {
            let nhwc = [
                input_desc.shape.n as u64,
                input_desc.shape.h as u64 + 2 * amount,
                input_desc.shape.w as u64 + 2 * amount,
                input_desc.shape.c as u64,
            ];
            let element = if input_desc.elem == ElemType::I8 {
                "i8"
            } else {
                "i16"
            };
            let buffers = self.analysis.buffers_with(&nhwc, element);
            let buffer = buffers.first().ok_or_else(|| LowerError::Analysis {
                detail: format!(
                    "layer {}: the optimized module has no {element} buffer of the padded input \
                     shape {nhwc:?}; padding did not materialize as the binding claims",
                    layer.layer
                ),
            })?;
            bytes = buffer.bytes.unwrap_or(0);
        }
        Ok((bytes, nest_ordinal, nest_depth, fused, others))
    }

    fn lower_add(&mut self, operator: &crate::mlir::OperatorBinding) -> Result<(), LowerError> {
        let input = self.tensor(&operator.consumes[0])?;
        let branch = self.tensor(&operator.consumes[1])?;
        let output = self.tensor(&operator.produces[0])?;
        let spec = operator
            .quantization
            .as_ref()
            .ok_or_else(|| LowerError::Artifact {
                detail: format!(
                    "add operator {} records no branch requantization",
                    operator.index
                ),
            })?;
        let requant = Requant {
            m0: spec.multiplier.first().copied().map(i64::from).unwrap_or(0),
            shift: spec.shift as i32,
            rounding: Rounding::HalfAwayFromZero,
            saturate_lo: spec.clamp.lower as i32,
            saturate_hi: spec.clamp.upper as i32,
            per_channel_multipliers: Vec::new(),
            multiplier_error_ppm: Some(ppm(spec.max_relative_error)),
        };
        // The output is *not* clamped: the reference leaves the sum unclamped, and
        // the interval the tensor table recorded is the emitter's exact sum of the
        // trunk interval and the requantization-clamped branch interval.
        let range = self
            .tensors
            .get(output.get() as usize)
            .and_then(|tensor| tensor.code_range)
            .unwrap_or(CodeRange::new(i16::MIN as i64, i16::MAX as i64));
        let name = self.elementwise_step_name(b'a', "add", operator)?;
        self.report.notes.push(format!(
            "{name}: branch requantized onto the destination grid (M0={} shift={}), sum carried \
             unclamped in [{}, {}]",
            requant.m0, requant.shift, range.lo, range.hi
        ));
        self.ops.push(PrimitiveOp {
            name,
            kind: OpKind::Add,
            inputs: vec![input, branch],
            outputs: vec![output],
            detail: OpDetail::Elementwise(ElementwiseDetail {
                requant: Some(requant),
                activation: Activation {
                    kind: ActivationKind::None,
                    clamp_lo: range.lo as i32,
                    clamp_hi: range.hi as i32,
                },
            }),
        });
        Ok(())
    }

    /// Name of a non-layer step, using the shared rule: the point it writes, with
    /// an execution-order suffix when the point is written more than once.
    fn elementwise_step_name(
        &mut self,
        kind: u8,
        prefix: &str,
        operator: &crate::mlir::OperatorBinding,
    ) -> Result<String, LowerError> {
        let point = operator.points.first().copied();
        let counter = self
            .elementwise_writes
            .entry((kind, point.unwrap_or(u32::MAX)))
            .or_insert(0);
        *counter += 1;
        let occurrence = *counter;
        if point.is_none() && operator.index == 0 {
            return Err(LowerError::Artifact {
                detail: format!(
                    "{prefix}: the operator binding names neither a point nor an index, so the \
                     step cannot be named"
                ),
            });
        }
        Ok(elementwise_name(prefix, point, occurrence))
    }

    fn lower_upsample(
        &mut self,
        operator: &crate::mlir::OperatorBinding,
    ) -> Result<(), LowerError> {
        let input = self.tensor(&operator.consumes[0])?;
        let output = self.tensor(&operator.produces[0])?;
        let name = self.elementwise_step_name(b'u', "upsample_nearest2x", operator)?;
        self.ops.push(PrimitiveOp {
            name,
            kind: OpKind::NearestUpsample2x,
            inputs: vec![input],
            outputs: vec![output],
            detail: OpDetail::Upsample(UpsampleDetail { scale: 2 }),
        });
        Ok(())
    }

    /// Materialize the unsigned output step.
    ///
    /// The module implements the head's unsigned clamp inside the convolution
    /// epilogue (`%upper = arith.const 255`), so the graph operator carries no
    /// operations of its own; the planner models the manifest's
    /// `quantize_u8_output` step explicitly, exactly as the graph lowering does,
    /// and the interval comes from the value the head produced.
    fn lower_quantize(
        &mut self,
        operator: &crate::mlir::OperatorBinding,
    ) -> Result<(), LowerError> {
        let image = self.tensor(&operator.consumes[0])?;
        let shape = self.tensors[image.get() as usize].shape;
        let upper = self
            .tensors
            .get(image.get() as usize)
            .and_then(|tensor| tensor.code_range)
            .map(|range| range.hi)
            .ok_or_else(|| LowerError::Artifact {
                detail: "the unsigned image tensor records no code interval".to_string(),
            })?;
        if upper <= 0 {
            return Err(LowerError::Analysis {
                detail: format!(
                    "operator {}: the value feeding the unsigned output carries no positive codes",
                    operator.index
                ),
            });
        }
        let output = self.push_tensor_coded(
            "output.u8".to_string(),
            shape,
            ElemType::U8,
            Some(CodeRange::new(0, upper)),
        );
        self.output = Some(output);
        self.report.notes.push(format!(
            "quantize_u8_output: the optimized module applies the head clamp [0, {upper}] inside \
             the convolution epilogue; the planner models it as this step"
        ));
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
                    clamp_hi: upper as i32,
                },
            }),
        });
        Ok(())
    }

    fn push_tensor(&mut self, name: String, shape: TensorShape, elem: ElemType) -> TensorId {
        let id = TensorId::new(self.tensors.len() as u32);
        self.tensors.push(TensorDesc {
            id,
            name,
            shape,
            elem,
            code_range: None,
        });
        id
    }

    /// Like [`Self::push_tensor`], with a code interval.
    fn push_tensor_coded(
        &mut self,
        name: String,
        shape: TensorShape,
        elem: ElemType,
        code_range: Option<CodeRange>,
    ) -> TensorId {
        let id = TensorId::new(self.tensors.len() as u32);
        self.tensors.push(TensorDesc {
            id,
            name,
            shape,
            elem,
            code_range,
        });
        id
    }

    fn finish(self) -> Result<(PlanInput, LowerReport), LowerError> {
        let input = self.input.ok_or_else(|| LowerError::Artifact {
            detail: "no input activation was lowered".to_string(),
        })?;
        let output = self
            .output
            .or_else(|| self.ops.last().and_then(|op| op.outputs.first().copied()))
            .ok_or_else(|| LowerError::Artifact {
                detail: "the artifact produced no output activation".to_string(),
            })?;
        let name = if self.artifact.function.trim().is_empty() {
            "tessera.mlir".to_string()
        } else {
            self.artifact.function.clone()
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
        Ok((plan, self.report))
    }
}

/// Convert a persisted code interval into the planner's vocabulary.
fn plan_range(range: crate::mlir::CodeRange) -> CodeRange {
    CodeRange::new(range.lo, range.hi)
}

/// Narrow a shape extent from the module's `u64` to the planner's `u32`.
fn narrow(value: u64) -> Result<u32, LowerError> {
    u32::try_from(value).map_err(|_| LowerError::Artifact {
        detail: format!("extent {value} does not fit the planner's shape"),
    })
}

/// Convert an `NHWC` tensor shape into the planner's `NCHW` shape.
fn plan_shape(shape: &[u64]) -> Result<TensorShape, LowerError> {
    let [n, h, w, c] = shape else {
        return Err(LowerError::Artifact {
            detail: format!("activation shape {shape:?} is not rank-4 NHWC"),
        });
    };
    if *n != 1 {
        return Err(LowerError::Artifact {
            detail: format!("v1 models are batch 1, but the module carries batch {n}"),
        });
    }
    let (c, h, w) = (narrow(*c)?, narrow(*h)?, narrow(*w)?);
    if c == 0 || h == 0 || w == 0 {
        return Err(LowerError::Artifact {
            detail: format!("activation shape {shape:?} has a zero extent"),
        });
    }
    Ok(TensorShape::new(1, c, h, w))
}

/// Fixed-point requantization of a persisted convolution binding.
fn requant_of_mlir(
    quantization: &crate::mlir::Quantization,
    layer: &str,
) -> Result<Requant, LowerError> {
    let m0 = quantization
        .multiplier
        .iter()
        .copied()
        .max()
        .ok_or_else(|| LowerError::Artifact {
            detail: format!("layer {layer}: the artifact records no requantization multipliers"),
        })?;
    Ok(Requant {
        m0: i64::from(m0),
        shift: quantization.shift as i32,
        rounding: Rounding::HalfAwayFromZero,
        saturate_lo: quantization.clamp.lower as i32,
        saturate_hi: quantization.clamp.upper as i32,
        per_channel_multipliers: quantization
            .multiplier
            .iter()
            .map(|value| i64::from(*value))
            .collect(),
        multiplier_error_ppm: Some(ppm(quantization.max_relative_error)),
    })
}

/// Saturation regime of a persisted convolution binding.
fn activation_of_mlir(quantization: &crate::mlir::Quantization) -> Activation {
    match quantization.clamp.kind {
        crate::mlir::ClampKind::UnsignedOutput => Activation {
            kind: ActivationKind::UnsignedClamp,
            clamp_lo: quantization.clamp.lower as i32,
            clamp_hi: quantization.clamp.upper as i32,
        },
        crate::mlir::ClampKind::Relu => Activation {
            kind: ActivationKind::Relu,
            clamp_lo: quantization.clamp.lower as i32,
            clamp_hi: quantization.clamp.upper as i32,
        },
        crate::mlir::ClampKind::Symmetric => Activation {
            kind: ActivationKind::None,
            clamp_lo: quantization.clamp.lower as i32,
            clamp_hi: quantization.clamp.upper as i32,
        },
    }
}

/// Padding rule of a persisted layer binding.
fn padding_of_mlir(padding: MlirPadding) -> PlanPadding {
    match padding {
        MlirPadding::None => PlanPadding::Zero {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        },
        MlirPadding::Reflect { .. } => PlanPadding::Reflect,
        MlirPadding::Zero { amount } => PlanPadding::Zero {
            top: amount,
            bottom: amount,
            left: amount,
            right: amount,
        },
    }
}
