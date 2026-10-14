//! Deterministic Graph IR -> MLIR text emission.
//!
//! The emitted module is tensor level and uses only generic dialects: `func`,
//! `tensor`, `linalg`, `arith`. No Tessera-owned MLIR dialect is involved, so
//! the artifact can be consumed by stock `mlir-opt` and by the pinned Melior
//! pipeline.
//!
//! Numeric contract, taken from `fpga_st_train/src/intref.rs`:
//!
//! * a convolution accumulates `bias_int32[oc] + sum(input_code * weight_code)`
//!   in `i32`; `linalg.fill` seeds the accumulator with zero and the bias is
//!   added in the epilogue, which is the same integer sum. The explicit
//!   `linalg.generic` form used for widened activations accumulates into its
//!   `outs` operand exactly like the named form, so it takes the same zero-filled
//!   seed — never an uninitialized `tensor.empty`;
//! * requantization is `round_clamp(acc * requant[oc], lo, hi)` with
//!   round-half-away-from-zero, emitted in the `(M0, shift)` fixed-point form
//!   produced by [`super::quant`] and verified against the NNgen oracle;
//! * the saturation regime is `[-qmax, qmax]`, `[0, qmax]` for `+relu`
//!   (`intref.rs:341-349`) or `[0, output_qmax]` for the unsigned head;
//! * padding is a `tensor.pad` whose region reflects indices exactly like
//!   `intref.rs::reflect_index`;
//! * `add` requantizes the branch onto the destination grid, clamps the branch
//!   and adds it to the trunk without clamping the sum (`intref.rs:257-268`);
//! * `upsample_nearest2x` duplicates each code into a 2x2 block
//!   (`intref.rs:540-552`).
//!
//! Weight, bias and multiplier values are never inlined into the text as data:
//! filters, `bias_int32` and per-channel multipliers are function arguments
//! whose external storage reference is recorded in [`MlirArgument`]. The
//! per-layer shift and clamp bounds are compile-time constants and *are*
//! emitted as literals, so the text shows the exact quantization contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::artifact::{
    ArgumentRole, ArtifactMetrics, BiasContribution, CodeRange, ElementType, FilterStorageLayout,
    LayerBinding, MlirArgument, MlirError, MlirFilterLayout, MlirOptions, OperatorBinding,
    OperatorKind, PaddingSpec, Quantization, TensorBinding, WeightPacking, WeightReference,
    tensor_type,
};
use super::quant::{ClampSpec, QuantError, RequantSpec, rounding_offsets};
use crate::graph::{
    Graph, GraphOp, Layer, LayerKind as GraphLayerKind, LayerName, Packing, Padding, PointId,
    PointRef,
};

/// Result of emitting one graph.
#[derive(Debug, Clone, PartialEq)]
pub struct EmittedModule {
    /// Textual MLIR, byte-for-byte deterministic for a given graph and options.
    pub module: String,
    /// Concrete `NHWC` input shape the module was emitted for.
    pub input_shape: Vec<u64>,
    /// Function arguments in declaration order.
    pub arguments: Vec<MlirArgument>,
    /// Activation tensors the module names, in emission order.
    pub tensors: Vec<TensorBinding>,
    /// Convolution layers in graph order.
    pub layers: Vec<LayerBinding>,
    /// Operators in execution order.
    pub operators: Vec<OperatorBinding>,
    /// Graph output tensor.
    pub output: TensorBinding,
    pub metrics: ArtifactMetrics,
}

/// Location namespace every Tessera-emitted operation carries.
///
/// The emitter tags each structural operation with `loc("tessera:<role>/<owner>")`
/// so that, after bufferization, loop lowering and fusion, the *optimized* IR
/// still says which layer or graph operator each affine loop body came from.
/// This is the stable mapping the primitive lowering and the fusion analysis use:
/// nothing recovers operator semantics by inspecting generated arithmetic.
pub const LOC_TAG_PREFIX: &str = "tessera:";

/// Role of the convolution itself.
pub const TAG_CONV: &str = "conv";
/// Role of the reflection/zero padding of a convolution input.
pub const TAG_PAD: &str = "pad";
/// Role of the requantization/saturation epilogue of a convolution.
pub const TAG_EPILOGUE: &str = "epilogue";
/// Role of an elementwise `add`.
pub const TAG_ADD: &str = "add";
/// Role of the nearest-neighbour 2x duplication.
pub const TAG_UPSAMPLE: &str = "upsample";

/// Render the location attribute for one tagged operation.
fn loc(tag: &str) -> String {
    format!(" loc(\"{LOC_TAG_PREFIX}{tag}\")")
}

/// Emit tensor-level MLIR for `graph`.
pub fn emit_module(graph: &Graph, options: &MlirOptions) -> Result<EmittedModule, MlirError> {
    options.validate()?;
    Emitter::new(graph, options)?.run()
}

/// Convenience for callers that only need the text.
pub fn emit_text(graph: &Graph, options: &MlirOptions) -> Result<String, MlirError> {
    emit_module(graph, options).map(|module| module.module)
}

/// One activation value tracked while walking the graph.
#[derive(Debug, Clone, PartialEq)]
struct TensorRef {
    ssa: String,
    shape: Vec<u64>,
    element: ElementType,
    point: Option<u32>,
    /// Exact interval of codes this value can hold.
    code_range: CodeRange,
}

impl TensorRef {
    /// Narrowest MLIR element type that transports the value: `i8` while the
    /// codes fit a signed byte, `i16` otherwise. A widened value is never
    /// narrowed, because the reference leaves an `add` sum unclamped
    /// (`fpga_st_train/src/intref.rs:257-268`).
    fn element_for(range: CodeRange) -> Result<ElementType, MlirError> {
        if range.fits_i8() {
            return Ok(ElementType::I8);
        }
        if range.lo >= i16::MIN as i64 && range.hi <= i16::MAX as i64 {
            return Ok(ElementType::I16);
        }
        Err(MlirError::Unsupported {
            feature: "activation code range".to_string(),
            detail: format!(
                "codes [{}, {}] need more than the 16-bit transport the v1 emitter supports",
                range.lo, range.hi
            ),
        })
    }
}

/// Padding mode of an emitted `tensor.pad`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PadMode {
    Zero,
    Reflect,
}

struct Emitter<'g> {
    graph: &'g Graph,
    options: &'g MlirOptions,
    body: String,
    arguments: Vec<MlirArgument>,
    /// Every tensor value the module names, in emission order.
    tensors_named: Vec<TensorBinding>,
    layers: Vec<LayerBinding>,
    operators: Vec<OperatorBinding>,
    /// Live activations, keyed by quantization point index.
    points: BTreeMap<u32, TensorRef>,
    /// Output of the unsigned head convolution.
    image: Option<TensorRef>,
    used_prefixes: BTreeSet<String>,
    tensors: usize,
    macs: u64,
    requantizations: u64,
    needs_zero_i32: bool,
    input: TensorRef,
}

impl<'g> Emitter<'g> {
    fn new(graph: &'g Graph, options: &'g MlirOptions) -> Result<Self, MlirError> {
        // `intref.rs:195-197` quantizes the input image onto point 0 with the
        // symmetric clamp, so the whole module consumes codes in [-qmax, qmax].
        let input_range = CodeRange::new(-(graph.act_qmax() as i64), graph.act_qmax() as i64);
        let input = TensorRef {
            ssa: "input".to_string(),
            shape: options.input_shape.clone(),
            element: ElementType::I8,
            point: Some(0),
            code_range: input_range,
        };
        let mut used_prefixes = BTreeSet::new();
        used_prefixes.insert("input".to_string());
        used_prefixes.insert("zero_i32".to_string());
        let input_binding = TensorBinding {
            ssa_name: input.ssa.clone(),
            shape: input.shape.clone(),
            element: input.element,
            point: input.point,
            scale: graph.point_scale(PointId::new(0)),
            code_range: Some(input_range),
        };
        Ok(Emitter {
            graph,
            options,
            body: String::new(),
            arguments: vec![MlirArgument {
                ssa_name: input.ssa.clone(),
                role: ArgumentRole::Input,
                element: ElementType::I8,
                shape: input.shape.clone(),
                layer: None,
                weight: None,
            }],
            tensors_named: vec![input_binding],
            layers: Vec::new(),
            operators: Vec::new(),
            points: BTreeMap::new(),
            image: None,
            used_prefixes,
            tensors: 0,
            macs: 0,
            requantizations: 0,
            needs_zero_i32: false,
            input,
        })
    }

    fn run(mut self) -> Result<EmittedModule, MlirError> {
        if self.input.shape.len() != 4 {
            return Err(MlirError::Shape {
                detail: format!(
                    "v1 emission needs rank-4 NHWC activations, found {}",
                    shape_text(&self.input.shape)
                ),
            });
        }
        self.check_input_channels()?;
        self.points.insert(0, self.input.clone());

        let ops = self.graph.ops();
        for (index, op) in ops.iter().enumerate() {
            self.emit_operator(index, op)?;
        }

        let output = self
            .image
            .clone()
            .or_else(|| self.last_live_point())
            .ok_or_else(|| MlirError::InvalidGraph {
                detail: "graph produced no live activation to return".to_string(),
            })?;
        let scale = output
            .point
            .and_then(|point| self.graph.point_scale(PointId::new(point)));

        let metrics = ArtifactMetrics {
            operators: self.operators.len(),
            layers: self.layers.len(),
            tensors: self.tensors,
            macs: self.macs,
            requantizations: self.requantizations,
            structural: None,
        };
        let output_binding = TensorBinding {
            ssa_name: output.ssa.clone(),
            shape: output.shape.clone(),
            element: output.element,
            point: output.point,
            scale,
            code_range: Some(output.code_range),
        };
        let module = self.render_module(&output);
        Ok(EmittedModule {
            module,
            input_shape: self.options.input_shape.clone(),
            arguments: self.arguments,
            tensors: self.tensors_named,
            layers: self.layers,
            operators: self.operators,
            output: output_binding,
            metrics,
        })
    }

    /// The manifest carries no spatial extents, so the caller supplies the input
    /// shape; its channel count must still agree with the convolution that
    /// consumes point 0.
    fn check_input_channels(&self) -> Result<(), MlirError> {
        let expected = self
            .graph
            .layers()
            .iter()
            .find(|layer| layer.in_point.get() == 0)
            .map(|layer| layer.in_channels.get());
        if let Some(expected) = expected {
            let found = self.input.shape[3];
            if found != expected as u64 {
                return Err(MlirError::Shape {
                    detail: format!(
                        "input shape {} has {found} channels, but the graph's first convolution reads {expected}",
                        shape_text(&self.input.shape)
                    ),
                });
            }
        }
        Ok(())
    }

    fn last_live_point(&self) -> Option<TensorRef> {
        self.points.values().next_back().cloned()
    }

    fn emit_operator(&mut self, index: usize, op: &GraphOp) -> Result<(), MlirError> {
        match op {
            GraphOp::Conv { layer, .. } => self.emit_conv(index, layer),
            GraphOp::Relu { layer, .. } => self.emit_relu(index, layer),
            GraphOp::Add { .. } => {
                let point = op.point().ok_or_else(|| MlirError::InvalidGraph {
                    detail: format!("graph[{index}]: add without a destination point"),
                })?;
                let source = op.source_point().ok_or_else(|| MlirError::InvalidGraph {
                    detail: format!("graph[{index}]: add without a source point"),
                })?;
                self.emit_add(index, point, source, op.scale().map(|scale| scale.get()))
            }
            GraphOp::UpsampleNearest2x { .. } => {
                let point = op.point().ok_or_else(|| MlirError::InvalidGraph {
                    detail: format!("graph[{index}]: upsample without a point"),
                })?;
                self.emit_upsample(index, point)
            }
            GraphOp::QuantizeU8Output { .. } => self.emit_quantize_output(index),
        }
    }

    fn layer_binding(&self, name: &LayerName) -> Result<&'g Layer, MlirError> {
        self.graph
            .layer_named(name.as_str())
            .ok_or_else(|| MlirError::InvalidGraph {
                detail: format!("graph references unknown layer {}", name.as_str()),
            })
    }

    fn emit_conv(&mut self, index: usize, name: &LayerName) -> Result<(), MlirError> {
        let layer = self.layer_binding(name)?;
        let layer_name = layer.name.as_str().to_string();
        let prefix = self.reserve_prefix(&layer_name)?;

        let source = self.lookup_input(layer.in_point.get(), &layer_name)?;
        if source.shape.len() != 4 {
            return Err(MlirError::Shape {
                detail: format!(
                    "layer {layer_name}: input tensor {} is not rank-4 NHWC",
                    shape_text(&source.shape)
                ),
            });
        }
        let in_channels = layer.in_channels.get();
        let out_channels = layer.out_channels.get();
        let groups = layer.groups.get();
        let kernel = [layer.kernel.h.get() as u64, layer.kernel.w.get() as u64];
        let stride = [layer.stride.h.get() as u64, layer.stride.w.get() as u64];
        let kind = classify(layer, kernel, groups, in_channels, out_channels)?;
        let (padding, pad_amount) = padding_spec(layer);

        if source.shape[3] != in_channels as u64 {
            return Err(MlirError::Shape {
                detail: format!(
                    "layer {layer_name} reads {in_channels} input channels, but its input tensor {} has {}",
                    shape_text(&source.shape),
                    source.shape[3]
                ),
            });
        }
        if stride[0] == 0 || stride[1] == 0 {
            return Err(MlirError::Shape {
                detail: format!("layer {layer_name} has a zero stride"),
            });
        }
        if kernel[0] == 0 || kernel[1] == 0 {
            return Err(MlirError::Shape {
                detail: format!("layer {layer_name} has a zero kernel extent"),
            });
        }
        if pad_amount as u64 >= source.shape[1].min(source.shape[2]) {
            return Err(MlirError::Unsupported {
                feature: "padding".to_string(),
                detail: format!(
                    "layer {layer_name} reflects by {pad_amount} on a {} input; v1 supports single-step reflection only (the amount must stay below both spatial extents)",
                    shape_text(&source.shape)
                ),
            });
        }

        let padded_shape = vec![
            source.shape[0],
            source.shape[1] + 2 * pad_amount as u64,
            source.shape[2] + 2 * pad_amount as u64,
            source.shape[3],
        ];
        if padded_shape[1] < kernel[0] || padded_shape[2] < kernel[1] {
            return Err(MlirError::Shape {
                detail: format!(
                    "layer {layer_name}: padded extent {}x{} is smaller than the {}x{} kernel",
                    padded_shape[1], padded_shape[2], kernel[0], kernel[1]
                ),
            });
        }
        let out_h = (padded_shape[1] - kernel[0]) / stride[0] + 1;
        let out_w = (padded_shape[2] - kernel[1]) / stride[1] + 1;
        if out_h == 0 || out_w == 0 {
            return Err(MlirError::Shape {
                detail: format!("layer {layer_name}: derived an empty output extent"),
            });
        }
        let acc_shape = vec![padded_shape[0], out_h, out_w, out_channels as u64];
        let out_shape = acc_shape.clone();

        let clamp = clamp_for(self.graph, layer)?;
        let requant = requant_for(layer, &layer_name, clamp)?;

        // 1. Padding.
        let padded = if pad_amount > 0 {
            let ssa_name = format!("{prefix}__padded");
            let mode = match padding {
                PaddingSpec::Reflect { .. } => PadMode::Reflect,
                PaddingSpec::Zero { .. } => PadMode::Zero,
                PaddingSpec::None => {
                    return Err(MlirError::Emission {
                        detail: format!(
                            "layer {layer_name}: padding amount without a padding mode"
                        ),
                    });
                }
            };
            let mut text = String::new();
            write_pad(
                &mut text,
                2,
                &ssa_name,
                &source.ssa,
                &source.shape,
                pad_amount,
                mode,
                &padded_shape,
                source.element,
                &format!("{TAG_PAD}#{index}"),
            );
            self.body.push_str(&text);
            let padded_ref = TensorRef {
                ssa: ssa_name,
                shape: padded_shape,
                element: source.element,
                point: source.point,
                code_range: source.code_range,
            };
            self.record(&padded_ref);
            padded_ref
        } else {
            source.clone()
        };

        // Legality: the emitted datapath accumulates in `i32`, so the worst-case
        // magnitude of `bias + sum(code * weight)` must fit. The weight bound is
        // the storage type's, because filter values live in the external blob and
        // are never inlined into the module.
        let acc_bound = accumulator_bound(&padded.code_range, layer, kind, kernel, in_channels)?;

        // 2. Filter argument.
        let mlir_layout = match kind {
            PlannerKind::DepthwiseConv2d => MlirFilterLayout::Hwc,
            PlannerKind::Conv2d | PlannerKind::PointwiseConv2d => MlirFilterLayout::Fhwc,
        };
        let filter_shape = match mlir_layout {
            MlirFilterLayout::Fhwc => vec![
                out_channels as u64,
                kernel[0],
                kernel[1],
                in_channels as u64,
            ],
            MlirFilterLayout::Hwc => vec![kernel[0], kernel[1], out_channels as u64],
        };
        let storage_layout = match kind {
            PlannerKind::DepthwiseConv2d => FilterStorageLayout::Chw,
            PlannerKind::Conv2d | PlannerKind::PointwiseConv2d => FilterStorageLayout::Oihw,
        };
        let weight = weight_reference(self.graph, layer, storage_layout, mlir_layout)?;
        let filter_ssa = format!("{prefix}__filter");
        self.push_argument(MlirArgument {
            ssa_name: filter_ssa.clone(),
            role: ArgumentRole::Filter,
            element: ElementType::I8,
            shape: filter_shape.clone(),
            layer: Some(layer_name.clone()),
            weight: Some(weight.clone()),
        })?;

        // 3. Bias and per-channel multiplier arguments.
        let channel_shape = vec![out_channels as u64];
        let bias_ssa = format!("{prefix}__bias");
        self.push_argument(MlirArgument {
            ssa_name: bias_ssa.clone(),
            role: ArgumentRole::Bias,
            element: ElementType::I32,
            shape: channel_shape.clone(),
            layer: Some(layer_name.clone()),
            weight: None,
        })?;
        let m0_ssa = format!("{prefix}__m0");
        self.push_argument(MlirArgument {
            ssa_name: m0_ssa.clone(),
            role: ArgumentRole::RequantMultiplier,
            element: ElementType::I32,
            shape: channel_shape,
            layer: Some(layer_name.clone()),
            weight: None,
        })?;

        // 4. Accumulator and convolution.
        self.needs_zero_i32 = true;
        let acc_init = format!("{prefix}__acc_init");
        let acc_zero = format!("{prefix}__acc_zero");
        let acc_ssa = format!("{prefix}__acc");
        let acc_type = tensor_type(&acc_shape, ElementType::I32);
        self.line(
            2,
            &format!(
                "{empty} = tensor.empty() : {acc_type}",
                empty = ssa(&acc_init)
            ),
        );
        self.line(
            2,
            &format!(
                "{zero} = linalg.fill ins(%zero_i32 : i32) outs({empty} : {acc_type}) -> {acc_type}",
                zero = ssa(&acc_zero),
                empty = ssa(&acc_init),
            ),
        );
        // The named linalg convolutions take one element type for both the
        // activation and the filter. A widened activation (codes outside a signed
        // byte) is therefore lowered through an explicit `linalg.generic`
        // convolution that widens the codes to the `i32` accumulator: same
        // arithmetic, no narrowing, and no unnamed element type on a named op.
        match padded.element {
            ElementType::I8 => {
                let op_name = match kind {
                    PlannerKind::DepthwiseConv2d => "linalg.depthwise_conv_2d_nhwc_hwc",
                    PlannerKind::Conv2d | PlannerKind::PointwiseConv2d => {
                        "linalg.conv_2d_nhwc_fhwc"
                    }
                };
                self.line(
                    2,
                    &format!(
                        "{acc} = {op} {{dilations = dense<1> : vector<2xi64>, strides = {strides}}} ins({input}, {filter} : {ptype}, {ftype}) outs({init} : {acc_type}) -> {acc_type}{tag}",
                        acc = ssa(&acc_ssa),
                        op = op_name,
                        strides = vector_attr(&stride),
                        input = ssa(&padded.ssa),
                        filter = ssa(&filter_ssa),
                        ptype = tensor_type(&padded.shape, ElementType::I8),
                        ftype = tensor_type(&filter_shape, ElementType::I8),
                        init = ssa(&acc_zero),
                        tag = loc(&format!("{TAG_CONV}#{index}")),
                    ),
                );
            }
            element => {
                // The explicit convolution *accumulates into* its `outs` operand,
                // exactly like the named form does. It therefore takes the same
                // zero-filled accumulator: seeding it from an uninitialized
                // `tensor.empty` would leave the sum of products on top of
                // whatever the buffer happened to hold.
                let mut text = String::new();
                write_generic_conv(
                    &mut text,
                    2,
                    &GenericConv {
                        result: &acc_ssa,
                        result_init: &acc_zero,
                        source: &padded.ssa,
                        filter: &filter_ssa,
                        acc_type: &acc_type,
                        source_shape: &padded.shape,
                        filter_shape: &filter_shape,
                        element,
                        kind,
                        stride,
                    },
                    &format!("{TAG_CONV}#{index}"),
                );
                self.body.push_str(&text);
            }
        }
        self.record(&TensorRef {
            ssa: acc_ssa.clone(),
            shape: acc_shape.clone(),
            element: ElementType::I32,
            point: point_id_of(&layer.out_point),
            code_range: CodeRange::new(-acc_bound, acc_bound),
        });

        // 5. Requantization and saturation epilogue.
        let out_init = format!("{prefix}__out_init");
        let out_ssa = format!("{prefix}__out");
        // The epilogue saturates, so the produced codes are the clamp's: the
        // transport follows from the clamp, not from a fixed byte width.
        let out_element = if clamp.unsigned {
            ElementType::I8
        } else {
            TensorRef::element_for(CodeRange::new(clamp.lower, clamp.upper))?
        };
        let out_type = tensor_type(&out_shape, out_element);
        self.line(
            2,
            &format!(
                "{empty} = tensor.empty() : {out_type}",
                empty = ssa(&out_init)
            ),
        );
        let mut text = String::new();
        write_conv_epilogue(
            &mut text,
            2,
            &ConvEpilogue {
                result: &out_ssa,
                result_init: &out_init,
                accumulator: &acc_ssa,
                bias: &bias_ssa,
                multiplier: &m0_ssa,
                acc_shape: &acc_shape,
                out_shape: &out_shape,
                requant: &requant,
                tag: &format!("{TAG_EPILOGUE}#{index}"),
            },
        );
        self.body.push_str(&text);

        let out_ref = TensorRef {
            ssa: out_ssa.clone(),
            shape: out_shape,
            element: out_element,
            point: point_id_of(&layer.out_point),
            code_range: CodeRange::new(clamp.lower, clamp.upper),
        };
        self.record(&out_ref);
        self.tensors += 1;
        let elements = out_ref.shape.iter().product::<u64>();
        self.macs += elements * kernel[0] * kernel[1] * taps(kind, in_channels);
        self.requantizations += elements;
        self.store_output(layer, &out_ref)?;

        self.layers.push(LayerBinding {
            layer: layer_name.clone(),
            kind,
            values: vec![
                padded.ssa.clone(),
                acc_zero.clone(),
                acc_ssa.clone(),
                out_ssa.clone(),
            ],
            input: tensor_binding(&source),
            output: tensor_binding(&out_ref),
            padding,
            groups,
            kernel: [kernel[0] as u32, kernel[1] as u32],
            stride: [stride[0] as u32, stride[1] as u32],
            quantization: requant.clone(),
            filter: weight,
            bias: BiasContribution::of(&layer.bias_int32),
        });
        self.operators.push(OperatorBinding {
            index,
            kind: operator_kind(kind),
            tag: format!("{TAG_CONV}#{index}"),
            layer: Some(layer_name),
            consumes: vec![source.ssa.clone()],
            produces: vec![out_ref.ssa],
            points: vec![layer.in_point.get()]
                .into_iter()
                .chain(point_id_of(&layer.out_point))
                .collect(),
            quantization: Some(requant),
        });
        Ok(())
    }

    fn emit_relu(&mut self, index: usize, name: &LayerName) -> Result<(), MlirError> {
        let layer = self.layer_binding(name)?;
        if !layer.out_relu {
            return Err(MlirError::InvalidGraph {
                detail: format!(
                    "graph schedules a relu for layer {}, whose saturation regime is not relu",
                    layer.name.as_str()
                ),
            });
        }
        if layer.out_unsigned {
            return Err(MlirError::Unsupported {
                feature: "relu".to_string(),
                detail: format!(
                    "layer {} is unsigned-output; its activation is the [0, output_qmax] head clamp, not a relu",
                    layer.name.as_str()
                ),
            });
        }
        // Clamping the symmetric grid at zero is exactly `relu` followed by
        // requantization (`intref.rs:341-349`), so the marker emits no operation.
        self.operators.push(OperatorBinding {
            index,
            kind: OperatorKind::Relu,
            tag: format!("{}#{index}", OperatorKind::Relu.as_str()),
            layer: Some(layer.name.as_str().to_string()),
            consumes: Vec::new(),
            produces: Vec::new(),
            points: vec![layer.in_point.get()],
            quantization: None,
        });
        Ok(())
    }

    fn emit_add(
        &mut self,
        index: usize,
        point: PointId,
        source_point: PointId,
        declared_trunk_scale: Option<f32>,
    ) -> Result<(), MlirError> {
        let point = point.get();
        let source_point = source_point.get();
        let trunk = self
            .points
            .get(&point)
            .cloned()
            .ok_or_else(|| MlirError::InvalidGraph {
                detail: format!("graph[{index}]: add trunk point {point} is not live"),
            })?;
        let branch =
            self.points
                .get(&source_point)
                .cloned()
                .ok_or_else(|| MlirError::InvalidGraph {
                    detail: format!("graph[{index}]: add branch point {source_point} is not live"),
                })?;
        if trunk.shape != branch.shape {
            return Err(MlirError::Shape {
                detail: format!(
                    "graph[{index}]: add operands differ in shape, trunk {} vs branch {}",
                    shape_text(&trunk.shape),
                    shape_text(&branch.shape)
                ),
            });
        }
        if trunk.element != ElementType::I8 || branch.element != ElementType::I8 {
            return Err(MlirError::Unsupported {
                feature: "add".to_string(),
                detail: format!("graph[{index}]: add operands must be i8 activations"),
            });
        }

        // `GraphOp::Add { scale }` is the destination point's scale, which the
        // graph validates; the branch is requantized onto that grid by
        // `point_scales[source_point] / point_scales[point]` (`intref.rs:261`).
        let trunk_scale = point_scale(self.graph, point, index)?;
        let branch_scale = point_scale(self.graph, source_point, index)?;
        if let Some(declared) = declared_trunk_scale {
            if declared != trunk_scale {
                return Err(MlirError::InvalidGraph {
                    detail: format!(
                        "graph[{index}]: add declares trunk scale {declared} but point {point} has scale {trunk_scale}"
                    ),
                });
            }
        }
        let factor = branch_scale / trunk_scale;

        let clamp = ClampSpec::symmetric_branch(self.graph.act_qmax());
        let (spec, error) = serialize(&[factor], 1, clamp)?;
        let prefix = format!("add_{index}");
        self.reserve_prefix_checked(&prefix)?;
        let out_ssa = format!("{prefix}__out");
        let out_init = format!("{prefix}__init");
        // The reference requantizes and clamps only the branch and leaves the sum
        // unclamped (`intref.rs:257-268`), so the result carries the *sum* of the
        // two operand intervals. The branch interval is the source interval
        // squeezed by the requantization clamp: the emitted datapath clamps it to
        // `[lower, upper]`, and a factor at most one cannot move a code outside the
        // interval it came from, so the intersection is exact. A factor above one
        // would widen, and then only the clamp bounds the branch.
        let branch_range = if factor <= 1.0 {
            CodeRange::new(
                branch.code_range.lo.max(spec.clamp.lower),
                branch.code_range.hi.min(spec.clamp.upper),
            )
        } else {
            CodeRange::new(spec.clamp.lower, spec.clamp.upper)
        };
        let out_range = trunk.code_range.sum(branch_range);
        // Resolved before emitting, so the writer and the recorded binding always
        // agree on the transport type.
        let out_element = TensorRef::element_for(out_range)?;
        let ty = tensor_type(&trunk.shape, out_element);
        self.line(
            2,
            &format!("{empty} = tensor.empty() : {ty}", empty = ssa(&out_init)),
        );
        let mut text = String::new();
        write_add(
            &mut text,
            2,
            &AddOp {
                result: &out_ssa,
                result_init: &out_init,
                trunk: &trunk.ssa,
                branch: &branch.ssa,
                shape: &trunk.shape,
                spec: &spec,
                trunk_element: trunk.element,
                branch_element: branch.element,
                result_element: out_element,
                tag: &format!("{TAG_ADD}#{index}"),
            },
        );
        self.body.push_str(&text);

        let out_ref = TensorRef {
            ssa: out_ssa.clone(),
            shape: trunk.shape.clone(),
            element: out_element,
            point: Some(point),
            code_range: out_range,
        };
        self.record(&out_ref);
        self.tensors += 1;
        self.requantizations += trunk.shape.iter().product::<u64>();
        self.points.insert(point, out_ref.clone());
        self.operators.push(OperatorBinding {
            index,
            kind: OperatorKind::Add,
            tag: format!("{TAG_ADD}#{index}"),
            layer: None,
            consumes: vec![trunk.ssa, branch.ssa],
            produces: vec![out_ref.ssa],
            points: vec![point, source_point],
            quantization: Some(Quantization::from_spec(
                &spec,
                branch_scale,
                trunk_scale,
                error,
            )),
        });
        Ok(())
    }

    fn emit_upsample(&mut self, index: usize, point: PointId) -> Result<(), MlirError> {
        let point = point.get();
        let source = self
            .points
            .get(&point)
            .cloned()
            .ok_or_else(|| MlirError::InvalidGraph {
                detail: format!("graph[{index}]: upsample point {point} is not live"),
            })?;
        if source.shape.len() != 4 {
            return Err(MlirError::Unsupported {
                feature: "upsample".to_string(),
                detail: format!("graph[{index}]: upsample operand is not rank-4 NHWC"),
            });
        }
        let out_shape = vec![
            source.shape[0],
            source.shape[1] * 2,
            source.shape[2] * 2,
            source.shape[3],
        ];
        let prefix = format!("up_{index}");
        self.reserve_prefix_checked(&prefix)?;
        let out_ssa = format!("{prefix}__out");
        let out_init = format!("{prefix}__init");
        let out_type = tensor_type(&out_shape, source.element);
        self.line(
            2,
            &format!(
                "{empty} = tensor.empty() : {out_type}",
                empty = ssa(&out_init)
            ),
        );
        let mut text = String::new();
        write_upsample(
            &mut text,
            2,
            &Upsample {
                result: &out_ssa,
                result_init: &out_init,
                source: &source.ssa,
                out_shape: &out_shape,
                element: source.element,
                tag: &format!("{TAG_UPSAMPLE}#{index}"),
            },
        );
        self.body.push_str(&text);

        // Duplication changes no code, so the range travels with the value.
        let out_ref = TensorRef {
            ssa: out_ssa.clone(),
            shape: out_shape,
            element: source.element,
            point: Some(point),
            code_range: source.code_range,
        };
        self.record(&out_ref);
        self.tensors += 1;
        self.points.insert(point, out_ref.clone());
        self.operators.push(OperatorBinding {
            index,
            kind: OperatorKind::UpsampleNearest2x,
            tag: format!("{TAG_UPSAMPLE}#{index}"),
            layer: None,
            consumes: vec![source.ssa],
            produces: vec![out_ref.ssa],
            points: vec![point],
            quantization: None,
        });
        Ok(())
    }

    fn emit_quantize_output(&mut self, index: usize) -> Result<(), MlirError> {
        let image = self.image.clone().ok_or_else(|| MlirError::InvalidGraph {
            detail: format!(
                "graph[{index}]: quantize_u8_output has no unsigned convolution result to consume"
            ),
        })?;
        self.operators.push(OperatorBinding {
            index,
            kind: OperatorKind::QuantizeU8,
            tag: format!("{}#{index}", OperatorKind::QuantizeU8.as_str()),
            layer: None,
            consumes: vec![image.ssa.clone()],
            produces: Vec::new(),
            points: image.point.into_iter().collect(),
            quantization: None,
        });
        Ok(())
    }

    /// Record a convolution result on its point, or as the graph image.
    ///
    /// The reference interpreter keeps an unsigned head's output in a dedicated
    /// image slot and never in `points` (`intref.rs:204-215`); the graph artifact
    /// may address that slot through [`PointRef::ImageSentinel`] or through a real
    /// point, so both forms are honoured and the image is always updated.
    fn store_output(&mut self, layer: &Layer, value: &TensorRef) -> Result<(), MlirError> {
        let name = layer.name.as_str();
        let out_point = match &layer.out_point {
            PointRef::ImageSentinel => None,
            PointRef::Point(out_point) => Some(out_point.get()),
        };
        if layer.out_unsigned {
            self.image = Some(value.clone());
            if let Some(id) = out_point {
                if id as usize >= self.graph.point_count() {
                    // The frontend encodes the image slot as the sentinel, but an
                    // index equal to the point count is the same slot.
                    if id as usize > self.graph.point_count() {
                        return Err(MlirError::InvalidGraph {
                            detail: format!(
                                "layer {name} writes point {id}, outside the {} declared points",
                                self.graph.point_count()
                            ),
                        });
                    }
                } else {
                    self.points.insert(id, value.clone());
                }
            }
            return Ok(());
        }
        match out_point {
            None => Err(MlirError::InvalidGraph {
                detail: format!(
                    "layer {name} writes the image sentinel but is not marked unsigned"
                ),
            }),
            Some(id) => {
                if id as usize >= self.graph.point_count() {
                    return Err(MlirError::InvalidGraph {
                        detail: format!(
                            "layer {name} writes point {id}, outside the {} declared points",
                            self.graph.point_count()
                        ),
                    });
                }
                self.points.insert(id, value.clone());
                Ok(())
            }
        }
    }

    fn lookup_input(&self, point: u32, layer: &str) -> Result<TensorRef, MlirError> {
        self.points
            .get(&point)
            .cloned()
            .ok_or_else(|| MlirError::InvalidGraph {
                detail: format!("layer {layer}: input point {point} is not live"),
            })
    }

    fn push_argument(&mut self, argument: MlirArgument) -> Result<(), MlirError> {
        if self
            .arguments
            .iter()
            .any(|existing| existing.ssa_name == argument.ssa_name)
        {
            return Err(MlirError::Emission {
                detail: format!("duplicate MLIR argument name {}", argument.ssa_name),
            });
        }
        self.arguments.push(argument);
        Ok(())
    }

    fn reserve_prefix(&mut self, layer_name: &str) -> Result<String, MlirError> {
        let prefix = sanitize(layer_name)?;
        self.reserve_prefix_checked(&prefix)?;
        Ok(prefix)
    }

    fn reserve_prefix_checked(&mut self, prefix: &str) -> Result<(), MlirError> {
        if !self.used_prefixes.insert(prefix.to_string()) {
            return Err(MlirError::Unsupported {
                feature: "symbol naming".to_string(),
                detail: format!(
                    "name {prefix} collides with another layer or operator name after sanitization"
                ),
            });
        }
        Ok(())
    }

    fn line(&mut self, indent: usize, text: &str) {
        push_line(&mut self.body, indent, text);
    }

    /// Name a tensor value in the artifact's tensor table.
    ///
    /// Every value the module refers to by SSA name is recorded once, in
    /// emission order, so a later stage can resolve an operator's operands and
    /// results to shapes and element types without re-deriving them from the
    /// graph (or from generated arithmetic).
    fn record(&mut self, reference: &TensorRef) {
        if self
            .tensors_named
            .iter()
            .any(|named| named.ssa_name == reference.ssa)
        {
            return;
        }
        self.tensors_named.push(TensorBinding {
            ssa_name: reference.ssa.clone(),
            shape: reference.shape.clone(),
            element: reference.element,
            point: reference.point,
            scale: reference
                .point
                .and_then(|point| self.graph.point_scale(PointId::new(point))),
            code_range: Some(reference.code_range),
        });
    }

    fn render_module(&self, output: &TensorRef) -> String {
        let mut text = String::from("module {\n");
        let mut signature = format!("  func.func @{}(", self.options.function_name);
        for (index, argument) in self.arguments.iter().enumerate() {
            if index > 0 {
                signature.push_str(", ");
            }
            let _ = write!(
                signature,
                "{}: {}",
                ssa(&argument.ssa_name),
                argument.mlir_type()
            );
        }
        let result_type = tensor_type(&output.shape, output.element);
        let _ = write!(signature, ") -> {result_type} {{\n");
        text.push_str(&signature);
        if self.needs_zero_i32 {
            push_line(&mut text, 2, "%zero_i32 = arith.constant 0 : i32");
        }
        text.push_str(&self.body);
        push_line(
            &mut text,
            2,
            &format!("return {} : {result_type}", ssa(&output.ssa)),
        );
        text.push_str("  }\n}\n");
        text
    }
}

/// Planner-facing convolution kind, derived from the graph layer.
type PlannerKind = super::artifact::LayerKind;

/// Classify a layer, rejecting the grouped convolutions that v1 cannot emit.
fn classify(
    layer: &Layer,
    kernel: [u64; 2],
    groups: u32,
    in_channels: u32,
    out_channels: u32,
) -> Result<PlannerKind, MlirError> {
    let name = layer.name.as_str();
    match layer.kind {
        GraphLayerKind::DepthwiseConv2d => {
            if groups != in_channels || in_channels != out_channels {
                return Err(MlirError::Unsupported {
                    feature: "depthwise groups".to_string(),
                    detail: format!(
                        "layer {name}: depthwise convolution requires groups == in_channels == out_channels, found groups={groups}, in={in_channels}, out={out_channels}"
                    ),
                });
            }
            Ok(PlannerKind::DepthwiseConv2d)
        }
        GraphLayerKind::PointwiseConv2d => {
            if groups != 1 {
                return Err(MlirError::Unsupported {
                    feature: "grouped convolution".to_string(),
                    detail: format!(
                        "layer {name}: pointwise convolution with groups={groups} is outside the v1 subset"
                    ),
                });
            }
            if kernel != [1, 1] {
                return Err(MlirError::InvalidGraph {
                    detail: format!(
                        "layer {name}: pointwise convolution declared with a {}x{} kernel",
                        kernel[0], kernel[1]
                    ),
                });
            }
            Ok(PlannerKind::PointwiseConv2d)
        }
        GraphLayerKind::Conv2d => {
            if groups != 1 {
                return Err(MlirError::Unsupported {
                    feature: "grouped convolution".to_string(),
                    detail: format!(
                        "layer {name}: groups={groups} is outside the v1 subset; only groups==1 (dense) and depthwise are supported"
                    ),
                });
            }
            if kernel == [1, 1] {
                Ok(PlannerKind::PointwiseConv2d)
            } else {
                Ok(PlannerKind::Conv2d)
            }
        }
    }
}

/// Saturation regime of a layer, matching `intref.rs:341-349`.
fn clamp_for(graph: &Graph, layer: &Layer) -> Result<ClampSpec, MlirError> {
    let name = layer.name.as_str();
    let act_qmax = graph.act_qmax();
    if act_qmax <= 0.0 {
        return Err(MlirError::InvalidGraph {
            detail: "graph act_qmax must be positive".to_string(),
        });
    }
    if layer.out_unsigned {
        if layer.out_relu {
            return Err(MlirError::InvalidGraph {
                detail: format!("layer {name} is marked both relu and unsigned"),
            });
        }
        let output_qmax = graph.output_qmax();
        if output_qmax <= 0.0 {
            return Err(MlirError::InvalidGraph {
                detail: "graph output_qmax must be positive".to_string(),
            });
        }
        Ok(ClampSpec::unsigned_output(output_qmax))
    } else if layer.out_relu {
        Ok(ClampSpec::relu(act_qmax))
    } else {
        Ok(ClampSpec::symmetric(act_qmax))
    }
}

fn requant_for(layer: &Layer, name: &str, clamp: ClampSpec) -> Result<Quantization, MlirError> {
    let values = layer.requant.values();
    let channels = layer.out_channels.get();
    if values.len() != channels as usize {
        return Err(MlirError::InvalidGraph {
            detail: format!(
                "layer {name}: {} requantization coefficients for {channels} output channels",
                values.len()
            ),
        });
    }
    if layer.bias_int32.len() != channels as usize {
        return Err(MlirError::InvalidGraph {
            detail: format!(
                "layer {name}: {} bias_int32 values for {channels} output channels",
                layer.bias_int32.len()
            ),
        });
    }
    let (spec, error) = serialize(values, channels, clamp)?;
    Ok(Quantization::from_spec(
        &spec,
        layer.in_scale.get(),
        layer.out_scale.get(),
        error,
    ))
}

/// Validate the external weight range of a filter and describe it.
fn weight_reference(
    graph: &Graph,
    layer: &Layer,
    storage_layout: FilterStorageLayout,
    mlir_layout: MlirFilterLayout,
) -> Result<WeightReference, MlirError> {
    let stream = graph.weight_stream();
    let packing = match stream.packing {
        Packing::I8x1 => WeightPacking::I8x1,
        Packing::I4x2 => WeightPacking::I4x2,
    };
    let name = layer.name.as_str();
    let code_count = layer.weight_count as u64;
    let byte_count = layer.weight_bytes as u64;
    // i4-x2 stores two signed nibbles per byte, low nibble first
    // (`fpga_st_train::quant::QuantManifest::load_weights`), so its byte count is
    // half its code count; i8-x1 is one code per byte.
    let codes_in_bytes = byte_count * packing.codes_per_byte();
    match packing {
        WeightPacking::I8x1 if code_count != codes_in_bytes => {
            return Err(MlirError::InvalidGraph {
                detail: format!(
                    "layer {name}: i8-x1 packing declares {code_count} codes in {byte_count} bytes"
                ),
            });
        }
        WeightPacking::I4x2 if code_count > codes_in_bytes => {
            return Err(MlirError::InvalidGraph {
                detail: format!(
                    "layer {name}: i4-x2 packing declares {code_count} codes but only {byte_count} bytes are stored"
                ),
            });
        }
        _ => {}
    }
    if layer.weight_offset as u64 + code_count > stream.code_count as u64 {
        return Err(MlirError::InvalidGraph {
            detail: format!(
                "layer {name}: codes [{}..{}) exceed the {} codes available in {}",
                layer.weight_offset,
                layer.weight_offset as u64 + code_count,
                stream.code_count,
                stream.file
            ),
        });
    }
    Ok(WeightReference {
        file: stream.file.clone(),
        packing,
        code_offset: layer.weight_offset as u64,
        code_count,
        byte_offset: layer.weight_byte_offset as u64,
        byte_count,
        storage_layout,
        mlir_layout,
    })
}

fn point_scale(graph: &Graph, point: u32, index: usize) -> Result<f32, MlirError> {
    graph
        .point_scale(PointId::new(point))
        .ok_or_else(|| MlirError::InvalidGraph {
            detail: format!("graph[{index}]: no scale for point {point}"),
        })
}

fn point_id_of(point: &PointRef) -> Option<u32> {
    point.id().map(|id| id.get())
}

fn padding_spec(layer: &Layer) -> (PaddingSpec, u32) {
    match &layer.padding {
        Padding::None => (PaddingSpec::None, 0),
        Padding::Reflect(extent) => (
            PaddingSpec::Reflect {
                amount: extent.get(),
            },
            extent.get(),
        ),
        Padding::Zero(extent) => (
            PaddingSpec::Zero {
                amount: extent.get(),
            },
            extent.get(),
        ),
    }
}

fn operator_kind(kind: PlannerKind) -> OperatorKind {
    match kind {
        PlannerKind::Conv2d => OperatorKind::Conv2d,
        PlannerKind::DepthwiseConv2d => OperatorKind::DepthwiseConv2d,
        PlannerKind::PointwiseConv2d => OperatorKind::PointwiseConv2d,
    }
}

fn taps(kind: PlannerKind, in_channels: u32) -> u64 {
    match kind {
        PlannerKind::DepthwiseConv2d => 1,
        PlannerKind::Conv2d | PlannerKind::PointwiseConv2d => in_channels as u64,
    }
}

fn serialize(
    values: &[f32],
    channels: u32,
    clamp: ClampSpec,
) -> Result<(RequantSpec, f64), MlirError> {
    super::quant::serialize_requant(values, channels, clamp).map_err(|error: QuantError| {
        MlirError::InvalidGraph {
            detail: error.to_string(),
        }
    })
}

fn tensor_binding(reference: &TensorRef) -> TensorBinding {
    TensorBinding {
        ssa_name: reference.ssa.clone(),
        shape: reference.shape.clone(),
        element: reference.element,
        point: reference.point,
        scale: None,
        code_range: Some(reference.code_range),
    }
}

fn ssa(name: &str) -> String {
    format!("%{name}")
}

fn push_line(text: &mut String, indent: usize, line: &str) {
    for _ in 0..indent {
        text.push_str("  ");
    }
    text.push_str(line);
    text.push('\n');
}

/// Sanitize a layer name into an MLIR SSA prefix.
fn sanitize(name: &str) -> Result<String, MlirError> {
    let mut out = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            out.push(character);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return Err(MlirError::Emission {
            detail: "empty layer name".to_string(),
        });
    }
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, 'L');
    }
    Ok(out)
}

fn shape_text(shape: &[u64]) -> String {
    shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn vector_attr(values: &[u64]) -> String {
    if values.iter().all(|value| *value == values[0]) {
        format!("dense<{}> : vector<2xi64>", values[0])
    } else {
        format!(
            "dense<[{}]> : vector<2xi64>",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn parallel_iters(rank: usize) -> String {
    let entries = vec!["\"parallel\""; rank];
    format!("[{}]", entries.join(", "))
}

fn identity_map(rank: usize) -> String {
    let dims = (0..rank)
        .map(|dim| format!("d{dim}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_map<({dims}) -> ({dims})>")
}

/// Map the output indices onto the channel axis, broadcasting a vector over the
/// spatial dimensions.
fn channel_vector_map(rank: usize) -> String {
    let dims = (0..rank)
        .map(|dim| format!("d{dim}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("affine_map<({dims}) -> (d{})>", rank - 1)
}

/// Reflection padding, emitted as a `tensor.pad` region matching
/// `intref.rs::reflect_index` for a single step.
///
/// The region is only evaluated for padded positions, so computing the source
/// index by reflection and extracting there is exactly the reference padding.
fn write_pad(
    text: &mut String,
    indent: usize,
    result: &str,
    source: &str,
    source_shape: &[u64],
    amount: u32,
    mode: PadMode,
    out_shape: &[u64],
    element: ElementType,
    tag: &str,
) {
    let source_type = tensor_type(source_shape, element);
    let out_type = tensor_type(out_shape, element);
    push_line(
        text,
        indent,
        &format!(
            "{result} = tensor.pad {source} low[0, {amount}, {amount}, 0] high[0, {amount}, {amount}, 0] {{",
            result = ssa(result),
            source = ssa(source),
        ),
    );
    let block = indent + 1;
    push_line(
        text,
        block,
        "^bb0(%i0: index, %i1: index, %i2: index, %i3: index):",
    );
    let inner = block + 1;
    match mode {
        PadMode::Zero => {
            push_line(
                text,
                inner,
                &format!("%zero = arith.constant 0 : {}", element.mlir()),
            );
            push_line(
                text,
                inner,
                &format!("tensor.yield %zero : {}", element.mlir()),
            );
        }
        PadMode::Reflect => {
            let mut reflections = Vec::new();
            for (axis, block_arg, extent) in
                [("h", "%i1", source_shape[1]), ("w", "%i2", source_shape[2])]
            {
                push_line(
                    text,
                    inner,
                    &format!("%c_{axis}_pad = arith.constant {amount} : index"),
                );
                push_line(
                    text,
                    inner,
                    &format!("%c_{axis}_zero = arith.constant 0 : index"),
                );
                push_line(
                    text,
                    inner,
                    &format!("%c_{axis}_n = arith.constant {extent} : index"),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%c_{axis}_mirror = arith.constant {} : index",
                        2 * (extent - 1)
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!("%{axis}_shifted = arith.subi {block_arg}, %c_{axis}_pad : index"),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_negative = arith.cmpi slt, %{axis}_shifted, %c_{axis}_zero : index"
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_absolute = arith.subi %c_{axis}_zero, %{axis}_shifted : index"
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_folded = arith.select %{axis}_negative, %{axis}_absolute, %{axis}_shifted : index"
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_overflow = arith.cmpi sge, %{axis}_folded, %c_{axis}_n : index"
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_mirrored = arith.subi %c_{axis}_mirror, %{axis}_folded : index"
                    ),
                );
                push_line(
                    text,
                    inner,
                    &format!(
                        "%{axis}_source = arith.select %{axis}_overflow, %{axis}_mirrored, %{axis}_folded : index"
                    ),
                );
                reflections.push(format!("%{axis}_source"));
            }
            push_line(
                text,
                inner,
                &format!(
                    "%value = tensor.extract {source}[%i0, {}, {}, %i3] : {source_type}",
                    reflections[0],
                    reflections[1],
                    source = ssa(source),
                ),
            );
            push_line(
                text,
                inner,
                &format!("tensor.yield %value : {}", element.mlir()),
            );
        }
    }
    push_line(
        text,
        indent,
        &format!("}} : {source_type} to {out_type}{}", loc(tag)),
    );
}

struct ConvEpilogue<'a> {
    result: &'a str,
    result_init: &'a str,
    accumulator: &'a str,
    bias: &'a str,
    multiplier: &'a str,
    acc_shape: &'a [u64],
    out_shape: &'a [u64],
    requant: &'a Quantization,
    /// Tessera location tag identifying the layer this epilogue belongs to.
    tag: &'a str,
}

/// Requantization and saturation epilogue of a convolution.
///
/// `%acc_b = acc + bias_int32`, then the `(M0, shift)` fixed point multiply with
/// round-half-away-from-zero, then the saturation clamp.
fn write_conv_epilogue(text: &mut String, indent: usize, epilogue: &ConvEpilogue<'_>) {
    let acc_type = tensor_type(epilogue.acc_shape, ElementType::I32);
    let out_type = tensor_type(epilogue.out_shape, ElementType::I8);
    let channel_type = tensor_type(&[epilogue.out_shape[3]], ElementType::I32);
    let maps = [
        identity_map(4),
        channel_vector_map(4),
        channel_vector_map(4),
        identity_map(4),
    ];
    push_line(
        text,
        indent,
        &format!(
            "{result} = linalg.generic {{indexing_maps = [{maps}], iterator_types = {iters}}} ins({acc}, {bias}, {m0} : {acc_type}, {channel_type}, {channel_type}) outs({init} : {out_type}) {{",
            result = ssa(epilogue.result),
            maps = maps.join(", "),
            iters = parallel_iters(4),
            acc = ssa(epilogue.accumulator),
            bias = ssa(epilogue.bias),
            m0 = ssa(epilogue.multiplier),
            init = ssa(epilogue.result_init),
        ),
    );
    let block = indent + 1;
    push_line(
        text,
        block,
        "^bb0(%acc: i32, %bias: i32, %m0: i32, %old: i8):",
    );
    let inner = block + 1;
    push_line(text, inner, "%acc_b = arith.addi %acc, %bias : i32");
    push_line(text, inner, "%acc64 = arith.extsi %acc_b : i32 to i64");
    push_line(text, inner, "%m0_64 = arith.extsi %m0 : i32 to i64");
    write_requant_arithmetic(
        text,
        inner,
        "%acc64",
        "%m0_64",
        epilogue.requant.shift,
        epilogue.requant.clamp,
    );
    push_line(text, inner, "linalg.yield %narrow : i8");
    push_line(
        text,
        indent,
        &format!("}} -> {out_type}{}", loc(epilogue.tag)),
    );
}

struct AddOp<'a> {
    result: &'a str,
    result_init: &'a str,
    trunk: &'a str,
    branch: &'a str,
    shape: &'a [u64],
    spec: &'a RequantSpec,
    /// Element type of the trunk operand.
    trunk_element: ElementType,
    /// Element type of the branch operand.
    branch_element: ElementType,
    /// Element type of the result: the sum's transport, never narrowed.
    result_element: ElementType,
    /// Tessera location tag identifying the graph operator this add implements.
    tag: &'a str,
}

/// Elementwise `add` with the branch requantized onto the destination grid and
/// clamped, and the sum left unclamped (`intref.rs:257-268`).
fn write_add(text: &mut String, indent: usize, op: &AddOp<'_>) {
    let ty = tensor_type(op.shape, op.result_element);
    let trunk_ty = tensor_type(op.shape, op.trunk_element);
    let branch_ty = tensor_type(op.shape, op.branch_element);
    let maps = [identity_map(4), identity_map(4), identity_map(4)];
    push_line(
        text,
        indent,
        &format!(
            "{result} = linalg.generic {{indexing_maps = [{maps}], iterator_types = {iters}}} ins({trunk}, {branch} : {trunk_ty}, {branch_ty}) outs({init} : {ty}) {{",
            result = ssa(op.result),
            maps = maps.join(", "),
            iters = parallel_iters(4),
            trunk = ssa(op.trunk),
            branch = ssa(op.branch),
            init = ssa(op.result_init),
        ),
    );
    let block = indent + 1;
    let result_element = op.result_element.mlir();
    push_line(
        text,
        block,
        &format!(
            "^bb0(%trunk: {trunk}, %branch: {branch}, %old: {result}):",
            trunk = op.trunk_element.mlir(),
            branch = op.branch_element.mlir(),
            result = result_element,
        ),
    );
    let inner = block + 1;
    push_line(
        text,
        inner,
        &format!(
            "%branch32 = arith.extsi %branch : {} to i32",
            op.branch_element.mlir()
        ),
    );
    push_line(
        text,
        inner,
        "%branch64 = arith.extsi %branch32 : i32 to i64",
    );
    let multiplier = op.spec.multiplier.first().copied().unwrap_or(0);
    push_line(
        text,
        inner,
        &format!("%m0_64 = arith.constant {multiplier} : i64"),
    );
    write_requant_arithmetic(
        text,
        inner,
        "%branch64",
        "%m0_64",
        op.spec.shift,
        op.spec.clamp,
    );
    push_line(
        text,
        inner,
        &format!(
            "%trunk32 = arith.extsi %trunk : {} to i32",
            op.trunk_element.mlir()
        ),
    );
    push_line(text, inner, "%narrow32 = arith.extsi %narrow : i8 to i32");
    push_line(text, inner, "%sum32 = arith.addi %trunk32, %narrow32 : i32");
    // The sum is not clamped (`intref.rs:257-268`); it is narrowed to the sum's
    // transport width, which the caller sized to hold both operands.
    push_line(
        text,
        inner,
        &format!("%sum = arith.trunci %sum32 : i32 to {result_element}"),
    );
    push_line(
        text,
        inner,
        &format!("linalg.yield %sum : {result_element}"),
    );
    push_line(text, indent, &format!("}} -> {ty}{}", loc(op.tag)));
}

/// Shared requantization arithmetic: multiply, round half away from zero,
/// arithmetic shift, saturate, narrow to `i8`. Produces `%narrow`.
fn write_requant_arithmetic(
    text: &mut String,
    indent: usize,
    product_lhs: &str,
    product_rhs: &str,
    shift: u32,
    clamp: ClampSpec,
) {
    push_line(
        text,
        indent,
        &format!("%product = arith.muli {product_lhs}, {product_rhs} : i64"),
    );
    push_line(text, indent, "%zero64 = arith.constant 0 : i64");
    push_line(
        text,
        indent,
        "%nonnegative = arith.cmpi sge, %product, %zero64 : i64",
    );
    let (round_positive, round_negative) = rounding_offsets(shift);
    push_line(
        text,
        indent,
        &format!("%round_positive = arith.constant {round_positive} : i64"),
    );
    push_line(
        text,
        indent,
        &format!("%round_negative = arith.constant {round_negative} : i64"),
    );
    push_line(
        text,
        indent,
        "%rounding = arith.select %nonnegative, %round_positive, %round_negative : i64",
    );
    push_line(
        text,
        indent,
        "%rounded = arith.addi %product, %rounding : i64",
    );
    push_line(
        text,
        indent,
        &format!("%shift64 = arith.constant {shift} : i64"),
    );
    push_line(
        text,
        indent,
        "%scaled = arith.shrsi %rounded, %shift64 : i64",
    );
    push_line(
        text,
        indent,
        &format!("%lower = arith.constant {} : i64", clamp.lower),
    );
    push_line(
        text,
        indent,
        "%saturated_low = arith.maxsi %scaled, %lower : i64",
    );
    push_line(
        text,
        indent,
        &format!("%upper = arith.constant {} : i64", clamp.upper),
    );
    push_line(
        text,
        indent,
        "%saturated = arith.minsi %saturated_low, %upper : i64",
    );
    push_line(
        text,
        indent,
        "%narrow = arith.trunci %saturated : i64 to i8",
    );
}

/// Worst-case magnitude of a convolution accumulator.
///
/// The emitted datapath accumulates in `i32`, so the module is only legal while
/// `|bias| + taps * |weight| * |code|` fits. Weights live in the external blob
/// and are never inlined here, so the bound uses the storage type's maximum
/// magnitude, which is the tightest bound the artifact can prove without reading
/// the blob.
fn accumulator_bound(
    input: &CodeRange,
    layer: &Layer,
    kind: PlannerKind,
    kernel: [u64; 2],
    in_channels: u32,
) -> Result<i64, MlirError> {
    let weight_bound: i64 = match layer.weight_bits.get() {
        8 => i8::MAX as i64,
        4 => 7,
        bits => {
            return Err(MlirError::Unsupported {
                feature: "weight width".to_string(),
                detail: format!(
                    "layer {}: weight width {bits} is outside the v1 set {{4, 8}}",
                    layer.name.as_str()
                ),
            });
        }
    };
    let code_bound = input.lo.unsigned_abs().max(input.hi.unsigned_abs()) as i64;
    let taps = (kernel[0] * kernel[1]).saturating_mul(taps(kind, in_channels));
    let bias_bound = layer
        .bias_int32
        .iter()
        .map(|value| i64::from(*value).unsigned_abs() as i64)
        .max()
        .unwrap_or(0);
    let bound = bias_bound.saturating_add(
        (taps as i64)
            .saturating_mul(weight_bound)
            .saturating_mul(code_bound),
    );
    if bound > i32::MAX as i64 {
        return Err(MlirError::Unsupported {
            feature: "accumulator width".to_string(),
            detail: format!(
                "layer {}: worst-case accumulator magnitude {bound} exceeds the i32 datapath \
                 (taps {taps}, |weight| <= {weight_bound}, |code| <= {code_bound}, |bias| <= {bias_bound})",
                layer.name.as_str()
            ),
        });
    }
    Ok(bound)
}

/// An explicit convolution for activations the named linalg form cannot express.
///
/// `linalg.conv_2d_nhwc_fhwc` and `linalg.depthwise_conv_2d_nhwc_hwc` take one
/// element type for the activation and the filter. When a point holds widened
/// codes, the activation is `i16` while the filter stays `i8`, so the
/// convolution is written out as a `linalg.generic` whose body widens both
/// operands to the `i32` accumulator. The arithmetic is the reference's:
/// `acc += code * weight` with no narrowing of the codes.
struct GenericConv<'a> {
    result: &'a str,
    result_init: &'a str,
    source: &'a str,
    filter: &'a str,
    acc_type: &'a str,
    source_shape: &'a [u64],
    filter_shape: &'a [u64],
    /// Element type of the activation operand.
    element: ElementType,
    kind: PlannerKind,
    stride: [u64; 2],
}

fn write_generic_conv(text: &mut String, indent: usize, op: &GenericConv<'_>, tag: &str) {
    let source_type = tensor_type(op.source_shape, op.element);
    let filter_type = tensor_type(op.filter_shape, ElementType::I8);
    let (input_map, filter_map) = match op.kind {
        PlannerKind::DepthwiseConv2d => (
            format!(
                "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * {sh} + d4, d2 * {sw} + d5, d3)>",
                sh = op.stride[0],
                sw = op.stride[1],
            ),
            "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3)>".to_string(),
        ),
        PlannerKind::Conv2d | PlannerKind::PointwiseConv2d => (
            format!(
                "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * {sh} + d4, d2 * {sw} + d5, d6)>",
                sh = op.stride[0],
                sw = op.stride[1],
            ),
            "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>".to_string(),
        ),
    };
    let output_map = "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>";
    let iters = r#"["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]"#;
    push_line(
        text,
        indent,
        &format!(
            "{result} = linalg.generic {{indexing_maps = [{maps}], iterator_types = {iters}}} ins({source}, {filter} : {source_type}, {filter_type}) outs({init} : {acc}) {{",
            result = ssa(op.result),
            maps = [input_map, filter_map, output_map.to_string()].join(", "),
            source = ssa(op.source),
            filter = ssa(op.filter),
            init = ssa(op.result_init),
            acc = op.acc_type,
        ),
    );
    let block = indent + 1;
    push_line(
        text,
        block,
        &format!(
            "^bb0(%activation: {element}, %weight: i8, %accumulator: i32):",
            element = op.element.mlir()
        ),
    );
    let inner = block + 1;
    push_line(
        text,
        inner,
        &format!(
            "%activation32 = arith.extsi %activation : {} to i32",
            op.element.mlir()
        ),
    );
    push_line(text, inner, "%weight32 = arith.extsi %weight : i8 to i32");
    push_line(
        text,
        inner,
        "%product = arith.muli %activation32, %weight32 : i32",
    );
    push_line(
        text,
        inner,
        "%sum = arith.addi %accumulator, %product : i32",
    );
    push_line(text, inner, "linalg.yield %sum : i32");
    push_line(text, indent, &format!("}} -> {}{}", op.acc_type, loc(tag)));
}

struct Upsample<'a> {
    result: &'a str,
    result_init: &'a str,
    source: &'a str,
    out_shape: &'a [u64],
    /// Element type of the source; duplication never changes the transport.
    element: ElementType,
    /// Tessera location tag identifying the graph operator this reads for.
    tag: &'a str,
}

/// Nearest-neighbour 2x duplication as a broadcast read.
fn write_upsample(text: &mut String, indent: usize, op: &Upsample<'_>) {
    let in_shape = vec![
        op.out_shape[0],
        op.out_shape[1] / 2,
        op.out_shape[2] / 2,
        op.out_shape[3],
    ];
    let in_type = tensor_type(&in_shape, op.element);
    let out_type = tensor_type(op.out_shape, op.element);
    let read_map =
        "affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 2, d2 floordiv 2, d3)>".to_string();
    let maps = [read_map, identity_map(4)];
    push_line(
        text,
        indent,
        &format!(
            "{result} = linalg.generic {{indexing_maps = [{maps}], iterator_types = {iters}}} ins({source} : {in_type}) outs({init} : {out_type}) {{",
            result = ssa(op.result),
            maps = maps.join(", "),
            iters = parallel_iters(4),
            source = ssa(op.source),
            init = ssa(op.result_init),
        ),
    );
    let block = indent + 1;
    let element = op.element.mlir();
    push_line(
        text,
        block,
        &format!("^bb0(%value: {element}, %old: {element}):"),
    );
    push_line(text, block + 1, &format!("linalg.yield %value : {element}"));
    push_line(text, indent, &format!("}} -> {out_type}{}", loc(op.tag)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GroupCount, Padding};
    use crate::mlir::LayerKind;
    use crate::mlir::fixtures::{sample_graph, sample_options};
    use crate::mlir::quant::{ClampKind, RoundingMode};

    fn spec(shift: u32, clamp: ClampSpec) -> RequantSpec {
        RequantSpec {
            multiplier: vec![104_887],
            shift,
            rounding: RoundingMode::HalfAwayFromZero,
            clamp,
        }
    }

    fn conv_quantization(shift: u32, clamp: ClampSpec) -> Quantization {
        Quantization::from_spec(&spec(shift, clamp), 0.007874016, 0.009141972, 0.0)
    }

    #[test]
    fn fixture_module_is_deterministic_and_covers_every_operator() {
        let graph = sample_graph();
        let options = sample_options();
        let first = emit_module(&graph, &options).expect("emit");
        let second = emit_module(&graph, &options).expect("emit again");
        assert_eq!(
            first.module, second.module,
            "emission must be deterministic"
        );

        // Input plus three arguments per convolution (filter, bias, multiplier).
        assert_eq!(first.arguments.len(), 1 + 3 * 3);
        assert_eq!(first.arguments[0].role, ArgumentRole::Input);
        assert_eq!(first.arguments[0].mlir_type(), "tensor<1x6x6x3xi8>");
        assert!(
            first
                .arguments
                .iter()
                .any(|argument| argument.role == ArgumentRole::Filter
                    && argument.weight.as_ref().unwrap().file == "weights.bin")
        );

        // Operator coverage and order.
        let kinds: Vec<OperatorKind> = first.operators.iter().map(|op| op.kind).collect();
        assert_eq!(
            kinds,
            vec![
                OperatorKind::Conv2d,
                OperatorKind::Relu,
                OperatorKind::PointwiseConv2d,
                OperatorKind::Add,
                OperatorKind::PointwiseConv2d,
                OperatorKind::QuantizeU8,
            ]
        );
        assert_eq!(first.layers.len(), 3);
        assert_eq!(first.layers[0].layer, "enc.0");
        assert_eq!(first.layers[0].padding, PaddingSpec::Reflect { amount: 1 });
        assert_eq!(first.layers[1].kind, LayerKind::PointwiseConv2d);
        assert!(first.layers[2].quantization.clamp.unsigned);

        // Text: reflect pad, named convolutions, requant epilogue, add, head clamp.
        let text = &first.module;
        assert!(text.starts_with("module {\n"));
        assert!(text.contains("func.func @tessera_graph(%input: tensor<1x6x6x3xi8>"));
        assert!(text.ends_with("  }\n}\n"));
        assert!(
            text.contains("%enc_0__padded = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0]")
        );
        assert!(text.contains("%c_h_mirror = arith.constant 10 : index"));
        assert!(text.contains(
            "%enc_0__acc = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}"
        ));
        assert!(text.contains("ins(%enc_0__acc, %enc_0__bias, %enc_0__m0 :"));
        assert!(text.contains("%add_3__out = linalg.generic"));
        assert!(text.contains("%upper = arith.constant 255 : i64"));
        assert!(text.contains("return %head__out : tensor<1x6x6x3xi8>"));
        assert!(!text.contains("linalg.depthwise_conv_2d_nhwc_hwc"));

        // Static metrics derived from the walk.
        assert_eq!(first.metrics.layers, 3);
        assert_eq!(first.metrics.operators, 6);
        assert_eq!(first.metrics.tensors, 4);
        // 6*6*4*(3*3*3) + 6*6*4*(1*1*3) + 6*6*3*(1*1*4)
        assert_eq!(first.metrics.macs, 3888 + 432 + 432);
        // 144 (enc.0) + 144 (res) + 144 (add) + 108 (head)
        assert_eq!(first.metrics.requantizations, 540);

        assert_eq!(first.output.ssa_name, "head__out");
        assert_eq!(first.output.shape, vec![1, 6, 6, 3]);
    }

    #[test]
    fn input_channel_mismatch_is_rejected() {
        let graph = sample_graph();
        let options = MlirOptions::new([1, 6, 6, 5]);
        assert!(matches!(
            emit_module(&graph, &options),
            Err(MlirError::Shape { .. })
        ));
    }

    #[test]
    fn grouped_convolution_is_rejected_with_a_reason() {
        let mut graph = sample_graph();
        graph.layers[0].groups = GroupCount::try_new(3).expect("groups");
        match emit_module(&graph, &sample_options()) {
            Err(MlirError::Unsupported { feature, detail }) => {
                assert_eq!(feature, "grouped convolution");
                assert!(detail.contains("enc.0"), "{detail}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn depthwise_with_inconsistent_groups_is_rejected() {
        let mut graph = sample_graph();
        graph.layers[1].kind = GraphLayerKind::DepthwiseConv2d;
        match emit_module(&graph, &sample_options()) {
            Err(MlirError::Unsupported { feature, .. }) => assert_eq!(feature, "depthwise groups"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn multi_step_reflection_is_rejected_instead_of_guessed() {
        let mut graph = sample_graph();
        graph.layers[0].padding = Padding::reflect(6).expect("padding");
        match emit_module(&graph, &sample_options()) {
            Err(MlirError::Unsupported { feature, detail }) => {
                assert_eq!(feature, "padding");
                assert!(detail.contains("single-step"), "{detail}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn relu_marker_for_a_plain_layer_is_rejected() {
        let mut graph = sample_graph();
        graph.ops[1] = crate::graph::GraphOp::Relu {
            layer: crate::graph::LayerName::new("res.0.c1"),
        };
        assert!(matches!(
            emit_module(&graph, &sample_options()),
            Err(MlirError::InvalidGraph { .. })
        ));
    }

    #[test]
    fn add_over_a_dead_point_is_rejected() {
        let mut graph = sample_graph();
        graph.ops[3] = crate::graph::GraphOp::Add {
            point: crate::graph::PointId::new(1),
            source_point: crate::graph::PointId::new(9),
            scale: crate::graph::Scale::try_new("scale", 1.0).expect("scale"),
        };
        assert!(matches!(
            emit_module(&graph, &sample_options()),
            Err(MlirError::InvalidGraph { .. })
        ));
    }

    #[test]
    fn reflect_pad_region_is_deterministic_and_explicit() {
        let mut first = String::new();
        write_pad(
            &mut first,
            2,
            "enc_0__padded",
            "input",
            &[1, 32, 32, 3],
            1,
            PadMode::Reflect,
            &[1, 34, 34, 3],
            ElementType::I8,
            &format!("{TAG_PAD}#0"),
        );
        let mut second = String::new();
        write_pad(
            &mut second,
            2,
            "enc_0__padded",
            "input",
            &[1, 32, 32, 3],
            1,
            PadMode::Reflect,
            &[1, 34, 34, 3],
            ElementType::I8,
            &format!("{TAG_PAD}#0"),
        );
        assert_eq!(first, second);
        assert!(first.contains("tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0]"));
        assert!(first.contains("%c_h_mirror = arith.constant 62 : index"));
        assert!(first.contains("%c_w_mirror = arith.constant 62 : index"));
        assert!(first.contains(
            "%value = tensor.extract %input[%i0, %h_source, %w_source, %i3] : tensor<1x32x32x3xi8>"
        ));
        assert!(first.contains("} : tensor<1x32x32x3xi8> to tensor<1x34x34x3xi8>"));
    }

    #[test]
    fn zero_pad_yields_a_constant() {
        let mut text = String::new();
        write_pad(
            &mut text,
            2,
            "enc_0__padded",
            "input",
            &[1, 8, 8, 4],
            2,
            PadMode::Zero,
            &[1, 12, 12, 4],
            ElementType::I8,
            &format!("{TAG_PAD}#0"),
        );
        assert!(text.contains("low[0, 2, 2, 0] high[0, 2, 2, 0]"));
        assert!(text.contains("%zero = arith.constant 0 : i8"));
        assert!(!text.contains("tensor.extract"));
    }

    #[test]
    fn conv_epilogue_preserves_quantization_and_clamp() {
        let mut text = String::new();
        let quantization = conv_quantization(26, ClampSpec::relu(127.0));
        write_conv_epilogue(
            &mut text,
            2,
            &ConvEpilogue {
                result: "enc_0__out",
                result_init: "enc_0__out_init",
                accumulator: "enc_0__acc",
                bias: "enc_0__bias",
                multiplier: "enc_0__m0",
                acc_shape: &[1, 32, 32, 32],
                out_shape: &[1, 32, 32, 32],
                requant: &quantization,
                tag: &format!("{TAG_EPILOGUE}#0"),
            },
        );
        assert!(text.contains("%round_positive = arith.constant 33554432 : i64"));
        assert!(text.contains("%round_negative = arith.constant 33554431 : i64"));
        assert!(text.contains("%shift64 = arith.constant 26 : i64"));
        assert!(text.contains("%lower = arith.constant 0 : i64"));
        assert!(text.contains("%upper = arith.constant 127 : i64"));
        assert!(text.contains("%acc_b = arith.addi %acc, %bias : i32"));
        assert!(text.contains("linalg.yield %narrow : i8"));
        assert_eq!(quantization.clamp.kind, ClampKind::Relu);
    }

    #[test]
    fn unsigned_head_clamps_to_255_and_stays_in_i8() {
        let mut text = String::new();
        let quantization = conv_quantization(31, ClampSpec::unsigned_output(255.0));
        write_conv_epilogue(
            &mut text,
            2,
            &ConvEpilogue {
                result: "head__out",
                result_init: "head__out_init",
                accumulator: "head__acc",
                bias: "head__bias",
                multiplier: "head__m0",
                acc_shape: &[1, 8, 8, 3],
                out_shape: &[1, 8, 8, 3],
                requant: &quantization,
                tag: &format!("{TAG_EPILOGUE}#0"),
            },
        );
        // `arith` accepts only signless integers, so unsigned codes are the same
        // bit pattern in an i8 tensor.
        assert!(text.contains("%upper = arith.constant 255 : i64"));
        assert!(text.contains("%narrow = arith.trunci %saturated : i64 to i8"));
        assert!(text.contains("} -> tensor<1x8x8x3xi8>"));
        assert!(quantization.clamp.unsigned);
    }

    #[test]
    fn add_requantizes_the_branch_and_leaves_the_sum_unclamped() {
        let mut text = String::new();
        let spec = spec(25, ClampSpec::symmetric_branch(127.0));
        write_add(
            &mut text,
            2,
            &AddOp {
                result: "add_3__out",
                result_init: "add_3__init",
                trunk: "up3_1__out",
                branch: "res_0_c2__out",
                shape: &[1, 16, 16, 32],
                spec: &spec,
                trunk_element: ElementType::I8,
                branch_element: ElementType::I8,
                result_element: ElementType::I16,
                tag: &format!("{TAG_ADD}#3"),
            },
        );
        assert!(text.contains(
            "ins(%up3_1__out, %res_0_c2__out : tensor<1x16x16x32xi8>, tensor<1x16x16x32xi8>)"
        ));
        assert!(text.contains("%m0_64 = arith.constant 104887 : i64"));
        assert!(text.contains("%shift64 = arith.constant 25 : i64"));
        // Branch clamp is [-127, 127]; the sum itself is not clamped.
        assert!(text.contains("%lower = arith.constant -127 : i64"));
        assert!(text.contains("%sum32 = arith.addi %trunk32, %narrow32 : i32"));
        assert!(!text.contains("%sum_upper"));
    }

    #[test]
    fn upsample_reads_with_a_floordiv_map() {
        let mut text = String::new();
        write_upsample(
            &mut text,
            2,
            &Upsample {
                result: "up_5__out",
                result_init: "up_5__init",
                source: "up3_1__out",
                out_shape: &[1, 16, 16, 32],
                element: ElementType::I8,
                tag: &format!("{TAG_UPSAMPLE}#5"),
            },
        );
        assert!(
            text.contains("affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 2, d2 floordiv 2, d3)>")
        );
        assert!(text.contains(
            "ins(%up3_1__out : tensor<1x8x8x32xi8>) outs(%up_5__init : tensor<1x16x16x32xi8>)"
        ));
        assert!(text.contains("} -> tensor<1x16x16x32xi8>"));
    }

    /// The emitted reflect region must reproduce `intref.rs::reflect_index`.
    ///
    /// `reflect_index` loops until the index is in range; the emitted region
    /// performs a single fold, so this test shows the two agree for every index
    /// the emitter accepts (padding strictly below both spatial extents) and that
    /// they would disagree outside that range, which is why the emitter rejects
    /// larger amounts instead of guessing.
    #[test]
    fn reflect_pad_formula_matches_the_reference_indexer() {
        // Verbatim transliteration of the ops `write_pad` emits.
        fn emitted(index: i64, amount: i64, extent: i64) -> i64 {
            let shifted = index - amount;
            let negative = shifted < 0;
            let absolute = 0 - shifted;
            let folded = if negative { absolute } else { shifted };
            let overflow = folded >= extent;
            let mirrored = 2 * (extent - 1) - folded;
            if overflow { mirrored } else { folded }
        }
        // `intref.rs::reflect_index`.
        fn reference(index: i64, extent: i64) -> i64 {
            if extent <= 1 {
                return 0;
            }
            let mut value = index;
            while value < 0 || value >= extent {
                if value < 0 {
                    value = -value;
                } else {
                    value = 2 * (extent - 1) - value;
                }
            }
            value
        }

        for extent in [2i64, 3, 5, 32] {
            for amount in 0..extent {
                for index in 0..(extent + 2 * amount) {
                    assert_eq!(
                        emitted(index, amount, extent),
                        reference(index - amount, extent),
                        "extent={extent} amount={amount} index={index}"
                    );
                }
            }
        }

        // Outside the accepted range a single fold is not enough, so the emitter
        // rejects those amounts (`multi_step_reflection_is_rejected_instead_of_guessed`).
        let extent = 4i64;
        let amount = 2 * extent - 1; // rejected by the emitter
        assert_ne!(
            emitted(0, amount, extent),
            reference(0 - amount, extent),
            "multi-step reflection must not be silently approximated"
        );
    }

    #[test]
    fn naming_is_stable_and_collisions_are_rejected() {
        assert_eq!(sanitize("enc.0.dw").unwrap(), "enc_0_dw");
        assert_eq!(sanitize("res.0.c1").unwrap(), "res_0_c1");
        assert_eq!(sanitize("0stem").unwrap(), "L0stem");
        assert!(sanitize("").is_err());
    }

    #[test]
    fn strided_attribute_is_explicit() {
        assert_eq!(vector_attr(&[2, 2]), "dense<2> : vector<2xi64>");
        assert_eq!(vector_attr(&[1, 2]), "dense<[1, 2]> : vector<2xi64>");
    }
}
