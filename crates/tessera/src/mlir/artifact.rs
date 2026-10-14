//! Persisted MLIR artifacts.
//!
//! ADR-0002 and the concept book make every compiler boundary an independently
//! versioned, inspectable artifact. This module owns the `tessera.mlir`
//! artifact: the textual MLIR for one graph, the deterministic list of function
//! arguments (weights stay symbolic external references), the per-operator
//! semantics the planner and RTL backends bind to, and provenance.
//!
//! Nothing here runs MLIR. The artifact is produced by [`crate::mlir::emit`]
//! and, when the `melior` feature is enabled, replaced by the
//! planning-stage module produced by [`crate::mlir::pipeline`].

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use super::analysis::ModuleAnalysis;
use super::quant::{ClampSpec, RequantSpec, RoundingMode};
use crate::graph::GraphArtifact;

/// Value of [`MlirArtifact::format`].
pub const MLIR_ARTIFACT_FORMAT: &str = "tessera.mlir";

/// Value of [`MlirArtifact::schema_version`].
///
/// Schema 2 adds the concrete input shape, the activation tensor table and the
/// structured analysis of the optimized module; without them a planning-stage
/// artifact cannot drive primitive lowering on its own.
pub const MLIR_ARTIFACT_SCHEMA_VERSION: u32 = 2;

/// Default name of the emitted MLIR function.
pub const DEFAULT_FUNCTION_NAME: &str = "tessera_graph";

/// Which optimization stage the stored module text was captured at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlirStage {
    /// Tensor-level `linalg` on tensors, as emitted from Graph IR.
    Tensor,
    /// Bufferized `memref` + `affine` planning primitives after the pinned pass
    /// pipeline.
    Planning,
}

impl MlirStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            MlirStage::Tensor => "tensor",
            MlirStage::Planning => "planning",
        }
    }
}

impl fmt::Display for MlirStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Element type of an emitted tensor or memref.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    /// `i8`: quantized activations and weights. Unsigned image codes are stored
    /// with the identical bit pattern, because `arith` accepts only signless
    /// integer types.
    I8,
    /// `i16`: activation codes whose range is wider than one signed byte.
    ///
    /// An elementwise `add` requantizes and clamps only its *branch* operand and
    /// leaves the sum unclamped (`fpga_st_train/src/intref.rs:257-268`), so the
    /// point it writes holds `[-2*qmax, 2*qmax]` codes. They are carried in
    /// `i16` — never narrowed to `i8`, which would wrap — and a convolution
    /// reading such a point widens them explicitly. The exact interval is
    /// recorded per tensor in [`TensorBinding::code_range`].
    I16,
    /// `i32`: convolution accumulators, biases, quantized linear codes and
    /// fixed-point multipliers.
    I32,
    /// `i64`: intermediate product of the requantization multiply.
    I64,
}

impl ElementType {
    /// MLIR type spelling.
    pub const fn mlir(self) -> &'static str {
        match self {
            ElementType::I8 => "i8",
            ElementType::I16 => "i16",
            ElementType::I32 => "i32",
            ElementType::I64 => "i64",
        }
    }

    /// Width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            ElementType::I8 => 8,
            ElementType::I16 => 16,
            ElementType::I32 => 32,
            ElementType::I64 => 64,
        }
    }

    /// Width in bytes.
    pub const fn bytes(self) -> u64 {
        (self.bits() / 8) as u64
    }
}

/// Inclusive interval of activation codes a tensor carries.
///
/// This is exact, not a container width: a point written by a convolution holds
/// `[-qmax, qmax]` (or `[0, qmax]`, `[0, output_qmax]` for the saturated and
/// unsigned regimes), while a point written by an `add` holds the sum of the two
/// operand intervals. The planner and the RTL backend read it to size their
/// datapath instead of assuming one signed byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeRange {
    /// Smallest code the tensor can hold.
    pub lo: i64,
    /// Largest code the tensor can hold.
    pub hi: i64,
}

impl CodeRange {
    /// Interval of a saturated point.
    pub const fn new(lo: i64, hi: i64) -> Self {
        CodeRange { lo, hi }
    }

    /// Narrowest signed container width, in bits, that holds the interval.
    ///
    /// `[0, 255]` needs 9 signed bits even though the unsigned image head stores
    /// it in an `i8` container, which is why this reports the signed width rather
    /// than the element width.
    pub fn signed_bits(&self) -> u32 {
        let bound = self.lo.unsigned_abs().max(self.hi.unsigned_abs());
        if bound == 0 {
            return 1;
        }
        (u64::BITS - bound.leading_zeros()) + 1
    }

    /// Sum of two intervals, as an `add` of the two produces.
    pub const fn sum(self, other: Self) -> Self {
        CodeRange {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Whether every code fits in a signed byte.
    pub const fn fits_i8(self) -> bool {
        self.lo >= i8::MIN as i64 && self.hi <= i8::MAX as i64
    }
}

/// Physical packing of a filter in the external weight file.
///
/// Persisted counterpart of [`crate::graph::Packing`], carrying the same two
/// layouts (`i8-x1`, `i4-x2`) under the artifact schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightPacking {
    /// One signed byte per code.
    I8x1,
    /// Two signed nibbles per byte, low nibble first.
    I4x2,
}

impl WeightPacking {
    /// Codes per byte of the external weight file.
    pub const fn codes_per_byte(self) -> u64 {
        match self {
            WeightPacking::I8x1 => 1,
            WeightPacking::I4x2 => 2,
        }
    }

    /// Bits per code.
    pub const fn code_bits(self) -> u32 {
        match self {
            WeightPacking::I8x1 => 8,
            WeightPacking::I4x2 => 4,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            WeightPacking::I8x1 => "i8-x1",
            WeightPacking::I4x2 => "i4-x2",
        }
    }
}

/// Layout of the filter codes as they are stored in `weights.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterStorageLayout {
    /// Dense convolution: `[out_channels][in_channels][kh][kw]` (`groups == 1`).
    Oihw,
    /// Depthwise convolution: `[channels][kh][kw]` with
    /// `groups == in_channels == out_channels`.
    Chw,
}

/// Layout the emitted MLIR expects for the filter tensor argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlirFilterLayout {
    /// `linalg.conv_2d_nhwc_fhwc`: `[out_channels][kh][kw][in_channels]`.
    Fhwc,
    /// `linalg.depthwise_conv_2d_nhwc_hwc`: `[kh][kw][channels]`.
    Hwc,
}

/// External reference to filter codes; weight values are never inlined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightReference {
    /// Name of the external weight file, e.g. `weights.bin`.
    pub file: String,
    pub packing: WeightPacking,
    /// Index of the first code of this filter in the unpacked code stream.
    pub code_offset: u64,
    /// Number of codes belonging to this filter.
    pub code_count: u64,
    /// Byte offset of the first encoded byte in the external file.
    pub byte_offset: u64,
    /// Number of encoded bytes belonging to this filter.
    pub byte_count: u64,
    pub storage_layout: FilterStorageLayout,
    pub mlir_layout: MlirFilterLayout,
}

/// Role of one function argument of the emitted MLIR function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgumentRole {
    /// Quantized input activation.
    Input,
    /// Filter codes, in the MLIR layout recorded in
    /// [`WeightReference::mlir_layout`].
    Filter,
    /// Per-output-channel `i32` bias (`bias_int32`).
    Bias,
    /// Per-output-channel fixed-point multiplier.
    RequantMultiplier,
    /// Per-output-channel shift. Constant across channels for one operator, but
    /// kept as a vector argument so the datapath reads it uniformly.
    RequantShift,
    /// Scalar branch multiplier of an elementwise `add`.
    AddMultiplier,
    /// Scalar branch shift of an elementwise `add`.
    AddShift,
}

/// One argument of the emitted MLIR function, in declaration order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlirArgument {
    /// SSA name without the leading `%`.
    pub ssa_name: String,
    pub role: ArgumentRole,
    pub element: ElementType,
    pub shape: Vec<u64>,
    /// Owning layer, when the argument belongs to a convolution.
    pub layer: Option<String>,
    /// External weight reference, present for [`ArgumentRole::Filter`].
    pub weight: Option<WeightReference>,
}

impl MlirArgument {
    /// MLIR type spelling, e.g. `tensor<1x32x32x3xi8>`.
    pub fn mlir_type(&self) -> String {
        tensor_type(&self.shape, self.element)
    }
}

/// A named activation tensor of the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorBinding {
    /// SSA name without the leading `%`.
    pub ssa_name: String,
    pub shape: Vec<u64>,
    pub element: ElementType,
    /// Quantization point index this tensor lives on, when the graph assigns one.
    pub point: Option<u32>,
    /// Scale of the point, when known.
    pub scale: Option<f32>,
    /// Exact interval of activation codes this value can hold, when the emitter
    /// can bound it. See [`CodeRange`]: this is what tells a planner or a backend
    /// that an `i16` value is a 9-bit code in a 16-bit transport, not a generic
    /// 16-bit integer.
    pub code_range: Option<CodeRange>,
}

/// Padding applied before a convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PaddingSpec {
    /// No padding.
    None,
    /// Constant zero padding of `amount` on every spatial edge.
    Zero { amount: u32 },
    /// Reflection padding that does not repeat the edge, matching
    /// `intref.rs::reflect_index`.
    Reflect { amount: u32 },
}

impl PaddingSpec {
    /// Padding amount on each spatial edge.
    pub const fn amount(self) -> u32 {
        match self {
            PaddingSpec::None => 0,
            PaddingSpec::Zero { amount } | PaddingSpec::Reflect { amount } => amount,
        }
    }
}

/// Operator kind of a convolution layer, as the planner sees it.
///
/// This is the *persisted* classification: it is versioned with the artifact
/// schema and is deliberately separate from the in-memory
/// [`crate::graph::LayerKind`] the frontend owns (ADR-0002, concept book §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Dense convolution, `groups == 1`, kernel larger than 1x1.
    Conv2d,
    /// Depthwise convolution, `groups == in_channels == out_channels`.
    DepthwiseConv2d,
    /// Dense 1x1 convolution, `groups == 1`.
    PointwiseConv2d,
}

/// Kind of an emitted graph operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    Conv2d,
    DepthwiseConv2d,
    PointwiseConv2d,
    /// Scheduling marker; a convolution whose quantization clamps to
    /// `[0, qmax]` already performs the ReLU.
    Relu,
    /// Elementwise branch requantization plus add.
    Add,
    /// Nearest-neighbour 2x duplication.
    UpsampleNearest2x,
    /// Unsigned image output: clamp to `[0, output_qmax]`.
    QuantizeU8,
}

impl OperatorKind {
    /// Stable lowercase name. It is also the location-tag role of the operators
    /// the emitter implements as a single structured operation.
    pub const fn as_str(self) -> &'static str {
        match self {
            OperatorKind::Conv2d => "conv2d",
            OperatorKind::DepthwiseConv2d => "depthwise_conv2d",
            OperatorKind::PointwiseConv2d => "pointwise_conv2d",
            OperatorKind::Relu => "relu",
            OperatorKind::Add => "add",
            OperatorKind::UpsampleNearest2x => "upsample_nearest2x",
            OperatorKind::QuantizeU8 => "quantize_u8",
        }
    }
}

/// Quantization contract of one emitted operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    /// Per-output-channel fixed-point multipliers.
    pub multiplier: Vec<i32>,
    /// Uniform shift applied after the multiply.
    pub shift: u32,
    pub rounding: RoundingMode,
    pub clamp: ClampSpec,
    /// Input quantization scale.
    pub in_scale: f32,
    /// Output quantization scale.
    pub out_scale: f32,
    /// Worst relative error of the fixed-point approximation.
    pub max_relative_error: f64,
}

impl Quantization {
    /// Build the persisted form from a serialized requantization spec.
    pub fn from_spec(spec: &RequantSpec, in_scale: f32, out_scale: f32, error: f64) -> Self {
        Quantization {
            multiplier: spec.multiplier.clone(),
            shift: spec.shift,
            rounding: spec.rounding,
            clamp: spec.clamp,
            in_scale,
            out_scale,
            max_relative_error: error,
        }
    }
}

/// Whether a layer's `bias_int32` vector contributes anything.
///
/// The bias codes themselves stay in the external weight blob; this records the
/// one decision downstream stages need, so no stage has to guess it from
/// lowered arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BiasContribution {
    /// Every entry is zero, so the bias add is an exact identity and the
    /// planner models the layer without a bias tensor.
    Identity,
    /// At least one entry is non-zero; the planner must carry a bias tensor.
    NonZero,
}

impl BiasContribution {
    /// Classify a `bias_int32` vector.
    pub fn of(bias_int32: &[i32]) -> Self {
        if bias_int32.iter().any(|value| *value != 0) {
            BiasContribution::NonZero
        } else {
            BiasContribution::Identity
        }
    }

    /// Whether the planner must carry a bias tensor for the layer.
    pub const fn requires_tensor(self) -> bool {
        matches!(self, BiasContribution::NonZero)
    }
}

/// A convolution layer, pre-bound to MLIR values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerBinding {
    /// Stable layer name from the graph artifact, e.g. `enc.0.dw`.
    pub layer: String,
    pub kind: LayerKind,
    /// SSA names this layer defines, in emission order.
    pub values: Vec<String>,
    pub input: TensorBinding,
    pub output: TensorBinding,
    pub padding: PaddingSpec,
    pub groups: u32,
    pub kernel: [u32; 2],
    pub stride: [u32; 2],
    pub quantization: Quantization,
    pub filter: WeightReference,
    /// Whether the layer's int32 bias is non-zero.
    pub bias: BiasContribution,
}

/// One operator of the emitted graph, in execution order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorBinding {
    /// Position in the graph artifact's execution order.
    pub index: usize,
    pub kind: OperatorKind,
    /// Stable identifier of this operator inside the emitted module.
    ///
    /// The emitter writes it into the MLIR location of every operation that
    /// implements the operator (`loc("tessera:conv#0")`), so the optimized IR
    /// still says which operator each loop nest came from. See
    /// [`crate::mlir::analysis`].
    pub tag: String,
    /// Owning layer for convolution and scheduling operators.
    pub layer: Option<String>,
    /// SSA names consumed, in operand order.
    pub consumes: Vec<String>,
    /// SSA names produced. Empty for the `relu` scheduling marker.
    pub produces: Vec<String>,
    /// Quantization points involved, in order.
    pub points: Vec<u32>,
    /// Branch requantization of `add`; the operator's own contract otherwise.
    pub quantization: Option<Quantization>,
}

/// Structural metrics of the emitted module text.
///
/// These are counts derived from the IR text. They are static structural
/// estimates, not measured DDR traffic or cycle counts
/// (`nngen/tools/mlir_optimizer/README.md:72`), and must be reported as such.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralMetrics {
    pub linalg_ops: usize,
    pub affine_loops: usize,
    pub memref_allocs: usize,
    /// Sum over the statically shaped `memref` allocations, in bytes.
    ///
    /// Allocations whose shape is not static contribute nothing and are counted
    /// separately by [`crate::mlir::analysis::AnalysisCounts::dynamic_buffers`];
    /// when there are any, this total is partial rather than measured.
    pub allocated_bytes: u64,
}

/// Graph-derived and structural metrics recorded next to the module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactMetrics {
    /// Number of operators in the graph artifact.
    pub operators: usize,
    /// Number of convolution layers.
    pub layers: usize,
    /// Number of activation/accumulator tensors named by the module.
    pub tensors: usize,
    /// Multiply-accumulate count of the emitted convolutions (static).
    pub macs: u64,
    /// Requantization multiplies (static): one per output element per operator.
    pub requantizations: u64,
    /// Present once the planning-stage pipeline has run.
    pub structural: Option<StructuralMetrics>,
}

/// Identity of the MLIR toolchain that produced a planning-stage module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeliorProvenance {
    /// Melior release the layer is written against; the crate is pinned to it
    /// because both the pass names and the Melior API are release specific.
    pub melior_version: String,
    /// Pass pipeline, in execution order.
    pub pipeline: Vec<PipelineStage>,
}

/// One pass of the optimization pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Pass name, e.g. `one-shot-bufferize`.
    pub pass: String,
    /// Pass options rendered as `{key=value,...}`.
    pub options: BTreeMap<String, String>,
    /// Whether the pass is nested in `func.func(...)`.
    pub nested_in_func: bool,
}

impl PipelineStage {
    /// Stage with no options, not nested.
    pub fn module(pass: &str) -> Self {
        PipelineStage {
            pass: pass.to_string(),
            options: BTreeMap::new(),
            nested_in_func: false,
        }
    }

    /// Stage nested in `func.func(...)`.
    pub fn func(pass: &str) -> Self {
        PipelineStage {
            pass: pass.to_string(),
            options: BTreeMap::new(),
            nested_in_func: true,
        }
    }

    /// Module-level stage with options.
    pub fn with_options(pass: &str, options: &[(&str, &str)]) -> Self {
        PipelineStage {
            pass: pass.to_string(),
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            nested_in_func: false,
        }
    }

    /// Stage nested in `func.func(...)` with options.
    pub fn func_with(pass: &str, options: &[(&str, &str)]) -> Self {
        PipelineStage {
            pass: pass.to_string(),
            options: options
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            nested_in_func: true,
        }
    }

    /// Render this stage: `pass{key=value,...}`. An option with an empty value
    /// is a valueless flag and renders as the bare key, which is how MLIR spells
    /// boolean pass options (`one-shot-bufferize{bufferize-function-boundaries}`).
    pub fn render(&self) -> String {
        if self.options.is_empty() {
            return self.pass.clone();
        }
        let body = self
            .options
            .iter()
            .map(|(key, value)| {
                if value.is_empty() {
                    key.clone()
                } else {
                    format!("{key}={value}")
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("{}{{{}}}", self.pass, body)
    }
}

/// Provenance of an MLIR artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirProvenance {
    /// Format of the source graph artifact.
    pub source_format: String,
    /// Schema version of the source graph artifact.
    pub source_schema_version: u32,
    /// Content hash of the source graph artifact.
    pub source_content_hash: String,
    /// Tessera compiler version that produced this artifact.
    pub compiler_version: String,
    /// MLIR identity, present for planning-stage modules.
    pub melior: Option<MeliorProvenance>,
    /// Deterministic hash of [`MlirArtifact::module`].
    pub module_hash: String,
}

/// A complete `tessera.mlir` artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlirArtifact {
    pub format: String,
    pub schema_version: u32,
    /// Name of the emitted function.
    pub function: String,
    pub stage: MlirStage,
    /// Textual MLIR.
    pub module: String,
    /// Concrete input activation shape in `NHWC` order.
    ///
    /// The manifest contract carries no spatial extents, so the shape the module
    /// was emitted for is recorded here: every later stage that has to propagate
    /// shapes (primitive lowering, layout search) reads it instead of being told
    /// again on the command line.
    pub input_shape: Vec<u64>,
    /// Function arguments in declaration order.
    pub arguments: Vec<MlirArgument>,
    /// Activation tensors the module names, in emission order.
    pub tensors: Vec<TensorBinding>,
    /// Convolution layers.
    pub layers: Vec<LayerBinding>,
    /// Operators in execution order.
    pub operators: Vec<OperatorBinding>,
    /// Graph output tensor.
    pub output: TensorBinding,
    pub metrics: ArtifactMetrics,
    /// Structured analysis of [`MlirArtifact::module`], present for
    /// planning-stage modules produced by [`crate::mlir::optimize`].
    pub analysis: Option<ModuleAnalysis>,
    pub provenance: MlirProvenance,
}

impl MlirArtifact {
    /// JSON rendering with a stable, pretty layout.
    pub fn to_json(&self) -> Result<String, MlirError> {
        serde_json::to_string_pretty(self).map_err(|error| MlirError::Serialization {
            detail: error.to_string(),
        })
    }

    /// Parse an artifact from its JSON rendering, checking format and schema.
    pub fn from_json(text: &str) -> Result<Self, MlirError> {
        let artifact: MlirArtifact =
            serde_json::from_str(text).map_err(|error| MlirError::Serialization {
                detail: error.to_string(),
            })?;
        artifact.validate_header()?;
        Ok(artifact)
    }

    /// Write the JSON artifact, creating parent directories.
    pub fn save(&self, path: &Path) -> Result<(), MlirError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|error| MlirError::Io {
                    detail: format!("{}: {error}", parent.display()),
                })?;
            }
        }
        std::fs::write(path, self.to_json()? + "\n").map_err(|error| MlirError::Io {
            detail: format!("{}: {error}", path.display()),
        })
    }

    /// Read a JSON artifact from disk.
    pub fn load(path: &Path) -> Result<Self, MlirError> {
        let text = std::fs::read_to_string(path).map_err(|error| MlirError::Io {
            detail: format!("{}: {error}", path.display()),
        })?;
        Self::from_json(&text)
    }

    /// Write only the textual MLIR, for tools that consume `.mlir` directly.
    pub fn save_module(&self, path: &Path) -> Result<(), MlirError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|error| MlirError::Io {
                    detail: format!("{}: {error}", parent.display()),
                })?;
            }
        }
        std::fs::write(path, &self.module).map_err(|error| MlirError::Io {
            detail: format!("{}: {error}", path.display()),
        })
    }

    /// Reject artifacts whose format or schema this build does not understand.
    pub fn validate_header(&self) -> Result<(), MlirError> {
        if self.format != MLIR_ARTIFACT_FORMAT {
            return Err(MlirError::IncompatibleArtifact {
                detail: format!(
                    "expected artifact format {MLIR_ARTIFACT_FORMAT}, found {}",
                    self.format
                ),
            });
        }
        if self.schema_version != MLIR_ARTIFACT_SCHEMA_VERSION {
            return Err(MlirError::IncompatibleArtifact {
                detail: format!(
                    "expected schema version {MLIR_ARTIFACT_SCHEMA_VERSION}, found {}",
                    self.schema_version
                ),
            });
        }
        if self.input_shape.len() != 4 || self.input_shape.contains(&0) {
            return Err(MlirError::Shape {
                detail: format!(
                    "artifact records input shape {:?}, which is not a non-empty NHWC shape",
                    self.input_shape
                ),
            });
        }
        if self.module.is_empty() {
            return Err(MlirError::IncompatibleArtifact {
                detail: "artifact carries no module text".to_string(),
            });
        }
        // The recorded hash is recomputed from the stored text rather than
        // trusted: it is part of the artifact's identity, and a stale one would
        // otherwise let edited text masquerade as the module an analysis
        // describes.
        let actual = content_hash(&self.module);
        if self.provenance.module_hash != actual {
            return Err(MlirError::Provenance {
                detail: format!(
                    "the artifact records module hash {}, but its stored text hashes to {actual}",
                    self.provenance.module_hash
                ),
            });
        }
        Ok(())
    }

    /// Re-check every hash and link this artifact promises.
    ///
    /// This is the check a later stage runs before trusting the artifact: the
    /// recorded module hash must match the module text, the recorded source hash
    /// must match the graph artifact it descends from, and a planning-stage
    /// module must carry an analysis of exactly the stored text.
    pub fn verify(
        &self,
        source: Option<&GraphArtifact>,
    ) -> Result<Vec<ProvenanceCheck>, MlirError> {
        self.validate_header()?;
        let mut checks = Vec::new();
        let module_hash = content_hash(&self.module);
        checks.push(ProvenanceCheck {
            check: "module hash".to_string(),
            ok: module_hash == self.provenance.module_hash,
            detail: format!(
                "records {}, text hashes to {module_hash}",
                self.provenance.module_hash
            ),
        });
        if let Some(graph) = source {
            let ok = self.provenance.source_content_hash == graph.provenance.manifest_sha256
                && self.provenance.source_format == graph.format
                && self.provenance.source_schema_version == graph.schema_version;
            checks.push(ProvenanceCheck {
                check: "descends from graph artifact".to_string(),
                ok,
                detail: format!(
                    "mlir source {} schema {} hash {}, graph is {} schema {} hash {}",
                    self.provenance.source_format,
                    self.provenance.source_schema_version,
                    self.provenance.source_content_hash,
                    graph.format,
                    graph.schema_version,
                    graph.provenance.manifest_sha256
                ),
            });
        }
        match (&self.stage, &self.analysis) {
            (MlirStage::Planning, recorded) => {
                // A plan may only descend from a module the compiler's own
                // pinned pipeline produced; a hand-edited pipeline claim is not
                // evidence that optimized this text.
                checks.push(self.verify_pipeline_provenance());
                match recorded {
                    Some(analysis) => {
                        let ok = analysis.module_hash == self.provenance.module_hash
                            && analysis.module_hash == module_hash;
                        checks.push(ProvenanceCheck {
                            check: "analysis describes the stored module".to_string(),
                            ok,
                            detail: format!(
                                "analysis records {}, artifact records {}, text hashes to \
                                 {module_hash}",
                                analysis.module_hash, self.provenance.module_hash
                            ),
                        });
                        // The link above is only a self-declared string. This
                        // check recomputes the analysis from the stored text, so
                        // a rewritten `tags`, `nests`, `buffers` or `operations`
                        // map that keeps the recorded hash is caught instead of
                        // trusted.
                        checks.push(self.verify_rederived_analysis(analysis));
                    }
                    None => {
                        checks.push(ProvenanceCheck {
                            check: "analysis describes the stored module".to_string(),
                            ok: false,
                            detail: "planning-stage artifact carries no structured analysis"
                                .to_string(),
                        });
                        checks.push(ProvenanceCheck {
                            check: "analysis matches the re-derived structure".to_string(),
                            ok: false,
                            detail: "there is no structured analysis to re-derive against"
                                .to_string(),
                        });
                    }
                }
            }
            (MlirStage::Tensor, _) => checks.push(ProvenanceCheck {
                check: "analysis describes the stored module".to_string(),
                ok: self.analysis.is_none(),
                detail: "tensor-stage artifact is not analyzed".to_string(),
            }),
        }
        Ok(checks)
    }

    /// Fail unless every provenance check holds.
    pub fn check(&self, source: Option<&GraphArtifact>) -> Result<(), MlirError> {
        for check in self.verify(source)? {
            if !check.ok {
                return Err(MlirError::Provenance {
                    detail: format!("{}: {}", check.check, check.detail),
                });
            }
        }
        Ok(())
    }

    /// The compiler's supported planning provenance, or why it does not hold.
    ///
    /// The pipeline is pinned: one pass order is supported, and a planning-stage
    /// artifact is only usable if it records that this build ran it. A
    /// hand-written pipeline claim is not evidence, so it is rejected.
    fn check_planning_provenance(&self) -> Result<(), String> {
        if self.stage != MlirStage::Planning {
            return Err(format!(
                "the artifact is at stage {}; primitive lowering is driven by the planning-stage \
                 optimized module",
                self.stage
            ));
        }
        let Some(melior) = &self.provenance.melior else {
            return Err(
                "the artifact records no Melior provenance, so it was not produced by the \
                 compiler's pinned optimization pipeline"
                    .to_string(),
            );
        };
        if melior.melior_version != super::pipeline::PINNED_MELIOR_VERSION {
            return Err(format!(
                "the artifact records melior {}, but this build pins {}",
                melior.melior_version,
                super::pipeline::PINNED_MELIOR_VERSION
            ));
        }
        let supported = super::pipeline::default_pipeline();
        if melior.pipeline != supported {
            return Err(format!(
                "the artifact records a pipeline of {} passes, not the compiler's pinned pipeline \
                 of {} passes",
                melior.pipeline.len(),
                supported.len()
            ));
        }
        Ok(())
    }

    /// The pinned-pipeline provenance as a check, for [`MlirArtifact::verify`].
    fn verify_pipeline_provenance(&self) -> ProvenanceCheck {
        match self.check_planning_provenance() {
            Ok(()) => {
                let melior = self
                    .provenance
                    .melior
                    .as_ref()
                    .expect("checked present just above");
                ProvenanceCheck {
                    check: "records the compiler's pinned pipeline".to_string(),
                    ok: true,
                    detail: format!(
                        "melior {} ran {} passes",
                        melior.melior_version,
                        melior.pipeline.len()
                    ),
                }
            }
            Err(detail) => ProvenanceCheck {
                check: "records the compiler's pinned pipeline".to_string(),
                ok: false,
                detail,
            },
        }
    }

    /// Re-derive the structured analysis from the stored module text.
    ///
    /// The analysis is recomputed by Melior from [`MlirArtifact::module`],
    /// with the melior identity and pipeline the artifact records. It is the
    /// authoritative description of the module; the embedded
    /// [`MlirArtifact::analysis`] is only a cache of it and is never trusted
    /// without being compared against this.
    pub fn reanalyze(&self) -> Result<ModuleAnalysis, MlirError> {
        let (version, pipeline) = match &self.provenance.melior {
            Some(melior) => (melior.melior_version.clone(), melior.pipeline.clone()),
            None => (
                super::pipeline::PINNED_MELIOR_VERSION.to_string(),
                Vec::new(),
            ),
        };
        super::analysis::analyze(&self.module, &version, &pipeline)
    }

    /// Re-derive the embedded analysis and compare it, field by field.
    ///
    /// This is the check that binds a plan to its module: the recorded analysis
    /// cannot satisfy it by editing metadata, because it is recomputed from the
    /// module text. In a build without the `melior` feature the re-derivation is
    /// impossible, and the check fails loudly rather than passing on metadata
    /// that was never verified.
    fn verify_rederived_analysis(&self, recorded: &ModuleAnalysis) -> ProvenanceCheck {
        match self.reanalyze() {
            Ok(derived) => {
                let diffs = recorded.diff(&derived);
                let ok = diffs.is_empty();
                ProvenanceCheck {
                    check: "analysis matches the re-derived structure".to_string(),
                    ok,
                    detail: if ok {
                        format!(
                            "re-derived from the stored text with Melior {} ({} operations, {} \
                             loops, {} allocations, {} loop nests)",
                            derived.melior_version,
                            derived.counts.operations,
                            derived.counts.loops,
                            derived.counts.buffers,
                            derived.nests.len()
                        )
                    } else {
                        format!(
                            "the recorded analysis does not describe the stored module: {}",
                            diffs.join("; ")
                        )
                    },
                }
            }
            Err(error) => ProvenanceCheck {
                check: "analysis matches the re-derived structure".to_string(),
                ok: false,
                detail: format!(
                    "the analysis cannot be re-derived ({error}); this build cannot verify an \
                     embedded analysis, so it does not report one as verified"
                ),
            },
        }
    }

    /// The structured analysis of a planning-stage module.
    ///
    /// Primitive lowering is driven by the optimized IR, so this re-derives the
    /// analysis from the module text with Melior and refuses an artifact whose
    /// recorded analysis disagrees with it — a rewritten `tags`, `nests`,
    /// `buffers` or `operations` map is caught here even if it kept the recorded
    /// `module_hash`. A module with dynamically shaped allocations is refused
    /// too: planning needs shapes whose size is known.
    ///
    /// Without the `melior` feature the analysis cannot be re-derived at all, so
    /// this errors instead of lowering from unverified metadata.
    pub fn analyzed(&self) -> Result<&ModuleAnalysis, MlirError> {
        let analysis = self.analysis.as_ref().ok_or_else(|| MlirError::Analysis {
            detail: format!(
                "artifact is at stage {} without a structured analysis; run the optimizer \
                 (`tessera optimize`) before lowering to primitives",
                self.stage
            ),
        })?;
        let recorded = content_hash(&self.module);
        if analysis.module_hash != recorded {
            return Err(MlirError::Analysis {
                detail: format!(
                    "analysis records module hash {}, the artifact's text hashes to {recorded}",
                    analysis.module_hash
                ),
            });
        }
        self.check_planning_provenance()
            .map_err(|detail| MlirError::Analysis { detail })?;
        let derived = self.reanalyze().map_err(|error| MlirError::Analysis {
            detail: format!(
                "the analysis of the stored module cannot be re-derived ({error}); refusing to \
                 lower from analysis metadata that cannot be verified"
            ),
        })?;
        let diffs = analysis.diff(&derived);
        if !diffs.is_empty() {
            return Err(MlirError::Analysis {
                detail: format!(
                    "the recorded analysis does not describe the stored module: {}",
                    diffs.join("; ")
                ),
            });
        }
        if derived.counts.dynamic_buffers > 0 {
            return Err(MlirError::Analysis {
                detail: format!(
                    "the optimized module allocates {} dynamically shaped buffer(s); planning \
                     requires statically shaped allocations whose size is known, and refuses to \
                     plan against a size it cannot determine",
                    derived.counts.dynamic_buffers
                ),
            });
        }
        Ok(analysis)
    }

    /// The recorded input shape as `[N, H, W, C]`.
    pub fn input_shape_nhwc(&self) -> Result<[u64; 4], MlirError> {
        self.input_shape
            .as_slice()
            .try_into()
            .map_err(|_| MlirError::Shape {
                detail: format!("artifact input shape {:?} is not rank 4", self.input_shape),
            })
    }

    /// Find an activation tensor by SSA name.
    pub fn tensor(&self, ssa_name: &str) -> Option<&TensorBinding> {
        self.tensors.iter().find(|t| t.ssa_name == ssa_name)
    }

    /// Find a function argument by role and, for filter arguments, by layer.
    pub fn argument(&self, role: ArgumentRole, layer: Option<&str>) -> Option<&MlirArgument> {
        self.arguments
            .iter()
            .find(|a| a.role == role && layer.is_none_or(|name| a.layer.as_deref() == Some(name)))
    }

    /// Look up a layer binding by its stable name.
    pub fn layer(&self, name: &str) -> Option<&LayerBinding> {
        self.layers.iter().find(|layer| layer.layer == name)
    }
}

/// Outcome of one provenance check on an MLIR artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceCheck {
    /// What was checked.
    pub check: String,
    /// Whether it held.
    pub ok: bool,
    /// The evidence: both hashes, both stages, whatever the check compared.
    pub detail: String,
}

/// Options that Graph IR cannot supply, because the manifest contract does not
/// carry spatial dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlirOptions {
    /// Input activation shape in `NHWC` order.
    pub input_shape: Vec<u64>,
    /// Emitted function name.
    pub function_name: String,
}

impl MlirOptions {
    /// Options for the given `NHWC` input shape.
    pub fn new(input_shape: [u64; 4]) -> Self {
        MlirOptions {
            input_shape: input_shape.to_vec(),
            function_name: DEFAULT_FUNCTION_NAME.to_string(),
        }
    }

    /// Override the emitted function name.
    pub fn with_function_name(mut self, name: &str) -> Self {
        self.function_name = name.to_string();
        self
    }

    /// Validate the input shape before emission.
    pub fn validate(&self) -> Result<(), MlirError> {
        if self.input_shape.len() != 4 {
            return Err(MlirError::Shape {
                detail: format!(
                    "input shape must be NHWC with four dimensions, found {} ({:?})",
                    self.input_shape.len(),
                    self.input_shape
                ),
            });
        }
        for (axis, extent) in self.input_shape.iter().enumerate() {
            if *extent == 0 {
                return Err(MlirError::Shape {
                    detail: format!(
                        "input shape dimension {axis} is zero ({:?})",
                        self.input_shape
                    ),
                });
            }
        }
        if self.function_name.is_empty() {
            return Err(MlirError::Emission {
                detail: "function name must not be empty".to_string(),
            });
        }
        if !self
            .function_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(MlirError::Emission {
                detail: format!(
                    "function name {} is not a valid MLIR symbol",
                    self.function_name
                ),
            });
        }
        Ok(())
    }
}

/// Errors raised by the MLIR layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlirError {
    /// The source graph artifact is internally inconsistent or unusable.
    InvalidGraph { detail: String },
    /// The graph uses a feature outside the validated v1 subset.
    Unsupported { feature: String, detail: String },
    /// Shapes or channel counts do not agree.
    Shape { detail: String },
    /// Emission produced something the emitter itself rejects.
    Emission { detail: String },
    /// A pass pipeline could not be built.
    Pipeline { detail: String },
    /// Optimization was requested from a build without the `melior` feature.
    MeliorUnavailable { detail: String },
    /// Melior reported a parse, verification or pass failure.
    Melior { detail: String },
    /// Artifact format or schema mismatch.
    IncompatibleArtifact { detail: String },
    /// An artifact's recorded provenance does not match its contents.
    Provenance { detail: String },
    /// The structured analysis of a module is missing or inconsistent.
    Analysis { detail: String },
    /// Artifact (de)serialization failure.
    Serialization { detail: String },
    /// Filesystem failure.
    Io { detail: String },
}

impl fmt::Display for MlirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlirError::InvalidGraph { detail } => write!(f, "invalid graph: {detail}"),
            MlirError::Unsupported { feature, detail } => {
                write!(f, "unsupported graph feature {feature}: {detail}")
            }
            MlirError::Shape { detail } => write!(f, "shape error: {detail}"),
            MlirError::Emission { detail } => write!(f, "MLIR emission error: {detail}"),
            MlirError::Pipeline { detail } => write!(f, "pass pipeline error: {detail}"),
            MlirError::MeliorUnavailable { detail } => {
                write!(f, "Melior optimization unavailable: {detail}")
            }
            MlirError::Melior { detail } => write!(f, "Melior error: {detail}"),
            MlirError::IncompatibleArtifact { detail } => write!(f, "artifact mismatch: {detail}"),
            MlirError::Provenance { detail } => write!(f, "provenance: {detail}"),
            MlirError::Analysis { detail } => write!(f, "module analysis: {detail}"),
            MlirError::Serialization { detail } => write!(f, "serialization error: {detail}"),
            MlirError::Io { detail } => write!(f, "io error: {detail}"),
        }
    }
}

impl std::error::Error for MlirError {}

/// MLIR tensor type spelling for a shape and element type.
pub fn tensor_type(shape: &[u64], element: ElementType) -> String {
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x");
    format!("tensor<{dims}x{}>", element.mlir())
}

/// MLIR memref type spelling for a shape and element type.
pub fn memref_type(shape: &[u64], element: ElementType) -> String {
    let dims = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join("x");
    format!("memref<{dims}x{}>", element.mlir())
}

/// Stable non-cryptographic content hash used for provenance. Two modules that
/// differ in any byte hash differently, which is all the artifact cache key
/// needs (`docs/concept-book.md:3.3`).
pub fn content_hash(text: &str) -> String {
    // FNV-1a, 128-bit, split into two independently mixed 64-bit lanes so that
    // long documents do not collapse into one weak lane.
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut low = OFFSET;
    let mut high = OFFSET ^ 0x9e37_79b9_7f4a_7c15;
    for (index, byte) in text.as_bytes().iter().enumerate() {
        low ^= *byte as u64;
        low = low.wrapping_mul(PRIME);
        if index % 8 == 0 {
            high ^= (*byte as u64).rotate_left(13);
            high = high.wrapping_mul(PRIME);
        }
    }
    format!("tessera-fnv1a128:{low:016x}{high:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_types_render_deterministically() {
        assert_eq!(
            tensor_type(&[1, 32, 32, 3], ElementType::I8),
            "tensor<1x32x32x3xi8>"
        );
        assert_eq!(
            memref_type(&[1, 16, 16, 64], ElementType::I32),
            "memref<1x16x16x64xi32>"
        );
    }

    #[test]
    fn pipeline_stage_encodes_options_in_sorted_order() {
        let stage = PipelineStage::func_with("affine-loop-fusion", &[("mode", "producer")]);
        assert_eq!(stage.render(), "affine-loop-fusion{mode=producer}");
        assert_eq!(
            PipelineStage::module("canonicalize").render(),
            "canonicalize"
        );
    }

    #[test]
    fn content_hash_is_stable_and_sensitive() {
        let a = content_hash("module {}");
        assert_eq!(a, content_hash("module {}"));
        assert_ne!(a, content_hash("module { }"));
        assert!(a.starts_with("tessera-fnv1a128:"));
    }

    #[test]
    fn options_reject_bad_shapes_and_names() {
        assert!(MlirOptions::new([1, 32, 32, 3]).validate().is_ok());
        assert!(MlirOptions::new([1, 0, 32, 3]).validate().is_err());
        let mut short = MlirOptions::new([1, 32, 32, 3]);
        short.input_shape = vec![1, 32, 3];
        assert!(matches!(short.validate(), Err(MlirError::Shape { .. })));
        assert!(
            MlirOptions::new([1, 32, 32, 3])
                .with_function_name("bad name")
                .validate()
                .is_err()
        );
    }

    #[test]
    fn artifact_header_is_checked() {
        let mut artifact = MlirArtifact {
            format: MLIR_ARTIFACT_FORMAT.to_string(),
            schema_version: MLIR_ARTIFACT_SCHEMA_VERSION,
            function: DEFAULT_FUNCTION_NAME.to_string(),
            stage: MlirStage::Tensor,
            module: "module {}".to_string(),
            input_shape: vec![1, 4, 4, 3],
            arguments: Vec::new(),
            tensors: Vec::new(),
            layers: Vec::new(),
            operators: Vec::new(),
            output: TensorBinding {
                ssa_name: "out".to_string(),
                shape: vec![1, 1, 1, 1],
                element: ElementType::I8,
                point: None,
                scale: None,
                code_range: Some(CodeRange::new(0, 255)),
            },
            metrics: ArtifactMetrics {
                operators: 0,
                layers: 0,
                tensors: 0,
                macs: 0,
                requantizations: 0,
                structural: None,
            },
            analysis: None,
            provenance: MlirProvenance {
                source_format: "tessera.graph".to_string(),
                source_schema_version: 1,
                source_content_hash: content_hash("graph"),
                compiler_version: "0.1.0".to_string(),
                melior: None,
                module_hash: content_hash("module {}"),
            },
        };
        assert!(artifact.validate_header().is_ok());
        let json = artifact.to_json().unwrap();
        let parsed = MlirArtifact::from_json(&json).unwrap();
        assert_eq!(parsed, artifact);

        artifact.schema_version = 99;
        assert!(matches!(
            artifact.validate_header(),
            Err(MlirError::IncompatibleArtifact { .. })
        ));
    }
}
