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

use super::quant::{ClampSpec, RequantSpec, RoundingMode};

/// Value of [`MlirArtifact::format`].
pub const MLIR_ARTIFACT_FORMAT: &str = "tessera.mlir";

/// Value of [`MlirArtifact::schema_version`].
pub const MLIR_ARTIFACT_SCHEMA_VERSION: u32 = 1;

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
            ElementType::I32 => "i32",
            ElementType::I64 => "i64",
        }
    }

    /// Width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            ElementType::I8 => 8,
            ElementType::I32 => 32,
            ElementType::I64 => 64,
        }
    }

    /// Width in bytes.
    pub const fn bytes(self) -> u64 {
        (self.bits() / 8) as u64
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
}

/// One operator of the emitted graph, in execution order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorBinding {
    /// Position in the graph artifact's execution order.
    pub index: usize,
    pub kind: OperatorKind,
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
    /// Sum over all allocated `memref` shapes, in bytes.
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
    /// Function arguments in declaration order.
    pub arguments: Vec<MlirArgument>,
    /// Convolution layers.
    pub layers: Vec<LayerBinding>,
    /// Operators in execution order.
    pub operators: Vec<OperatorBinding>,
    /// Graph output tensor.
    pub output: TensorBinding,
    pub metrics: ArtifactMetrics,
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
        Ok(())
    }

    /// Look up a layer binding by its stable name.
    pub fn layer(&self, name: &str) -> Option<&LayerBinding> {
        self.layers.iter().find(|layer| layer.layer == name)
    }
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
            arguments: Vec::new(),
            layers: Vec::new(),
            operators: Vec::new(),
            output: TensorBinding {
                ssa_name: "out".to_string(),
                shape: vec![1, 1, 1, 1],
                element: ElementType::I8,
                point: None,
                scale: None,
            },
            metrics: ArtifactMetrics {
                operators: 0,
                layers: 0,
                tensors: 0,
                macs: 0,
                requantizations: 0,
                structural: None,
            },
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
