//! Tessera SystemVerilog RTL backend (v1).
//!
//! The backend consumes a *validated* hardware plan and emits a deterministic
//! RTL artifact bundle (SystemVerilog sources, a self-checking testbench,
//! memory images, and metadata). It rejects every plan construct it cannot
//! faithfully implement; it never reinterprets an unsupported layout.
//!
//! # Supported subset (`tessera.pw_dot.i8i8.v1`)
//!
//! Exactly one schedule step, and that step must be:
//!
//! * a **pointwise (1x1) convolution** (`PointwiseConv2d`, or `Conv2d` with a
//!   1x1 kernel) with `groups == 1`, `stride == [1, 1]` and
//!   `tap_parallel == 1` — i.e. a signed int8 x signed int8 dot-product
//!   engine over `in_channels`;
//! * `in_channels` folded into `simd` activation lanes per input beat, so
//!   `k = in_channels / simd` beats form one reduction window;
//! * `out_channels` folded into `pe` lanes per output beat, so
//!   `nf = out_channels / pe` output beats are produced per window (the same
//!   SF/NF folding used by the NNgen `sts_mvau` streaming MAC unit);
//! * symmetric quantization: both zero-points must be zero;
//! * weights resident in a generated inline ROM (the tensor must be listed in
//!   `ExecutionPlan::weights`), in signed int8 or the manifest `i4-x2` packing,
//!   which the backend unpacks (low nibble first, signed);
//! * activations streamed from a `Register` or `Fifo` buffer through a
//!   valid/ready interface, and the output streamed into a `Register`/`Fifo`
//!   buffer;
//! * requantization in the `sts_rq` fx form: the step's scalar
//!   `Requant { m0, shift, rounding: HalfAwayFromZero, saturate_lo, saturate_hi }` is
//!   replicated across every output channel (the emitted ROM is per-channel),
//!   intersected with the step's activation clamp, applied as
//!   `((|acc| * m0) + 2^(shift-1)) >> shift` with the accumulator sign
//!   restored and the result clamped to `[lo, hi]`;
//! * no bias term.
//!
//! # Explicitly unsupported (rejected with [`RtlError`])
//!
//! Every other operator (depthwise, 2D convolution with any kernel other than
//! 1x1, add, standalone requant/activation, nearest upsample, unsigned output
//! quantization), i4/u8/i32/f32 *activation* element types, asymmetric
//! quantization, `m0` outside the 17-bit field range `[0, 2^17)`, rounding other than
//! half-away-from-zero,
//! missing requantization, bias terms, tap parallelism, shared-tile or hybrid
//! execution modes, `Bram`/`Ddr`/`LineBuffer` activation buffers, weights that
//! are not ROM-resident, and plans with zero or more than one schedule step.
//! Each rejection names the step, the offending construct, and what the
//! backend does support.
//!
//! # Interface contract (v1)
//!
//! ```text
//! clk, rst                    synchronous, active-high reset
//! in_valid / in_ready         accepts K beats per reduction window
//! in_data  [SIMD*8-1:0]       signed int8 lanes, lane 0 in the low byte
//! out_valid / out_ready       one beat per output fold, NF beats per window
//! out_data [NOC*8-1:0]        signed int8 requantized lanes, lane 0 low byte
//! ```
//!
//! Scheduled cost of one window is `K + NF*(K + 2)` clock cycles: `K` load
//! cycles, then per output fold `K` MAC cycles plus two pipeline cycles. The
//! value is recorded in the bundle metadata as an *estimate* of this module,
//! separate from the plan's `estimated_cycles`.
//!
//! # Relationship to `sts_lib.v`
//!
//! `nngen/examples/st_stream/rtl/sts_lib.v` is the reference streaming library
//! for the same numerical contract (`sts_rq` requantizes, `sts_mvau` is the
//! folding MAC array, `sts_rom` initializes weight ROMs, `sts_fifo` matches
//! output rates). The generated datapath evaluates exactly the `sts_rq` fx
//! formula, with `PRE = 0` and an absolute shift where `sts_rq` carries a
//! relative shift (`SHMIN`, `SHB`) and the same `SIMD`/`PE` folding as
//! `sts_mvau`, so substituting the library modules is a translation of the
//! shift encoding plus a port match rather than a numerics change.
//!
//! The library is *not* copied into generated output: the backend emits a
//! self-contained module and records the substitution candidates in the bundle
//! metadata (`reuse_candidates`).

use std::error::Error;
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::layout::{
    ActivationKind, ElemType as LayoutElemType, ExecutionMode, ExecutionPlan, MemoryPlacement,
    OpKind, Rounding, TensorRef,
};

pub mod full;
pub mod mapped;
pub mod reference;

pub use mapped::{MappedSummary, lower_mapped};

/// Format tag stamped into the serialized RTL bundle metadata.
pub const RTL_BUNDLE_FORMAT: &str = "tessera.rtl_bundle";
/// Schema version of the RTL bundle metadata document.
pub const RTL_BUNDLE_SCHEMA_VERSION: u32 = 1;
/// Stable identifier of the kernel family implemented by this backend.
pub const RTL_KERNEL_ID: &str = "tessera.pw_dot.i8i8.v1";
/// Width of the emitted per-channel multiplier field.
///
/// The planner clips `Requant::m0` to `[0, 2^17 - 1]`, and that is exactly the
/// range the generated ROM can represent. The kernel does not require the
/// `sts_rq` normalization `2^16 <= M0 < 2^17`: the product field is wide enough
/// for `|acc| * M0` at either end of the range, so the arithmetic stays exact.
pub const MULTIPLIER_BITS: u32 = 17;
const MULTIPLIER_MAX_EXCL: u32 = 1 << MULTIPLIER_BITS;
/// Magnitude field of the accumulator (`ACCW + 1`, as in `sts_rq`).
const ACC_MAG_BITS: u32 = 33;
/// Width of the emitted signed accumulator.
const ACC_WIDTH: u32 = 32;
/// Upper bound on the generated weight ROM depth, to keep emission sane.
const MAX_WEIGHT_ROM_ENTRIES: u64 = 1 << 20;
/// Upper bound on the generated activation window buffer depth.
const MAX_WINDOW_BUFFER_ENTRIES: u64 = 1 << 12;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the RTL backend. Every variant is user-actionable and
/// names the step or the offending construct.
#[derive(Debug)]
pub enum RtlError {
    /// The plan asked for an operator this backend cannot implement.
    UnsupportedOperator {
        layer: String,
        op: PlannedOp,
        supported: &'static str,
    },
    /// Element type outside the supported subset.
    UnsupportedElementType {
        layer: String,
        tensor: String,
        element_type: ElementType,
    },
    /// Requantization or activation contract this backend cannot reproduce.
    UnsupportedQuantization { layer: String, detail: String },
    /// Placement (memory / engine realization) this backend cannot emit.
    UnsupportedPlacement {
        layer: String,
        what: &'static str,
        placement: PlacementHint,
    },
    /// Weights are not ROM-resident in the plan.
    WeightsNotResident {
        layer: String,
        tensor: String,
        detail: String,
    },
    /// A step references a buffer that the plan does not define.
    UnknownBuffer {
        layer: String,
        what: &'static str,
        name: String,
    },
    /// Plan shape is outside the narrow v1 envelope (step count, tiling).
    UnsupportedLayout { detail: String },
    /// A tensor/attribute length does not match the declared tile geometry.
    ShapeMismatch {
        layer: String,
        what: String,
        expected: u64,
        actual: u64,
    },
    /// A tile parameter is out of range or inconsistent.
    InvalidKernelParam {
        layer: String,
        field: &'static str,
        detail: String,
    },
    /// The declared accumulator width cannot hold the worst-case dot product.
    AccumulatorOverflow {
        layer: String,
        max_abs: u64,
        acc_width: u32,
    },
    /// The plan did not carry the provenance required by VERSIONING.md.
    MissingProvenance { field: &'static str },
    /// Metadata could not be serialized.
    Serialize {
        context: &'static str,
        source: serde_json::Error,
    },
    /// Filesystem failure while materializing the bundle.
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl fmt::Display for RtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RtlError::UnsupportedOperator {
                layer,
                op,
                supported,
            } => write!(
                f,
                "rtl: step {layer}: operator {op} is not implemented by {RTL_KERNEL_ID} \
                 (supported: {supported})"
            ),
            RtlError::UnsupportedElementType {
                layer,
                tensor,
                element_type,
            } => write!(
                f,
                "rtl: step {layer}: tensor {tensor} has element type {element_type}, which \
                 {RTL_KERNEL_ID} cannot consume (supported: i8, and i4 packed as i4-x2 weights)"
            ),
            RtlError::UnsupportedQuantization { layer, detail } => {
                write!(f, "rtl: step {layer}: unsupported requantization: {detail}")
            }
            RtlError::UnsupportedPlacement {
                layer,
                what,
                placement,
            } => write!(
                f,
                "rtl: step {layer}: {what} placement {placement} is not emitted by \
                 {RTL_KERNEL_ID} (supported: weights in ROM, activations streamed via \
                 register/FIFO buffers)"
            ),
            RtlError::WeightsNotResident {
                layer,
                tensor,
                detail,
            } => write!(
                f,
                "rtl: step {layer}: weights tensor {tensor} is not ROM-resident: {detail}"
            ),
            RtlError::UnknownBuffer { layer, what, name } => write!(
                f,
                "rtl: step {layer}: {what} buffer `{name}` is not defined in the plan's buffer \
                 table"
            ),
            RtlError::UnsupportedLayout { detail } => {
                write!(f, "rtl: unsupported layout: {detail}")
            }
            RtlError::ShapeMismatch {
                layer,
                what,
                expected,
                actual,
            } => write!(
                f,
                "rtl: step {layer}: {what} length mismatch: expected {expected}, got {actual}"
            ),
            RtlError::InvalidKernelParam {
                layer,
                field,
                detail,
            } => {
                write!(f, "rtl: step {layer}: invalid {field}: {detail}")
            }
            RtlError::AccumulatorOverflow {
                layer,
                max_abs,
                acc_width,
            } => write!(
                f,
                "rtl: step {layer}: accumulator width {acc_width} cannot hold the worst-case \
                 |acc| {max_abs}; the planner must widen the fold or split the reduction"
            ),
            RtlError::MissingProvenance { field } => write!(
                f,
                "rtl: plan provenance is missing required field `{field}`; the backend refuses to \
                 emit unattributable RTL"
            ),
            RtlError::Serialize { context, source } => write!(f, "rtl: {context}: {source}"),
            RtlError::Io { context, source } => write!(f, "rtl: {context}: {source}"),
        }
    }
}

impl Error for RtlError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RtlError::Io { source, .. } => Some(source),
            RtlError::Serialize { source, .. } => Some(source),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Planner-facing plan description
// ---------------------------------------------------------------------------

/// Operator kind as planned by the hardware planner.
///
/// Only [`PlannedOp::PointwiseDot`] is executable in v1; the remaining
/// variants exist so the backend can *name* what it is rejecting instead of
/// silently dropping it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlannedOp {
    PointwiseDot,
    Depthwise,
    Conv2d,
    Requant,
    Activation,
    Add,
    UpsampleNearest,
    QuantizeOutput,
}

impl fmt::Display for PlannedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PlannedOp::PointwiseDot => "pointwise_dot",
            PlannedOp::Depthwise => "depthwise",
            PlannedOp::Conv2d => "conv2d",
            PlannedOp::Requant => "requant",
            PlannedOp::Activation => "activation",
            PlannedOp::Add => "add",
            PlannedOp::UpsampleNearest => "upsample_nearest",
            PlannedOp::QuantizeOutput => "quantize_output",
        };
        f.write_str(s)
    }
}

/// Element type of a stream or tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    I8,
    /// Signed 9-bit value (an unclamped residual sum of two i8 codes).
    I9,
    /// Signed 16-bit transport container.
    I16,
    I4,
    U8,
    I32,
    F32,
}

impl From<LayoutElemType> for ElementType {
    fn from(value: LayoutElemType) -> Self {
        match value {
            LayoutElemType::I8 => ElementType::I8,
            LayoutElemType::I9 => ElementType::I9,
            LayoutElemType::I16 => ElementType::I16,
            LayoutElemType::I4 => ElementType::I4,
            LayoutElemType::U8 => ElementType::U8,
            LayoutElemType::I32 => ElementType::I32,
            LayoutElemType::F32 => ElementType::F32,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ElementType::I8 => "i8",
            ElementType::I9 => "i9",
            ElementType::I16 => "i16",
            ElementType::I4 => "i4",
            ElementType::U8 => "u8",
            ElementType::I32 => "i32",
            ElementType::F32 => "f32",
        };
        f.write_str(s)
    }
}

/// Where the planner decided to keep a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementHint {
    /// Weights resident in a read-only ROM inside the generated module.
    WeightRom,
    /// Activations streamed through the engine (register or FIFO transport).
    StreamedActivation,
    /// Registers.
    Register,
    /// Handshake FIFO / SRL storage.
    Fifo,
    Bram,
    Ddr,
    LineBuffer,
}

impl From<MemoryPlacement> for PlacementHint {
    fn from(value: MemoryPlacement) -> Self {
        match value {
            MemoryPlacement::Register => PlacementHint::Register,
            MemoryPlacement::Fifo => PlacementHint::Fifo,
            MemoryPlacement::Bram => PlacementHint::Bram,
            MemoryPlacement::Ddr => PlacementHint::Ddr,
            MemoryPlacement::LineBuffer => PlacementHint::LineBuffer,
        }
    }
}

impl fmt::Display for PlacementHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PlacementHint::WeightRom => "weight_rom",
            PlacementHint::StreamedActivation => "streamed_activation",
            PlacementHint::Register => "register",
            PlacementHint::Fifo => "fifo",
            PlacementHint::Bram => "bram",
            PlacementHint::Ddr => "ddr",
            PlacementHint::LineBuffer => "line_buffer",
        };
        f.write_str(s)
    }
}

/// Rounding mode of the requantization step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundingMode {
    /// `(mag*M0 + 2^(sh-1)) >> sh` — the `sts_rq` fx form.
    RoundHalfUpMagnitude,
    Truncate,
    RoundHalfEven,
}

/// Tile geometry of a planned kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileGeometry {
    /// Activation lanes per input beat (`in_channels / k`).
    pub simd: u32,
    /// Output channels per output beat (`out_channels / fold`).
    pub noc: u32,
    /// Reduction beats per window (`in_channels / simd`).
    pub k: u32,
    /// Output beats per window (`out_channels / noc`).
    pub fold: u32,
}

impl TileGeometry {
    /// Input channels covered by one reduction window.
    pub fn in_channels(&self) -> u64 {
        u64::from(self.simd) * u64::from(self.k)
    }

    /// Output channels covered by one window.
    pub fn out_channels(&self) -> u64 {
        u64::from(self.noc) * u64::from(self.fold)
    }
}

/// Plan-level context of a kernel, recorded verbatim in bundle metadata and in
/// the generated file headers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanContext {
    pub plan_format: String,
    pub plan_schema_version: u32,
    pub execution_mode: String,
    pub granularity: String,
    pub engine: String,
    pub step_index: u32,
    pub step_count: u32,
    pub estimated_cycles: u64,
    pub spatial_in: [u32; 2],
    pub spatial_out: [u32; 2],
}

impl PlanContext {
    /// Context for a plan source that is not a planner artifact (tests and
    /// direct backend use). It states that plainly instead of inventing plan
    /// fields.
    pub fn direct(layer: &str) -> Self {
        PlanContext {
            plan_format: "tessera.direct_kernel".to_string(),
            plan_schema_version: RTL_BUNDLE_SCHEMA_VERSION,
            execution_mode: "streaming".to_string(),
            granularity: "window".to_string(),
            engine: format!("direct:{layer}"),
            step_index: 0,
            step_count: 1,
            estimated_cycles: 0,
            spatial_in: [0, 0],
            spatial_out: [0, 0],
        }
    }
}

/// Quantization contract of a planned kernel. Values are semantic data and are
/// never re-derived from lowered arithmetic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantPlan {
    /// Activation zero-point (must be 0 in v1).
    pub activation_zero_point: i32,
    /// Weight zero-point (must be 0 in v1).
    pub weight_zero_point: i32,
    pub rounding: RoundingMode,
    /// Per-output-channel requantization, `fold * noc` entries in
    /// `(fold, noc)` order.
    pub channels: Vec<ChannelRequant>,
    /// Declared maximum relative error of the fixed-point requantization
    /// approximation, in ppm, when the producer artifact states one. This is a
    /// plan estimate, never a measurement.
    #[serde(default)]
    pub multiplier_error_ppm: Option<u32>,
}

/// One output channel's requantization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRequant {
    /// Normalized multiplier in `[2^16, 2^17)`.
    pub multiplier: u32,
    /// Right shift applied after the multiply.
    pub shift: u8,
    /// Saturation low bound (inclusive).
    pub lo: i16,
    /// Saturation high bound (inclusive).
    pub hi: i16,
}

/// Signed int8 weight tile in `(fold, k, noc, simd)` order, flattened.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensor {
    pub data: Vec<i8>,
}

/// Everything the RTL backend needs about one planned kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelPlanView {
    pub layer: String,
    pub op: PlannedOp,
    pub activation_element: ElementType,
    pub weight_element: ElementType,
    pub output_element: ElementType,
    pub tile: TileGeometry,
    pub quant: QuantPlan,
    pub weights: WeightTensor,
    /// Where the planner kept the weights / activations.
    pub weight_placement: PlacementHint,
    pub activation_placement: PlacementHint,
    pub context: PlanContext,
}

/// A planner artifact that the RTL backend can consume.
///
/// Implemented for [`crate::layout::ExecutionPlan`] through
/// [`ExecutionPlanSource`]; the backend never inspects planner internals and
/// never guesses a layout.
pub trait RtlPlanSource {
    /// Format tag of the planner artifact (must be non-empty).
    fn artifact_format(&self) -> &str;
    /// Schema version of the planner artifact.
    fn schema_version(&self) -> u32;
    /// Provenance required by VERSIONING.md §3.
    fn provenance(&self) -> Provenance;
    /// Planned kernels, in deterministic plan order.
    fn kernels(&self) -> Vec<KernelPlanView>;
}

/// Supplies the stored bytes of a weight tensor referenced by the plan.
///
/// The planner records weight *references* only, so contents come from the
/// graph/manifest layer. Bytes must be in `(n, c, h, w)` row-major order for
/// the tensor's declared shape, using the element type's storage encoding
/// (`i4-x2`: two signed nibbles per byte, low nibble first).
pub trait WeightResolver {
    fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError>;
}

/// Unpacks the manifest `i4-x2` encoding: two signed 4-bit codes per byte, low
/// nibble first, each sign-extended to 8 bits.
pub fn unpack_i4_x2(
    layer: &str,
    tensor: &str,
    packed: &[u8],
    elements: u64,
) -> Result<Vec<i8>, RtlError> {
    let needed = elements.div_ceil(2);
    if packed.len() as u64 != needed {
        return Err(RtlError::ShapeMismatch {
            layer: layer.to_string(),
            what: format!("i4-x2 packed bytes of {tensor}"),
            expected: needed,
            actual: packed.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(elements as usize);
    for byte in packed {
        for nibble in [byte & 0x0f, byte >> 4] {
            if out.len() as u64 == elements {
                break;
            }
            // Exact sign extension of a 4-bit code (nibble 8..15 is -8..-1).
            out.push((nibble ^ 8) as i8 - 8);
        }
    }
    Ok(out)
}

/// Reorders a pointwise weight matrix in `(out_channel, in_channel)` order into
/// the kernel ROM order `(fold, k, noc, simd)`.
pub fn reorder_pointwise_weights(
    layer: &str,
    weights: &[i8],
    tile: TileGeometry,
) -> Result<Vec<i8>, RtlError> {
    let simd = tile.simd as usize;
    let noc = tile.noc as usize;
    let fold = tile.fold as usize;
    let k = tile.k as usize;
    let in_channels = k * simd;
    let out_channels = fold * noc;
    if weights.len() != out_channels * in_channels {
        return Err(RtlError::ShapeMismatch {
            layer: layer.to_string(),
            what: "weights (out_channels*in_channels)".to_string(),
            expected: (out_channels * in_channels) as u64,
            actual: weights.len() as u64,
        });
    }
    let mut reordered = vec![0i8; weights.len()];
    for f in 0..fold {
        for kk in 0..k {
            for n in 0..noc {
                for l in 0..simd {
                    let in_index = kk * simd + l;
                    let out_channel = f * noc + n;
                    let out_index = ((f * k + kk) * noc + n) * simd + l;
                    reordered[out_index] = weights[out_channel * in_channels + in_index];
                }
            }
        }
    }
    Ok(reordered)
}

/// Replicates one scalar requantization record across every output channel.
pub fn uniform_requant(count: u32, channel: ChannelRequant) -> Vec<ChannelRequant> {
    vec![channel; count as usize]
}

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// Artifact provenance (VERSIONING.md §3). Mirrored into every generated file
/// and into the bundle metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub compiler_version: String,
    pub git_revision: String,
    pub dirty: bool,
    pub graph_schema_version: u32,
    pub plan_schema_version: u32,
    pub target: String,
    pub target_revision: String,
    pub input_manifest_sha256: String,
    /// Digest of the weight bytes actually emitted.
    ///
    /// The planner artifact carries no weight hash, so the backend records the
    /// digest of the bytes it embedded and names the algorithm it used
    /// (`fnv1a64:<hex>`); a `sha256:` digest is passed through unchanged when a
    /// producer supplies one.
    pub weights_sha256: String,
    pub planner_config_sha256: String,
}

impl Provenance {
    /// Rejects provenance that cannot attribute the generated RTL.
    pub fn validate(&self) -> Result<(), RtlError> {
        let fields: [(&'static str, &str); 8] = [
            ("compiler_version", &self.compiler_version),
            ("git_revision", &self.git_revision),
            ("target", &self.target),
            ("target_revision", &self.target_revision),
            ("input_manifest_sha256", &self.input_manifest_sha256),
            ("weights_sha256", &self.weights_sha256),
            ("planner_config_sha256", &self.planner_config_sha256),
            ("plan_schema_version", ""),
        ];
        for (name, value) in fields {
            if name != "plan_schema_version" && value.trim().is_empty() {
                return Err(RtlError::MissingProvenance { field: name });
            }
        }
        if self.plan_schema_version == 0 {
            return Err(RtlError::MissingProvenance {
                field: "plan_schema_version",
            });
        }
        Ok(())
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            git_revision: "unknown".to_string(),
            dirty: false,
            graph_schema_version: 0,
            plan_schema_version: 1,
            target: "unspecified".to_string(),
            target_revision: "unspecified".to_string(),
            input_manifest_sha256: "unspecified".to_string(),
            weights_sha256: "unspecified".to_string(),
            planner_config_sha256: "unspecified".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Validated kernel
// ---------------------------------------------------------------------------

/// A kernel that passed every legality check the backend can perform before
/// emission. Constructed only through [`RtlKernel::analyze`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlKernel {
    pub layer: String,
    pub module_name: String,
    pub tile: TileGeometry,
    /// Signed int8 weights in `(fold, k, noc, simd)` order.
    pub weights: Vec<i8>,
    /// `fold * noc` requantization records in `(fold, noc)` order.
    pub requant: Vec<ChannelRequant>,
    pub acc_width: u32,
    pub out_width: u32,
    pub weight_width: u32,
    pub input_width: u32,
    pub context: PlanContext,
    /// Declared requantization approximation error (plan estimate, in ppm).
    pub multiplier_error_ppm: Option<u32>,
}

impl RtlKernel {
    /// Validates a planner kernel view and returns the emittable kernel.
    pub fn analyze(view: &KernelPlanView) -> Result<RtlKernel, RtlError> {
        let layer = view.layer.clone();
        if layer.trim().is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: "kernel has an empty step name; the RTL backend needs a stable identity"
                    .to_string(),
            });
        }
        if view.op != PlannedOp::PointwiseDot {
            return Err(RtlError::UnsupportedOperator {
                layer,
                op: view.op,
                supported: "pointwise (1x1) dot-product engine `pointwise_dot`",
            });
        }
        for ty in [
            view.activation_element,
            view.weight_element,
            view.output_element,
        ] {
            if ty != ElementType::I8 {
                return Err(RtlError::UnsupportedElementType {
                    layer,
                    tensor: "kernel stream".to_string(),
                    element_type: ty,
                });
            }
        }
        if !matches!(view.weight_placement, PlacementHint::WeightRom) {
            return Err(RtlError::UnsupportedPlacement {
                layer,
                what: "weight",
                placement: view.weight_placement,
            });
        }
        if view.activation_placement != PlacementHint::StreamedActivation {
            return Err(RtlError::UnsupportedPlacement {
                layer,
                what: "activation",
                placement: view.activation_placement,
            });
        }

        let TileGeometry { simd, noc, k, fold } = view.tile;
        for (field, value) in [("simd", simd), ("noc", noc), ("k", k), ("fold", fold)] {
            if value == 0 {
                return Err(RtlError::InvalidKernelParam {
                    layer,
                    field,
                    detail: "must be at least 1".to_string(),
                });
            }
        }
        // The generated datapath uses a fixed signed accumulator; prove the
        // worst-case dot product fits before emitting it. Checked before the
        // buffer bounds so an overflowing fold is reported as such.
        let max_abs: u64 = u64::from(k) * u64::from(simd) * 128 * 128;
        if max_abs >= 1u64 << (ACC_WIDTH - 1) {
            return Err(RtlError::AccumulatorOverflow {
                layer,
                max_abs,
                acc_width: ACC_WIDTH,
            });
        }
        let window_entries = u64::from(k) * u64::from(simd);
        if window_entries > MAX_WINDOW_BUFFER_ENTRIES {
            return Err(RtlError::InvalidKernelParam {
                layer,
                field: "k * simd",
                detail: format!(
                    "activation window buffer of {window_entries} entries exceeds the emitted \
                     limit of {MAX_WINDOW_BUFFER_ENTRIES}; reduce the fold or the SIMD width"
                ),
            });
        }
        let weight_entries = u64::from(fold) * window_entries * u64::from(noc);
        if weight_entries > MAX_WEIGHT_ROM_ENTRIES {
            return Err(RtlError::InvalidKernelParam {
                layer,
                field: "fold * k * noc * simd",
                detail: format!(
                    "weight ROM of {weight_entries} entries exceeds the emitted limit of \
                     {MAX_WEIGHT_ROM_ENTRIES}; the planner must tile the layer"
                ),
            });
        }

        // Quantization contract.
        if view.quant.activation_zero_point != 0 || view.quant.weight_zero_point != 0 {
            return Err(RtlError::UnsupportedQuantization {
                layer,
                detail: format!(
                    "asymmetric zero-points (activation {}, weight {}) are not implemented; \
                     the v1 kernel is symmetric-only",
                    view.quant.activation_zero_point, view.quant.weight_zero_point
                ),
            });
        }
        if view.quant.rounding != RoundingMode::RoundHalfUpMagnitude {
            return Err(RtlError::UnsupportedQuantization {
                layer,
                detail: format!(
                    "rounding mode {:?} is not reproducible by the generated datapath; \
                     supported: RoundHalfUpMagnitude (sts_rq fx semantics)",
                    view.quant.rounding
                ),
            });
        }
        let channels_expected = u64::from(noc) * u64::from(fold);
        if view.quant.channels.len() as u64 != channels_expected {
            return Err(RtlError::ShapeMismatch {
                layer,
                what: "per-channel requantization (fold*noc)".to_string(),
                expected: channels_expected,
                actual: view.quant.channels.len() as u64,
            });
        }
        for (index, channel) in view.quant.channels.iter().enumerate() {
            if channel.multiplier >= MULTIPLIER_MAX_EXCL {
                return Err(RtlError::UnsupportedQuantization {
                    layer,
                    detail: format!(
                        "channel {index}: multiplier {} does not fit the {MULTIPLIER_BITS}-bit \
                         multiplier ROM (representable range [0, 2^17))",
                        channel.multiplier
                    ),
                });
            }
            if channel.shift > 31 {
                return Err(RtlError::UnsupportedQuantization {
                    layer,
                    detail: format!("channel {index}: shift {} exceeds 31", channel.shift),
                });
            }
            if channel.lo > channel.hi {
                return Err(RtlError::InvalidKernelParam {
                    layer,
                    field: "requant clamp",
                    detail: format!(
                        "channel {index}: saturation range is inverted (lo {} > hi {})",
                        channel.lo, channel.hi
                    ),
                });
            }
            for (name, value) in [("lo", channel.lo), ("hi", channel.hi)] {
                if !(-512..=511).contains(&value) {
                    return Err(RtlError::InvalidKernelParam {
                        layer,
                        field: "requant clamp",
                        detail: format!(
                            "channel {index}: {name} {value} does not fit 10 signed bits"
                        ),
                    });
                }
            }
        }

        if view.weights.data.len() as u64 != weight_entries {
            return Err(RtlError::ShapeMismatch {
                layer,
                what: "weight tile (fold*k*noc*simd)".to_string(),
                expected: weight_entries,
                actual: view.weights.data.len() as u64,
            });
        }

        Ok(RtlKernel {
            module_name: format!("{}_pw_dot_i8i8_v1", slug(&layer)),
            layer,
            tile: view.tile,
            weights: view.weights.data.clone(),
            requant: view.quant.channels.clone(),
            acc_width: ACC_WIDTH,
            out_width: 8,
            weight_width: 8,
            input_width: 8,
            context: view.context.clone(),
            multiplier_error_ppm: view.quant.multiplier_error_ppm,
        })
    }

    fn k_bits(&self) -> u32 {
        bits_for(self.tile.k)
    }

    fn fold_bits(&self) -> u32 {
        bits_for(self.tile.fold)
    }

    fn shift_bits(&self) -> u32 {
        let max = self.requant.iter().map(|c| c.shift).max().unwrap_or(0);
        bits_for(u32::from(max) + 1)
    }

    /// Worst-case accumulation magnitude covered by the emitted accumulator.
    fn max_abs_acc(&self) -> u64 {
        u64::from(self.tile.k) * u64::from(self.tile.simd) * 128 * 128
    }

    /// Estimated clock cycles of one reduction window: `k` load beats, then
    /// `fold * (k + 2)` for the MAC sweep and the requant/output pipeline.
    pub fn cycles_per_window(&self) -> u64 {
        let k = u64::from(self.tile.k);
        k + u64::from(self.tile.fold) * (k + 2)
    }
}

/// `ceil(log2(value))`, clamped to at least 1 so SystemVerilog never sees a
/// zero-width vector index.
pub fn bits_for(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    u32::BITS - (value - 1).leading_zeros()
}

/// Deterministic identifier fragment: lowercase, `[a-z0-9_]`, leading letter.
pub fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("unnamed");
    }
    if !out.starts_with(|c: char| c.is_ascii_alphabetic()) {
        out.insert(0, 'k');
    }
    out
}

// ---------------------------------------------------------------------------
// Software reference model
// ---------------------------------------------------------------------------

/// Reference requantization, bit-identical to `sts_rq`'s fx semantics:
/// `k = (|acc|*M0 + 2^(sh-1)) >> sh`, sign restored, then clamped.
pub fn reference_requant(acc: i64, channel: &ChannelRequant) -> i64 {
    let multiplier = u128::from(channel.multiplier);
    let magnitude = acc.unsigned_abs() as u128;
    let product = magnitude * multiplier;
    let rounded = if channel.shift == 0 {
        product
    } else {
        product + (1u128 << (channel.shift - 1))
    };
    let shifted = rounded >> channel.shift;
    let shifted = shifted.min(i64::MAX as u128) as i64;
    let signed = if acc < 0 { -shifted } else { shifted };
    signed.clamp(i64::from(channel.lo), i64::from(channel.hi))
}

/// Reference dot product for one reduction window: `fold * noc` accumulators
/// over `k` beats of `simd` int8 lanes, weights in `(fold, k, noc, simd)` order
/// and `activation[beat]` holding `simd` lanes.
pub fn reference_dot(tile: TileGeometry, weights: &[i8], activation: &[Vec<i8>]) -> Vec<i64> {
    let simd = tile.simd as usize;
    let noc = tile.noc as usize;
    let k = tile.k as usize;
    let fold = tile.fold as usize;
    let mut acc = vec![0i64; fold * noc];
    for f in 0..fold {
        for beat in 0..k {
            for n in 0..noc {
                for lane in 0..simd {
                    let w = weights[((f * k + beat) * noc + n) * simd + lane] as i64;
                    let x = activation[beat][lane] as i64;
                    acc[f * noc + n] += w * x;
                }
            }
        }
    }
    acc
}

/// Full reference evaluation of one window, in output-beat order
/// (`fold` beats of `noc` requantized values).
pub fn reference_window_output(kernel: &RtlKernel, activation: &[Vec<i8>]) -> Vec<i64> {
    let acc = reference_dot(kernel.tile, &kernel.weights, activation);
    acc.iter()
        .zip(kernel.requant.iter())
        .map(|(value, channel)| reference_requant(*value, channel))
        .collect()
}

// ---------------------------------------------------------------------------
// Fixture generation
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* stream, so stimulus is reproducible from a seed
/// (VERSIONING.md §8: randomness must be explicitly seeded).
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Xorshift64 {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform value in `[-128, 127]`.
    pub fn next_i8(&mut self) -> i8 {
        self.next_u64() as u8 as i8
    }
}

/// Stimulus plus expected outputs for the generated testbench.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fixture {
    /// `windows * k` input beats, each with `simd` lanes.
    pub stimulus: Vec<Vec<i8>>,
    /// `windows * fold` output beats, each with `noc` values.
    pub expected: Vec<Vec<i64>>,
    pub window_count: usize,
}

/// Builds a deterministic fixture for a kernel.
pub fn build_fixture(kernel: &RtlKernel, windows: usize, seed: u64) -> Fixture {
    let mut rng = Xorshift64::new(seed);
    let k = kernel.tile.k as usize;
    let fold = kernel.tile.fold as usize;
    let mut stimulus = Vec::with_capacity(windows * k);
    for _ in 0..windows * k {
        let mut beat = Vec::with_capacity(kernel.tile.simd as usize);
        for _ in 0..kernel.tile.simd {
            beat.push(rng.next_i8());
        }
        stimulus.push(beat);
    }
    let mut expected = Vec::with_capacity(windows * fold);
    for window in 0..windows {
        let start = window * k;
        let slice = stimulus[start..start + k].to_vec();
        let values = reference_window_output(kernel, &slice);
        for beat in 0..fold {
            let base = beat * kernel.tile.noc as usize;
            expected.push(values[base..base + kernel.tile.noc as usize].to_vec());
        }
    }
    Fixture {
        stimulus,
        expected,
        window_count: windows,
    }
}

// ---------------------------------------------------------------------------
// Configuration and bundle
// ---------------------------------------------------------------------------

/// Backend configuration. All fields affect generated text, so the bundle
/// metadata records the resolved values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlConfig {
    /// Prefix of the generated module names.
    pub module_prefix: String,
    /// Emit the self-checking testbench.
    pub emit_testbench: bool,
    /// Emit `$readmemh`-loadable images next to the inline ROM contents.
    pub emit_memory_images: bool,
    /// Number of stimulus windows in the generated testbench.
    pub fixture_windows: usize,
    /// Seed of the fixture stimulus stream.
    pub fixture_seed: u64,
}

impl Default for RtlConfig {
    fn default() -> Self {
        RtlConfig {
            module_prefix: "tessera".to_string(),
            emit_testbench: true,
            emit_memory_images: true,
            fixture_windows: 16,
            fixture_seed: 0x5eed_1234_abcd_0001,
        }
    }
}

impl RtlConfig {
    /// Validates backend options before any file is produced.
    pub fn validate(&self) -> Result<(), RtlError> {
        if self.module_prefix.trim().is_empty() {
            return Err(RtlError::InvalidKernelParam {
                layer: "<config>".to_string(),
                field: "module_prefix",
                detail: "must be non-empty".to_string(),
            });
        }
        if self.emit_testbench && self.fixture_windows == 0 {
            return Err(RtlError::InvalidKernelParam {
                layer: "<config>".to_string(),
                field: "fixture_windows",
                detail: "a testbench needs at least one stimulus window".to_string(),
            });
        }
        Ok(())
    }

    fn note(&self) -> String {
        format!(
            "module_prefix={} emit_testbench={} emit_memory_images={} fixture_windows={} fixture_seed={:#018x}",
            self.module_prefix,
            self.emit_testbench,
            self.emit_memory_images,
            self.fixture_windows,
            self.fixture_seed
        )
    }
}

/// Class of a generated file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RtlFileKind {
    SystemVerilog,
    Testbench,
    MemoryImage,
    Script,
    Metadata,
}

/// One generated file. Deterministic content, deterministic name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlFile {
    pub name: String,
    pub kind: RtlFileKind,
    pub contents: String,
}

/// Verified digest entry for one generated file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub name: String,
    pub kind: RtlFileKind,
    pub bytes: usize,
    /// FNV-1a-64 of the file contents (dependency-free, stable across builds).
    pub fnv1a64: String,
}

/// Metadata document emitted alongside the RTL sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlMetadata {
    pub format: String,
    pub schema_version: u32,
    pub kernel_id: String,
    pub provenance: Provenance,
    pub plan: PlanContext,
    pub top_module: String,
    pub kernel_module: String,
    pub layer: String,
    pub tile: TileGeometry,
    pub in_channels: u64,
    pub out_channels: u64,
    pub accumulator_width: u32,
    pub max_abs_accumulator: u64,
    pub kernel_cycles_per_window: u64,
    pub requantization: Vec<ChannelRequant>,
    /// Plan-declared requantization approximation error, in ppm.
    pub requant_multiplier_error_ppm: Option<u32>,
    pub config: RtlConfig,
    pub files: Vec<FileDigest>,
    pub supported_operators: Vec<String>,
    pub supported_interface: Vec<String>,
    pub unsupported: Vec<String>,
    /// Reference library modules that can later replace generated logic without
    /// changing numerics.
    pub reuse_candidates: Vec<String>,
    /// Exact commands that reproduce and verify this bundle.
    pub reproduce: Vec<String>,
    pub bundle_digest: String,
    /// Present only for the full-graph backend: the generated structure, its
    /// exact activation arena, MAC lane count and cycle accounting, next to the
    /// plan's estimates. `None` for the single-kernel pointwise envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_graph: Option<full::FullGraphSummary>,
    /// Present only for the mapped multi-engine backend: the generated engine
    /// instances, multiplier lanes, pipeline links and shared arena.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapped: Option<mapped::MappedSummary>,
}

/// The emitted artifact bundle: sources plus metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlBundle {
    pub metadata: RtlMetadata,
    files: Vec<RtlFile>,
}

impl RtlBundle {
    /// Generated files, sorted by name.
    pub fn files(&self) -> &[RtlFile] {
        &self.files
    }

    /// Looks up one generated file by name.
    pub fn file(&self, name: &str) -> Option<&RtlFile> {
        self.files.iter().find(|f| f.name == name)
    }

    /// The SystemVerilog sources in deterministic compile order (DUT first).
    pub fn verilog_sources(&self) -> Vec<&str> {
        let mut sources: Vec<&RtlFile> = self
            .files
            .iter()
            .filter(|f| matches!(f.kind, RtlFileKind::SystemVerilog | RtlFileKind::Testbench))
            .collect();
        let rank = |f: &RtlFile| match f.kind {
            RtlFileKind::SystemVerilog => 0u8,
            RtlFileKind::Testbench => 1,
            _ => 2,
        };
        sources.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.name.cmp(&b.name)));
        sources.into_iter().map(|f| f.name.as_str()).collect()
    }

    /// Writes every file into `dir`, creating it if needed.
    pub fn write_to(&self, dir: &Path) -> Result<Vec<PathBuf>, RtlError> {
        fs::create_dir_all(dir).map_err(|source| RtlError::Io {
            context: format!("creating {}", dir.display()),
            source,
        })?;
        let mut written = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let path = dir.join(&file.name);
            fs::write(&path, &file.contents).map_err(|source| RtlError::Io {
                context: format!("writing {}", path.display()),
                source,
            })?;
            written.push(path);
        }
        Ok(written)
    }
}

/// Stable, dependency-free content digest. Honest about the algorithm: this is
/// FNV-1a-64, used for reproducibility checks inside a bundle, not for
/// cryptographic attribution (that stays with the planner's SHA-256 hashes).
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// The SystemVerilog backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemVerilogBackend;

impl SystemVerilogBackend {
    /// Lowers a planner artifact into an RTL bundle.
    ///
    /// Fails closed: any operator, quantization form, placement, or plan shape
    /// outside the supported subset is reported as an [`RtlError`].
    pub fn lower(
        &self,
        plan: &dyn RtlPlanSource,
        config: &RtlConfig,
    ) -> Result<RtlBundle, RtlError> {
        config.validate()?;
        let provenance = plan.provenance();
        provenance.validate()?;
        let format = plan.artifact_format();
        if format.trim().is_empty() {
            return Err(RtlError::MissingProvenance {
                field: "plan artifact format",
            });
        }
        let plan_schema_version = plan.schema_version();
        if plan_schema_version == 0 {
            return Err(RtlError::MissingProvenance {
                field: "plan schema_version",
            });
        }

        let views = plan.kernels();
        if views.is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: "the plan contains no executable step; v1 emits exactly one kernel"
                    .to_string(),
            });
        }
        if views.len() > 1 {
            let names: Vec<&str> = views.iter().map(|v| v.layer.as_str()).collect();
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "v1 emits exactly one kernel, but the plan contains {} steps [{}]; select a \
                     single pointwise step or split the plan",
                    views.len(),
                    names.join(", ")
                ),
            });
        }
        let kernel = RtlKernel::analyze(&views[0])?;

        let dut_name = format!("{}_{}", slug(&config.module_prefix), kernel.module_name);
        let tb_name = format!("{dut_name}_tb");
        let top_module = if config.emit_testbench {
            tb_name.clone()
        } else {
            dut_name.clone()
        };

        let fixture = if config.emit_testbench {
            Some(build_fixture(
                &kernel,
                config.fixture_windows,
                config.fixture_seed,
            ))
        } else {
            None
        };

        let mut files: Vec<RtlFile> = Vec::new();
        files.push(RtlFile {
            name: format!("{dut_name}.sv"),
            kind: RtlFileKind::SystemVerilog,
            contents: emit_dut(&kernel, &dut_name, &provenance, &config.note()),
        });
        if config.emit_memory_images {
            files.push(RtlFile {
                name: format!("{dut_name}.weights.mem"),
                kind: RtlFileKind::MemoryImage,
                contents: emit_weight_mem(&kernel),
            });
            files.push(RtlFile {
                name: format!("{dut_name}.requant.mem"),
                kind: RtlFileKind::MemoryImage,
                contents: emit_requant_mem(&kernel),
            });
        }
        if let Some(fixture) = &fixture {
            files.push(RtlFile {
                name: format!("{tb_name}.sv"),
                kind: RtlFileKind::Testbench,
                contents: emit_testbench(&kernel, &dut_name, &tb_name, fixture, &provenance),
            });
            files.push(RtlFile {
                name: "run_verilator.sh".to_string(),
                kind: RtlFileKind::Script,
                contents: emit_run_script(&dut_name, &tb_name, &kernel),
            });
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));

        let digests: Vec<FileDigest> = files
            .iter()
            .map(|f| FileDigest {
                name: f.name.clone(),
                kind: f.kind,
                bytes: f.contents.len(),
                fnv1a64: format!("{:016x}", fnv1a64(f.contents.as_bytes())),
            })
            .collect();

        let mut digest_input = String::new();
        for digest in &digests {
            let _ = writeln!(
                digest_input,
                "{} {} {}",
                digest.name, digest.bytes, digest.fnv1a64
            );
        }
        let bundle_digest = format!("{:016x}", fnv1a64(digest_input.as_bytes()));

        let simulatable: Vec<String> = files
            .iter()
            .filter(|f| matches!(f.kind, RtlFileKind::SystemVerilog | RtlFileKind::Testbench))
            .map(|f| f.name.clone())
            .collect();

        let metadata = RtlMetadata {
            format: RTL_BUNDLE_FORMAT.to_string(),
            schema_version: RTL_BUNDLE_SCHEMA_VERSION,
            kernel_id: RTL_KERNEL_ID.to_string(),
            provenance: provenance.clone(),
            plan: kernel.context.clone(),
            top_module,
            kernel_module: dut_name.clone(),
            layer: kernel.layer.clone(),
            tile: kernel.tile,
            in_channels: kernel.tile.in_channels(),
            out_channels: kernel.tile.out_channels(),
            accumulator_width: kernel.acc_width,
            max_abs_accumulator: kernel.max_abs_acc(),
            kernel_cycles_per_window: kernel.cycles_per_window(),
            requantization: kernel.requant.clone(),
            requant_multiplier_error_ppm: kernel.multiplier_error_ppm,
            config: config.clone(),
            files: digests,
            supported_operators: vec![format!(
                "pointwise_dot: 1x1 conv, groups=1, stride=1, i8xi8 symmetric, packed i4 weights \
                 unpacked [{RTL_KERNEL_ID}]"
            )],
            supported_interface: vec![
                "clk, rst (synchronous, active-high reset)".to_string(),
                format!(
                    "in_data [{}:0] signed int8 x{} lanes, in_valid/in_ready, {} beats per window",
                    kernel.tile.simd * 8 - 1,
                    kernel.tile.simd,
                    kernel.tile.k
                ),
                format!(
                    "out_data [{}:0] signed int8 x{} lanes, out_valid/out_ready, {} beats per window",
                    kernel.tile.noc * 8 - 1,
                    kernel.tile.noc,
                    kernel.tile.fold
                ),
            ],
            unsupported: unsupported_summary(),
            reuse_candidates: vec![
                "nngen/examples/st_stream/rtl/sts_lib.v: sts_rq (bit-identical fx requantization)"
                    .to_string(),
                "nngen/examples/st_stream/rtl/sts_lib.v: sts_mvau (streaming SIMD/PE folding MAC \
                 array; replaces the generated MAC and window buffer when the plan moves from a \
                 single step to a pipelined graph)"
                    .to_string(),
                "nngen/examples/st_stream/rtl/sts_lib.v: sts_fifo (output rate matching and \
                 backpressure buffering)"
                    .to_string(),
                "nngen/examples/st_stream/rtl/sts_lib.v: sts_rom (weight ROM with $readmemh \
                 initialization, matching the emitted ROM layout)"
                    .to_string(),
            ],
            reproduce: vec![
                format!(
                    "emit: tessera RTL backend v1 (SystemVerilogBackend::lower_execution_plan) -> \
                     {dut_name}.sv + {tb_name}.sv + rtl_bundle.json"
                ),
                format!(
                    "simulate: verilator --binary --timing -Wno-fatal --top-module {tb_name} -o \
                     {tb_name}_sim {}",
                    simulatable.join(" ")
                ),
                "verify: the testbench prints TESSA_RTL_PASS with mismatches=0".to_string(),
            ],
            bundle_digest,
            full_graph: None,
            mapped: None,
        };

        let metadata_json =
            serde_json::to_string_pretty(&metadata).map_err(|source| RtlError::Serialize {
                context: "serializing bundle metadata",
                source,
            })?;
        files.push(RtlFile {
            name: "rtl_bundle.json".to_string(),
            kind: RtlFileKind::Metadata,
            contents: metadata_json,
        });

        Ok(RtlBundle { metadata, files })
    }

    /// Lowers a planner [`ExecutionPlan`], resolving weight contents through
    /// `weights`.
    pub fn lower_execution_plan(
        &self,
        plan: &ExecutionPlan,
        weights: &dyn WeightResolver,
        config: &RtlConfig,
    ) -> Result<RtlBundle, RtlError> {
        let source = ExecutionPlanSource::build(plan, weights)?;
        self.lower(&source, config)
    }
}

fn unsupported_summary() -> Vec<String> {
    vec![
        "operators: depthwise, conv2d with any kernel other than 1x1, add, standalone \
         requant/activation, nearest upsample, unsigned output quantization"
            .to_string(),
        "activation or output element types other than signed i8".to_string(),
        "asymmetric quantization (non-zero activation or weight zero-points)".to_string(),
        "requantization without a normalized M0 in [2^16, 2^17), or with rounding other than \
         half-away-from-zero"
            .to_string(),
        "bias terms".to_string(),
        "kernel tap parallelism (tap_parallel != 1)".to_string(),
        "shared-tile and hybrid execution modes".to_string(),
        "activation buffers other than register/FIFO streams (bram, ddr, line buffer)".to_string(),
        "weights that are not listed as ROM-resident in the execution plan".to_string(),
        "plans with more than one step, or with no step".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Execution plan adapter
// ---------------------------------------------------------------------------

/// Adapts a planner [`ExecutionPlan`] to the backend contract, rejecting every
/// construct outside the supported subset.
pub struct ExecutionPlanSource<'a> {
    plan: &'a ExecutionPlan,
    views: Vec<KernelPlanView>,
    weights_digest: String,
}

impl<'a> ExecutionPlanSource<'a> {
    /// Builds the backend view of `plan`, resolving weight contents through
    /// `weights`.
    pub fn build(plan: &'a ExecutionPlan, weights: &dyn WeightResolver) -> Result<Self, RtlError> {
        let mut digests: Vec<u8> = Vec::new();
        let views = plan_kernels(plan, weights, &mut digests)?;
        Ok(ExecutionPlanSource {
            plan,
            views,
            weights_digest: format!("fnv1a64:{:016x}", fnv1a64(&digests)),
        })
    }

    /// The plan this source was built from.
    pub fn plan(&self) -> &'a ExecutionPlan {
        self.plan
    }
}

impl RtlPlanSource for ExecutionPlanSource<'_> {
    fn artifact_format(&self) -> &str {
        &self.plan.format
    }

    fn schema_version(&self) -> u32 {
        self.plan.schema_version
    }

    fn provenance(&self) -> Provenance {
        let source = &self.plan.provenance;
        Provenance {
            compiler_version: source.compiler_version.clone(),
            git_revision: option(&source.options, "git_revision")
                .unwrap_or_else(|| "unknown".into()),
            dirty: option(&source.options, "dirty")
                .map(|v| v == "true")
                .unwrap_or(false),
            graph_schema_version: option(&source.options, "graph_schema_version")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            plan_schema_version: self.plan.schema_version,
            target: source.target_name.clone(),
            target_revision: source.target_revision.clone(),
            input_manifest_sha256: source.graph_sha256.clone(),
            weights_sha256: self.weights_digest.clone(),
            planner_config_sha256: source.policy_sha256.clone(),
        }
    }

    fn kernels(&self) -> Vec<KernelPlanView> {
        self.views.clone()
    }
}

fn option(options: &std::collections::BTreeMap<String, String>, key: &str) -> Option<String> {
    options.get(key).filter(|v| !v.trim().is_empty()).cloned()
}

/// Builds the backend kernel views for an execution plan.
///
/// `weight_digest_input` accumulates the resolved weight bytes so the caller can
/// record a digest of the data actually embedded in the RTL.
pub fn plan_kernels(
    plan: &ExecutionPlan,
    weights: &dyn WeightResolver,
    weight_digest_input: &mut Vec<u8>,
) -> Result<Vec<KernelPlanView>, RtlError> {
    if plan.steps.len() != 1 {
        let names: Vec<String> = plan
            .steps
            .iter()
            .map(|s| format!("{}({:?})", s.op, s.op_kind))
            .collect();
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "v1 emits exactly one kernel, but the plan has {} steps [{}]; the supported \
                 envelope is a single pointwise step",
                plan.steps.len(),
                names.join(", ")
            ),
        });
    }
    if !matches!(
        plan.execution_mode,
        ExecutionMode::Dedicated | ExecutionMode::Streaming
    ) {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "execution mode {} requires a scheduled shared engine and buffers, which v1 does \
                 not emit (supported: dedicated, streaming)",
                plan.execution_mode.as_str()
            ),
        });
    }
    let step = &plan.steps[0];
    let layer = step.op.clone();

    let op = match step.op_kind {
        OpKind::PointwiseConv2d => PlannedOp::PointwiseDot,
        OpKind::Conv2d if step.kernel == [1, 1] && step.groups == 1 => PlannedOp::PointwiseDot,
        other => {
            return Err(RtlError::UnsupportedOperator {
                layer,
                op: planned_op(other),
                supported: "pointwise_dot: PointwiseConv2d, or Conv2d with kernel [1,1] and groups=1",
            });
        }
    };

    if step.tap_parallel != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {layer}: tap_parallel {} is not emitted by {RTL_KERNEL_ID}; a 1x1 kernel has \
                 a single tap, so the planner must assign tap_parallel = 1",
                step.tap_parallel
            ),
        });
    }
    if step.kernel != [1, 1] {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {layer}: kernel {:?} is not a 1x1 pointwise kernel",
                step.kernel
            ),
        });
    }
    if step.stride != [1, 1] {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {layer}: stride {:?} changes the spatial mapping, which the streaming dot \
                 kernel does not implement",
                step.stride
            ),
        });
    }
    if let Some(bias) = &step.bias {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {layer}: bias tensor {} is not implemented by {RTL_KERNEL_ID}; the planner \
                 must fold the bias into the requantization or clear it",
                bias.name
            ),
        });
    }

    let weight_ref = step
        .weights
        .as_ref()
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!("step {layer}: a pointwise step without a weight tensor is malformed"),
        })?;
    if !plan.weights.iter().any(|w| w.id == weight_ref.id) {
        return Err(RtlError::WeightsNotResident {
            layer,
            tensor: weight_ref.name.clone(),
            detail: format!(
                "it is absent from ExecutionPlan::weights, so the plan does not keep it resident \
                 in ROM (placement would be {:?})",
                buffer_placement(plan, &step.input_buffer)
            ),
        });
    }

    let shape = weight_ref.shape;
    if shape.h != 1 || shape.w != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {layer}: weight tensor {} has shape {}x{}x{}x{}, but only pointwise 1x1 \
                 tensors are executable",
                weight_ref.name, shape.n, shape.c, shape.h, shape.w
            ),
        });
    }
    if shape.n != step.out_channels || shape.c != step.in_channels {
        return Err(RtlError::ShapeMismatch {
            layer,
            what: format!(
                "weight tensor {} shape (n=out_channels, c=in_channels)",
                weight_ref.name
            ),
            expected: u64::from(step.out_channels) * u64::from(step.in_channels),
            actual: u64::from(shape.n) * u64::from(shape.c),
        });
    }

    let simd = step.simd;
    let noc = step.pe;
    if simd == 0 || noc == 0 {
        return Err(RtlError::InvalidKernelParam {
            layer,
            field: "simd/pe",
            detail: format!("engine folding must be non-zero (simd {simd}, pe {noc})"),
        });
    }
    if step.in_channels % simd != 0 {
        return Err(RtlError::InvalidKernelParam {
            layer,
            field: "simd",
            detail: format!(
                "in_channels {} is not divisible by simd {simd}",
                step.in_channels
            ),
        });
    }
    if step.out_channels % noc != 0 {
        return Err(RtlError::InvalidKernelParam {
            layer,
            field: "pe",
            detail: format!(
                "out_channels {} is not divisible by pe {noc}",
                step.out_channels
            ),
        });
    }
    let tile = TileGeometry {
        simd,
        noc,
        k: step.in_channels / simd,
        fold: step.out_channels / noc,
    };

    let raw = weights.tensor_bytes(weight_ref)?;
    weight_digest_input.extend_from_slice(&raw);
    let expected_elements = u64::from(step.out_channels) * u64::from(step.in_channels);
    let flat = match weight_ref.elem {
        LayoutElemType::I8 => {
            if raw.len() as u64 != expected_elements {
                return Err(RtlError::ShapeMismatch {
                    layer,
                    what: format!("weight tensor {} bytes", weight_ref.name),
                    expected: expected_elements,
                    actual: raw.len() as u64,
                });
            }
            raw.iter().map(|b| *b as i8).collect::<Vec<i8>>()
        }
        LayoutElemType::I4 => unpack_i4_x2(&layer, &weight_ref.name, &raw, expected_elements)?,
        other => {
            return Err(RtlError::UnsupportedElementType {
                layer,
                tensor: weight_ref.name.clone(),
                element_type: other.into(),
            });
        }
    };
    let ordered = reorder_pointwise_weights(&layer, &flat, tile)?;

    let channels_count = u64::from(tile.noc) * u64::from(tile.fold);
    if channels_count > MAX_WEIGHT_ROM_ENTRIES {
        return Err(RtlError::InvalidKernelParam {
            layer,
            field: "pe * (out_channels / pe)",
            detail: format!(
                "the folded layer needs {channels_count} requantization channels, above the \
                 emitted limit of {MAX_WEIGHT_ROM_ENTRIES}; the planner must tile the layer"
            ),
        });
    }
    // The emitted requant ROM is per output channel, so a plan that carries
    // per-channel multipliers is honoured verbatim.
    let (channels, multiplier_error_ppm) =
        map_requant_channels(step, &layer, channels_count as usize)?;

    let input_placement = buffer_stream_placement(plan, &layer, "input", &step.input_buffer)?;
    buffer_stream_placement(plan, &layer, "output", &step.output_buffer)?;

    Ok(vec![KernelPlanView {
        layer: layer.clone(),
        op,
        activation_element: ElementType::I8,
        weight_element: ElementType::I8,
        output_element: ElementType::I8,
        tile,
        quant: QuantPlan {
            activation_zero_point: 0,
            weight_zero_point: 0,
            rounding: RoundingMode::RoundHalfUpMagnitude,
            channels,
            multiplier_error_ppm,
        },
        weights: WeightTensor { data: ordered },
        weight_placement: PlacementHint::WeightRom,
        activation_placement: input_placement,
        context: PlanContext {
            plan_format: plan.format.clone(),
            plan_schema_version: plan.schema_version,
            execution_mode: plan.execution_mode.as_str().to_string(),
            granularity: granularity_name(plan),
            engine: plan.engine.name.clone(),
            step_index: step.step,
            step_count: plan.steps.len() as u32,
            estimated_cycles: step.estimated_cycles,
            spatial_in: step.in_spatial,
            spatial_out: step.out_spatial,
        },
    }])
}

/// Maps the plan's requantization plus activation clamp into the kernel's
/// per-channel requantization. `Requant::per_channel_multipliers` (one entry per
/// output channel, execution order) is used verbatim when present; otherwise the
/// scalar `m0` is replicated.
fn map_requant_channels(
    step: &crate::layout::ScheduleStep,
    layer: &str,
    channels_count: usize,
) -> Result<(Vec<ChannelRequant>, Option<u32>), RtlError> {
    let requant = step.requant.as_ref().ok_or_else(|| RtlError::UnsupportedQuantization {
        layer: layer.to_string(),
        detail: format!(
            "step has no requantization contract (`Requant: None`); {RTL_KERNEL_ID} never infers \
             one"
        ),
    })?;
    if requant.rounding != Rounding::HalfAwayFromZero {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "rounding {:?} is not reproducible by the generated datapath; supported: \
                 HalfAwayFromZero",
                requant.rounding
            ),
        });
    }
    if !(0..=crate::layout::REQUANT_MAX_MULTIPLIER).contains(&requant.m0) {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "m0 {} does not fit the {MULTIPLIER_BITS}-bit multiplier ROM (representable \
                 range [0, {}])",
                requant.m0,
                crate::layout::REQUANT_MAX_MULTIPLIER
            ),
        });
    }
    if !(0..=31).contains(&requant.shift) {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "shift {} is outside the emitted range 0..=31",
                requant.shift
            ),
        });
    }
    if !requant.per_channel_multipliers.is_empty()
        && requant.per_channel_multipliers.len() != channels_count
    {
        return Err(RtlError::ShapeMismatch {
            layer: layer.to_string(),
            what: "per-channel requantization multipliers".to_string(),
            expected: channels_count as u64,
            actual: requant.per_channel_multipliers.len() as u64,
        });
    }
    for (index, multiplier) in requant.per_channel_multipliers.iter().enumerate() {
        if !(0..=crate::layout::REQUANT_MAX_MULTIPLIER).contains(multiplier) {
            return Err(RtlError::UnsupportedQuantization {
                layer: layer.to_string(),
                detail: format!(
                    "channel {index}: multiplier {multiplier} does not fit the \
                     {MULTIPLIER_BITS}-bit multiplier ROM (representable range [0, {}])",
                    crate::layout::REQUANT_MAX_MULTIPLIER
                ),
            });
        }
    }

    let (mut lo, mut hi) = step.activation.effective_clamp(Some(requant));
    match step.activation.kind {
        ActivationKind::None => {}
        ActivationKind::Relu => lo = lo.max(0),
        ActivationKind::Relu6 => {
            lo = lo.max(0);
            hi = hi.min(6);
        }
        ActivationKind::UnsignedClamp => {
            return Err(RtlError::UnsupportedQuantization {
                layer: layer.to_string(),
                detail: "activation kind UnsignedClamp produces an unsigned output range, which \
                         the signed int8 kernel cannot represent"
                    .to_string(),
            });
        }
    }
    if lo > hi {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "the requantization saturate range and the activation clamp do not intersect \
                 (lo {lo} > hi {hi})"
            ),
        });
    }
    if lo < -128 || hi > 127 {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "clamp range [{lo}, {hi}] does not fit the signed int8 output stream; the planner \
                 must emit QuantizeOutput for wider ranges"
            ),
        });
    }
    let template = ChannelRequant {
        multiplier: requant.m0 as u32,
        shift: requant.shift as u8,
        lo: lo as i16,
        hi: hi as i16,
    };
    let channels = if requant.per_channel_multipliers.is_empty() {
        uniform_requant(channels_count as u32, template)
    } else {
        requant
            .per_channel_multipliers
            .iter()
            .map(|multiplier| ChannelRequant {
                multiplier: *multiplier as u32,
                ..template
            })
            .collect()
    };
    Ok((channels, requant.multiplier_error_ppm))
}

fn buffer_stream_placement(
    plan: &ExecutionPlan,
    layer: &str,
    what: &'static str,
    name: &str,
) -> Result<PlacementHint, RtlError> {
    let binding = plan
        .buffers
        .iter()
        .find(|b| b.name == name)
        .ok_or_else(|| RtlError::UnknownBuffer {
            layer: layer.to_string(),
            what,
            name: name.to_string(),
        })?;
    match binding.placement {
        MemoryPlacement::Register | MemoryPlacement::Fifo => Ok(PlacementHint::StreamedActivation),
        other => Ok(other.into()),
    }
}

fn buffer_placement(plan: &ExecutionPlan, name: &str) -> Option<MemoryPlacement> {
    plan.buffers
        .iter()
        .find(|b| b.name == name)
        .map(|b| b.placement)
}

fn granularity_name(plan: &ExecutionPlan) -> String {
    if plan.steps.len() == 1 {
        match plan.steps[0].mode {
            crate::layout::Granularity::Frame => "frame",
            crate::layout::Granularity::Row => "row",
            crate::layout::Granularity::Tile => "tile",
        }
        .to_string()
    } else {
        "mixed".to_string()
    }
}

fn planned_op(kind: OpKind) -> PlannedOp {
    match kind {
        OpKind::PointwiseConv2d => PlannedOp::PointwiseDot,
        OpKind::Conv2d => PlannedOp::Conv2d,
        OpKind::DepthwiseConv2d => PlannedOp::Depthwise,
        OpKind::Add => PlannedOp::Add,
        OpKind::Requant => PlannedOp::Requant,
        OpKind::Activation => PlannedOp::Activation,
        OpKind::NearestUpsample2x => PlannedOp::UpsampleNearest,
        OpKind::QuantizeOutput => PlannedOp::QuantizeOutput,
    }
}

// ---------------------------------------------------------------------------
// SystemVerilog emission
// ---------------------------------------------------------------------------

/// SystemVerilog signed decimal literal. Negative values are emitted as a
/// unary minus on a magnitude literal, because `8'sd-4` is not valid syntax.
fn sv_signed(width: u32, value: i64) -> String {
    if value < 0 {
        format!("-{width}'sd{}", value.unsigned_abs())
    } else {
        format!("{width}'sd{value}")
    }
}

fn provenance_header(provenance: &Provenance, note: &str, kernel: &RtlKernel) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated by tessera {RTL_KERNEL_ID}. Do not edit by hand."
    );
    let _ = writeln!(
        out,
        "// compiler_version: {}  git_revision: {}  dirty: {}",
        provenance.compiler_version, provenance.git_revision, provenance.dirty
    );
    let _ = writeln!(
        out,
        "// target: {} ({})  plan_schema_version: {}  graph_schema_version: {}",
        provenance.target,
        provenance.target_revision,
        provenance.plan_schema_version,
        provenance.graph_schema_version
    );
    let _ = writeln!(
        out,
        "// input_manifest_sha256: {}  weights_digest: {}  planner_config_sha256: {}",
        provenance.input_manifest_sha256,
        provenance.weights_sha256,
        provenance.planner_config_sha256
    );
    let _ = writeln!(
        out,
        "// plan: {} ({})  mode: {}  granularity: {}  engine: {}  step {}/{}  \
         plan_estimated_cycles: {}",
        kernel.context.plan_format,
        kernel.context.plan_schema_version,
        kernel.context.execution_mode,
        kernel.context.granularity,
        kernel.context.engine,
        kernel.context.step_index,
        kernel.context.step_count,
        kernel.context.estimated_cycles
    );
    let _ = writeln!(
        out,
        "// layer: {}  tile: simd={} noc={} k={} fold={}  in_channels={} out_channels={}",
        kernel.layer,
        kernel.tile.simd,
        kernel.tile.noc,
        kernel.tile.k,
        kernel.tile.fold,
        kernel.tile.in_channels(),
        kernel.tile.out_channels()
    );
    let _ = writeln!(
        out,
        "// acc_width={}  max_abs_acc={}  cycles_per_window={} (estimate)",
        kernel.acc_width,
        kernel.max_abs_acc(),
        kernel.cycles_per_window()
    );
    let _ = writeln!(out, "// config: {note}");
    out
}

fn emit_dut(kernel: &RtlKernel, dut_name: &str, provenance: &Provenance, note: &str) -> String {
    let simd = kernel.tile.simd;
    let noc = kernel.tile.noc;
    let k = kernel.tile.k;
    let fold = kernel.tile.fold;
    let xw = kernel.input_width;
    let ww = kernel.weight_width;
    let ow = kernel.out_width;
    let accw = kernel.acc_width;
    let wab = kernel.k_bits();
    let fab = kernel.fold_bits();
    let shw = kernel.shift_bits();
    let pw = ACC_MAG_BITS + MULTIPLIER_BITS;
    let out_channels = (fold * noc) as usize;

    let mut s = provenance_header(provenance, note, kernel);
    let _ = writeln!(s, "//");
    let _ = writeln!(
        s,
        "// Streaming pointwise engine: K={k} beats of {simd} signed int8 activation lanes form a"
    );
    let _ = writeln!(
        s,
        "// reduction window; the engine emits NF={fold} beats of {noc} requantized signed int8"
    );
    let _ = writeln!(
        s,
        "// outputs (out_channels={out_channels}). Weights live in the inline ROM."
    );
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s, "module {dut_name} #(");
    let _ = writeln!(s, "    parameter int    SIMD  = {simd},");
    let _ = writeln!(s, "    parameter int    NOC   = {noc},");
    let _ = writeln!(s, "    parameter int    K     = {k},");
    let _ = writeln!(s, "    parameter int    NF    = {fold},");
    let _ = writeln!(s, "    parameter int    XW    = {xw},");
    let _ = writeln!(s, "    parameter int    WW    = {ww},");
    let _ = writeln!(s, "    parameter int    OW    = {ow},");
    let _ = writeln!(s, "    parameter int    ACCW  = {accw},");
    let _ = writeln!(
        s,
        "    // Optional external weight ROM image; empty selects the inline contents below."
    );
    let _ = writeln!(s, "    parameter string WFILE = \"\"");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  logic               clk,");
    let _ = writeln!(s, "    input  logic               rst,");
    let _ = writeln!(s, "    input  logic [SIMD*XW-1:0] in_data,");
    let _ = writeln!(s, "    input  logic               in_valid,");
    let _ = writeln!(s, "    output logic               in_ready,");
    let _ = writeln!(s, "    output logic [NOC*OW-1:0]  out_data,");
    let _ = writeln!(s, "    output logic               out_valid,");
    let _ = writeln!(s, "    input  logic               out_ready");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    localparam int IWA = SIMD*XW;");
    let _ = writeln!(s, "    localparam int OWA = NOC*OW;");
    let _ = writeln!(
        s,
        "    localparam int OC  = NF*NOC;          // output channels per window"
    );
    let _ = writeln!(
        s,
        "    localparam int WAB = {wab};   // reduction beat index width"
    );
    let _ = writeln!(
        s,
        "    localparam int FAB = {fab};   // output fold index width"
    );
    let _ = writeln!(
        s,
        "    localparam int SHW = {shw};   // per-channel shift width"
    );
    let _ = writeln!(
        s,
        "    localparam int PW  = {pw};   // (ACCW+1)-bit magnitude x {MULTIPLIER_BITS}-bit M0"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    typedef enum logic [1:0] {{S_LOAD, S_MAC, S_RQ, S_OUT}} state_t;"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // ---- weight ROM: (fold, k, noc, simd) signed int8 ----"
    );
    let _ = writeln!(s, "    logic signed [WW-1:0] w_rom [0:NF*K*NOC*SIMD-1];");
    let _ = writeln!(
        s,
        "    // ---- per-channel requantization ROM, (fold, noc) order ----"
    );
    let _ = writeln!(
        s,
        "    logic        [{}:0] m_rom  [0:OC-1];",
        MULTIPLIER_BITS - 1
    );
    let _ = writeln!(s, "    logic        [SHW-1:0] sh_rom [0:OC-1];");
    let _ = writeln!(s, "    logic signed [9:0]     lo_rom [0:OC-1];");
    let _ = writeln!(s, "    logic signed [9:0]     hi_rom [0:OC-1];");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        if (WFILE == \"\") begin");
    for (index, weight) in kernel.weights.iter().enumerate() {
        let _ = writeln!(
            s,
            "            w_rom[{index}] = {};",
            sv_signed(ww, i64::from(*weight))
        );
    }
    for (index, channel) in kernel.requant.iter().enumerate() {
        let _ = writeln!(
            s,
            "            m_rom[{index}]  = {MULTIPLIER_BITS}'d{};",
            channel.multiplier
        );
        let _ = writeln!(s, "            sh_rom[{index}] = {shw}'d{};", channel.shift);
        let _ = writeln!(
            s,
            "            lo_rom[{index}] = {};",
            sv_signed(10, i64::from(channel.lo))
        );
        let _ = writeln!(
            s,
            "            hi_rom[{index}] = {};",
            sv_signed(10, i64::from(channel.hi))
        );
    }
    let _ = writeln!(s, "        end else begin");
    let _ = writeln!(s, "            $readmemh(WFILE, w_rom);");
    let _ = writeln!(
        s,
        "            // Per-channel parameters stay inline: they are plan data,"
    );
    let _ = writeln!(s, "            // not a swappable memory image.");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    // Activation window buffer: K beats of SIMD lanes.");
    let _ = writeln!(s, "    logic signed [XW-1:0] x_buf [0:K*SIMD-1];");
    let _ = writeln!(s);
    let _ = writeln!(s, "    state_t  state;");
    let _ = writeln!(s, "    logic [WAB-1:0] kk;");
    let _ = writeln!(s, "    logic [FAB-1:0] ff;");
    let _ = writeln!(s, "    logic signed [ACCW-1:0] acc [0:NOC-1];");
    let _ = writeln!(s, "    logic [OWA-1:0] q_r;");
    let _ = writeln!(s, "    logic out_v;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    assign in_ready  = (state == S_LOAD);");
    let _ = writeln!(s, "    assign out_valid = out_v;");
    let _ = writeln!(s, "    assign out_data  = q_r;");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // One MAC step of fold ff: first ? product : acc[n] + sum_l w[ff][kk][n][l] * x[kk][l]"
    );
    let _ = writeln!(
        s,
        "    function automatic logic signed [ACCW-1:0] mac_step(input int n, input logic first);"
    );
    let _ = writeln!(s, "        logic signed [ACCW-1:0] sum;");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            sum = first ? ACCW'(0) : acc[n];");
    let _ = writeln!(s, "            for (int l = 0; l < SIMD; l = l + 1) begin");
    let _ = writeln!(
        s,
        "                sum = sum + ACCW'($signed(w_rom[((int'(ff)*K + int'(kk))*NOC + n)*SIMD + l]) * $signed(x_buf[kk*SIMD + l]));"
    );
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "            mac_step = sum;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endfunction");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // sts_rq fx requantization for output lane n of fold ff:"
    );
    let _ = writeln!(
        s,
        "    //   k = ((|acc| * M0) + 2^(sh-1)) >> sh ; out = clamp(sign(acc)*k, lo, hi)"
    );
    let _ = writeln!(
        s,
        "    function automatic logic signed [OW-1:0] requant(input int n);"
    );
    let _ = writeln!(s, "        logic [ACCW:0]         sx;");
    let _ = writeln!(s, "        logic [ACCW:0]         mag;");
    let _ = writeln!(s, "        logic [PW-1:0]         prod;");
    let _ = writeln!(s, "        logic [PW-1:0]         rounded;");
    let _ = writeln!(s, "        logic [SHW-1:0]        sh;");
    let _ = writeln!(s, "        logic signed [PW-1:0]  kv;");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            sh   = sh_rom[ff*NOC + n];");
    let _ = writeln!(s, "            sx   = {{acc[n][ACCW-1], acc[n]}};");
    let _ = writeln!(s, "            mag  = acc[n][ACCW-1] ? (~sx + 1'b1) : sx;");
    let zero_pad = format!("{{{MULTIPLIER_BITS}{{1'b0}}}}");
    let mag_ext = format!("{{{zero_pad}, mag}}");
    let _ = writeln!(
        s,
        "            prod = PW'({mag_ext}) * PW'(m_rom[ff*NOC + n]);"
    );
    let _ = writeln!(
        s,
        "            // M0 is applied at full precision; no pre-shift is required."
    );
    let _ = writeln!(
        s,
        "            rounded = (sh == '0) ? prod : (prod + (PW'(1) << (sh - 1'b1)));"
    );
    let _ = writeln!(s, "            kv = PW'(rounded >> sh);");
    let _ = writeln!(s, "            if (acc[n][ACCW-1]) kv = -kv;");
    let _ = writeln!(
        s,
        "            if (kv > $signed(PW'(hi_rom[ff*NOC + n])))       requant = OW'(hi_rom[ff*NOC + n]);"
    );
    let _ = writeln!(
        s,
        "            else if (kv < $signed(PW'(lo_rom[ff*NOC + n])))  requant = OW'(lo_rom[ff*NOC + n]);"
    );
    let _ = writeln!(
        s,
        "            else                                            requant = OW'(kv);"
    );
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endfunction");
    let _ = writeln!(s);
    let _ = writeln!(s, "    always_ff @(posedge clk) begin");
    let _ = writeln!(s, "        if (rst) begin");
    let _ = writeln!(s, "            state <= S_LOAD;");
    let _ = writeln!(s, "            kk    <= '0;");
    let _ = writeln!(s, "            ff    <= '0;");
    let _ = writeln!(s, "            out_v <= 1'b0;");
    let _ = writeln!(s, "            q_r   <= '0;");
    let _ = writeln!(
        s,
        "            for (int n = 0; n < NOC; n = n + 1) acc[n] <= '0;"
    );
    let _ = writeln!(
        s,
        "            for (int i = 0; i < K*SIMD; i = i + 1) x_buf[i] <= '0;"
    );
    let _ = writeln!(s, "        end else begin");
    let _ = writeln!(s, "            unique case (state)");
    let _ = writeln!(s, "                S_LOAD: begin");
    let _ = writeln!(s, "                    if (in_valid) begin");
    let _ = writeln!(
        s,
        "                        for (int l = 0; l < SIMD; l = l + 1) x_buf[kk*SIMD + l] <= in_data[l*XW +: XW];"
    );
    let _ = writeln!(s, "                        if (kk == WAB'(K-1)) begin");
    let _ = writeln!(s, "                            kk    <= '0;");
    let _ = writeln!(s, "                            state <= S_MAC;");
    let _ = writeln!(s, "                        end else begin");
    let _ = writeln!(s, "                            kk <= kk + 1'b1;");
    let _ = writeln!(s, "                        end");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                S_MAC: begin");
    let _ = writeln!(
        s,
        "                    for (int n = 0; n < NOC; n = n + 1) acc[n] <= mac_step(n, kk == '0);"
    );
    let _ = writeln!(s, "                    if (kk == WAB'(K-1)) state <= S_RQ;");
    let _ = writeln!(
        s,
        "                    else                 kk    <= kk + 1'b1;"
    );
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                S_RQ: begin");
    let _ = writeln!(
        s,
        "                    for (int n = 0; n < NOC; n = n + 1) begin"
    );
    let _ = writeln!(s, "                        q_r[n*OW +: OW] <= requant(n);");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                    out_v <= 1'b1;");
    let _ = writeln!(s, "                    state <= S_OUT;");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                S_OUT: begin");
    let _ = writeln!(s, "                    if (out_ready) begin");
    let _ = writeln!(s, "                        out_v <= 1'b0;");
    let _ = writeln!(s, "                        kk    <= '0;");
    let _ = writeln!(s, "                        if (ff == FAB'(NF-1)) begin");
    let _ = writeln!(s, "                            ff    <= '0;");
    let _ = writeln!(s, "                            state <= S_LOAD;");
    let _ = writeln!(s, "                        end else begin");
    let _ = writeln!(s, "                            ff    <= ff + 1'b1;");
    let _ = writeln!(s, "                            state <= S_MAC;");
    let _ = writeln!(s, "                        end");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                default: state <= S_LOAD;");
    let _ = writeln!(s, "            endcase");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s, "`default_nettype wire");
    s
}

fn emit_weight_mem(kernel: &RtlKernel) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "// tessera {RTL_KERNEL_ID} weight ROM image");
    let _ = writeln!(
        s,
        "// layer: {}  order: (fold, k, noc, simd)  entries: {}",
        kernel.layer,
        kernel.weights.len()
    );
    for weight in &kernel.weights {
        let _ = writeln!(s, "{:02x}", (*weight as u8) as u32);
    }
    s
}

fn emit_requant_mem(kernel: &RtlKernel) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "// tessera {RTL_KERNEL_ID} per-channel requantization: m0 (17b), shift, lo, hi"
    );
    for channel in &kernel.requant {
        let _ = writeln!(
            s,
            "{:05x} {:02x} {:03x} {:03x}",
            channel.multiplier,
            channel.shift,
            (channel.lo as u16) & 0x3ff,
            (channel.hi as u16) & 0x3ff
        );
    }
    s
}

fn emit_testbench(
    kernel: &RtlKernel,
    dut_name: &str,
    tb_name: &str,
    fixture: &Fixture,
    provenance: &Provenance,
) -> String {
    let simd = kernel.tile.simd;
    let noc = kernel.tile.noc;
    let k = kernel.tile.k;
    let fold = kernel.tile.fold;
    let iwa = simd * kernel.input_width;
    let owa = noc * kernel.out_width;
    let nv = fixture.window_count;
    let beats = nv * (fold as usize);
    let timeout = 4000 + (beats as u64) * (k as u64 + 16) * 4;

    let mut s = provenance_header(provenance, "self-checking testbench", kernel);
    let _ = writeln!(s, "//");
    let _ = writeln!(
        s,
        "// Drives {nv} reduction windows ({k} input beats each) through the DUT and compares all"
    );
    let _ = writeln!(
        s,
        "// {beats} output beats against the Rust reference (tessera::rtl::reference_window_output)."
    );
    let _ = writeln!(s, "// Pass: `TESSA_RTL_PASS` with mismatches=0.");
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s, "module {tb_name};");
    let _ = writeln!(s, "    localparam int SIMD = {simd};");
    let _ = writeln!(s, "    localparam int NOC  = {noc};");
    let _ = writeln!(s, "    localparam int K    = {k};");
    let _ = writeln!(s, "    localparam int NF   = {fold};");
    let _ = writeln!(s, "    localparam int IWA  = {iwa};");
    let _ = writeln!(s, "    localparam int OWA  = {owa};");
    let _ = writeln!(s, "    localparam int NV   = {nv};");
    let _ = writeln!(s, "    localparam int NB   = {};", beats);
    let _ = writeln!(s, "    localparam int TIMEOUT_CYCLES = {timeout};");
    let _ = writeln!(s);
    let _ = writeln!(s, "    logic clk = 1'b0;");
    let _ = writeln!(s, "    logic rst = 1'b1;");
    let _ = writeln!(s, "    logic in_valid = 1'b0;");
    let _ = writeln!(s, "    logic in_ready;");
    let _ = writeln!(s, "    logic [IWA-1:0] in_data = '0;");
    let _ = writeln!(s, "    logic out_valid;");
    let _ = writeln!(s, "    logic out_ready = 1'b0;");
    let _ = writeln!(s, "    logic [OWA-1:0] out_data;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    logic [IWA-1:0] stim [0:NV*K-1];");
    let _ = writeln!(s, "    logic [OWA-1:0] exp_q [0:NB-1];");
    let _ = writeln!(s, "    logic [OWA-1:0] got;");
    let _ = writeln!(s, "    int window = 0;");
    let _ = writeln!(s, "    int beat = 0;");
    let _ = writeln!(s, "    int mism = 0;");
    let _ = writeln!(s, "    int checks = 0;");
    let _ = writeln!(s, "    int cyc = 0;");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    {dut_name} #(.SIMD(SIMD), .NOC(NOC), .K(K), .NF(NF)) dut ("
    );
    let _ = writeln!(s, "        .clk(clk), .rst(rst),");
    let _ = writeln!(
        s,
        "        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),"
    );
    let _ = writeln!(
        s,
        "        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready));"
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        forever #5 clk = ~clk;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let iw_hex = (iwa as usize).div_ceil(4);
    let ow_hex = (owa as usize).div_ceil(4);
    for (index, values) in fixture.stimulus.iter().enumerate() {
        let packed = pack_lanes(values, kernel.input_width);
        let _ = writeln!(s, "        stim[{index}] = {iwa}'h{packed:0iw_hex$x};");
    }
    for (index, values) in fixture.expected.iter().enumerate() {
        let packed = pack_signed(values, kernel.out_width);
        let _ = writeln!(s, "        exp_q[{index}] = {owa}'h{packed:0ow_hex$x};");
    }
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        cyc <= cyc + 1;");
    let _ = writeln!(s, "        if (cyc > TIMEOUT_CYCLES) begin");
    let _ = writeln!(
        s,
        "            $display(\"TESSA_RTL_TIMEOUT cycles=%0d\", cyc);"
    );
    let _ = writeln!(s, "            $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "            $finish;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    task automatic send(input logic [IWA-1:0] data);");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            in_data  = data;");
    let _ = writeln!(s, "            in_valid = 1'b1;");
    let _ = writeln!(s, "            while (!in_ready) @(negedge clk);");
    let _ = writeln!(s, "            @(posedge clk);");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endtask");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // Pulls one output beat. Out of courtesy to the DUT's contract it"
    );
    let _ = writeln!(
        s,
        "    // holds out_ready low for two cycles and checks that out_data stays"
    );
    let _ = writeln!(s, "    // stable while out_valid is asserted.");
    let _ = writeln!(s, "    task automatic recv();");
    let _ = writeln!(s, "        logic [OWA-1:0] hold;");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            in_valid = 1'b0;");
    let _ = writeln!(s, "            in_data  = '0;");
    let _ = writeln!(s, "            while (!out_valid) @(negedge clk);");
    let _ = writeln!(s, "            hold = out_data;");
    let _ = writeln!(s, "            repeat (2) begin");
    let _ = writeln!(s, "                @(posedge clk);");
    let _ = writeln!(s, "                if (out_data !== hold) begin");
    let _ = writeln!(
        s,
        "                    $display(\"TESSA_RTL_ERROR unstable_output window=%0d\", window);"
    );
    let _ = writeln!(s, "                    mism = mism + 1;");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "            out_ready = 1'b1;");
    let _ = writeln!(s, "            @(posedge clk);");
    let _ = writeln!(s, "            out_ready = 1'b0;");
    let _ = writeln!(s, "            got = hold;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endtask");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        repeat (4) @(negedge clk);");
    let _ = writeln!(s, "        rst = 1'b0;");
    let _ = writeln!(
        s,
        "        for (window = 0; window < NV; window = window + 1) begin"
    );
    let _ = writeln!(
        s,
        "            for (beat = 0; beat < K; beat = beat + 1) begin"
    );
    let _ = writeln!(s, "                send(stim[window*K + beat]);");
    let _ = writeln!(s, "            end");
    let _ = writeln!(
        s,
        "            for (beat = 0; beat < NF; beat = beat + 1) begin"
    );
    let _ = writeln!(s, "                recv();");
    let _ = writeln!(s, "                checks = checks + 1;");
    let _ = writeln!(
        s,
        "                if (got !== exp_q[window*NF + beat]) begin"
    );
    let _ = writeln!(s, "                    mism = mism + 1;");
    let _ = writeln!(
        s,
        "                    $display(\"TESSA_RTL_MISMATCH window=%0d fold=%0d expected=%h actual=%h\", window, beat, exp_q[window*NF + beat], got);"
    );
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    let _ = writeln!(
        s,
        "        $display(\"TESSA_RTL_RESULT module={tb_name} windows=%0d beats=%0d checks=%0d mismatches=%0d\", NV, NB, checks, mism);"
    );
    let _ = writeln!(
        s,
        "        if (mism == 0 && checks == NB) $display(\"TESSA_RTL_PASS\");"
    );
    let _ = writeln!(s, "        else $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "        $finish;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s, "`default_nettype wire");
    s
}

/// Packs signed lanes little-endian (lane 0 in the low bits) for a testbench
/// literal. Negative lanes become two's-complement bit patterns.
fn pack_signed(values: &[i64], width: u32) -> u128 {
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let mut packed = 0u128;
    for (index, value) in values.iter().enumerate() {
        packed |= ((*value as u128) & mask) << (index as u32 * width);
    }
    packed
}

fn pack_lanes(values: &[i8], width: u32) -> u128 {
    let mask = if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    };
    let mut packed = 0u128;
    for (index, value) in values.iter().enumerate() {
        packed |= ((*value as u8 as u128) & mask) << (index as u32 * width);
    }
    packed
}

fn emit_run_script(dut_name: &str, tb_name: &str, kernel: &RtlKernel) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "#!/bin/sh");
    let _ = writeln!(
        s,
        "# Generated by tessera {RTL_KERNEL_ID} for step {}.",
        kernel.layer
    );
    let _ = writeln!(s, "# Requires Verilator (https://verilator.org).");
    let _ = writeln!(s, "set -eu");
    let _ = writeln!(s, "cd \"$(dirname \"$0\")\"");
    let _ = writeln!(s, "verilator --binary --timing -Wno-fatal \\");
    let _ = writeln!(
        s,
        "    --top-module {tb_name} -o {tb_name}_sim --Mdir obj_dir \\"
    );
    let _ = writeln!(s, "    {dut_name}.sv {tb_name}.sv");
    let _ = writeln!(s, "./obj_dir/{tb_name}_sim");
    s
}

// ---------------------------------------------------------------------------
// Verilator simulation scaffold
// ---------------------------------------------------------------------------

/// Result of trying to simulate a bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimulationOutcome {
    /// The testbench ran and reported zero mismatches.
    Passed { log: String },
    /// The testbench ran and reported mismatches, or a tool failed.
    Failed { log: String, detail: String },
    /// No simulator was available; callers must treat this as *not verified*
    /// and skip rather than claim success.
    ToolUnavailable { tool: String, reason: String },
}

impl SimulationOutcome {
    /// True only when the RTL was actually simulated and matched the reference.
    pub fn is_pass(&self) -> bool {
        matches!(self, SimulationOutcome::Passed { .. })
    }

    pub fn log(&self) -> &str {
        match self {
            SimulationOutcome::Passed { log } | SimulationOutcome::Failed { log, .. } => log,
            SimulationOutcome::ToolUnavailable { reason, .. } => reason,
        }
    }
}

/// Locates the Verilator binary, honouring `TESSERA_VERILATOR`.
pub fn verilator_version() -> Option<String> {
    let binary = std::env::var("TESSERA_VERILATOR").unwrap_or_else(|_| "verilator".to_string());
    let output = Command::new(&binary).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if version.is_empty() {
        None
    } else {
        Some(version)
    }
}

/// Writes the bundle to `work_dir`, compiles it with Verilator, runs the
/// generated testbench, and reports the result.
///
/// Returns [`SimulationOutcome::ToolUnavailable`] when Verilator is absent, so
/// callers can skip cleanly instead of asserting a fake success.
pub fn simulate_with_verilator(
    bundle: &RtlBundle,
    work_dir: &Path,
) -> Result<SimulationOutcome, RtlError> {
    let binary = std::env::var("TESSERA_VERILATOR").unwrap_or_else(|_| "verilator".to_string());
    let version = match verilator_version() {
        Some(version) => version,
        None => {
            return Ok(SimulationOutcome::ToolUnavailable {
                tool: binary,
                reason: "verilator not found on PATH (set TESSERA_VERILATOR to override)"
                    .to_string(),
            });
        }
    };

    if bundle
        .files()
        .iter()
        .all(|f| f.kind != RtlFileKind::Testbench)
    {
        return Err(RtlError::UnsupportedLayout {
            detail: "bundle has no testbench; enable RtlConfig::emit_testbench".to_string(),
        });
    }

    bundle.write_to(work_dir)?;
    let sources: Vec<String> = bundle
        .verilog_sources()
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    let obj_dir = work_dir.join("obj_dir");
    let top = bundle.metadata.top_module.clone();

    let mut command = Command::new(&binary);
    command
        .current_dir(work_dir)
        .arg("--binary")
        .arg("--timing")
        .arg("-Wno-fatal")
        .arg("-Wno-DECLFILENAME")
        .arg("--top-module")
        .arg(&top)
        .arg("--Mdir")
        .arg(&obj_dir)
        .arg("-o")
        .arg(format!("{top}_sim"));
    for source in &sources {
        command.arg(source);
    }
    let compile = command.output().map_err(|source| RtlError::Io {
        context: format!("running {binary}"),
        source,
    })?;
    let mut log = format!("--- verilator ({version}) ---\n");
    log.push_str(&String::from_utf8_lossy(&compile.stdout));
    log.push_str(&String::from_utf8_lossy(&compile.stderr));
    if !compile.status.success() {
        return Ok(SimulationOutcome::Failed {
            log,
            detail: format!("verilator exited with {}", compile.status),
        });
    }

    let sim_path = obj_dir.join(format!("{top}_sim"));
    let run = Command::new(&sim_path)
        .current_dir(work_dir)
        .output()
        .map_err(|source| RtlError::Io {
            context: format!("running {}", sim_path.display()),
            source,
        })?;
    log.push_str("--- simulation ---\n");
    log.push_str(&String::from_utf8_lossy(&run.stdout));
    log.push_str(&String::from_utf8_lossy(&run.stderr));

    if run.status.success() && log.contains("TESSA_RTL_PASS") {
        Ok(SimulationOutcome::Passed { log })
    } else {
        Ok(SimulationOutcome::Failed {
            log,
            detail: format!("simulator exited with {}", run.status),
        })
    }
}

// ---------------------------------------------------------------------------
// Textual self-tests (no external tools required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{
        Activation, EngineConfig, Granularity, Provenance as PlanProvenance, Requant,
        ResourceEstimate, ScheduleStep, TensorId, TensorShape,
    };

    /// Minimal plan source used by the textual tests.
    struct DirectPlan {
        views: Vec<KernelPlanView>,
        provenance: Provenance,
    }

    impl DirectPlan {
        fn single(view: KernelPlanView) -> Self {
            DirectPlan {
                views: vec![view],
                provenance: Provenance {
                    target: "ti60f225".to_string(),
                    target_revision: "2026-10-14".to_string(),
                    git_revision: "test".to_string(),
                    input_manifest_sha256: "11".repeat(32),
                    weights_sha256: "22".repeat(32),
                    planner_config_sha256: "33".repeat(32),
                    ..Provenance::default()
                },
            }
        }

        fn with_views(views: Vec<KernelPlanView>) -> Self {
            let provenance = DirectPlan::single(sample_view()).provenance;
            DirectPlan { views, provenance }
        }
    }

    impl RtlPlanSource for DirectPlan {
        fn artifact_format(&self) -> &str {
            "tessera.direct_kernel"
        }
        fn schema_version(&self) -> u32 {
            1
        }
        fn provenance(&self) -> Provenance {
            self.provenance.clone()
        }
        fn kernels(&self) -> Vec<KernelPlanView> {
            self.views.clone()
        }
    }

    fn sample_view() -> KernelPlanView {
        sample_view_tiled(TileGeometry {
            simd: 2,
            noc: 2,
            k: 3,
            fold: 2,
        })
    }

    fn sample_view_tiled(tile: TileGeometry) -> KernelPlanView {
        let entries = (tile.fold * tile.k * tile.noc * tile.simd) as usize;
        let weights: Vec<i8> = (0..entries).map(|i| ((i as i16 % 7) - 3) as i8).collect();
        let channel = ChannelRequant {
            multiplier: 1 << 16,
            shift: 8,
            lo: -128,
            hi: 127,
        };
        KernelPlanView {
            layer: "enc.0.pw".to_string(),
            op: PlannedOp::PointwiseDot,
            activation_element: ElementType::I8,
            weight_element: ElementType::I8,
            output_element: ElementType::I8,
            tile,
            quant: QuantPlan {
                activation_zero_point: 0,
                weight_zero_point: 0,
                rounding: RoundingMode::RoundHalfUpMagnitude,
                channels: uniform_requant(tile.noc * tile.fold, channel),
                multiplier_error_ppm: None,
            },
            weights: WeightTensor { data: weights },
            weight_placement: PlacementHint::WeightRom,
            activation_placement: PlacementHint::StreamedActivation,
            context: PlanContext::direct("enc.0.pw"),
        }
    }

    /// A real planner execution plan, built field by field so the adapter is
    /// exercised against the actual artifact contract.
    fn sample_execution_plan() -> ExecutionPlan {
        let in_channels = 8;
        let out_channels = 4;
        let mut buffers = vec![MemoryPlacement::Fifo, MemoryPlacement::Register]
            .into_iter()
            .enumerate()
            .map(|(index, placement)| crate::layout::BufferBinding {
                name: format!("buf{index}"),
                placement,
                region: crate::layout::BufferRegion::Stream,
                base_offset_bytes: 0,
                tensor: None,
                width_bits: 8,
                depth: 64,
                ram10_blocks: 1,
                bytes: 64,
                read_ports: 1,
                write_ports: 1,
                ping_pong_depth: 1,
                producers: vec![],
                consumers: vec![],
            })
            .collect::<Vec<_>>();
        buffers[0].producers.push("enc.0.pw".to_string());
        buffers[1].consumers.push("enc.0.pw".to_string());

        let weight_ref = TensorRef {
            id: TensorId(7),
            name: "enc.0.pw.weight".to_string(),
            shape: TensorShape {
                n: out_channels,
                c: in_channels,
                h: 1,
                w: 1,
            },
            elem: LayoutElemType::I8,
        };

        ExecutionPlan {
            format: crate::layout::PLAN_FORMAT.to_string(),
            schema_version: crate::layout::PLAN_SCHEMA_VERSION,
            provenance: PlanProvenance {
                graph_sha256: "aa".repeat(32),
                target_sha256: "bb".repeat(32),
                policy_sha256: "cc".repeat(32),
                compiler_version: "0.1.0".to_string(),
                target_name: "ti60f225".to_string(),
                target_revision: "2026-10-14".to_string(),
                policy_schema_version: 1,
                seed: 7,
                options: std::collections::BTreeMap::from([
                    ("git_revision".to_string(), "deadbeef".to_string()),
                    ("graph_schema_version".to_string(), "1".to_string()),
                ]),
            },
            certainty: crate::layout::PlanCertainty::BestEffort,
            execution_mode: ExecutionMode::Streaming,
            engine: EngineConfig {
                name: "stream_conv0".to_string(),
                simd: 2,
                pe: 2,
                tap_parallel: 1,
                activation_bits: 8,
                accumulator_bits: 32,
                weight_bits: 8,
                arena_words: 0,
            },
            tile: None,
            steps: vec![ScheduleStep {
                step: 0,
                op: "enc.0.pw".to_string(),
                op_kind: OpKind::PointwiseConv2d,
                engine: "stream_conv0".to_string(),
                mode: Granularity::Frame,
                tile: None,
                simd: 2,
                pe: 2,
                tap_parallel: 1,
                in_channels,
                out_channels,
                in_spatial: [64, 48],
                out_spatial: [64, 48],
                kernel: [1, 1],
                groups: 1,
                stride: [1, 1],
                padding: None,
                upsample_scale: None,
                in_code_range: None,
                out_code_range: None,
                requant: Some(Requant::uniform(
                    (1 << 16) + 4096,
                    18,
                    Rounding::HalfAwayFromZero,
                    -128,
                    127,
                )),
                activation: Activation::identity(-128, 127),
                weights: Some(weight_ref.clone()),
                bias: None,
                inputs: vec![TensorId(0)],
                outputs: vec![TensorId(1)],
                input_buffer: "buf0".to_string(),
                output_buffer: "buf1".to_string(),
                estimated_cycles: 12_345,
                cost: crate::layout::StepCost {
                    lanes: 4,
                    lane_iterations: 12_345,
                    macs: 12_345 * 4,
                    setup_cycles: 0,
                    utilisation_percent: 100,
                },
                fused_with: vec![],
            }],
            buffers,
            tensors: vec![weight_ref.clone()],
            weights: vec![weight_ref],
            cycles: crate::layout::CycleBreakdown {
                utilisation_percent: 100,
                mac_cycles: 12_345,
                setup_cycles: 0,
                dma_cycles: 0,
                total_cycles: 12_345,
            },
            resources: ResourceEstimate::ZERO,
            mapping: None,
        }
    }

    /// Resolver over a fixed weight table.
    struct FixedWeights(Vec<u8>);

    impl WeightResolver for FixedWeights {
        fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError> {
            let expected = tensor.shape.elements() as usize;
            if tensor.elem == LayoutElemType::I4 {
                assert_eq!(self.0.len(), expected.div_ceil(2));
            } else {
                assert_eq!(self.0.len(), expected);
            }
            Ok(self.0.clone())
        }
    }

    #[test]
    fn emits_deterministic_bundle_for_supported_kernel() {
        let plan = DirectPlan::single(sample_view());
        let config = RtlConfig::default();
        let backend = SystemVerilogBackend;
        let first = backend
            .lower(&plan, &config)
            .expect("supported plan lowers");
        let second = backend.lower(&plan, &config).expect("second lowering");

        assert_eq!(first.files(), second.files());
        assert_eq!(first.metadata.bundle_digest, second.metadata.bundle_digest);
        assert_eq!(
            first.metadata.tile,
            TileGeometry {
                simd: 2,
                noc: 2,
                k: 3,
                fold: 2
            }
        );
        assert_eq!(first.metadata.in_channels, 6);
        assert_eq!(first.metadata.out_channels, 4);
        assert_eq!(first.metadata.kernel_cycles_per_window, 3 + 2 * 5);
        assert_eq!(first.metadata.format, RTL_BUNDLE_FORMAT);
        assert_eq!(first.metadata.schema_version, RTL_BUNDLE_SCHEMA_VERSION);
        assert_eq!(first.metadata.provenance.target, "ti60f225");

        let dut = first
            .file("tessera_enc_0_pw_pw_dot_i8i8_v1.sv")
            .expect("dut file present");
        assert!(
            dut.contents
                .contains("module tessera_enc_0_pw_pw_dot_i8i8_v1")
        );
        assert!(dut.contents.contains("input_manifest_sha256"));
        assert!(dut.contents.contains("layer: enc.0.pw"));
        assert!(dut.contents.contains("parameter int    NF    = 2"));
        assert!(
            first
                .file("tessera_enc_0_pw_pw_dot_i8i8_v1_tb.sv")
                .is_some()
        );
        assert!(
            first
                .file("tessera_enc_0_pw_pw_dot_i8i8_v1.weights.mem")
                .is_some()
        );
        assert!(first.file("rtl_bundle.json").is_some());
        let json = first.file("rtl_bundle.json").unwrap();
        let parsed: RtlMetadata = serde_json::from_str(&json.contents).expect("metadata parses");
        assert_eq!(parsed.bundle_digest, first.metadata.bundle_digest);
    }

    #[test]
    fn rejects_unsupported_operators_with_the_step_name() {
        for (op, expected) in [
            (PlannedOp::Depthwise, "depthwise"),
            (PlannedOp::Conv2d, "conv2d"),
            (PlannedOp::Add, "add"),
            (PlannedOp::UpsampleNearest, "upsample_nearest"),
            (PlannedOp::Requant, "requant"),
            (PlannedOp::Activation, "activation"),
            (PlannedOp::QuantizeOutput, "quantize_output"),
        ] {
            let mut view = sample_view();
            view.op = op;
            let err = SystemVerilogBackend
                .lower(&DirectPlan::single(view), &RtlConfig::default())
                .expect_err("unsupported operator must be rejected");
            match err {
                RtlError::UnsupportedOperator {
                    ref layer, ref op, ..
                } => {
                    assert_eq!(layer, "enc.0.pw");
                    assert_eq!(op.to_string(), expected);
                }
                other => panic!("unexpected error: {other}"),
            }
            assert!(err.to_string().contains("enc.0.pw"));
        }
    }

    #[test]
    fn rejects_unsupported_quantization_elements_and_placement() {
        let mut view = sample_view();
        view.activation_element = ElementType::I4;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedElementType { .. })
        ));

        let mut view = sample_view();
        view.quant.activation_zero_point = -3;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedQuantization { .. })
        ));

        let mut view = sample_view();
        view.quant.rounding = RoundingMode::RoundHalfEven;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedQuantization { .. })
        ));

        let mut view = sample_view();
        view.quant.channels[1].multiplier = 1 << 17;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedQuantization { .. })
        ));

        // A denormalized (small) multiplier is representable and accepted.
        let mut view = sample_view();
        for channel in &mut view.quant.channels {
            channel.multiplier = 3;
            channel.shift = 1;
        }
        assert!(
            SystemVerilogBackend
                .lower(&DirectPlan::single(view), &RtlConfig::default())
                .is_ok()
        );

        let mut view = sample_view();
        view.weight_placement = PlacementHint::Ddr;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedPlacement { .. })
        ));

        let mut view = sample_view();
        view.activation_placement = PlacementHint::Bram;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::UnsupportedPlacement { .. })
        ));
    }

    #[test]
    fn rejects_geometry_and_shape_mismatches() {
        let mut view = sample_view();
        view.weights.data.pop();
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::ShapeMismatch { .. })
        ));

        let mut view = sample_view();
        view.quant.channels.pop();
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::ShapeMismatch { .. })
        ));

        let mut view = sample_view();
        view.tile.k = 0;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::InvalidKernelParam { .. })
        ));

        let mut view = sample_view();
        view.tile.k = 100_000;
        assert!(matches!(
            SystemVerilogBackend.lower(&DirectPlan::single(view), &RtlConfig::default()),
            Err(RtlError::AccumulatorOverflow { .. })
        ));
    }

    #[test]
    fn rejects_multi_kernel_and_empty_plans() {
        let plan = DirectPlan::with_views(vec![sample_view(), sample_view()]);
        let err = SystemVerilogBackend
            .lower(&plan, &RtlConfig::default())
            .expect_err("multi-step plan must be rejected");
        assert!(matches!(err, RtlError::UnsupportedLayout { .. }));
        assert!(err.to_string().contains("2 steps"));

        let plan = DirectPlan::with_views(vec![]);
        assert!(matches!(
            SystemVerilogBackend.lower(&plan, &RtlConfig::default()),
            Err(RtlError::UnsupportedLayout { .. })
        ));
    }

    #[test]
    fn rejects_missing_provenance() {
        let mut plan = DirectPlan::single(sample_view());
        plan.provenance.input_manifest_sha256 = String::new();
        assert!(matches!(
            SystemVerilogBackend.lower(&plan, &RtlConfig::default()),
            Err(RtlError::MissingProvenance { .. })
        ));
    }

    #[test]
    fn requant_reference_matches_sts_rq_fx_rounding() {
        let channel = ChannelRequant {
            multiplier: 1 << 16,
            shift: 24,
            lo: -128,
            hi: 127,
        };
        // M0 / 2^shift = 2^16 / 2^24 = 1/256 with round-half-away-from-zero.
        assert_eq!(reference_requant(0, &channel), 0);
        assert_eq!(reference_requant(255, &channel), 1);
        assert_eq!(reference_requant(256, &channel), 1);
        assert_eq!(reference_requant(-256, &channel), -1);
        assert_eq!(reference_requant(-128, &channel), -1);
        assert_eq!(reference_requant(65_536, &channel), 127);

        let channel = ChannelRequant {
            multiplier: (1 << 16) + (1 << 15),
            shift: 16,
            lo: 0,
            hi: 127,
        };
        assert_eq!(reference_requant(3, &channel), 5);
        assert_eq!(reference_requant(-3, &channel), 0);
    }

    #[test]
    fn reference_model_covers_every_fold() {
        let kernel = RtlKernel::analyze(&sample_view()).unwrap();
        let fixture = build_fixture(&kernel, 5, 42);
        assert_eq!(fixture.window_count, 5);
        assert_eq!(fixture.stimulus.len(), 5 * kernel.tile.k as usize);
        assert_eq!(fixture.expected.len(), 5 * kernel.tile.fold as usize);
        assert_eq!(fixture, build_fixture(&kernel, 5, 42));
        for window in 0..5 {
            let start = window * kernel.tile.k as usize;
            let slice = fixture.stimulus[start..start + kernel.tile.k as usize].to_vec();
            let values = reference_window_output(&kernel, &slice);
            for beat in 0..kernel.tile.fold as usize {
                let base = beat * kernel.tile.noc as usize;
                assert_eq!(
                    &fixture.expected[window * kernel.tile.fold as usize + beat],
                    &values[base..base + kernel.tile.noc as usize]
                );
            }
        }
    }

    #[test]
    fn testbench_embeds_the_reference_expectations() {
        let kernel = RtlKernel::analyze(&sample_view()).unwrap();
        let fixture = build_fixture(&kernel, 4, 7);
        let tb = emit_testbench(&kernel, "dut", "dut_tb", &fixture, &Provenance::default());
        assert!(tb.contains("TESSA_RTL_PASS"));
        assert!(tb.contains("TESSA_RTL_MISMATCH"));
        assert!(tb.contains(".NF(NF)"));
        for values in &fixture.expected {
            let packed = pack_signed(values, 8);
            assert!(tb.contains(&format!("{packed:04x}")));
        }
    }

    #[test]
    fn adapter_lowers_a_real_execution_plan() {
        let plan = sample_execution_plan();
        let weights = FixedWeights(
            (0..32)
                .map(|i| ((i as i16 % 9) - 4) as i8 as u8)
                .collect::<Vec<u8>>(),
        );
        let source = ExecutionPlanSource::build(&plan, &weights).expect("plan adapts");
        assert_eq!(source.artifact_format(), crate::layout::PLAN_FORMAT);
        let provenance = source.provenance();
        assert_eq!(provenance.target, "ti60f225");
        assert_eq!(provenance.git_revision, "deadbeef");
        assert_eq!(provenance.input_manifest_sha256, "aa".repeat(32));
        assert_eq!(provenance.planner_config_sha256, "cc".repeat(32));
        assert!(provenance.weights_sha256.starts_with("fnv1a64:"));

        let bundle = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect("plan lowers");
        assert_eq!(
            bundle.metadata.tile,
            TileGeometry {
                simd: 2,
                noc: 2,
                k: 4,
                fold: 2
            }
        );
        assert_eq!(bundle.metadata.plan.engine, "stream_conv0");
        assert_eq!(bundle.metadata.plan.estimated_cycles, 12_345);
        assert_eq!(bundle.metadata.plan.spatial_in, [64, 48]);
        assert_eq!(bundle.metadata.plan.execution_mode, "streaming");
        assert_eq!(bundle.metadata.in_channels, 8);
        assert_eq!(bundle.metadata.out_channels, 4);
    }

    #[test]
    fn adapter_rejects_unrepresentable_plans() {
        // Two steps: the plan is a graph, not a single kernel.
        let mut plan = sample_execution_plan();
        let extra = plan.steps[0].clone();
        plan.steps.push(extra);
        plan.steps[1].op = "enc.1.pw".to_string();
        let weights = FixedWeights(vec![0u8; 32]);
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("two steps must be rejected");
        assert!(err.to_string().contains("2 steps"), "{err}");

        // Depthwise op.
        let mut plan = sample_execution_plan();
        plan.steps[0].op_kind = OpKind::DepthwiseConv2d;
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("depthwise must be rejected");
        assert!(matches!(err, RtlError::UnsupportedOperator { .. }), "{err}");

        // 3x3 kernel.
        let mut plan = sample_execution_plan();
        plan.steps[0].op_kind = OpKind::Conv2d;
        plan.steps[0].kernel = [3, 3];
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("3x3 must be rejected");
        assert!(matches!(err, RtlError::UnsupportedOperator { .. }), "{err}");

        // Missing requantization.
        let mut plan = sample_execution_plan();
        plan.steps[0].requant = None;
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("missing requant must be rejected");
        assert!(
            matches!(err, RtlError::UnsupportedQuantization { .. }),
            "{err}"
        );

        // Multiplier that does not fit the 17-bit field.
        let mut plan = sample_execution_plan();
        plan.steps[0].requant = Some(Requant::uniform(
            1 << 17,
            3,
            Rounding::HalfAwayFromZero,
            -128,
            127,
        ));
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("out-of-range m0 must be rejected");
        assert!(err.to_string().contains("m0 131072"), "{err}");

        // A small (denormalized) multiplier is representable, so it lowers.
        let mut plan = sample_execution_plan();
        plan.steps[0].requant = Some(Requant::uniform(
            5,
            3,
            Rounding::HalfAwayFromZero,
            -128,
            127,
        ));
        let bundle = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect("denormalized m0 is representable");
        assert_eq!(bundle.metadata.requantization[0].multiplier, 5);

        // Bias term.
        let mut plan = sample_execution_plan();
        plan.steps[0].bias = Some(TensorRef {
            id: TensorId(9),
            name: "enc.0.pw.bias".to_string(),
            shape: TensorShape {
                n: 4,
                c: 1,
                h: 1,
                w: 1,
            },
            elem: LayoutElemType::I32,
        });
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("bias must be rejected");
        assert!(err.to_string().contains("bias"), "{err}");

        // Shared-tile execution mode.
        let mut plan = sample_execution_plan();
        plan.execution_mode = ExecutionMode::SharedTile;
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("shared tile must be rejected");
        assert!(err.to_string().contains("shared_tile"), "{err}");

        // DDR-resident activation buffer.
        let mut plan = sample_execution_plan();
        plan.buffers[0].placement = MemoryPlacement::Ddr;
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("ddr activation must be rejected");
        assert!(
            matches!(err, RtlError::UnsupportedPlacement { .. }),
            "{err}"
        );

        // Unknown buffer.
        let mut plan = sample_execution_plan();
        plan.steps[0].input_buffer = "missing".to_string();
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("unknown buffer must be rejected");
        assert!(matches!(err, RtlError::UnknownBuffer { .. }), "{err}");

        // Non-resident weights.
        let mut plan = sample_execution_plan();
        plan.weights.clear();
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("non-resident weights must be rejected");
        assert!(matches!(err, RtlError::WeightsNotResident { .. }), "{err}");

        // Indivisible folding.
        let mut plan = sample_execution_plan();
        plan.steps[0].simd = 3;
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("indivisible simd must be rejected");
        assert!(err.to_string().contains("not divisible"), "{err}");
    }

    #[test]
    fn adapter_rejects_unsigned_output_activation() {
        let mut plan = sample_execution_plan();
        plan.steps[0].activation = Activation {
            kind: ActivationKind::UnsignedClamp,
            clamp_lo: 0,
            clamp_hi: 255,
        };
        let weights = FixedWeights(vec![0u8; 32]);
        let err = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect_err("unsigned output must be rejected");
        assert!(
            matches!(err, RtlError::UnsupportedQuantization { .. }),
            "{err}"
        );
    }

    #[test]
    fn adapter_honours_per_channel_multipliers() {
        let mut plan = sample_execution_plan();
        let requant = plan.steps[0].requant.as_mut().expect("requant present");
        requant.per_channel_multipliers =
            vec![1 << 16, (1 << 16) + 1, (1 << 16) + 2, (1 << 16) + 3];
        requant.multiplier_error_ppm = Some(1500);
        let weights = FixedWeights(vec![0u8; 32]);

        let source = ExecutionPlanSource::build(&plan, &weights).expect("plan adapts");
        let channels = source.kernels()[0].quant.channels.clone();
        assert_eq!(channels.len(), 4, "one entry per output channel");
        assert_eq!(channels[0].multiplier, 1 << 16);
        assert_eq!(channels[3].multiplier, (1 << 16) + 3);
        assert_eq!(source.kernels()[0].quant.multiplier_error_ppm, Some(1500));

        let bundle = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect("per-channel plan lowers");
        assert_eq!(bundle.metadata.requant_multiplier_error_ppm, Some(1500));
        assert_eq!(bundle.metadata.requantization[3].multiplier, (1 << 16) + 3);
        let dut = bundle
            .file("tessera_enc_0_pw_pw_dot_i8i8_v1.sv")
            .expect("dut file");
        assert!(
            dut.contents
                .contains(&format!("m_rom[3]  = 17'd{};", (1 << 16) + 3)),
            "per-channel multiplier must reach the ROM"
        );

        // Wrong per-channel length is a shape error, not a silent scalar fallback.
        let mut plan = sample_execution_plan();
        plan.steps[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = vec![1 << 16];
        assert!(matches!(
            SystemVerilogBackend.lower_execution_plan(&plan, &weights, &RtlConfig::default()),
            Err(RtlError::ShapeMismatch { .. })
        ));

        // A per-channel multiplier outside the 17-bit field is rejected.
        let mut plan = sample_execution_plan();
        plan.steps[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = vec![1 << 16, 1 << 16, 1 << 17, 1 << 16];
        assert!(matches!(
            SystemVerilogBackend.lower_execution_plan(&plan, &weights, &RtlConfig::default()),
            Err(RtlError::UnsupportedQuantization { .. })
        ));
    }

    #[test]
    fn adapter_unpacks_i4_x2_weights_into_the_rom() {
        let mut plan = sample_execution_plan();
        plan.steps[0].weights.as_mut().unwrap().elem = LayoutElemType::I4;
        plan.weights[0].elem = LayoutElemType::I4;

        // 32 weight elements (4 out x 8 in) packed two per byte, low nibble
        // first; every code stays in 0..=7 so signed decoding is the identity.
        let packed: Vec<u8> = (0..16u32)
            .map(|index| {
                let low = (index % 8) as u8;
                let high = (7 - index % 8) as u8;
                (high << 4) | low
            })
            .collect();
        let weights = FixedWeights(packed);

        let bundle = SystemVerilogBackend
            .lower_execution_plan(&plan, &weights, &RtlConfig::default())
            .expect("i4 weights lower");
        let dut = bundle
            .file("tessera_enc_0_pw_pw_dot_i8i8_v1.sv")
            .expect("dut file");
        // out_channel 0, in_channels 0 and 1 are the two nibbles of byte 0.
        assert!(
            dut.contents.contains("w_rom[0] = 8'sd0;"),
            "{}",
            dut.contents
        );
        assert!(
            dut.contents.contains("w_rom[1] = 8'sd7;"),
            "{}",
            dut.contents
        );
        assert_eq!(bundle.metadata.requantization.len(), 4);
    }

    #[test]
    fn adapter_applies_relu_clamp() {
        let mut plan = sample_execution_plan();
        plan.steps[0].activation = Activation {
            kind: ActivationKind::Relu6,
            clamp_lo: -128,
            clamp_hi: 127,
        };
        let weights = FixedWeights(vec![0u8; 32]);
        let source = ExecutionPlanSource::build(&plan, &weights).expect("plan adapts");
        let channels = source.kernels()[0].quant.channels.clone();
        assert_eq!(channels.len(), 4);
        for channel in channels {
            assert_eq!((channel.lo, channel.hi), (0, 6));
        }
    }

    #[test]
    fn unpacks_i4_x2_low_nibble_first_with_sign() {
        // 0x7f -> low 0xf = -1, high 0x7 = 7 ; 0x80 -> low 0x0 = 0, high 0x8 = -8
        let unpacked = unpack_i4_x2("l", "t", &[0x7f, 0x80], 4).unwrap();
        assert_eq!(unpacked, vec![-1, 7, 0, -8]);
        // Odd element count: the trailing (high) nibble is padding and ignored.
        let unpacked = unpack_i4_x2("l", "t", &[0x7f, 0x0f], 3).unwrap();
        assert_eq!(unpacked, vec![-1, 7, -1]);
        assert!(unpack_i4_x2("l", "t", &[0x7f], 3).is_err());
        assert_eq!(unpack_i4_x2("l", "t", &[], 0).unwrap(), Vec::<i8>::new());
    }

    #[test]
    fn reorders_weights_into_fold_k_noc_simd_order() {
        let tile = TileGeometry {
            simd: 2,
            noc: 2,
            k: 2,
            fold: 2,
        };
        // (out_channel, in_channel) with out_channel = fold*noc ordering.
        let weights: Vec<i8> = (0..16).collect();
        let ordered = reorder_pointwise_weights("l", &weights, tile).unwrap();
        // fold 0, k 0, noc 1 -> out_channel 1, in channels 0,1
        assert_eq!(&ordered[2..4], &[4, 5]);
        // fold 0, k 1, noc 0 -> out_channel 0, in channels 2,3
        assert_eq!(&ordered[4..6], &[2, 3]);
        // fold 1, k 0, noc 1 -> out_channel 3, in channels 0,1
        assert_eq!(&ordered[10..12], &[12, 13]);
        // fold 1, k 1, noc 1 -> out_channel 3, in channels 2,3
        assert_eq!(&ordered[14..16], &[14, 15]);
        assert!(reorder_pointwise_weights("l", &weights[..4], tile).is_err());
    }

    #[test]
    fn digest_is_stable_and_content_sensitive() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_ne!(fnv1a64(b"abc"), fnv1a64(b"abd"));
    }

    #[test]
    fn slug_is_module_safe() {
        assert_eq!(slug("enc.0.pw"), "enc_0_pw");
        assert_eq!(slug("1_stage"), "k1_stage");
        assert_eq!(slug(""), "unnamed");
    }

    #[test]
    fn write_to_materializes_every_file() {
        let plan = DirectPlan::single(sample_view());
        let bundle = SystemVerilogBackend
            .lower(&plan, &RtlConfig::default())
            .unwrap();
        let dir = std::env::temp_dir().join(format!(
            "tessera_rtl_write_{}_{}",
            std::process::id(),
            fnv1a64(b"write_to")
        ));
        let _ = fs::remove_dir_all(&dir);
        let written = bundle.write_to(&dir).expect("bundle writes");
        assert_eq!(written.len(), bundle.files().len());
        for file in bundle.files() {
            let path = dir.join(&file.name);
            assert!(path.exists(), "{} missing", file.name);
            assert_eq!(fs::read_to_string(&path).unwrap(), file.contents);
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
