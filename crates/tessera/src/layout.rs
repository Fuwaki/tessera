//! Layout vocabulary, the persisted `LayoutArtifact`, and the backend-ready
//! `ExecutionPlan` projection.
//!
//! A [`LayoutArtifact`] is the reproducible output of layout search: the chosen
//! execution mode, engine, tile geometry, fusion decisions, buffer placement,
//! schedule, resource/cycle/DDR estimates, slack, the objective that ranked
//! candidates, and a summary of rejected alternatives. It is self-describing
//! (`format`, `schema_version`, `provenance`) and self-contained: the resource
//! limits it was solved against are embedded so that [`LayoutArtifact::verify`]
//! and [`LayoutArtifact::execution_plan`] need no external inputs.
//!
//! Persistence uses the project TOML subset ([`crate::target::toml_lite`]) and
//! is deterministic: collections are normalized (sorted) before serialization,
//! so two equal artifacts always render byte-identically.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::target::toml_lite::{self, TomlError};

/// Layout artifact format tag.
pub const LAYOUT_FORMAT: &str = "tessera.layout";
/// Layout artifact schema version implemented by this build.
pub const LAYOUT_SCHEMA_VERSION: u32 = 1;
/// Execution-plan format tag.
pub const PLAN_FORMAT: &str = "tessera.execution_plan";
/// Execution-plan schema version implemented by this build.
pub const PLAN_SCHEMA_VERSION: u32 = 1;
/// Target format tag (re-exported so callers need not import `target`).
pub use crate::target::TARGET_FORMAT;
/// Target schema version supported by this build.
pub use crate::target::TARGET_SCHEMA_VERSION;

// ---------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------

/// Stable tensor identity within one graph artifact.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct TensorId(pub u32);

impl TensorId {
    /// Construct a tensor id.
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Raw index.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Stored element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElemType {
    /// Signed 8-bit.
    I8,
    /// Signed 4-bit packed codes (two per byte, low nibble first).
    I4,
    /// Unsigned 8-bit.
    U8,
    /// Signed 32-bit accumulator or bias.
    I32,
    /// IEEE binary32 (analysis only; never a hardware operand in v1).
    F32,
}

impl ElemType {
    /// Element width in bits.
    pub const fn bits(self) -> u32 {
        match self {
            ElemType::I8 | ElemType::U8 => 8,
            ElemType::I4 => 4,
            ElemType::I32 | ElemType::F32 => 32,
        }
    }
}

/// NCHW tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorShape {
    /// Batch (v1 models are batch-1).
    pub n: u32,
    /// Channels.
    pub c: u32,
    /// Height.
    pub h: u32,
    /// Width.
    pub w: u32,
}

impl TensorShape {
    /// Construct a shape.
    pub const fn new(n: u32, c: u32, h: u32, w: u32) -> Self {
        Self { n, c, h, w }
    }

    /// Total element count.
    pub fn elements(self) -> u64 {
        self.n as u64 * self.c as u64 * self.h as u64 * self.w as u64
    }

    /// Bytes needed to store this tensor at the given element type.
    pub fn bytes(self, elem: ElemType) -> u64 {
        // I4 packs two codes per byte.
        match elem {
            ElemType::I4 => self.elements().div_ceil(2),
            other => self.elements() * (other.bits() as u64 / 8),
        }
    }

    /// Spatial extent `[height, width]`.
    pub const fn spatial(self) -> [u32; 2] {
        [self.h, self.w]
    }
}

/// A tensor reference carried through the artifact: identity, name, shape and
/// element type. Never tensor contents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRef {
    /// Stable id.
    pub id: TensorId,
    /// Stable symbol name.
    pub name: String,
    /// Shape.
    pub shape: TensorShape,
    /// Stored element type.
    pub elem: ElemType,
}

impl TensorRef {
    /// Storage bytes of this tensor.
    pub fn bytes(&self) -> u64 {
        self.shape.bytes(self.elem)
    }
}

/// Operator kind in the lowered primitive graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    /// Dense 3x3 (or NxN) convolution.
    Conv2d,
    /// Depthwise convolution (`groups == in_channels`).
    DepthwiseConv2d,
    /// 1x1 convolution (`groups == 1`, kernel 1x1).
    PointwiseConv2d,
    /// Elementwise addition (residual/skip).
    Add,
    /// Standalone requantization.
    Requant,
    /// Standalone activation clamp.
    Activation,
    /// 2x nearest-neighbor upsampling.
    NearestUpsample2x,
    /// Final unsigned 8-bit output quantization.
    QuantizeOutput,
}

impl OpKind {
    /// Whether the operator performs a MAC reduction.
    pub const fn is_convolution(self) -> bool {
        matches!(
            self,
            OpKind::Conv2d | OpKind::DepthwiseConv2d | OpKind::PointwiseConv2d
        )
    }

    /// Whether the operator is a pure elementwise transform.
    pub const fn is_elementwise(self) -> bool {
        matches!(
            self,
            OpKind::Add | OpKind::Requant | OpKind::Activation | OpKind::QuantizeOutput
        )
    }
}

/// Physically distinct execution organisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// One specialized engine per operator.
    Dedicated,
    /// Layer-pipelined streaming engines with ready/valid handshakes.
    Streaming,
    /// One shared engine reused across operators with tile scheduling.
    SharedTile,
    /// Streaming elementwise paths plus a shared convolution engine.
    Hybrid,
}

impl ExecutionMode {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            ExecutionMode::Dedicated => "dedicated",
            ExecutionMode::Streaming => "streaming",
            ExecutionMode::SharedTile => "shared_tile",
            ExecutionMode::Hybrid => "hybrid",
        }
    }
}

/// Schedule granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Granularity {
    /// One pass over the whole frame.
    Frame,
    /// Row stripes.
    Row,
    /// Rectangular tiles.
    Tile,
}

/// Where a buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPlacement {
    /// Flip-flop registers.
    Register,
    /// Handshake FIFO / SRL storage.
    Fifo,
    /// Convolution line buffer.
    LineBuffer,
    /// Block RAM (RAM10).
    Bram,
    /// External DDR.
    Ddr,
}

/// Requantization rounding policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rounding {
    /// Round half away from zero (`rnd_half_away`), the manifest contract.
    HalfUp,
    /// Round half to even.
    HalfEven,
    /// Truncate toward negative infinity.
    Truncate,
}

/// Requant contract: normalized multiplier, right shift, rounding and the
/// quantizer's own output clamp applied after the multiply-shift.
///
/// `m0` is the scalar hardware operand (uniform across output channels). When
/// the source model declares per-output-channel multipliers they are carried
/// verbatim in `per_channel_multipliers` (same 17-bit encoding, one entry per
/// output channel, execution order); an empty vector means "uniform, use
/// `m0`". The planner never samples or rescales either form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requant {
    /// Normalized multiplier `M0` in `[0, 2^17 - 1]`.
    pub m0: i64,
    /// Post-multiply arithmetic right shift `S`, in `0..=31`.
    pub shift: i32,
    /// Rounding mode.
    pub rounding: Rounding,
    /// Inclusive lower clamp of the quantizer output.
    pub saturate_lo: i32,
    /// Inclusive upper clamp of the quantizer output.
    pub saturate_hi: i32,
    /// Per-output-channel multipliers, verbatim from the source artifact.
    /// Empty means uniform (all channels use `m0`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_channel_multipliers: Vec<i64>,
    /// Maximum relative error of the fixed-point requant approximation, in
    /// parts per million, when the source artifact declares one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiplier_error_ppm: Option<u32>,
}

/// Highest representable normalized requant multiplier (`2^17 - 1`).
pub const REQUANT_MAX_MULTIPLIER: i64 = (1 << 17) - 1;

impl Requant {
    /// A uniform requant contract.
    pub const fn uniform(m0: i64, shift: i32, rounding: Rounding, lo: i32, hi: i32) -> Self {
        Self {
            m0,
            shift,
            rounding,
            saturate_lo: lo,
            saturate_hi: hi,
            per_channel_multipliers: Vec::new(),
            multiplier_error_ppm: None,
        }
    }

    /// Validate the hardware-operand representation.
    pub fn validate(&self, out_channels: u32) -> Result<(), String> {
        if self.m0 < 0 || self.m0 > REQUANT_MAX_MULTIPLIER {
            return Err(format!(
                "requant m0={} is outside the 17-bit multiplier field [0, {}]",
                self.m0, REQUANT_MAX_MULTIPLIER
            ));
        }
        if !(0..=31).contains(&self.shift) {
            return Err(format!("requant shift={} must be in 0..=31", self.shift));
        }
        if self.saturate_lo > self.saturate_hi {
            return Err(format!(
                "requant saturate range [{}, {}] is empty",
                self.saturate_lo, self.saturate_hi
            ));
        }
        if !self.per_channel_multipliers.is_empty() {
            if self.per_channel_multipliers.len() != out_channels as usize {
                return Err(format!(
                    "per-channel multipliers: expected {} entries (out_channels), found {}",
                    out_channels,
                    self.per_channel_multipliers.len()
                ));
            }
            if let Some(bad) = self
                .per_channel_multipliers
                .iter()
                .find(|m| **m < 0 || **m > REQUANT_MAX_MULTIPLIER)
            {
                return Err(format!(
                    "per-channel multiplier {bad} is outside the 17-bit field [0, {}]",
                    REQUANT_MAX_MULTIPLIER
                ));
            }
        }
        Ok(())
    }
}

/// Activation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    /// Identity.
    None,
    /// ReLU: clamp below at zero.
    Relu,
    /// ReLU6: clamp to `[0, 6]` in the quantized domain.
    Relu6,
    /// Unsigned clamp to `[0, 255]`; only legal on the final output op.
    UnsignedClamp,
}

/// Activation attached to an operator output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Activation {
    /// Activation kind.
    pub kind: ActivationKind,
    /// Inclusive lower clamp.
    pub clamp_lo: i32,
    /// Inclusive upper clamp.
    pub clamp_hi: i32,
}

impl Activation {
    /// Identity activation over the given quantized range.
    pub const fn identity(clamp_lo: i32, clamp_hi: i32) -> Self {
        Self {
            kind: ActivationKind::None,
            clamp_lo,
            clamp_hi,
        }
    }

    /// The effective clamp after intersecting with a requant saturate range.
    pub fn effective_clamp(self, requant: Option<&Requant>) -> (i32, i32) {
        match requant {
            Some(r) => (
                self.clamp_lo.max(r.saturate_lo),
                self.clamp_hi.min(r.saturate_hi),
            ),
            None => (self.clamp_lo, self.clamp_hi),
        }
    }
}

/// Rectangular tile geometry in output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileGeometry {
    /// Tile width in output pixels.
    pub width: u32,
    /// Tile height in output pixels.
    pub height: u32,
}

impl TileGeometry {
    /// Construct a tile geometry.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Tile area in pixels.
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// Convolution engine folding configuration.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    /// Engine name (`shared_conv0`, `stream_dw0`, …).
    pub name: String,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
}

impl EngineConfig {
    /// Total multiplier lanes.
    pub const fn lanes(&self) -> u64 {
        self.simd as u64 * self.pe as u64 * self.tap_parallel as u64
    }
}

/// Objective metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveTerm {
    /// Minimize frame cycles.
    MinimizeCycles,
    /// Minimize total DDR bytes per frame.
    MinimizeDdrBytes,
    /// Minimize RAM10 blocks.
    MinimizeRam10,
    /// Minimize DSP48 blocks.
    MinimizeDsp,
    /// Minimize XLR / logic units.
    MinimizeLogic,
}

impl ObjectiveTerm {
    /// Extract this metric from a resource vector.
    pub fn value(self, estimate: &ResourceEstimate) -> u64 {
        match self {
            ObjectiveTerm::MinimizeCycles => estimate.frame_cycles,
            ObjectiveTerm::MinimizeDdrBytes => {
                estimate.ddr_read_bytes_per_frame + estimate.ddr_write_bytes_per_frame
            }
            ObjectiveTerm::MinimizeRam10 => estimate.ram10,
            ObjectiveTerm::MinimizeDsp => estimate.dsp48,
            ObjectiveTerm::MinimizeLogic => estimate.xlr,
        }
    }
}

/// Lexicographic objective: primary term, then ordered tie-breakers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveSpec {
    /// Primary metric.
    pub primary: ObjectiveTerm,
    /// Tie-breakers, most significant first.
    #[serde(default)]
    pub secondary: Vec<ObjectiveTerm>,
}

impl ObjectiveSpec {
    /// Rank vector for an estimate.
    pub fn rank(&self, estimate: &ResourceEstimate) -> Vec<u64> {
        std::iter::once(self.primary)
            .chain(self.secondary.iter().copied())
            .map(|term| term.value(estimate))
            .collect()
    }

    /// Whether `a` is strictly better than `b`.
    pub fn is_better(&self, a: &ResourceEstimate, b: &ResourceEstimate) -> bool {
        self.rank(a) < self.rank(b)
    }
}

/// Legal producer/consumer fusions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionKind {
    /// Depthwise → requant → pointwise in one tile, keeping the intermediate
    /// activation on chip.
    DepthwiseRequantPointwise,
    /// Short residual add fused into the consuming tile.
    ShortResidual,
}

impl FusionKind {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            FusionKind::DepthwiseRequantPointwise => "depthwise_requant_pointwise",
            FusionKind::ShortResidual => "short_residual",
        }
    }
}

/// Conditions that make a producer/consumer boundary illegal to fuse across.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionBoundary {
    /// The producer value has more than one consumer.
    FanoutValue,
    /// The pair spans a DDR round trip.
    CrossDdrBoundary,
}

impl FusionBoundary {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            FusionBoundary::FanoutValue => "fanout_value",
            FusionBoundary::CrossDdrBoundary => "cross_ddr_boundary",
        }
    }
}

// ---------------------------------------------------------------------------
// Estimates, limits, slack
// ---------------------------------------------------------------------------

/// Resource and cost estimate for a plan, produced by the target model.
///
/// Also used as the embedded limit vector ([`PlanLimits`]) with identical
/// fields, so slack is a straight subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEstimate {
    /// DSP48 blocks.
    pub dsp48: u64,
    /// RAM10 blocks.
    pub ram10: u64,
    /// XLR / logic units.
    pub xlr: u64,
    /// Frame cycles.
    pub frame_cycles: u64,
    /// DDR bytes read per frame.
    pub ddr_read_bytes_per_frame: u64,
    /// DDR bytes written per frame.
    pub ddr_write_bytes_per_frame: u64,
}

impl ResourceEstimate {
    /// All-zero estimate.
    pub const ZERO: Self = Self {
        dsp48: 0,
        ram10: 0,
        xlr: 0,
        frame_cycles: 0,
        ddr_read_bytes_per_frame: 0,
        ddr_write_bytes_per_frame: 0,
    };

    /// Whether every resource fits within `limits`.
    pub fn fits_within(&self, limits: &ResourceEstimate) -> bool {
        self.dsp48 <= limits.dsp48
            && self.ram10 <= limits.ram10
            && self.xlr <= limits.xlr
            && self.frame_cycles <= limits.frame_cycles
            && self.ddr_read_bytes_per_frame <= limits.ddr_read_bytes_per_frame
            && self.ddr_write_bytes_per_frame <= limits.ddr_write_bytes_per_frame
    }

    /// Legible description of the first violated limit, if any.
    pub fn first_violation(&self, limits: &ResourceEstimate) -> Option<(&'static str, u64, u64)> {
        let checks = [
            ("dsp48", self.dsp48, limits.dsp48),
            ("ram10", self.ram10, limits.ram10),
            ("xlr", self.xlr, limits.xlr),
            ("frame_cycles", self.frame_cycles, limits.frame_cycles),
            (
                "ddr_read_bytes_per_frame",
                self.ddr_read_bytes_per_frame,
                limits.ddr_read_bytes_per_frame,
            ),
            (
                "ddr_write_bytes_per_frame",
                self.ddr_write_bytes_per_frame,
                limits.ddr_write_bytes_per_frame,
            ),
        ];
        checks.into_iter().find(|(_, used, limit)| used > limit)
    }
}

/// The resource limits a layout was solved against, embedded in the artifact.
pub type PlanLimits = ResourceEstimate;

/// Signed headroom: `limit - used`, positive when a budget is respected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSlack {
    /// DSP48 headroom.
    pub dsp48: i64,
    /// RAM10 headroom.
    pub ram10: i64,
    /// XLR headroom.
    pub xlr: i64,
    /// Frame-cycle headroom.
    pub frame_cycles: i64,
    /// DDR read headroom.
    pub ddr_read_bytes_per_frame: i64,
    /// DDR write headroom.
    pub ddr_write_bytes_per_frame: i64,
}

impl ResourceSlack {
    /// Slack of `used` against `limits`.
    pub fn between(used: &ResourceEstimate, limits: &PlanLimits) -> Self {
        fn d(limit: u64, used: u64) -> i64 {
            let v = limit as i128 - used as i128;
            v.clamp(i64::MIN as i128, i64::MAX as i128) as i64
        }
        Self {
            dsp48: d(limits.dsp48, used.dsp48),
            ram10: d(limits.ram10, used.ram10),
            xlr: d(limits.xlr, used.xlr),
            frame_cycles: d(limits.frame_cycles, used.frame_cycles),
            ddr_read_bytes_per_frame: d(
                limits.ddr_read_bytes_per_frame,
                used.ddr_read_bytes_per_frame,
            ),
            ddr_write_bytes_per_frame: d(
                limits.ddr_write_bytes_per_frame,
                used.ddr_write_bytes_per_frame,
            ),
        }
    }

    /// Whether every budget is respected.
    pub const fn all_non_negative(&self) -> bool {
        self.dsp48 >= 0
            && self.ram10 >= 0
            && self.xlr >= 0
            && self.frame_cycles >= 0
            && self.ddr_read_bytes_per_frame >= 0
            && self.ddr_write_bytes_per_frame >= 0
    }
}

// ---------------------------------------------------------------------------
// Provenance and artifact contents
// ---------------------------------------------------------------------------

/// Reproducibility metadata for a persisted artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// Content hash of the input graph artifact.
    pub graph_sha256: String,
    /// Content hash of the target description.
    pub target_sha256: String,
    /// Content hash of the policy description.
    pub policy_sha256: String,
    /// Compiler version string.
    pub compiler_version: String,
    /// Target name.
    pub target_name: String,
    /// Target revision.
    pub target_revision: String,
    /// Policy schema version.
    pub policy_schema_version: u32,
    /// Search seed (v1 search is deterministic; the seed is recorded and
    /// recomputed identically).
    pub seed: u64,
    /// Additional solver options that affect the result.
    #[serde(default)]
    pub options: BTreeMap<String, String>,
}

impl Provenance {
    /// Compute a stable combined key for cache invalidation.
    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            self.graph_sha256,
            self.target_sha256,
            self.policy_sha256,
            self.compiler_version,
            self.target_name,
            self.target_revision,
            self.seed
        )
    }
}

/// A concrete plan was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// A legal plan meeting every hard constraint was found.
    Feasible,
    /// No legal plan exists under these target/policy constraints.
    Infeasible,
}

/// One scheduled operator execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleStep {
    /// Monotone step index.
    pub step: u32,
    /// Stable operator/layer name.
    pub op: String,
    /// Operator kind.
    pub op_kind: OpKind,
    /// Engine name.
    pub engine: String,
    /// Schedule granularity.
    pub mode: Granularity,
    /// Tile geometry, when the granularity is tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Input channels.
    pub in_channels: u32,
    /// Output channels.
    pub out_channels: u32,
    /// Input spatial extent `[h, w]`.
    pub in_spatial: [u32; 2],
    /// Output spatial extent `[h, w]`.
    pub out_spatial: [u32; 2],
    /// Kernel extent `[kh, kw]`.
    pub kernel: [u32; 2],
    /// Convolution groups.
    pub groups: u32,
    /// Stride `[sh, sw]`.
    pub stride: [u32; 2],
    /// Requant contract, when the operator requantizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requant: Option<Requant>,
    /// Activation applied to the output.
    pub activation: Activation,
    /// Weight tensor reference (never contents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<TensorRef>,
    /// Bias tensor reference (never contents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<TensorRef>,
    /// Input tensors.
    pub inputs: Vec<TensorId>,
    /// Output tensors.
    pub outputs: Vec<TensorId>,
    /// Input buffer name (resolves into the artifact's `buffers`).
    pub input_buffer: String,
    /// Output buffer name (resolves into the artifact's `buffers`).
    pub output_buffer: String,
    /// Estimated cycles for this step.
    pub estimated_cycles: u64,
    /// Operators fused into this step.
    #[serde(default)]
    pub fused_with: Vec<String>,
}

/// A buffer binding produced by the planner and mapped by the target model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferBinding {
    /// Buffer name.
    pub name: String,
    /// Placement.
    pub placement: MemoryPlacement,
    /// Tensor held, when the buffer backs a graph tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<TensorId>,
    /// Element width in bits.
    pub width_bits: u64,
    /// Number of words.
    pub depth: u64,
    /// Physical RAM10 blocks consumed (`0` for register/FIFO/DDR buffers).
    pub ram10_blocks: u64,
    /// Logical bytes.
    pub bytes: u64,
    /// Read ports required.
    pub read_ports: u64,
    /// Write ports required.
    pub write_ports: u64,
    /// Ping-pong depth (1 = single-buffered).
    pub ping_pong_depth: u32,
    /// Producers of the value held.
    #[serde(default)]
    pub producers: Vec<String>,
    /// Consumers of the value held.
    #[serde(default)]
    pub consumers: Vec<String>,
}

/// Per-operator fold assignment in the selected plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpAssignment {
    /// Operator name.
    pub op: String,
    /// Operator kind.
    pub op_kind: OpKind,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Input buffer name.
    pub input_buffer: String,
    /// Output buffer name.
    pub output_buffer: String,
    /// Operators fused into this one.
    #[serde(default)]
    pub fused_with: Vec<String>,
}

/// The selected execution organisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    /// Execution mode.
    pub execution_mode: ExecutionMode,
    /// Granularity.
    pub granularity: Granularity,
    /// Tile geometry when tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Engine folding configuration.
    pub engine: EngineConfig,
    /// Weight residency.
    pub weight_placement: MemoryPlacement,
    /// Activation/intermediate residency.
    pub intermediate_placement: MemoryPlacement,
    /// Whether legal fusions were applied.
    pub fusion_enabled: bool,
    /// Per-operator assignments, ordered by operator name.
    pub ops: Vec<OpAssignment>,
}

/// A fusion that the selected plan applied.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionDecision {
    /// Producer operator name.
    pub producer: String,
    /// Consumer operator name.
    pub consumer: String,
    /// Fusion kind.
    pub kind: FusionKind,
    /// DDR bytes per frame avoided by keeping the intermediate on chip.
    pub saved_ddr_bytes: u64,
}

/// Why a candidate was rejected.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    /// DSP48 budget exceeded.
    DspBudget { used: u64, budget: u64 },
    /// RAM10 budget exceeded.
    RamBudget { used: u64, budget: u64 },
    /// XLR budget exceeded.
    XlrBudget { used: u64, budget: u64 },
    /// Frame cycle budget exceeded.
    FrameCycles { used: u64, budget: u64 },
    /// DDR read budget exceeded.
    DdrReadBudget { used: u64, budget: u64 },
    /// DDR write budget exceeded.
    DdrWriteBudget { used: u64, budget: u64 },
    /// A fold factor does not divide the operator dimension.
    IllegalFold { op: String, detail: String },
    /// Operator kind is not supported by the v1 planner.
    UnsupportedOperator { op: String, kind: OpKind },
    /// Execution mode not permitted by policy.
    UnsupportedMode { mode: ExecutionMode },
    /// Memory placement not permitted by policy.
    UnsupportedPlacement { placement: MemoryPlacement },
    /// The input graph is malformed.
    InvalidGraph { detail: String },
    /// A policy rule forbids the candidate.
    PolicyViolation { detail: String },
    /// Search exhausted without a feasible candidate.
    NoCandidate { detail: String },
}

impl RejectReason {
    /// Stable category used for counting and reporting.
    pub const fn category(&self) -> &'static str {
        match self {
            RejectReason::DspBudget { .. } => "dsp_budget",
            RejectReason::RamBudget { .. } => "ram_budget",
            RejectReason::XlrBudget { .. } => "xlr_budget",
            RejectReason::FrameCycles { .. } => "frame_cycles",
            RejectReason::DdrReadBudget { .. } => "ddr_read_budget",
            RejectReason::DdrWriteBudget { .. } => "ddr_write_budget",
            RejectReason::IllegalFold { .. } => "illegal_fold",
            RejectReason::UnsupportedOperator { .. } => "unsupported_operator",
            RejectReason::UnsupportedMode { .. } => "unsupported_mode",
            RejectReason::UnsupportedPlacement { .. } => "unsupported_placement",
            RejectReason::InvalidGraph { .. } => "invalid_graph",
            RejectReason::PolicyViolation { .. } => "policy_violation",
            RejectReason::NoCandidate { .. } => "no_candidate",
        }
    }
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::DspBudget { used, budget } => {
                write!(f, "DSP budget exceeded: {used} > {budget}")
            }
            RejectReason::RamBudget { used, budget } => {
                write!(f, "RAM10 budget exceeded: {used} > {budget}")
            }
            RejectReason::XlrBudget { used, budget } => {
                write!(f, "XLR budget exceeded: {used} > {budget}")
            }
            RejectReason::FrameCycles { used, budget } => {
                write!(f, "frame cycles exceeded: {used} > {budget}")
            }
            RejectReason::DdrReadBudget { used, budget } => {
                write!(f, "DDR read budget exceeded: {used} > {budget}")
            }
            RejectReason::DdrWriteBudget { used, budget } => {
                write!(f, "DDR write budget exceeded: {used} > {budget}")
            }
            RejectReason::IllegalFold { op, detail } => write!(f, "illegal fold on {op}: {detail}"),
            RejectReason::UnsupportedOperator { op, kind } => {
                write!(f, "unsupported operator {op} ({kind:?})")
            }
            RejectReason::UnsupportedMode { mode } => {
                write!(f, "execution mode {mode:?} not permitted")
            }
            RejectReason::UnsupportedPlacement { placement } => {
                write!(f, "placement {placement:?} not permitted")
            }
            RejectReason::InvalidGraph { detail } => write!(f, "invalid graph: {detail}"),
            RejectReason::PolicyViolation { detail } => write!(f, "policy violation: {detail}"),
            RejectReason::NoCandidate { detail } => write!(f, "no feasible candidate: {detail}"),
        }
    }
}

/// One rejected candidate, recorded for inspection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedCandidate {
    /// Candidate execution mode.
    pub execution_mode: ExecutionMode,
    /// Candidate granularity.
    pub granularity: Granularity,
    /// Candidate tile geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Candidate weight placement.
    pub weight_placement: MemoryPlacement,
    /// Candidate intermediate placement.
    pub intermediate_placement: MemoryPlacement,
    /// Whether fusions were enabled.
    pub fusion_enabled: bool,
    /// Why it was rejected.
    pub reason: RejectReason,
}

/// Bounded summary of the rejected search space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectionSummary {
    /// Number of rejected candidate configurations.
    pub rejected_candidates: u64,
    /// Count per [`RejectReason::category`].
    #[serde(default)]
    pub counts: BTreeMap<String, u64>,
    /// First few rejected candidates, in deterministic order.
    #[serde(default)]
    pub samples: Vec<RejectedCandidate>,
}

/// Search statistics for reproducibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchStats {
    /// Candidate configurations explored.
    pub explored: u64,
    /// Node expansions performed by the branch-and-bound inner search.
    pub expanded_nodes: u64,
    /// Candidate cap from policy.
    pub max_candidates: u64,
    /// Whether the cap stopped the search early.
    pub truncated: bool,
}

/// The persisted layout artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutArtifact {
    /// Must equal [`LAYOUT_FORMAT`].
    pub format: String,
    /// Must equal [`LAYOUT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Feasibility status.
    pub status: PlanStatus,
    /// Reproducibility metadata.
    pub provenance: Provenance,
    /// Resource limits the search was performed against.
    pub limits: PlanLimits,
    /// Objective used to rank candidates.
    pub objective: ObjectiveSpec,
    /// Selected organisation, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    /// Resource estimate, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceEstimate>,
    /// Slack, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slack: Option<ResourceSlack>,
    /// Tensor table (activations, weights, graph I/O).
    #[serde(default)]
    pub tensors: Vec<TensorRef>,
    /// Schedule steps, ordered by `step`.
    #[serde(default)]
    pub schedule: Vec<ScheduleStep>,
    /// Buffers, ordered deterministically.
    #[serde(default)]
    pub buffers: Vec<BufferBinding>,
    /// Applied fusions, ordered deterministically.
    #[serde(default)]
    pub fusions: Vec<FusionDecision>,
    /// Rejected-candidate summary.
    pub rejections: RejectionSummary,
    /// Search statistics.
    pub search: SearchStats,
}

/// Errors produced while loading, validating or projecting a layout artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// TOML syntax or schema binding failure.
    Config(TomlError),
    /// Wrong `format` tag.
    WrongFormat { expected: String, found: String },
    /// Unsupported `schema_version`.
    UnsupportedSchema { found: u32, supported: u32 },
    /// Structural inconsistency.
    Invalid { detail: String },
    /// A resource exceeds the embedded limit.
    OverBudget {
        resource: &'static str,
        used: u64,
        budget: u64,
    },
    /// The artifact records an infeasible plan and cannot be projected.
    Infeasible { detail: String },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::Config(e) => write!(f, "{e}"),
            LayoutError::WrongFormat { expected, found } => {
                write!(
                    f,
                    "wrong format tag: expected {expected:?}, found {found:?}"
                )
            }
            LayoutError::UnsupportedSchema { found, supported } => write!(
                f,
                "unsupported schema_version {found} (this build supports {supported})"
            ),
            LayoutError::Invalid { detail } => write!(f, "invalid layout artifact: {detail}"),
            LayoutError::OverBudget {
                resource,
                used,
                budget,
            } => write!(f, "over budget: {resource} used {used} > limit {budget}"),
            LayoutError::Infeasible { detail } => {
                write!(f, "layout artifact is infeasible: {detail}")
            }
        }
    }
}

impl std::error::Error for LayoutError {}

impl From<TomlError> for LayoutError {
    fn from(e: TomlError) -> Self {
        LayoutError::Config(e)
    }
}

impl LayoutArtifact {
    /// Parse an artifact from TOML text, normalize it, and validate it.
    pub fn from_toml_str(src: &str) -> Result<Self, LayoutError> {
        let mut artifact: LayoutArtifact = toml_lite::from_toml_str(src)?;
        artifact.normalize();
        artifact.validate()?;
        Ok(artifact)
    }

    /// Read and validate an artifact from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, LayoutError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot read {}: {e}", path.as_ref().display()),
        })?;
        Self::from_toml_str(&text)
    }

    /// Deterministic TOML rendering (normalized, then serialized).
    pub fn to_toml_string(&self) -> Result<String, LayoutError> {
        let mut normalized = self.clone();
        normalized.normalize();
        Ok(toml_lite::to_toml_string_of(&normalized)?)
    }

    /// Write the artifact to disk as TOML.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), LayoutError> {
        let text = self.to_toml_string()?;
        std::fs::write(path.as_ref(), text).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot write {}: {e}", path.as_ref().display()),
        })
    }

    /// JSON rendering, for tooling that prefers JSON.
    pub fn to_json_string(&self) -> Result<String, LayoutError> {
        let mut normalized = self.clone();
        normalized.normalize();
        serde_json::to_string_pretty(&normalized).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot encode JSON: {e}"),
        })
    }

    /// Parse an artifact from JSON text, normalize it, and validate it.
    pub fn from_json_str(src: &str) -> Result<Self, LayoutError> {
        let mut artifact: LayoutArtifact =
            serde_json::from_str(src).map_err(|e| LayoutError::Invalid {
                detail: format!("cannot decode JSON: {e}"),
            })?;
        artifact.normalize();
        artifact.validate()?;
        Ok(artifact)
    }

    /// Put every collection in deterministic order.
    pub fn normalize(&mut self) {
        self.tensors.sort_by_key(|t| t.id);
        self.schedule.sort_by_key(|s| s.step);
        self.buffers
            .sort_by(|a, b| (a.placement, &a.name).cmp(&(b.placement, &b.name)));
        self.fusions.sort();
        if let Some(selection) = &mut self.selection {
            selection.ops.sort_by(|a, b| a.op.cmp(&b.op));
            for step in &mut self.schedule {
                step.fused_with.sort();
            }
        }
        self.rejections.samples.sort();
    }

    /// Validate the artifact's structure and, when feasible, its budgets.
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.format != LAYOUT_FORMAT {
            return Err(LayoutError::WrongFormat {
                expected: LAYOUT_FORMAT.into(),
                found: self.format.clone(),
            });
        }
        if self.schema_version != LAYOUT_SCHEMA_VERSION {
            return Err(LayoutError::UnsupportedSchema {
                found: self.schema_version,
                supported: LAYOUT_SCHEMA_VERSION,
            });
        }

        // Schedule ordering and buffer resolution.
        let mut last: Option<u32> = None;
        for step in &self.schedule {
            if let Some(prev) = last {
                if step.step <= prev {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "schedule steps are not strictly increasing: {} after {}",
                            step.step, prev
                        ),
                    });
                }
            }
            last = Some(step.step);
            if let Some(requant) = &step.requant {
                requant
                    .validate(step.out_channels)
                    .map_err(|detail| LayoutError::Invalid {
                        detail: format!("step {} ({}): {detail}", step.step, step.op),
                    })?;
            }
            for buffer in [&step.input_buffer, &step.output_buffer] {
                if !buffer.is_empty() && !self.buffers.iter().any(|b| &b.name == buffer) {
                    return Err(LayoutError::Invalid {
                        detail: format!("step {} references unknown buffer {buffer:?}", step.step),
                    });
                }
            }
        }

        // Tensor-table consistency for weight/bias references.
        for step in &self.schedule {
            for reference in [&step.weights, &step.bias].into_iter().flatten() {
                if !self.tensors.iter().any(|t| t.id == reference.id) {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "step {} references tensor {} absent from the tensor table",
                            step.step, reference.id
                        ),
                    });
                }
            }
        }

        match self.status {
            PlanStatus::Feasible => {
                let selection = self
                    .selection
                    .as_ref()
                    .ok_or_else(|| LayoutError::Invalid {
                        detail: "feasible artifact has no selection".into(),
                    })?;
                let resources = self
                    .resources
                    .as_ref()
                    .ok_or_else(|| LayoutError::Invalid {
                        detail: "feasible artifact has no resource estimate".into(),
                    })?;
                let slack = self.slack.as_ref().ok_or_else(|| LayoutError::Invalid {
                    detail: "feasible artifact has no slack report".into(),
                })?;
                if self.schedule.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact has an empty schedule".into(),
                    });
                }
                if selection.ops.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact has no operator assignments".into(),
                    });
                }
                if let Some((resource, used, budget)) = resources.first_violation(&self.limits) {
                    return Err(LayoutError::OverBudget {
                        resource,
                        used,
                        budget,
                    });
                }
                let recomputed = ResourceSlack::between(resources, &self.limits);
                if &recomputed != slack || !slack.all_non_negative() {
                    return Err(LayoutError::Invalid {
                        detail: "slack report is inconsistent with the resource estimate".into(),
                    });
                }
            }
            PlanStatus::Infeasible => {
                if self.selection.is_some() || self.resources.is_some() || self.slack.is_some() {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must not carry a selection".into(),
                    });
                }
                if !self.schedule.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must not carry a schedule".into(),
                    });
                }
                if self.rejections.rejected_candidates == 0 {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must explain why no candidate was legal"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Re-check every hard constraint against the embedded limits.
    pub fn verify(&self) -> Result<(), LayoutError> {
        self.validate()?;
        if self.status != PlanStatus::Feasible {
            return Err(LayoutError::Infeasible {
                detail: "artifact records an infeasible plan".into(),
            });
        }
        Ok(())
    }

    /// Project a validated artifact onto the backend-ready [`ExecutionPlan`].
    pub fn execution_plan(&self) -> Result<ExecutionPlan, LayoutError> {
        self.verify()?;
        let selection = self
            .selection
            .as_ref()
            .ok_or_else(|| LayoutError::Invalid {
                detail: "feasible artifact has no selection".into(),
            })?;
        let resources = self.resources.ok_or_else(|| LayoutError::Invalid {
            detail: "feasible artifact has no resource estimate".into(),
        })?;

        // Deterministic weight/bias tensor table: unique references by id.
        let mut weights: Vec<TensorRef> = Vec::new();
        for step in &self.schedule {
            for reference in [&step.weights, &step.bias].into_iter().flatten() {
                if !weights.iter().any(|w| w.id == reference.id) {
                    weights.push(reference.clone());
                }
            }
        }
        weights.sort_by_key(|w| w.id);

        Ok(ExecutionPlan {
            format: PLAN_FORMAT.into(),
            schema_version: PLAN_SCHEMA_VERSION,
            provenance: self.provenance.clone(),
            execution_mode: selection.execution_mode,
            engine: selection.engine.clone(),
            tile: selection.tile,
            steps: self.schedule.clone(),
            buffers: self.buffers.clone(),
            tensors: self.tensors.clone(),
            weights,
            resources,
        })
    }
}

/// Backend-ready projection of a validated layout artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    /// Must equal [`PLAN_FORMAT`].
    pub format: String,
    /// Must equal [`PLAN_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Provenance inherited from the layout.
    pub provenance: Provenance,
    /// Execution mode.
    pub execution_mode: ExecutionMode,
    /// Engine configuration.
    pub engine: EngineConfig,
    /// Tile geometry, when tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Schedule steps, ordered by `step`.
    pub steps: Vec<ScheduleStep>,
    /// Buffers.
    pub buffers: Vec<BufferBinding>,
    /// Tensor table.
    pub tensors: Vec<TensorRef>,
    /// Weight and bias tensors, ordered by id.
    pub weights: Vec<TensorRef>,
    /// Resource estimate.
    pub resources: ResourceEstimate,
}

impl ExecutionPlan {
    /// Deterministic serialization and re-parse, primarily for RTL metadata.
    pub fn to_toml_string(&self) -> Result<String, LayoutError> {
        Ok(toml_lite::to_toml_string_of(self)?)
    }

    /// Parse an execution plan from TOML text.
    pub fn from_toml_str(src: &str) -> Result<Self, LayoutError> {
        let plan: ExecutionPlan = toml_lite::from_toml_str(src)?;
        if plan.format != PLAN_FORMAT {
            return Err(LayoutError::WrongFormat {
                expected: PLAN_FORMAT.into(),
                found: plan.format.clone(),
            });
        }
        if plan.schema_version != PLAN_SCHEMA_VERSION {
            return Err(LayoutError::UnsupportedSchema {
                found: plan.schema_version,
                supported: PLAN_SCHEMA_VERSION,
            });
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> PlanLimits {
        PlanLimits {
            dsp48: 152,
            ram10: 88,
            xlr: 40_568,
            frame_cycles: 10_000_000,
            ddr_read_bytes_per_frame: 90_000_000,
            ddr_write_bytes_per_frame: 70_000_000,
        }
    }

    fn feasible_artifact() -> LayoutArtifact {
        let tensors = vec![
            TensorRef {
                id: TensorId(0),
                name: "image".into(),
                shape: TensorShape::new(1, 3, 480, 640),
                elem: ElemType::I8,
            },
            TensorRef {
                id: TensorId(1),
                name: "enc.0.w".into(),
                shape: TensorShape::new(32, 3, 3, 3),
                elem: ElemType::I8,
            },
            TensorRef {
                id: TensorId(2),
                name: "enc.0.b".into(),
                shape: TensorShape::new(32, 1, 1, 1),
                elem: ElemType::I32,
            },
        ];
        let resources = ResourceEstimate {
            dsp48: 96,
            ram10: 80,
            xlr: 12_000,
            frame_cycles: 9_866_272,
            ddr_read_bytes_per_frame: 921_600,
            ddr_write_bytes_per_frame: 921_600,
        };
        LayoutArtifact {
            format: LAYOUT_FORMAT.into(),
            schema_version: LAYOUT_SCHEMA_VERSION,
            status: PlanStatus::Feasible,
            provenance: Provenance {
                graph_sha256: "g".into(),
                target_sha256: "t".into(),
                policy_sha256: "p".into(),
                compiler_version: "0.1.0".into(),
                target_name: "ti60f225".into(),
                target_revision: "r1".into(),
                policy_schema_version: 1,
                seed: 1,
                options: BTreeMap::new(),
            },
            limits: limits(),
            objective: ObjectiveSpec {
                primary: ObjectiveTerm::MinimizeCycles,
                secondary: vec![
                    ObjectiveTerm::MinimizeDdrBytes,
                    ObjectiveTerm::MinimizeRam10,
                ],
            },
            selection: Some(Selection {
                execution_mode: ExecutionMode::SharedTile,
                granularity: Granularity::Tile,
                tile: Some(TileGeometry::new(80, 60)),
                engine: EngineConfig {
                    name: "shared_conv0".into(),
                    simd: 8,
                    pe: 8,
                    tap_parallel: 1,
                },
                weight_placement: MemoryPlacement::Bram,
                intermediate_placement: MemoryPlacement::Ddr,
                fusion_enabled: true,
                ops: vec![OpAssignment {
                    op: "enc.0".into(),
                    op_kind: OpKind::Conv2d,
                    simd: 8,
                    pe: 8,
                    tap_parallel: 1,
                    input_buffer: "image_buf".into(),
                    output_buffer: "enc0_out".into(),
                    fused_with: vec![],
                }],
            }),
            resources: Some(resources),
            slack: Some(ResourceSlack::between(&resources, &limits())),
            tensors,
            schedule: vec![ScheduleStep {
                step: 0,
                op: "enc.0".into(),
                op_kind: OpKind::Conv2d,
                engine: "shared_conv0".into(),
                mode: Granularity::Tile,
                tile: Some(TileGeometry::new(80, 60)),
                simd: 8,
                pe: 8,
                tap_parallel: 1,
                in_channels: 3,
                out_channels: 32,
                in_spatial: [480, 640],
                out_spatial: [480, 640],
                kernel: [3, 3],
                groups: 1,
                stride: [1, 1],
                requant: Some(Requant {
                    m0: 98_304,
                    shift: 16,
                    rounding: Rounding::HalfUp,
                    saturate_lo: -128,
                    saturate_hi: 127,
                    per_channel_multipliers: Vec::new(),
                    multiplier_error_ppm: None,
                }),
                activation: Activation {
                    kind: ActivationKind::Relu,
                    clamp_lo: 0,
                    clamp_hi: 127,
                },
                weights: Some(TensorRef {
                    id: TensorId(1),
                    name: "enc.0.w".into(),
                    shape: TensorShape::new(32, 3, 3, 3),
                    elem: ElemType::I8,
                }),
                bias: Some(TensorRef {
                    id: TensorId(2),
                    name: "enc.0.b".into(),
                    shape: TensorShape::new(32, 1, 1, 1),
                    elem: ElemType::I32,
                }),
                inputs: vec![TensorId(0)],
                outputs: vec![],
                input_buffer: "image_buf".into(),
                output_buffer: "enc0_out".into(),
                estimated_cycles: 9_866_272,
                fused_with: vec![],
            }],
            buffers: vec![
                BufferBinding {
                    name: "image_buf".into(),
                    placement: MemoryPlacement::Ddr,
                    tensor: Some(TensorId(0)),
                    width_bits: 8,
                    depth: 921_600,
                    ram10_blocks: 0,
                    bytes: 921_600,
                    read_ports: 1,
                    write_ports: 1,
                    ping_pong_depth: 1,
                    producers: vec![],
                    consumers: vec!["enc.0".into()],
                },
                BufferBinding {
                    name: "enc0_out".into(),
                    placement: MemoryPlacement::Ddr,
                    tensor: None,
                    width_bits: 8,
                    depth: 921_600,
                    ram10_blocks: 0,
                    bytes: 921_600,
                    read_ports: 1,
                    write_ports: 1,
                    ping_pong_depth: 1,
                    producers: vec!["enc.0".into()],
                    consumers: vec![],
                },
            ],
            fusions: vec![],
            rejections: RejectionSummary {
                rejected_candidates: 12,
                counts: BTreeMap::from([("frame_cycles".to_string(), 12)]),
                samples: vec![],
            },
            search: SearchStats {
                explored: 13,
                expanded_nodes: 240,
                max_candidates: 100_000,
                truncated: false,
            },
        }
    }

    #[test]
    fn artifact_round_trips_through_toml() {
        let mut a = feasible_artifact();
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn artifact_round_trips_through_json() {
        let mut a = feasible_artifact();
        a.normalize();
        let text = a.to_json_string().unwrap();
        let b = LayoutArtifact::from_json_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn serialization_is_deterministic_across_input_order() {
        let a = feasible_artifact();
        let mut b = feasible_artifact();
        b.buffers.reverse();
        b.schedule.reverse();
        b.tensors.reverse();
        assert_eq!(
            a.to_toml_string().unwrap(),
            b.to_toml_string().unwrap(),
            "normalization must remove input ordering differences"
        );
    }

    #[test]
    fn over_budget_estimate_is_rejected() {
        let mut a = feasible_artifact();
        let resources = a.resources.as_mut().unwrap();
        resources.dsp48 = 999;
        let err = a.validate().unwrap_err();
        assert!(matches!(
            err,
            LayoutError::OverBudget {
                resource: "dsp48",
                ..
            }
        ));

        // Slack must also match the (fixed) resources.
        let mut b = feasible_artifact();
        b.resources.as_mut().unwrap().frame_cycles = 11_000_000;
        assert!(b.validate().is_err());
    }

    #[test]
    fn infeasible_artifact_must_explain_itself() {
        let mut a = feasible_artifact();
        a.status = PlanStatus::Infeasible;
        a.selection = None;
        a.resources = None;
        a.slack = None;
        a.schedule.clear();
        a.rejections.rejected_candidates = 0;
        assert!(a.validate().is_err());

        a.rejections.rejected_candidates = 4;
        a.rejections.counts.insert("dsp_budget".to_string(), 4);
        a.rejections.samples.push(RejectedCandidate {
            execution_mode: ExecutionMode::Streaming,
            granularity: Granularity::Row,
            tile: None,
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Fifo,
            fusion_enabled: false,
            reason: RejectReason::DspBudget {
                used: 304,
                budget: 152,
            },
        });
        a.validate().unwrap();
        assert!(matches!(
            a.execution_plan(),
            Err(LayoutError::Infeasible { .. })
        ));
    }

    #[test]
    fn per_channel_multipliers_round_trip_and_validate() {
        let mut a = feasible_artifact();
        a.schedule[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = (0..32).map(|i| 65_536 + i).collect();
        a.schedule[0].requant.as_mut().unwrap().multiplier_error_ppm = Some(37);
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            b.schedule[0]
                .requant
                .as_ref()
                .unwrap()
                .per_channel_multipliers
                .len(),
            32
        );

        // Wrong length is rejected at the artifact boundary.
        let mut c = feasible_artifact();
        c.schedule[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = vec![1, 2, 3];
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejection_reason_samples_round_trip_through_toml() {
        let mut a = feasible_artifact();
        a.status = PlanStatus::Infeasible;
        a.selection = None;
        a.resources = None;
        a.slack = None;
        a.schedule.clear();
        a.rejections.rejected_candidates = 1;
        a.rejections.counts.insert("ram_budget".into(), 1);
        a.rejections.samples.push(RejectedCandidate {
            execution_mode: ExecutionMode::Hybrid,
            granularity: Granularity::Tile,
            tile: Some(TileGeometry::new(80, 60)),
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Bram,
            fusion_enabled: true,
            reason: RejectReason::RamBudget {
                used: 120,
                budget: 88,
            },
        });
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn execution_plan_projects_weights_and_steps() {
        let a = feasible_artifact();
        let plan = a.execution_plan().unwrap();
        assert_eq!(plan.format, PLAN_FORMAT);
        assert_eq!(plan.execution_mode, ExecutionMode::SharedTile);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.weights.len(), 2);
        assert_eq!(plan.weights[0].id, TensorId(1));
        assert_eq!(plan.weights[1].id, TensorId(2));

        let text = plan.to_toml_string().unwrap();
        let reparsed = ExecutionPlan::from_toml_str(&text).unwrap();
        assert_eq!(plan, reparsed);
    }

    #[test]
    fn wrong_format_and_schema_are_rejected_on_load() {
        let a = feasible_artifact();
        let text = a
            .to_toml_string()
            .unwrap()
            .replace("format = \"tessera.layout\"", "format = \"tessera.nope\"");
        assert!(matches!(
            LayoutArtifact::from_toml_str(&text),
            Err(LayoutError::WrongFormat { .. })
        ));

        let mut b = feasible_artifact();
        b.schema_version = 9;
        assert!(matches!(
            b.validate(),
            Err(LayoutError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn schedule_referencing_unknown_buffer_is_rejected() {
        let mut a = feasible_artifact();
        a.schedule[0].output_buffer = "missing".into();
        assert!(matches!(a.validate(), Err(LayoutError::Invalid { .. })));
    }

    #[test]
    fn slack_arithmetic_handles_deficit() {
        let used = ResourceEstimate {
            dsp48: 160,
            ..ResourceEstimate::ZERO
        };
        let slack = ResourceSlack::between(&used, &limits());
        assert_eq!(slack.dsp48, -8);
        assert!(!slack.all_non_negative());
    }

    #[test]
    fn objective_ordering_is_lexicographic() {
        let objective = ObjectiveSpec {
            primary: ObjectiveTerm::MinimizeCycles,
            secondary: vec![ObjectiveTerm::MinimizeDdrBytes],
        };
        let a = ResourceEstimate {
            frame_cycles: 100,
            ddr_read_bytes_per_frame: 50,
            ddr_write_bytes_per_frame: 0,
            ..ResourceEstimate::ZERO
        };
        let b = ResourceEstimate {
            frame_cycles: 100,
            ddr_read_bytes_per_frame: 10,
            ddr_write_bytes_per_frame: 0,
            ..ResourceEstimate::ZERO
        };
        assert!(objective.is_better(&b, &a));
        assert!(!objective.is_better(&a, &b));
    }
}
