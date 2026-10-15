//! Target-aware candidate planner for the Ti60 family.
//!
//! # What this module does
//!
//! It consumes a typed, lowered primitive graph ([`PlanInput`]), a
//! [`Ti60Target`], and a user [`Policy`], and produces a [`LayoutArtifact`]:
//! the chosen execution mode, engine folding, buffer placement, fusion
//! decisions, a static schedule with per-step MAC and cycle derivations,
//! resource and cycle estimates, slack, and a summary of rejected candidates.
//!
//! # Search structure
//!
//! ```text
//! enumerate every outer candidate (mode × granularity × tile × weight
//!                                 placement × intermediate placement × fusion)
//! give each candidate an equal share of the node-expansion budget
//! for each candidate:
//!     reject it if no v1 backend realizes its architecture
//!     compute DDR traffic; reject the whole candidate if it exceeds budget
//!     build per-operator fold options (legality + Pareto reduction)
//!     branch-and-bound depth-first over operators:
//!         prune when a monotone *lower bound* of the final resource vector
//!         already exceeds DSP / RAM / cycle budgets, or when the partial
//!         cycle count already meets the incumbent (primary = cycles)
//!     at the leaf, recompute the exact pooled memory figure from the pooled
//!     ROM word counts and the schedule's peak live activation set, and
//!     re-check every budget before accepting the plan
//! ```
//!
//! Sharing the budget fairly is what keeps truncation meaningful: a single
//! greedy first candidate used to be able to consume the whole cap, leaving
//! entire execution modes unexplored. Now every enumerated candidate is
//! searched, and `search.truncated` says whether any of them was cut.
//!
//! # Physical memory model
//!
//! The RAM figure a plan is accepted on is the mapping of the memories the
//! generated structure actually instantiates, and the engine's storage is
//! **pooled**, so the figure is never a sum of per-layer roundings:
//!
//! ```text
//! w_rom   8-bit weight codes, every layer's slice in one ROM, one read port
//! b_rom   accumulator-width bias words, one slot per output channel of every
//!         step (an elementwise step reserves zeroes so the base index and the
//!         fill order cannot drift apart)
//! m0_rom  17-bit requantization multipliers, one per output channel of every
//!         step that requantizes
//! act     16-bit activation words, sized from the lifetime allocation of the
//!         schedule's live ranges (one read port and one write port)
//! ```
//!
//! Each pool is mapped once, with the target's legal RAM organizations, and
//! each layer's slice is recorded as a code offset inside its pool rather than
//! as a memory of its own. The activation arena is a *lifetime* problem, not a
//! sum: the step-major sequential schedule only keeps the tensors that are
//! still live when it runs, and the plan charges the high water mark of the
//! shared allocator
//! ([`crate::layout::allocate_arena`]) together with its offsets, reported as
//! [`EngineConfig::arena_words`](crate::layout::EngineConfig::arena_words) so a
//! backend can check that what it materializes fits.
//!
//! The engine's MAC and requantization multipliers are charged the same way:
//! one lane (the shared engine is single-lane, so the fold search is clamped to
//! it) and one requantization multiplier whose 32-bit accumulator against a
//! 17-bit `M0` costs `ceil(32 / 18)` DSP48, because the DSP48 multiplier takes
//! an 18-bit operand and cascades the rest.
//!
//! # Realized architectures (v1)
//!
//! The planner enumerates every mode the policy permits and rejects the ones no
//! backend implements, with a reason that names the mode:
//!
//! * [`ExecutionMode::SharedTile`] — one shared, single-lane, RAM-backed engine
//!   running a frame-sequential static FSM over every step. Weights and
//!   activations are RAM-resident, so the intermediate placement must be
//!   on-chip and the granularity must be [`Granularity::Frame`]. Its
//!   activation memory stores 16-bit words, because a residual `add` is
//!   unclamped and needs nine bits.
//! * [`ExecutionMode::Streaming`] / [`ExecutionMode::Dedicated`] — the legacy
//!   single-step pointwise-dot kernel: one plain 1x1, stride-1, bias-free
//!   convolution with `simd × pe` lanes.
//! * [`ExecutionMode::Hybrid`] — not implemented; rejected explicitly.
//!
//! # Cycle model
//!
//! Cycles follow the *generated structure*, not an abstract lane count. A step
//! records its true MAC count and the lane-array sweeps those MACs need
//! (`macs <= lane_iterations × lanes` is enforced at the artifact boundary),
//! and `estimated_cycles = ceil(lane_iterations × 100 / utilisation) + setup`.
//! The backend is expected to reproduce the same derivation; anything it runs
//! beyond it is reported as measured, never as the plan's estimate.
//!
//! # Documented limits of v1
//!
//! * Enumeration + branch-and-bound only — no CP-SAT, no Pareto-frontier dump.
//! * One global execution mode and granularity per plan; per-operator folds may
//!   differ, and the shared engine is sized to the widest assignment.
//! * Streaming overlaps DMA with compute; every other organisation pays the
//!   transfer serially.
//! * XLR is only modelled when the target declares `[logic]` coefficients.
//! * The activation arena is allocated by tensor lifetime, which assumes the
//!   step-major sequential schedule the shared engine implements. The plan
//!   carries the allocator's offsets and its high water mark; a backend must
//!   call the same allocator, because a placement that assumes the peak live
//!   word count is attainable can put a range through a surviving allocation.
//! * The three read-only memories are mapped as RAM10 blocks. Whether the
//!   synthesizer builds a small pool from LUTs instead is a synthesis decision
//!   the target does not declare, so the plan reports the RAM mapping — the
//!   conservative direction — rather than inventing a threshold.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::layout::{
    Activation, BufferBinding, BufferRegion, CodeRange, CycleBreakdown, ElemType, EngineConfig,
    ExecutionMode, FusionBoundary, FusionDecision, FusionKind, Granularity, LAYOUT_FORMAT,
    LAYOUT_SCHEMA_VERSION, LayoutArtifact, MemoryPlacement, ModeSummary, ObjectiveSpec,
    ObjectiveTerm, OpAssignment, OpKind, PaddingSpec, PlanCertainty, PlanLimits, PlanStatus,
    Provenance, RejectReason, RejectedCandidate, RejectionSummary, Requant, ResourceEstimate,
    ResourceSlack, ScheduleStep, SearchStats, Selection, StepCost, TensorId, TensorRef,
    TensorShape, TileGeometry,
};
use crate::target::{BufferRequest, FrameCycles, MacLanes, TargetError, Ti60Target};

/// Policy file format tag.
pub const POLICY_FORMAT: &str = "tessera.policy";
/// Policy file schema version implemented by this build.
pub const POLICY_SCHEMA_VERSION: u32 = 1;

/// Fixed per-tile cycle overhead charged for tile setup/teardown.
const TILE_SETUP_CYCLES: u64 = 64;
/// Maximum number of rejected candidates retained in the artifact.
const MAX_REJECTION_SAMPLES: usize = 16;

// ---------------------------------------------------------------------------
// v1 engine architecture
// ---------------------------------------------------------------------------

/// MAC lanes the RAM-backed shared engine physically instantiates.
///
/// The v1 full-graph emitter is a single-lane, memory-backed sequential engine:
/// one activation read, one weight read and one accumulation per cycle. A plan
/// that folded more lanes would claim parallelism the generated structure does
/// not have, so the shared mode's fold search is clamped to this width and its
/// per-step `lane_iterations` equals its MAC count.
const SHARED_ENGINE_LANES: u64 = 1;
/// Largest multiplier arena the shared emitter will build.
///
/// A capability bound, not a target resource: a plan that needs more
/// multipliers than this is rejected as unrealizable instead of being handed to
/// a backend that cannot build it.
const SHARED_ENGINE_LANE_CAP: u64 = 64;
/// Activation memory word width of the shared engine, in bits.
///
/// The residual `add` is unclamped, so a step's activation can be a nine-bit
/// signed sum; the engine's activation memory therefore stores every activation
/// in a signed 16-bit word.
const SHARED_ENGINE_ACTIVATION_BITS: u32 = 16;
/// Accumulator width of the shared engine, in bits.
const SHARED_ENGINE_ACCUMULATOR_BITS: u32 = 32;
/// Stored weight word width of the shared engine, in bits.
const SHARED_ENGINE_WEIGHT_BITS: u32 = 8;
/// Signed width of one stored requantization multiplier, in bits.
///
/// The multiplier is a non-negative fixed-point value bounded by
/// [`crate::layout::REQUANT_MAX_MULTIPLIER`] (`2^17 − 1`), so the engine's
/// `m0_rom` entry is 17 bits wide and the pooled multiplier ROM is mapped at
/// that width.
const REQUANT_MULTIPLIER_BITS: u64 = 17;
/// Activation word width of the legacy streaming pointwise-dot envelope.
const LEGACY_ENGINE_ACTIVATION_BITS: u32 = 8;

/// Datapath the emitter realizes for a mode: `(lane_limit, activation_bits,
/// accumulator_bits, weight_bits)`.
fn engine_datapath(mode: ExecutionMode) -> (u64, u32, u32, u32) {
    match mode {
        ExecutionMode::SharedTile => (
            SHARED_ENGINE_LANES,
            SHARED_ENGINE_ACTIVATION_BITS,
            SHARED_ENGINE_ACCUMULATOR_BITS,
            SHARED_ENGINE_WEIGHT_BITS,
        ),
        // The legacy envelope is one pointwise-dot engine per step; its lanes
        // are bounded by the DSP budget rather than by a declared limit.
        ExecutionMode::Streaming | ExecutionMode::Dedicated => (
            u64::MAX,
            LEGACY_ENGINE_ACTIVATION_BITS,
            SHARED_ENGINE_ACCUMULATOR_BITS,
            SHARED_ENGINE_WEIGHT_BITS,
        ),
        // Hybrid mixes a shared convolution engine with streaming elementwise
        // paths; the streaming half has no v1 emitter, so it is rejected before
        // its datapath is ever used.
        ExecutionMode::Hybrid => (
            u64::MAX,
            SHARED_ENGINE_ACTIVATION_BITS,
            SHARED_ENGINE_ACCUMULATOR_BITS,
            SHARED_ENGINE_WEIGHT_BITS,
        ),
    }
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// Hard constraints declared by the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConstraints {
    /// Maximum frame cycles.
    pub max_frame_cycles: u64,
    /// Require a bit-exact *realization of the compiled fixed-point contract*:
    /// `(acc * M0 + round) >> shift` with half-away-from-zero rounding, exactly as the
    /// emitted datapath computes it.
    ///
    /// This is deliberately not a claim about the training `f32` requantization
    /// factors. Those are approximated by the serialized 17-bit `M0` and the
    /// shift, and each step records the worst-case relative error it incurred
    /// (`Requant::multiplier_error_ppm`). `false` is accepted and grants nothing,
    /// because v1 has no approximate realization to fall back to.
    #[serde(default)]
    pub require_bit_exact: bool,
    /// Permit CPU layer dispatch. v1 never uses it.
    #[serde(default)]
    pub allow_cpu_layer_dispatch: bool,
}

/// Search space description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchPolicy {
    /// Execution modes to consider.
    pub execution_modes: Vec<ExecutionMode>,
    /// Granularities to consider.
    pub granularities: Vec<Granularity>,
    /// Candidate tile widths.
    pub tile_widths: Vec<u32>,
    /// Candidate tile heights / row stripe heights.
    pub tile_heights: Vec<u32>,
    /// Cap on branch-and-bound node expansions.
    pub max_candidates: u64,
    /// Search seed, recorded for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// MAC-array utilisation in percent applied to compute cycles.
    #[serde(default = "default_utilisation")]
    pub utilisation_percent: u32,
}

fn default_seed() -> u64 {
    1
}

fn default_utilisation() -> u32 {
    100
}

/// Fusion permissions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionPolicy {
    /// Fusions the planner may apply.
    #[serde(default)]
    pub allow: Vec<FusionKind>,
    /// Boundaries the planner must not fuse across.
    #[serde(default)]
    pub forbid: Vec<FusionBoundary>,
}

/// Memory permissions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryPolicy {
    /// Permitted placements.
    #[serde(default = "default_placements")]
    pub placements: Vec<MemoryPlacement>,
    /// Whether ping-pong buffering is permitted.
    #[serde(default = "default_true")]
    pub allow_ping_pong: bool,
    /// Maximum ping-pong buffers.
    #[serde(default = "default_ping_pong")]
    pub max_ping_pong_buffers: u32,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            placements: default_placements(),
            allow_ping_pong: true,
            max_ping_pong_buffers: default_ping_pong(),
        }
    }
}

fn default_placements() -> Vec<MemoryPlacement> {
    vec![
        MemoryPlacement::LineBuffer,
        MemoryPlacement::Bram,
        MemoryPlacement::Ddr,
    ]
}

fn default_true() -> bool {
    true
}

fn default_ping_pong() -> u32 {
    2
}

/// Per-operator restriction or prioritisation of fold factors.
///
/// Empty lists mean "no restriction". Present lists are intersected with the
/// legality rules (`simd` divides the fold dimension, `pe` divides output
/// channels, `tap` divides the kernel taps) and can never make an illegal fold
/// legal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorOverride {
    /// Operator name, or an [`OpKind`] tag such as `depthwise_conv2d`.
    pub pattern: String,
    /// Permitted SIMD factors.
    #[serde(default)]
    pub allowed_simd: Vec<u32>,
    /// Permitted PE factors.
    #[serde(default)]
    pub allowed_pe: Vec<u32>,
    /// Permitted tap-parallel factors.
    #[serde(default)]
    pub allowed_tap_parallel: Vec<u32>,
}

/// Soft hints that influence candidate ordering only.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hints {
    /// Prefer fused candidates.
    #[serde(default)]
    pub prefer_fusion: bool,
    /// Prefer on-chip over DDR placement.
    #[serde(default)]
    pub prefer_bram_over_ddr: bool,
    /// Prefer the shared engine.
    #[serde(default)]
    pub prefer_shared_engines: bool,
}

/// Planning policy: what the user permits and optimises.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    /// Must equal [`POLICY_FORMAT`].
    pub format: String,
    /// Must equal [`POLICY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Objective.
    pub objective: ObjectiveSpec,
    /// Hard constraints.
    pub constraints: PolicyConstraints,
    /// Search space.
    pub search: SearchPolicy,
    /// Fusion permissions.
    #[serde(default)]
    pub fusion: FusionPolicy,
    /// Memory permissions.
    #[serde(default)]
    pub memory: MemoryPolicy,
    /// Per-operator restrictions.
    #[serde(default)]
    pub operator_override: Vec<OperatorOverride>,
    /// Soft hints.
    #[serde(default)]
    pub hint: Hints,
}

/// A policy validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyError {
    /// Offending field.
    pub field: String,
    /// What is wrong.
    pub detail: String,
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "policy.{}: {}", self.field, self.detail)
    }
}

impl std::error::Error for PolicyError {}

impl Policy {
    /// Parse a policy file (schema binding only; call [`Policy::validate`] for
    /// semantic checks).
    pub fn from_toml_str(src: &str) -> Result<Self, crate::target::toml_lite::TomlError> {
        crate::target::toml_lite::from_toml_str(src)
    }

    /// Read and parse a policy file from disk.
    pub fn from_toml_file(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::target::toml_lite::TomlError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            crate::target::toml_lite::TomlError::plain(format!(
                "cannot read {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Self::from_toml_str(&text)
    }

    /// Deterministic TOML rendering.
    pub fn to_toml_string(&self) -> Result<String, crate::target::toml_lite::TomlError> {
        crate::target::toml_lite::to_toml_string_of(self)
    }

    /// Validate the policy for internal consistency.
    pub fn validate(&self) -> Result<(), PolicyError> {
        let err = |field: &str, detail: String| PolicyError {
            field: field.into(),
            detail,
        };
        if self.format != POLICY_FORMAT {
            return Err(err(
                "format",
                format!("expected {POLICY_FORMAT:?}, found {:?}", self.format),
            ));
        }
        if self.schema_version != POLICY_SCHEMA_VERSION {
            return Err(err(
                "schema_version",
                format!(
                    "unsupported schema_version {} (this build supports {POLICY_SCHEMA_VERSION})",
                    self.schema_version
                ),
            ));
        }
        if self.constraints.max_frame_cycles == 0 {
            return Err(err(
                "constraints.max_frame_cycles",
                "must be non-zero".into(),
            ));
        }
        if self.search.execution_modes.is_empty() {
            return Err(err("search.execution_modes", "must not be empty".into()));
        }
        if self.search.granularities.is_empty() {
            return Err(err("search.granularities", "must not be empty".into()));
        }
        if self.search.max_candidates == 0 {
            return Err(err("search.max_candidates", "must be non-zero".into()));
        }
        if self.search.utilisation_percent == 0 || self.search.utilisation_percent > 100 {
            return Err(err(
                "search.utilisation_percent",
                "must be in 1..=100".into(),
            ));
        }
        if self.search.granularities.contains(&Granularity::Tile)
            && (self.search.tile_widths.is_empty() || self.search.tile_heights.is_empty())
        {
            return Err(err(
                "search.tile_widths/tile_heights",
                "tile granularity requires at least one tile width and height".into(),
            ));
        }
        if self.search.granularities.contains(&Granularity::Row)
            && self.search.tile_heights.is_empty()
        {
            return Err(err(
                "search.tile_heights",
                "row granularity requires at least one stripe height".into(),
            ));
        }
        if self
            .search
            .tile_widths
            .iter()
            .chain(&self.search.tile_heights)
            .any(|d| *d == 0)
        {
            return Err(err(
                "search.tile_widths/tile_heights",
                "tile dimensions must be non-zero".into(),
            ));
        }
        if self.memory.placements.is_empty() {
            return Err(err("memory.placements", "must not be empty".into()));
        }
        let has_weight_placement = self
            .memory
            .placements
            .iter()
            .any(|p| matches!(p, MemoryPlacement::Bram | MemoryPlacement::Ddr));
        if !has_weight_placement {
            return Err(err(
                "memory.placements",
                "must permit at least one of bram or ddr for weight residency".into(),
            ));
        }
        if self.memory.allow_ping_pong
            && (self.memory.max_ping_pong_buffers == 0 || self.memory.max_ping_pong_buffers > 4)
        {
            return Err(err(
                "memory.max_ping_pong_buffers",
                "must be in 1..=4 when ping-pong is enabled".into(),
            ));
        }
        for (i, o) in self.operator_override.iter().enumerate() {
            if o.pattern.trim().is_empty() {
                return Err(err(
                    &format!("operator_override[{i}].pattern"),
                    "must not be empty".into(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Planner input contract
// ---------------------------------------------------------------------------

/// Padding description used to derive output spatial extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Padding {
    /// Explicit asymmetric zero padding.
    Zero {
        /// Top padding.
        top: u32,
        /// Bottom padding.
        bottom: u32,
        /// Left padding.
        left: u32,
        /// Right padding.
        right: u32,
    },
    /// Reflect padding; shape-equivalent to `same`.
    Reflect,
    /// Padding chosen to preserve the input extent for the given stride.
    Same,
}

/// Convolution detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvDetail {
    /// Weight tensor id.
    pub weights: TensorId,
    /// Optional bias tensor id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<TensorId>,
    /// Convolution groups (`1` dense, `c_in` depthwise).
    pub groups: u32,
    /// Stride `[sh, sw]`.
    pub stride: [u32; 2],
    /// Padding.
    pub padding: Padding,
    /// Requant contract.
    pub requant: Requant,
    /// Output activation.
    pub activation: Activation,
}

/// Elementwise detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElementwiseDetail {
    /// Optional requant contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requant: Option<Requant>,
    /// Output activation.
    pub activation: Activation,
}

/// Upsample detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpsampleDetail {
    /// Integer spatial scale factor.
    pub scale: u32,
}

/// Operator-specific payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpDetail {
    /// Convolution payload.
    Conv(ConvDetail),
    /// Elementwise payload.
    Elementwise(ElementwiseDetail),
    /// Upsample payload.
    Upsample(UpsampleDetail),
}

/// A tensor in the lowered primitive graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDesc {
    /// Stable id.
    pub id: TensorId,
    /// Stable symbol name.
    pub name: String,
    /// Shape.
    pub shape: TensorShape,
    /// Element type.
    pub elem: ElemType,
    /// Exact inclusive code interval this tensor carries, when the lowering can
    /// bound it.
    ///
    /// This is what sizes the MAC path: a point written by a residual `add`
    /// spans `[-254, 254]` and must be multiplied at nine bits even though it
    /// travels in a 16-bit word. Absent means "unknown", and an unknown width
    /// wider than a byte is rejected rather than guessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_range: Option<CodeRange>,
}

/// A lowered primitive operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveOp {
    /// Stable operator/layer name.
    pub name: String,
    /// Operator kind.
    pub kind: OpKind,
    /// Input tensor ids, in operand order.
    pub inputs: Vec<TensorId>,
    /// Output tensor ids.
    pub outputs: Vec<TensorId>,
    /// Operator payload.
    pub detail: OpDetail,
}

/// The planner's input contract: a typed, lowered primitive graph.
///
/// The integration layer adapts the frontend graph / lowered MLIR artifact to
/// this type; the planner deliberately does not parse manifests itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanInput {
    /// Graph name.
    pub name: String,
    /// Tensor table.
    pub tensors: Vec<TensorDesc>,
    /// Operators in topological execution order.
    pub ops: Vec<PrimitiveOp>,
    /// Graph input tensors.
    pub inputs: Vec<TensorId>,
    /// Graph output tensors.
    pub outputs: Vec<TensorId>,
}

/// Alias kept for the interface agreed with the integration agent.
pub type PrimitiveGraph = PlanInput;

impl PlanInput {
    /// Tensor descriptor by id.
    pub fn tensor(&self, id: TensorId) -> Option<&TensorDesc> {
        self.tensors.iter().find(|t| t.id == id)
    }

    /// Fan-out of a tensor: how many operators consume it.
    pub fn fanout(&self, id: TensorId) -> usize {
        self.ops.iter().filter(|op| op.inputs.contains(&id)).count()
    }

    /// Validate shapes, quantisation and connectivity.
    ///
    /// A malformed graph is a fatal error, never an infeasible plan.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("graph name must not be empty".into());
        }
        if self.tensors.is_empty() {
            return Err("graph has no tensors".into());
        }
        if self.ops.is_empty() {
            return Err("graph has no operators".into());
        }
        let mut seen_ids = BTreeSet::new();
        for t in &self.tensors {
            if !seen_ids.insert(t.id) {
                return Err(format!("duplicate tensor id {}", t.id));
            }
            if t.name.trim().is_empty() {
                return Err(format!("tensor {} has an empty name", t.id));
            }
            if t.shape.n == 0 || t.shape.c == 0 || t.shape.h == 0 || t.shape.w == 0 {
                return Err(format!("tensor {} has a zero dimension", t.name));
            }
            // The declared interval and the declared container must agree. A
            // value that needs nine bits cannot travel in an int8 word: that is
            // exactly how an unclamped residual gets silently narrowed, so it is
            // a hard error here rather than a wrong number later.
            if let Some(range) = t.code_range {
                if !range.is_valid() {
                    return Err(format!(
                        "tensor {} declares an empty code interval [{}, {}]",
                        t.name, range.lo, range.hi
                    ));
                }
                let (lo, hi) = t.elem.value_range();
                if range.lo < i64::from(lo) || range.hi > i64::from(hi) {
                    return Err(format!(
                        "tensor {} declares codes [{}, {}] which do not fit its {:?} container \
                         [{}, {}]; widen the container instead of narrowing the value",
                        t.name, range.lo, range.hi, t.elem, lo, hi
                    ));
                }
            }
        }
        let tensor = |id: TensorId| -> Result<&TensorDesc, String> {
            self.tensor(id)
                .ok_or_else(|| format!("unknown tensor {id} referenced by the graph"))
        };

        let mut op_names = BTreeSet::new();
        for op in &self.ops {
            if !op_names.insert(op.name.clone()) {
                return Err(format!("duplicate operator name {:?}", op.name));
            }
            if op.inputs.is_empty() || op.outputs.is_empty() {
                return Err(format!(
                    "operator {:?} must have inputs and outputs",
                    op.name
                ));
            }
            for id in op.inputs.iter().chain(&op.outputs) {
                tensor(*id)?;
            }
            match (&op.kind, &op.detail) {
                (kind, OpDetail::Conv(detail)) if kind.is_convolution() => {
                    if op.inputs.len() != 1 || op.outputs.len() != 1 {
                        return Err(format!(
                            "convolution {:?} must have exactly one input and one output",
                            op.name
                        ));
                    }
                    let input = tensor(op.inputs[0])?;
                    let output = tensor(op.outputs[0])?;
                    let weights = tensor(detail.weights)?;
                    if let Some(bias) = detail.bias {
                        let bias = tensor(bias)?;
                        if bias.shape.n != weights.shape.n
                            || bias.shape.c != 1
                            || bias.shape.h != 1
                            || bias.shape.w != 1
                        {
                            return Err(format!(
                                "operator {:?}: bias must be a {}-element vector",
                                op.name, weights.shape.n
                            ));
                        }
                    }
                    if detail.groups == 0 || detail.stride.contains(&0) {
                        return Err(format!("operator {:?} has zero groups or stride", op.name));
                    }
                    if input.shape.c % detail.groups != 0 || output.shape.c % detail.groups != 0 {
                        return Err(format!(
                            "operator {:?}: groups {} does not divide channels {} -> {}",
                            op.name, detail.groups, input.shape.c, output.shape.c
                        ));
                    }
                    if weights.shape.h == 0 || weights.shape.w == 0 {
                        return Err(format!("operator {:?}: empty kernel", op.name));
                    }
                    let (kh, kw) = (weights.shape.h, weights.shape.w);
                    match op.kind {
                        OpKind::DepthwiseConv2d => {
                            if detail.groups != input.shape.c || output.shape.c != input.shape.c {
                                return Err(format!(
                                    "operator {:?}: depthwise requires groups == in_channels == out_channels",
                                    op.name
                                ));
                            }
                            if weights.shape.c != 1 {
                                return Err(format!(
                                    "operator {:?}: depthwise weights must have one channel per group",
                                    op.name
                                ));
                            }
                        }
                        OpKind::PointwiseConv2d => {
                            if kh != 1 || kw != 1 || detail.groups != 1 {
                                return Err(format!(
                                    "operator {:?}: pointwise requires a 1x1 kernel and groups == 1",
                                    op.name
                                ));
                            }
                        }
                        _ => {}
                    }
                    let expected_ic = input.shape.c / detail.groups;
                    if weights.shape.n != output.shape.c || weights.shape.c != expected_ic {
                        return Err(format!(
                            "operator {:?}: weight shape {}x{}x{}x{} does not match {}->{} with groups {}",
                            op.name,
                            weights.shape.n,
                            weights.shape.c,
                            weights.shape.h,
                            weights.shape.w,
                            input.shape.c,
                            output.shape.c,
                            detail.groups
                        ));
                    }
                    if weights.elem != ElemType::I8 && weights.elem != ElemType::I4 {
                        return Err(format!(
                            "operator {:?}: v1 convolution weights must be I8 or I4",
                            op.name
                        ));
                    }
                    let out_hw = conv_out_spatial(
                        [input.shape.h, input.shape.w],
                        [kh, kw],
                        detail.stride,
                        detail.padding,
                    )
                    .ok_or_else(|| {
                        format!(
                            "operator {:?}: kernel {}x{} exceeds the padded input",
                            op.name, kh, kw
                        )
                    })?;
                    if [output.shape.h, output.shape.w] != out_hw {
                        return Err(format!(
                            "operator {:?}: declared output spatial {:?} does not match computed {:?}",
                            op.name,
                            [output.shape.h, output.shape.w],
                            out_hw
                        ));
                    }
                    // The container may be I8, the I9 residual transport, or the
                    // wider I16 transport; the *interval* is what sizes the
                    // multiplier, and it is stated per tensor.
                    if !matches!(input.elem, ElemType::I8 | ElemType::I9 | ElemType::I16) {
                        return Err(format!(
                            "operator {:?}: v1 convolution input must be I8 or a widened I9/I16 \
                             residual transport, found {:?}",
                            op.name, input.elem
                        ));
                    }
                    detail
                        .requant
                        .validate(output.shape.c)
                        .map_err(|detail| format!("operator {:?}: {detail}", op.name))?;
                }
                (OpKind::Add, OpDetail::Elementwise(_)) => {
                    if op.inputs.len() != 2 || op.outputs.len() != 1 {
                        return Err(format!("add {:?} must have two inputs", op.name));
                    }
                    let a = tensor(op.inputs[0])?;
                    let b = tensor(op.inputs[1])?;
                    let out = tensor(op.outputs[0])?;
                    if a.shape != b.shape || a.shape != out.shape {
                        return Err(format!("add {:?}: operand shapes must match", op.name));
                    }
                    for operand in [a, b] {
                        if !matches!(operand.elem, ElemType::I8 | ElemType::I9 | ElemType::I16) {
                            return Err(format!(
                                "add {:?}: operands must be I8 or a widened I9/I16 transport, \
                                 found {:?}",
                                op.name, operand.elem
                            ));
                        }
                    }
                    // A residual sum of two signed 8-bit codes spans -256..=255,
                    // which is exactly I9. Clamping it back to int8 would change
                    // the model's arithmetic, so it is rejected rather than
                    // silently narrowed.
                    if !out.elem.can_hold(ElemType::I9) {
                        return Err(format!(
                            "add {:?}: the unclamped residual sum needs an I9 (or wider) output, \
                             found {:?}; the planner never clamps a residual to int8",
                            op.name, out.elem
                        ));
                    }
                }
                (kind, OpDetail::Elementwise(_))
                    if matches!(
                        kind,
                        OpKind::Requant | OpKind::Activation | OpKind::QuantizeOutput
                    ) =>
                {
                    if op.inputs.len() != 1 || op.outputs.len() != 1 {
                        return Err(format!(
                            "elementwise {:?} must have exactly one input and output",
                            op.name
                        ));
                    }
                    let a = tensor(op.inputs[0])?;
                    let out = tensor(op.outputs[0])?;
                    if a.shape != out.shape {
                        return Err(format!(
                            "elementwise {:?}: shapes must be preserved",
                            op.name
                        ));
                    }
                    match kind {
                        OpKind::QuantizeOutput => {
                            if out.elem != ElemType::U8 {
                                return Err(format!(
                                    "quantize {:?}: the unsigned head must produce U8, found {:?}",
                                    op.name, out.elem
                                ));
                            }
                        }
                        // Requant and activation narrow values but never widen
                        // them; the transport type is preserved so a 9-bit
                        // residual stays 9-bit until something requantizes it.
                        _ => {
                            if out.elem != a.elem {
                                return Err(format!(
                                    "elementwise {:?}: element type {:?} must be preserved, found \
                                     {:?}",
                                    op.name, a.elem, out.elem
                                ));
                            }
                        }
                    }
                }
                (OpKind::NearestUpsample2x, OpDetail::Upsample(detail)) => {
                    if op.inputs.len() != 1 || op.outputs.len() != 1 {
                        return Err(format!("upsample {:?} must have one input", op.name));
                    }
                    if detail.scale == 0 {
                        return Err(format!("upsample {:?}: scale must be non-zero", op.name));
                    }
                    let a = tensor(op.inputs[0])?;
                    let out = tensor(op.outputs[0])?;
                    if out.elem != a.elem {
                        return Err(format!(
                            "upsample {:?}: element type {:?} must be preserved, found {:?}",
                            op.name, a.elem, out.elem
                        ));
                    }
                    if out.shape.c != a.shape.c
                        || out.shape.h != a.shape.h * detail.scale
                        || out.shape.w != a.shape.w * detail.scale
                    {
                        return Err(format!(
                            "upsample {:?}: output shape must be the scaled input",
                            op.name
                        ));
                    }
                }
                _ => {
                    return Err(format!(
                        "operator {:?}: kind {:?} does not match its detail payload",
                        op.name, op.kind
                    ));
                }
            }
        }

        let produced: BTreeSet<TensorId> = self
            .ops
            .iter()
            .flat_map(|op| op.outputs.iter().copied())
            .collect();
        for id in &self.inputs {
            tensor(*id)?;
            if produced.contains(id) {
                return Err(format!("graph input {id} is also produced by an operator"));
            }
        }
        for id in &self.outputs {
            tensor(*id)?;
            let producers = self.ops.iter().filter(|op| op.outputs.contains(id)).count();
            if producers != 1 {
                return Err(format!(
                    "graph output {id} must be produced exactly once (found {producers})"
                ));
            }
        }
        Ok(())
    }
}

/// Output spatial extent for a convolution.
fn conv_out_spatial(
    in_hw: [u32; 2],
    kernel: [u32; 2],
    stride: [u32; 2],
    padding: Padding,
) -> Option<[u32; 2]> {
    let axis = |input: u32, k: u32, s: u32, pad_a: u32, pad_b: u32| -> Option<u32> {
        if input + pad_a + pad_b < k {
            return None;
        }
        Some((input + pad_a + pad_b - k) / s + 1)
    };
    match padding {
        Padding::Zero {
            top,
            bottom,
            left,
            right,
        } => Some([
            axis(in_hw[0], kernel[0], stride[0], top, bottom)?,
            axis(in_hw[1], kernel[1], stride[1], left, right)?,
        ]),
        Padding::Reflect | Padding::Same => {
            let keep = |input: u32, s: u32| input.div_ceil(s);
            let oh = keep(in_hw[0], stride[0]);
            let ow = keep(in_hw[1], stride[1]);
            let pad_h = ((oh - 1) * stride[0] + kernel[0]).saturating_sub(in_hw[0]);
            let pad_w = ((ow - 1) * stride[1] + kernel[1]).saturating_sub(in_hw[1]);
            let a = axis(in_hw[0], kernel[0], stride[0], pad_h / 2, pad_h - pad_h / 2)?;
            let b = axis(in_hw[1], kernel[1], stride[1], pad_w / 2, pad_w - pad_w / 2)?;
            Some([a, b])
        }
    }
}

// ---------------------------------------------------------------------------
// Errors and provenance
// ---------------------------------------------------------------------------

/// Planner failure. Malformed inputs and targets are errors; budget
/// infeasibility is a *result* ([`PlanStatus::Infeasible`]), not an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    /// The input graph is malformed.
    InvalidInput { detail: String },
    /// The target description is invalid.
    Target(TargetError),
    /// The policy is invalid.
    Policy { detail: String },
}

impl std::fmt::Display for PlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlannerError::InvalidInput { detail } => write!(f, "invalid planner input: {detail}"),
            PlannerError::Target(e) => write!(f, "invalid target: {e}"),
            PlannerError::Policy { detail } => write!(f, "invalid policy: {detail}"),
        }
    }
}

impl std::error::Error for PlannerError {}

impl From<TargetError> for PlannerError {
    fn from(e: TargetError) -> Self {
        PlannerError::Target(e)
    }
}

/// Provenance inputs supplied by the caller (content hashes of the artifacts
/// the solve consumed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceInput {
    /// Hash of the input graph artifact.
    pub graph_sha256: String,
    /// Hash of the target file.
    pub target_sha256: String,
    /// Hash of the policy file.
    pub policy_sha256: String,
    /// Compiler version string.
    pub compiler_version: String,
}

// Search helpers deleted: planner::solve forwards to crate::solver.

pub fn solve(
    input: &PlanInput,
    target: &Ti60Target,
    policy: &Policy,
    provenance: &ProvenanceInput,
) -> Result<LayoutArtifact, PlannerError> {
    let bounds = crate::solver::MappingBounds::hybrid_for(target, policy).map_err(|error| {
        PlannerError::InvalidInput {
            detail: error.to_string(),
        }
    })?;
    crate::solver::solve_mapping(input, target, policy, provenance, &bounds).map_err(|error| {
        PlannerError::InvalidInput {
            detail: error.to_string(),
        }
    })
}
