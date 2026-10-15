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

// ---------------------------------------------------------------------------
// Precomputed operator information
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpInfo {
    name: String,
    kind: OpKind,
    is_conv: bool,
    in_channels: u32,
    out_channels: u32,
    in_spatial: [u32; 2],
    out_spatial: [u32; 2],
    kernel: [u32; 2],
    groups: u32,
    stride: [u32; 2],
    /// Channels folded by SIMD: per-group input channels for dense convs,
    /// output channels for depthwise and elementwise operators.
    channel_lanes: u32,
    /// Multiply-accumulate reduction length per output element
    /// (`taps × input channels per group` for a convolution, `1` otherwise).
    reduction: u32,
    act_bits: u32,
    weight_bits: u32,
    requant: Option<Requant>,
    activation: Activation,
    weights: Option<TensorRef>,
    bias: Option<TensorRef>,
    /// Number of weight codes (words), independent of their packing.
    weight_elems: u64,
    /// Stored weight element type.
    weight_elem: ElemType,
    /// Number of bias codes.
    bias_elems: u64,
    /// Resolved spatial padding, for spatial operators.
    padding: Option<PaddingSpec>,
    /// Nearest-neighbour scale, for upsample operators.
    upsample_scale: Option<u32>,
    /// Exact code interval of the activation this operator reads.
    in_code_range: Option<CodeRange>,
    /// Exact code interval of the activation this operator writes.
    out_code_range: Option<CodeRange>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    out_elems: u64,
    weight_bytes: u64,
}

impl OpInfo {
    fn taps(&self) -> u32 {
        self.kernel[0] * self.kernel[1]
    }

    fn input_tensor(&self) -> TensorId {
        self.inputs[0]
    }

    /// Output positions of the whole frame.
    fn out_pixels(&self) -> u64 {
        self.out_spatial[0].max(1) as u64 * self.out_spatial[1].max(1) as u64
    }

    /// Multiply-accumulate products this operator performs for the whole frame.
    fn macs(&self) -> u64 {
        self.out_elems.saturating_mul(self.reduction as u64)
    }

    /// Whether the operator is a 1x1, stride-1, single-group, bias-free
    /// pointwise convolution: the legacy streaming envelope's only shape.
    fn is_plain_pointwise(&self) -> bool {
        self.is_conv
            && self.kind != OpKind::DepthwiseConv2d
            && self.kernel == [1, 1]
            && self.groups == 1
            && self.stride == [1, 1]
            && self.bias.is_none()
    }

    /// Entries this operator contributes to the pooled requantization ROM.
    ///
    /// The engine stores one 17-bit `M0` per *output channel* of the step,
    /// whether the plan carries a scalar multiplier (replicated across the
    /// channels) or an explicit per-channel list; an operator with no
    /// requantization at all contributes nothing. This is the same count the
    /// backend bakes into `m0_rom`, so the pooled depth is not a second guess.
    fn requant_entries(&self) -> u64 {
        if self.requant.is_some() {
            u64::from(self.out_channels).max(1)
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FoldConfig {
    simd: u32,
    pe: u32,
    tap: u32,
}

impl FoldConfig {
    fn lanes(self) -> u64 {
        self.simd as u64 * self.pe as u64 * self.tap as u64
    }
}

/// Word counts one operator contributes to the engine's *pooled* read-only
/// memories.
///
/// The shared engine emits one `w_rom`, one `b_rom` and one `m0_rom` holding
/// every step's slice, so what the budget sees is the total word count of each
/// pool, not the sum of per-layer block roundings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RomWords {
    /// 8-bit weight codes (the manifest's `i4-x2` packing is unpacked to one
    /// signed code per entry before it reaches the ROM).
    weights: u64,
    /// Accumulator-width bias words, one per output channel of every step the
    /// generated `b_rom` reserves a slot for.
    bias: u64,
    /// 17-bit requantization multipliers, one per output channel of every
    /// step that requantizes.
    multipliers: u64,
}

impl RomWords {
    fn plus(self, other: RomWords) -> RomWords {
        RomWords {
            weights: self.weights.saturating_add(other.weights),
            bias: self.bias.saturating_add(other.bias),
            multipliers: self.multipliers.saturating_add(other.multipliers),
        }
    }

    /// Total words, for ranking only.
    fn total(self) -> u64 {
        self.weights
            .saturating_add(self.bias)
            .saturating_add(self.multipliers)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpEval {
    fold: FoldConfig,
    lanes: u64,
    dsp_mac: u32,
    dsp_requant: u32,
    /// This step's slice of the pooled ROMs.
    rom: RomWords,
    /// Standalone RAM10 blocks of this step's own buffers. Informational and
    /// used only for Pareto ranking; the accepted plan is judged on the pooled
    /// figure, which is what the hardware actually instantiates.
    standalone_ram: u32,
    cost: StepCost,
    cycles: u64,
    buffers: Vec<BufferBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GlobalCandidate {
    mode: ExecutionMode,
    granularity: Granularity,
    /// Rectangular tile, only for `Granularity::Tile`.
    tile: Option<TileGeometry>,
    /// Row stripe height, only for `Granularity::Row`.
    stripe_height: Option<u32>,
    weight_placement: MemoryPlacement,
    intermediate_placement: MemoryPlacement,
    fusion_enabled: bool,
}

impl GlobalCandidate {
    /// Effective per-operator tile `(height, width)` in output pixels.
    fn effective_tile(&self, info: &OpInfo) -> (u32, u32) {
        let oh = info.out_spatial[0].max(1);
        let ow = info.out_spatial[1].max(1);
        match self.granularity {
            Granularity::Frame => (oh, ow),
            Granularity::Row => (self.stripe_height.unwrap_or(1).max(1), ow),
            Granularity::Tile => {
                let t = self.tile.unwrap_or(TileGeometry::new(1, 1));
                (t.height.max(1), t.width.max(1))
            }
        }
    }
}

/// Monotone search state. Every field only grows along a DFS branch, so the
/// derived lower bounds are sound.
#[derive(Debug, Clone, Copy, Default)]
struct Partial {
    conv_mac_sum: u32,
    conv_mac_max: u32,
    nonconv_mac_sum: u32,
    nonconv_mac_max: u32,
    conv_rq_sum: u32,
    conv_rq_max: u32,
    nonconv_rq_sum: u32,
    nonconv_rq_max: u32,
    conv_lanes_max: u64,
    nonconv_lanes_max: u64,
    nonconv_lanes_sum: u64,
    /// Pooled ROM words contributed by the prefix. Every pool is monotone in
    /// its word count, so the pooled block figure of a prefix is a sound lower
    /// bound of the finished plan's.
    rom: RomWords,
    cycles_sum: u64,
}

impl Partial {
    fn add(&mut self, eval: &OpEval, is_conv: bool) {
        if is_conv {
            self.conv_mac_sum = self.conv_mac_sum.saturating_add(eval.dsp_mac);
            self.conv_mac_max = self.conv_mac_max.max(eval.dsp_mac);
            self.conv_rq_sum = self.conv_rq_sum.saturating_add(eval.dsp_requant);
            self.conv_rq_max = self.conv_rq_max.max(eval.dsp_requant);
            self.conv_lanes_max = self.conv_lanes_max.max(eval.lanes);
        } else {
            self.nonconv_mac_sum = self.nonconv_mac_sum.saturating_add(eval.dsp_mac);
            self.nonconv_mac_max = self.nonconv_mac_max.max(eval.dsp_mac);
            self.nonconv_rq_sum = self.nonconv_rq_sum.saturating_add(eval.dsp_requant);
            self.nonconv_rq_max = self.nonconv_rq_max.max(eval.dsp_requant);
            self.nonconv_lanes_max = self.nonconv_lanes_max.max(eval.lanes);
            self.nonconv_lanes_sum = self.nonconv_lanes_sum.saturating_add(eval.lanes);
        }
        self.rom = self.rom.plus(eval.rom);
        self.cycles_sum = self.cycles_sum.saturating_add(eval.cycles);
    }

    /// Exact `(dsp48, physical_lanes)` for `mode`. These are also valid lower
    /// bounds for a prefix of the operator list.
    ///
    /// RAM is *not* part of this: the engine's memories are pooled, and only
    /// the leaf knows the arena allocation's extent, so the RAM bound is
    /// computed separately by
    /// [`Search::ram_lower_bound`].
    fn aggregate(&self, mode: ExecutionMode) -> (u32, u64) {
        let mac = match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => {
                self.conv_mac_sum + self.nonconv_mac_sum
            }
            ExecutionMode::SharedTile => self.conv_mac_max.max(self.nonconv_mac_max),
            ExecutionMode::Hybrid => self.conv_mac_max + self.nonconv_mac_sum,
        };
        let rq = match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => {
                self.conv_rq_sum + self.nonconv_rq_sum
            }
            ExecutionMode::SharedTile => self.conv_rq_max.max(self.nonconv_rq_max),
            ExecutionMode::Hybrid => self.conv_rq_max + self.nonconv_rq_sum,
        };
        let lanes = match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => {
                self.conv_lanes_max + self.nonconv_lanes_sum
            }
            ExecutionMode::SharedTile => self.conv_lanes_max.max(self.nonconv_lanes_max),
            ExecutionMode::Hybrid => self.conv_lanes_max + self.nonconv_lanes_sum,
        };
        (mac + rq, lanes)
    }
}

/// Physical mapping of the memories an organisation actually instantiates.
///
/// The engine's storage is *pooled*: one weight ROM, one bias ROM, one
/// multiplier ROM and one activation arena, each read by the single memory port
/// the sequential engine has. Mapping the pool once — instead of rounding each
/// layer's slice up to whole blocks and adding the roundings — is what makes
/// the figure the plan is accepted on the figure the RTL realizes.
#[derive(Debug, Clone, Copy)]
struct PoolModel<'a> {
    target: &'a Ti60Target,
    primitive: &'a str,
    /// Activation word width, in bits.
    activation_bits: u64,
    /// Stored weight word width, in bits: the manifest's packed weights are
    /// unpacked to one code per entry before they reach the ROM.
    weight_bits: u64,
    /// Accumulator word width, in bits, which is the bias ROM's word width.
    accumulator_bits: u64,
    /// Whether the organisation keeps a RAM activation arena at all. The legacy
    /// streaming envelope streams activations through registers and FIFOs, so
    /// its activation footprint is genuinely zero.
    keeps_arena: bool,
    /// Whether the organisation keeps a separate bias ROM.
    keeps_bias_rom: bool,
}

impl PoolModel<'_> {
    /// Blocks one pooled memory of `width_bits × depth` needs.
    fn blocks(
        &self,
        width_bits: u64,
        depth: u64,
        read_ports: u64,
        write_ports: u64,
    ) -> Result<u64, RejectReason> {
        Ok(self
            .target
            .map_buffer(
                self.primitive,
                BufferRequest {
                    width_bits: width_bits.max(1),
                    depth: depth.max(1),
                    read_ports,
                    write_ports,
                },
            )
            .map_err(|e| RejectReason::IllegalFold {
                op: "pooled-memory".into(),
                detail: e.to_string(),
            })?
            .blocks
            .get())
    }

    /// Blocks for the ROM pools plus a `live_words`-word activation arena.
    ///
    /// Every pool is mapped at its own word width, because the engine emits one
    /// physical memory per pool: 8-bit weight codes, accumulator-width bias
    /// words, 17-bit multipliers, and 16-bit activation words.
    fn pooled(&self, rom: RomWords, live_words: u64) -> Result<u64, RejectReason> {
        let mut total = 0u64;
        // `map_buffer` rejects a zero-depth request, and an absent pool is
        // genuinely zero blocks, so empty pools are skipped rather than
        // rounded up to one block.
        if rom.weights > 0 {
            total = total.saturating_add(self.blocks(self.weight_bits, rom.weights, 1, 0)?);
        }
        if self.keeps_bias_rom && rom.bias > 0 {
            total = total.saturating_add(self.blocks(self.accumulator_bits, rom.bias, 1, 0)?);
        }
        if rom.multipliers > 0 {
            total = total.saturating_add(self.blocks(
                REQUANT_MULTIPLIER_BITS,
                rom.multipliers,
                1,
                0,
            )?);
        }
        if self.keeps_arena && live_words > 0 {
            // One synchronous read port for the MAC/elementwise operand path and
            // one synchronous write port for the epilogue store. The
            // verification port is *not* a second read port: it is muxed onto
            // the single read port while the FSM is idle.
            total = total.saturating_add(self.blocks(self.activation_bits, live_words, 1, 1)?);
        }
        Ok(total)
    }
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

type Best = (ResourceEstimate, GlobalCandidate, Vec<OpEval>);

struct Search<'a> {
    input: &'a PlanInput,
    target: &'a Ti60Target,
    policy: &'a Policy,
    ops: Vec<OpInfo>,
    tensors: Vec<TensorRef>,
    graph_inputs: BTreeSet<TensorId>,
    graph_outputs: BTreeSet<TensorId>,
    limits: PlanLimits,
    candidate_traffic: (u64, u64),
    options: Vec<Vec<OpEval>>,
    best: Option<Best>,
    expanded: u64,
    explored: u64,
    /// Node-expansion share granted to each outer candidate.
    per_candidate_budget: u64,
    /// Outer candidates whose inner search was cut.
    truncated_candidates: u64,
    truncated: bool,
    first_reason: Option<RejectReason>,
    rejections: Vec<RejectedCandidate>,
    rejection_counts: BTreeMap<String, u64>,
    rejected_total: u64,
    mode_summaries: BTreeMap<String, ModeSummary>,
}

impl<'a> Search<'a> {
    fn new(
        input: &'a PlanInput,
        target: &'a Ti60Target,
        policy: &'a Policy,
    ) -> Result<Self, PlannerError> {
        let usable = target.usable_budget()?;
        let ddr = target.ddr_budget(FrameCycles(policy.constraints.max_frame_cycles));
        let limits = PlanLimits {
            dsp48: usable.dsp48.get(),
            ram10: usable.ram10.get(),
            xlr: usable.xlr.get(),
            frame_cycles: policy.constraints.max_frame_cycles,
            ddr_read_bytes_per_frame: ddr.read_bytes.get(),
            ddr_write_bytes_per_frame: ddr.write_bytes.get(),
        };
        let ops = build_op_infos(input)?;
        let tensors = input
            .tensors
            .iter()
            .map(|t| TensorRef {
                id: t.id,
                name: t.name.clone(),
                shape: t.shape,
                elem: t.elem,
            })
            .collect();
        Ok(Self {
            input,
            target,
            policy,
            ops,
            tensors,
            graph_inputs: input.inputs.iter().copied().collect(),
            graph_outputs: input.outputs.iter().copied().collect(),
            limits,
            candidate_traffic: (0, 0),
            options: Vec::new(),
            best: None,
            expanded: 0,
            explored: 0,
            per_candidate_budget: policy.search.max_candidates,
            truncated_candidates: 0,
            truncated: false,
            first_reason: None,
            rejections: Vec::new(),
            rejection_counts: BTreeMap::new(),
            rejected_total: 0,
            mode_summaries: BTreeMap::new(),
        })
    }

    /// Per-mode outcome record, created on demand.
    fn mode_summary(&mut self, mode: ExecutionMode) -> &mut ModeSummary {
        self.mode_summaries
            .entry(mode.as_str().to_string())
            .or_insert_with(|| ModeSummary {
                candidates: 0,
                feasible: 0,
                counts: BTreeMap::new(),
                first_reason: None,
            })
    }

    /// Whether the v1 emitters can realize this candidate's architecture.
    ///
    /// The planner enumerates every mode the policy permits, then rejects the
    /// ones no backend implements *explicitly* — a plan that claims a
    /// streaming engine per operator, or a hybrid of a shared convolution
    /// engine with streaming elementwise paths, would describe hardware that
    /// does not exist.
    ///
    /// Each accepted mode names a *realization*, and the realization fixes the
    /// rest of the plan's shape:
    ///
    /// * [`ExecutionMode::SharedTile`] — one shared, single-lane, RAM-backed
    ///   engine running a static FSM over every step. Its activations live in
    ///   the activation arena, so the whole frame is resident and the
    ///   granularity is [`Granularity::Frame`]; a tiled schedule would need
    ///   each tile's halo from a neighbouring tile that this FSM never
    ///   produces, and would silently re-read DDR.
    /// * [`ExecutionMode::Streaming`] / [`ExecutionMode::Dedicated`] — the
    ///   legacy single-step pointwise-dot kernel, which is a real, emitted
    ///   envelope but only for one plain 1x1 convolution.
    fn mode_support(&self, candidate: &GlobalCandidate) -> Result<(), RejectReason> {
        let unsupported = |detail: &str| RejectReason::UnsupportedArchitecture {
            mode: candidate.mode,
            detail: detail.to_string(),
        };
        match candidate.mode {
            ExecutionMode::SharedTile => {
                if candidate.granularity != Granularity::Frame {
                    return Err(unsupported(
                        "the RAM-backed engine runs a frame-sequential static FSM; a tiled or \
                         row-striped schedule would need each tile's halo from a neighbouring \
                         tile, which that FSM does not produce",
                    ));
                }
                if candidate.intermediate_placement == MemoryPlacement::Ddr {
                    return Err(unsupported(
                        "the RAM-backed engine keeps every activation on chip; a DDR-resident \
                         intermediate would need a DMA stage between steps",
                    ));
                }
                if candidate.intermediate_placement == MemoryPlacement::LineBuffer {
                    return Err(unsupported(
                        "the RAM-backed engine reads its frame-resident activation arena at \
                         computed addresses; it instantiates no per-step line buffer, so an \
                         intermediate cannot be placed in one",
                    ));
                }
                if candidate.weight_placement == MemoryPlacement::Ddr {
                    return Err(unsupported(
                        "weights must be RAM-resident; a DDR weight placement needs a weight DMA \
                         that v1 does not emit",
                    ));
                }
                if self.ops.iter().any(|op| !op.is_conv && op.reduction != 1) {
                    return Err(unsupported(
                        "the shared engine's elementwise path is a single-element datapath",
                    ));
                }
                let (lanes, _, _, _) = engine_datapath(candidate.mode);
                if lanes > SHARED_ENGINE_LANE_CAP {
                    return Err(unsupported(
                        "the shared emitter builds at most 64 multipliers",
                    ));
                }
                Ok(())
            }
            ExecutionMode::Streaming | ExecutionMode::Dedicated => {
                if self.ops.len() != 1 {
                    return Err(unsupported(
                        "one specialized/pipelined engine per operator is not emitted; the \
                         supported envelope is a single pointwise step",
                    ));
                }
                if !self.ops[0].is_plain_pointwise() {
                    return Err(unsupported(
                        "the streaming envelope emits a 1x1, stride-1, single-group, bias-free \
                         pointwise dot kernel only",
                    ));
                }
                if candidate.weight_placement == MemoryPlacement::Ddr {
                    return Err(unsupported(
                        "the pointwise kernel reads its weights from an on-chip ROM; a DDR weight \
                         placement is not emitted",
                    ));
                }
                Ok(())
            }
            ExecutionMode::Hybrid => Err(unsupported(
                "v1 emits one shared RAM-backed engine; the streaming elementwise paths of the \
                 hybrid organisation are not implemented",
            )),
        }
    }

    fn weight_placements(&self) -> Vec<MemoryPlacement> {
        let mut out = Vec::new();
        for p in &self.policy.memory.placements {
            if matches!(p, MemoryPlacement::Bram | MemoryPlacement::Ddr) && !out.contains(p) {
                out.push(*p);
            }
        }
        out
    }

    fn intermediate_placements(&self, mode: ExecutionMode) -> Vec<MemoryPlacement> {
        if matches!(mode, ExecutionMode::Streaming | ExecutionMode::Dedicated) {
            // The legacy pointwise envelope streams its activation between
            // engines; there is no intermediate storage to place.
            return vec![MemoryPlacement::Fifo];
        }
        let mut out = Vec::new();
        for p in &self.policy.memory.placements {
            if matches!(
                p,
                MemoryPlacement::Bram | MemoryPlacement::LineBuffer | MemoryPlacement::Ddr
            ) && !out.contains(p)
            {
                out.push(*p);
            }
        }
        if out.is_empty() {
            out.push(MemoryPlacement::Bram);
        }
        out
    }

    /// Enumerate the whole outer space before searching it.
    ///
    /// The node-expansion budget is then shared fairly across candidates. A
    /// single greedy first-candidate sweep can consume the entire budget and
    /// leave whole execution modes unexplored, which makes the result
    /// uninformative; a per-candidate share guarantees every enumerated mode
    /// and placement is actually searched.
    fn enumerate_candidates(&self) -> Vec<GlobalCandidate> {
        let mut modes = self.policy.search.execution_modes.clone();
        modes.dedup();
        let mut granularities = self.policy.search.granularities.clone();
        granularities.dedup();
        let weight_placements = self.weight_placements();
        let mut out = Vec::new();
        for mode in modes {
            for granularity in &granularities {
                let tiles: Vec<(Option<TileGeometry>, Option<u32>)> = match granularity {
                    Granularity::Frame => vec![(None, None)],
                    Granularity::Row => self
                        .policy
                        .search
                        .tile_heights
                        .iter()
                        .map(|h| (None, Some(*h)))
                        .collect(),
                    Granularity::Tile => {
                        let mut v = Vec::new();
                        for h in &self.policy.search.tile_heights {
                            for w in &self.policy.search.tile_widths {
                                v.push((Some(TileGeometry::new(*w, *h)), None));
                            }
                        }
                        v
                    }
                };
                for (tile, stripe) in tiles {
                    for wp in &weight_placements {
                        for ip in self.intermediate_placements(mode) {
                            for fusion_enabled in [true, false] {
                                out.push(GlobalCandidate {
                                    mode,
                                    granularity: *granularity,
                                    tile,
                                    stripe_height: stripe,
                                    weight_placement: *wp,
                                    intermediate_placement: ip,
                                    fusion_enabled,
                                });
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn run(&mut self) {
        let candidates = self.enumerate_candidates();
        let count = candidates.len().max(1) as u64;
        let total = self.policy.search.max_candidates;
        let per_candidate = (total / count).max(1);
        self.per_candidate_budget = per_candidate;
        for candidate in candidates {
            let limit = self.expanded.saturating_add(per_candidate);
            self.explored += 1;
            self.mode_summary(candidate.mode).candidates += 1;
            self.search_candidate(candidate, limit);
            if self.expanded >= limit {
                self.truncated = true;
                self.truncated_candidates += 1;
            }
        }
    }

    fn search_candidate(&mut self, candidate: GlobalCandidate, node_limit: u64) {
        if let Err(reason) = self.mode_support(&candidate) {
            self.reject(&candidate, reason);
            return;
        }
        self.first_reason = None;
        self.candidate_traffic = self.compute_ddr_traffic(&candidate);
        if self.candidate_traffic.0 > self.limits.ddr_read_bytes_per_frame {
            self.reject(
                &candidate,
                RejectReason::DdrReadBudget {
                    used: self.candidate_traffic.0,
                    budget: self.limits.ddr_read_bytes_per_frame,
                },
            );
            return;
        }
        if self.candidate_traffic.1 > self.limits.ddr_write_bytes_per_frame {
            self.reject(
                &candidate,
                RejectReason::DdrWriteBudget {
                    used: self.candidate_traffic.1,
                    budget: self.limits.ddr_write_bytes_per_frame,
                },
            );
            return;
        }

        let mut options: Vec<Vec<OpEval>> = Vec::with_capacity(self.ops.len());
        for info in &self.ops {
            match self.op_options(info, &candidate) {
                Ok(list) if !list.is_empty() => options.push(list),
                Ok(_) => {
                    self.reject(
                        &candidate,
                        RejectReason::IllegalFold {
                            op: info.name.clone(),
                            detail: "no legal fold factors under this policy".into(),
                        },
                    );
                    return;
                }
                Err(reason) => {
                    self.reject(&candidate, reason);
                    return;
                }
            }
        }
        self.options = options;

        let mut assign = vec![0usize; self.ops.len()];
        let before = self.best.clone();
        self.dfs(0, Partial::default(), &mut assign, &candidate, node_limit);
        if self.best == before {
            let reason = self
                .first_reason
                .take()
                .unwrap_or(RejectReason::NoCandidate {
                    detail: "no fold assignment satisfies every budget".into(),
                });
            self.reject(&candidate, reason);
        } else {
            self.mode_summary(candidate.mode).feasible += 1;
        }
    }

    fn reject(&mut self, candidate: &GlobalCandidate, reason: RejectReason) {
        self.rejected_total += 1;
        *self
            .rejection_counts
            .entry(reason.category().to_string())
            .or_insert(0) += 1;
        {
            let summary = self.mode_summary(candidate.mode);
            *summary
                .counts
                .entry(reason.category().to_string())
                .or_insert(0) += 1;
            if summary.first_reason.is_none() {
                summary.first_reason = Some(reason.clone());
            }
        }
        if self.rejections.len() < MAX_REJECTION_SAMPLES {
            self.rejections.push(RejectedCandidate {
                execution_mode: candidate.mode,
                granularity: candidate.granularity,
                tile: candidate.tile,
                weight_placement: candidate.weight_placement,
                intermediate_placement: candidate.intermediate_placement,
                fusion_enabled: candidate.fusion_enabled,
                reason,
            });
        }
    }

    fn dfs(
        &mut self,
        index: usize,
        partial: Partial,
        assign: &mut [usize],
        candidate: &GlobalCandidate,
        node_limit: u64,
    ) {
        if index == self.ops.len() {
            self.leaf(partial, assign, candidate);
            return;
        }
        let n_options = self.options[index].len();
        for choice in 0..n_options {
            if self.expanded >= node_limit {
                self.truncated = true;
                return;
            }
            self.expanded += 1;
            let eval = self.options[index][choice].clone();
            let mut next = partial;
            next.add(&eval, self.ops[index].is_conv);

            let (dsp, _lanes) = next.aggregate(candidate.mode);
            if dsp as u64 > self.limits.dsp48 {
                self.note_reason(RejectReason::DspBudget {
                    used: dsp as u64,
                    budget: self.limits.dsp48,
                });
                continue;
            }
            match self.ram_lower_bound(candidate.mode, next.rom) {
                Ok(ram) if ram > self.limits.ram10 => {
                    self.note_reason(RejectReason::RamBudget {
                        used: ram,
                        budget: self.limits.ram10,
                    });
                    continue;
                }
                Ok(_) => {}
                Err(reason) => {
                    self.note_reason(reason);
                    continue;
                }
            }
            if next.cycles_sum > self.limits.frame_cycles {
                self.note_reason(RejectReason::FrameCycles {
                    used: next.cycles_sum,
                    budget: self.limits.frame_cycles,
                });
                continue;
            }
            if self.policy.objective.primary == ObjectiveTerm::MinimizeCycles {
                if let Some((incumbent, _, _)) = &self.best {
                    if next.cycles_sum >= incumbent.frame_cycles {
                        continue;
                    }
                }
            }
            assign[index] = choice;
            self.dfs(index + 1, next, assign, candidate, node_limit);
            if self.expanded >= node_limit {
                self.truncated = true;
                return;
            }
        }
    }

    fn leaf(&mut self, partial: Partial, assign: &[usize], candidate: &GlobalCandidate) {
        let (dsp, lanes) = partial.aggregate(candidate.mode);
        let chosen: Vec<OpEval> = (0..self.ops.len())
            .map(|i| self.options[i][assign[i]].clone())
            .collect();
        // The pruning bound above is a lower bound; the accepted estimate must
        // be the exact pooled figure, so a plan can never be accepted on a RAM
        // number that ignores a live frame or the activation arena.
        let ram = match self.exact_ram_blocks(candidate, &chosen) {
            Ok(ram) => ram,
            Err(reason) => {
                self.note_reason(reason);
                return;
            }
        };
        let dma_cycles = self.dma_cycles_for(candidate.mode);
        let cycles = partial.cycles_sum.saturating_add(dma_cycles);
        let engines = self.engine_count(candidate.mode);
        let xlr = self.target.estimate_xlr(lanes, engines, ram).get();
        let estimate = ResourceEstimate {
            dsp48: dsp as u64,
            ram10: ram,
            xlr,
            frame_cycles: cycles,
            ddr_read_bytes_per_frame: self.candidate_traffic.0,
            ddr_write_bytes_per_frame: self.candidate_traffic.1,
        };
        if let Some((resource, used, budget)) = estimate.first_violation(&self.limits) {
            let reason = match resource {
                "dsp48" => RejectReason::DspBudget { used, budget },
                "ram10" => RejectReason::RamBudget { used, budget },
                "xlr" => RejectReason::XlrBudget { used, budget },
                "frame_cycles" => RejectReason::FrameCycles { used, budget },
                "ddr_read_bytes_per_frame" => RejectReason::DdrReadBudget { used, budget },
                _ => RejectReason::DdrWriteBudget { used, budget },
            };
            self.note_reason(reason);
            return;
        }
        let better = match &self.best {
            None => true,
            Some((incumbent, _, _)) => self.policy.objective.is_better(&estimate, incumbent),
        };
        if better {
            self.best = Some((estimate, candidate.clone(), chosen));
        }
    }

    fn note_reason(&mut self, reason: RejectReason) {
        if self.first_reason.is_none() {
            self.first_reason = Some(reason);
        }
    }

    fn engine_count(&self, mode: ExecutionMode) -> u64 {
        match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => self.ops.len() as u64,
            ExecutionMode::SharedTile => 1,
            ExecutionMode::Hybrid => 1 + self.ops.iter().filter(|o| !o.is_conv).count() as u64,
        }
    }

    /// DMA cycles charged for a mode.
    ///
    /// Only the streaming organisation overlaps DDR transfer with compute
    /// (that is what its FIFO interfaces are for). Every other organisation —
    /// including the RAM-backed shared engine — pays the transfer serially.
    fn dma_cycles_for(&self, mode: ExecutionMode) -> u64 {
        if matches!(mode, ExecutionMode::Streaming) {
            0
        } else {
            self.dma_cycles()
        }
    }

    fn dma_cycles(&self) -> u64 {
        let clock = self.target.clock().hz();
        if clock == 0 {
            return 0;
        }
        let read_bpc = self.target.budget.ddr_read_bytes_per_s / clock;
        let write_bpc = self.target.budget.ddr_write_bytes_per_s / clock;
        let read = if read_bpc == 0 {
            0
        } else {
            self.candidate_traffic.0.div_ceil(read_bpc)
        };
        let write = if write_bpc == 0 {
            0
        } else {
            self.candidate_traffic.1.div_ceil(write_bpc)
        };
        read + write
    }

    fn operator_override(&self, info: &OpInfo) -> Option<OperatorOverride> {
        let kind_tag = match info.kind {
            OpKind::Conv2d => "conv2d",
            OpKind::DepthwiseConv2d => "depthwise_conv2d",
            OpKind::PointwiseConv2d => "pointwise_conv2d",
            OpKind::Add => "add",
            OpKind::Requant => "requant",
            OpKind::Activation => "activation",
            OpKind::NearestUpsample2x => "nearest_upsample2x",
            OpKind::QuantizeOutput => "quantize_output",
        };
        self.policy
            .operator_override
            .iter()
            .find(|o| o.pattern == info.name || o.pattern == kind_tag)
            .cloned()
    }

    fn op_options(
        &self,
        info: &OpInfo,
        candidate: &GlobalCandidate,
    ) -> Result<Vec<OpEval>, RejectReason> {
        let override_ = self.operator_override(info);
        let restrict = |pool: Vec<u32>, allowed: &[u32], dividend: u32| -> Vec<u32> {
            if allowed.is_empty() {
                pool
            } else {
                allowed
                    .iter()
                    .copied()
                    .filter(|f| *f > 0 && dividend % *f == 0)
                    .collect()
            }
        };

        let simd_pool = restrict(
            divisors(info.channel_lanes),
            override_
                .as_ref()
                .map(|o| o.allowed_simd.as_slice())
                .unwrap_or(&[]),
            info.channel_lanes,
        );
        let pe_pool = if info.kind == OpKind::DepthwiseConv2d || !info.is_conv {
            vec![1u32]
        } else {
            restrict(
                divisors(info.out_channels),
                override_
                    .as_ref()
                    .map(|o| o.allowed_pe.as_slice())
                    .unwrap_or(&[]),
                info.out_channels,
            )
        };
        let tap_pool = if info.is_conv {
            restrict(
                divisors(info.taps()),
                override_
                    .as_ref()
                    .map(|o| o.allowed_tap_parallel.as_slice())
                    .unwrap_or(&[]),
                info.taps(),
            )
        } else {
            vec![1u32]
        };

        let mut options: Vec<OpEval> = Vec::new();
        let (lane_limit, _, _, _) = engine_datapath(candidate.mode);
        for simd in dedup_sorted(simd_pool) {
            for pe in dedup_sorted(pe_pool.clone()) {
                for tap in dedup_sorted(tap_pool.clone()) {
                    if simd == 0 || pe == 0 || tap == 0 {
                        continue;
                    }
                    // The fold is the *physical* lane count, so it may never
                    // exceed what the selected architecture instantiates.
                    if u64::from(simd) * u64::from(pe) * u64::from(tap) > lane_limit {
                        continue;
                    }
                    let eval = self.eval_fold(info, candidate, FoldConfig { simd, pe, tap })?;
                    options.push(eval);
                }
            }
        }
        pareto_reduce(&mut options);
        options.sort_by_key(|e| (e.cycles, e.lanes, e.fold.simd, e.fold.pe, e.fold.tap));
        Ok(options)
    }

    fn eval_fold(
        &self,
        info: &OpInfo,
        candidate: &GlobalCandidate,
        fold: FoldConfig,
    ) -> Result<OpEval, RejectReason> {
        let lanes = fold.lanes();
        let (dsp_mac, multipliers_per_lane) = if info.is_conv {
            let mapping = self
                .target
                .map_mac(info.act_bits, info.weight_bits, MacLanes(lanes))
                .map_err(|e| RejectReason::IllegalFold {
                    op: info.name.clone(),
                    detail: e.to_string(),
                })?;
            (mapping.dsp48.get() as u32, mapping.multipliers_per_lane)
        } else {
            (0, 1)
        };
        let requant_lanes = if info.is_conv { fold.pe } else { fold.simd };
        let dsp_requant = if info.requant.is_some() {
            // The wider operand of the requantization multiply is the
            // accumulator, so a 32-bit accumulator costs two DSP48 per lane,
            // not one.
            self.target
                .map_requant(
                    MacLanes(requant_lanes.max(1) as u64),
                    engine_datapath(candidate.mode).2,
                )
                .get() as u32
        } else {
            0
        };
        let cost = self.op_cost(info, candidate, fold, multipliers_per_lane);
        let cycles = cost.total_cycles();
        let (buffers, rom, standalone_ram) = self.op_buffers(info, candidate)?;
        Ok(OpEval {
            fold,
            lanes,
            dsp_mac,
            dsp_requant,
            rom,
            standalone_ram,
            cost,
            cycles,
            buffers,
        })
    }

    /// Derive the step's MAC count and cycle cost from the generated structure.
    ///
    /// `macs` is the true product count the engine performs for the whole frame
    /// (halo re-reads add *reads*, not MACs); `lane_iterations` is how many
    /// sweeps of the lane array those products need, times the partial-product
    /// passes a widened activation costs. Tiling does not change either — it
    /// adds per-tile setup cycles and changes the buffers, which is exactly what
    /// the trade-off between tile size and RAM should be.
    fn op_cost(
        &self,
        info: &OpInfo,
        candidate: &GlobalCandidate,
        fold: FoldConfig,
        multipliers_per_lane: u64,
    ) -> StepCost {
        let util = self.policy.search.utilisation_percent.max(1);
        let lanes = fold.lanes().max(1);
        let sweeps = if info.is_conv {
            let tap_rounds = div_ceil_u64(info.taps() as u64, fold.tap.max(1) as u64);
            let (ic_rounds, oc_rounds) = if info.kind == OpKind::DepthwiseConv2d {
                (
                    div_ceil_u64(info.out_channels as u64, fold.simd.max(1) as u64),
                    1,
                )
            } else {
                (
                    div_ceil_u64(info.channel_lanes as u64, fold.simd.max(1) as u64),
                    div_ceil_u64(info.out_channels as u64, fold.pe.max(1) as u64),
                )
            };
            info.out_pixels()
                .saturating_mul(oc_rounds)
                .saturating_mul(ic_rounds)
                .saturating_mul(tap_rounds)
        } else {
            // An elementwise step issues one memory read per input operand per
            // output element, on the engine's single read port. A residual `add`
            // reads its trunk *and* its branch, so it costs two issue beats per
            // element while an upsample or a requantize costs one.
            div_ceil_u64(info.out_elems, lanes).saturating_mul(info.inputs.len().max(1) as u64)
        };
        let lane_iterations = sweeps.saturating_mul(multipliers_per_lane.max(1));
        let tiles = match candidate.granularity {
            Granularity::Frame => 1,
            Granularity::Row => {
                let h = candidate.stripe_height.unwrap_or(1).max(1) as u64;
                div_ceil_u64(info.out_spatial[0].max(1) as u64, h).max(1)
            }
            Granularity::Tile => {
                let (th, tw) = candidate.effective_tile(info);
                div_ceil_u64(info.out_spatial[0].max(1) as u64, th.max(1) as u64)
                    .max(1)
                    .saturating_mul(
                        div_ceil_u64(info.out_spatial[1].max(1) as u64, tw.max(1) as u64).max(1),
                    )
            }
        };
        StepCost {
            lanes,
            lane_iterations,
            macs: info.macs(),
            setup_cycles: tiles.saturating_mul(TILE_SETUP_CYCLES),
            utilisation_percent: util,
        }
    }

    fn op_buffers(
        &self,
        info: &OpInfo,
        candidate: &GlobalCandidate,
    ) -> Result<(Vec<BufferBinding>, RomWords, u32), RejectReason> {
        // Double buffering exists to overlap a DMA with compute. The RAM-backed
        // shared engine has no overlapped DMA, so it is single-buffered; only
        // the streaming organisation may pay for ping-pong storage.
        let overlap = matches!(candidate.mode, ExecutionMode::Streaming);
        let ping_pong = if overlap && self.policy.memory.allow_ping_pong {
            self.policy.memory.max_ping_pong_buffers.max(1)
        } else {
            1
        };
        let (_, act_word_bits, acc_word_bits, weight_word_bits) = engine_datapath(candidate.mode);
        let act_bits = u64::from(act_word_bits);
        let mut buffers = Vec::new();
        let mut rom = RomWords::default();

        let make = |name: String,
                    placement: MemoryPlacement,
                    region: BufferRegion,
                    tensor: Option<TensorId>,
                    width_bits: u64,
                    depth: u64,
                    writes: u64,
                    producers: Vec<String>,
                    consumers: Vec<String>|
         -> Result<BufferBinding, RejectReason> {
            let width_bits = width_bits.max(1);
            let depth = depth.max(1);
            // Ping-pong storage is a physical cost: the arena holds
            // `ping_pong_depth` copies, so the mapped geometry must too.
            let physical_depth = if region == BufferRegion::ActivationRam {
                depth.saturating_mul(u64::from(ping_pong))
            } else {
                depth
            };
            let ram10_blocks = if region == BufferRegion::ActivationRam
                || matches!(
                    placement,
                    MemoryPlacement::Bram | MemoryPlacement::LineBuffer
                ) {
                self.target
                    .map_buffer(
                        self.default_ram_name(),
                        BufferRequest {
                            width_bits,
                            depth: physical_depth,
                            read_ports: 1,
                            write_ports: writes,
                        },
                    )
                    .map_err(|e| RejectReason::IllegalFold {
                        op: info.name.clone(),
                        detail: e.to_string(),
                    })?
                    .blocks
                    .get()
            } else {
                0
            };
            Ok(BufferBinding {
                name,
                placement,
                region,
                // Assigned once the whole buffer table is known; the planner
                // lays out each arena deterministically.
                base_offset_bytes: 0,
                tensor,
                width_bits,
                depth,
                ram10_blocks,
                // Whole bytes of storage for the buffer's words. A 17-bit
                // multiplier word is three bytes, so the span is the byte
                // footprint, which is what keeps the region's offsets
                // comparable and non-overlapping.
                bytes: width_bits.div_ceil(8).saturating_mul(physical_depth),
                read_ports: 1,
                write_ports: writes,
                ping_pong_depth: ping_pong,
                producers,
                consumers,
            })
        };

        if info.is_conv {
            // Weights are codes, not bytes: the depth is the code count and the
            // width is the width of one *stored* entry, which is the engine's
            // weight word — the manifest's `i4-x2` packing is unpacked to one
            // signed code per entry before it reaches the ROM, so a 4-bit
            // tensor costs one 8-bit entry per code, not half of one. Every
            // layer's codes are a *slice* of one pooled ROM, so the offsets
            // recorded here are the layer's code offsets inside that ROM.
            let weight_buf = make(
                format!("{}.w", info.name),
                candidate.weight_placement,
                BufferRegion::WeightRom,
                info.weights.as_ref().map(|w| w.id),
                u64::from(weight_word_bits),
                info.weight_elems.max(1),
                0,
                vec![],
                vec![info.name.clone()],
            )?;
            rom.weights = rom.weights.saturating_add(weight_buf.depth);
            buffers.push(weight_buf);
        }

        // Bias slots: the generated `b_rom` reserves one accumulator-width word
        // per output channel of *every* step, whether or not the layer carries a
        // bias tensor, so the per-step base index and the fill order cannot
        // drift apart. An elementwise or upsample step therefore reserves its
        // channels as zero words, and the pool's depth is the sum over the
        // whole schedule rather than over the biased convolutions only.
        if candidate.mode == ExecutionMode::SharedTile {
            rom.bias = rom.bias.saturating_add(u64::from(info.out_channels).max(1));
            buffers.push(make(
                format!("{}.b", info.name),
                MemoryPlacement::Bram,
                BufferRegion::BiasRom,
                info.bias.as_ref().map(|b| b.id),
                u64::from(acc_word_bits),
                u64::from(info.out_channels).max(1),
                0,
                vec![],
                vec![info.name.clone()],
            )?);
        } else if let Some(bias) = &info.bias {
            // The legacy pointwise envelope emits no bias ROM; a bias tensor is
            // rejected by the backend, so this arm only records the slot for a
            // plan a backend will refuse with its own message.
            buffers.push(make(
                format!("{}.b", info.name),
                MemoryPlacement::Bram,
                BufferRegion::BiasRom,
                Some(bias.id),
                u64::from(acc_word_bits),
                info.bias_elems.max(1),
                0,
                vec![],
                vec![info.name.clone()],
            )?);
        }

        // The requantization multipliers are a third pooled ROM: one 17-bit
        // `M0` per output channel of every step that requantizes.
        let requant_entries = info.requant_entries();
        if requant_entries > 0 {
            if candidate.mode == ExecutionMode::SharedTile {
                rom.multipliers = rom.multipliers.saturating_add(requant_entries);
            }
            buffers.push(make(
                format!("{}.m0", info.name),
                MemoryPlacement::Bram,
                BufferRegion::MultiplierRom,
                None,
                REQUANT_MULTIPLIER_BITS,
                requant_entries,
                0,
                vec![],
                vec![info.name.clone()],
            )?);
        }

        // A graph input is staged from its DDR frame buffer into activation RAM,
        // because the shared engine never reads DDR directly. The staged frame
        // is budgeted like any other activation, not allocated silently. The
        // legacy envelope streams its input instead, so it needs no staging.
        let input_id = info.input_tensor();
        if candidate.mode == ExecutionMode::SharedTile && self.graph_inputs.contains(&input_id) {
            if let Some(t) = self.input.tensor(input_id) {
                let port = format!("in.{}", t.name);
                let consumers: Vec<String> = self
                    .ops
                    .iter()
                    .filter(|o| o.inputs.contains(&input_id))
                    .map(|o| o.name.clone())
                    .collect();
                buffers.push(make(
                    format!("stage.in.{}", t.name),
                    MemoryPlacement::Bram,
                    BufferRegion::ActivationRam,
                    Some(t.id),
                    act_bits,
                    info.in_spatial[0]
                        .max(1)
                        .saturating_mul(info.in_spatial[1].max(1))
                        .saturating_mul(info.in_channels) as u64,
                    1,
                    vec![port],
                    consumers,
                )?);
            }
        }

        // There is no per-step window or line buffer. The shared engine's FSM
        // computes its activation addresses from its own counters and reads the
        // frame-resident arena directly (the padded border is generated from the
        // tap index), so charging a line buffer would invent storage the
        // generated structure does not contain. The legacy envelope is a 1x1
        // pointwise dot, which has no window either.

        // The activation this operator writes, in the engine's activation word
        // width — not in the manifest code width. The residual `add` alone makes
        // that difference real: nine-bit sums need a 16-bit word. The shared
        // engine is frame-sequential, so its buffers hold the whole frame; the
        // legacy envelope follows its granularity.
        let depth = if candidate.mode == ExecutionMode::SharedTile {
            info.out_elems
        } else {
            let rows = match candidate.granularity {
                Granularity::Frame => info.out_spatial[0].max(1) as u64,
                Granularity::Row => 1,
                Granularity::Tile => candidate.tile.map(|t| t.height.max(1) as u64).unwrap_or(1),
            };
            (rows * info.out_spatial[1].max(1) as u64)
                .saturating_mul(info.out_channels as u64)
                .max(1)
        };
        buffers.push(make(
            format!("onchip.{}", info.name),
            candidate.intermediate_placement,
            BufferRegion::ActivationRam,
            info.outputs.first().copied(),
            act_bits,
            depth.max(1),
            1,
            vec![info.name.clone()],
            vec![],
        )?);

        let standalone_ram: u32 = buffers
            .iter()
            .filter(|b| {
                b.region == BufferRegion::ActivationRam
                    || b.region == BufferRegion::WeightRom
                    || b.region == BufferRegion::BiasRom
                    || b.region == BufferRegion::MultiplierRom
            })
            .map(|b| b.ram10_blocks as u32)
            .sum();
        Ok((buffers, rom, standalone_ram))
    }

    /// The physical mapping of the organisation's pooled memories.
    fn pool_model(&self, mode: ExecutionMode) -> PoolModel<'_> {
        let (_, act_word_bits, acc_word_bits, weight_word_bits) = engine_datapath(mode);
        PoolModel {
            target: self.target,
            primitive: self.default_ram_name(),
            activation_bits: u64::from(act_word_bits),
            weight_bits: u64::from(weight_word_bits),
            accumulator_bits: u64::from(acc_word_bits),
            // The legacy pointwise envelope streams its activations through
            // registers/FIFOs and keeps only its weights in RAM, so its
            // activation footprint is genuinely zero; claiming an arena for it
            // would overstate the generated structure.
            keeps_arena: mode == ExecutionMode::SharedTile,
            keeps_bias_rom: mode == ExecutionMode::SharedTile,
        }
    }

    /// Lower bound of the pooled RAM10 figure for a *prefix* of the schedule.
    ///
    /// The pruning bound must never exceed the leaf's exact value, or a
    /// feasible assignment would be cut. Every pool is monotone in its word
    /// count and the arena is not yet known at all (only its liveness at the
    /// leaf is), so mapping the prefix's ROM words with an empty arena is
    /// sound and tight enough to prune the absurd.
    fn ram_lower_bound(&self, mode: ExecutionMode, rom: RomWords) -> Result<u64, RejectReason> {
        self.pool_model(mode).pooled(rom, 0)
    }

    /// Physical RAM10 blocks the chosen plan really needs.
    ///
    /// Two facts decide this figure, and neither is a per-layer rounding:
    ///
    /// * The engine's read-only storage is **pooled**. It emits one weight ROM,
    ///   one bias ROM and one requantization ROM, each holding every step's
    ///   slice and each read through the one memory port the sequential engine
    ///   has. Mapping the pool once — instead of mapping each layer's slice
    ///   separately and adding the roundings — is what the hardware does, and
    ///   on a 29-layer model it is the difference between 49 blocks of weight
    ///   ROM and a block per layer.
    /// * The activation arena is a *lifetime* problem, not a sum: the
    ///   step-major sequential engine only has to keep the tensors that are
    ///   still live when it runs, so the arena is sized from the lifetime
    ///   allocation's high water mark, which is at least the peak live set and
    ///   can be larger.
    ///   Its span is therefore mapped once, at the engine's 16-bit word.
    ///
    /// Every pool is mapped at its own word width, because each is a separate
    /// physical memory.
    fn exact_ram_blocks(
        &self,
        candidate: &GlobalCandidate,
        chosen: &[OpEval],
    ) -> Result<u64, RejectReason> {
        let mut rom = RomWords::default();
        for eval in chosen {
            rom = rom.plus(eval.rom);
        }
        let arena_words = match candidate.mode {
            ExecutionMode::SharedTile => self.arena_allocation(chosen).1,
            _ => 0,
        };
        self.pool_model(candidate.mode).pooled(rom, arena_words)
    }

    /// Live interval of every activation tensor, in step indices.
    ///
    /// A tensor is live from the step that produces it to the step that last
    /// consumes it (the graph inputs are live from the start, a tensor with no
    /// consumer — a graph output — until the end of the schedule). Weight and
    /// bias tensors are resident ROM, not activations, so they are excluded
    /// here; they are counted through [`RomWords`] instead.
    fn live_intervals(&self) -> (BTreeMap<TensorId, usize>, BTreeMap<TensorId, usize>) {
        let mut resident: BTreeSet<TensorId> = BTreeSet::new();
        for op in &self.ops {
            if let Some(w) = &op.weights {
                resident.insert(w.id);
            }
            if let Some(b) = &op.bias {
                resident.insert(b.id);
            }
        }
        let mut first_step: BTreeMap<TensorId, usize> = BTreeMap::new();
        let mut last_step: BTreeMap<TensorId, usize> = BTreeMap::new();
        for (index, op) in self.ops.iter().enumerate() {
            for output in &op.outputs {
                first_step.entry(*output).or_insert(index);
            }
            for input in &op.inputs {
                last_step.insert(*input, index);
            }
        }
        for id in &self.graph_inputs {
            first_step.entry(*id).or_insert(0);
        }
        for id in resident {
            first_step.remove(&id);
            last_step.remove(&id);
        }
        (first_step, last_step)
    }

    /// Lifetime allocation of the activation arena for a chosen assignment.
    ///
    /// The ranges are the plan's `ActivationRam` buffers, one per tensor they
    /// hold, ordered by name so the placement is reproducible. The offsets come
    /// from [`crate::layout::allocate_arena`] — the same function the backend
    /// calls — so `engine.arena_words` is the extent the structure realizes,
    /// not a bound on it, and the plan carries the resulting offsets.
    fn arena_allocation(&self, chosen: &[OpEval]) -> (BTreeMap<String, u64>, u64) {
        let (first_step, last_step) = self.live_intervals();
        let final_step = self.ops.len().saturating_sub(1);

        // One range per tensor: two buffers that back the same tensor are the
        // same storage, so they share its offset.
        let mut per_tensor: BTreeMap<TensorId, (String, u64)> = BTreeMap::new();
        for eval in chosen {
            for buffer in &eval.buffers {
                if buffer.region != BufferRegion::ActivationRam {
                    continue;
                }
                let id = buffer.tensor.unwrap_or(TensorId(u32::MAX));
                per_tensor
                    .entry(id)
                    .or_insert_with(|| (buffer.name.clone(), buffer.depth.max(1)));
            }
        }
        let mut entries: Vec<(TensorId, String, u64)> = per_tensor
            .into_iter()
            .map(|(id, (name, words))| (id, name, words))
            .collect();
        entries.sort_by(|a, b| a.1.cmp(&b.1));
        let ranges: Vec<crate::layout::ArenaRange> = entries
            .iter()
            .map(|(id, _, words)| crate::layout::ArenaRange {
                start: first_step.get(id).copied().unwrap_or(0) as u64,
                end: last_step.get(id).copied().unwrap_or(final_step) as u64,
                words: *words,
            })
            .collect();
        let (offsets, high) = crate::layout::allocate_arena(&ranges);
        let mut by_name = BTreeMap::new();
        for ((_, name, _), offset) in entries.iter().zip(offsets) {
            by_name.insert(name.clone(), offset);
        }
        (by_name, high)
    }

    fn default_ram_name(&self) -> &str {
        self.target
            .default_ram_primitive()
            .map(|p| p.name.as_str())
            .unwrap_or("ram10")
    }

    fn compute_ddr_traffic(&self, candidate: &GlobalCandidate) -> (u64, u64) {
        let mut read: u64 = 0;
        let mut write: u64 = 0;

        if candidate.weight_placement == MemoryPlacement::Ddr {
            for op in &self.ops {
                if let Some(w) = &op.weights {
                    read = read.saturating_add(w.bytes());
                }
                if let Some(b) = &op.bias {
                    read = read.saturating_add(b.bytes());
                }
            }
        }

        let fused: BTreeSet<TensorId> =
            legal_fusions(self.input, &self.ops, self.policy, candidate)
                .into_iter()
                .flat_map(|(_, ids)| ids)
                .collect();

        for tensor in &self.tensors {
            let is_input = self.graph_inputs.contains(&tensor.id);
            let is_output = self.graph_outputs.contains(&tensor.id);
            if !is_input && !is_output && fused.contains(&tensor.id) {
                continue;
            }
            let bytes = tensor.bytes();
            if is_input {
                read =
                    read.saturating_add(self.read_bytes_by_consumers(tensor.id, bytes, candidate));
                continue;
            }
            if is_output {
                write = write.saturating_add(bytes);
                continue;
            }
            if candidate.intermediate_placement == MemoryPlacement::Ddr {
                write = write.saturating_add(bytes);
                read =
                    read.saturating_add(self.read_bytes_by_consumers(tensor.id, bytes, candidate));
            }
        }
        (read, write)
    }

    fn read_bytes_by_consumers(
        &self,
        tensor: TensorId,
        bytes: u64,
        candidate: &GlobalCandidate,
    ) -> u64 {
        let mut total = 0u64;
        for op in &self.ops {
            if op.inputs.contains(&tensor) {
                total = total.saturating_add(halo_read_bytes(op, bytes, candidate));
            }
        }
        total
    }
}

/// Legal producer/consumer fusions for a candidate.
///
/// A fusion is legal only when every one of these holds, and each is checked
/// rather than assumed:
///
/// * the producer value has a single consumer (a fan-out value is needed
///   elsewhere, so it cannot be swallowed);
/// * the producer and consumer run on the *same* engine — a per-operator
///   streaming pipeline cannot fuse across two engines;
/// * the intermediate is not a graph output (the graph boundary must be
///   observable);
/// * the policy allows the fusion kind and forbids neither a fan-out value nor
///   a DDR round trip that this pair would cross.
fn legal_fusions(
    input: &PlanInput,
    ops: &[OpInfo],
    policy: &Policy,
    candidate: &GlobalCandidate,
) -> Vec<(FusionDecision, Vec<TensorId>)> {
    let mut out = Vec::new();
    if !candidate.fusion_enabled {
        return out;
    }
    // Only the shared engine hosts several operators; the legacy envelope gives
    // every operator its own engine, so there is nothing to fuse across.
    if candidate.mode != ExecutionMode::SharedTile {
        return out;
    }
    let forbid_ddr = policy
        .fusion
        .forbid
        .contains(&FusionBoundary::CrossDdrBoundary);
    for i in 0..ops.len().saturating_sub(1) {
        let producer = &ops[i];
        let consumer = &ops[i + 1];
        let intermediate = producer.outputs[0];
        // A fan-out value must survive for its other consumers; the policy's
        // `fanout_value` boundary states the same rule explicitly, so a plan
        // never fuses a value another operator still reads.
        if input.fanout(intermediate) > 1 {
            continue;
        }
        if input.outputs.contains(&intermediate) {
            continue;
        }
        if forbid_ddr && candidate.intermediate_placement == MemoryPlacement::Ddr {
            continue;
        }
        let kind = match (producer.kind, consumer.kind) {
            (OpKind::DepthwiseConv2d, OpKind::PointwiseConv2d)
                if policy
                    .fusion
                    .allow
                    .contains(&FusionKind::DepthwiseRequantPointwise) =>
            {
                FusionKind::DepthwiseRequantPointwise
            }
            _ => continue,
        };
        let bytes = input
            .tensor(intermediate)
            .map(|t| t.shape.bytes(t.elem))
            .unwrap_or(0);
        // Fusion saves DDR bytes only when the intermediate would otherwise be
        // written to and read back from DDR; an on-chip intermediate saves
        // on-chip traffic instead, which this field does not claim.
        let saved = if candidate.intermediate_placement == MemoryPlacement::Ddr {
            bytes.saturating_mul(2)
        } else {
            0
        };
        out.push((
            FusionDecision {
                producer: producer.name.clone(),
                consumer: consumer.name.clone(),
                kind,
                saved_ddr_bytes: saved,
            },
            vec![intermediate],
        ));
    }
    out
}

fn halo_read_bytes(consumer: &OpInfo, bytes: u64, candidate: &GlobalCandidate) -> u64 {
    let tiled = consumer.is_conv
        && matches!(
            candidate.mode,
            ExecutionMode::SharedTile | ExecutionMode::Hybrid
        );
    if !tiled {
        return bytes;
    }
    let (th, tw) = candidate.effective_tile(consumer);
    let oh = consumer.out_spatial[0].max(1);
    let ow = consumer.out_spatial[1].max(1);
    let in_h = th.min(oh);
    let in_w = tw.min(ow);
    let halo_h = (in_h + consumer.kernel[0].saturating_sub(1)) as u64;
    let halo_w = (in_w + consumer.kernel[1].saturating_sub(1)) as u64;
    let denom = in_h as u64 * in_w as u64;
    bytes
        .saturating_mul(halo_h)
        .saturating_mul(halo_w)
        .div_ceil(denom.max(1))
}

// ---------------------------------------------------------------------------
// Graph lowering helpers
// ---------------------------------------------------------------------------

fn build_op_infos(input: &PlanInput) -> Result<Vec<OpInfo>, PlannerError> {
    let tensor = |id: TensorId| -> Result<&TensorDesc, PlannerError> {
        input.tensor(id).ok_or_else(|| PlannerError::InvalidInput {
            detail: format!("unknown tensor {id}"),
        })
    };
    let mut out = Vec::with_capacity(input.ops.len());
    for op in &input.ops {
        let is_conv = op.kind.is_convolution();
        let input_desc = tensor(op.inputs[0])?;
        let output_desc = tensor(op.outputs[0])?;
        let mut kernel = [1u32, 1];
        let mut groups = 1u32;
        let mut stride = [1u32, 1];
        let mut requant = None;
        let mut activation = Activation::identity(-128, 127);
        let mut weights = None;
        let mut bias = None;
        let mut padding = None;
        let mut upsample_scale = None;
        match &op.detail {
            OpDetail::Conv(detail) if is_conv => {
                let w = tensor(detail.weights)?;
                kernel = [w.shape.h, w.shape.w];
                groups = detail.groups;
                stride = detail.stride;
                requant = Some(detail.requant.clone());
                activation = detail.activation;
                padding = Some(resolve_padding(
                    [input_desc.shape.h, input_desc.shape.w],
                    kernel,
                    stride,
                    detail.padding,
                ));
                weights = Some(TensorRef {
                    id: w.id,
                    name: w.name.clone(),
                    shape: w.shape,
                    elem: w.elem,
                });
                bias = match detail.bias {
                    Some(id) => {
                        let b = tensor(id)?;
                        Some(TensorRef {
                            id: b.id,
                            name: b.name.clone(),
                            shape: b.shape,
                            elem: b.elem,
                        })
                    }
                    None => None,
                };
            }
            OpDetail::Elementwise(detail) => {
                requant = detail.requant.clone();
                activation = detail.activation;
            }
            OpDetail::Upsample(detail) => {
                upsample_scale = Some(detail.scale);
            }
            _ => {
                return Err(PlannerError::InvalidInput {
                    detail: format!("operator {:?}: detail does not match kind", op.name),
                });
            }
        }
        let channel_lanes = if op.kind == OpKind::DepthwiseConv2d || !is_conv {
            output_desc.shape.c
        } else {
            (input_desc.shape.c / groups).max(1)
        };
        // The reduction length per output element. Depthwise convolutions have
        // exactly one input channel per group, so their `channel_lanes` is the
        // *output* channel fold and must not be used as a reduction length.
        let reduction = if !is_conv {
            1
        } else if op.kind == OpKind::DepthwiseConv2d {
            kernel[0].max(1) * kernel[1].max(1)
        } else {
            channel_lanes.max(1) * kernel[0].max(1) * kernel[1].max(1)
        };
        let weight_elems = weights.as_ref().map(|w| w.shape.elements()).unwrap_or(0);
        let weight_elem = weights.as_ref().map(|w| w.elem).unwrap_or(ElemType::I8);
        let weight_bytes = weights.as_ref().map(|w| w.bytes()).unwrap_or(0);
        let bias_elems = bias.as_ref().map(|b| b.shape.elements()).unwrap_or(0);
        out.push(OpInfo {
            name: op.name.clone(),
            kind: op.kind,
            is_conv,
            in_channels: input_desc.shape.c,
            out_channels: output_desc.shape.c,
            in_spatial: [input_desc.shape.h, input_desc.shape.w],
            out_spatial: [output_desc.shape.h, output_desc.shape.w],
            kernel,
            groups,
            stride,
            channel_lanes,
            reduction,
            act_bits: activation_bits(input_desc),
            weight_bits: weight_elem.bits(),
            requant,
            activation,
            weights,
            bias,
            weight_elems,
            weight_elem,
            bias_elems,
            padding,
            upsample_scale,
            in_code_range: input_desc.code_range,
            out_code_range: output_desc.code_range,
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            out_elems: output_desc.shape.elements(),
            weight_bytes,
        });
    }
    Ok(out)
}

/// Effective activation width for the MAC path.
///
/// The exact code interval wins over the container type: a widened activation
/// stored in an `i16` word whose interval is `[-254, 254]` multiplies as nine
/// bits, and the target's 9-bit single-product path is what realizes it. An
/// `i16` with no interval is 16 bits, which no declared MAC mode supports — the
/// lowering must state the interval rather than leave the width to a guess.
fn activation_bits(tensor: &TensorDesc) -> u32 {
    match tensor.code_range {
        Some(range) => range.signed_bits(),
        None => tensor.elem.bits(),
    }
}

/// Resolve a declared padding rule into concrete input-pixel borders.
fn resolve_padding(
    in_hw: [u32; 2],
    kernel: [u32; 2],
    stride: [u32; 2],
    padding: Padding,
) -> PaddingSpec {
    let axis = |input: u32, k: u32, s: u32| -> (u32, u32) {
        let s = s.max(1);
        let out = input.div_ceil(s).max(1);
        let total = ((out - 1) * s + k).saturating_sub(input);
        (total / 2, total - total / 2)
    };
    match padding {
        Padding::Zero {
            top,
            bottom,
            left,
            right,
        } => PaddingSpec {
            top,
            bottom,
            left,
            right,
            reflect: false,
        },
        Padding::Reflect => {
            let (top, bottom) = axis(in_hw[0], kernel[0], stride[0]);
            let (left, right) = axis(in_hw[1], kernel[1], stride[1]);
            PaddingSpec {
                top,
                bottom,
                left,
                right,
                reflect: true,
            }
        }
        Padding::Same => {
            let (top, bottom) = axis(in_hw[0], kernel[0], stride[0]);
            let (left, right) = axis(in_hw[1], kernel[1], stride[1]);
            PaddingSpec {
                top,
                bottom,
                left,
                right,
                reflect: false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

fn div_ceil_u64(a: u64, b: u64) -> u64 {
    if b == 0 { 0 } else { a.div_ceil(b) }
}

fn divisors(n: u32) -> Vec<u32> {
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let mut d = 1u32;
    while d <= n / d {
        if n % d == 0 {
            out.push(d);
            if d != n / d {
                out.push(n / d);
            }
        }
        d += 1;
    }
    out.sort_unstable();
    out
}

fn dedup_sorted(mut v: Vec<u32>) -> Vec<u32> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Keep only Pareto-undominated options.
fn pareto_reduce(options: &mut Vec<OpEval>) {
    let n = options.len();
    let mut keep = vec![true; n];
    let cost = |e: &OpEval| {
        (
            e.cycles,
            e.dsp_mac + e.dsp_requant,
            u64::from(e.standalone_ram) + e.rom.total(),
        )
    };
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !keep[j] {
                continue;
            }
            let a = cost(&options[i]);
            let b = cost(&options[j]);
            let a_worse_or_equal = a.0 >= b.0 && a.1 >= b.1 && a.2 >= b.2;
            let a_strictly_worse = a.0 > b.0 || a.1 > b.1 || a.2 > b.2;
            if a_worse_or_equal && a_strictly_worse {
                keep[i] = false;
                break;
            }
        }
    }
    let mut idx = 0;
    options.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Solve the layout search and return a persisted-candidate artifact.
///
/// Infeasibility is reported through [`PlanStatus::Infeasible`] with rejection
/// reasons; only malformed inputs, targets or policies produce `Err`.
pub fn solve(
    input: &PlanInput,
    target: &Ti60Target,
    policy: &Policy,
    provenance: &ProvenanceInput,
) -> Result<LayoutArtifact, PlannerError> {
    input
        .validate()
        .map_err(|detail| PlannerError::InvalidInput { detail })?;
    target.validate()?;
    policy.validate().map_err(|e| PlannerError::Policy {
        detail: e.to_string(),
    })?;

    let mut search = Search::new(input, target, policy)?;
    search.run();

    let provenance = Provenance {
        graph_sha256: provenance.graph_sha256.clone(),
        target_sha256: provenance.target_sha256.clone(),
        policy_sha256: provenance.policy_sha256.clone(),
        compiler_version: provenance.compiler_version.clone(),
        target_name: target.name.clone(),
        target_revision: target.revision.clone(),
        policy_schema_version: policy.schema_version,
        seed: policy.search.seed,
        options: BTreeMap::new(),
    };
    let search_stats = SearchStats {
        explored: search.explored,
        expanded_nodes: search.expanded,
        max_candidates: policy.search.max_candidates,
        per_candidate_budget: search.per_candidate_budget,
        truncated_candidates: search.truncated_candidates,
        truncated: search.truncated,
    };
    let rejections = RejectionSummary {
        rejected_candidates: search.rejected_total,
        counts: search.rejection_counts.clone(),
        samples: search.rejections.clone(),
        by_mode: search.mode_summaries.clone(),
    };

    let Some((estimate, candidate, chosen)) = search.best.clone() else {
        let mut rejections = rejections;
        if rejections.rejected_candidates == 0 {
            rejections.rejected_candidates = 1;
            rejections.counts.insert("no_candidate".into(), 1);
            rejections.samples.push(RejectedCandidate {
                execution_mode: ExecutionMode::Streaming,
                granularity: Granularity::Row,
                tile: None,
                weight_placement: MemoryPlacement::Ddr,
                intermediate_placement: MemoryPlacement::Fifo,
                fusion_enabled: false,
                reason: RejectReason::NoCandidate {
                    detail: "search space is empty under this policy".into(),
                },
            });
        }
        let mut artifact = LayoutArtifact {
            format: LAYOUT_FORMAT.into(),
            schema_version: LAYOUT_SCHEMA_VERSION,
            status: PlanStatus::Infeasible,
            certainty: PlanCertainty::from_search(PlanStatus::Infeasible, search.truncated),
            provenance,
            limits: search.limits,
            objective: policy.objective.clone(),
            selection: None,
            resources: None,
            cycles: None,
            slack: None,
            tensors: search.tensors.clone(),
            schedule: Vec::new(),
            buffers: Vec::new(),
            fusions: Vec::new(),
            rejections,
            search: search_stats,
            mapping: None,
        };
        artifact.normalize();
        return Ok(artifact);
    };

    // Buffer names: a graph input is staged from its DDR frame buffer into the
    // engine's activation RAM, so the step interface names the on-chip staging
    // buffer; every other activation is the producing operator's on-chip
    // buffer. The engine never reads or writes DDR directly.
    let tensor_buffer_name = |op: &OpInfo, is_input_port: bool| -> String {
        if is_input_port {
            let id = op.input_tensor();
            if search.graph_inputs.contains(&id) {
                return search
                    .input
                    .tensor(id)
                    .map(|t| format!("stage.in.{}", t.name))
                    .unwrap_or_else(|| format!("stage.in.{}", id));
            }
            // Produced by an earlier operator.
            return search
                .ops
                .iter()
                .find(|o| o.outputs.contains(&id))
                .map(|o| format!("onchip.{}", o.name))
                .unwrap_or_else(|| format!("onchip.{}", id));
        }
        // The producing operator owns the activation it writes, whether or not
        // the graph boundary drains it afterwards.
        format!("onchip.{}", op.name)
    };

    let mut schedule = Vec::with_capacity(search.ops.len());
    let mut buffers: Vec<BufferBinding> = Vec::new();
    let mut assignments = Vec::with_capacity(search.ops.len());
    for (i, info) in search.ops.iter().enumerate() {
        let eval = &chosen[i];
        for b in &eval.buffers {
            if !buffers.iter().any(|x| x.name == b.name) {
                buffers.push(b.clone());
            }
        }
        let input_buffer = tensor_buffer_name(info, true);
        let output_buffer = tensor_buffer_name(info, false);
        // The legacy pointwise-dot kernel really does have ready/valid ports and
        // streams its activations, so that envelope keeps an explicit stream
        // buffer at each graph boundary. The shared engine has no such port, so
        // it names the on-chip buffer it reads instead of inventing a FIFO.
        let (input_buffer, output_buffer) = if matches!(
            candidate.mode,
            ExecutionMode::Streaming | ExecutionMode::Dedicated
        ) {
            let mut input_buffer = input_buffer;
            let mut output_buffer = output_buffer;
            let in_id = info.input_tensor();
            if search.graph_inputs.contains(&in_id) {
                if let Some(t) = search.input.tensor(in_id) {
                    input_buffer = format!("stream.{}.in", info.name);
                    buffers.push(stream_buffer(
                        input_buffer.clone(),
                        t,
                        vec![format!("in.{}", t.name)],
                        vec![info.name.clone()],
                    ));
                }
            }
            let out_id = info.outputs[0];
            if search.graph_outputs.contains(&out_id) {
                if let Some(t) = search.input.tensor(out_id) {
                    output_buffer = format!("stream.{}.out", info.name);
                    buffers.push(stream_buffer(
                        output_buffer.clone(),
                        t,
                        vec![info.name.clone()],
                        vec![format!("out.{}", t.name)],
                    ));
                }
            }
            (input_buffer, output_buffer)
        } else {
            (input_buffer, output_buffer)
        };
        assignments.push(OpAssignment {
            op: info.name.clone(),
            op_kind: info.kind,
            simd: eval.fold.simd,
            pe: eval.fold.pe,
            tap_parallel: eval.fold.tap,
            input_buffer: input_buffer.clone(),
            output_buffer: output_buffer.clone(),
            fused_with: Vec::new(),
        });
        let tile = match candidate.granularity {
            Granularity::Frame => None,
            _ => {
                let (th, tw) = candidate.effective_tile(info);
                Some(TileGeometry::new(tw, th))
            }
        };
        schedule.push(ScheduleStep {
            step: i as u32,
            op: info.name.clone(),
            op_kind: info.kind,
            engine: engine_name(candidate.mode, &info.name),
            mode: candidate.granularity,
            tile,
            simd: eval.fold.simd,
            pe: eval.fold.pe,
            tap_parallel: eval.fold.tap,
            in_channels: info.in_channels,
            out_channels: info.out_channels,
            in_spatial: info.in_spatial,
            out_spatial: info.out_spatial,
            kernel: info.kernel,
            groups: info.groups,
            stride: info.stride,
            padding: info.padding,
            upsample_scale: info.upsample_scale,
            in_code_range: info.in_code_range,
            out_code_range: info.out_code_range,
            requant: info.requant.clone(),
            activation: info.activation,
            weights: info.weights.clone(),
            bias: info.bias.clone(),
            inputs: info.inputs.clone(),
            outputs: info.outputs.clone(),
            input_buffer,
            output_buffer,
            estimated_cycles: eval.cycles,
            cost: eval.cost,
            fused_with: Vec::new(),
        });
    }

    // Declare the DDR frame-buffer endpoints. A Ddr buffer with no producer is
    // an input port; with no consumer it is an output sink. The activation-RAM
    // staging buffers the engine reads are already part of each step's buffer
    // set, so they were budgeted by the search.
    for id in &search.graph_inputs.clone() {
        if let Some(t) = search.input.tensor(*id) {
            let staging = format!("stage.in.{}", t.name);
            let mut port_buffer = tensor_buffer(format!("in.{}", t.name), t, MemoryPlacement::Ddr);
            port_buffer.region = BufferRegion::FrameBuffer;
            port_buffer.consumers = vec![staging];
            buffers.push(port_buffer);
        }
    }
    for id in &search.graph_outputs.clone() {
        if let Some(t) = search.input.tensor(*id) {
            let mut b = tensor_buffer(format!("out.{}", t.name), t, MemoryPlacement::Ddr);
            b.region = BufferRegion::FrameBuffer;
            b.producers = search
                .ops
                .iter()
                .filter(|o| o.outputs.contains(id))
                .map(|o| o.name.clone())
                .collect();
            buffers.push(b);
        }
    }

    let fusions: Vec<FusionDecision> = legal_fusions(search.input, &search.ops, policy, &candidate)
        .into_iter()
        .map(|(decision, _)| decision)
        .collect();

    // Record fusions on the consuming steps.
    for step in &mut schedule {
        for f in &fusions {
            if f.consumer == step.op {
                step.fused_with.push(f.producer.clone());
            }
        }
        for a in &mut assignments {
            if a.op == step.op {
                a.fused_with = step.fused_with.clone();
            }
        }
    }

    // The activation arena is allocated by tensor lifetime with the same
    // function the backend calls, so the plan carries the offsets the structure
    // realizes and `engine.arena_words` is that allocation's extent rather than
    // a bound on it. The read-only pools are a plain code-offset concatenation.
    let (arena_offsets, arena_words) = search.arena_allocation(&chosen);
    assign_arena_addresses(&mut buffers);
    for buffer in buffers
        .iter_mut()
        .filter(|b| b.region == BufferRegion::ActivationRam)
    {
        if let Some(offset) = arena_offsets.get(&buffer.name) {
            // The allocator works in words; the plan records byte offsets, so
            // the two agree with the buffer's `bytes` span.
            buffer.base_offset_bytes = offset.saturating_mul(buffer.width_bits.div_ceil(8));
        }
    }

    let (_, act_bits, acc_bits, weight_bits) = engine_datapath(candidate.mode);
    let engine = EngineConfig {
        name: artifact_engine_name(candidate.mode).into(),
        // The engine instantiates one lane array for every step, sized to the
        // widest fold; a step that folds less simply uses fewer lanes.
        simd: chosen.iter().map(|e| e.fold.simd).max().unwrap_or(1),
        pe: chosen.iter().map(|e| e.fold.pe).max().unwrap_or(1),
        tap_parallel: chosen.iter().map(|e| e.fold.tap).max().unwrap_or(1),
        activation_bits: act_bits,
        accumulator_bits: acc_bits,
        weight_bits,
        // The arena the plan budgets is the lifetime allocation's high water
        // mark, computed by the shared allocator from the schedule's live
        // ranges; the backend must materialize exactly that arena.
        arena_words,
    };
    let selection = Selection {
        execution_mode: candidate.mode,
        granularity: candidate.granularity,
        tile: candidate.tile,
        engine,
        weight_placement: candidate.weight_placement,
        intermediate_placement: candidate.intermediate_placement,
        fusion_enabled: candidate.fusion_enabled,
        ops: assignments,
    };

    let dma_cycles = search.dma_cycles_for(candidate.mode);
    let mut mac_cycles = 0u64;
    let mut setup_cycles = 0u64;
    for step in &schedule {
        mac_cycles = mac_cycles.saturating_add(step.cost.mac_cycles());
        setup_cycles = setup_cycles.saturating_add(step.cost.setup_cycles);
    }
    let cycles = CycleBreakdown::new(
        policy.search.utilisation_percent.max(1),
        mac_cycles,
        setup_cycles,
        dma_cycles,
    );
    let mut estimate = estimate;
    estimate.frame_cycles = cycles.total_cycles;
    let slack = ResourceSlack::between(&estimate, &search.limits);

    let mut artifact = LayoutArtifact {
        format: LAYOUT_FORMAT.into(),
        schema_version: LAYOUT_SCHEMA_VERSION,
        status: PlanStatus::Feasible,
        certainty: PlanCertainty::from_search(PlanStatus::Feasible, search.truncated),
        provenance,
        limits: search.limits,
        objective: policy.objective.clone(),
        selection: Some(selection),
        resources: Some(estimate),
        cycles: Some(cycles),
        slack: Some(slack),
        tensors: search.tensors.clone(),
        schedule,
        buffers,
        fusions,
        rejections,
        search: search_stats,
        mapping: None,
    };
    artifact.normalize();
    Ok(artifact)
}

/// Lay every addressable arena out deterministically.
///
/// Buffers are placed in name order inside their region, so the same plan
/// always yields the same addresses and the backend never has to invent an
/// allocation order. Stream buffers carry no address.
fn assign_arena_addresses(buffers: &mut [BufferBinding]) {
    for region in [
        BufferRegion::WeightRom,
        BufferRegion::BiasRom,
        BufferRegion::MultiplierRom,
        BufferRegion::FrameBuffer,
    ] {
        let mut indices: Vec<usize> = buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| b.region == region)
            .map(|(i, _)| i)
            .collect();
        indices.sort_by(|a, b| buffers[*a].name.cmp(&buffers[*b].name));
        let mut offset = 0u64;
        for index in indices {
            buffers[index].base_offset_bytes = offset;
            offset = offset.saturating_add(buffers[index].bytes);
        }
    }
    for buffer in buffers
        .iter_mut()
        .filter(|b| b.region == BufferRegion::Stream)
    {
        buffer.base_offset_bytes = 0;
    }
}

fn engine_name(mode: ExecutionMode, op: &str) -> String {
    match mode {
        ExecutionMode::Streaming | ExecutionMode::Dedicated => format!("stream.{op}"),
        ExecutionMode::SharedTile | ExecutionMode::Hybrid => "shared_conv0".into(),
    }
}

/// Name of the engine recorded in `Selection.engine`: a single shared engine,
/// or the pipeline that owns all streaming engines.
fn artifact_engine_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Streaming | ExecutionMode::Dedicated => "stream_pipeline",
        ExecutionMode::SharedTile | ExecutionMode::Hybrid => "shared_conv0",
    }
}

/// A streamed activation interface of one engine step.
///
/// The legacy pointwise-dot kernel has ready/valid ports, so its graph-boundary
/// activations are carried by a FIFO that the DDR frame-buffer port feeds. This
/// is a real interface of an emitted kernel, not a placeholder for an
/// architecture that does not exist: the shared engine, which has no such port,
/// never gets one.
fn stream_buffer(
    name: String,
    tensor: &TensorDesc,
    producers: Vec<String>,
    consumers: Vec<String>,
) -> BufferBinding {
    let mut binding = tensor_buffer(name, tensor, MemoryPlacement::Fifo);
    binding.region = BufferRegion::Stream;
    binding.producers = producers;
    binding.consumers = consumers;
    binding
}

/// A buffer that backs a graph tensor.
///
/// The arena is *not* inferred from the placement: a BRAM-resident activation
/// and a BRAM-resident weight both use RAM10 blocks but belong to different
/// address spaces, and the backend needs the region to place them. Callers set
/// `region` (and the engine word width) explicitly.
fn tensor_buffer(name: String, tensor: &TensorDesc, placement: MemoryPlacement) -> BufferBinding {
    let width_bits = tensor.elem.storage_bits() as u64;
    let depth = tensor.shape.elements().max(1);
    BufferBinding {
        name,
        placement,
        region: BufferRegion::FrameBuffer,
        base_offset_bytes: 0,
        tensor: Some(tensor.id),
        width_bits,
        depth,
        ram10_blocks: 0,
        bytes: width_bits.div_ceil(8).saturating_mul(depth),
        read_ports: 1,
        write_ports: 1,
        ping_pong_depth: 1,
        producers: Vec::new(),
        consumers: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{ActivationKind, ElemType, Rounding};

    fn requant() -> Requant {
        Requant {
            m0: 98_304,
            shift: 16,
            rounding: Rounding::HalfAwayFromZero,
            saturate_lo: -128,
            saturate_hi: 127,
            per_channel_multipliers: Vec::new(),
            multiplier_error_ppm: None,
        }
    }

    fn relu() -> Activation {
        Activation {
            kind: ActivationKind::Relu,
            clamp_lo: 0,
            clamp_hi: 127,
        }
    }

    /// A two-operator depthwise → pointwise graph: 16 channels, 32x32.
    fn toy_input() -> PlanInput {
        PlanInput {
            name: "toy".into(),
            tensors: vec![
                TensorDesc {
                    id: TensorId(0),
                    name: "x".into(),
                    shape: TensorShape::new(1, 16, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(1),
                    name: "dw.w".into(),
                    shape: TensorShape::new(16, 1, 3, 3),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(2),
                    name: "dw.b".into(),
                    shape: TensorShape::new(16, 1, 1, 1),
                    elem: ElemType::I32,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(3),
                    name: "dw.out".into(),
                    shape: TensorShape::new(1, 16, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(4),
                    name: "pw.w".into(),
                    shape: TensorShape::new(32, 16, 1, 1),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(5),
                    name: "pw.b".into(),
                    shape: TensorShape::new(32, 1, 1, 1),
                    elem: ElemType::I32,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(6),
                    name: "y".into(),
                    shape: TensorShape::new(1, 32, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
            ],
            ops: vec![
                PrimitiveOp {
                    name: "dw".into(),
                    kind: OpKind::DepthwiseConv2d,
                    inputs: vec![TensorId(0)],
                    outputs: vec![TensorId(3)],
                    detail: OpDetail::Conv(ConvDetail {
                        weights: TensorId(1),
                        bias: Some(TensorId(2)),
                        groups: 16,
                        stride: [1, 1],
                        padding: Padding::Zero {
                            top: 1,
                            bottom: 1,
                            left: 1,
                            right: 1,
                        },
                        requant: requant(),
                        activation: relu(),
                    }),
                },
                PrimitiveOp {
                    name: "pw".into(),
                    kind: OpKind::PointwiseConv2d,
                    inputs: vec![TensorId(3)],
                    outputs: vec![TensorId(6)],
                    detail: OpDetail::Conv(ConvDetail {
                        weights: TensorId(4),
                        bias: Some(TensorId(5)),
                        groups: 1,
                        stride: [1, 1],
                        padding: Padding::Zero {
                            top: 0,
                            bottom: 0,
                            left: 0,
                            right: 0,
                        },
                        requant: requant(),
                        activation: relu(),
                    }),
                },
            ],
            inputs: vec![TensorId(0)],
            outputs: vec![TensorId(6)],
        }
    }

    fn toy_policy() -> Policy {
        Policy {
            format: POLICY_FORMAT.into(),
            schema_version: POLICY_SCHEMA_VERSION,
            objective: ObjectiveSpec {
                primary: ObjectiveTerm::MinimizeCycles,
                secondary: vec![ObjectiveTerm::MinimizeDdrBytes],
            },
            constraints: PolicyConstraints {
                max_frame_cycles: 10_000_000,
                require_bit_exact: true,
                allow_cpu_layer_dispatch: false,
            },
            search: SearchPolicy {
                execution_modes: vec![
                    ExecutionMode::SharedTile,
                    ExecutionMode::Streaming,
                    ExecutionMode::Hybrid,
                ],
                granularities: vec![Granularity::Frame, Granularity::Row, Granularity::Tile],
                tile_widths: vec![16, 32],
                tile_heights: vec![16, 32],
                max_candidates: 200_000,
                seed: 1,
                utilisation_percent: 85,
            },
            fusion: FusionPolicy {
                allow: vec![FusionKind::DepthwiseRequantPointwise],
                forbid: vec![FusionBoundary::FanoutValue],
            },
            memory: MemoryPolicy {
                placements: vec![
                    MemoryPlacement::LineBuffer,
                    MemoryPlacement::Bram,
                    MemoryPlacement::Ddr,
                ],
                allow_ping_pong: true,
                max_ping_pong_buffers: 2,
            },
            operator_override: vec![
                OperatorOverride {
                    pattern: "depthwise_conv2d".into(),
                    allowed_simd: vec![1, 2, 4, 8, 16],
                    allowed_pe: vec![],
                    allowed_tap_parallel: vec![1, 3, 9],
                },
                OperatorOverride {
                    pattern: "pointwise_conv2d".into(),
                    allowed_simd: vec![1, 2, 4, 8, 16],
                    allowed_pe: vec![1, 2, 4, 8, 16],
                    allowed_tap_parallel: vec![1],
                },
            ],
            hint: Hints::default(),
        }
    }

    fn provenance() -> ProvenanceInput {
        ProvenanceInput {
            graph_sha256: "graph".into(),
            target_sha256: "target".into(),
            policy_sha256: "policy".into(),
            compiler_version: "0.1.0".into(),
        }
    }

    fn reference() -> Ti60Target {
        let mut t = Ti60Target::ti60f225_reference();
        t.logic.xlr_per_engine = 2_500;
        t
    }

    #[test]
    fn feasible_small_graph_is_solved_and_verifies() {
        let input = toy_input();
        input.validate().unwrap();
        let artifact = solve(&input, &reference(), &toy_policy(), &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        artifact.validate().unwrap();
        artifact.verify().unwrap();

        let plan = artifact.execution_plan().unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.weights.len(), 4);
        assert!(plan.resources.frame_cycles <= artifact.limits.frame_cycles);
        assert!(artifact.slack.unwrap().all_non_negative());
        // Depthwise → pointwise may fuse.
        assert!(
            artifact.fusions.iter().any(|f| f.consumer == "pw"),
            "expected a depthwise→pointwise fusion decision"
        );
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let input = toy_input();
        let a = solve(&input, &reference(), &toy_policy(), &provenance()).unwrap();
        let b = solve(&input, &reference(), &toy_policy(), &provenance()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_toml_string().unwrap(), b.to_toml_string().unwrap());
    }

    #[test]
    fn dsp_budget_exhaustion_is_infeasible_not_over_budget() {
        let input = toy_input();
        let mut target = reference();
        target.reserved.dsp48 = target.budget.dsp48; // usable DSP = 0
        let artifact = solve(&input, &target, &toy_policy(), &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Infeasible);
        assert!(artifact.selection.is_none());
        assert!(artifact.rejections.rejected_candidates > 0);
        assert!(artifact.rejections.counts.contains_key("dsp_budget"));
        artifact.validate().unwrap();
    }

    #[test]
    fn impossible_cycle_budget_is_infeasible() {
        let input = toy_input();
        let mut policy = toy_policy();
        // Far below the transfer/compute floor of even the smallest plan.
        policy.constraints.max_frame_cycles = 1_000;
        let artifact = solve(&input, &reference(), &policy, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Infeasible);
        assert!(artifact.selection.is_none());
        assert!(artifact.rejections.rejected_candidates > 0);
        assert!(!artifact.rejections.counts.is_empty());
    }

    #[test]
    fn illegal_fold_override_is_rejected_not_bypassed() {
        let input = toy_input();
        let mut policy = toy_policy();
        // 3 does not divide a 16-channel depthwise fold dimension.
        policy.operator_override[0].allowed_simd = vec![3];
        let artifact = solve(&input, &reference(), &policy, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Infeasible);
        assert!(
            artifact.rejections.counts.contains_key("illegal_fold")
                || artifact.rejections.counts.contains_key("frame_cycles")
        );
    }

    #[test]
    fn policy_parses_from_documented_toml() {
        let src = r#"
format = "tessera.policy"
schema_version = 1

[objective]
primary = "minimize_cycles"
secondary = ["minimize_ddr_bytes", "minimize_ram10"]

[constraints]
max_frame_cycles = 10_000_000
require_bit_exact = true
allow_cpu_layer_dispatch = false

[search]
execution_modes = ["streaming", "shared_tile", "hybrid"]
granularities = ["row", "tile"]
tile_widths = [40, 80, 160]
tile_heights = [40, 60, 80]
max_candidates = 100_000
seed = 1
utilisation_percent = 85

[fusion]
allow = ["depthwise_requant_pointwise", "short_residual"]
forbid = ["fanout_value", "cross_ddr_boundary"]

[memory]
placements = ["line_buffer", "bram", "ddr"]
allow_ping_pong = true
max_ping_pong_buffers = 2

[[operator_override]]
pattern = "depthwise_conv2d"
allowed_simd = [1, 2, 4, 8, 16, 32]
allowed_tap_parallel = [1, 3, 9]

[[operator_override]]
pattern = "pointwise_conv2d"
allowed_simd = [1, 2, 4, 8, 16, 32]
allowed_pe = [1, 2, 4, 8, 16]
"#;
        let policy = Policy::from_toml_str(src).unwrap();
        policy.validate().unwrap();
        assert_eq!(policy.search.tile_widths, vec![40, 80, 160]);
        assert_eq!(policy.operator_override.len(), 2);
        // Round-trip.
        let text = policy.to_toml_string().unwrap();
        let again = Policy::from_toml_str(&text).unwrap();
        assert_eq!(policy, again);
    }

    #[test]
    fn unknown_policy_value_is_an_error() {
        let src = r#"
format = "tessera.policy"
schema_version = 1
[objective]
primary = "minimize_cycles"
[constraints]
max_frame_cycles = 100
[search]
execution_modes = ["streaming", "quantum"]
granularities = ["row"]
tile_widths = [8]
tile_heights = [8]
max_candidates = 10
"#;
        let err = Policy::from_toml_str(src).unwrap_err();
        assert!(err.to_string().contains("quantum"), "got: {err}");
    }

    #[test]
    fn malformed_graph_is_an_error() {
        let mut input = toy_input();
        // Declare the wrong output spatial extent for the depthwise op.
        input.tensors[3].shape = TensorShape::new(1, 16, 31, 31);
        let err = solve(&input, &reference(), &toy_policy(), &provenance()).unwrap_err();
        assert!(matches!(err, PlannerError::InvalidInput { .. }));
    }

    #[test]
    fn fold_options_respect_divisibility() {
        let input = toy_input();
        let target = reference();
        let policy = toy_policy();
        let search = Search::new(&input, &target, &policy).unwrap();
        let info = search
            .ops
            .iter()
            .find(|o| o.kind == OpKind::PointwiseConv2d)
            .unwrap();
        let candidate = GlobalCandidate {
            mode: ExecutionMode::SharedTile,
            granularity: Granularity::Tile,
            tile: Some(TileGeometry::new(16, 16)),
            stripe_height: None,
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Ddr,
            fusion_enabled: false,
        };
        let options = search.op_options(info, &candidate).unwrap();
        assert!(!options.is_empty());
        for option in &options {
            assert_eq!(info.channel_lanes % option.fold.simd, 0);
            assert_eq!(info.out_channels % option.fold.pe, 0);
            assert_eq!(info.taps() % option.fold.tap, 0);
        }
    }

    /// The bug this guards: the planner used to invent a streaming engine per
    /// operator and fake FIFO boundaries for it. A two-operator graph has no
    /// v1 streaming architecture, so the mode must be rejected with a reason
    /// naming the mode — not silently planned as if the hardware existed.
    #[test]
    fn unimplemented_modes_are_rejected_by_name() {
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::Streaming];
        policy.search.granularities = vec![Granularity::Row];
        let artifact = solve(&toy_input(), &reference(), &policy, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Infeasible);
        assert_eq!(artifact.certainty, PlanCertainty::ProvenInfeasible);
        let streaming = artifact
            .rejections
            .by_mode
            .get("streaming")
            .expect("the streaming mode must appear in the per-mode summary");
        assert_eq!(streaming.feasible, 0);
        assert!(
            streaming.counts.contains_key("unsupported_architecture"),
            "expected an explicit unsupported-architecture rejection, got {:?}",
            streaming.counts
        );
        assert!(matches!(
            streaming.first_reason,
            Some(RejectReason::UnsupportedArchitecture { .. })
        ));
        artifact.validate().unwrap();
    }

    /// The bug this guards: a nine-bit residual carried in an int8 word is
    /// silently narrowed. The declared interval must fit the container.
    #[test]
    fn a_nine_bit_interval_cannot_hide_in_an_int8_container() {
        let mut input = residual_input(ElemType::I8, Some(CodeRange::new(-254, 254)));
        let err = input.validate().unwrap_err();
        assert!(
            err.contains("do not fit its I8 container"),
            "unexpected error: {err}"
        );

        // The same interval in a container that can hold it is fine.
        input.tensors[4].elem = ElemType::I9;
        input.validate().unwrap();

        // An empty interval is never a valid range.
        input.tensors[4].code_range = Some(CodeRange::new(5, -5));
        assert!(
            input
                .validate()
                .unwrap_err()
                .contains("empty code interval")
        );

        // A container wide enough for its interval always passes.
        input.tensors[4].code_range = Some(CodeRange::new(-256, 255));
        input.validate().unwrap();
    }

    /// A multi-operator graph is planned by the shared RAM-backed engine.
    #[test]
    fn the_shared_engine_plans_a_multi_operator_graph() {
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::SharedTile];
        policy.search.granularities = vec![Granularity::Frame];
        let artifact = solve(&toy_input(), &reference(), &policy, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        assert_eq!(
            artifact.selection.as_ref().map(|s| s.granularity),
            Some(Granularity::Frame),
            "the frame-sequential engine is the only realized shared schedule"
        );
        let plan = artifact.execution_plan().unwrap();
        assert_eq!(plan.execution_mode, ExecutionMode::SharedTile);
        assert_eq!(plan.steps.len(), 2);
        // The emitter instantiates exactly one MAC lane; the plan must not claim
        // more parallelism than that, and its cycles must follow from the MACs.
        assert_eq!(plan.engine.lanes(), 1);
        assert_eq!(plan.engine.activation_bits, 16);
        for step in &plan.steps {
            assert_eq!(step.simd, 1);
            assert_eq!(step.pe, 1);
            assert_eq!(step.tap_parallel, 1);
            assert_eq!(step.cost.lanes, 1);
            assert_eq!(step.cost.lane_iterations, step.cost.macs);
            assert!(step.cost.macs > 0);
        }
        // No fabricated FIFO interface survives.
        for step in &plan.steps {
            for name in [&step.input_buffer, &step.output_buffer] {
                let binding = plan.buffers.iter().find(|b| &b.name == name).unwrap();
                assert_ne!(binding.placement, MemoryPlacement::Fifo);
                assert_ne!(binding.region, BufferRegion::Stream);
            }
        }
    }

    /// `conv -> residual add -> conv`, with the residual point written at the
    /// caller's element type.
    fn residual_input(add_elem: ElemType, add_range: Option<CodeRange>) -> PlanInput {
        let conv = |weights: TensorId, bias: TensorId| ConvDetail {
            weights,
            bias: Some(bias),
            groups: 4,
            stride: [1, 1],
            padding: Padding::Zero {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
            requant: requant(),
            activation: relu(),
        };
        PlanInput {
            name: "residual".into(),
            tensors: vec![
                TensorDesc {
                    id: TensorId(0),
                    name: "x".into(),
                    shape: TensorShape::new(1, 4, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(1),
                    name: "dw.w".into(),
                    shape: TensorShape::new(4, 1, 3, 3),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(2),
                    name: "dw.b".into(),
                    shape: TensorShape::new(4, 1, 1, 1),
                    elem: ElemType::I32,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(3),
                    name: "dw.out".into(),
                    shape: TensorShape::new(1, 4, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(4),
                    name: "sum".into(),
                    shape: TensorShape::new(1, 4, 8, 8),
                    elem: add_elem,
                    code_range: add_range,
                },
                TensorDesc {
                    id: TensorId(5),
                    name: "pw.w".into(),
                    shape: TensorShape::new(4, 4, 1, 1),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(6),
                    name: "pw.b".into(),
                    shape: TensorShape::new(4, 1, 1, 1),
                    elem: ElemType::I32,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(7),
                    name: "y".into(),
                    shape: TensorShape::new(1, 4, 8, 8),
                    elem: ElemType::I8,
                    code_range: None,
                },
            ],
            ops: vec![
                PrimitiveOp {
                    name: "dw".into(),
                    kind: OpKind::DepthwiseConv2d,
                    inputs: vec![TensorId(0)],
                    outputs: vec![TensorId(3)],
                    detail: OpDetail::Conv(conv(TensorId(1), TensorId(2))),
                },
                PrimitiveOp {
                    name: "add".into(),
                    kind: OpKind::Add,
                    inputs: vec![TensorId(3), TensorId(0)],
                    outputs: vec![TensorId(4)],
                    detail: OpDetail::Elementwise(ElementwiseDetail {
                        requant: None,
                        activation: Activation::identity(-256, 255),
                    }),
                },
                PrimitiveOp {
                    name: "pw".into(),
                    kind: OpKind::PointwiseConv2d,
                    inputs: vec![TensorId(4)],
                    outputs: vec![TensorId(7)],
                    detail: OpDetail::Conv(ConvDetail {
                        weights: TensorId(5),
                        bias: Some(TensorId(6)),
                        groups: 1,
                        stride: [1, 1],
                        padding: Padding::Zero {
                            top: 0,
                            bottom: 0,
                            left: 0,
                            right: 0,
                        },
                        requant: requant(),
                        activation: relu(),
                    }),
                },
            ],
            inputs: vec![TensorId(0)],
            outputs: vec![TensorId(7)],
        }
    }

    fn shared_only_policy() -> Policy {
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::SharedTile];
        policy.search.granularities = vec![Granularity::Frame];
        policy
    }

    /// The residual sum of two signed int8 codes spans -254..=254. Clamping it
    /// back into an int8 container would change the arithmetic, so the graph is
    /// rejected rather than silently narrowed.
    #[test]
    fn a_residual_add_must_not_be_clamped_to_int8() {
        let clamped = residual_input(ElemType::I8, None);
        let err = solve(&clamped, &reference(), &shared_only_policy(), &provenance()).unwrap_err();
        assert!(
            err.to_string().contains("residual"),
            "the error must explain the residual width, got: {err}"
        );

        // I9 and the wider I16 transport are both accepted.
        for (elem, range) in [
            (ElemType::I9, None),
            (ElemType::I16, Some(CodeRange::new(-254, 254))),
        ] {
            let input = residual_input(elem, range);
            input.validate().unwrap();
            let artifact = solve(&input, &reference(), &shared_only_policy(), &provenance())
                .unwrap_or_else(|e| panic!("{elem:?} must lower: {e}"));
            assert_eq!(artifact.status, PlanStatus::Feasible);
            artifact.verify().unwrap();
        }
    }

    /// A convolution that reads a widened activation multiplies at nine bits,
    /// which selects the target's single-product path: half the lanes per DSP48
    /// that the dual int8 path packs.
    #[test]
    fn a_widened_activation_selects_the_single_product_mac_path() {
        let build = |elem: ElemType, range: Option<CodeRange>| {
            let mut input = residual_input(elem, range);
            // Keep only the pointwise tail so the legacy envelope accepts it,
            // and drop the bias, which that envelope does not emit.
            input.ops.retain(|op| op.name == "pw");
            if let OpDetail::Conv(detail) = &mut input.ops[0].detail {
                detail.bias = None;
            }
            input.inputs = vec![TensorId(4)];
            input
        };
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::Streaming];
        policy.search.granularities = vec![Granularity::Frame];
        policy.operator_override = vec![OperatorOverride {
            pattern: "pointwise_conv2d".into(),
            allowed_simd: vec![4],
            allowed_pe: vec![1],
            allowed_tap_parallel: vec![1],
        }];

        let narrow = solve(
            &build(ElemType::I8, None),
            &reference(),
            &policy,
            &provenance(),
        )
        .unwrap();
        let wide = solve(
            &build(ElemType::I9, Some(CodeRange::new(-254, 254))),
            &reference(),
            &policy,
            &provenance(),
        )
        .unwrap();
        assert_eq!(narrow.status, PlanStatus::Feasible);
        assert_eq!(wide.status, PlanStatus::Feasible);
        let narrow_dsp = narrow.resources.unwrap().dsp48;
        let wide_dsp = wide.resources.unwrap().dsp48;
        assert!(
            wide_dsp > narrow_dsp,
            "four 9-bit lanes need four DSP48, four int8 lanes need two: narrow {narrow_dsp}, \
             wide {wide_dsp}"
        );
        // Four int8 lanes pack two per DSP48 (2 MAC blocks); four 9-bit lanes
        // take the single-product path (4 MAC blocks). The one requant
        // multiplier is shared by both plans and costs two DSP48, because its
        // wider operand is the 32-bit accumulator, not the 8-bit activation:
        // ceil(32 / 18) = 2.
        assert_eq!(narrow_dsp, 2 + 2);
        assert_eq!(wide_dsp, 4 + 2);
    }

    /// A wider interval has no wider multiplier: it pays for nine-bit partial
    /// products, so both the DSP count and the cycles grow with the code width.
    /// Nothing is silently truncated to int8 or int9.
    #[test]
    fn a_wide_activation_pays_for_extra_multipliers() {
        let build = |elem: ElemType, range: Option<CodeRange>| {
            let mut input = residual_input(elem, range);
            input.ops.retain(|op| op.name == "pw");
            if let OpDetail::Conv(detail) = &mut input.ops[0].detail {
                detail.bias = None;
            }
            input.inputs = vec![TensorId(4)];
            input
        };
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::Streaming];
        policy.search.granularities = vec![Granularity::Frame];
        policy.operator_override = vec![OperatorOverride {
            pattern: "pointwise_conv2d".into(),
            allowed_simd: vec![4],
            allowed_pe: vec![1],
            allowed_tap_parallel: vec![1],
        }];

        // Nine bits: one nine-bit multiplier per lane.
        let nine = solve(
            &build(ElemType::I9, Some(CodeRange::new(-254, 254))),
            &reference(),
            &policy,
            &provenance(),
        )
        .unwrap();
        // A 16-bit transport with no stated interval is a 16-bit operand, which
        // needs two partial products.
        let sixteen = solve(
            &build(ElemType::I16, None),
            &reference(),
            &policy,
            &provenance(),
        )
        .unwrap();
        assert_eq!(nine.status, PlanStatus::Feasible);
        assert_eq!(sixteen.status, PlanStatus::Feasible);
        let nine_plan = nine.execution_plan().unwrap();
        let sixteen_plan = sixteen.execution_plan().unwrap();
        // Four 9-bit lanes cost four DSP48 and four 16-bit lanes cost eight (two
        // nine-bit partial products each); the single requant multiplier, whose
        // 32-bit accumulator against a 17-bit `M0` cascades two partial
        // products, costs two DSP48 in both plans and does not double.
        assert_eq!(nine.resources.unwrap().dsp48, 4 + 2);
        assert_eq!(
            sixteen.resources.unwrap().dsp48,
            8 + 2,
            "the second partial product doubles the MAC DSPs but not the requant unit"
        );
        assert!(
            sixteen_plan.steps[0].cost.lane_iterations
                == nine_plan.steps[0].cost.lane_iterations * 2,
            "two partial products take two passes per lane"
        );
        assert_eq!(
            sixteen_plan.steps[0].cost.macs, nine_plan.steps[0].cost.macs,
            "the MAC count itself is unchanged"
        );

        // An operand width with no realizable multiplier path is an explicit
        // fold rejection, not a silent narrowing to int8: a fabric that
        // advertises only the dual int8 packing cannot execute the single
        // nine-bit product path.
        let mut int8_only = reference();
        int8_only.stream.primitive[0].valid_modes = vec![crate::target::MacMode::Int8MacDual];
        let artifact = solve(
            &build(ElemType::I9, Some(CodeRange::new(-254, 254))),
            &int8_only,
            &policy,
            &provenance(),
        )
        .unwrap();
        assert_eq!(artifact.status, PlanStatus::Infeasible);
        assert!(
            artifact.rejections.counts.contains_key("illegal_fold"),
            "an unrealizable operand width must be an explicit rejection, got {:?}",
            artifact.rejections.counts
        );

        // An interval that does not fit its own container is malformed input
        // instead, and says so: nothing may be narrowed silently.
        let mut inconsistent = build(ElemType::I16, None);
        inconsistent.tensors[4].code_range = Some(CodeRange::new(-(1 << 30), 1 << 30));
        let err = solve(&inconsistent, &reference(), &policy, &provenance()).unwrap_err();
        assert!(
            err.to_string().contains("do not fit its I16 container"),
            "unexpected error: {err}"
        );
    }

    /// Padding is data in the plan: a mirrored border and a zero border of the
    /// same size are different hardware, so both the amounts and the rule are
    /// recorded.
    #[test]
    fn padding_is_resolved_into_the_plan() {
        let artifact = solve(
            &toy_input(),
            &reference(),
            &shared_only_policy(),
            &provenance(),
        )
        .unwrap();
        let plan = artifact.execution_plan().unwrap();
        let dw = plan.steps.iter().find(|s| s.op == "dw").unwrap();
        let padding = dw
            .padding
            .expect("a 3x3 convolution must record its padding");
        assert_eq!((padding.top, padding.bottom), (1, 1));
        assert_eq!((padding.left, padding.right), (1, 1));
        assert!(!padding.reflect, "the toy depthwise zero-fills");

        // A reflected border resolves to the same geometry but is marked.
        let mut input = toy_input();
        if let OpDetail::Conv(detail) = &mut input.ops[0].detail {
            detail.padding = Padding::Reflect;
        }
        let artifact = solve(&input, &reference(), &shared_only_policy(), &provenance()).unwrap();
        let plan = artifact.execution_plan().unwrap();
        let dw = plan.steps.iter().find(|s| s.op == "dw").unwrap();
        let padding = dw.padding.unwrap();
        assert!(padding.reflect);
        assert_eq!((padding.top, padding.left), (1, 1));
    }

    /// The activation arena stores signed 16-bit words, because a residual sum
    /// needs nine bits. Sizing it at the manifest's one byte would understate
    /// the generated structure.
    #[test]
    fn activation_memory_is_sized_in_engine_words() {
        let artifact = solve(
            &toy_input(),
            &reference(),
            &shared_only_policy(),
            &provenance(),
        )
        .unwrap();
        let plan = artifact.execution_plan().unwrap();
        let onchip = plan
            .buffers
            .iter()
            .find(|b| b.name == "onchip.dw")
            .expect("the depthwise output buffer");
        assert_eq!(onchip.region, BufferRegion::ActivationRam);
        assert_eq!(
            onchip.width_bits, 16,
            "one activation element per 16-bit engine word"
        );
        assert_eq!(onchip.depth, 16 * 8 * 8, "the whole depthwise frame");
        assert!(onchip.ram10_blocks > 0);
        assert_eq!(onchip.bytes, onchip.width_bits * onchip.depth / 8);
    }

    /// A chain of `layers` plain pointwise convolutions over a 4x4 frame, each
    /// with `channels` input and output channels. Every layer's weight tile is
    /// `channels × channels` codes, so the *pooled* code count is what the
    /// weight ROM is mapped from.
    fn chain_input(layers: usize, channels: u32) -> PlanInput {
        let mut tensors = vec![TensorDesc {
            id: TensorId(0),
            name: "x".into(),
            shape: TensorShape::new(1, channels, 4, 4),
            elem: ElemType::I8,
            code_range: None,
        }];
        let mut ops = Vec::new();
        let mut next = 1u32;
        let mut prev = TensorId(0);
        for l in 0..layers {
            let w = TensorId(next);
            next += 1;
            let b = TensorId(next);
            next += 1;
            let out = TensorId(next);
            next += 1;
            tensors.push(TensorDesc {
                id: w,
                name: format!("pw{l}.w"),
                shape: TensorShape::new(channels, channels, 1, 1),
                elem: ElemType::I8,
                code_range: None,
            });
            tensors.push(TensorDesc {
                id: b,
                name: format!("pw{l}.b"),
                shape: TensorShape::new(channels, 1, 1, 1),
                elem: ElemType::I32,
                code_range: None,
            });
            tensors.push(TensorDesc {
                id: out,
                name: format!("pw{l}.out"),
                shape: TensorShape::new(1, channels, 4, 4),
                elem: ElemType::I8,
                code_range: None,
            });
            ops.push(PrimitiveOp {
                name: format!("pw{l}"),
                kind: OpKind::PointwiseConv2d,
                inputs: vec![prev],
                outputs: vec![out],
                detail: OpDetail::Conv(ConvDetail {
                    weights: w,
                    bias: Some(b),
                    groups: 1,
                    stride: [1, 1],
                    padding: Padding::Zero {
                        top: 0,
                        bottom: 0,
                        left: 0,
                        right: 0,
                    },
                    requant: requant(),
                    activation: relu(),
                }),
            });
            prev = out;
        }
        PlanInput {
            name: "chain".into(),
            tensors,
            ops,
            inputs: vec![TensorId(0)],
            outputs: vec![prev],
        }
    }

    fn rom_blocks(target: &Ti60Target, width_bits: u64, depth: u64, read_ports: u64) -> u64 {
        target
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits,
                    depth: depth.max(1),
                    read_ports,
                    write_ports: 0,
                },
            )
            .unwrap()
            .blocks
            .get()
    }

    /// The engine emits **one** weight ROM holding every layer's codes, so the
    /// RAM10 figure is the pool's own mapping — not the sum of each layer's
    /// separately-rounded block. On a chain of small tiles that is the whole
    /// difference: eight 64-code tiles are eight blocks apart, and one block
    /// together.
    #[test]
    fn small_weight_tensors_pack_into_one_physical_rom_block() {
        let target = reference();
        let input = chain_input(8, 8);
        let artifact = solve(&input, &target, &shared_only_policy(), &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        let plan = artifact.execution_plan().unwrap();

        let word_count = |region: BufferRegion| -> u64 {
            plan.buffers
                .iter()
                .filter(|b| b.region == region)
                .map(|b| b.depth)
                .sum()
        };
        let standalone = |region: BufferRegion| -> u64 {
            plan.buffers
                .iter()
                .filter(|b| b.region == region)
                .map(|b| b.ram10_blocks)
                .sum()
        };

        let weight_words = word_count(BufferRegion::WeightRom);
        let bias_words = word_count(BufferRegion::BiasRom);
        let m0_words = word_count(BufferRegion::MultiplierRom);
        assert_eq!(weight_words, 8 * 64, "eight 8x8 tiles of one code each");
        assert!(bias_words > 0 && m0_words > 0);

        // One pooled ROM, read through the engine's single read port.
        assert_eq!(rom_blocks(&target, 8, weight_words, 1), 1);
        // Each layer mapped on its own would round up to a whole block.
        assert_eq!(
            standalone(BufferRegion::WeightRom),
            8,
            "each small tile alone needs its own block"
        );

        // The accepted figure is the pooled mapping of the four memories, not
        // the sum of the per-slice roundings.
        let arena_words = plan.engine.arena_words;
        assert!(arena_words > 0, "a shared plan budgets an activation arena");
        let expected = rom_blocks(&target, 8, weight_words, 1)
            + rom_blocks(&target, 32, bias_words, 1)
            + rom_blocks(&target, 17, m0_words, 1)
            + target
                .map_buffer(
                    "ram10",
                    BufferRequest {
                        width_bits: 16,
                        depth: arena_words,
                        read_ports: 1,
                        write_ports: 1,
                    },
                )
                .unwrap()
                .blocks
                .get();
        assert_eq!(plan.resources.ram10, expected);

        let per_slice = standalone(BufferRegion::WeightRom)
            + standalone(BufferRegion::BiasRom)
            + standalone(BufferRegion::MultiplierRom)
            + target
                .map_buffer(
                    "ram10",
                    BufferRequest {
                        width_bits: 16,
                        depth: arena_words,
                        read_ports: 1,
                        write_ports: 1,
                    },
                )
                .unwrap()
                .blocks
                .get();
        assert!(
            plan.resources.ram10 < per_slice,
            "pooling must not be a sum of per-layer roundings: pooled {}, per-slice {}",
            plan.resources.ram10,
            per_slice
        );
    }

    /// Sharing is what licenses the pooled mapping: the same codes served by
    /// one read port are one block, while the same codes behind eight
    /// independent ports (one per layer, no sharing) replicate the block eight
    /// times. The plan records the single port the sequential engine has.
    #[test]
    fn a_pooled_rom_is_only_cheap_because_one_port_serves_every_layer() {
        let target = reference();
        let shared = rom_blocks(&target, 8, 1024, 1);
        let unshared = target
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 8,
                    depth: 1024,
                    read_ports: 8,
                    write_ports: 0,
                },
            )
            .unwrap();
        assert_eq!(shared, 1);
        assert_eq!(unshared.replication, 8);
        assert_eq!(unshared.blocks.get(), 8);

        let input = chain_input(8, 8);
        let artifact = solve(&input, &target, &shared_only_policy(), &provenance()).unwrap();
        let plan = artifact.execution_plan().unwrap();
        for buffer in plan
            .buffers
            .iter()
            .filter(|b| b.region == BufferRegion::WeightRom)
        {
            assert_eq!(buffer.read_ports, 1, "one port serves every layer");
        }
    }

    /// A packed `i4-x2` weight tensor is unpacked to one signed code per entry
    /// before it reaches the ROM, so its plan buffer is 8 bits wide and as deep
    /// as the code count — not 4 bits wide. The pooled ROM depth follows the
    /// code count, which is what the emitter's `$readmemh` image holds.
    #[test]
    fn packed_i4_weights_occupy_one_stored_word_per_code() {
        let target = reference();
        let mut input = chain_input(2, 8);
        for tensor in &mut input.tensors {
            if tensor.name.ends_with(".w") {
                tensor.elem = ElemType::I4;
            }
        }
        let artifact = solve(&input, &target, &shared_only_policy(), &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        let plan = artifact.execution_plan().unwrap();

        let mut codes = 0u64;
        for buffer in plan
            .buffers
            .iter()
            .filter(|b| b.region == BufferRegion::WeightRom)
        {
            assert_eq!(
                buffer.width_bits, 8,
                "{} stores one unpacked code per entry",
                buffer.name
            );
            assert_eq!(
                buffer.depth, 64,
                "{} holds its layer's code count, not half of it",
                buffer.name
            );
            codes += buffer.depth;
        }
        assert_eq!(codes, 2 * 64);
        assert_eq!(rom_blocks(&target, 8, codes, 1), 1);
    }

    /// Every layer's codes are an offset inside its pool, and the offsets of a
    /// region tile it without a gap: the backend can turn a step's base index
    /// into a word address without re-inventing an allocation order.
    #[test]
    fn pooled_rom_offsets_tile_each_region_without_gaps() {
        let artifact = solve(
            &chain_input(8, 8),
            &reference(),
            &shared_only_policy(),
            &provenance(),
        )
        .unwrap();
        let plan = artifact.execution_plan().unwrap();
        for region in [
            BufferRegion::WeightRom,
            BufferRegion::BiasRom,
            BufferRegion::MultiplierRom,
        ] {
            let mut buffers: Vec<&crate::layout::BufferBinding> =
                plan.buffers.iter().filter(|b| b.region == region).collect();
            assert!(!buffers.is_empty(), "{region:?} is unused");
            buffers.sort_by_key(|b| b.base_offset_bytes);
            let mut cursor = 0u64;
            for buffer in buffers {
                assert_eq!(
                    buffer.base_offset_bytes, cursor,
                    "{region:?} buffer {} does not start where the previous one ended",
                    buffer.name
                );
                assert!(buffer.bytes > 0);
                cursor = cursor.saturating_add(buffer.bytes);
            }
        }
    }

    /// The shared engine reads its frame-resident arena at computed addresses,
    /// so it has no line buffer. Charging one would invent storage the
    /// generated structure does not contain.
    #[test]
    fn the_shared_engine_charges_no_line_buffer() {
        let artifact = solve(
            &toy_input(),
            &reference(),
            &shared_only_policy(),
            &provenance(),
        )
        .unwrap();
        let plan = artifact.execution_plan().unwrap();
        assert!(
            plan.buffers
                .iter()
                .all(|b| b.placement != MemoryPlacement::LineBuffer),
            "a 3x3 depthwise convolution is frame-resident, not windowed"
        );
        assert!(
            plan.buffers.iter().all(|b| !b.name.ends_with(".line")),
            "no line buffer is instantiated"
        );
    }

    /// The pruning bound must never exceed the leaf's exact figure, or a
    /// feasible assignment would be cut before it is scored. This checks the
    /// bound the DFS uses against the exact pooled mapping on a real plan.
    #[test]
    fn the_ram_pruning_bound_is_below_the_exact_pooled_figure() {
        let target = reference();
        let input = chain_input(8, 8);
        let policy = shared_only_policy();
        let search = Search::new(&input, &target, &policy).unwrap();
        let candidate = search
            .enumerate_candidates()
            .into_iter()
            .find(|c| c.mode == ExecutionMode::SharedTile)
            .expect("the shared organisation is enumerable");

        // One assignment: the first legal fold of every operator.
        let chosen: Vec<OpEval> = search
            .ops
            .iter()
            .map(|info| search.op_options(info, &candidate).unwrap()[0].clone())
            .collect();
        let exact = search.exact_ram_blocks(&candidate, &chosen).unwrap();

        let mut partial = Partial::default();
        for (info, eval) in search.ops.iter().zip(&chosen) {
            partial.add(eval, info.is_conv);
            let bound = search
                .ram_lower_bound(ExecutionMode::SharedTile, partial.rom)
                .unwrap();
            assert!(
                bound <= exact,
                "the prefix bound {bound} exceeds the exact figure {exact}"
            );
        }
        // The finished prefix covers every ROM pool, so the only gap left is
        // the activation arena — exactly the allocation's extent.
        let bound = search
            .ram_lower_bound(ExecutionMode::SharedTile, partial.rom)
            .unwrap();
        let arena = target
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 16,
                    depth: search.arena_allocation(&chosen).1,
                    read_ports: 1,
                    write_ports: 1,
                },
            )
            .unwrap()
            .blocks
            .get();
        assert_eq!(bound + arena, exact);
    }

    /// The arena the plan budgets is the extent of the lifetime allocation the
    /// plan and the backend share, and every property that has to hold does:
    /// it is at least the peak simultaneous word count (nothing live at the
    /// same time is ever placed on top of anything else), it is at most the
    /// no-reuse span, and two activations whose live ranges overlap never share
    /// a word. The figure is the structure's, not a bound on it.
    #[test]
    fn the_arena_budget_is_the_shared_allocations_extent() {
        let target = reference();
        let input = toy_input();
        let policy = shared_only_policy();
        let search = Search::new(&input, &target, &policy).unwrap();

        let artifact = solve(&input, &target, &policy, &provenance()).unwrap();
        let plan = artifact.execution_plan().unwrap();

        let candidate = search
            .enumerate_candidates()
            .into_iter()
            .find(|c| c.mode == ExecutionMode::SharedTile)
            .unwrap();
        let chosen: Vec<OpEval> = search
            .ops
            .iter()
            .map(|info| search.op_options(info, &candidate).unwrap()[0].clone())
            .collect();
        let (offsets, extent) = search.arena_allocation(&chosen);
        assert_eq!(plan.engine.arena_words, extent);
        assert!(extent > 0, "the toy graph has live activations");
        // The allocation is never smaller than the peak simultaneous word
        // count: at the peak step every live tensor is resident at once, so
        // their slices are pairwise disjoint.
        let (first_step, last_step) = search.live_intervals();
        let final_step = search.ops.len().saturating_sub(1);
        let peak = (0..search.ops.len())
            .map(|step| {
                search
                    .input
                    .tensors
                    .iter()
                    .filter(|tensor| {
                        let Some(start) = first_step.get(&tensor.id).copied() else {
                            return false;
                        };
                        let end = last_step.get(&tensor.id).copied().unwrap_or(final_step);
                        start <= step && step <= end
                    })
                    .map(|tensor| tensor.shape.elements().max(1))
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
        assert!(
            extent >= peak,
            "no allocation can be smaller than the peak live set"
        );

        let span_words: u64 = plan
            .buffers
            .iter()
            .filter(|b| b.region == BufferRegion::ActivationRam)
            .map(|b| b.depth)
            .sum();
        assert!(
            extent <= span_words,
            "the allocation {extent} must fit the no-reuse span {span_words}"
        );

        // The plan carries the allocation's offsets.
        for buffer in plan
            .buffers
            .iter()
            .filter(|b| b.region == BufferRegion::ActivationRam)
        {
            let expected = offsets
                .get(&buffer.name)
                .unwrap_or_else(|| panic!("{} was not allocated", buffer.name));
            assert_eq!(
                buffer.base_offset_bytes,
                expected * buffer.width_bits.div_ceil(8),
                "{} carries the allocated offset",
                buffer.name
            );
            assert!(
                buffer.base_offset_bytes + buffer.bytes <= extent * 2,
                "{} lies inside the arena",
                buffer.name
            );
        }
    }

    /// A residual `add` reads two operands through the engine's single read
    /// port, so it costs two issue beats per output element while an upsample
    /// costs one. The step's recorded lane iterations must say so, or the plan
    /// would claim a schedule the FSM cannot issue.
    #[test]
    fn an_elementwise_step_pays_for_each_operand_it_reads() {
        // `x` -> residual add of `x` with itself -> nearest upsample 2x.
        let input = PlanInput {
            name: "elementwise".into(),
            tensors: vec![
                TensorDesc {
                    id: TensorId(0),
                    name: "x".into(),
                    shape: TensorShape::new(1, 4, 4, 4),
                    elem: ElemType::I8,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(1),
                    name: "sum".into(),
                    shape: TensorShape::new(1, 4, 4, 4),
                    elem: ElemType::I16,
                    code_range: None,
                },
                TensorDesc {
                    id: TensorId(2),
                    name: "up".into(),
                    shape: TensorShape::new(1, 4, 8, 8),
                    elem: ElemType::I16,
                    code_range: None,
                },
            ],
            ops: vec![
                PrimitiveOp {
                    name: "res".into(),
                    kind: OpKind::Add,
                    inputs: vec![TensorId(0), TensorId(0)],
                    outputs: vec![TensorId(1)],
                    detail: OpDetail::Elementwise(ElementwiseDetail {
                        requant: Some(requant()),
                        activation: Activation::identity(-128, 127),
                    }),
                },
                PrimitiveOp {
                    name: "up".into(),
                    kind: OpKind::NearestUpsample2x,
                    inputs: vec![TensorId(1)],
                    outputs: vec![TensorId(2)],
                    detail: OpDetail::Upsample(UpsampleDetail { scale: 2 }),
                },
            ],
            inputs: vec![TensorId(0)],
            outputs: vec![TensorId(2)],
        };

        let artifact = solve(&input, &reference(), &shared_only_policy(), &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        let plan = artifact.execution_plan().unwrap();

        let add = plan.steps.iter().find(|s| s.op == "res").unwrap();
        assert_eq!(add.inputs.len(), 2);
        assert_eq!(
            add.cost.lane_iterations,
            add.cost.macs * 2,
            "two operands, two issue beats per element"
        );

        let up = plan.steps.iter().find(|s| s.op == "up").unwrap();
        assert_eq!(up.inputs.len(), 1);
        assert_eq!(
            up.cost.lane_iterations, up.cost.macs,
            "one operand, one issue beat per element"
        );
    }

    /// The node budget is shared across candidates, so every enumerated mode is
    /// actually searched even when the budget is far smaller than the space.
    #[test]
    fn a_small_budget_still_explores_every_mode() {
        let mut policy = toy_policy();
        policy.search.max_candidates = 3;
        let artifact = solve(&toy_input(), &reference(), &policy, &provenance()).unwrap();
        assert!(artifact.search.truncated);
        assert!(artifact.search.per_candidate_budget >= 1);
        for mode in [
            ExecutionMode::SharedTile,
            ExecutionMode::Streaming,
            ExecutionMode::Hybrid,
        ] {
            assert!(
                artifact.rejections.by_mode.contains_key(mode.as_str()),
                "mode {} was never explored",
                mode.as_str()
            );
        }
        // Truncation is not optimality, and the artifact says so.
        assert_ne!(artifact.certainty, PlanCertainty::Optimal);
        assert_eq!(
            artifact.certainty,
            if artifact.status == PlanStatus::Feasible {
                PlanCertainty::BestEffort
            } else {
                PlanCertainty::Inconclusive
            }
        );
    }

    /// The frame cycle total is the sum of the schedule's own derivation.
    #[test]
    fn cycles_follow_from_the_schedule() {
        let artifact = solve(
            &toy_input(),
            &reference(),
            &shared_only_policy(),
            &provenance(),
        )
        .unwrap();
        let cycles = artifact.cycles.expect("a feasible plan records its cycles");
        let mut mac = 0u64;
        let mut setup = 0u64;
        for step in &artifact.schedule {
            mac += step.cost.mac_cycles();
            setup += step.cost.setup_cycles;
            assert_eq!(step.estimated_cycles, step.cost.total_cycles());
        }
        assert_eq!(cycles.mac_cycles, mac);
        assert_eq!(cycles.setup_cycles, setup);
        assert_eq!(
            cycles.total_cycles,
            artifact.resources.unwrap().frame_cycles
        );
        assert_eq!(cycles.total_cycles, mac + setup + cycles.dma_cycles);
    }

    /// A fusion is never applied across a DDR round trip when the policy forbids
    /// it, and never in a mode whose operators run on separate engines.
    #[test]
    fn fusion_respects_its_boundaries() {
        let mut policy = shared_only_policy();
        policy.fusion.forbid = vec![FusionBoundary::CrossDdrBoundary];
        policy.memory.placements = vec![
            MemoryPlacement::Bram,
            MemoryPlacement::Ddr,
            MemoryPlacement::LineBuffer,
        ];
        let artifact = solve(&toy_input(), &reference(), &policy, &provenance()).unwrap();
        if artifact
            .selection
            .as_ref()
            .map(|s| s.intermediate_placement == MemoryPlacement::Ddr)
            .unwrap_or(false)
        {
            assert!(
                artifact.fusions.is_empty(),
                "a DDR-crossing fusion is forbidden by policy"
            );
        }

        // The legacy envelope runs one engine per operator, so nothing fuses.
        let mut streaming = toy_policy();
        streaming.search.execution_modes = vec![ExecutionMode::Streaming];
        streaming.search.granularities = vec![Granularity::Frame];
        let input = {
            let mut input = toy_input();
            input.ops.retain(|op| op.name == "pw");
            if let OpDetail::Conv(detail) = &mut input.ops[0].detail {
                detail.bias = None;
            }
            input.inputs = vec![TensorId(3)];
            input
        };
        let artifact = solve(&input, &reference(), &streaming, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        assert!(artifact.fusions.is_empty());
    }
}
