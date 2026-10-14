//! Target-aware candidate planner for the Ti60 family.
//!
//! # What this module does
//!
//! It consumes a typed, lowered primitive graph ([`PlanInput`]), a
//! [`Ti60Target`], and a user [`Policy`], and produces a [`LayoutArtifact`]:
//! the chosen execution mode, engine folding, tile geometry, buffer placement,
//! fusion decisions, a static schedule, resource and cycle estimates, slack,
//! and a summary of rejected candidates.
//!
//! # Search structure
//!
//! ```text
//! for each global candidate (mode × granularity × tile/stripe × weight
//!                           placement × intermediate placement × fusion):
//!     compute DDR traffic; reject the whole candidate if it exceeds budget
//!     build per-operator fold options (legality + Pareto reduction)
//!     branch-and-bound depth-first over operators:
//!         prune when a monotone *lower bound* of the final resource vector
//!         already exceeds DSP / RAM / cycle budgets, or when the partial
//!         cycle count already meets the incumbent (primary = cycles)
//!     keep the best complete assignment under the lexicographic objective
//! ```
//!
//! Every aggregate used for pruning is a genuine lower bound of the final
//! aggregate (sums for dedicated modes, maxima for the shared engine), so the
//! search never prunes a feasible plan. The final estimate is re-checked
//! against every budget before a plan is accepted, and the artifact re-checks
//! it again on load, so the planner never silently over-budgets.
//!
//! # Documented limits of v1
//!
//! * Enumeration + branch-and-bound only — no CP-SAT, no Pareto-frontier dump.
//!   The search is capped by `policy.search.max_candidates` (node expansions);
//!   hitting the cap sets `search.truncated` and returns the best plan found.
//! * One global execution mode and granularity per plan; per-operator folds may
//!   differ (a shared engine is sized to the widest assignment).
//! * Cycle model: integer loop-nest count
//!   `work_pixels × ceil(oc/pe) × ceil(ic_per_group/simd) × ceil(taps/tap)`
//!   scaled by the declared MAC utilisation, plus per-tile setup cycles and,
//!   for non-overlapping modes, DDR transfer cycles. Tiled granularities charge
//!   the full halo of every tile: a conservative over-estimate.
//! * Streaming/dedicated modes overlap DMA with compute; shared-tile/hybrid do
//!   not (per `docs/st_accel_architecture.md`, DDR overlap must be measured).
//! * XLR is only modelled when the target declares `[logic]` coefficients.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::layout::{
    Activation, BufferBinding, ElemType, EngineConfig, ExecutionMode, FusionBoundary,
    FusionDecision, FusionKind, Granularity, LAYOUT_FORMAT, LAYOUT_SCHEMA_VERSION, LayoutArtifact,
    MemoryPlacement, ObjectiveSpec, ObjectiveTerm, OpAssignment, OpKind, PlanLimits, PlanStatus,
    Provenance, RejectReason, RejectedCandidate, RejectionSummary, Requant, ResourceEstimate,
    ResourceSlack, ScheduleStep, SearchStats, Selection, TensorId, TensorRef, TensorShape,
    TileGeometry,
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
// Policy
// ---------------------------------------------------------------------------

/// Hard constraints declared by the user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyConstraints {
    /// Maximum frame cycles.
    pub max_frame_cycles: u64,
    /// Require bit-exact realizations. v1 only produces bit-exact plans, so
    /// `false` is accepted but grants nothing.
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
                    if input.elem != ElemType::I8 {
                        return Err(format!(
                            "operator {:?}: v1 convolution input must be I8",
                            op.name
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
    act_bits: u32,
    weight_bits: u32,
    requant: Option<Requant>,
    activation: Activation,
    weights: Option<TensorRef>,
    bias: Option<TensorRef>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpEval {
    fold: FoldConfig,
    lanes: u64,
    dsp_mac: u32,
    dsp_requant: u32,
    ram_weights: u32,
    ram_buffers: u32,
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
    weights_sum: u32,
    buf_sum: u32,
    conv_buf_max: u32,
    nonconv_buf_max: u32,
    nonconv_buf_sum: u32,
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
            self.conv_buf_max = self.conv_buf_max.max(eval.ram_buffers);
        } else {
            self.nonconv_mac_sum = self.nonconv_mac_sum.saturating_add(eval.dsp_mac);
            self.nonconv_mac_max = self.nonconv_mac_max.max(eval.dsp_mac);
            self.nonconv_rq_sum = self.nonconv_rq_sum.saturating_add(eval.dsp_requant);
            self.nonconv_rq_max = self.nonconv_rq_max.max(eval.dsp_requant);
            self.nonconv_lanes_max = self.nonconv_lanes_max.max(eval.lanes);
            self.nonconv_lanes_sum = self.nonconv_lanes_sum.saturating_add(eval.lanes);
            self.nonconv_buf_max = self.nonconv_buf_max.max(eval.ram_buffers);
            self.nonconv_buf_sum = self.nonconv_buf_sum.saturating_add(eval.ram_buffers);
        }
        self.weights_sum = self.weights_sum.saturating_add(eval.ram_weights);
        self.buf_sum = self.buf_sum.saturating_add(eval.ram_buffers);
        self.cycles_sum = self.cycles_sum.saturating_add(eval.cycles);
    }

    /// Exact `(dsp48, ram10, physical_lanes)` for `mode`. These are also valid
    /// lower bounds for a prefix of the operator list.
    fn aggregate(&self, mode: ExecutionMode) -> (u32, u32, u64) {
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
        let buffers = match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => self.buf_sum,
            ExecutionMode::SharedTile => self.conv_buf_max.max(self.nonconv_buf_max),
            ExecutionMode::Hybrid => self.conv_buf_max + self.nonconv_buf_sum,
        };
        let lanes = match mode {
            ExecutionMode::Streaming | ExecutionMode::Dedicated => {
                self.conv_lanes_max + self.nonconv_lanes_sum
            }
            ExecutionMode::SharedTile => self.conv_lanes_max.max(self.nonconv_lanes_max),
            ExecutionMode::Hybrid => self.conv_lanes_max + self.nonconv_lanes_sum,
        };
        (mac + rq, self.weights_sum + buffers, lanes)
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
    truncated: bool,
    first_reason: Option<RejectReason>,
    rejections: Vec<RejectedCandidate>,
    rejection_counts: BTreeMap<String, u64>,
    rejected_total: u64,
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
            truncated: false,
            first_reason: None,
            rejections: Vec::new(),
            rejection_counts: BTreeMap::new(),
            rejected_total: 0,
        })
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
            return vec![MemoryPlacement::Fifo];
        }
        let mut out = Vec::new();
        for p in &self.policy.memory.placements {
            if matches!(p, MemoryPlacement::Bram | MemoryPlacement::Ddr) && !out.contains(p) {
                out.push(*p);
            }
        }
        if out.is_empty() {
            out.push(MemoryPlacement::Ddr);
        }
        out
    }

    fn run(&mut self) {
        let mut modes = self.policy.search.execution_modes.clone();
        modes.dedup();
        let mut granularities = self.policy.search.granularities.clone();
        granularities.dedup();
        let weight_placements = self.weight_placements();
        let max_candidates = self.policy.search.max_candidates;

        'outer: for mode in modes {
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
                                if self.expanded >= max_candidates {
                                    self.truncated = true;
                                    break 'outer;
                                }
                                let candidate = GlobalCandidate {
                                    mode,
                                    granularity: *granularity,
                                    tile,
                                    stripe_height: stripe,
                                    weight_placement: *wp,
                                    intermediate_placement: ip,
                                    fusion_enabled,
                                };
                                self.explored += 1;
                                self.search_candidate(candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    fn search_candidate(&mut self, candidate: GlobalCandidate) {
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

        let max_candidates = self.policy.search.max_candidates;
        let mut assign = vec![0usize; self.ops.len()];
        let before = self.best.clone();
        self.dfs(
            0,
            Partial::default(),
            &mut assign,
            &candidate,
            max_candidates,
        );
        if self.best == before {
            let reason = self
                .first_reason
                .take()
                .unwrap_or(RejectReason::NoCandidate {
                    detail: "no fold assignment satisfies every budget".into(),
                });
            self.reject(&candidate, reason);
        }
    }

    fn reject(&mut self, candidate: &GlobalCandidate, reason: RejectReason) {
        self.rejected_total += 1;
        *self
            .rejection_counts
            .entry(reason.category().to_string())
            .or_insert(0) += 1;
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
        max_candidates: u64,
    ) {
        if index == self.ops.len() {
            self.leaf(partial, assign, candidate);
            return;
        }
        let n_options = self.options[index].len();
        for choice in 0..n_options {
            if self.expanded >= max_candidates {
                self.truncated = true;
                return;
            }
            self.expanded += 1;
            let eval = self.options[index][choice].clone();
            let mut next = partial;
            next.add(&eval, self.ops[index].is_conv);

            let (dsp, ram, _lanes) = next.aggregate(candidate.mode);
            if dsp as u64 > self.limits.dsp48 {
                self.note_reason(RejectReason::DspBudget {
                    used: dsp as u64,
                    budget: self.limits.dsp48,
                });
                continue;
            }
            if ram as u64 > self.limits.ram10 {
                self.note_reason(RejectReason::RamBudget {
                    used: ram as u64,
                    budget: self.limits.ram10,
                });
                continue;
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
            self.dfs(index + 1, next, assign, candidate, max_candidates);
            if self.truncated {
                return;
            }
        }
    }

    fn leaf(&mut self, partial: Partial, assign: &[usize], candidate: &GlobalCandidate) {
        let (dsp, ram, lanes) = partial.aggregate(candidate.mode);
        let overlap = matches!(candidate.mode, ExecutionMode::Streaming);
        let dma_cycles = if overlap { 0 } else { self.dma_cycles() };
        let cycles = partial.cycles_sum.saturating_add(dma_cycles);
        let engines = self.engine_count(candidate.mode);
        let xlr = self.target.estimate_xlr(lanes, engines, ram as u64).get();
        let estimate = ResourceEstimate {
            dsp48: dsp as u64,
            ram10: ram as u64,
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
            let chosen: Vec<OpEval> = (0..self.ops.len())
                .map(|i| self.options[i][assign[i]].clone())
                .collect();
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
        for simd in dedup_sorted(simd_pool) {
            for pe in dedup_sorted(pe_pool.clone()) {
                for tap in dedup_sorted(tap_pool.clone()) {
                    if simd == 0 || pe == 0 || tap == 0 {
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
        let dsp_mac = if info.is_conv {
            self.target
                .map_mac(info.act_bits, info.weight_bits, MacLanes(lanes))
                .map_err(|e| RejectReason::IllegalFold {
                    op: info.name.clone(),
                    detail: e.to_string(),
                })?
                .dsp48
                .get() as u32
        } else {
            0
        };
        let requant_lanes = if info.is_conv { fold.pe } else { fold.simd };
        let dsp_requant = if info.requant.is_some() {
            self.target
                .map_requant(MacLanes(requant_lanes.max(1) as u64))
                .get() as u32
        } else {
            0
        };
        let cycles = self.op_cycles(info, candidate, fold);
        let (buffers, ram_weights, ram_buffers) = self.op_buffers(info, candidate)?;
        Ok(OpEval {
            fold,
            lanes,
            dsp_mac,
            dsp_requant,
            ram_weights,
            ram_buffers,
            cycles,
            buffers,
        })
    }

    fn op_cycles(&self, info: &OpInfo, candidate: &GlobalCandidate, fold: FoldConfig) -> u64 {
        let util = self.policy.search.utilisation_percent.max(1) as u64;
        let raw = if info.is_conv {
            let tiled = matches!(
                candidate.mode,
                ExecutionMode::SharedTile | ExecutionMode::Hybrid
            );
            let (work_pixels, tiles) = if tiled {
                let (th, tw) = candidate.effective_tile(info);
                let oh = info.out_spatial[0].max(1) as u64;
                let ow = info.out_spatial[1].max(1) as u64;
                let tiles_h = div_ceil_u64(oh, th as u64);
                let tiles_w = div_ceil_u64(ow, tw as u64);
                let tiles = tiles_h.max(1) * tiles_w.max(1);
                let halo_h = (th.min(oh as u32) + info.kernel[0].saturating_sub(1)) as u64;
                let halo_w = (tw.min(ow as u32) + info.kernel[1].saturating_sub(1)) as u64;
                (tiles * halo_h * halo_w, tiles)
            } else {
                (
                    info.out_spatial[0].max(1) as u64 * info.out_spatial[1].max(1) as u64,
                    0,
                )
            };
            let tap_rounds = div_ceil_u64(info.taps() as u64, fold.tap as u64);
            let (ic_rounds, oc_rounds) = if info.kind == OpKind::DepthwiseConv2d {
                (div_ceil_u64(info.out_channels as u64, fold.simd as u64), 1)
            } else {
                (
                    div_ceil_u64(info.channel_lanes as u64, fold.simd as u64),
                    div_ceil_u64(info.out_channels as u64, fold.pe as u64),
                )
            };
            work_pixels
                .saturating_mul(oc_rounds)
                .saturating_mul(ic_rounds)
                .saturating_mul(tap_rounds)
                .saturating_add(tiles.saturating_mul(TILE_SETUP_CYCLES))
        } else {
            div_ceil_u64(info.out_elems, fold.lanes().max(1))
        };
        div_ceil_u64(raw.saturating_mul(100), util)
    }

    fn op_buffers(
        &self,
        info: &OpInfo,
        candidate: &GlobalCandidate,
    ) -> Result<(Vec<BufferBinding>, u32, u32), RejectReason> {
        let ping_pong = if self.policy.memory.allow_ping_pong {
            self.policy.memory.max_ping_pong_buffers.max(1)
        } else {
            1
        };
        let act_bits = info.act_bits as u64;
        let mut buffers = Vec::new();
        let mut ram_weights = 0u32;

        let make = |name: String,
                    placement: MemoryPlacement,
                    tensor: Option<TensorId>,
                    width_bits: u64,
                    depth: u64,
                    writes: u64,
                    producers: Vec<String>,
                    consumers: Vec<String>|
         -> Result<BufferBinding, RejectReason> {
            let width_bits = width_bits.max(1);
            let depth = depth.max(1);
            let ram10_blocks = if matches!(
                placement,
                MemoryPlacement::Bram | MemoryPlacement::LineBuffer
            ) {
                self.target
                    .map_buffer(
                        self.default_ram_name(),
                        BufferRequest {
                            width_bits,
                            depth,
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
                tensor,
                width_bits,
                depth,
                ram10_blocks,
                bytes: width_bits.saturating_mul(depth).div_ceil(8),
                read_ports: 1,
                write_ports: writes,
                ping_pong_depth: ping_pong,
                producers,
                consumers,
            })
        };

        if info.is_conv {
            let weight_buf = make(
                format!("{}.w", info.name),
                candidate.weight_placement,
                info.weights.as_ref().map(|w| w.id),
                info.weight_bits as u64,
                info.weight_bytes.max(1),
                0,
                vec![],
                vec![info.name.clone()],
            )?;
            ram_weights = weight_buf.ram10_blocks as u32;
            buffers.push(weight_buf);

            let tiled = matches!(
                candidate.mode,
                ExecutionMode::SharedTile | ExecutionMode::Hybrid
            );
            if tiled {
                let (th, tw) = candidate.effective_tile(info);
                let halo_h =
                    (th.min(info.out_spatial[0].max(1)) + info.kernel[0].saturating_sub(1)) as u64;
                let halo_w =
                    (tw.min(info.out_spatial[1].max(1)) + info.kernel[1].saturating_sub(1)) as u64;
                buffers.push(make(
                    format!("{}.tin", info.name),
                    candidate.intermediate_placement,
                    None,
                    info.in_channels as u64 * act_bits,
                    halo_h * halo_w,
                    1,
                    vec![],
                    vec![info.name.clone()],
                )?);
            } else if info.kernel[0] > 1 || info.kernel[1] > 1 {
                let depth = (info.kernel[0].saturating_sub(1)) as u64 * info.in_spatial[1] as u64
                    + (info.kernel[1].saturating_sub(1)) as u64;
                buffers.push(make(
                    format!("{}.line", info.name),
                    MemoryPlacement::LineBuffer,
                    None,
                    info.in_channels as u64 * act_bits,
                    depth.max(1),
                    1,
                    vec![],
                    vec![info.name.clone()],
                )?);
            }
        }

        let rows = match candidate.granularity {
            Granularity::Frame => info.out_spatial[0].max(1) as u64,
            Granularity::Row => 1,
            Granularity::Tile => candidate.tile.map(|t| t.height.max(1) as u64).unwrap_or(1),
        };
        buffers.push(make(
            format!("onchip.{}", info.name),
            candidate.intermediate_placement,
            info.outputs.first().copied(),
            info.out_channels as u64 * act_bits,
            (rows * info.out_spatial[1].max(1) as u64).max(1),
            1,
            vec![info.name.clone()],
            vec![],
        )?);

        let ram_buffers: u32 = buffers
            .iter()
            .filter(|b| b.name != format!("{}.w", info.name))
            .map(|b| b.ram10_blocks as u32)
            .sum();
        Ok((buffers, ram_weights, ram_buffers))
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
    for i in 0..ops.len().saturating_sub(1) {
        let producer = &ops[i];
        let consumer = &ops[i + 1];
        let intermediate = producer.outputs[0];
        if input.fanout(intermediate) > 1 {
            continue; // never fuse a fan-out value
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
        out.push((
            FusionDecision {
                producer: producer.name.clone(),
                consumer: consumer.name.clone(),
                kind,
                saved_ddr_bytes: bytes.saturating_mul(2),
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
        match &op.detail {
            OpDetail::Conv(detail) if is_conv => {
                let w = tensor(detail.weights)?;
                kernel = [w.shape.h, w.shape.w];
                groups = detail.groups;
                stride = detail.stride;
                requant = Some(detail.requant.clone());
                activation = detail.activation;
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
            OpDetail::Upsample(_) => {}
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
        let weight_bytes = weights.as_ref().map(|w| w.bytes()).unwrap_or(0);
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
            act_bits: input_desc.elem.bits(),
            weight_bits: weights.as_ref().map(|w| w.elem.bits()).unwrap_or(8),
            requant,
            activation,
            weights,
            bias,
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            out_elems: output_desc.shape.elements(),
            weight_bytes,
        });
    }
    Ok(out)
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
            e.ram_buffers + e.ram_weights,
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
        truncated: search.truncated,
    };
    let rejections = RejectionSummary {
        rejected_candidates: search.rejected_total,
        counts: search.rejection_counts.clone(),
        samples: search.rejections.clone(),
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
            provenance,
            limits: search.limits,
            objective: policy.objective.clone(),
            selection: None,
            resources: None,
            slack: None,
            tensors: search.tensors.clone(),
            schedule: Vec::new(),
            buffers: Vec::new(),
            fusions: Vec::new(),
            rejections,
            search: search_stats,
        };
        artifact.normalize();
        return Ok(artifact);
    };

    // Buffer names: graph I/O by tensor name; everything else per operator.
    let tensor_buffer_name = |op: &OpInfo, is_input_port: bool| -> String {
        if is_input_port {
            let id = op.input_tensor();
            if search.graph_inputs.contains(&id) {
                return search
                    .input
                    .tensor(id)
                    .map(|t| format!("in.{}", t.name))
                    .unwrap_or_else(|| format!("in.{}", id));
            }
            // Produced by an earlier operator.
            return search
                .ops
                .iter()
                .find(|o| o.outputs.contains(&id))
                .map(|o| format!("onchip.{}", o.name))
                .unwrap_or_else(|| format!("onchip.{}", id));
        }
        let id = op.outputs[0];
        if search.graph_outputs.contains(&id) {
            return search
                .input
                .tensor(id)
                .map(|t| format!("out.{}", t.name))
                .unwrap_or_else(|| format!("out.{}", id));
        }
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
        // A dedicated or streaming engine has ready/valid ports, so an
        // activation that crosses the graph boundary is carried by an on-chip
        // stream buffer. The DDR frame-buffer port is declared below as the
        // external endpoint of that stream, which is what keeps the schedule's
        // engine interface free of an implicit DMA.
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
                        Vec::new(),
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
                        Vec::new(),
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
            requant: info.requant.clone(),
            activation: info.activation,
            weights: info.weights.clone(),
            bias: info.bias.clone(),
            inputs: info.inputs.clone(),
            outputs: info.outputs.clone(),
            input_buffer,
            output_buffer,
            estimated_cycles: eval.cycles,
            fused_with: Vec::new(),
        });
    }

    // Declare graph I/O buffers so every schedule reference resolves. A Ddr
    // buffer with no producer is a frame-buffer input port; with no consumer it
    // is a frame-buffer output sink.
    for id in &search.graph_inputs.clone() {
        if let Some(t) = search.input.tensor(*id) {
            let mut b = tensor_buffer(format!("in.{}", t.name), t, MemoryPlacement::Ddr);
            b.consumers = search
                .ops
                .iter()
                .filter(|o| o.inputs.contains(id))
                .map(|o| o.name.clone())
                .collect();
            buffers.push(b);
        }
    }
    for id in &search.graph_outputs.clone() {
        if let Some(t) = search.input.tensor(*id) {
            let mut b = tensor_buffer(format!("out.{}", t.name), t, MemoryPlacement::Ddr);
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

    let slack = ResourceSlack::between(&estimate, &search.limits);
    let engine = EngineConfig {
        name: artifact_engine_name(candidate.mode).into(),
        simd: chosen.iter().map(|e| e.fold.simd).max().unwrap_or(1),
        pe: chosen.iter().map(|e| e.fold.pe).max().unwrap_or(1),
        tap_parallel: chosen.iter().map(|e| e.fold.tap).max().unwrap_or(1),
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

    let mut artifact = LayoutArtifact {
        format: LAYOUT_FORMAT.into(),
        schema_version: LAYOUT_SCHEMA_VERSION,
        status: PlanStatus::Feasible,
        provenance,
        limits: search.limits,
        objective: policy.objective.clone(),
        selection: Some(selection),
        resources: Some(estimate),
        slack: Some(slack),
        tensors: search.tensors.clone(),
        schedule,
        buffers,
        fusions,
        rejections,
        search: search_stats,
    };
    artifact.normalize();
    Ok(artifact)
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
/// The scheduler keeps the external DDR port buffers (`in.*`, `out.*`) in the
/// table as the frame-buffer endpoints, and connects each engine to them
/// through these FIFO buffers, so a step's input and output buffers describe the
/// engine interface rather than the memory the DMA talks to.
fn stream_buffer(
    name: String,
    tensor: &TensorDesc,
    producers: Vec<String>,
    consumers: Vec<String>,
) -> BufferBinding {
    let mut binding = tensor_buffer(name, tensor, MemoryPlacement::Fifo);
    binding.producers = producers;
    binding.consumers = consumers;
    binding
}

fn tensor_buffer(name: String, tensor: &TensorDesc, placement: MemoryPlacement) -> BufferBinding {
    let width_bits = tensor.elem.bits() as u64;
    let depth = tensor.shape.elements().max(1);
    BufferBinding {
        name,
        placement,
        tensor: Some(tensor.id),
        width_bits,
        depth,
        ram10_blocks: 0,
        bytes: tensor.shape.bytes(tensor.elem),
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
            rounding: Rounding::HalfUp,
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
                    shape: TensorShape::new(1, 16, 32, 32),
                    elem: ElemType::I8,
                },
                TensorDesc {
                    id: TensorId(1),
                    name: "dw.w".into(),
                    shape: TensorShape::new(16, 1, 3, 3),
                    elem: ElemType::I8,
                },
                TensorDesc {
                    id: TensorId(2),
                    name: "dw.b".into(),
                    shape: TensorShape::new(16, 1, 1, 1),
                    elem: ElemType::I32,
                },
                TensorDesc {
                    id: TensorId(3),
                    name: "dw.out".into(),
                    shape: TensorShape::new(1, 16, 32, 32),
                    elem: ElemType::I8,
                },
                TensorDesc {
                    id: TensorId(4),
                    name: "pw.w".into(),
                    shape: TensorShape::new(32, 16, 1, 1),
                    elem: ElemType::I8,
                },
                TensorDesc {
                    id: TensorId(5),
                    name: "pw.b".into(),
                    shape: TensorShape::new(32, 1, 1, 1),
                    elem: ElemType::I32,
                },
                TensorDesc {
                    id: TensorId(6),
                    name: "y".into(),
                    shape: TensorShape::new(1, 32, 32, 32),
                    elem: ElemType::I8,
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
                granularities: vec![Granularity::Row, Granularity::Tile],
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

    #[test]
    fn streaming_mode_iterates_all_ops() {
        let mut policy = toy_policy();
        policy.search.execution_modes = vec![ExecutionMode::Streaming];
        policy.search.granularities = vec![Granularity::Row];
        let artifact = solve(&toy_input(), &reference(), &policy, &provenance()).unwrap();
        assert_eq!(artifact.status, PlanStatus::Feasible);
        let plan = artifact.execution_plan().unwrap();
        assert_eq!(plan.execution_mode, ExecutionMode::Streaming);
        assert_eq!(plan.steps.len(), 2);
    }
}
