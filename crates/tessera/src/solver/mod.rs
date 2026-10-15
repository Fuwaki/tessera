//! General hardware-mapping solver.
//!
//! Where [`crate::planner`] enumerates one execution organization at a time,
//! this module solves the mapping *jointly*: how many MAC engines to
//! instantiate, which operator runs on which engine, what fold each uses, when
//! each runs, whether two engines are pipelined across a tensor, and where the
//! intermediate activations live. The problem is handed to a real constraint
//! solver (Google OR-Tools CP-SAT) as the solver-agnostic [`model`] program;
//! the assignment is then *independently* re-checked by [`check`] before
//! [`lower`] turns it into a schema-2 [`LayoutArtifact`](crate::layout::LayoutArtifact).
//!
//! The four organizations the tests exercise are not modes: they are four
//! different [`MappingBounds`] on the same input graph, and the solver derives
//! one shared engine, one wide shared engine, two pipelined shared engines or a
//! dedicated/shared split from the bounds alone.

mod backend;
pub mod candidates;
pub mod check;
pub mod dialect;
pub use dialect::{DialectSolve, solve_tessera};
pub mod encode;
pub mod lower;
pub mod model;

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::graph::Graph;
use crate::layout::{
    Activation, ActivationKind, CodeRange, ElemType, ExecutionMode, Granularity, LAYOUT_FORMAT,
    LAYOUT_SCHEMA_VERSION_V2, LayoutArtifact, LayoutError, MemoryPlacement, OpKind, PlanCertainty,
    PlanLimits, PlanStatus, Provenance, RejectReason, RejectedCandidate, RejectionSummary, Requant,
    Rounding, SearchStats, TensorId, TensorRef, TensorShape,
};
use crate::planner::{
    ConvDetail, OpDetail, Padding, PlanInput, Policy, PrimitiveOp, ProvenanceInput,
};
use crate::target::{FrameCycles, TargetError, Ti60Target};

use self::candidates::candidates_for;
use self::encode::{EdgeSpec, OpSpec, Problem, TensorSpec};
use self::model::ConstraintModel;

/// A spatial fold: SIMD lanes × output-channel PEs × kernel taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialShape {
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
}

impl SpatialShape {
    /// Multiply-accumulate lanes the shape instantiates.
    pub const fn lanes(self) -> u64 {
        self.simd as u64 * self.pe as u64 * self.tap_parallel as u64
    }

    /// Stable name, used by [`crate::layout::OpEngineBinding::candidate`].
    pub fn name(self) -> String {
        format!("{}x{}x{}", self.simd, self.pe, self.tap_parallel)
    }
}

/// One legal fold the search may choose for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// The shape.
    pub shape: SpatialShape,
    /// Its lane count.
    pub lanes: u64,
    /// Lane-array sweeps the operator costs on it.
    pub temporal_iterations: u64,
}

/// The search space the caller allows.
///
/// Each field is a *bound*, not a template: the same input graph solves to one
/// narrow shared engine, a wide shared engine, two pipelined engines or a
/// dedicated/shared split purely by changing these numbers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingBounds {
    /// Minimum instantiated engines.
    pub min_engines: u32,
    /// Maximum instantiated engines.
    pub max_engines: u32,
    /// Minimum lanes per engine.
    pub min_lanes_per_engine: u32,
    /// Maximum lanes per engine.
    pub max_lanes_per_engine: u32,
    /// Minimum pipeline links.
    pub min_pipeline_links: u32,
    /// Maximum pipeline links.
    pub max_pipeline_links: u32,
    /// Minimum dedicated engines.
    pub min_dedicated_engines: u32,
    /// Minimum shared engines.
    pub min_shared_engines: u32,
    /// Arena capacity in words.
    pub max_arena_words: u64,
    /// Enumerated legal spatial shapes (`tap_parallel == 1` for this backend).
    pub shapes: Vec<SpatialShape>,
}

impl Default for MappingBounds {
    fn default() -> Self {
        Self {
            min_engines: 1,
            max_engines: 1,
            min_lanes_per_engine: 1,
            max_lanes_per_engine: 1,
            min_pipeline_links: 0,
            max_pipeline_links: 0,
            min_dedicated_engines: 0,
            min_shared_engines: 1,
            max_arena_words: 1 << 20,
            shapes: vec![SpatialShape {
                simd: 1,
                pe: 1,
                tap_parallel: 1,
            }],
        }
    }
}

impl MappingBounds {
    /// Default hybrid search window for a Ti60 solve.
    ///
    /// Opens unit plus power-of-two SIMD/PE products up to 64 lanes, an engine
    /// count capped by the usable DSP budget, and optional pipeline cuts. The
    /// solver still chooses dedicated vs shared and the actual counts.
    pub fn hybrid_for(target: &Ti60Target, _policy: &Policy) -> Result<Self, SolverError> {
        let usable = target.usable_budget()?;
        let max_engines = u32::try_from(usable.dsp48.get().clamp(1, 32)).unwrap_or(32);
        let mut shapes = Vec::new();
        for simd in [1u32, 2, 4, 8, 16] {
            for pe in [1u32, 2, 4, 8, 16] {
                if simd.saturating_mul(pe) > 64 {
                    continue;
                }
                shapes.push(SpatialShape {
                    simd,
                    pe,
                    tap_parallel: 1,
                });
            }
        }
        shapes.sort();
        shapes.dedup();
        Ok(Self {
            min_engines: 1,
            max_engines,
            min_lanes_per_engine: 1,
            max_lanes_per_engine: 64,
            min_pipeline_links: 0,
            max_pipeline_links: max_engines.saturating_sub(1),
            min_dedicated_engines: 0,
            min_shared_engines: 0,
            max_arena_words: 1 << 20,
            shapes,
        })
    }
}

/// Solver failure. A *malformed* input or a broken backend is an error; a
/// bound set that no organization satisfies is a result
/// ([`PlanStatus::Infeasible`](crate::layout::PlanStatus::Infeasible)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverError {
    /// The input graph is malformed.
    InvalidInput {
        /// What is wrong.
        detail: String,
    },
    /// The requested bound set is malformed.
    InvalidBounds {
        /// What is wrong.
        detail: String,
    },
    /// The target description is invalid.
    Target(TargetError),
    /// The constraint backend could not be run.
    Backend {
        /// What is wrong.
        detail: String,
    },
    /// The backend returned something that is not a valid assignment.
    Protocol {
        /// What is wrong.
        detail: String,
    },
    /// The independent checker rejected the solver's assignment.
    Check {
        /// What is wrong.
        detail: String,
    },
    /// Lowering produced an artifact that failed its own validation.
    Layout(LayoutError),
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::InvalidInput { detail } => write!(f, "invalid solver input: {detail}"),
            SolverError::InvalidBounds { detail } => write!(f, "invalid mapping bounds: {detail}"),
            SolverError::Target(e) => write!(f, "invalid target: {e}"),
            SolverError::Backend { detail } => write!(f, "constraint backend failed: {detail}"),
            SolverError::Protocol { detail } => write!(f, "constraint backend protocol: {detail}"),
            SolverError::Check { detail } => write!(f, "mapping check failed: {detail}"),
            SolverError::Layout(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SolverError {}

impl From<TargetError> for SolverError {
    fn from(e: TargetError) -> Self {
        SolverError::Target(e)
    }
}

/// Everything the lowering needs about one operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlanOp {
    pub(crate) name: String,
    pub(crate) kind: OpKind,
    pub(crate) in_channels: u32,
    pub(crate) out_channels: u32,
    pub(crate) in_spatial: [u32; 2],
    pub(crate) out_spatial: [u32; 2],
    pub(crate) kernel: [u32; 2],
    pub(crate) groups: u32,
    pub(crate) stride: [u32; 2],
    pub(crate) requant: Option<Requant>,
    pub(crate) activation: Activation,
    pub(crate) weights: Option<TensorRef>,
    pub(crate) bias: Option<TensorRef>,
    pub(crate) inputs: Vec<TensorId>,
    pub(crate) outputs: Vec<TensorId>,
    pub(crate) in_code_range: Option<CodeRange>,
    pub(crate) out_code_range: Option<CodeRange>,
    pub(crate) macs: u64,
    pub(crate) candidates: Vec<Candidate>,
}

/// The context built once from the input graph and bounds.
pub(crate) struct Context {
    pub(crate) ops: Vec<PlanOp>,
    pub(crate) problem: Problem,
    pub(crate) graph_inputs: Vec<TensorId>,
    pub(crate) activation_bits: u32,
    pub(crate) accumulator_bits: u32,
    pub(crate) weight_bits: u32,
}

fn fail_bounds(detail: impl Into<String>) -> SolverError {
    SolverError::InvalidBounds {
        detail: detail.into(),
    }
}

/// Build the operator/tensor/edge model from the lowered primitive graph.
pub(crate) fn build_context(
    input: &PlanInput,
    bounds: &MappingBounds,
) -> Result<Context, SolverError> {
    if bounds.min_engines > bounds.max_engines {
        return Err(fail_bounds("min_engines exceeds max_engines"));
    }
    if bounds.min_lanes_per_engine > bounds.max_lanes_per_engine {
        return Err(fail_bounds(
            "min_lanes_per_engine exceeds max_lanes_per_engine",
        ));
    }
    if bounds.min_pipeline_links > bounds.max_pipeline_links {
        return Err(fail_bounds("min_pipeline_links exceeds max_pipeline_links"));
    }
    if bounds.shapes.is_empty() {
        return Err(fail_bounds("no spatial shapes were enumerated"));
    }

    let mut ops = Vec::with_capacity(input.ops.len());
    for op in &input.ops {
        ops.push(plan_op(input, op, bounds)?);
    }

    // Activation tensors: every operator input/output, with the operator that
    // produces it (graph inputs have none).
    let mut tensors: Vec<TensorSpec> = Vec::new();
    for op in &input.ops {
        for id in op.inputs.iter().chain(&op.outputs) {
            if tensors.iter().any(|t| t.id == *id) {
                continue;
            }
            let desc = input.tensor(*id).ok_or_else(|| SolverError::InvalidInput {
                detail: format!("operator {:?} references unknown tensor {id}", op.name),
            })?;
            let producer = input
                .ops
                .iter()
                .position(|candidate| candidate.outputs.contains(id));
            tensors.push(TensorSpec {
                id: *id,
                words: desc.shape.elements(),
                producer,
            });
        }
    }

    // Producer→consumer crossings between two operators.
    let mut edges: Vec<EdgeSpec> = Vec::new();
    for (consumer, op) in input.ops.iter().enumerate() {
        for id in &op.inputs {
            if let Some(producer) = input
                .ops
                .iter()
                .position(|candidate| candidate.outputs.contains(id))
            {
                edges.push(EdgeSpec {
                    producer,
                    consumer,
                    tensor: *id,
                });
            }
        }
    }

    // Distinct shapes, in deterministic order.
    let mut shapes: Vec<SpatialShape> = Vec::new();
    for op in &ops {
        for candidate in &op.candidates {
            if !shapes.contains(&candidate.shape) {
                shapes.push(candidate.shape);
            }
        }
    }
    shapes.sort_by_key(|s| (s.simd, s.pe, s.tap_parallel));

    let activation_bits = ops
        .iter()
        .filter_map(|op| op.inputs.first())
        .filter_map(|id| input.tensor(*id))
        .map(|t| t.elem.storage_bits())
        .next()
        .unwrap_or(8);
    let weight_bits = ops
        .iter()
        .filter_map(|op| op.weights.as_ref().map(|w| w.elem.storage_bits()))
        .max()
        .unwrap_or(8);

    let problem = Problem {
        ops: ops
            .iter()
            .map(|op| OpSpec {
                name: op.name.clone(),
                kind: op.kind,
                macs: op.macs,
                in_channels: op.in_channels,
                out_channels: op.out_channels,
                candidates: op.candidates.clone(),
            })
            .collect(),
        edges,
        tensors,
        shapes,
        bounds: bounds.clone(),
        activation_bits,
        accumulator_bits: 32,
        weight_bits,
    };

    Ok(Context {
        ops,
        problem,
        graph_inputs: input.inputs.clone(),
        activation_bits,
        accumulator_bits: 32,
        weight_bits,
    })
}

fn plan_op(
    input: &PlanInput,
    op: &PrimitiveOp,
    bounds: &MappingBounds,
) -> Result<PlanOp, SolverError> {
    let tensor = |id: TensorId| -> Result<&crate::planner::TensorDesc, SolverError> {
        input.tensor(id).ok_or_else(|| SolverError::InvalidInput {
            detail: format!("operator {:?} references unknown tensor {id}", op.name),
        })
    };
    let input_desc = tensor(*op.inputs.first().ok_or_else(|| SolverError::InvalidInput {
        detail: format!("operator {:?} has no inputs", op.name),
    })?)?;
    let output_desc = tensor(*op.outputs.first().ok_or_else(|| SolverError::InvalidInput {
        detail: format!("operator {:?} has no outputs", op.name),
    })?)?;
    let in_channels = input_desc.shape.c;
    let out_channels = output_desc.shape.c;
    let in_spatial = [input_desc.shape.h, input_desc.shape.w];
    let out_spatial = [output_desc.shape.h, output_desc.shape.w];

    match &op.detail {
        OpDetail::Conv(detail) => {
            if !op.kind.is_convolution() {
                return Err(SolverError::InvalidInput {
                    detail: format!("operator {:?} is not a convolution kind", op.name),
                });
            }
            let weight_desc = tensor(detail.weights)?;
            let kernel = [weight_desc.shape.h, weight_desc.shape.w];
            // Depthwise: each output channel reduces over one input channel and the
            // kernel taps, not over the full in_channels*out_channels product.
            let reduction_channels = if op.kind == OpKind::DepthwiseConv2d {
                1u64
            } else {
                u64::from(in_channels)
            };
            let macs = (out_spatial[0] as u64)
                .saturating_mul(out_spatial[1] as u64)
                .saturating_mul(reduction_channels)
                .saturating_mul(u64::from(out_channels))
                .saturating_mul(u64::from(kernel[0]))
                .saturating_mul(u64::from(kernel[1]));
            let candidates = candidates_for(in_channels, out_channels, macs, bounds);
            let weights = Some(TensorRef {
                id: weight_desc.id,
                name: weight_desc.name.clone(),
                shape: weight_desc.shape,
                elem: weight_desc.elem,
            });
            let bias = detail
                .bias
                .and_then(|id| input.tensor(id))
                .map(|t| TensorRef {
                    id: t.id,
                    name: t.name.clone(),
                    shape: t.shape,
                    elem: t.elem,
                });
            Ok(PlanOp {
                name: op.name.clone(),
                kind: op.kind,
                in_channels,
                out_channels,
                in_spatial,
                out_spatial,
                kernel,
                groups: detail.groups,
                stride: detail.stride,
                requant: Some(detail.requant.clone()),
                activation: detail.activation,
                weights,
                bias,
                inputs: op.inputs.clone(),
                outputs: op.outputs.clone(),
                in_code_range: input_desc.code_range,
                out_code_range: output_desc.code_range,
                macs,
                candidates,
            })
        }
        OpDetail::Elementwise(detail) => {
            plan_elementwise(
                op,
                bounds,
                input_desc,
                output_desc,
                in_channels,
                out_channels,
                in_spatial,
                out_spatial,
                detail.requant.clone(),
                detail.activation,
            )
        }
        OpDetail::Upsample(_) => {
            plan_elementwise(
                op,
                bounds,
                input_desc,
                output_desc,
                in_channels,
                out_channels,
                in_spatial,
                out_spatial,
                None,
                Activation {
                    kind: crate::layout::ActivationKind::None,
                    clamp_lo: i32::MIN,
                    clamp_hi: i32::MAX,
                },
            )
        }
    }
}

fn plan_elementwise(
    op: &PrimitiveOp,
    bounds: &MappingBounds,
    input_desc: &crate::planner::TensorDesc,
    output_desc: &crate::planner::TensorDesc,
    in_channels: u32,
    out_channels: u32,
    in_spatial: [u32; 2],
    out_spatial: [u32; 2],
    requant: Option<Requant>,
    activation: Activation,
) -> Result<PlanOp, SolverError> {
    // Elementwise / upsample / quantize: one code per cycle on a unit engine.
    let work = match op.kind {
        OpKind::Add => 2u64.saturating_mul(output_desc.shape.elements()),
        _ => output_desc.shape.elements(),
    };
    let unit = SpatialShape {
        simd: 1,
        pe: 1,
        tap_parallel: 1,
    };
    if !bounds.shapes.is_empty() && !bounds.shapes.contains(&unit) {
        return Err(SolverError::InvalidInput {
            detail: format!(
                "operator {:?} needs the unit shape (simd=pe=tap=1), which is not in the                  enumerated shapes",
                op.name
            ),
        });
    }
    let lanes = 1u64;
    if lanes < u64::from(bounds.min_lanes_per_engine)
        || lanes > u64::from(bounds.max_lanes_per_engine)
    {
        return Err(SolverError::InvalidInput {
            detail: format!(
                "operator {:?} needs one lane, outside the per-engine lane bounds [{}, {}]",
                op.name, bounds.min_lanes_per_engine, bounds.max_lanes_per_engine
            ),
        });
    }
    Ok(PlanOp {
        name: op.name.clone(),
        kind: op.kind,
        in_channels,
        out_channels,
        in_spatial,
        out_spatial,
        kernel: [1, 1],
        groups: 1,
        stride: [1, 1],
        requant,
        activation,
        weights: None,
        bias: None,
        inputs: op.inputs.clone(),
        outputs: op.outputs.clone(),
        in_code_range: input_desc.code_range,
        out_code_range: output_desc.code_range,
        macs: work,
        candidates: vec![Candidate {
            shape: unit,
            lanes,
            temporal_iterations: work,
        }],
    })
}

/// Serialize the constraint program the solver runs for `input` under `bounds`.
///
/// Exposed so tooling and tests can inspect the exact program, and so a test
/// can prove the description never names the backend implementation.
pub fn encode_model(
    input: &PlanInput,
    target: &Ti60Target,
    bounds: &MappingBounds,
) -> Result<ConstraintModel, SolverError> {
    let context = build_context(input, bounds)?;
    Ok(encode::encode(&context.problem, target))
}

/// Solve the hardware mapping for `input` under `bounds`.
///
/// Returns a schema-2 artifact whose mapping has already passed the independent
/// checker, or a schema-1 artifact recording that the bounds admit nothing.
pub fn solve_mapping(
    input: &PlanInput,
    target: &Ti60Target,
    policy: &Policy,
    provenance_input: &ProvenanceInput,
    bounds: &MappingBounds,
) -> Result<LayoutArtifact, SolverError> {
    input
        .validate()
        .map_err(|detail| SolverError::InvalidInput { detail })?;
    target.validate()?;
    policy
        .validate()
        .map_err(|error| SolverError::InvalidInput {
            detail: format!("invalid policy: {error}"),
        })?;

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

    let context = build_context(input, bounds)?;
    let provenance = Provenance {
        graph_sha256: provenance_input.graph_sha256.clone(),
        target_sha256: provenance_input.target_sha256.clone(),
        policy_sha256: provenance_input.policy_sha256.clone(),
        compiler_version: provenance_input.compiler_version.clone(),
        target_name: target.name.clone(),
        target_revision: target.revision.clone(),
        policy_schema_version: policy.schema_version,
        seed: policy.search.seed,
        options: BTreeMap::new(),
    };

    if context.ops.iter().any(|op| op.candidates.is_empty()) || context.problem.ops.is_empty() {
        return Ok(infeasible_artifact(
            input,
            target,
            policy,
            provenance,
            limits,
            "no operator has a legal fold under these bounds",
        ));
    }

    let model = encode::encode(&context.problem, target);
    let values = match backend::solve(&model)? {
        backend::Outcome::Solved(values) => values,
        backend::Outcome::Infeasible => {
            return Ok(infeasible_artifact(
                input,
                target,
                policy,
                provenance,
                limits,
                "no organization satisfies the requested bounds",
            ));
        }
    };
    let solution = encode::decode(&context.problem, &values)?;
    let checked = check::verify(&context.problem, &solution, target)?;

    let artifact = lower::lower(&lower::LowerInput {
        input,
        target,
        policy,
        provenance: provenance.clone(),
        limits,
        context: &context,
        checked: &checked,
    })?;
    match artifact.validate() {
        Ok(()) => Ok(artifact),
        Err(LayoutError::OverBudget { .. }) => Ok(infeasible_artifact(
            input,
            target,
            policy,
            provenance,
            limits,
            "the best organization does not fit the target budget",
        )),
        Err(error) => Err(SolverError::Layout(error)),
    }
}

/// The empty-selection artifact the v1 rules require for an infeasible result.
fn infeasible_artifact(
    input: &PlanInput,
    target: &Ti60Target,
    policy: &Policy,
    provenance: Provenance,
    limits: PlanLimits,
    detail: &str,
) -> LayoutArtifact {
    let _ = target;
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
    let rejections = RejectionSummary {
        rejected_candidates: 1,
        counts: BTreeMap::from([("no_candidate".to_string(), 1)]),
        samples: vec![RejectedCandidate {
            execution_mode: ExecutionMode::SharedTile,
            granularity: Granularity::Frame,
            tile: None,
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Bram,
            fusion_enabled: false,
            reason: RejectReason::NoCandidate {
                detail: detail.to_string(),
            },
        }],
        by_mode: BTreeMap::new(),
    };
    let search = SearchStats {
        explored: 1,
        expanded_nodes: 0,
        max_candidates: policy.search.max_candidates,
        per_candidate_budget: policy.search.max_candidates,
        truncated_candidates: 0,
        truncated: false,
    };
    let mut artifact = LayoutArtifact {
        format: LAYOUT_FORMAT.into(),
        schema_version: LAYOUT_SCHEMA_VERSION_V2,
        status: PlanStatus::Infeasible,
        certainty: PlanCertainty::from_search(PlanStatus::Infeasible, false),
        provenance,
        limits,
        objective: policy.objective.clone(),
        selection: None,
        resources: None,
        cycles: None,
        slack: None,
        tensors,
        schedule: Vec::new(),
        buffers: Vec::new(),
        fusions: Vec::new(),
        rejections,
        search,
        mapping: None,
    };
    artifact.normalize();
    artifact
}

/// A public four-layer pointwise chain and its real lowered inputs.
///
/// Four 1×1, stride-1, single-group pointwise convolutions `pw0`…`pw3`, each
/// 4→4 channels over a 1×2×2×4 activation, weights in `{-1, 0, 1}`, zero bias,
/// requant `M0 = 1`, `S = 0`, half-away-from-zero, signed int8 clamp. The plan
/// input is built directly so the numeric contract is exact; `graph` is a
/// valid graph-IR chain with the same layer names and weight tensors, used by
/// the RTL backend, and `weights` holds the raw int8 weight bytes keyed
/// `weights:pw0`…`weights:pw3` (row-major `[out_c][in_c]`).
pub struct ChainFixture {
    /// A valid graph-IR chain with layers `pw0`…`pw3`.
    pub graph: Graph,
    /// The lowered primitive graph the solver consumes.
    pub input: PlanInput,
    /// Raw int8 weights keyed by weight-tensor name.
    pub weights: BTreeMap<String, Vec<i8>>,
}

/// Build the four-layer pointwise chain fixture.
pub fn pointwise_chain4() -> ChainFixture {
    let names = ["pw0", "pw1", "pw2", "pw3"];
    let channels = 4u32;
    let spatial = 2u32;
    let per_layer = (channels * channels) as usize;

    let mut tensors = vec![crate::planner::TensorDesc {
        id: TensorId(0),
        name: "x".into(),
        shape: TensorShape::new(1, channels, spatial, spatial),
        elem: ElemType::I8,
        code_range: None,
    }];
    for (layer, name) in names.iter().enumerate() {
        tensors.push(crate::planner::TensorDesc {
            id: TensorId(1 + layer as u32),
            name: format!("{name}.out"),
            shape: TensorShape::new(1, channels, spatial, spatial),
            elem: ElemType::I8,
            code_range: None,
        });
        tensors.push(crate::planner::TensorDesc {
            id: TensorId(5 + layer as u32),
            name: format!("weights:{name}"),
            shape: TensorShape::new(channels, channels, 1, 1),
            elem: ElemType::I8,
            code_range: None,
        });
    }

    let requant = Requant {
        m0: 1,
        shift: 0,
        rounding: Rounding::HalfAwayFromZero,
        saturate_lo: -128,
        saturate_hi: 127,
        per_channel_multipliers: Vec::new(),
        multiplier_error_ppm: None,
    };
    let activation = Activation {
        kind: ActivationKind::None,
        clamp_lo: -128,
        clamp_hi: 127,
    };
    let ops = (0..4usize)
        .map(|layer| PrimitiveOp {
            name: names[layer].to_string(),
            kind: OpKind::PointwiseConv2d,
            inputs: vec![TensorId(layer as u32)],
            outputs: vec![TensorId(1 + layer as u32)],
            detail: OpDetail::Conv(ConvDetail {
                weights: TensorId(5 + layer as u32),
                bias: None,
                groups: 1,
                stride: [1, 1],
                padding: Padding::Zero {
                    top: 0,
                    bottom: 0,
                    left: 0,
                    right: 0,
                },
                requant: requant.clone(),
                activation,
            }),
        })
        .collect();

    let input = PlanInput {
        name: "pointwise-chain4".into(),
        tensors,
        ops,
        inputs: vec![TensorId(0)],
        outputs: vec![TensorId(4)],
    };

    let mut weights = BTreeMap::new();
    for (layer, name) in names.iter().enumerate() {
        weights.insert(
            format!("weights:{name}"),
            (0..per_layer)
                .map(|k| (((layer * 5 + k * 3) % 3) as i8) - 1)
                .collect(),
        );
    }

    ChainFixture {
        graph: chain_graph(&names, channels, per_layer),
        input,
        weights,
    }
}

/// Build a valid graph-IR chain with the same layer and weight-tensor names.
fn chain_graph(names: &[&str; 4], channels: u32, per_layer: usize) -> Graph {
    use crate::graph::{
        BitWidth, Bits, Channels, Dtype, GraphOp, GroupCount, Kernel, Layer, LayerId, LayerKind,
        LayerName, LayerOp, MANIFEST_FORMAT, Packing, Padding as GraphPadding, Point, PointId,
        PointName, PointRef, QMax, QuantContract, RequantCoefficients, Rounding as GraphRounding,
        Scale, Stride, WeightStream, ZeroPoint,
    };

    let bit = |bits: u8| BitWidth::try_new("chain fixture", bits).expect("valid width");
    let scale = |value: f32| Scale::try_new("chain fixture", value).expect("positive scale");
    let qmax = |value: f32| QMax::try_new("chain fixture", value).expect("positive qmax");
    let scale_value = 0.01f32;

    let points: Vec<Point> = (0..4u32)
        .map(|index| Point {
            id: PointId::new(index),
            name: PointName::new(format!("p{index}")),
            scale: scale(scale_value),
        })
        .collect();

    let layers: Vec<Layer> = names
        .iter()
        .enumerate()
        .map(|(layer, name)| {
            let head = layer == names.len() - 1;
            Layer {
                id: LayerId::new(layer as u32),
                name: LayerName::new(*name),
                op: LayerOp::Conv2d,
                kind: LayerKind::PointwiseConv2d,
                in_channels: Channels::try_new(channels).expect("non-zero"),
                out_channels: Channels::try_new(channels).expect("non-zero"),
                groups: GroupCount::try_new(1).expect("non-zero"),
                act_bits: bit(8),
                weight_bits: bit(8),
                act_zero_point: ZeroPoint::symmetric(),
                out_relu: false,
                out_unsigned: head,
                kernel: Kernel::try_new(1, 1).expect("non-zero"),
                stride: Stride::try_new(1, 1).expect("non-zero"),
                padding: GraphPadding::none(),
                in_point: PointId::new(layer as u32),
                out_point: if head {
                    PointRef::ImageSentinel
                } else {
                    PointRef::Point(PointId::new(layer as u32 + 1))
                },
                in_scale: scale(scale_value),
                out_scale: scale(if head { 1.0 / 255.0 } else { scale_value }),
                weight_scale: vec![scale_value; channels as usize],
                bias: Vec::new(),
                bias_int32: vec![0; channels as usize],
                requant: RequantCoefficients::try_new(vec![1.0; channels as usize])
                    .expect("positive requant"),
                weight_offset: layer * per_layer,
                weight_count: per_layer,
                weight_byte_offset: layer * per_layer,
                weight_bytes: per_layer,
            }
        })
        .collect();

    let mut ops = Vec::new();
    for (layer, name) in names.iter().enumerate() {
        let head = layer == names.len() - 1;
        ops.push(GraphOp::Conv {
            layer: LayerName::new(*name),
            in_point: PointId::new(layer as u32),
            out_point: if head {
                PointRef::ImageSentinel
            } else {
                PointRef::Point(PointId::new(layer as u32 + 1))
            },
        });
    }
    ops.push(GraphOp::QuantizeU8Output {
        scale: scale(1.0 / 255.0),
    });

    let code_count = names.len() * per_layer;
    Graph {
        source_format: MANIFEST_FORMAT.to_string(),
        arch: "pointwise-chain4".to_string(),
        quant: QuantContract {
            bits: Bits {
                acts: bit(8),
                weights: bit(8),
            },
            act_qmax: qmax(127.0),
            output_qmax: qmax(255.0),
            weight_qmax: qmax(127.0),
            act_zero_point: ZeroPoint::symmetric(),
            accumulator: Dtype::Int32,
            requant_rounding: GraphRounding::HalfAwayFromZero,
            packing: Packing::I8x1,
        },
        points,
        ops,
        layers,
        weight_stream: WeightStream {
            file: "weights.bin".to_string(),
            byte_len: code_count,
            code_count,
            packing: Packing::I8x1,
        },
        notes: vec!["4x2x2 pointwise chain fixture".to_string()],
    }
}
