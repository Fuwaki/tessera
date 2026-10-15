//! Tests for the general hardware-mapping solver (`tessera::solver`).
//!
//! Every test drives the real pipeline: the four-layer pointwise chain fixture
//! through the constraint encoder, the in-process CP-SAT backend, the
//! independent checker and the schema-2 lowering. The four organizations the
//! contract asks for are four different `MappingBounds` on the *same* input
//! graph, so a passing run is evidence the solver derives them rather than
//! looking them up.

use std::path::{Path, PathBuf};

use tessera::layout::{EngineRole, LAYOUT_SCHEMA_VERSION, LAYOUT_SCHEMA_VERSION_V2, PlanStatus};
use tessera::planner::{Policy, ProvenanceInput};
use tessera::solver::{
    MappingBounds, SolverError, SpatialShape, encode_model, pointwise_chain4, solve_mapping,
};
use tessera::target::Ti60Target;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("load the target")
}

fn policy() -> Policy {
    Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("load the policy")
}

fn provenance() -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: "chain-graph".to_string(),
        target_sha256: "ti60f225".to_string(),
        policy_sha256: "explore".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// The shapes every organization enumerates from.
fn shapes() -> Vec<SpatialShape> {
    vec![
        SpatialShape {
            simd: 1,
            pe: 1,
            tap_parallel: 1,
        },
        SpatialShape {
            simd: 1,
            pe: 2,
            tap_parallel: 1,
        },
        SpatialShape {
            simd: 2,
            pe: 1,
            tap_parallel: 1,
        },
        SpatialShape {
            simd: 1,
            pe: 4,
            tap_parallel: 1,
        },
        SpatialShape {
            simd: 2,
            pe: 2,
            tap_parallel: 1,
        },
        SpatialShape {
            simd: 4,
            pe: 1,
            tap_parallel: 1,
        },
    ]
}

fn bounds(
    min_engines: u32,
    max_engines: u32,
    min_lanes: u32,
    max_lanes: u32,
    min_pipeline: u32,
    max_pipeline: u32,
    min_dedicated: u32,
    min_shared: u32,
) -> MappingBounds {
    MappingBounds {
        min_engines,
        max_engines,
        min_lanes_per_engine: min_lanes,
        max_lanes_per_engine: max_lanes,
        min_pipeline_links: min_pipeline,
        max_pipeline_links: max_pipeline,
        min_dedicated_engines: min_dedicated,
        min_shared_engines: min_shared,
        max_arena_words: 1 << 20,
        shapes: shapes(),
    }
}

/// Organization 1: one narrow shared engine, lanes == 1.
fn org1() -> MappingBounds {
    bounds(1, 1, 1, 1, 0, 0, 0, 1)
}

/// Organization 2: one wide shared engine, lanes == 4.
fn org2() -> MappingBounds {
    bounds(1, 1, 4, 4, 0, 0, 0, 1)
}

/// Organization 3: two shared engines with one pipeline link.
fn org3() -> MappingBounds {
    bounds(2, 2, 1, 4, 1, 1, 0, 2)
}

/// Organization 4: one dedicated and one shared engine, no pipeline.
fn org4() -> MappingBounds {
    bounds(2, 2, 1, 4, 0, 0, 1, 1)
}

/// An operator streams its output plane while it still reads its input plane
/// in place, so the two must never share arena words. The liveness rule is
/// inclusive at both ends (a tensor stays live through the step that consumes
/// it), which makes an operator's input and output overlap at that operator's
/// own step and therefore blocks the alias.
fn assert_step_io_is_disjoint(layout: &tessera::layout::LayoutArtifact) {
    let mapping = layout.mapping.as_ref().expect("a mapping");
    for step in &layout.schedule {
        let words = |tensor| -> Option<(u64, u64)> {
            mapping
                .buffers
                .iter()
                .find(|residence| {
                    residence.tensor == tensor
                        && residence.kind == tessera::layout::BufferKind::Arena
                })
                .map(|residence| (residence.offset_words, residence.words))
        };
        if let (Some((in_lo, in_len)), Some((out_lo, out_len))) =
            (words(step.inputs[0]), words(step.outputs[0]))
        {
            assert!(
                in_lo >= out_lo + out_len || out_lo >= in_lo + in_len,
                "step {:?} input [{}..{}) and output [{}..{}) share arena words",
                step.op,
                in_lo,
                in_lo + in_len,
                out_lo,
                out_lo + out_len
            );
        }
    }
}

fn solve(bounds: &MappingBounds) -> tessera::layout::LayoutArtifact {
    let fixture = pointwise_chain4();
    let layout = solve_mapping(&fixture.input, &target(), &policy(), &provenance(), bounds)
        .expect("the solver must not fail on the chain fixture");
    layout
        .validate()
        .expect("a returned artifact must validate");
    if layout.mapping.is_some() {
        assert_step_io_is_disjoint(&layout);
    }
    layout
}

#[test]
fn the_chain_fixture_is_a_real_lowerable_graph() {
    let fixture = pointwise_chain4();
    fixture.graph.validate().expect("the graph must validate");
    assert_eq!(fixture.graph.layers().len(), 4);
    for name in ["pw0", "pw1", "pw2", "pw3"] {
        assert!(
            fixture.graph.layer_named(name).is_some(),
            "the graph must carry layer {name}"
        );
    }
    fixture
        .input
        .validate()
        .expect("the plan input must validate");
    assert_eq!(fixture.input.ops.len(), 4);
    assert_eq!(fixture.weights.len(), 4);
    for (layer, weights) in &fixture.weights {
        assert_eq!(weights.len(), 16, "{layer} must carry 16 weight codes");
        assert!(
            weights.iter().all(|w| (-1..=1).contains(w)),
            "{layer} weights must be ternary"
        );
    }
    // The graph must lower through the real frontend path, proving it is not a
    // decorative shell next to the hand-built plan input.
    let (lowered, _report) = tessera::lower::lower_graph(
        &fixture.graph,
        tessera::lower::InputShape::new(4, 2, 2).expect("a 4x2x2 input shape"),
    )
    .expect("the fixture graph must lower");
    assert!(
        lowered.ops.len() >= 4,
        "the lowered graph must contain the four convolutions"
    );
    for (lowered_op, plan_op) in lowered.ops.iter().zip(fixture.input.ops.iter()) {
        assert_eq!(lowered_op.name, plan_op.name);
    }
}

#[test]
fn organization_one_is_a_single_narrow_shared_engine() {
    let layout = solve(&org1());
    let mapping = layout.mapping.as_ref().expect("schema 2 carries a mapping");
    assert_eq!(layout.schema_version, LAYOUT_SCHEMA_VERSION_V2);
    assert_eq!(mapping.engines.len(), 1, "exactly one engine instance");
    assert_eq!(mapping.engines[0].role, EngineRole::Shared);
    assert_eq!(mapping.engines[0].lanes(), 1);
    assert_eq!(mapping.bindings.len(), 4, "all four ops must be bound");
    assert!(
        mapping.pipeline.is_empty(),
        "no pipeline under these bounds"
    );
    assert_eq!(mapping.makespan_cycles, 4 * (64 + 1));
    // One lane: ceil(1/2) MAC DSP48 plus lanes × ceil(32/18) requant DSP48.
    assert_eq!(layout.resources.expect("resources").dsp48, 1 + 2);
    let plan = layout.execution_plan().expect("projection");
    assert!(plan.mapping.is_some());
}

#[test]
fn organization_two_widens_the_shared_engine() {
    let layout = solve(&org2());
    let mapping = layout.mapping.as_ref().expect("schema 2 carries a mapping");
    assert_eq!(mapping.engines.len(), 1);
    assert_eq!(mapping.engines[0].role, EngineRole::Shared);
    assert!(
        mapping.engines[0].lanes() >= 4,
        "the wide organization must instantiate at least four lanes"
    );
    assert_eq!(mapping.bindings.len(), 4);
    assert!(mapping.pipeline.is_empty());
    // Four lanes cost 4/2 MAC DSP48 plus 4 × 2 requant DSP48 on one engine.
    assert_eq!(layout.resources.expect("resources").dsp48, 2 + 8);
}

#[test]
fn organization_three_pipelines_two_shared_engines() {
    let layout = solve(&org3());
    let mapping = layout.mapping.as_ref().expect("schema 2 carries a mapping");
    assert_eq!(mapping.engines.len(), 2);
    assert!(
        mapping
            .engines
            .iter()
            .all(|engine| engine.role == EngineRole::Shared)
    );
    assert_eq!(mapping.bindings.len(), 4);
    // Every engine carries exactly two operators.
    for engine in &mapping.engines {
        let count = mapping
            .bindings
            .iter()
            .filter(|binding| binding.engine == engine.name)
            .count();
        assert_eq!(count, 2, "engine {} must carry two ops", engine.name);
    }
    assert_eq!(mapping.pipeline.len(), 1, "exactly one pipeline link");
    let link = &mapping.pipeline[0];
    assert_ne!(link.producer_engine, link.consumer_engine);
    assert!(link.depth_elements >= 1);
    // The pipelined frame is strictly shorter than the sequential one.
    let sequential: u64 = layout
        .schedule
        .iter()
        .map(|step| step.estimated_cycles)
        .sum();
    assert!(mapping.makespan_cycles < sequential);
    assert_eq!(
        layout.resources.expect("resources").frame_cycles,
        mapping.makespan_cycles
    );
    // Two engines of four lanes: 2 × (4/2 MAC DSP48 + 4 × 2 requant DSP48).
    assert_eq!(layout.resources.expect("resources").dsp48, 2 * (2 + 8));
}

#[test]
fn organization_four_splits_dedicated_and_shared() {
    let layout = solve(&org4());
    let mapping = layout.mapping.as_ref().expect("schema 2 carries a mapping");
    assert_eq!(mapping.engines.len(), 2);
    let dedicated: Vec<_> = mapping
        .engines
        .iter()
        .filter(|engine| engine.role == EngineRole::Dedicated)
        .collect();
    let shared: Vec<_> = mapping
        .engines
        .iter()
        .filter(|engine| engine.role == EngineRole::Shared)
        .collect();
    assert_eq!(dedicated.len(), 1, "exactly one dedicated engine");
    assert_eq!(shared.len(), 1, "exactly one shared engine");
    let dedicated_bindings = mapping
        .bindings
        .iter()
        .filter(|binding| binding.engine == dedicated[0].name)
        .count();
    let shared_bindings = mapping
        .bindings
        .iter()
        .filter(|binding| binding.engine == shared[0].name)
        .count();
    assert_eq!(dedicated_bindings, 1);
    assert_eq!(shared_bindings, 3);
    assert!(mapping.pipeline.is_empty());
    // Both engines instantiate four lanes, so both cost 2 + 8 DSP48.
    assert_eq!(layout.resources.expect("resources").dsp48, 2 * (2 + 8));
}

#[test]
fn solving_twice_is_byte_identical() {
    let first = solve(&org3());
    let second = solve(&org3());
    assert_eq!(first, second);
    assert_eq!(
        first.to_toml_string().expect("toml"),
        second.to_toml_string().expect("toml")
    );
    assert_eq!(
        first.to_json_string().expect("json"),
        second.to_json_string().expect("json")
    );
    // A schema-2 artifact must survive both persisted forms, because the CLI
    // saves and reloads it between `solve` and `emit`.
    let toml = first.to_toml_string().expect("toml");
    let reloaded = tessera::layout::LayoutArtifact::from_toml_str(&toml)
        .expect("a schema-2 artifact must round-trip through TOML");
    assert_eq!(reloaded, first);
    let json = first.to_json_string().expect("json");
    let reloaded = tessera::layout::LayoutArtifact::from_json_str(&json)
        .expect("a schema-2 artifact must round-trip through JSON");
    assert_eq!(reloaded, first);
}

#[test]
fn impossible_lane_bounds_are_infeasible_not_a_fake_plan() {
    let impossible = bounds(1, 1, 1000, 1000, 0, 0, 0, 1);
    let layout = solve(&impossible);
    assert_eq!(layout.status, PlanStatus::Infeasible);
    assert_eq!(layout.schema_version, LAYOUT_SCHEMA_VERSION);
    assert!(layout.selection.is_none());
    assert!(layout.resources.is_none());
    assert!(layout.mapping.is_none());
    assert!(layout.schedule.is_empty());
    assert!(layout.rejections.rejected_candidates > 0);
}

#[test]
fn a_dedicated_engine_with_two_bindings_is_rejected() {
    let layout = solve(&org4());
    let mut broken = layout.clone();
    let mapping = broken.mapping.as_mut().expect("a mapping");
    // Find the shared engine and relabel it dedicated: it carries three ops, so
    // the artifact must no longer validate.
    let shared_name = mapping
        .engines
        .iter()
        .find(|engine| engine.role == EngineRole::Shared)
        .expect("a shared engine")
        .name
        .clone();
    for engine in &mut mapping.engines {
        if engine.name == shared_name {
            engine.role = EngineRole::Dedicated;
        }
    }
    let error = broken
        .validate()
        .expect_err("a dedicated engine bound to two operators must not validate");
    assert!(
        error.to_string().contains("dedicated"),
        "the error must name the role violation: {error}"
    );
}

#[test]
fn the_constraint_model_never_names_the_backend() {
    let fixture = pointwise_chain4();
    let model = encode_model(&fixture.input, &target(), &org3()).expect("encode");
    let json = model.to_json().expect("serialize");
    assert!(
        !json.contains("ortools"),
        "the model must not name OR-Tools"
    );
    assert!(!json.contains("CpModel"), "the model must not name CpModel");
    assert!(
        !json.contains("cp_model"),
        "the model must not name the binding"
    );
    // The document is a real program: it declares variables, constraints and a
    // lexicographic objective list.
    assert!(!model.variables.is_empty());
    assert!(!model.constraints.is_empty());
    assert!(model.objectives.len() >= 2);
    let round_trip: tessera::solver::model::ConstraintModel =
        serde_json::from_str(&json).expect("the model must round-trip through JSON");
    assert_eq!(round_trip, model);
}

#[test]
fn an_invalid_input_is_an_error_not_an_infeasible_plan() {
    let fixture = pointwise_chain4();
    let mut broken = fixture.input.clone();
    broken.ops.clear();
    let error = solve_mapping(&broken, &target(), &policy(), &provenance(), &org1())
        .expect_err("a malformed graph must be an error");
    assert!(matches!(error, SolverError::InvalidInput { .. }));
}
