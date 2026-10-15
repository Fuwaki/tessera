//! Verilator simulation of the mapped multi-engine backend.
//!
//! The same four-layer pointwise chain is solved four ways through
//! [`solve_mapping`], lowered with [`lower_mapped`], and simulated. Beyond
//! "mismatches == 0" the tests check the *structure* that the mapping asked for:
//! one engine with one lane, at least four multiplier lanes, two engines with a
//! ready/valid FIFO between them, and two engines that are never allowed to
//! overlap.
//!
//! Verilator is optional: when it is absent the simulation assertion prints a
//! `SKIP` line and passes **without** claiming a verified result. The structural
//! assertions still run, because they read the emitted text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tessera::layout::{ExecutionPlan, LayoutArtifact, PlanStatus, TensorRef};
use tessera::planner::{Policy, ProvenanceInput};
use tessera::rtl::{
    RtlBundle, RtlConfig, RtlError, SimulationOutcome, WeightResolver, lower_mapped,
    simulate_with_verilator, verilator_version,
};
use tessera::solver::{MappingBounds, SpatialShape, pointwise_chain4, solve_mapping};
use tessera::target::Ti60Target;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Scratch directory for one test. A Verilator build is hundreds of megabytes,
/// so the default is `/var/tmp`, not `/tmp`.
fn temp_dir(tag: &str) -> PathBuf {
    let base = std::env::var_os("TESSERA_TEST_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/tmp"));
    let dir = base.join(format!("tessera-mapping-rtl-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml"))
        .expect("load the real Ti60")
}

fn policy() -> Policy {
    Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("load the policy")
}

fn provenance() -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: "graph-sha256".to_string(),
        target_sha256: "target-sha256".to_string(),
        policy_sha256: "policy-sha256".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Weight bytes of the chain fixture, keyed by tensor name.
struct FixtureWeights<'a> {
    weights: &'a BTreeMap<String, Vec<i8>>,
}

impl WeightResolver for FixtureWeights<'_> {
    fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError> {
        self.weights
            .get(&tensor.name)
            .map(|codes| codes.iter().map(|code| *code as u8).collect())
            .ok_or_else(|| RtlError::WeightsNotResident {
                layer: tensor.name.clone(),
                tensor: tensor.name.clone(),
                detail: "the fixture carries no weight tensor with this name".to_string(),
            })
    }
}

/// Every legal spatial shape for the four pointwise layers (`pe` and `simd` must
/// divide four channels; the kernel has one tap).
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
    min_links: u32,
    max_links: u32,
    min_dedicated: u32,
    min_shared: u32,
) -> MappingBounds {
    MappingBounds {
        min_engines,
        max_engines,
        min_lanes_per_engine: min_lanes,
        max_lanes_per_engine: max_lanes,
        min_pipeline_links: min_links,
        max_pipeline_links: max_links,
        min_dedicated_engines: min_dedicated,
        min_shared_engines: min_shared,
        max_arena_words: 1 << 20,
        shapes: shapes(),
    }
}

/// Solves one organization and lowers it, with the structural checks that must
/// hold before any simulation runs.
struct Solved {
    bundle: RtlBundle,
    /// Contents of the emitted DUT source, for the structural assertions.
    dut: String,
}

impl Solved {
    fn summary(&self) -> &tessera::rtl::MappedSummary {
        self.bundle
            .metadata
            .mapped
            .as_ref()
            .expect("the mapped backend records its structure")
    }
}

fn solve_and_lower(tag: &str, bounds: &MappingBounds) -> Solved {
    let fixture = pointwise_chain4();
    let artifact: LayoutArtifact =
        solve_mapping(&fixture.input, &target(), &policy(), &provenance(), bounds)
            .expect("the mapping solve succeeds");
    assert_eq!(
        artifact.status,
        PlanStatus::Feasible,
        "{tag}: the organization must be feasible"
    );
    let plan: ExecutionPlan = artifact
        .execution_plan()
        .expect("project the execution plan");
    assert!(
        plan.mapping.is_some(),
        "{tag}: a schema-2 plan carries the mapping"
    );
    let resolver = FixtureWeights {
        weights: &fixture.weights,
    };
    let config = RtlConfig {
        module_prefix: "tessera".to_string(),
        emit_testbench: true,
        emit_memory_images: false,
        fixture_windows: 3,
        fixture_seed: 0x5eed_0000_0000_0001,
    };
    let bundle = lower_mapped(&plan, &fixture.graph, &resolver, &config)
        .unwrap_or_else(|error| panic!("{tag}: lowering must succeed: {error}"));
    assert_eq!(bundle.metadata.kernel_id, "tessera.mapped.i8i8.v1");
    let dut = bundle
        .verilog_sources()
        .into_iter()
        .find(|name| !name.ends_with("_tb.sv"))
        .expect("the bundle has a DUT source")
        .to_string();
    let source = bundle
        .file(&dut)
        .expect("the DUT source is in the bundle")
        .contents
        .clone();
    Solved {
        bundle,
        dut: source,
    }
}

/// Simulations are skipped cleanly when Verilator is absent.
fn simulate(tag: &str, bundle: &RtlBundle) -> Option<String> {
    let dir = temp_dir(tag);
    let outcome = simulate_with_verilator(bundle, &dir).expect("the simulator scaffold runs");
    match outcome {
        SimulationOutcome::Passed { log } => {
            let _ = std::fs::remove_dir_all(&dir);
            Some(log)
        }
        SimulationOutcome::ToolUnavailable { tool, reason } => {
            eprintln!("SKIP {tag}: {tool} unavailable ({reason}); RTL not simulated");
            None
        }
        SimulationOutcome::Failed { log, detail } => {
            panic!("{tag} failed: {detail}\n{log}");
        }
    }
}

fn count_instances(source: &str, marker: &str) -> usize {
    source.matches(marker).count()
}

/// `TESSA_RTL_PIPELINE engines=N links=M overlap_cycles=K`
fn pipeline_counts(log: &str) -> (usize, usize, usize) {
    let line = log
        .lines()
        .find(|line| line.contains("TESSA_RTL_PIPELINE"))
        .unwrap_or_else(|| panic!("the testbench reports its pipeline structure:\n{log}"));
    let field = |key: &str| -> usize {
        line.split_whitespace()
            .find_map(|token| token.strip_prefix(key))
            .unwrap_or_else(|| panic!("missing {key} in {line:?}"))
            .parse()
            .unwrap_or_else(|_| panic!("non-numeric {key} in {line:?}"))
    };
    (field("engines="), field("links="), field("overlap_cycles="))
}

fn require_verilator() -> bool {
    if verilator_version().is_none() {
        eprintln!("SKIP: verilator unavailable; the mapped RTL is not simulated");
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Organization 1: one shared engine, one lane
// ---------------------------------------------------------------------------

#[test]
fn one_shared_one_lane_engine_simulates_the_chain() {
    let solved = solve_and_lower("org1", &bounds(1, 1, 1, 1, 0, 0, 0, 1));
    let summary = solved.summary();
    assert_eq!(summary.engines.len(), 1, "one engine instance");
    assert_eq!(summary.engines[0].role, "shared");
    assert_eq!(summary.engines[0].lanes, 1, "lanes == 1");
    assert_eq!(
        summary.engines[0].operators.len(),
        4,
        "all four operators share the engine"
    );
    assert!(summary.links.is_empty(), "no pipeline link");
    assert_eq!(summary.multiplier_lanes, 1);
    assert_eq!(
        count_instances(&solved.dut, " u_lane_"),
        1,
        "a one-lane array instantiates exactly one multiplier"
    );
    assert_eq!(
        count_instances(&solved.dut, " u_engine"),
        1,
        "one engine instance"
    );
    assert_eq!(count_instances(&solved.dut, " u_fifo"), 0);
    if !require_verilator() {
        return;
    }
    let log = simulate("org1", &solved.bundle).expect("verilator ran");
    assert!(log.contains("mismatches=0"), "{log}");
    let (engines, links, overlap) = pipeline_counts(&log);
    assert_eq!((engines, links), (1, 0));
    assert_eq!(overlap, 0, "a single engine can never overlap");
}

// ---------------------------------------------------------------------------
// Organization 2: one shared engine, at least four lanes
// ---------------------------------------------------------------------------

#[test]
fn one_shared_multi_lane_engine_simulates_the_chain() {
    let solved = solve_and_lower("org2", &bounds(1, 1, 4, 4, 0, 0, 0, 1));
    let summary = solved.summary();
    assert_eq!(summary.engines.len(), 1);
    assert_eq!(summary.engines[0].role, "shared");
    assert!(
        summary.engines[0].lanes >= 4,
        "lanes {} >= 4",
        summary.engines[0].lanes
    );
    assert_eq!(summary.engines[0].operators.len(), 4);
    assert!(summary.links.is_empty());
    let multipliers = count_instances(&solved.dut, " u_lane_");
    assert!(
        multipliers >= 4,
        "the multi-lane array instantiates {multipliers} multipliers, at least four"
    );
    assert_eq!(
        multipliers as u64, summary.engines[0].lanes,
        "the multiplier count is the engine's lane count"
    );
    if !require_verilator() {
        return;
    }
    let log = simulate("org2", &solved.bundle).expect("verilator ran");
    assert!(log.contains("mismatches=0"), "{log}");
}

// ---------------------------------------------------------------------------
// Organization 3: two shared engines, one pipeline link
// ---------------------------------------------------------------------------

#[test]
fn two_shared_engines_with_a_pipeline_link_simulate_the_chain() {
    let solved = solve_and_lower("org3", &bounds(2, 2, 1, 4, 1, 1, 0, 2));
    let summary = solved.summary();
    assert_eq!(summary.engines.len(), 2, "two engine instances");
    assert_eq!(summary.links.len(), 1, "exactly one pipeline link");
    for engine in &summary.engines {
        assert_eq!(engine.role, "shared");
        assert_eq!(
            engine.operators.len(),
            2,
            "each engine carries two operators"
        );
    }
    assert!(summary.pipelined());
    assert_eq!(
        count_instances(&solved.dut, " u_engine"),
        2,
        "two engine instances are instantiated"
    );
    assert_eq!(
        count_instances(&solved.dut, " u_fifo"),
        1,
        "the pipeline link instantiates a ready/valid FIFO"
    );
    assert!(solved.dut.contains("_fifo"), "the FIFO module is emitted");
    if !require_verilator() {
        return;
    }
    let log = simulate("org3", &solved.bundle).expect("verilator ran");
    assert!(log.contains("mismatches=0"), "{log}");
    let (engines, links, overlap) = pipeline_counts(&log);
    assert_eq!((engines, links), (2, 1));
    assert!(
        overlap > 0,
        "a pipeline link licenses overlap: the testbench must observe both engines busy in the \
         same cycle, saw {overlap}"
    );
}

// ---------------------------------------------------------------------------
// Organization 4: one dedicated and one shared engine, no pipeline link
// ---------------------------------------------------------------------------

#[test]
fn dedicated_and_shared_engines_never_overlap() {
    let solved = solve_and_lower("org4", &bounds(2, 2, 1, 4, 0, 0, 1, 1));
    let summary = solved.summary();
    assert_eq!(summary.engines.len(), 2);
    assert!(
        summary
            .engines
            .iter()
            .any(|engine| engine.role == "dedicated"),
        "at least one dedicated engine"
    );
    assert!(
        summary.engines.iter().any(|engine| engine.role == "shared"),
        "at least one shared engine"
    );
    assert!(
        summary
            .engines
            .iter()
            .any(|engine| engine.role == "dedicated" && engine.operators.len() == 1),
        "a dedicated engine serves exactly one operator"
    );
    assert!(summary.links.is_empty(), "no pipeline link");
    assert_eq!(
        count_instances(&solved.dut, " u_engine"),
        2,
        "both engine arrays are still instantiated"
    );
    assert_eq!(count_instances(&solved.dut, " u_fifo"), 0);
    if !require_verilator() {
        return;
    }
    let log = simulate("org4", &solved.bundle).expect("verilator ran");
    assert!(log.contains("mismatches=0"), "{log}");
    let (engines, links, overlap) = pipeline_counts(&log);
    assert_eq!((engines, links), (2, 0));
    assert_eq!(
        overlap, 0,
        "without a pipeline link the consumer engine starts only after the producer engine is \
         done, so no two engines are ever busy together"
    );
}

// ---------------------------------------------------------------------------
// Rejection
// ---------------------------------------------------------------------------

#[test]
fn an_operator_outside_the_subset_is_rejected() {
    let fixture = pointwise_chain4();
    let artifact = solve_mapping(
        &fixture.input,
        &target(),
        &policy(),
        &provenance(),
        &bounds(1, 1, 1, 1, 0, 0, 0, 1),
    )
    .expect("the mapping solve succeeds");
    let mut plan = artifact
        .execution_plan()
        .expect("project the execution plan");
    // A 3x3 kernel is a different operator: the mapped datapath implements one
    // tap sweep over a 1x1 kernel and must say so instead of approximating.
    plan.steps[2].kernel = [3, 3];
    let resolver = FixtureWeights {
        weights: &fixture.weights,
    };
    let error = lower_mapped(&plan, &fixture.graph, &resolver, &RtlConfig::default())
        .expect_err("a 3x3 kernel is not in the mapped subset");
    let message = error.to_string();
    assert!(
        message.contains(&plan.steps[2].op),
        "the rejection names the step: {message}"
    );
    assert!(
        message.contains("1x1") || message.contains("kernel"),
        "the rejection names the construct: {message}"
    );
}
