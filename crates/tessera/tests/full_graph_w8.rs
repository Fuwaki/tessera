//! End-to-end tests for the full-graph SystemVerilog backend (`tessera::rtl::full`).
//!
//! Every test drives the real compiler: manifest import, MLIR-independent
//! lowering, layout search, execution-plan projection, weight resolution,
//! full-graph emission and (when Verilator is present) simulation against the
//! independent Rust scalar reference in `tessera::rtl::reference`, plus the
//! boundary fixture's own hand-written scalar chain.
//!
//! The layout policy below is the *emit policy*: it constrains the search to the
//! static-arena `shared_tile` organization with `tap_parallel = 1` and `frame`
//! granularity, so the selected layout is exactly what the backend
//! instantiates - one shared single-lane MAC engine over a whole-frame arena. A
//! plan outside that envelope is rejected by the backend by name; the envelope
//! test below asserts each rejection.
//!
//! Two things the emitted structure is checked for *structurally*, because they
//! are what the planner's cost model assumes and a metadata claim cannot prove:
//! one shared multiplier (the count of real multiplication operators does not
//! grow with the step count) and one shared activation read port (the debug port
//! is muxed onto it while the FSM is idle or held, never a second port).

use std::path::{Path, PathBuf};

use tessera::frontend;
use tessera::graph::GraphArtifact;
use tessera::layout::{
    ExecutionMode, ExecutionPlan, Granularity, OpKind, PlanStatus, TileGeometry,
};
use tessera::lower::{InputShape, lower_graph};
use tessera::planner::{self, Policy, ProvenanceInput};
use tessera::rtl::full::{self, FullGraphModel};
use tessera::rtl::reference;
use tessera::rtl::{RtlBundle, RtlConfig, SimulationOutcome, simulate_with_verilator};
use tessera::target::Ti60Target;
use tessera::weights::GraphWeightResolver;

/// The backend's envelope, expressed as a planner policy.
///
/// `shared_tile` + `frame` only: that is the organization the emitted engine
/// implements (one static-schedule activation arena, one shared single-lane MAC
/// engine, one whole-frame pass per step, so no tile halo is ever needed).
/// Frame granularity and `tap_parallel = 1` keep every selected step inside the
/// generated structure; the shared mode's own lane clamp keeps the fold at one.
const EMIT_POLICY: &str = r#"
format = "tessera.policy"
schema_version = 1

[constraints]
max_frame_cycles = 50000000
require_bit_exact = false
allow_cpu_layer_dispatch = false

[objective]
primary = "minimize_cycles"
secondary = ["minimize_ddr_bytes"]

[search]
execution_modes = ["shared_tile"]
granularities = ["frame"]
tile_widths = [16, 32]
tile_heights = [16, 32]
max_candidates = 200000
seed = 1
utilisation_percent = 100

[fusion]
allow = ["depthwise_requant_pointwise", "short_residual"]
forbid = ["fanout_value"]

[memory]
allow_ping_pong = true
max_ping_pong_buffers = 2
placements = ["line_buffer", "bram", "ddr"]

[hint]
prefer_fusion = false
prefer_bram_over_ddr = true
prefer_shared_engines = true

[[operator_override]]
pattern = "conv2d"
allowed_tap_parallel = [1]
allowed_simd = [1, 2, 3, 4, 8, 16, 32]
allowed_pe = [1, 2, 3, 4, 8]

[[operator_override]]
pattern = "pointwise_conv2d"
allowed_tap_parallel = [1]
allowed_simd = [1, 2, 3, 4, 8, 16, 32]
allowed_pe = [1, 2, 3, 4, 8]

[[operator_override]]
pattern = "depthwise_conv2d"
allowed_tap_parallel = [1]
allowed_simd = [1, 2, 3, 4, 8, 16, 32]
allowed_pe = [1]
"#;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-full-graph-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("load target")
}

fn emit_policy() -> Policy {
    Policy::from_toml_str(EMIT_POLICY).expect("parse the emit policy")
}

fn provenance(artifact: &GraphArtifact) -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: artifact.provenance().manifest_sha256.clone(),
        target_sha256: "target-sha".to_string(),
        policy_sha256: "policy-sha".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Import a manifest, solve the emit policy, project the plan, read the blob.
fn chain(dir_name: &str, shape: InputShape) -> (GraphArtifact, ExecutionPlan, Vec<u8>) {
    let artifact = frontend::import_manifest(&fixtures().join(dir_name))
        .unwrap_or_else(|error| panic!("import {dir_name}: {error}"));
    let (plan_input, _report) = lower_graph(artifact.graph(), shape)
        .unwrap_or_else(|error| panic!("lower {dir_name}: {error}"));
    let layout = planner::solve(
        &plan_input,
        &target(),
        &emit_policy(),
        &provenance(&artifact),
    )
    .unwrap_or_else(|error| panic!("solve {dir_name}: {error}"));
    if layout.status != PlanStatus::Feasible {
        panic!(
            "the emit policy produced an infeasible layout for {dir_name}: {:?}",
            layout.rejections
        );
    }
    let plan = layout.execution_plan().expect("project the plan");
    let blob = std::fs::read(fixtures().join(dir_name).join("weights.bin")).expect("read weights");
    (artifact, plan, blob)
}

fn config() -> RtlConfig {
    RtlConfig {
        module_prefix: "tessera".to_string(),
        emit_testbench: true,
        emit_memory_images: true,
        fixture_windows: 2,
        fixture_seed: 0x5eed_1234_abcd_0001,
    }
}

fn lower(dir_name: &str, shape: InputShape) -> (FullGraphModel, ExecutionPlan, RtlBundle) {
    let (artifact, plan, blob) = chain(dir_name, shape);
    let resolver =
        GraphWeightResolver::from_bytes(&artifact, blob).expect("resolve the weight blob");
    let model = FullGraphModel::resolve(artifact.graph(), &plan, &resolver)
        .unwrap_or_else(|error| panic!("resolve the model for {dir_name}: {error}"));
    let bundle = full::lower_full_graph(&plan, &model, &config())
        .unwrap_or_else(|error| panic!("emit {dir_name}: {error}"));
    (model, plan, bundle)
}

/// Strict simulation gate: with `TESSERA_ACCEPTANCE_STRICT` (or the narrower
/// `TESSERA_RTL_STRICT`) set, a missing simulator fails instead of skipping, so
/// an unavailable tool can never be mistaken for a verified result.
fn strict_simulation() -> bool {
    let set = |name: &str| {
        std::env::var(name)
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    };
    set("TESSERA_ACCEPTANCE_STRICT") || set("TESSERA_RTL_STRICT")
}

/// Outcome of trying to simulate a bundle: verified, or truthfully skipped.
enum Simulation {
    /// The RTL ran and reported the given log.
    Verified(String),
    /// No simulator was available; nothing was verified.
    Skipped(String),
}

impl Simulation {
    fn log(&self) -> &str {
        match self {
            Simulation::Verified(log) => log,
            Simulation::Skipped(reason) => reason,
        }
    }

    fn is_verified(&self) -> bool {
        matches!(self, Simulation::Verified(_))
    }
}

fn simulate(bundle: &RtlBundle, dir: &Path) -> Simulation {
    match simulate_with_verilator(bundle, dir).expect("simulation scaffold") {
        SimulationOutcome::Passed { log } => {
            assert!(
                log.contains("TESSA_RTL_PASS"),
                "a passing simulation must say so: {log}"
            );
            assert!(
                log.contains("mismatches=0"),
                "a passing simulation has zero mismatches: {log}"
            );
            Simulation::Verified(log)
        }
        SimulationOutcome::ToolUnavailable { tool, reason } => {
            let message = format!(
                "{tool} unavailable ({reason}); the emitted RTL was NOT simulated and this run \
                 verifies nothing about the hardware behaviour"
            );
            if strict_simulation() {
                panic!("the strict simulation gate cannot be satisfied: {message}");
            }
            eprintln!("SKIP: {message}");
            Simulation::Skipped(message)
        }
        SimulationOutcome::Failed { log, detail } => {
            panic!("generated RTL did not match the reference: {detail}\n{log}")
        }
    }
}

/// Codes of one trace block: `header` names the `frame N <keyword> ...` line and
/// the trailing shape literals give the count.
fn trace_block(text: &str, header: &str) -> Vec<i64> {
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.starts_with(header) {
            let numbers: Vec<u64> = line
                .trim_end()
                .split_whitespace()
                .filter_map(|token| token.parse::<u64>().ok())
                .collect();
            assert!(numbers.len() >= 3, "trace header without a shape: {line:?}");
            let count: u64 = numbers[numbers.len() - 3..].iter().product();
            return lines
                .take(count as usize)
                .map(|code| {
                    code.trim()
                        .parse::<i64>()
                        .unwrap_or_else(|_| panic!("trace code line {code:?} is not a code"))
                })
                .collect();
        }
    }
    Vec::new()
}

/// Real multiplication operators in the emitted DUT.
///
/// The emitter writes every *real* multiplier with surrounding spaces
/// (`x_ext * w_ext`, `a * m`) and writes address/index arithmetic without them,
/// so this counts multipliers rather than index products.
fn multiplication_operators(source: &str) -> usize {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(|line| line.matches(" * ").count())
        .sum()
}

/// Simultaneously live slots must never share a word.
///
/// This is the artifact-level check the reuse contract needs: the emitter
/// publishes each slot's inclusive `[first_step, last_step]` live range and its
/// word offset, so an independent reader can rediscover the placement and prove
/// that two slots live at the same step are address-disjoint (a lifetime arena
/// that aliases two live tensors would corrupt the frame).
fn assert_simultaneous_slots_are_disjoint(model: &FullGraphModel) {
    for (index, left) in model.slots.iter().enumerate() {
        for right in model.slots.iter().skip(index + 1) {
            let live_overlap =
                left.first_step <= right.last_step && right.first_step <= left.last_step;
            if !live_overlap {
                continue;
            }
            let left_range = (left.offset, left.offset + left.shape.elements().max(1));
            let right_range = (right.offset, right.offset + right.shape.elements().max(1));
            assert!(
                left_range.1 <= right_range.0 || right_range.1 <= left_range.0,
                "slots {} [{}, {}) and {} [{}, {}) are live at the same step but share words",
                left.name,
                left_range.0,
                left_range.1,
                right.name,
                right_range.0,
                right_range.1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// W8 fixture: four layers, ten graph ops, 6x6 input
// ---------------------------------------------------------------------------

#[test]
fn w8_full_graph_emits_and_simulates_against_the_scalar_reference() {
    let (model, plan, bundle) = lower("w8", InputShape::new(3, 6, 6).expect("shape"));

    for name in [
        "tessera_full_graph_v1.sv",
        "tessera_full_graph_v1_tb.sv",
        "stim_input.hex",
        "expected.hex",
        "reference_steps.txt",
        "rtl_bundle.json",
    ] {
        assert!(bundle.file(name).is_some(), "bundle is missing {name}");
    }
    assert_eq!(
        model.steps.len(),
        7,
        "the plan must cover every graph operator"
    );
    assert_eq!(
        model.mac_lanes, 1,
        "the emitted engine is one shared multiplier"
    );
    assert_eq!(
        model.word_width, 16,
        "9-bit residual codes need a 16-bit word"
    );

    // The arena is a lifetime peak, not a sum: the plan budgets the same peak.
    assert!(
        model.arena_words < model.arena_words_without_reuse,
        "the w8 schedule must reuse arena words ({} peak vs {} without reuse)",
        model.arena_words,
        model.arena_words_without_reuse
    );
    assert!(plan.engine.arena_words > 0, "the plan budgets an arena");
    assert!(
        model.arena_words <= plan.engine.arena_words,
        "the emitted arena {} must fit the plan's budget {}",
        model.arena_words,
        plan.engine.arena_words
    );
    assert_simultaneous_slots_are_disjoint(&model);

    // Accumulator bound: the whole channel reduction of the widest step.
    let widest = model
        .steps
        .iter()
        .filter(|step| step.is_conv())
        .map(|step| {
            let operand = model.slots[step.inputs[0]].max_abs;
            operand * 128 * step.taps as i64 * i64::from(step.in_per_group)
        })
        .max()
        .expect("a convolution step");
    assert!(
        model.acc_bound >= widest,
        "the accumulator bound {} must cover the widest step's reduction {}",
        model.acc_bound,
        widest
    );
    let needed = u64::from(64 - (widest as u64).leading_zeros()) + 1;
    assert!(
        u64::from(model.acc_width) >= needed,
        "acc_width {} must hold |acc| <= {} (needs {needed} bits)",
        model.acc_width,
        widest
    );

    let dir = temp_dir("w8");
    let simulation = simulate(&bundle, &dir);
    if !simulation.is_verified() {
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let log = simulation.log();
    assert!(log.contains("TESSA_RTL_PASS"), "log: {log}");
    assert!(log.contains("mismatches=0"), "log: {log}");

    let trace = dir.join("sim_outputs.txt");
    let text = std::fs::read_to_string(&trace).expect("the testbench must write sim_outputs.txt");
    assert!(
        text.starts_with("format tessera.sim_trace.v1"),
        "unexpected trace header: {:?}",
        text.lines().next()
    );

    // Frame 0 measures the unstalled schedule; frame 1 verifies the arena at
    // every step boundary (one block per step) plus the input and the image.
    let frames = 2;
    assert_eq!(
        text.matches(" input ").count(),
        frames,
        "one input block per frame"
    );
    assert_eq!(
        text.matches(" step ").count(),
        model.steps.len(),
        "one arena checkpoint per step (frame 1)"
    );
    assert_eq!(text.matches(" output ").count(), frames);
    assert_eq!(text.matches(" cycles ").count(), frames);
    assert_eq!(text.matches(" stalls ").count(), frames);
    for line in text.lines().filter(|line| line.starts_with("frame ")) {
        if line.contains(" cycles ") {
            assert!(
                line.contains("expected"),
                "the cycle line must carry the recorded schedule: {line}"
            );
        }
    }

    // Every step block must be the step's own codes, in the reference's order.
    let run =
        reference::run_integer(&model, &full::build_frames(&model, &config())).expect("reference");
    for step in &model.steps {
        let header = format!("frame 1 step {} ", step.index);
        let block = trace_block(&text, &header);
        assert_eq!(
            block.len() as u64,
            model.slots[step.output].shape.elements(),
            "step {} block length",
            step.index
        );
        let expected = &run.frames[1].steps[step.index].codes;
        assert_eq!(
            &block, expected,
            "the trace's step {} block must be the reference's codes",
            step.index
        );
    }

    // The residual carry is labelled i16 and the head unsigned codes are u8.
    assert!(
        text.contains("i16"),
        "the residual add block must be labelled i16, found: {:?}",
        text.lines()
            .filter(|l| l.contains(" step "))
            .collect::<Vec<_>>()
    );
    assert!(text.contains("u8"), "the unsigned head must be labelled u8");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn w8_residual_sum_is_carried_nine_bits_not_clamped() {
    let (model, _plan, _bundle) = lower("w8", InputShape::new(3, 6, 6).expect("shape"));
    let frames = full::build_frames(&model, &config());
    let run = reference::run_integer(&model, &frames).expect("reference");

    // `intref.rs:257-268`: the branch is requantized and clamped, the sum is not.
    let add_step = model
        .steps
        .iter()
        .find(|step| step.op_kind == OpKind::Add)
        .expect("the graph has a residual add");
    let slot = &model.slots[add_step.output];
    assert!(
        slot.max_abs > 128,
        "a residual sum reaches beyond the i8 domain, got {}",
        slot.max_abs
    );

    let mut carried = 0usize;
    for frame in &run.frames {
        let block = &frame.steps[add_step.index].codes;
        for code in block {
            if *code > 127 || *code < -128 {
                carried += 1;
            }
        }
    }
    assert!(
        carried > 0,
        "no residual code left the i8 domain on the default stimulus, so the 9-bit carry is \
         not exercised (a clamped implementation would pass this test by accident otherwise)"
    );

    // The trunk operand and the requantized branch are both in range, so the sum
    // is exactly their difference from the trunk.
    let trunk = model.steps[add_step.index].inputs[0];
    assert!(model.slots[trunk].max_abs <= 128, "the trunk stays in i8");
}

#[test]
fn w8_divergence_report_is_per_step_and_grounded_in_the_manifest_coefficients() {
    let (model, _plan, _bundle) = lower("w8", InputShape::new(3, 6, 6).expect("shape"));
    let frames = full::build_frames(&model, &config());
    let integer = reference::run_integer(&model, &frames).expect("reference");
    let again = reference::run_integer(&model, &frames).expect("reference again");
    assert_eq!(integer, again, "the reference must be deterministic");

    // The graph input is loaded unchanged, and the final block is the unsigned
    // image produced by the head convolution.
    assert_eq!(integer.frames.len(), frames.len());
    for frame in &integer.frames {
        assert_eq!(
            frame.input.len(),
            model.slots[model.input_slot].shape.elements() as usize
        );
        assert_eq!(frame.steps.len(), model.steps.len());
        assert_eq!(
            frame.steps.last().expect("a last step").codes.len(),
            model.slots[model.output_slot].shape.elements() as usize
        );
        for code in &frame.steps.last().expect("a last step").codes {
            assert!(
                (0..=255).contains(code),
                "the unsigned head must produce u8 codes, got {code}"
            );
        }
    }

    // Manifest f32 coefficients are *not* claimed to be reproduced bit for bit;
    // the reference quantifies the distance instead of asserting zero. Every
    // reported entry must be a real per-step distance, and every step that
    // carries manifest coefficients must be either reported or explicitly
    // unavailable - `divergence` is a zip of the two runs, so `<= len` alone
    // would hold for any input.
    let float = reference::run_f32(&model, &frames).expect("f32 reference");
    let divergence = reference::divergence(&integer, &float);
    for entry in &divergence {
        assert!(
            entry.index < model.steps.len(),
            "divergence entry {} is not a schedule index",
            entry.index
        );
        assert!(
            entry.differing > 0 && entry.max_abs_diff > 0,
            "entry {} claims a difference of zero, which is not a divergence",
            entry.index
        );
        assert!(
            entry.differing <= entry.total,
            "entry {}: {} of {} codes differ",
            entry.index,
            entry.differing,
            entry.total
        );
        assert_eq!(
            entry.total as u64,
            model.slots[model.steps[entry.index].output]
                .shape
                .elements(),
            "entry {} must cover the step's whole code block",
            entry.index
        );
        assert_eq!(
            entry.name, model.steps[entry.index].name,
            "entry {} must name the step it measured",
            entry.index
        );
        // The reported distance is really measured on the compared runs.
        let step = entry.index;
        let integer_codes = &integer.frames[0].steps[step].codes;
        let float_codes = &float.frames[0].steps[step].codes;
        let measured = integer_codes.len().min(float_codes.len());
        let differing = (0..measured)
            .filter(|index| integer_codes[*index] != float_codes[*index])
            .count();
        let worst = (0..measured)
            .map(|index| (integer_codes[index] - float_codes[index]).abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            entry.differing, differing,
            "entry {step} must report the codes that actually differ"
        );
        assert_eq!(
            entry.max_abs_diff, worst,
            "entry {step} must report the measured distance"
        );
        // A convolution step's distance is measured against the manifest
        // coefficients it carries; an elementwise step (the residual add) has
        // none of its own and is measured through the branch it requantizes.
        if model.steps[step].is_conv() {
            assert!(
                model.steps[step].requant_f32.len()
                    == model.slots[model.steps[step].output].shape.c as usize,
                "a reported convolution step must carry its manifest coefficients"
            );
        }
    }

    // The divergence is a property of the datapath and is measured on codes, so a
    // step whose fixed-point result happens to land on the manifest's codes is
    // legitimately absent. What must hold: the report is not empty (the M0/shift
    // approximation is real on this fixture), and every step that carries
    // manifest coefficients is either reported or explicitly unavailable.
    let reported: std::collections::BTreeSet<usize> =
        divergence.iter().map(|entry| entry.index).collect();
    assert!(
        !reported.is_empty(),
        "the fixed-point datapath must differ from the manifest f32 coefficients somewhere: {:?}",
        float.f32_unavailable
    );
    // The divergence originates in the steps whose manifest coefficients the
    // datapath approximates and propagates downstream, so a reported step need
    // not carry coefficients of its own (an upsample inherits the difference
    // through its input). What must hold: the report is not empty, it has more
    // than one step on this fixture (the approximation is not a single rounding
    // artifact), and at least one *coefficient-carrying* convolution is reported
    // - i.e. the divergence is attributed to where it is generated.
    assert!(
        divergence.len() > 1,
        "the report must attribute the divergence to more than one step: {divergence:?}"
    );
    assert!(
        divergence.iter().any(|entry| {
            let step = &model.steps[entry.index];
            step.is_conv() && !step.requant_f32.is_empty()
        }),
        "at least one convolution with manifest coefficients must be reported"
    );
    assert_eq!(
        model
            .steps
            .iter()
            .filter(|step| !step.requant_f32.is_empty())
            .count(),
        model.steps.iter().filter(|step| step.is_conv()).count(),
        "the w8 layers carry f32 requantization vectors"
    );
}

#[test]
fn full_graph_rejects_plans_outside_the_static_arena_envelope() {
    let (artifact, plan, blob) = chain("w8", InputShape::new(3, 6, 6).expect("shape"));
    let resolver = GraphWeightResolver::from_bytes(&artifact, blob).expect("resolve the weights");

    // A streaming/dedicated plan describes per-operator engines; the backend
    // emits one shared arena and refuses to relabel the plan.
    let mut streaming = plan.clone();
    streaming.execution_mode = ExecutionMode::Streaming;
    let error = FullGraphModel::resolve(artifact.graph(), &streaming, &resolver)
        .expect_err("a streaming plan must be rejected");
    assert!(
        error.to_string().contains("shared_tile"),
        "the error must name the required organization: {error}"
    );

    // Tap parallelism above one would make the plan's cycle model claim
    // parallelism the sequential shared engine does not have.
    let mut tapped = plan.clone();
    tapped.steps[0].tap_parallel = 3;
    let error = FullGraphModel::resolve(artifact.graph(), &tapped, &resolver)
        .expect_err("tap_parallel != 1 must be rejected");
    assert!(
        error.to_string().contains("tap_parallel"),
        "the error must name the offending field: {error}"
    );

    // Any fold other than the shared engine's single lane is refused, not
    // silently serialized.
    let mut wide = plan.clone();
    wide.steps[0].simd = 2;
    wide.steps[0].pe = 2;
    let error = FullGraphModel::resolve(artifact.graph(), &wide, &resolver)
        .expect_err("a fold beyond the shared lane must be rejected");
    assert!(
        error.to_string().contains("single-lane"),
        "the error must state the shared engine's single lane: {error}"
    );

    // A tiled plan is a different schedule: its tile geometry, halo re-reads and
    // per-tile setup cycles would be discarded by flattening it into one pass.
    let mut tiled = plan.clone();
    tiled.tile = Some(TileGeometry::new(16, 16));
    tiled.steps[0].mode = Granularity::Row;
    let error = FullGraphModel::resolve(artifact.graph(), &tiled, &resolver)
        .expect_err("a plan carrying a tile geometry must be rejected");
    assert!(
        error.to_string().contains("whole-frame") || error.to_string().contains("row/tile"),
        "the error must state the granularity the engine implements: {error}"
    );

    let mut tiled = plan.clone();
    tiled.steps[0].mode = Granularity::Tile;
    tiled.steps[0].tile = Some(TileGeometry::new(16, 16));
    let error = FullGraphModel::resolve(artifact.graph(), &tiled, &resolver)
        .expect_err("a step scheduled at tile granularity must be rejected");
    assert!(
        error.to_string().contains("whole-frame") || error.to_string().contains("row/tile"),
        "the error must state the granularity the engine implements: {error}"
    );
}

/// The emitted structure shares one multiplier and one arena read port.
///
/// This is the property the planner's `StepCost::lanes == 1` and its pooled
/// RAM10 figure assume; a per-step MAC circuit or a second asynchronous arena
/// port would make those numbers fiction, and no amount of metadata would show
/// it.
#[test]
fn emitted_engine_shares_one_multiplier_and_one_arena_read_port_across_steps() {
    let (_model, _plan, bundle) = lower("w8", InputShape::new(3, 6, 6).expect("shape"));
    let dut = bundle
        .file("tessera_full_graph_v1.sv")
        .expect("the DUT source")
        .contents
        .clone();
    // One MAC multiplier and one requantization multiplier, whatever the step
    // count: the engine is shared, not unrolled per step.
    assert_eq!(
        multiplication_operators(&dut),
        2,
        "the engine must instantiate exactly one MAC and one requantization multiplier"
    );
    // A single registered arena read: one read address register, one read data
    // register, and the debug port muxed onto them.
    assert_eq!(
        dut.matches("mem_raddr").count() > 0,
        true,
        "the arena read port must exist"
    );
    assert_eq!(dut.matches("mem_radeg").count(), 0, "no second read port");
    assert!(
        dut.contains("assign mem_raddr = dbg_sel ? dbg_addr : el_mem_addr;"),
        "the debug read must be muxed onto the same read port"
    );
    assert!(
        dut.contains("(state == S_IDLE) || (state == S_HOLD)"),
        "the debug read must only be serviced while the FSM issues no reads"
    );
    // No per-step accumulation function, and no per-step MAC instantiation.
    for step in ["acc0", "acc1", "acc2", "acc3"] {
        assert!(
            !dut.contains(&format!(
                "function automatic logic signed [ACCW-1:0] {step}("
            )),
            "the emitted engine must not contain a per-step accumulation function {step}"
        );
    }
}

// ---------------------------------------------------------------------------
// Boundary fixture: all-maximum codes through a multi-channel dense chain
// ---------------------------------------------------------------------------

/// Manifest of the boundary fixture: two dense 3x3 convolutions, a ReLU
/// (negative clamp) and an unsigned head (0..255 clamp), three input channels.
const BOUNDARY_MANIFEST: &str = r#"{
  "format": "fpga_st_train.int8.v1",
  "arch": "fixture-boundary",
  "bits": { "acts": 8, "weights": 8 },
  "act_qmax": 127.0,
  "output_qmax": 255.0,
  "weight_qmax": 127.0,
  "notes": ["boundary fixture: all-maximum input codes, multi-channel 3x3 dense chain, both clamps"],
  "points": ["image", "b1.in"],
  "point_scales": [0.007874016, 0.01],
  "weight_file": "weights.bin",
  "weight_packing": "i8-x1",
  "weight_bytes": 90,
  "graph": [
    { "op": "conv", "layer": "b1", "point": 0, "scale": 0.007874016 },
    { "op": "relu", "layer": "b1" },
    { "op": "conv", "layer": "b2", "point": 1, "scale": 0.01 },
    { "op": "quantize_u8_output", "scale": 0.003921569 }
  ],
  "layers": [
    {
      "name": "b1", "op": "conv2d+relu",
      "in_channels": 3, "out_channels": 2,
      "act_bits": 8, "weight_bits": 8, "act_zero_point": 0,
      "out_relu": true, "out_unsigned": false,
      "kernel": [3, 3], "stride": [1, 1], "reflect_pad": 1,
      "in_point": 0, "out_point": 1,
      "in_scale": 0.007874016, "out_scale": 0.01,
      "weight_scale": [0.02, 0.02],
      "bias": [0.0, 0.0],
      "bias_int32": [17, -23],
      "requant": [0.02, 0.02],
      "weight_offset": 0, "weight_count": 54,
      "weight_byte_offset": 0, "weight_bytes": 54
    },
    {
      "name": "b2", "op": "conv2d",
      "in_channels": 2, "out_channels": 2,
      "act_bits": 8, "weight_bits": 8, "act_zero_point": 0,
      "out_relu": false, "out_unsigned": true,
      "kernel": [3, 3], "stride": [1, 1], "reflect_pad": 1,
      "in_point": 1, "out_point": 2,
      "in_scale": 0.01, "out_scale": 0.003921569,
      "weight_scale": [0.05, 0.05],
      "bias": [0.0, 0.0],
      "bias_int32": [-3000, 11],
      "requant": [0.05, 0.05],
      "weight_offset": 54, "weight_count": 36,
      "weight_byte_offset": 54, "weight_bytes": 36
    }
  ]
}
"#;

/// `(out_channel, in_channel, kh, kw)` weights of the boundary fixture.
///
/// `b1` alternates sign per tap so its two channels straddle zero (the ReLU
/// lower clamp is reached), `b2` is strongly positive so the unsigned head
/// saturates at 255 for the all-maximum stimulus.
fn boundary_weights() -> Vec<i8> {
    let mut weights = Vec::with_capacity(90);
    for out in 0..2 {
        for input in 0..3 {
            for tap in 0..9 {
                let code = if (tap + out + input) % 2 == 0 {
                    127
                } else {
                    -127
                };
                weights.push(code);
            }
        }
    }
    // The head's channel 0 is strongly negative (its bias is negative too, so
    // its codes saturate at the unsigned *lower* bound) and channel 1 is
    // strongly positive (its codes saturate at the *upper* bound), so the
    // fixture reaches both clamps of the unsigned head.
    for out in 0..2 {
        for _input in 0..2 {
            for _tap in 0..9 {
                weights.push(if out == 0 { -127 } else { 127 });
            }
        }
    }
    weights
}

fn write_boundary_fixture(dir: &Path) -> Vec<i8> {
    std::fs::create_dir_all(dir).expect("create the fixture dir");
    std::fs::write(dir.join("manifest.json"), BOUNDARY_MANIFEST).expect("write the manifest");
    let weights = boundary_weights();
    let bytes: Vec<u8> = weights.iter().map(|code| *code as u8).collect();
    std::fs::write(dir.join("weights.bin"), &bytes).expect("write the weights");
    weights
}

/// Reflection that does not repeat the edge (`intref.rs::reflect_index`),
/// implemented here independently of the emitter.
fn reflect_index(index: i64, extent: i64) -> i64 {
    if extent <= 1 {
        return 0;
    }
    let period = 2 * (extent - 1);
    let mut folded = index.rem_euclid(period);
    if folded >= extent {
        folded = period - folded;
    }
    folded
}

/// The datapath contract, written out independently of the emitter and of
/// `reference.rs`: `(acc*M0 + round) >> shift`, round half away from zero, then
/// saturate.
fn requant_scalar(acc: i64, m0: i64, shift: u32, lo: i64, hi: i64) -> i64 {
    let product = acc * m0;
    let offset = if shift == 0 {
        0
    } else if product >= 0 {
        1i64 << (shift - 1)
    } else {
        (1i64 << (shift - 1)) - 1
    };
    let scaled = (product + offset) >> shift;
    scaled.clamp(lo, hi)
}

#[test]
fn all_max_dense_chain_matches_an_independent_scalar_and_both_clamps() {
    let dir = temp_dir("boundary-fixture");
    let weights = write_boundary_fixture(&dir);

    let artifact = frontend::import_manifest(&dir).expect("import the boundary fixture");
    let shape = InputShape::new(3, 5, 5).expect("5x5x3 input");
    let (plan_input, _report) =
        lower_graph(artifact.graph(), shape).expect("lower the boundary fixture");
    let layout = planner::solve(
        &plan_input,
        &target(),
        &emit_policy(),
        &provenance(&artifact),
    )
    .expect("solve the boundary fixture");
    assert_eq!(
        layout.status,
        PlanStatus::Feasible,
        "the boundary fixture must fit: {:?}",
        layout.rejections
    );
    let plan = layout.execution_plan().expect("project the plan");
    let resolver = GraphWeightResolver::from_bytes(&artifact, weights_written(&dir))
        .expect("resolve the boundary weights");
    let model = FullGraphModel::resolve(artifact.graph(), &plan, &resolver)
        .expect("resolve the boundary model");
    // The ReLU is fused into the first convolution's step (the manifest marks
    // the layer `conv2d+relu`), so the schedule is conv, head, quantize.
    assert_eq!(
        model.steps.len(),
        3,
        "two convolutions and the output quantize"
    );

    // The stimulus: frame 0 is every code at +127, frame 1 alternates the
    // extremes, so a wrapping accumulator and a one-sided clamp both show up.
    let elements = model.slots[model.input_slot].shape.elements();
    let mut frames = vec![vec![127i64; elements as usize]];
    frames.push(
        (0..elements)
            .map(|index| if index % 2 == 0 { 127 } else { -127 })
            .collect(),
    );
    let bundle = full::lower_full_graph_with_frames(&plan, &model, &config(), &frames)
        .expect("emit the boundary fixture");

    // The independent scalar chain over the same weights.
    let mut expected: Vec<Vec<i64>> = Vec::new();
    for frame in &frames {
        expected.push(boundary_scalar(&model, &weights, frame));
    }
    assert!(
        expected[0].iter().any(|code| *code == 255),
        "the boundary fixture must reach the unsigned upper clamp, got {:?}",
        expected[0]
    );
    assert!(
        expected.iter().flatten().any(|code| *code == 0),
        "the boundary fixture must reach a lower clamp"
    );
    assert!(
        expected
            .iter()
            .flatten()
            .all(|code| (0..=255).contains(code)),
        "the unsigned head must stay inside its clamp"
    );

    let sim_dir = temp_dir("boundary-sim");
    let simulation = simulate(&bundle, &sim_dir);
    if !simulation.is_verified() {
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&sim_dir);
        return;
    }
    let text = std::fs::read_to_string(sim_dir.join("sim_outputs.txt")).expect("trace");
    for (frame, want) in expected.iter().enumerate() {
        let header = format!("frame {frame} output ");
        let got = trace_block(&text, &header);
        assert_eq!(
            got.len(),
            want.len(),
            "frame {frame} output block length ({header:?})"
        );
        assert_eq!(
            &got, want,
            "frame {frame}: the emitted engine must match the independent scalar chain"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&sim_dir);
}

fn weights_written(dir: &Path) -> Vec<u8> {
    std::fs::read(dir.join("weights.bin")).expect("read the written weights")
}

/// Hand-written evaluation of the boundary chain, independent of the emitter and
/// of `reference.rs`: plain `i64` accumulation over the fixture's own weights,
/// the plan's `M0`/shift contract and the step clamps.
fn boundary_scalar(model: &FullGraphModel, weights: &[i8], frame: &[i64]) -> Vec<i64> {
    let mut slots: Vec<Vec<i64>> = model
        .slots
        .iter()
        .map(|slot| vec![0i64; slot.shape.elements() as usize])
        .collect();
    slots[model.input_slot] = frame.to_vec();
    let mut weight_cursor = 0usize;
    for step in &model.steps {
        let out = &step.out_shape;
        let (oh, ow, oc) = (out.h as i64, out.w as i64, out.c as usize);
        let mut codes = vec![0i64; out.elements() as usize];
        match step.op_kind {
            OpKind::Conv2d | OpKind::PointwiseConv2d => {
                let input = slots[step.inputs[0]].clone();
                let (ih, iw) = (step.in_shape.h as i64, step.in_shape.w as i64);
                let icg = step.in_per_group as usize;
                let (kh, kw) = (step.kernel[0] as i64, step.kernel[1] as i64);
                let pad = i64::from(step.border.amount());
                let layer_weights =
                    &weights[weight_cursor..weight_cursor + oc * icg * step.taps as usize];
                weight_cursor += layer_weights.len();
                for channel in 0..oc {
                    let requant = step.requant[channel];
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = i64::from(step.bias.get(channel).copied().unwrap_or(0));
                            for ic in 0..icg {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let y = oy * i64::from(step.stride[0]) + ky - pad;
                                        let x = ox * i64::from(step.stride[1]) + kx - pad;
                                        let (y, x) = match step.border {
                                            full::Border::Reflect { .. } => {
                                                (reflect_index(y, ih), reflect_index(x, iw))
                                            }
                                            full::Border::Zero { .. } => {
                                                if y < 0 || y >= ih || x < 0 || x >= iw {
                                                    continue;
                                                }
                                                (y, x)
                                            }
                                        };
                                        let value = input[((ic as i64 * ih + y) * iw + x) as usize];
                                        let weight = layer_weights[((channel * icg + ic)
                                            * kh as usize
                                            + ky as usize)
                                            * kw as usize
                                            + kx as usize];
                                        acc += value * i64::from(weight);
                                    }
                                }
                            }
                            codes[(channel * oh as usize + oy as usize) * ow as usize
                                + ox as usize] = requant_scalar(
                                acc,
                                i64::from(requant.multiplier),
                                u32::from(requant.shift),
                                i64::from(requant.lo),
                                i64::from(requant.hi),
                            );
                        }
                    }
                }
            }
            OpKind::Activation | OpKind::Requant | OpKind::QuantizeOutput => {
                let input = slots[step.inputs[0]].clone();
                let requant = step.requant.first().copied();
                for (index, code) in input.iter().enumerate() {
                    codes[index] = match requant {
                        Some(channel) => requant_scalar(
                            *code,
                            i64::from(channel.multiplier),
                            u32::from(channel.shift),
                            i64::from(channel.lo),
                            i64::from(channel.hi),
                        ),
                        None => (*code).clamp(step.clamp.0, step.clamp.1),
                    };
                }
            }
            other => panic!("the boundary fixture has no {other:?} step"),
        }
        slots[step.output] = codes;
    }
    slots[model.output_slot].clone()
}

// ---------------------------------------------------------------------------
// ink_dw: a real 29-layer model at a small resolution
// ---------------------------------------------------------------------------

#[test]
fn ink_dw_real_model_emits_and_simulates_at_a_small_resolution() {
    let dir_name = "ink_dw";
    if !fixtures().join(dir_name).join("manifest.json").exists() {
        eprintln!("SKIP: fixtures/{dir_name} is not present");
        return;
    }
    // 1x8x8x3: the smallest real-resolution input the whole model still solves
    // on the shipping Ti60 budget. The arena is a lifetime peak (the plan's
    // `engine.arena_words`), so this is the case that decides whether the real
    // model fits once pooling and reuse are honest.
    let shape = InputShape::new(3, 8, 8).expect("8x8x3 input");
    let artifact = frontend::import_manifest(&fixtures().join(dir_name)).expect("import ink_dw");
    let (plan_input, _report) =
        lower_graph(artifact.graph(), shape).expect("the real model must lower");
    let layout = planner::solve(
        &plan_input,
        &target(),
        &emit_policy(),
        &provenance(&artifact),
    )
    .expect("solve ink_dw on the shipping budget");
    let plan = match layout.status {
        PlanStatus::Feasible => layout.execution_plan().expect("project the plan"),
        PlanStatus::Infeasible => {
            let named = layout.rejections.samples.iter().all(|candidate| {
                matches!(
                    candidate.reason,
                    tessera::layout::RejectReason::RamBudget { .. }
                        | tessera::layout::RejectReason::UnsupportedArchitecture { .. }
                )
            });
            assert!(
                named,
                "an infeasible whole-model plan must be refused with a named resource: {:?}",
                layout.rejections.samples
            );
            eprintln!(
                "ink_dw 8x8 is refused by the shipping budget with a named resource \
                 ({} candidates rejected); nothing was emitted",
                layout.rejections.rejected_candidates
            );
            return;
        }
    };
    let blob = std::fs::read(fixtures().join(dir_name).join("weights.bin")).expect("weights");
    let resolver =
        GraphWeightResolver::from_bytes(&artifact, blob).expect("resolve the weight blob");
    let model = FullGraphModel::resolve(artifact.graph(), &plan, &resolver)
        .expect("resolve the real model");
    assert!(
        model.steps.len() > 20,
        "the real model has many steps, got {}",
        model.steps.len()
    );
    assert_eq!(
        model.mac_lanes, 1,
        "one shared multiplier whatever the depth"
    );
    // The arena the engine materializes must be the plan's budget, not the sum
    // of every tensor's storage.
    assert!(plan.engine.arena_words > 0, "the plan budgets an arena");
    assert!(
        model.arena_words <= plan.engine.arena_words,
        "the emitted arena {} must fit the plan's {} words",
        model.arena_words,
        plan.engine.arena_words
    );
    assert!(
        model.arena_words < model.arena_words_without_reuse,
        "the real model must reuse arena words ({} peak vs {} without reuse)",
        model.arena_words,
        model.arena_words_without_reuse
    );
    assert_simultaneous_slots_are_disjoint(&model);
    for slot in &model.slots {
        assert!(
            slot.max_abs < 1 << 15,
            "slot {} magnitude {} exceeds the 16-bit word",
            slot.name,
            slot.max_abs
        );
    }
    eprintln!(
        "ink_dw 8x8: {} steps, arena {} words of {} budget ({} words without reuse), {} \
         multiplier(s), generated beats {} vs plan estimate {}, plan RAM10 {}",
        model.steps.len(),
        model.arena_words,
        plan.engine.arena_words,
        model.arena_words_without_reuse,
        model.mac_lanes,
        model.cycles_unstalled,
        model.plan_cycles,
        layout.resources.map(|r| r.ram10).unwrap_or(0)
    );
    let bundle = full::lower_full_graph(&plan, &model, &config()).expect("emit the real model");
    let dut = bundle
        .file("tessera_full_graph_v1.sv")
        .expect("the DUT source")
        .contents
        .clone();
    assert_eq!(
        multiplication_operators(&dut),
        2,
        "a 29-layer model must still instantiate exactly one MAC and one requantization multiplier"
    );
    let sim_dir = temp_dir("ink-dw");
    let simulation = simulate(&bundle, &sim_dir);
    if !simulation.is_verified() {
        let _ = std::fs::remove_dir_all(&sim_dir);
        return;
    }
    let log = simulation.log();
    assert!(log.contains("TESSA_RTL_PASS"), "log: {log}");
    let text = std::fs::read_to_string(sim_dir.join("sim_outputs.txt")).expect("trace");
    assert_eq!(
        text.matches(" step ").count(),
        model.steps.len(),
        "one arena checkpoint per step"
    );
    let _ = std::fs::remove_dir_all(&sim_dir);
}
