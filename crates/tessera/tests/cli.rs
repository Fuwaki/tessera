//! CLI smoke tests: the acceptance path, driven exactly as a user drives it.
//!
//! Every test runs the real `tessera` binary against the checked-in examples
//! under `fixtures/` and a fresh temporary stage directory. Paths come from
//! `CARGO_MANIFEST_DIR`, so nothing is machine specific; Verilator is optional
//! and its absence is reported, never hidden.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn fx(relative: &str) -> String {
    fixtures().join(relative).display().to_string()
}

/// Scratch directory for one test, under `TESSERA_TEST_TMPDIR` (default
/// `/var/tmp`), for the same reason as the pipeline tests: full-graph bundles and
/// Verilator builds are large and `/tmp` is often a small tmpfs.
fn temp_dir(tag: &str) -> PathBuf {
    let base = std::env::var_os("TESSERA_TEST_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/tmp"));
    let dir = base.join(format!("tessera-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn run(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .output()
        .expect("run the tessera binary")
}

fn run_ok(args: &[String]) -> String {
    let output = run(args);
    assert!(
        output.status.success(),
        "`tessera {}` failed\nstdout:\n{}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn run_err(args: &[String]) -> String {
    let output = run(args);
    assert!(
        !output.status.success(),
        "`tessera {}` was expected to fail but succeeded\nstdout:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8_lossy(&output.stderr).to_string()
}

fn s(value: impl AsRef<str>) -> String {
    value.as_ref().to_string()
}

// ---------------------------------------------------------------------------
// The supported end-to-end path
// ---------------------------------------------------------------------------

#[test]
fn pointwise_fixture_runs_solve_emit_verify_and_simulate() {
    let out = temp_dir("pointwise");
    let layout = out.join("layout.tlayout").display().to_string();
    let weights = fx("pointwise/weights");

    let solved = run_ok(&[
        s("solve"),
        s("--primitive"),
        fx("pointwise/plan_input.json"),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/streaming.toml"),
        s("--out"),
        out.display().to_string(),
    ]);
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("step 0: pw"), "{solved}");
    assert!(layout_file_exists(&out), "the layout artifact is written");

    let emitted = run_ok(&[
        s("emit"),
        layout.clone(),
        s("--weights"),
        weights.clone(),
        s("--out"),
        out.display().to_string(),
    ]);
    assert!(
        emitted.contains("emitted kernel tessera.pw_dot.i8i8.v1"),
        "{emitted}"
    );

    let verified = run_ok(&[s("verify"), out.display().to_string()]);
    assert!(verified.contains("verify: ok"), "{verified}");
    assert!(verified.contains("rtl"), "{verified}");

    let simulated = run_ok(&[
        s("simulate"),
        layout,
        s("--weights"),
        weights,
        s("--out"),
        out.display().to_string(),
    ]);
    assert!(
        simulated.contains("simulation: PASS") || simulated.contains("simulation: SKIP"),
        "simulation must either pass or skip transparently:\n{simulated}"
    );
    assert!(
        !simulated.contains("simulation: FAIL"),
        "the generated RTL must not fail its own testbench:\n{simulated}"
    );

    let _ = std::fs::remove_dir_all(&out);
}

fn layout_file_exists(dir: &Path) -> bool {
    dir.join(tessera::stages::LAYOUT_FILE).exists()
}

#[test]
fn rtl_emission_is_byte_deterministic_across_runs() {
    let out = temp_dir("determinism");
    let layout = out.join("layout.tlayout").display().to_string();
    let weights = fx("pointwise/weights");
    run_ok(&[
        s("solve"),
        s("--primitive"),
        fx("pointwise/plan_input.json"),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/streaming.toml"),
        s("--out"),
        out.display().to_string(),
    ]);

    let emit = |again: &str| -> (String, Vec<u8>) {
        let stdout = run_ok(&[
            s("emit"),
            layout.clone(),
            s("--weights"),
            weights.clone(),
            s("--out"),
            out.display().to_string(),
        ]);
        let source = std::fs::read(out.join(tessera::stages::RTL_DIR).join(again))
            .expect("read the emitted DUT");
        (stdout, source)
    };

    let (first_log, first) = emit("tessera_pw_pw_dot_i8i8_v1.sv");
    let (second_log, second) = emit("tessera_pw_pw_dot_i8i8_v1.sv");
    assert_eq!(first, second, "RTL emission must be reproducible");
    let digest = |log: &str| {
        log.lines()
            .find(|line| line.contains("bundle digest:"))
            .map(str::to_string)
            .expect("the CLI reports the bundle digest")
    };
    assert_eq!(digest(&first_log), digest(&second_log));

    let _ = std::fs::remove_dir_all(&out);
}

// ---------------------------------------------------------------------------
// The whole-network path
// ---------------------------------------------------------------------------

#[test]
fn manifest_pipeline_reaches_an_emittable_layout_through_the_generic_policy() {
    let out = temp_dir("w8");
    let out_arg = out.display().to_string();

    let imported = run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(imported.contains("4 layers"), "{imported}");
    assert!(
        out.join("weights.bin").exists() && out.join("graph.tgraph").exists(),
        "the stage directory keeps the graph and its weight blob"
    );

    // No --input-shape here: the shape recorded at import time is reused.
    let emitted = run_ok(&[
        s("emit-mlir"),
        out.join("graph.tgraph").display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(emitted.contains("10 operators"), "{emitted}");

    // The generic policy is the unconstrained one, so this is the planner's own
    // choice of organization rather than a policy shaped to the backend.
    let solved = run_ok(&[
        s("solve"),
        s("--graph"),
        out.join("graph.tgraph").display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/explore.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("7 steps"), "{solved}");
    assert!(solved.contains("quantize_u8_output"), "{solved}");
    assert!(
        solved.contains("lower: layer dw: int32 bias is all zero"),
        "a zero int32 bias is dropped with a recorded note: {solved}"
    );

    // The whole model emits through the full-graph backend, which reports the
    // structure it generated next to the planner's estimate.
    let bundle = run_ok(&[
        s("emit"),
        out.join("layout.tlayout").display().to_string(),
        s("--graph"),
        out.join("graph.tgraph").display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(bundle.contains("full-graph backend"), "{bundle}");
    assert!(bundle.contains("full graph:"), "{bundle}");
    assert!(bundle.contains("arena"), "{bundle}");
    assert!(bundle.contains("cycles:"), "{bundle}");
    assert!(bundle.contains("bundle digest:"), "{bundle}");

    let verified = run_ok(&[s("verify"), out_arg.clone()]);
    assert!(verified.contains("verify: ok"), "{verified}");
    assert!(verified.contains("graph.tgraph"), "{verified}");

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn verify_detects_a_tampered_target_copy() {
    let out = temp_dir("tamper");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("solve"),
        s("--primitive"),
        fx("pointwise/plan_input.json"),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/streaming.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[s("verify"), out_arg.clone()]);

    let target = out.join(tessera::stages::TARGET_FILE);
    let mut text = std::fs::read_to_string(&target).expect("read the staged target");
    text.push_str("# tampered\n");
    std::fs::write(&target, text).expect("tamper with the staged target");

    let error = run_err(&[s("verify"), out_arg]);
    assert!(error.contains("target hash"), "{error}");

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn solving_is_rejected_without_an_input_shape() {
    let out = temp_dir("no-shape");
    let out_arg = out.display().to_string();
    run_ok(&[s("import"), fx("w8"), s("--out"), out_arg.clone()]);
    let error = run_err(&[
        s("solve"),
        s("--graph"),
        out.join("graph.tgraph").display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/streaming.toml"),
        s("--out"),
        out_arg,
    ]);
    assert!(error.contains("--input-shape"), "{error}");

    let _ = std::fs::remove_dir_all(&out);
}

// ---------------------------------------------------------------------------
// Inspection and the optional MLIR stage
// ---------------------------------------------------------------------------

#[test]
fn inspect_reports_every_artifact_kind() {
    let out = temp_dir("inspect");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit-mlir"),
        out.join("graph.tgraph").display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("solve"),
        s("--primitive"),
        fx("pointwise/plan_input.json"),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/streaming.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit"),
        out.join("layout.tlayout").display().to_string(),
        s("--weights"),
        fx("pointwise/weights"),
        s("--out"),
        out_arg.clone(),
    ]);

    for (file, expected) in [
        ("graph.tgraph", "manifest sha256"),
        ("mlir.json", "module hash"),
        ("primitive.json", "PointwiseConv2d"),
        ("layout.tlayout", "status: feasible"),
        ("plan.tplan", "mode: streaming"),
        ("rtl/rtl_bundle.json", "tessera.pw_dot.i8i8.v1"),
    ] {
        let text = run_ok(&[s("inspect"), out.join(file).display().to_string()]);
        assert!(
            text.contains(expected),
            "inspect {file} did not report {expected}:\n{text}"
        );
    }

    // An artifact this build does not understand is reported, not guessed at.
    let unknown = out.join("unknown.json");
    std::fs::write(&unknown, "{\"hello\":1}").expect("write");
    let error = run_err(&[s("inspect"), unknown.display().to_string()]);
    assert!(
        error.contains("not a recognised tessera artifact"),
        "{error}"
    );

    let _ = std::fs::remove_dir_all(&out);
}

#[cfg(not(feature = "melior"))]
#[test]
fn optimize_explains_the_missing_feature_instead_of_faking_success() {
    let out = temp_dir("optimize");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit-mlir"),
        out.join("graph.tgraph").display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);

    let error = run_err(&[
        s("optimize"),
        out.join(tessera::stages::MLIR_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(error.contains("melior"), "{error}");
    assert!(error.contains("0.28.2"), "{error}");
    assert!(
        !out.join(tessera::stages::OPTIMIZED_FILE).exists(),
        "a failed stage must not leave an artifact behind"
    );

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn whole_model_w8_emits_and_simulates_through_the_full_graph_backend() {
    let out = temp_dir("w8-full");
    let out_arg = out.display().to_string();
    let layout = out.join(tessera::stages::LAYOUT_FILE).display().to_string();

    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    let solved = run_ok(&[
        s("solve"),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("certainty"), "{solved}");
    assert!(solved.contains("7 steps"), "{solved}");

    let emitted = run_ok(&[
        s("emit"),
        layout.clone(),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(emitted.contains("full-graph backend"), "{emitted}");
    assert!(emitted.contains("full graph:"), "{emitted}");
    assert!(emitted.contains("arena"), "{emitted}");

    let verified = run_ok(&[s("verify"), out_arg.clone()]);
    assert!(verified.contains("verify: ok"), "{verified}");

    let simulated = run_ok(&[
        s("simulate"),
        layout,
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg,
    ]);
    assert!(
        simulated.contains("simulation: PASS") || simulated.contains("simulation: SKIP"),
        "the whole model must simulate or skip transparently:\n{simulated}"
    );
    assert!(!simulated.contains("simulation: FAIL"), "{simulated}");
    if simulated.contains("simulation: PASS") {
        let trace = out.join("sim").join("sim_outputs.txt");
        let text = std::fs::read_to_string(&trace).expect("a PASS with a readable trace");
        let blocks = text.lines().filter(|line| line.contains(" step ")).count();
        assert!(
            blocks >= 7,
            "a PASS must be re-readable: one block per scheduled step, found {blocks}"
        );
    }

    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn multi_step_plan_refuses_a_blob_only_resolver() {
    // A multi-step plan needs bias from the model, which weight blobs cannot
    // supply; the CLI must say so instead of emitting unattributable RTL.
    let out = temp_dir("w8-blobs");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("solve"),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    let error = run_err(&[
        s("emit"),
        out.join(tessera::stages::LAYOUT_FILE).display().to_string(),
        s("--weights"),
        fx("pointwise/weights"),
        s("--out"),
        out_arg,
    ]);
    assert!(error.contains("--graph"), "{error}");
    let _ = std::fs::remove_dir_all(&out);
}

/// The native MLIR path: emitted module -> pinned pipeline -> primitive graph ->
/// plan, driven only through the CLI. Requires the `melior` feature and an
/// LLVM/MLIR 23.x toolchain (see docs/developer-setup.md); the default build
/// skips it rather than pretending the stage ran.
#[cfg(feature = "melior")]
#[test]
fn optimized_mlir_drives_the_downstream_solve_and_emit() {
    let out = temp_dir("mlir-path");
    let out_arg = out.display().to_string();

    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    let emitted = run_ok(&[
        s("emit-mlir"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(emitted.contains("module hash"), "{emitted}");

    let optimized = run_ok(&[
        s("optimize"),
        out.join(tessera::stages::MLIR_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(optimized.contains("melior"), "{optimized}");
    assert!(
        out.join(tessera::stages::OPTIMIZED_MODULE_FILE).exists(),
        "the optimized module text is written"
    );

    // The plan must come from the optimized module, not from a graph re-lowering.
    let solved = run_ok(&[
        s("solve"),
        s("--mlir"),
        out.join(tessera::stages::OPTIMIZED_FILE)
            .display()
            .to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(solved.contains("fusion groups"), "{solved}");
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("certainty"), "{solved}");
    assert!(out.join(tessera::stages::PRIMITIVES_FILE).exists());
    assert!(out.join(tessera::stages::PLAN_FILE).exists());

    // Every provenance link in the chain, including primitives -> optimized.
    let verified = run_ok(&[s("verify"), out_arg.clone()]);
    assert!(verified.contains("verify: ok"), "{verified}");
    assert!(
        verified.contains("descends from the optimized module"),
        "the primitive artifact must be linked to the module it came from:\n{verified}"
    );

    let emitted_rtl = run_ok(&[
        s("emit"),
        out.join(tessera::stages::LAYOUT_FILE).display().to_string(),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg,
    ]);
    assert!(emitted_rtl.contains("full-graph backend"), "{emitted_rtl}");

    let _ = std::fs::remove_dir_all(&out);
}

/// `verify` must recompute, not compare two artifacts' self-reports: editing the
/// module text inside a saved artifact has to be detected even though every
/// recorded hash still agrees with every other recorded hash.
#[cfg(feature = "melior")]
#[test]
fn verify_recomputes_the_module_text_hash() {
    let out = temp_dir("verify-module-hash");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit-mlir"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("optimize"),
        out.join(tessera::stages::MLIR_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[s("verify"), out_arg.clone()]);

    // Perturb the stored module text only. The artifact's own provenance still
    // agrees with itself, so only a recomputation can catch this.
    let path = out.join(tessera::stages::OPTIMIZED_FILE);
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
    let module = value["module"].as_str().expect("module text").to_string();
    assert!(
        module.contains("func.func"),
        "the stored module must be real MLIR text"
    );
    value["module"] = serde_json::Value::String(module.replacen("func.func", "func.fun", 1));
    std::fs::write(&path, serde_json::to_string(&value).expect("serialize")).expect("write");

    let report = run_err(&[s("verify"), out_arg]);
    assert!(
        report.contains("module hash"),
        "the recomputed module hash must be reported as the broken check: {report}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// A primitive artifact is only as good as its link to the module it claims to
/// come from; editing that link must be detected.
#[cfg(feature = "melior")]
#[test]
fn verify_detects_a_tampered_primitive_artifact() {
    let out = temp_dir("verify-primitives");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit-mlir"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("optimize"),
        out.join(tessera::stages::MLIR_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("solve"),
        s("--mlir"),
        out.join(tessera::stages::OPTIMIZED_FILE)
            .display()
            .to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[s("verify"), out_arg.clone()]);

    let path = out.join(tessera::stages::PRIMITIVES_FILE);
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
    value["provenance"]["mlir_module_hash"] =
        serde_json::Value::String("fnv1a64:deadbeefdeadbeef".to_string());
    std::fs::write(&path, serde_json::to_string(&value).expect("serialize")).expect("write");

    let report = run_err(&[s("verify"), out_arg]);
    assert!(
        report.contains("descends from the optimized module"),
        "the primitive-to-module link must be the named failure: {report}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// A saved optimized module is a standalone input: it must solve in a directory
/// that holds no graph, and the same artifact must solve again against a changed
/// policy without repeating import or optimization (ADR-0002).
#[cfg(feature = "melior")]
#[test]
fn optimized_artifact_solves_standalone_and_against_a_second_policy() {
    let source = temp_dir("optimized-source");
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        source.display().to_string(),
    ]);
    run_ok(&[
        s("emit-mlir"),
        source
            .join(tessera::stages::GRAPH_FILE)
            .display()
            .to_string(),
        s("--out"),
        source.display().to_string(),
    ]);
    run_ok(&[
        s("optimize"),
        source
            .join(tessera::stages::MLIR_FILE)
            .display()
            .to_string(),
        s("--out"),
        source.display().to_string(),
    ]);

    for policy in ["emit.toml", "explore.toml"] {
        let out = temp_dir(&format!(
            "optimized-standalone-{}",
            policy.replace('.', "-")
        ));
        // Only the optimized artifact is carried over: no graph, no module text file.
        std::fs::copy(
            source.join(tessera::stages::OPTIMIZED_FILE),
            out.join(tessera::stages::OPTIMIZED_FILE),
        )
        .expect("carry the optimized artifact");
        assert!(!out.join(tessera::stages::GRAPH_FILE).exists());

        let solved = run_ok(&[
            s("solve"),
            s("--mlir"),
            out.join(tessera::stages::OPTIMIZED_FILE)
                .display()
                .to_string(),
            s("--target"),
            fx("targets/ti60f225.toml"),
            s("--policy"),
            fx(&format!("policies/{policy}")),
            s("--out"),
            out.display().to_string(),
        ]);
        assert!(solved.contains("layout: feasible"), "{policy}: {solved}");
        let verified = run_ok(&[s("verify"), out.display().to_string()]);
        assert!(verified.contains("verify: ok"), "{policy}: {verified}");
        let _ = std::fs::remove_dir_all(&out);
    }
    let _ = std::fs::remove_dir_all(&source);
}

/// The real 29-layer export through the whole chain, at the smallest resolution
/// that fits the Ti60 reference budget: import -> solve -> verify -> emit ->
/// simulate, with the per-step trace the generated testbench writes. The
/// acceptance script requires this run to report PASS; the test itself only
/// requires that it does not report FAIL, because the simulator is optional.
#[test]
fn real_export_runs_the_whole_chain_and_dumps_a_trace() {
    let ink = fixtures().join("ink_dw");
    if !ink.join("manifest.json").exists() {
        eprintln!("SKIP: no real export at {}", ink.display());
        return;
    }
    let out = temp_dir("ink-chain");
    let out_arg = out.display().to_string();
    let layout = out.join(tessera::stages::LAYOUT_FILE).display().to_string();

    let imported = run_ok(&[
        s("import"),
        ink.display().to_string(),
        s("--input-shape"),
        s("1,8,8,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(imported.contains("29 layers"), "{imported}");

    let solved = run_ok(&[
        s("solve"),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("37 steps"), "{solved}");
    assert!(solved.contains("certainty"), "{solved}");

    let verified = run_ok(&[s("verify"), out_arg.clone()]);
    assert!(verified.contains("verify: ok"), "{verified}");

    let emitted = run_ok(&[
        s("emit"),
        layout.clone(),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(emitted.contains("full-graph backend"), "{emitted}");
    assert!(emitted.contains("arena"), "{emitted}");

    let simulated = run_ok(&[
        s("simulate"),
        layout,
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(
        !simulated.contains("simulation: FAIL"),
        "the real model's RTL must not fail its own testbench:\n{simulated}"
    );
    if simulated.contains("simulation: PASS") {
        assert!(simulated.contains("mismatches=0"), "{simulated}");
        let trace = out.join("sim").join("sim_outputs.txt");
        assert!(
            trace.exists(),
            "a simulated run must leave the per-step trace at {}",
            trace.display()
        );
        let text = std::fs::read_to_string(&trace).expect("read the trace");
        // A PASS must be re-readable per step, so the trace needs one block per
        // scheduled step: a bare line count would accept a trace that carries only
        // the frame input and output.
        let blocks = text
            .lines()
            .filter(|line| line.contains(" step ") && !line.contains("input"))
            .count();
        assert!(
            blocks >= 37,
            "the trace must carry a block per scheduled step of the real model, found {blocks}"
        );
    } else {
        eprintln!("SKIP: no simulator; the real chain was verified without a trace");
    }

    let _ = std::fs::remove_dir_all(&out);
}

/// The allocator may reuse words, so the invariant is not "no two slots overlap"
/// but "no two *simultaneously live* slots overlap". This test builds the
/// violating case deliberately (two slots given the same address and an
/// overlapping live range) and asserts `verify` rejects it, so the check is known
/// to be able to fail rather than merely present.
#[test]
fn verify_rejects_simultaneously_live_slots_that_share_addresses() {
    let out = temp_dir("arena-overlap");
    let out_arg = out.display().to_string();
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("solve"),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        out_arg.clone(),
    ]);
    run_ok(&[
        s("emit"),
        out.join(tessera::stages::LAYOUT_FILE).display().to_string(),
        s("--graph"),
        out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    let verified = run_ok(&[s("verify"), out_arg.clone()]);
    assert!(
        verified.contains("simultaneous address disjoint"),
        "the check must be part of a passing chain too:\n{verified}"
    );

    let path = out
        .join(tessera::stages::RTL_DIR)
        .join(tessera::stages::RTL_METADATA_FILE);
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read the bundle"))
            .expect("parse");
    let slots = value["full_graph"]["slots"]
        .as_array_mut()
        .expect("the bundle carries arena slots");
    assert!(slots.len() >= 2, "at least two slots are expected");
    let offset = slots[0]["offset"].as_u64().expect("offset");
    let elements = slots[0]["elements"].as_u64().expect("elements");
    assert!(elements > 0, "a slot holds at least one word");
    slots[1]["offset"] = serde_json::json!(offset);
    slots[1]["elements"] = serde_json::json!(elements);
    for slot in slots.iter_mut() {
        slot["live"] = serde_json::json!([0, u32::MAX]);
    }
    std::fs::write(&path, serde_json::to_string(&value).expect("serialize")).expect("write");

    let report = run_err(&[s("verify"), out_arg]);
    assert!(
        report.contains("simultaneous address disjoint"),
        "the overlapping pair must be reported by name: {report}"
    );
    assert!(
        report.contains("both live over steps"),
        "the report must say why they conflict: {report}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// A graph imported into one directory must solve into another without repeating
/// `--input-shape`: the shape recorded at import time belongs to the artifact, and
/// the stage directory a solve writes to may be empty. (This caught a real bug:
/// the resolver looked at the staged *copy*, i.e. at the output directory.)
#[test]
fn solve_reads_the_recorded_input_shape_from_the_importing_directory() {
    let imported = temp_dir("shape-import");
    let solved = temp_dir("shape-solve");
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        imported.display().to_string(),
    ]);
    assert!(imported.join(tessera::stages::INPUT_SHAPE_FILE).exists());

    let out = run_ok(&[
        s("solve"),
        s("--graph"),
        imported
            .join(tessera::stages::GRAPH_FILE)
            .display()
            .to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/emit.toml"),
        s("--out"),
        solved.display().to_string(),
    ]);
    assert!(out.contains("layout: feasible"), "{out}");
    // The solving directory becomes self-contained for later stages.
    assert!(
        solved.join(tessera::stages::INPUT_SHAPE_FILE).exists()
            && solved.join(tessera::stages::TARGET_FILE).exists(),
        "the stage directory must record the shape and the constraints it used"
    );
    let verified = run_ok(&[s("verify"), solved.display().to_string()]);
    assert!(verified.contains("verify: ok"), "{verified}");

    let _ = std::fs::remove_dir_all(&imported);
    let _ = std::fs::remove_dir_all(&solved);
}
