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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-cli-{tag}-{}", std::process::id()));
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
fn manifest_pipeline_reaches_a_layout_and_rejects_whole_model_rtl_clearly() {
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
        "{solved}"
    );

    // The narrow v1 backend refuses the whole model, and says which step.
    let rejection = run_err(&[
        s("emit"),
        out.join("layout.tlayout").display().to_string(),
        s("--graph"),
        out.join("graph.tgraph").display().to_string(),
        s("--out"),
        out_arg.clone(),
    ]);
    assert!(rejection.contains("single pointwise step"), "{rejection}");
    assert!(rejection.contains("stem"), "{rejection}");

    let verified = run_ok(&[s("verify"), out_arg]);
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
