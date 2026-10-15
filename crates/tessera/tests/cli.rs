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

#[test]
fn solve_tessera_writes_solved_mlir_for_w8() {
    let imported = temp_dir("tessera-import");
    let emitted = temp_dir("tessera-emit");
    let solved = temp_dir("tessera-solve");
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--out"),
        imported.display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
    ]);
    run_ok(&[
        s("emit-mlir"),
        imported.join("graph.tgraph").display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        emitted.display().to_string(),
    ]);
    let out = run_ok(&[
        s("solve"),
        s("--tessera"),
        emitted.join("module.mlir").display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/explore.toml"),
        s("--out"),
        solved.display().to_string(),
    ]);
    assert!(out.contains("solved:"), "{out}");
    assert!(solved.join("solved.mlir").is_file(), "solved.mlir missing");
    assert!(solved.join("tessera.mlir").is_file(), "tessera.mlir missing");
    let text = std::fs::read_to_string(solved.join("solved.mlir")).expect("read solved");
    assert!(text.contains("tessera.engine") || text.contains("\"tessera.engine\""), "{text}");
    assert!(text.contains("tessera.schedule") || text.contains("\"tessera.schedule\""), "{text}");
}

#[test]
fn solve_requires_tessera_input() {
    let out = temp_dir("solve-no-input");
    let err = run_err(&[
        s("solve"),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/explore.toml"),
        s("--out"),
        out.display().to_string(),
    ]);
    assert!(
        err.contains("--tessera") || err.contains("required"),
        "{err}"
    );
}

#[test]
fn emit_tessera_circt_writes_sv_for_w8() {
    let imported = temp_dir("circt-import");
    let emitted = temp_dir("circt-emit-mlir");
    let solved = temp_dir("circt-solve");
    let rtl = temp_dir("circt-rtl");
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--out"),
        imported.display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
    ]);
    run_ok(&[
        s("emit-mlir"),
        imported.join("graph.tgraph").display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        emitted.display().to_string(),
    ]);
    run_ok(&[
        s("solve"),
        s("--tessera"),
        emitted.join("module.mlir").display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/explore.toml"),
        s("--out"),
        solved.display().to_string(),
    ]);
    let out = run_ok(&[
        s("emit"),
        s("--tessera"),
        solved.join("solved.mlir").display().to_string(),
        s("--graph"),
        imported.join("graph.tgraph").display().to_string(),
        s("--out"),
        rtl.display().to_string(),
        s("--module-prefix"),
        s("w8"),
    ]);
    assert!(out.contains("CIRCT hybrid"), "{out}");
    assert!(out.contains("engines"), "{out}");
    let dut = rtl.join("rtl").join("w8_hybrid_circt.sv");
    assert!(dut.is_file(), "missing {}", dut.display());
    let sv = std::fs::read_to_string(&dut).expect("sv");
    assert!(
        sv.contains("module w8_hybrid_circt") || sv.contains("w8_hybrid_circt"),
        "top missing in {} bytes",
        sv.len()
    );
    assert!(sv.contains("mac_lane") || sv.contains("tessera_mac_lane"), "mac lane missing");
    assert!(sv.contains("sched_core"), "sched_core missing");
    assert!(
        !sv.contains("_engine_0"),
        "stub engine modules must not be emitted"
    );
}

#[test]
fn emit_tessera_circt_bitexact_bundle_for_w8() {
    let imported = temp_dir("circt-be-import");
    let emitted = temp_dir("circt-be-mlir");
    let solved = temp_dir("circt-be-solve");
    let rtl = temp_dir("circt-be-rtl");
    run_ok(&[
        s("import"),
        fx("w8"),
        s("--out"),
        imported.display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
    ]);
    run_ok(&[
        s("emit-mlir"),
        imported.join("graph.tgraph").display().to_string(),
        s("--input-shape"),
        s("1,6,6,3"),
        s("--out"),
        emitted.display().to_string(),
    ]);
    run_ok(&[
        s("solve"),
        s("--tessera"),
        emitted.join("module.mlir").display().to_string(),
        s("--target"),
        fx("targets/ti60f225.toml"),
        s("--policy"),
        fx("policies/explore.toml"),
        s("--out"),
        solved.display().to_string(),
    ]);
    let out = run_ok(&[
        s("emit"),
        s("--tessera"),
        solved.join("solved.mlir").display().to_string(),
        s("--graph"),
        imported.join("graph.tgraph").display().to_string(),
        s("--out"),
        rtl.display().to_string(),
        s("--module-prefix"),
        s("w8"),
    ]);
    assert!(out.contains("CIRCT hybrid"), "{out}");
    let dir = rtl.join("rtl");
    assert!(dir.join("stim_input.hex").is_file(), "stim missing");
    assert!(dir.join("expected.hex").is_file(), "expected missing");
    let sv = std::fs::read_to_string(dir.join("w8_hybrid_circt.sv")).expect("sv");
    assert!(sv.contains("sched_core") || sv.contains("_sched_core"), "sched_core missing");
    assert!(sv.contains("tessera_requant") || sv.contains("requant"), "requant missing");
}

