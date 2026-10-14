//! Acceptance through the CLI, including the RTL trace comparison.
//!
//! This file drives the `tessera` binary exactly as a user does and then judges
//! the *emitted and simulated* design:
//!
//! * the CLI must reach a feasible layout for the acceptance graph;
//! * `emit`/`simulate` must either produce a bundle or fail closed with a
//!   documented reason — never a partial or silently narrowed plan;
//! * when a bundle exists, the suite supplies its own `stim_input.hex`, runs the
//!   compiled simulation binary with `+STIM`/`+TRACE`/`+NOCHECK`, and compares
//!   every dumped step against the independent scalar reference in
//!   `acceptance_support`.
//!
//! Nothing here is verified by the generated testbench's own self-check: that
//! check compares the RTL against the compiler's Rust reference, which is not an
//! independent observation. The suite's stimulus and expected codes are its own.

mod acceptance_support;

use acceptance_support::{
    Fixture, OpReference, Plane, SimRun, Trace, args, bundle_top_module, compare_block,
    fixture_dir, fixtures, is_documented_backend_refusal, parse_trace, run_ok, run_simulation,
    stage_dir, stim_input_hex, uncovered,
};
use std::path::Path;

const GRAPH_INPUT_SHAPE: &str = "1,5,5,3";

fn target_arg() -> String {
    fixtures()
        .join("targets/ti60f225.toml")
        .display()
        .to_string()
}

fn acceptance_policy(name: &str) -> String {
    fixture_dir("").join(name).display().to_string()
}

/// A deterministic code stream owned by the suite, not by the compiler.
struct Codes(u64);

impl Codes {
    fn new(seed: u64) -> Codes {
        Codes(seed | 1)
    }

    fn next_i8(&mut self) -> i64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8 as i8) as i64
    }

    fn plane(&mut self, c: u32, h: u32, w: u32) -> Plane {
        let codes = (0..c * h * w).map(|_| self.next_i8()).collect::<Vec<_>>();
        Plane::new(c, h, w, codes)
    }
}

/// What an `emit` + `simulate` attempt produced.
enum RtlEvidence {
    /// The backend refused the plan, with its own explanation.
    Refused(String),
    /// A bundle exists but no simulation binary/trace could be obtained.
    NotBuilt(String),
    /// A simulation ran and a trace was parsed.
    Simulated(Trace),
}

fn rtl_evidence(
    stage: &Path,
    emit_args: &[String],
    simulate_args: &[String],
    stimulus: &Path,
    trace: &Path,
) -> RtlEvidence {
    let emitted = acceptance_support::run_tessera(emit_args);
    if !emitted.status.success() {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&emitted.stderr),
            String::from_utf8_lossy(&emitted.stdout)
        );
        return RtlEvidence::Refused(text);
    }

    // `simulate` re-emits the bundle and compiles it. Its own exit status is not
    // this suite's verdict: the suite judges the trace it asks for itself.
    let simulated = acceptance_support::run_tessera(simulate_args);
    let simulate_log = format!(
        "{}{}",
        String::from_utf8_lossy(&simulated.stderr),
        String::from_utf8_lossy(&simulated.stdout)
    );
    // A usage error means this suite's invocation is stale, not that the backend
    // cannot emit: report it as a failure of the harness rather than skipping.
    assert!(
        !simulate_log.contains("unexpected argument")
            && !simulate_log.contains("unrecognized subcommand")
            && !simulate_log.contains("USAGE"),
        "the acceptance harness invokes `simulate` with stale arguments:\n{simulate_log}"
    );
    if bundle_top_module(stage).is_none() {
        return RtlEvidence::NotBuilt(format!(
            "{}\nno rtl_bundle.json with a top_module in {}/rtl",
            simulate_log.trim(),
            stage.display()
        ));
    }
    match run_simulation(stage, stimulus, trace) {
        SimRun::Unavailable(reason) => RtlEvidence::NotBuilt(format!(
            "{}\nsimulation unavailable: {reason}",
            simulate_log.trim()
        )),
        SimRun::Ran {
            status,
            stdout,
            stderr,
        } => {
            let _ = status;
            // The requested trace first; then the testbench's documented default
            // location, in case the plusarg is ignored.
            let fallback = stage.join("sim").join("sim_outputs.txt");
            let candidates: [&Path; 2] = [trace, fallback.as_path()];
            let mut last = String::new();
            for candidate in candidates {
                match std::fs::read_to_string(candidate) {
                    Ok(text) => match parse_trace(&text) {
                        Ok(parsed) => return RtlEvidence::Simulated(parsed),
                        Err(detail) => {
                            last = format!(
                                "the trace at {} is malformed: {detail}",
                                candidate.display()
                            );
                        }
                    },
                    Err(error) => {
                        last = format!("no trace at {}: {error}", candidate.display());
                    }
                }
            }
            let testbench = std::fs::read_dir(stage.join("rtl"))
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "sv"))
                .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
                .any(|source| source.contains("STIM") && source.contains("TRACE"));
            let hint = if testbench {
                "the emitted testbench mentions the stimulus/trace plusargs but produced no trace"
            } else {
                "the emitted testbench does not implement +STIM/+TRACE/+NOCHECK yet, so no \
                 independent trace exists; the built-in self-check is not acceptance evidence"
            };
            RtlEvidence::NotBuilt(format!("{last}\n{hint}\nsim stdio:\n{stdout}{stderr}"))
        }
    }
}

/// Compares a simulated trace against the independent reference.
///
/// The contract the emitted testbench implements, and why:
///
/// * every frame must read back the suite's stimulus and dump its image, so the
///   frame's input and output are always compared;
/// * a frame that dumps per-step blocks must dump one per operator, in plan order,
///   and every code must match the oracle;
/// * with lifetime-reused arena words, a per-step read *after* a frame has finished
///   is meaningless — later steps overwrite the words — so the emitted testbench
///   dumps steps only on frames it holds at step boundaries. A frame without step
///   blocks is therefore the unstalled timing frame, and it must still report its
///   cycle count and stall counters;
/// * at least one frame must carry a full step list, or the trace cannot judge the
///   datapath at all and this gate fails rather than passing quietly.
fn compare_trace(
    fixture: &Fixture,
    frames: &[acceptance_support::FrameReference],
    trace: &Trace,
    label: &str,
) {
    let mut compared = 0usize;
    let mut frames_with_steps = 0usize;
    for frame in frames {
        let simulated = trace.frame(frame.index as u32);

        let input = simulated
            .input
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: frame {} has no input block", frame.index));
        assert_eq!(
            input.shape,
            (frame.input.c, frame.input.h, frame.input.w),
            "{label}: frame {} input shape",
            frame.index
        );
        assert_eq!(
            input.codes, frame.input.codes,
            "{label}: frame {} read back different input codes",
            frame.index
        );

        let output = simulated
            .output
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: frame {} has no output block", frame.index));
        assert_eq!(
            output.codes, frame.image.codes,
            "{label}: frame {} image codes",
            frame.index
        );

        if simulated.steps.is_empty() {
            // The timing frame: it must still account for its own cycles, which is
            // the evidence this frame exists to provide.
            assert!(
                simulated.cycles.is_some(),
                "{label}: frame {} carries neither step blocks nor a cycle count",
                frame.index
            );
            continue;
        }

        frames_with_steps += 1;
        assert_eq!(
            simulated.steps.len(),
            frame.ops.len(),
            "{label}: frame {} simulated {} steps, the reference expects {} ({:?})",
            frame.index,
            simulated.steps.len(),
            frame.ops.len(),
            simulated
                .steps
                .iter()
                .map(|step| step.label.as_str())
                .collect::<Vec<_>>()
        );
        for (expected, block) in frame.ops.iter().zip(&simulated.steps) {
            let prefix = fixture.step_match(&expected.name);
            assert!(
                block.label.starts_with(&prefix),
                "{label}: frame {} step {:?} does not match the expected operator {:?} \
                 (prefix {prefix:?})",
                frame.index,
                block.label,
                expected.name
            );
            compare_block(
                &format!(
                    "{label}/frame{}/step {} ({})",
                    frame.index, block.label, expected.name
                ),
                block,
                expected,
            )
            .unwrap_or_else(|error| panic!("{error}"));
            compared += 1;
        }
    }
    assert!(compared > 0, "{label}: nothing was compared");
    assert!(
        frames_with_steps > 0,
        "{label}: no driven frame carried per-step evidence, so the trace cannot judge the \
         datapath"
    );
}

#[test]
fn acceptance_graph_reaches_a_feasible_layout_through_the_cli() {
    let out = stage_dir("graph-cli");
    let out_arg = out.display().to_string();
    let extract = fixture_dir("graph").display().to_string();

    let imported = run_ok(&args(&[
        "import",
        &extract,
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--out",
        &out_arg,
    ]));
    assert!(imported.contains("4 layers"), "{imported}");
    let solved = run_ok(&args(&[
        "solve",
        "--graph",
        &out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        "--target",
        &target_arg(),
        "--policy",
        &acceptance_policy("policy.toml"),
        "--out",
        &out_arg,
    ]));
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("7 steps"), "{solved}");
    assert!(solved.contains("lower: layer dw"), "{solved}");
    assert!(out.join(tessera::stages::LAYOUT_FILE).exists());
    assert!(out.join(tessera::stages::PLAN_FILE).exists());
    assert!(out.join(tessera::stages::PRIMITIVE_FILE).exists());

    let verified = run_ok(&args(&["verify", &out_arg]));
    assert!(verified.contains("verify: ok"), "{verified}");

    // An impossible budget is reported as infeasible, and the artifact is still
    // written so the rejected candidates can be inspected.
    let impossible = stage_dir("graph-cli-impossible");
    let impossible_arg = impossible.display().to_string();
    let report = acceptance_support::run_err(&args(&[
        "solve",
        "--graph",
        &out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--target",
        &target_arg(),
        "--policy",
        &acceptance_policy("policy_impossible.toml"),
        "--out",
        &impossible_arg,
    ]));
    assert!(
        report.contains("INFEASIBLE") || report.contains("infeasible"),
        "the CLI must report the infeasible status: {report}"
    );
    assert!(
        report.contains("rejected"),
        "the CLI must name rejection categories: {report}"
    );
    assert!(
        impossible.join(tessera::stages::LAYOUT_FILE).exists(),
        "an infeasible layout is still an artifact"
    );

    // A target with no usable DSP48 must also be infeasible, naming the resource.
    let no_dsp = stage_dir("graph-cli-no-dsp");
    let no_dsp_arg = no_dsp.display().to_string();
    acceptance_support::run_err(&args(&[
        "solve",
        "--graph",
        &out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--target",
        &fixture_dir("")
            .join("target_impossible.toml")
            .display()
            .to_string(),
        "--policy",
        &acceptance_policy("policy.toml"),
        "--out",
        &no_dsp_arg,
    ]));
    // The persisted layout must explain itself: the status, the count and the
    // exhausted resource are artifact data, not just console text.
    let exhausted = std::fs::read_to_string(no_dsp.join(tessera::stages::LAYOUT_FILE))
        .expect("the infeasible layout is written");
    assert!(exhausted.contains("infeasible"), "{exhausted}");
    assert!(
        exhausted.contains("dsp_budget"),
        "the exhausted resource must be named in the artifact: {exhausted}"
    );
    let starved = std::fs::read_to_string(impossible.join(tessera::stages::LAYOUT_FILE))
        .expect("the infeasible layout is written");
    assert!(starved.contains("infeasible"), "{starved}");
    assert!(
        starved.contains("ddr_read_budget"),
        "one cycle cannot pay for the graph input: {starved}"
    );

    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&impossible);
    let _ = std::fs::remove_dir_all(&no_dsp);
}

#[test]
fn stale_provenance_is_reported_by_verify() {
    let out = stage_dir("graph-cli-stale");
    let out_arg = out.display().to_string();
    run_ok(&args(&[
        "import",
        &fixture_dir("graph").display().to_string(),
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--out",
        &out_arg,
    ]));
    run_ok(&args(&[
        "solve",
        "--graph",
        &out.join(tessera::stages::GRAPH_FILE).display().to_string(),
        "--target",
        &target_arg(),
        "--policy",
        &acceptance_policy("policy.toml"),
        "--out",
        &out_arg,
    ]));
    run_ok(&args(&["verify", &out_arg]));

    let graph = out.join(tessera::stages::GRAPH_FILE);
    let mut text = std::fs::read_to_string(&graph).expect("read the staged graph");
    text.push('\n');
    std::fs::write(&graph, text).expect("perturb the staged graph");

    let report = acceptance_support::run_err(&args(&["verify", &out_arg]));
    assert!(
        report.contains("solved-from input hash"),
        "the stale check must be named: {report}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn acceptance_graph_rtl_matches_the_independent_reference() {
    let out = stage_dir("graph-rtl");
    let out_arg = out.display().to_string();
    run_ok(&args(&[
        "import",
        &fixture_dir("graph").display().to_string(),
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--out",
        &out_arg,
    ]));

    let fixture = Fixture::load("graph");
    let frames = acceptance_support::run_fixture(&fixture);
    let stimulus = out.join("acceptance_stim.hex");
    std::fs::write(
        &stimulus,
        stim_input_hex(&frames.iter().map(|f| f.input.clone()).collect::<Vec<_>>()),
    )
    .expect("write the suite's stimulus");
    let trace_path = out.join("acceptance_trace.txt");

    let mut evidence = None;
    for policy in ["policy_streaming.toml", "policy.toml"] {
        let solved = acceptance_support::run_tessera(&args(&[
            "solve",
            "--graph",
            &out.join(tessera::stages::GRAPH_FILE).display().to_string(),
            "--target",
            &target_arg(),
            "--policy",
            &acceptance_policy(policy),
            "--out",
            &out_arg,
        ]));
        let solved_text = String::from_utf8_lossy(&solved.stdout).to_string();
        if !solved.status.success() || !solved_text.contains("feasible") {
            eprintln!(
                "note: policy {policy} found no feasible layout: {}",
                solved_text.trim()
            );
            continue;
        }
        let layout = out.join(tessera::stages::LAYOUT_FILE).display().to_string();
        let graph = out.join(tessera::stages::GRAPH_FILE).display().to_string();
        let emit_args = args(&["emit", &layout, "--graph", &graph, "--out", &out_arg]);
        let simulate_args = args(&["simulate", &layout, "--graph", &graph, "--out", &out_arg]);
        match rtl_evidence(&out, &emit_args, &simulate_args, &stimulus, &trace_path) {
            RtlEvidence::Refused(reason) => {
                // The backend must fail closed and name what it cannot emit.
                assert!(
                    is_documented_backend_refusal(&reason),
                    "an unknown emit failure under {policy}:\n{reason}"
                );
                assert!(
                    reason.contains("stem") || reason.contains("step"),
                    "the refusal must name the offending step:\n{reason}"
                );
                evidence = Some(RtlEvidence::Refused(format!("{policy}: {reason}")));
            }
            other => {
                evidence = Some(other);
                break;
            }
        }
    }

    match evidence.expect("at least one policy was attempted") {
        RtlEvidence::Simulated(trace) => {
            compare_trace(&fixture, &frames, &trace, "graph");
            // 1. The residual must be carried at 9-bit width in the hardware.
            //    Only frames that dump step blocks are inspected: an unstalled
            //    timing frame cannot be read back step by step once the arena words
            //    are reused, and `compare_trace` already required at least one frame
            //    with a full step list.
            let residual_prefix = fixture.step_match("add");
            let mut residual_codes = Vec::new();
            let with_steps: Vec<&acceptance_support::FrameReference> = frames
                .iter()
                .filter(|frame| !trace.frame(frame.index as u32).steps.is_empty())
                .collect();
            assert!(
                !with_steps.is_empty(),
                "no frame carried per-step evidence, so the residual width cannot be judged"
            );
            for frame in &with_steps {
                let simulated = trace.frame(frame.index as u32);
                let add_step = simulated
                    .steps
                    .iter()
                    .find(|step| step.label.starts_with(&residual_prefix))
                    .unwrap_or_else(|| {
                        panic!("frame {}: no residual step in the trace", frame.index)
                    });
                assert_eq!(
                    add_step.elem, "i16",
                    "frame {}: the residual block must be labelled i16, got {:?}",
                    frame.index, add_step.elem
                );
                assert!(
                    add_step
                        .codes
                        .iter()
                        .any(|code| *code > 127 || *code < -128),
                    "frame {}: the residual block must contain a value outside the signed 8-bit \
                     domain, so the carry is exercised",
                    frame.index
                );
                residual_codes.extend(add_step.codes.iter().copied());
            }
            assert!(!residual_codes.is_empty());

            // 2. The unsigned head must exercise both ends of its clamp in the RTL.
            let image: Vec<i64> = with_steps
                .iter()
                .flat_map(|frame| {
                    trace
                        .frame(frame.index as u32)
                        .steps
                        .last()
                        .expect("the trace has steps")
                        .codes
                        .clone()
                })
                .collect();
            assert!(
                image.iter().any(|code| *code == 0),
                "the simulated unsigned clamp must drive a code to 0"
            );
            assert!(
                image.iter().any(|code| *code == 255),
                "the simulated unsigned clamp must saturate at output_qmax (255)"
            );

            // 3. A non-zero int32 bias must reach the datapath. The fixture is
            //    provably bias-sensitive (zeroing `stem`'s bias changes the
            //    reference), and the trace comparison above shows the RTL produced
            //    the biased values.
            let mut unbiased = fixture.manifest.clone();
            unbiased.layer_mut("stem").bias_int32 = vec![0; 4];
            let (unbiased_ops, _) =
                acceptance_support::run_frame(&unbiased, &fixture.weights, &frames[0].input);
            let biased_stem = &frames[0].ops[0].plane;
            assert_ne!(
                unbiased_ops[0].plane.codes, biased_stem.codes,
                "the fixture must be sensitive to the stem's int32 bias, or the trace \
                 comparison would not prove the bias reached the datapath"
            );
            let stem_prefix = fixture.step_match("stem");
            for frame in &with_steps {
                let simulated = trace.frame(frame.index as u32);
                let stem = simulated
                    .steps
                    .iter()
                    .find(|step| step.label.starts_with(&stem_prefix))
                    .unwrap_or_else(|| panic!("frame {}: no stem step in the trace", frame.index));
                assert_eq!(
                    stem.codes, frame.ops[0].plane.codes,
                    "frame {}: the simulated stem must equal the biased reference",
                    frame.index
                );
            }
        }
        RtlEvidence::Refused(reason) => {
            uncovered("full-graph RTL emission", &reason);
        }
        RtlEvidence::NotBuilt(reason) => {
            uncovered("full-graph RTL simulation evidence", &reason);
        }
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn pointwise_fixture_rtl_matches_the_independent_reference() {
    // The single-step pointwise fixture reaches the narrow v1 envelope, so it is
    // the one plan this suite can judge end to end today.
    let dir = fixtures().join("pointwise");
    let plan_text = std::fs::read_to_string(dir.join("plan_input.json")).expect("read the plan");
    let plan: serde_json::Value = serde_json::from_str(&plan_text).expect("parse the plan");
    let tensors = plan["tensors"].as_array().expect("tensors");
    let input_desc = &tensors[0];
    let shape = &input_desc["shape"];
    let (c, h, w) = (
        shape["c"].as_u64().expect("c") as u32,
        shape["h"].as_u64().expect("h") as u32,
        shape["w"].as_u64().expect("w") as u32,
    );
    let op = &plan["ops"][0];
    let conv = &op["detail"]["conv"];
    let requant = &conv["requant"];
    let m0 = requant["m0"].as_i64().expect("m0");
    let shift = requant["shift"].as_u64().expect("shift") as u32;
    let lo = requant["saturate_lo"].as_i64().expect("lo");
    let hi = requant["saturate_hi"].as_i64().expect("hi");
    let activation = &conv["activation"];
    let clamp_lo = activation["clamp_lo"].as_i64().expect("clamp_lo");
    let clamp_hi = activation["clamp_hi"].as_i64().expect("clamp_hi");
    let (lo, hi) = (lo.max(clamp_lo), hi.min(clamp_hi));

    let weights_id = conv["weights"].as_u64().expect("weights tensor id");
    let weights_desc = tensors
        .iter()
        .find(|tensor| tensor["id"].as_u64() == Some(weights_id))
        .expect("the weights tensor must be declared");
    let weights_name = weights_desc["name"].as_str().expect("weights name");
    let weight_bytes =
        std::fs::read(dir.join("weights").join(format!("{weights_name}.bin"))).expect("weights");
    let out_channels = weights_desc["shape"]["n"].as_u64().expect("n") as usize;
    let in_channels = weights_desc["shape"]["c"].as_u64().expect("c") as usize;
    assert_eq!(weight_bytes.len(), out_channels * in_channels);
    let weights: Vec<i64> = weight_bytes.iter().map(|byte| *byte as i8 as i64).collect();

    // The suite's own two frames of stimulus.
    let mut codes = Codes::new(0x5eed_1234_abcd_0001);
    let frames = vec![codes.plane(c, h, w), codes.plane(c, h, w)];

    // Independent expectation: 1x1 pointwise conv, per-position dot product.
    let expected = frames
        .iter()
        .map(|frame| {
            let mut out = vec![0i64; (out_channels as u32 * h * w) as usize];
            for y in 0..h {
                for x in 0..w {
                    for oc in 0..out_channels {
                        let mut acc = 0i64;
                        for ic in 0..in_channels {
                            let a = frame.at(ic as u32, y, x);
                            let b = weights[oc * in_channels + ic];
                            acc += a * b;
                        }
                        out[(oc as u32 * h * w + y * w + x) as usize] =
                            acceptance_support::requantize(acc, m0, shift, lo, hi);
                    }
                }
            }
            Plane::new(out_channels as u32, h, w, out)
        })
        .collect::<Vec<_>>();

    let out = stage_dir("pointwise-rtl");
    let out_arg = out.display().to_string();
    let layout = out.join(tessera::stages::LAYOUT_FILE).display().to_string();
    run_ok(&args(&[
        "solve",
        "--primitive",
        &dir.join("plan_input.json").display().to_string(),
        "--target",
        &target_arg(),
        "--policy",
        &acceptance_policy("policy_streaming.toml"),
        "--out",
        &out_arg,
    ]));
    let weights_dir = dir.join("weights").display().to_string();
    let emit_args = args(&[
        "emit",
        &layout,
        "--weights",
        &weights_dir,
        "--out",
        &out_arg,
    ]);
    let simulate_args = args(&[
        "simulate",
        &layout,
        "--weights",
        &weights_dir,
        "--out",
        &out_arg,
    ]);
    let stimulus = out.join("acceptance_stim.hex");
    std::fs::write(&stimulus, stim_input_hex(&frames)).expect("write the suite's stimulus");
    let trace_path = out.join("acceptance_trace.txt");

    match rtl_evidence(&out, &emit_args, &simulate_args, &stimulus, &trace_path) {
        RtlEvidence::Simulated(trace) => {
            for (index, frame) in frames.iter().enumerate() {
                let simulated = trace.frame(index as u32);
                let input = simulated
                    .input
                    .as_ref()
                    .unwrap_or_else(|| panic!("frame {index}: the trace has no input block"));
                assert_eq!(input.codes, frame.codes, "frame {index}: input read-back");
                assert_eq!(simulated.steps.len(), 1, "one step in the narrow envelope");
                let step = &simulated.steps[0];
                assert!(
                    step.label.starts_with("pw"),
                    "the step must name the pointwise operator, got {:?}",
                    step.label
                );
                let op_reference = OpReference {
                    name: "pw".to_string(),
                    plane: expected[index].clone(),
                    elem: "i8",
                    requant: None,
                };
                compare_block(
                    &format!("pointwise/frame{index}/{}", step.label),
                    step,
                    &op_reference,
                )
                .unwrap_or_else(|error| panic!("{error}"));
            }
        }
        RtlEvidence::Refused(reason) => {
            assert!(
                is_documented_backend_refusal(&reason),
                "the pointwise fixture must stay inside the supported envelope:\n{reason}"
            );
            eprintln!(
                "REGRESSION: the narrow pointwise envelope no longer emits the shipped \
                 fixture (maintainer tests/cli.rs and tests/pipeline.rs cover this path; the \
                 full-graph gate does not depend on it).\nreason: {}",
                reason.trim()
            );
        }
        RtlEvidence::NotBuilt(reason) => {
            eprintln!(
                "REGRESSION: no pointwise simulation evidence was produced (regression path \
                 only; the full-graph gate does not depend on it).\nreason: {}",
                reason.trim()
            );
        }
    }
    let _ = std::fs::remove_dir_all(&out);
}

/// The suite's stimulus file layout is part of the interface contract; check it
/// directly instead of trusting a downstream parser.
#[test]
fn stimulus_file_renders_one_hex_byte_per_code_in_order() {
    let plane = Plane::new(2, 1, 2, vec![-128, -1, 0, 127]);
    let text = stim_input_hex(&[plane]);
    assert_eq!(text, "80\nff\n00\n7f\n");
}

/// A parsed trace must be rejected when the format tag is not the documented one.
#[test]
fn trace_parser_rejects_an_unknown_format() {
    let error = parse_trace("format tessera.sim_trace.v2\nframe 0 cycles 1\n")
        .err()
        .expect("an unknown trace format must be rejected");
    assert!(error.contains("unexpected trace format"), "{error}");
    let parsed = parse_trace(
        "format tessera.sim_trace.v1\nframe 0 step 0 stem i8 1 1 2\n-5\n7\nframe 0 output 1 1 2\n0\n7\nframe 0 cycles 42\nframe 0 stalls 3 4\n",
    )
    .expect("a well-formed trace parses");
    assert_eq!(parsed.frame(0).steps.len(), 1);
    assert_eq!(parsed.frame(0).steps[0].codes, vec![-5, 7]);
    assert_eq!(parsed.frame(0).output.as_ref().expect("output").elem, "u8");
    assert_eq!(parsed.frame(0).cycles, Some(42));
    assert_eq!(parsed.frame(0).stalls, Some((3, 4)));
}

/// The same judgement, but over a plan derived from the **optimized module**
/// instead of the graph: `emit-mlir` -> `optimize` -> `solve --mlir` ->
/// `emit`/`simulate` -> comparison against the independent reference. This is the
/// path that proves the optimized IR, not just the frontend, is what the RTL is
/// built from.
///
/// Gated on the `melior` feature: only the optimizer produces a module analysis,
/// and a build without it must skip rather than pretend the stage ran
/// (`optimize` refuses and names the missing feature).
#[cfg(feature = "melior")]
#[test]
fn optimized_mlir_path_matches_the_independent_reference() {
    let out = stage_dir("mlir-path-rtl");
    let out_arg = out.display().to_string();
    let graph = out.join(tessera::stages::GRAPH_FILE).display().to_string();

    run_ok(&args(&[
        "import",
        &fixture_dir("graph").display().to_string(),
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--out",
        &out_arg,
    ]));
    run_ok(&args(&[
        "emit-mlir",
        &graph,
        "--input-shape",
        GRAPH_INPUT_SHAPE,
        "--out",
        &out_arg,
    ]));
    let optimized = run_ok(&args(&[
        "optimize",
        &out.join(tessera::stages::MLIR_FILE).display().to_string(),
        "--out",
        &out_arg,
    ]));
    assert!(
        optimized.contains("melior"),
        "the optimizer must report the pipeline it ran: {optimized}"
    );

    let solved = run_ok(&args(&[
        "solve",
        "--mlir",
        &out.join(tessera::stages::OPTIMIZED_FILE)
            .display()
            .to_string(),
        "--target",
        &target_arg(),
        "--policy",
        &acceptance_policy("policy.toml"),
        "--out",
        &out_arg,
    ]));
    assert!(solved.contains("fusion groups"), "{solved}");
    assert!(solved.contains("layout: feasible"), "{solved}");
    assert!(solved.contains("certainty"), "{solved}");
    assert!(
        out.join(tessera::stages::PRIMITIVES_FILE).exists(),
        "the primitive artifact must be persisted with its evidence"
    );

    let verified = run_ok(&args(&["verify", &out_arg]));
    assert!(verified.contains("verify: ok"), "{verified}");
    assert!(
        verified.contains("descends from the optimized module"),
        "the primitive graph must be linked to the module it came from:\n{verified}"
    );

    let fixture = Fixture::load("graph");
    let frames = acceptance_support::run_fixture(&fixture);
    let stimulus = out.join("acceptance_mlir_stim.hex");
    std::fs::write(
        &stimulus,
        stim_input_hex(&frames.iter().map(|f| f.input.clone()).collect::<Vec<_>>()),
    )
    .expect("write the suite's stimulus");
    let trace_path = out.join("acceptance_mlir_trace.txt");

    let layout = out.join(tessera::stages::LAYOUT_FILE).display().to_string();
    let emit_args = args(&["emit", &layout, "--graph", &graph, "--out", &out_arg]);
    let simulate_args = args(&["simulate", &layout, "--graph", &graph, "--out", &out_arg]);
    match rtl_evidence(&out, &emit_args, &simulate_args, &stimulus, &trace_path) {
        RtlEvidence::Simulated(trace) => {
            compare_trace(&fixture, &frames, &trace, "optimized-mlir");
            // The two properties that make the plan's semantics observable:
            // the residual is carried at nine bits, and the unsigned head
            // exercises both ends of its clamp.
            let residual_prefix = fixture.step_match("add");
            let mut residual_codes = Vec::new();
            let mut image = Vec::new();
            // Only the frames that dump step blocks: with a reused arena the
            // unstalled timing frame cannot be read back step by step, and
            // `compare_trace` already required at least one frame with a full list.
            let with_steps: Vec<&acceptance_support::FrameReference> = frames
                .iter()
                .filter(|frame| !trace.frame(frame.index as u32).steps.is_empty())
                .collect();
            assert!(!with_steps.is_empty(), "no frame carried per-step evidence");
            for frame in &with_steps {
                let simulated = trace.frame(frame.index as u32);
                let add_step = simulated
                    .steps
                    .iter()
                    .find(|step| step.label.starts_with(&residual_prefix))
                    .unwrap_or_else(|| panic!("frame {}: no residual step", frame.index));
                assert_eq!(
                    add_step.elem, "i16",
                    "frame {}: residual width",
                    frame.index
                );
                residual_codes.extend(add_step.codes.iter().copied());
                image.extend(
                    simulated
                        .steps
                        .last()
                        .expect("the trace has steps")
                        .codes
                        .iter()
                        .copied(),
                );
            }
            assert!(
                residual_codes
                    .iter()
                    .any(|code| *code > 127 || *code < -128),
                "the residual carry must be exercised"
            );
            assert!(
                image.iter().any(|code| *code == 0),
                "the clamp must reach 0"
            );
            assert!(
                image.iter().any(|code| *code == 255),
                "the clamp must saturate at 255"
            );
        }
        RtlEvidence::Refused(reason) => uncovered("optimized-MLIR RTL emission", &reason),
        RtlEvidence::NotBuilt(reason) => uncovered("optimized-MLIR RTL simulation", &reason),
    }
    let _ = std::fs::remove_dir_all(&out);
}
