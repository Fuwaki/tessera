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

