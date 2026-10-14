//! Verilator simulation of the emitted SystemVerilog kernel.
//!
//! Each case lowers a supported kernel into a bundle, materializes it, and runs
//! the generated self-checking testbench against the Rust reference model.
//! Verilator is optional: when it is not installed the test prints a `SKIP`
//! line and passes **without** claiming a verified result.

use std::fs;
use std::path::PathBuf;

use tessera::rtl::{
    ChannelRequant, ElementType, KernelPlanView, PlacementHint, PlanContext, PlannedOp, Provenance,
    QuantPlan, RoundingMode, RtlConfig, RtlPlanSource, SimulationOutcome, SystemVerilogBackend,
    TileGeometry, WeightTensor, simulate_with_verilator, verilator_version,
};

struct SingleKernelPlan {
    view: KernelPlanView,
}

impl RtlPlanSource for SingleKernelPlan {
    fn artifact_format(&self) -> &str {
        "tessera.execution_plan"
    }

    fn schema_version(&self) -> u32 {
        1
    }

    fn provenance(&self) -> Provenance {
        Provenance {
            compiler_version: "0.1.0".to_string(),
            git_revision: "cafebabe".to_string(),
            dirty: false,
            graph_schema_version: 1,
            plan_schema_version: 1,
            target: "ti60f225".to_string(),
            target_revision: "2026-10-14".to_string(),
            input_manifest_sha256: "ab".repeat(32),
            weights_sha256: "cd".repeat(32),
            planner_config_sha256: "ef".repeat(32),
        }
    }

    fn kernels(&self) -> Vec<KernelPlanView> {
        vec![self.view.clone()]
    }
}

/// Builds a supported pointwise kernel with deterministic weights and
/// per-channel requantization that varies across folds and lanes.
fn pointwise_view(layer: &str, simd: u32, noc: u32, k: u32, fold: u32) -> KernelPlanView {
    let tile = TileGeometry { simd, noc, k, fold };
    let entries = (fold * k * noc * simd) as usize;
    // Weights derived from a fixed recurrence so the ROM is non-trivial.
    let mut state: u32 = 0x1234_5678;
    let weights: Vec<i8> = (0..entries)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8 as i8
        })
        .collect();
    let channels: Vec<ChannelRequant> = (0..(fold * noc))
        .map(|channel| ChannelRequant {
            // 2^16 + c*997 stays inside the 17-bit multiplier field.
            multiplier: (1 << 16) + channel * 997,
            shift: 17 + (channel % 4) as u8,
            lo: -128 + channel as i16,
            hi: 127 - channel as i16,
        })
        .collect();
    KernelPlanView {
        layer: layer.to_string(),
        op: PlannedOp::PointwiseDot,
        activation_element: ElementType::I8,
        weight_element: ElementType::I8,
        output_element: ElementType::I8,
        tile,
        quant: QuantPlan {
            activation_zero_point: 0,
            weight_zero_point: 0,
            rounding: RoundingMode::RoundHalfUpMagnitude,
            channels,
            multiplier_error_ppm: None,
        },
        weights: WeightTensor { data: weights },
        weight_placement: PlacementHint::WeightRom,
        activation_placement: PlacementHint::StreamedActivation,
        context: PlanContext::direct(layer),
    }
}

fn work_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tessera_rtl_sim_{}_{}", std::process::id(), tag))
}

/// Lowers a kernel, simulates it, and returns the outcome.
fn simulate(tag: &str, view: KernelPlanView, windows: usize) -> SimulationOutcome {
    let config = RtlConfig {
        fixture_windows: windows,
        ..RtlConfig::default()
    };
    let bundle = SystemVerilogBackend
        .lower(&SingleKernelPlan { view }, &config)
        .expect("supported kernel lowers");
    let dir = work_dir(tag);
    let _ = fs::remove_dir_all(&dir);
    let outcome = simulate_with_verilator(&bundle, &dir).expect("simulator scaffold runs");
    if matches!(outcome, SimulationOutcome::ToolUnavailable { .. }) {
        return outcome;
    }
    if let SimulationOutcome::Failed { log, detail } = &outcome {
        panic!("{tag} failed: {detail}\n{log}");
    }
    let _ = fs::remove_dir_all(&dir);
    outcome
}

/// Runs one case, skipping cleanly when Verilator is absent.
fn run_case(tag: &str, view: KernelPlanView, windows: usize) {
    let outcome = simulate(tag, view, windows);
    match outcome {
        SimulationOutcome::Passed { .. } => {}
        SimulationOutcome::ToolUnavailable { tool, reason } => {
            eprintln!("SKIP {tag}: {tool} unavailable ({reason}); RTL not simulated");
        }
        SimulationOutcome::Failed { log, detail } => {
            panic!("{tag} failed: {detail}\n{log}");
        }
    }
}

#[test]
fn emitted_kernel_matches_the_reference_for_single_fold() {
    if verilator_version().is_none() {
        eprintln!(
            "SKIP emitted_kernel_matches_the_reference_for_single_fold: verilator unavailable"
        );
        return;
    }
    run_case("fold1", pointwise_view("enc.0.pw", 2, 2, 3, 1), 24);
}

#[test]
fn emitted_kernel_matches_the_reference_for_multiple_folds() {
    if verilator_version().is_none() {
        eprintln!(
            "SKIP emitted_kernel_matches_the_reference_for_multiple_folds: verilator unavailable"
        );
        return;
    }
    // Non-power-of-two reduction and fold counts, varied per-channel requant.
    run_case("fold3", pointwise_view("enc.3.pw", 4, 2, 5, 3), 32);
    run_case("noc3", pointwise_view("enc.4.pw", 2, 3, 2, 2), 24);
}

#[test]
fn emitted_kernel_matches_the_reference_for_the_minimal_shape() {
    if verilator_version().is_none() {
        eprintln!(
            "SKIP emitted_kernel_matches_the_reference_for_the_minimal_shape: verilator unavailable"
        );
        return;
    }
    run_case("minimal", pointwise_view("tiny", 1, 1, 1, 1), 8);
}
