//! Textual, tool-free tests for the SystemVerilog RTL backend.
//!
//! These exercise the public backend boundary only: they build an RTL bundle
//! from a plan source, check determinism and provenance, and check that every
//! unsupported construct is rejected with an actionable error. Simulation with
//! an external tool lives in `rtl_verilator.rs`.

use std::fs;

use tessera::rtl::{
    ChannelRequant, ElementType, KernelPlanView, PlacementHint, PlanContext, PlannedOp, Provenance,
    QuantPlan, RoundingMode, RtlBundle, RtlConfig, RtlFileKind, RtlMetadata, RtlPlanSource,
    SystemVerilogBackend, TileGeometry, WeightTensor,
};

/// Minimal plan source: one kernel view, no planner artifact.
struct SingleKernelPlan {
    view: KernelPlanView,
    provenance: Provenance,
}

impl SingleKernelPlan {
    fn new(view: KernelPlanView) -> Self {
        SingleKernelPlan {
            view,
            provenance: Provenance {
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
            },
        }
    }
}

impl RtlPlanSource for SingleKernelPlan {
    fn artifact_format(&self) -> &str {
        "tessera.execution_plan"
    }

    fn schema_version(&self) -> u32 {
        1
    }

    fn provenance(&self) -> Provenance {
        self.provenance.clone()
    }

    fn kernels(&self) -> Vec<KernelPlanView> {
        vec![self.view.clone()]
    }
}

/// A supported kernel view: `enc.0.pw`, 4 in channels x 4 out channels folded
/// as simd=2/noc=2 and k=2/fold=2.
fn supported_view() -> KernelPlanView {
    let tile = TileGeometry {
        simd: 2,
        noc: 2,
        k: 2,
        fold: 2,
    };
    let entries = (tile.fold * tile.k * tile.noc * tile.simd) as usize;
    let weights: Vec<i8> = (0..entries)
        .map(|index| ((index as i16 % 11) - 5) as i8)
        .collect();
    KernelPlanView {
        layer: "enc.0.pw".to_string(),
        op: PlannedOp::PointwiseDot,
        activation_element: ElementType::I8,
        weight_element: ElementType::I8,
        output_element: ElementType::I8,
        tile,
        quant: QuantPlan {
            activation_zero_point: 0,
            weight_zero_point: 0,
            rounding: RoundingMode::RoundHalfUpMagnitude,
            channels: (0..(tile.fold * tile.noc))
                .map(|channel| ChannelRequant {
                    multiplier: (1 << 16) + channel * 997,
                    shift: 18 + (channel % 3) as u8,
                    lo: -128 + channel as i16,
                    hi: 127 - channel as i16,
                })
                .collect(),
            multiplier_error_ppm: None,
        },
        weights: WeightTensor { data: weights },
        weight_placement: PlacementHint::WeightRom,
        activation_placement: PlacementHint::StreamedActivation,
        context: PlanContext {
            plan_format: "tessera.execution_plan".to_string(),
            plan_schema_version: 1,
            execution_mode: "streaming".to_string(),
            granularity: "frame".to_string(),
            engine: "stream.enc.0.pw".to_string(),
            step_index: 0,
            step_count: 1,
            estimated_cycles: 48_000,
            spatial_in: [64, 48],
            spatial_out: [64, 48],
        },
    }
}

fn lower(view: KernelPlanView) -> RtlBundle {
    SystemVerilogBackend
        .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
        .expect("supported kernel lowers")
}

#[test]
fn bundle_is_deterministic_and_carries_provenance() {
    let first = lower(supported_view());
    let second = lower(supported_view());

    assert_eq!(
        first.files(),
        second.files(),
        "emission must be reproducible"
    );
    assert_eq!(first.metadata.bundle_digest, second.metadata.bundle_digest);

    let metadata = &first.metadata;
    assert_eq!(metadata.format, "tessera.rtl_bundle");
    assert_eq!(metadata.schema_version, 1);
    assert_eq!(metadata.kernel_id, "tessera.pw_dot.i8i8.v1");
    assert_eq!(metadata.provenance.target, "ti60f225");
    assert_eq!(metadata.provenance.git_revision, "cafebabe");
    assert_eq!(metadata.provenance.input_manifest_sha256, "ab".repeat(32));
    assert_eq!(metadata.plan.engine, "stream.enc.0.pw");
    assert_eq!(metadata.plan.estimated_cycles, 48_000);
    assert_eq!(metadata.tile, supported_view().tile);
    assert_eq!(metadata.in_channels, 4);
    assert_eq!(metadata.out_channels, 4);
    assert_eq!(metadata.kernel_cycles_per_window, 2 + 2 * (2 + 2));
    assert_eq!(metadata.max_abs_accumulator, 2 * 2 * 128 * 128);

    // Every generated file is attributable on its own.
    for file in first.files() {
        if matches!(
            file.kind,
            RtlFileKind::SystemVerilog | RtlFileKind::Testbench
        ) {
            assert!(
                file.contents.contains("compiler_version: 0.1.0"),
                "{} lacks provenance",
                file.name
            );
            assert!(
                file.contents.contains(&"ab".repeat(32)),
                "{} lacks the input manifest hash",
                file.name
            );
        }
    }

    let dut = first
        .file("tessera_enc_0_pw_pw_dot_i8i8_v1.sv")
        .expect("module name is stable");
    assert!(
        dut.contents
            .contains("module tessera_enc_0_pw_pw_dot_i8i8_v1 #(")
    );
    assert!(dut.contents.contains("layer: enc.0.pw"));
    assert!(dut.contents.contains("parameter int    NF    = 2"));
    assert!(dut.contents.contains("in_ready"));
    assert!(dut.contents.contains("out_ready"));
    assert!(
        !dut.contents.contains("TODO") && !dut.contents.contains("FIXME"),
        "generated RTL must not contain placeholders"
    );

    // Metadata file re-parses and agrees with the in-memory metadata.
    let json = first.file("rtl_bundle.json").expect("metadata file");
    let parsed: RtlMetadata = serde_json::from_str(&json.contents).expect("metadata parses");
    assert_eq!(parsed, first.metadata);
    assert_eq!(parsed.requantization.len(), 4);

    // Source ordering for a simulator: DUT before testbench.
    let sources = first.verilog_sources();
    assert_eq!(
        sources,
        vec![
            "tessera_enc_0_pw_pw_dot_i8i8_v1.sv",
            "tessera_enc_0_pw_pw_dot_i8i8_v1_tb.sv"
        ]
    );
}

#[test]
fn bundle_materializes_to_disk_byte_for_byte() {
    let bundle = lower(supported_view());
    let dir = std::env::temp_dir().join(format!("tessera_rtl_artifacts_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);

    let written = bundle.write_to(&dir).expect("bundle writes");
    assert_eq!(written.len(), bundle.files().len());
    for file in bundle.files() {
        let path = dir.join(&file.name);
        assert_eq!(
            fs::read_to_string(&path).expect("generated file is readable"),
            file.contents
        );
    }
    // The memory images describe exactly the embedded ROM contents.
    let weights = bundle
        .file("tessera_enc_0_pw_pw_dot_i8i8_v1.weights.mem")
        .expect("weight image");
    assert_eq!(
        weights
            .contents
            .lines()
            .filter(|l| !l.starts_with("//"))
            .count(),
        16
    );
    let requant = bundle
        .file("tessera_enc_0_pw_pw_dot_i8i8_v1.requant.mem")
        .expect("requant image");
    assert_eq!(
        requant
            .contents
            .lines()
            .filter(|l| !l.starts_with("//"))
            .count(),
        4
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn memory_images_can_be_switched_off() {
    let config = RtlConfig {
        emit_memory_images: false,
        emit_testbench: false,
        ..RtlConfig::default()
    };
    let bundle = SystemVerilogBackend
        .lower(&SingleKernelPlan::new(supported_view()), &config)
        .expect("supported kernel lowers");
    assert_eq!(bundle.files().len(), 2, "one source plus metadata");
    assert_eq!(bundle.metadata.top_module, bundle.metadata.kernel_module);
    assert!(bundle.verilog_sources().len() == 1);
}

#[test]
fn unsupported_operators_are_rejected_with_the_step_name() {
    for (op, name) in [
        (PlannedOp::Depthwise, "depthwise"),
        (PlannedOp::Conv2d, "conv2d"),
        (PlannedOp::Add, "add"),
        (PlannedOp::Requant, "requant"),
        (PlannedOp::Activation, "activation"),
        (PlannedOp::UpsampleNearest, "upsample_nearest"),
        (PlannedOp::QuantizeOutput, "quantize_output"),
    ] {
        let mut view = supported_view();
        view.op = op;
        let err = SystemVerilogBackend
            .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
            .expect_err("operator must be rejected");
        let text = err.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("enc.0.pw"), "{text}");
        assert!(text.contains("pointwise_dot"), "{text}");
    }
}

#[test]
fn unsupported_element_types_and_placement_are_rejected() {
    let mut view = supported_view();
    view.weight_element = ElementType::I4;
    assert!(
        SystemVerilogBackend
            .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
            .is_err()
    );

    let mut view = supported_view();
    view.weight_placement = PlacementHint::Bram;
    let err = SystemVerilogBackend
        .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
        .expect_err("bram-resident weights are not emitted");
    assert!(err.to_string().contains("bram"), "{err}");
    assert!(err.to_string().contains("weights in ROM"), "{err}");

    let mut view = supported_view();
    view.activation_placement = PlacementHint::Ddr;
    let err = SystemVerilogBackend
        .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
        .expect_err("ddr-resident activations are not emitted");
    assert!(err.to_string().contains("ddr"), "{err}");
    assert!(err.to_string().contains("activations streamed"), "{err}");
}

#[test]
fn unsupported_quantization_is_rejected() {
    let mut view = supported_view();
    view.quant.activation_zero_point = 5;
    assert!(
        SystemVerilogBackend
            .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
            .is_err()
    );

    let mut view = supported_view();
    view.quant.rounding = RoundingMode::RoundHalfEven;
    assert!(
        SystemVerilogBackend
            .lower(&SingleKernelPlan::new(view), &RtlConfig::default())
            .is_err()
    );
}
