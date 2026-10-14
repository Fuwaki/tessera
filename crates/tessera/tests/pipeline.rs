//! End-to-end integration tests for the persisted pipeline.
//!
//! These exercise the real modules against the checked-in examples under
//! `fixtures/`: manifest import, Graph IR lower to the planner's primitive
//! graph, layout search, execution-plan projection, weight resolution and RTL
//! emission. Nothing here is mocked, and no path is hard-coded to a particular
//! machine: every fixture is resolved from `CARGO_MANIFEST_DIR`.
//!
//! Verilator is optional; the simulation assertion skips cleanly when the tool
//! is absent instead of claiming a verified result.

use std::path::{Path, PathBuf};

use tessera::frontend;
use tessera::graph::GraphArtifact;
use tessera::layout::{ExecutionMode, ExecutionPlan, Granularity, LayoutArtifact, PlanStatus};
use tessera::lower::{InputShape, lower_graph};
use tessera::mlir::{self, MlirOptions, SourceGraph};
use tessera::planner::{self, Policy, ProvenanceInput};
use tessera::rtl::{
    RtlBundle, RtlConfig, SimulationOutcome, SystemVerilogBackend, WeightResolver,
    simulate_with_verilator,
};
use tessera::target::Ti60Target;
use tessera::weights::{GraphWeightResolver, NamedBlobResolver};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// Scratch directory for one test, under `TESSERA_TEST_TMPDIR` (default
/// `/var/tmp`): a full-graph bundle plus a Verilator build is hundreds of
/// megabytes, and `/tmp` is often a small tmpfs shared with other work.
fn temp_dir(tag: &str) -> PathBuf {
    let base = std::env::var_os("TESSERA_TEST_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/tmp"));
    let dir = base.join(format!("tessera-pipeline-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn w8_artifact() -> GraphArtifact {
    frontend::import_manifest(&fixtures().join("w8")).expect("import the W8 fixture")
}

fn w8_shape() -> InputShape {
    InputShape::new(3, 6, 6).expect("6x6x3 input")
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("load target")
}

fn policy(name: &str) -> Policy {
    Policy::from_toml_file(fixtures().join("policies").join(name)).expect("load policy")
}

fn provenance(graph_sha256: &str) -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: graph_sha256.to_string(),
        target_sha256: "target-sha".to_string(),
        policy_sha256: "policy-sha".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

// ---------------------------------------------------------------------------
// Graph artifact and MLIR stage
// ---------------------------------------------------------------------------

#[test]
fn graph_artifact_round_trips_byte_for_byte() {
    let artifact = w8_artifact();
    let dir = temp_dir("graph-round-trip");
    let first = dir.join("graph.tgraph");
    let second = dir.join("graph.second.tgraph");

    frontend::save_artifact(&artifact, &first).expect("save");
    let loaded = frontend::load_artifact(&first).expect("load");
    assert_eq!(loaded, artifact, "the artifact must survive a round trip");
    frontend::save_artifact(&loaded, &second).expect("save again");

    // Deterministic serialization: the same content produces the same bytes.
    assert_eq!(
        std::fs::read(&first).expect("read first"),
        std::fs::read(&second).expect("read second"),
        "artifact serialization must be reproducible"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mlir_emission_is_deterministic_and_linked_to_the_graph_artifact() {
    let artifact = w8_artifact();
    let options = MlirOptions::new(w8_shape().to_nhwc());
    let source = SourceGraph::from(&artifact);

    let first = mlir::lower(artifact.graph(), source, &options).expect("emit");
    let second = mlir::lower(artifact.graph(), source, &options).expect("emit again");
    assert_eq!(first.module, second.module, "emission must be reproducible");
    assert_eq!(first.provenance.module_hash, second.provenance.module_hash);

    // Provenance is data: the MLIR artifact names the graph artifact it descends
    // from, and that hash is the manifest hash the frontend recorded.
    assert_eq!(first.provenance.source_format, tessera::graph::GRAPH_FORMAT);
    assert_eq!(
        first.provenance.source_schema_version,
        tessera::graph::GRAPH_SCHEMA_VERSION
    );
    assert_eq!(artifact.provenance().source_format, "fpga_st_train.int8.v1");
    assert_eq!(
        first.provenance.source_content_hash,
        artifact.provenance().manifest_sha256
    );
    assert!(
        first.provenance.melior.is_none(),
        "the tensor stage is pure Rust"
    );
    assert_eq!(first.operators.len(), artifact.graph().ops().len());
}

#[test]
fn optimize_reports_that_melior_is_unavailable_without_the_feature() {
    let artifact = w8_artifact();
    let emitted = mlir::lower(
        artifact.graph(),
        SourceGraph::from(&artifact),
        &MlirOptions::new(w8_shape().to_nhwc()),
    )
    .expect("emit");

    let result = mlir::optimize(&emitted);
    if mlir::melior_available() {
        let optimized = result.expect("a Melior build optimizes the artifact");
        assert_eq!(optimized.stage.as_str(), "planning");
        assert!(optimized.provenance.melior.is_some());
    } else {
        let error = result.expect_err("a build without Melior refuses to claim optimization");
        assert!(
            error.to_string().contains("melior"),
            "the error must name the missing feature: {error}"
        );
    }
}

// ---------------------------------------------------------------------------
// Graph IR -> planner primitive graph
// ---------------------------------------------------------------------------

#[test]
fn lowering_preserves_weights_bias_and_per_channel_requantization() {
    let artifact = w8_artifact();
    let (plan, report) = lower_graph(artifact.graph(), w8_shape()).expect("lower");
    plan.validate()
        .expect("the lowered graph is a valid planner input");

    // One weight tensor per layer, named by the shared convention.
    for layer in artifact.graph().layers() {
        let name = tessera::lower::weight_tensor_name(layer.name.as_str());
        assert_eq!(
            report
                .weight_tensors
                .get(layer.name.as_str())
                .map(String::as_str),
            Some(name.as_str())
        );
        let tensor = plan
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("weight tensor {name} is missing"));
        assert_eq!(tensor.shape.n, layer.out_channels.get());
        assert_eq!(tensor.shape.h, layer.kernel.h.get());
        assert_eq!(tensor.shape.w, layer.kernel.w.get());
    }

    // Two operators per convolution, in graph order, minus the relu markers.
    let names: Vec<&str> = plan.ops.iter().map(|op| op.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "stem",
            "dw",
            "mid",
            // Named by the point they write, with an occurrence suffix only when a
            // model writes a point twice.
            "add.3",
            "upsample_nearest2x.3",
            "head",
            "quantize_u8_output"
        ]
    );

    // The manifest's per-output-channel requantization survives lowering.
    let stem = plan.ops.iter().find(|op| op.name == "stem").expect("stem");
    let tessera::planner::OpDetail::Conv(detail) = &stem.detail else {
        panic!("stem must lower to a convolution");
    };
    let layer = artifact.graph().layer_named("stem").expect("stem layer");
    assert_eq!(
        detail.requant.per_channel_multipliers.len(),
        layer.out_channels.get() as usize
    );
    assert!(
        detail
            .requant
            .per_channel_multipliers
            .windows(2)
            .any(|w| w[0] != w[1]),
        "the fixture declares distinct per-channel factors"
    );
    assert!(detail.requant.multiplier_error_ppm.is_some());
    // The declared int32 bias is non-zero for `stem`, so it is preserved.
    assert!(detail.bias.is_some(), "a non-zero bias must not be dropped");

    // A zero bias is the identity and is dropped, with a recorded note.
    let mid = plan.ops.iter().find(|op| op.name == "mid").expect("mid");
    let tessera::planner::OpDetail::Conv(mid_detail) = &mid.detail else {
        panic!("mid must lower to a convolution");
    };
    assert!(mid_detail.bias.is_none());
    assert!(
        report
            .notes
            .iter()
            .any(|note| note.contains("mid") && note.contains("zero"))
    );
    // Depthwise weight tensors carry one input channel per group.
    let dw = plan.ops.iter().find(|op| op.name == "dw").expect("dw");
    let tessera::planner::OpDetail::Conv(dw_detail) = &dw.detail else {
        panic!("dw must lower to a convolution");
    };
    assert_eq!(dw_detail.groups, 4);
    let dw_weights = plan.tensor(dw_detail.weights).expect("dw weights");
    assert_eq!(dw_weights.shape.c, 1);
}

#[test]
fn lowering_propagates_the_input_resolution_through_every_operator() {
    let artifact = w8_artifact();
    let (plan, _) =
        lower_graph(artifact.graph(), InputShape::new(3, 8, 8).expect("shape")).expect("lower");

    // The fixture doubles the spatial extent once (nearest upsample) and keeps
    // the head at three channels, so the unsigned image is 16x16x3.
    let output = plan
        .tensor(plan.outputs[0])
        .expect("the lowered graph has an output tensor");
    assert_eq!(
        (output.shape.c, output.shape.h, output.shape.w),
        (3, 16, 16)
    );
    assert_eq!(output.elem, tessera::layout::ElemType::U8);

    // The head reads the upsampled point, not the 8x8 point it was written from.
    let head = plan.ops.iter().find(|op| op.name == "head").expect("head");
    let head_input = plan.tensor(head.inputs[0]).expect("head input");
    assert_eq!((head_input.shape.h, head_input.shape.w), (16, 16));
}

#[test]
fn an_unsigned_head_that_writes_a_real_point_still_produces_the_image() {
    // The graph IR allows the unsigned head to write a real point instead of the
    // image sentinel; both forms must lower, and both must feed the output
    // quantization with an unsigned tensor.
    let manifest = r#"{
      "format": "fpga_st_train.int8.v1",
      "arch": "unsigned-point-head",
      "act_qmax": 127.0,
      "output_qmax": 255.0,
      "weight_qmax": 127.0,
      "points": ["in", "out"],
      "point_scales": [0.01, 0.003921569],
      "weight_file": "weights.bin",
      "weight_packing": "i8-x1",
      "weight_bytes": 6,
      "graph": [
        { "op": "conv", "layer": "head", "point": 0, "scale": 0.01 },
        { "op": "quantize_u8_output", "scale": 0.003921569 }
      ],
      "layers": [
        {
          "name": "head", "op": "conv2d",
          "in_channels": 3, "out_channels": 2,
          "out_relu": false, "out_unsigned": true,
          "kernel": [1, 1], "stride": [1, 1], "reflect_pad": 0,
          "in_point": 0, "out_point": 1,
          "in_scale": 0.01, "out_scale": 0.003921569,
          "weight_scale": [0.5, 0.5],
          "bias": [0.0, 0.0],
          "bias_int32": [0, 0],
          "requant": [0.25, 0.25],
          "weight_offset": 0, "weight_count": 6
        }
      ]
    }"#;
    let artifact =
        frontend::import_manifest_str(manifest, &[1u8; 6], "unsigned-head.json").expect("import");
    let (plan, _) =
        lower_graph(artifact.graph(), InputShape::new(3, 4, 4).expect("shape")).expect("lower");

    let head = plan.ops.iter().find(|op| op.name == "head").expect("head");
    let tessera::planner::OpDetail::Conv(detail) = &head.detail else {
        panic!("head must lower to a convolution");
    };
    assert!(matches!(
        detail.activation.kind,
        tessera::layout::ActivationKind::UnsignedClamp
    ));
    let head_output = plan.tensor(head.outputs[0]).expect("head output");
    assert_eq!(head_output.elem, tessera::layout::ElemType::U8);
    // The output quantization reads that same unsigned tensor, so the lowered
    // graph is a two-step chain rather than a broken one.
    let quantize = plan
        .ops
        .iter()
        .find(|op| op.name == "quantize_u8_output")
        .expect("output quantization");
    assert_eq!(quantize.inputs, head.outputs);
}

#[test]
fn input_shape_parsing_rejects_anything_the_compiler_cannot_lower() {
    for bad in ["6,6,3", "2,6,6,3", "1,0,6,3", "1,6,6,-3", "1,6,6"] {
        assert!(
            InputShape::parse_nhwc(bad).is_err(),
            "{bad:?} must not parse into an input shape"
        );
    }
    let shape = InputShape::parse_nhwc("1,6,6,3").expect("a valid NHWC shape");
    assert_eq!(shape.to_nhwc(), [1, 6, 6, 3]);
    assert_eq!(
        InputShape::parse_nhwc("1, 6, 6, 3").expect("whitespace is tolerated"),
        shape
    );
}

// ---------------------------------------------------------------------------
// Layout search -> execution plan
// ---------------------------------------------------------------------------

#[test]
fn whole_network_plan_fits_the_emitted_envelope_and_the_single_kernel_backend_rejects_it() {
    let artifact = w8_artifact();
    let (plan, _) = lower_graph(artifact.graph(), w8_shape()).expect("lower");
    let layout = planner::solve(
        &plan,
        &target(),
        &policy("emit.toml"),
        &provenance("graph-sha"),
    )
    .expect("solve");
    assert_eq!(layout.status, PlanStatus::Feasible);
    layout.verify().expect("the layout re-verifies");

    let execution = layout.execution_plan().expect("projection");
    assert_eq!(
        execution.steps.len(),
        plan.ops.len(),
        "every lowered operator is scheduled"
    );
    // The generated engine is one shared array over an on-chip arena, so the
    // plan must be a shared-tile frame schedule with a single lane per step.
    assert_eq!(execution.execution_mode, ExecutionMode::SharedTile);
    assert_eq!(
        layout.selection.as_ref().unwrap().granularity,
        Granularity::Frame
    );
    for step in &execution.steps {
        assert_eq!(
            step.tap_parallel, 1,
            "step {} asks for tap parallelism the engine does not have",
            step.op
        );
        for name in [&step.input_buffer, &step.output_buffer] {
            assert!(
                execution.buffers.iter().any(|b| &b.name == name),
                "step {} buffer {name} is undeclared",
                step.op
            );
        }
    }

    // The single-kernel backend still refuses a multi-step plan: either because
    // the plan is not one pointwise step, or because its shared-tile engine is
    // not the interface that backend emits.
    let resolver = GraphWeightResolver::load(&artifact, fixtures().join("w8")).expect("resolver");
    let error = SystemVerilogBackend
        .lower_execution_plan(&execution, &resolver, &RtlConfig::default())
        .expect_err("a seven-step plan is outside the single-kernel envelope");
    let text = error.to_string();
    assert!(
        text.contains("single pointwise step") || text.contains("shared_tile"),
        "the rejection must name the plan shape: {text}"
    );
}

#[test]
fn emit_policy_keeps_every_convolution_step_within_the_emitted_envelope() {
    // The generated engine instantiates `simd * pe` real multipliers and iterates
    // taps sequentially, so an emittable plan needs tap_parallel == 1 and at most
    // 64 lanes per convolution step. This locks the fixture policy to that
    // envelope, so editing the policy cannot silently produce plans the backend
    // must reject.
    for (name, graph_shape) in [("w8", w8_shape())] {
        let artifact = w8_artifact();
        let (plan, _) = lower_graph(artifact.graph(), graph_shape).expect("lower");
        let layout = planner::solve(&plan, &target(), &policy("emit.toml"), &provenance("sha"))
            .unwrap_or_else(|error| panic!("{name}: solve: {error}"));
        layout
            .verify()
            .unwrap_or_else(|error| panic!("{name}: verify: {error}"));
        let execution = layout.execution_plan().expect("projection");
        for step in &execution.steps {
            assert_eq!(step.tap_parallel, 1, "{name}: step {}", step.op);
            if step.op_kind.is_convolution() {
                assert!(
                    u64::from(step.simd) * u64::from(step.pe) <= 64,
                    "{name}: step {} wants {} lanes",
                    step.op,
                    u64::from(step.simd) * u64::from(step.pe)
                );
            }
        }
    }
}

#[test]
fn per_channel_requantization_reaches_the_layout() {
    let artifact = w8_artifact();
    let (plan, _) = lower_graph(artifact.graph(), w8_shape()).expect("lower");
    let layout: LayoutArtifact = planner::solve(
        &plan,
        &target(),
        &policy("explore.toml"),
        &provenance("graph-sha"),
    )
    .expect("solve");
    let stem = layout
        .schedule
        .iter()
        .find(|step| step.op == "stem")
        .expect("stem is scheduled");
    let requant = stem.requant.as_ref().expect("stem requantizes");
    assert_eq!(requant.per_channel_multipliers.len(), 4);
    assert!(requant.per_channel_multipliers.iter().all(|m| *m >= 0));
    assert!(requant.multiplier_error_ppm.is_some());
}

// ---------------------------------------------------------------------------
// Weight resolution
// ---------------------------------------------------------------------------

#[test]
fn graph_weight_resolver_slices_the_manifest_blob_per_layer() {
    let artifact = w8_artifact();
    let blob = std::fs::read(fixtures().join("w8/weights.bin")).expect("read blob");
    let resolver = GraphWeightResolver::from_bytes(&artifact, blob.clone()).expect("resolver");

    for layer in artifact.graph().layers() {
        let channels_per_group = if layer.kind == tessera::graph::LayerKind::DepthwiseConv2d {
            1
        } else {
            layer.in_channels.get() / layer.groups.get().max(1)
        };
        let tensor = tessera::layout::TensorRef {
            id: tessera::layout::TensorId(layer.id.get()),
            name: tessera::lower::weight_tensor_name(layer.name.as_str()),
            shape: tessera::layout::TensorShape::new(
                layer.out_channels.get(),
                channels_per_group,
                layer.kernel.h.get(),
                layer.kernel.w.get(),
            ),
            elem: tessera::layout::ElemType::I8,
        };
        let bytes = resolver.tensor_bytes(&tensor).expect("resolve");
        assert_eq!(
            bytes,
            blob[layer.weight_byte_offset..layer.weight_byte_offset + layer.weight_bytes].to_vec(),
            "layer {} bytes must come from its declared range",
            layer.name
        );
    }
}

#[test]
fn graph_weight_resolver_checks_the_blob_digest() {
    let artifact = w8_artifact();
    let mut blob = std::fs::read(fixtures().join("w8/weights.bin")).expect("read blob");
    blob[0] ^= 0xff;
    let error = GraphWeightResolver::from_bytes(&artifact, blob)
        .err()
        .expect("a tampered blob must be rejected");
    assert!(
        error.to_string().contains("weights_sha256"),
        "the error must name the broken provenance field: {error}"
    );
}

// ---------------------------------------------------------------------------
// RTL emission and simulation
// ---------------------------------------------------------------------------

fn pointwise_plan() -> ExecutionPlan {
    let dir = fixtures().join("pointwise");
    let text = std::fs::read_to_string(dir.join("plan_input.json")).expect("read fixture");
    let input = tessera::stages::parse_primitive(&text).expect("parse fixture");
    let layout = planner::solve(
        &input,
        &target(),
        &policy("streaming.toml"),
        &provenance("primitive-sha"),
    )
    .expect("solve");
    assert_eq!(layout.status, PlanStatus::Feasible);
    layout.execution_plan().expect("projection")
}

fn pointwise_fixture_resolver() -> NamedBlobResolver {
    NamedBlobResolver::from_dir(fixtures().join("pointwise/weights")).expect("weights")
}

fn lower_pointwise() -> RtlBundle {
    let plan = pointwise_plan();
    SystemVerilogBackend
        .lower_execution_plan(&plan, &pointwise_fixture_resolver(), &RtlConfig::default())
        .expect("the narrow envelope is emittable")
}

#[test]
fn manifest_weight_bytes_reach_the_emitted_rom() {
    // A one-layer manifest whose head is a 4x4 1x1 convolution, and a
    // single-step primitive plan whose weight tensor is named after that layer:
    // the resolver must fetch the bytes from the manifest blob the frontend
    // hashed, not from anywhere else.
    const MANIFEST: &str = r#"{
      "format": "fpga_st_train.int8.v1",
      "arch": "graph-weight-fixture",
      "act_qmax": 127.0,
      "output_qmax": 255.0,
      "weight_qmax": 127.0,
      "points": ["in", "out"],
      "point_scales": [0.01, 0.003921569],
      "weight_file": "weights.bin",
      "weight_packing": "i8-x1",
      "weight_bytes": 16,
      "graph": [
        { "op": "conv", "layer": "head", "point": 0, "scale": 0.01 },
        { "op": "quantize_u8_output", "scale": 0.003921569 }
      ],
      "layers": [
        {
          "name": "head", "op": "conv2d",
          "in_channels": 4, "out_channels": 4,
          "out_relu": false, "out_unsigned": true,
          "kernel": [1, 1], "stride": [1, 1], "reflect_pad": 0,
          "in_point": 0, "out_point": 1,
          "in_scale": 0.01, "out_scale": 0.003921569,
          "weight_scale": [0.5, 0.5, 0.5, 0.5],
          "bias": [0.0, 0.0, 0.0, 0.0],
          "bias_int32": [0, 0, 0, 0],
          "requant": [0.25, 0.25, 0.25, 0.25],
          "weight_offset": 0, "weight_count": 16
        }
      ]
    }"#;
    const PLAN: &str = r#"{
      "name": "graph-weight-fixture",
      "tensors": [
        { "id": 0, "name": "x", "shape": { "n": 1, "c": 4, "h": 8, "w": 8 }, "elem": "i8" },
        { "id": 1, "name": "weights:head", "shape": { "n": 4, "c": 4, "h": 1, "w": 1 }, "elem": "i8" },
        { "id": 2, "name": "y", "shape": { "n": 1, "c": 4, "h": 8, "w": 8 }, "elem": "i8" }
      ],
      "ops": [
        {
          "name": "head", "kind": "pointwise_conv2d",
          "inputs": [0], "outputs": [2],
          "detail": { "conv": {
            "weights": 1, "bias": null, "groups": 1, "stride": [1, 1],
            "padding": { "zero": { "top": 0, "bottom": 0, "left": 0, "right": 0 } },
            "requant": {
              "m0": 65536, "shift": 18, "rounding": "half_away_from_zero",
              "saturate_lo": -128, "saturate_hi": 127
            },
            "activation": { "kind": "relu", "clamp_lo": 0, "clamp_hi": 127 }
          } }
        }
      ],
      "inputs": [0],
      "outputs": [2]
    }"#;

    // Distinct, non-zero bytes: a default or zero-filled ROM would not match.
    let blob: Vec<u8> = (1..=16u8).collect();
    let artifact = frontend::import_manifest_str(MANIFEST, &blob, "graph-weight-fixture.json")
        .expect("import");
    let input = tessera::stages::parse_primitive(PLAN).expect("parse the primitive plan");
    let layout = planner::solve(
        &input,
        &target(),
        &policy("streaming.toml"),
        &provenance(&artifact.provenance().manifest_sha256),
    )
    .expect("solve");
    let execution = layout.execution_plan().expect("projection");
    assert_eq!(execution.steps.len(), 1);

    let resolver = GraphWeightResolver::from_bytes(&artifact, blob.clone()).expect("resolver");
    let bundle = SystemVerilogBackend
        .lower_execution_plan(&execution, &resolver, &RtlConfig::default())
        .expect("the manifest weights satisfy the narrow envelope");

    let image = bundle
        .files()
        .iter()
        .find(|f| f.name.ends_with(".weights.mem"))
        .expect("the weight ROM image is emitted");
    let values: Vec<u8> = image
        .contents
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(|line| u8::from_str_radix(line.trim(), 16).expect("hex byte"))
        .collect();
    // For this tile geometry (simd=4, k=1, noc=4, fold=1) the ROM order is the
    // manifest's (out, in) order, so the bytes must appear verbatim.
    assert_eq!(values, blob, "the ROM must carry the manifest weight bytes");
}

#[test]
fn pointwise_fixture_emits_deterministic_systemverilog() {
    let first = lower_pointwise();
    let second = lower_pointwise();
    assert_eq!(
        first.files(),
        second.files(),
        "emission must be reproducible"
    );
    assert_eq!(first.metadata.bundle_digest, second.metadata.bundle_digest);

    let source = first
        .files()
        .iter()
        .find(|f| f.name.ends_with("pw_dot_i8i8_v1.sv"))
        .expect("the DUT is emitted");
    assert!(source.contents.contains("module tessera_pw_pw_dot_i8i8_v1"));
    assert!(source.contents.contains("layer: pw"));
    assert!(source.contents.contains("in_ready") && source.contents.contains("out_ready"));
    assert!(
        !source.contents.contains("TODO") && !source.contents.contains("FIXME"),
        "generated RTL must not contain placeholders"
    );
    // The weight ROM carries the fixture's 4x4 matrix, packed in (fold,k,noc,simd)
    // order: the first entry is the weight the fixture's weights.bin declares.
    let image = first
        .file("tessera_pw_pw_dot_i8i8_v1.weights.mem")
        .expect("weight image");
    assert!(
        image.contents.contains("//") && image.contents.lines().count() > 1,
        "the weight image documents itself"
    );
}

#[test]
fn pointwise_fixture_matches_the_reference_model_under_verilator() {
    let bundle = lower_pointwise();
    let dir = temp_dir("verilator");
    match simulate_with_verilator(&bundle, &dir).expect("simulator invocation") {
        SimulationOutcome::Passed { log } => {
            assert!(log.contains("TESSA_RTL_PASS"), "{log}");
            assert!(log.contains("mismatches=0"), "{log}");
        }
        SimulationOutcome::ToolUnavailable { tool, reason } => {
            eprintln!("SKIP: {tool} unavailable ({reason}); RTL was not simulated");
        }
        SimulationOutcome::Failed { detail, log } => {
            panic!("generated RTL failed simulation: {detail}\n{log}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Serves a graph's weight bytes with one tensor's contents replaced, so a
/// fixture property can be tested by changing the model rather than by reasoning
/// about it.
struct HeadPatchedResolver<'a> {
    inner: GraphWeightResolver<'a>,
    tensor: String,
    head_bytes: Vec<u8>,
}

impl WeightResolver for HeadPatchedResolver<'_> {
    fn tensor_bytes(
        &self,
        tensor: &tessera::layout::TensorRef,
    ) -> Result<Vec<u8>, tessera::rtl::RtlError> {
        let mut bytes = self.inner.tensor_bytes(tensor)?;
        if tensor.name == self.tensor {
            assert_eq!(bytes.len(), self.head_bytes.len(), "patched tensor length");
            bytes.copy_from_slice(&self.head_bytes);
        }
        Ok(bytes)
    }
}

#[test]
fn w8_image_evidence_is_sized_to_the_fixture_it_comes_from() {
    // The head reads the upsampled tensor, which is never negative, so a head row
    // with no positive coefficient is identically zero for *every* stimulus and
    // its output channel is structurally dead. This test states the consequence
    // of the fixture as it stands and switches to the stronger requirement once
    // the rows are fixed, so an image-level comparison can never be read as
    // evidence it is not -- and fixing the rows cannot be mistaken for a
    // semantic change in the compiler.
    let artifact = w8_artifact();
    let head = artifact.graph().layer_named("head").expect("head layer");
    assert_eq!(
        (head.out_channels.get(), head.in_channels.get()),
        (3, 4),
        "the fixture's head geometry is part of the assertion"
    );
    let blob = std::fs::read(fixtures().join("w8/weights.bin")).expect("weight blob");
    let codes: Vec<i8> = blob[head.weight_byte_offset..head.weight_byte_offset + head.weight_bytes]
        .iter()
        .map(|byte| *byte as i8)
        .collect();
    assert_eq!(codes.len(), 12, "the head is a 3x4 pointwise matrix");
    let dead_rows = codes
        .chunks(4)
        .filter(|row| !row.iter().any(|code| *code > 0))
        .count();

    // The image the reference produces for a deterministic stimulus set.
    let (plan, _) = lower_graph(artifact.graph(), w8_shape()).expect("lower");
    let layout =
        planner::solve(&plan, &target(), &policy("emit.toml"), &provenance("sha")).expect("solve");
    let execution = layout.execution_plan().expect("projection");
    let resolver = GraphWeightResolver::load(&artifact, fixtures().join("w8")).expect("resolver");
    let model =
        tessera::rtl::full::FullGraphModel::resolve(artifact.graph(), &execution, &resolver)
            .expect("full-graph model");
    let elements = plan
        .tensor(plan.inputs[0])
        .expect("input tensor")
        .shape
        .elements() as usize;
    let mut seen = std::collections::BTreeSet::new();
    let mut stimuli = Vec::new();
    for seed in [1u64, 7, 0x5eed] {
        let mut state = seed | 1;
        stimuli.push(
            (0..elements)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    ((state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as i64 % 255) - 127
                })
                .collect::<Vec<i64>>(),
        );
    }
    stimuli.push(vec![127i64; elements]);
    for frame in stimuli {
        let run =
            tessera::rtl::reference::run_integer(&model, &[frame]).expect("reference evaluation");
        let image = run.frames[0]
            .steps
            .last()
            .expect("the image step is scheduled");
        assert_eq!(image.name, "quantize_u8_output");
        seen.extend(image.codes.iter().copied());
    }

    if dead_rows > 0 {
        // A row with no positive coefficient contributes nothing: the head reads
        // a non-negative tensor, so its accumulator can only be <= 0 and the
        // clamp maps it to 0. Prove that dynamically by zeroing those rows and
        // re-running the reference: the image must be *identical*, which is what
        // makes an image-level comparison on this fixture weaker evidence than
        // the step traces (fixtures/acceptance/graph carries the discriminating
        // coverage).
        let mut head_bytes = codes.iter().map(|code| *code as u8).collect::<Vec<u8>>();
        for (index, row) in codes.chunks(4).enumerate() {
            if !row.iter().any(|code| *code > 0) {
                for offset in 0..4 {
                    head_bytes[index * 4 + offset] = 0;
                }
            }
        }
        let inner = GraphWeightResolver::load(&artifact, fixtures().join("w8")).expect("resolver");
        let patched = HeadPatchedResolver {
            inner,
            tensor: tessera::lower::weight_tensor_name("head"),
            head_bytes,
        };
        let patched_model =
            tessera::rtl::full::FullGraphModel::resolve(artifact.graph(), &execution, &patched)
                .expect("full-graph model with the dead rows zeroed");
        let mut patched_seen = std::collections::BTreeSet::new();
        let mut stimuli = Vec::new();
        let mut state = 0x5eedu64 | 1;
        stimuli.push(
            (0..elements)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    ((state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 33) as i64 % 255) - 127
                })
                .collect::<Vec<i64>>(),
        );
        stimuli.push(vec![127i64; elements]);
        for frame in stimuli {
            let run = tessera::rtl::reference::run_integer(&patched_model, &[frame])
                .expect("reference evaluation");
            patched_seen.extend(
                run.frames[0]
                    .steps
                    .last()
                    .expect("the image step is scheduled")
                    .codes
                    .iter()
                    .copied(),
            );
        }
        assert_eq!(
            seen, patched_seen,
            "a head row without a positive coefficient must not move the image at all"
        );
        eprintln!(
            "fixtures/w8: {dead_rows} of 3 head rows have no positive coefficient, so that many \
             output channels are identically zero and image codes are limited to {seen:?}; \
             zeroing those rows leaves the image unchanged. Discriminating image evidence comes \
             from fixtures/acceptance/graph."
        );
        return;
    }

    // Once every head row has both signs, the image must actually vary and clamp
    // at both ends; anything less means the fixture regressed.
    assert!(
        seen.len() >= 4,
        "the W8 image must vary over the stimulus set, found {seen:?}"
    );
    assert!(
        seen.contains(&0) && seen.contains(&255),
        "the unsigned clamp must actually clamp at both ends, found {seen:?}"
    );
    assert!(
        seen.iter().any(|code| *code > 0 && *code < 255),
        "the image must contain an interior value, found {seen:?}"
    );
}

/// The datapath rounds **half away from zero**, which is not the same rule as
/// "half up" (toward positive infinity): the two disagree exactly at negative
/// ties. This test states both rules and asserts that the emitted requantization
/// matches away-from-zero while a toward-+infinity rule would differ, so the
/// contract's name is backed by behaviour rather than by a comment.
#[test]
fn negative_ties_round_away_from_zero_not_toward_positive_infinity() {
    use tessera::rtl::{ChannelRequant, reference_requant};

    // `m0 = 2^16`, `shift = 17`: the product of one accumulator step is exactly
    // one half in the scaled domain, so every assertion below is a tie or a
    // quarter, never a rounding accident.
    let channel = ChannelRequant {
        multiplier: 1 << 16,
        shift: 17,
        lo: -128,
        hi: 127,
    };

    // The rule the datapath implements, spelled out: add `2^(shift-1)` to a
    // non-negative product and `2^(shift-1) - 1` to a negative one, then
    // arithmetic-shift right.
    let away_from_zero = |acc: i64| -> i64 {
        let product = acc * i64::from(channel.multiplier);
        let offset = if product >= 0 { 1 << 16 } else { (1 << 16) - 1 };
        (product + offset) >> channel.shift
    };
    // The misleading rule: "half up" as in toward positive infinity.
    let toward_positive_infinity =
        |acc: i64| -> i64 { (acc * i64::from(channel.multiplier) + (1 << 16)) >> channel.shift };

    assert_eq!(reference_requant(1, &channel), 1);
    // -1 * 2^16 = -0.5 exactly: away from zero is -1, toward +inf is 0.
    assert_eq!(reference_requant(-1, &channel), -1);
    // -3 * 2^16 = -1.5 exactly: away from zero is -2, toward +inf is -1.
    assert_eq!(reference_requant(-3, &channel), -2);

    for acc in [-3i64, -1, 1, 3] {
        assert_eq!(
            reference_requant(acc, &channel),
            away_from_zero(acc),
            "acc {acc}: the datapath must be half-away-from-zero"
        );
    }
    for acc in [-3i64, -1] {
        assert_ne!(
            reference_requant(acc, &channel),
            toward_positive_infinity(acc),
            "acc {acc}: the two rules must be distinguishable, or the contract's name is \
             untested"
        );
    }
}
