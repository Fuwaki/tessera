//! Acceptance pipeline: the fixture graph through import, lowering and planning.
//!
//! The point of this file is that every number it compares against comes from
//! `manifest.json` plus `weights.bin` through the independent reference in
//! `acceptance_support`, never from the artifact under test. If lowering and the
//! reference disagree about `M0`, the shift, the clamp, the per-channel
//! multipliers, the tensor shapes, the padding rule or the residual's code
//! width, this file fails and names the operator.

mod acceptance_support;

use acceptance_support::{
    Fixture, Manifest, ManifestLayer, ResidualCarry, fixture_dir, fixtures, plane_from_nhwc,
    plane_to_nhwc, regenerate_or_compare, run_fixture, run_frame, run_frame_with,
    serialize_requant, stage_dir,
};
use tessera::frontend;
use tessera::layout::{
    ActivationKind, CodeRange, ElemType, ExecutionPlan, PlanStatus, Rounding, TensorRef,
};
use tessera::lower::{InputShape, lower_graph};
use tessera::planner::{self, OpDetail, Padding, Policy, ProvenanceInput};
use tessera::rtl::WeightResolver;
use tessera::stages::{self, StageDir};
use tessera::target::Ti60Target;
use tessera::weights::GraphWeightResolver;

/// The acceptance graph's input resolution.
const GRAPH_SHAPE: (u32, u32, u32) = (3, 5, 5);

/// The reference Ti60 target the acceptance graph is solved against.
fn target_path() -> std::path::PathBuf {
    acceptance_support::fixtures().join("targets/ti60f225.toml")
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(target_path()).expect("load the reference target")
}

fn policy(name: &str) -> Policy {
    Policy::from_toml_file(fixture_dir("").join(name)).expect("load the acceptance policy")
}

fn provenance(graph_sha256: &str) -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: graph_sha256.to_string(),
        target_sha256: "acceptance-target".to_string(),
        policy_sha256: "acceptance-policy".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// The requantization contract the reference derives for one manifest layer:
/// `(M0 per channel, shift, (clamp lo, clamp hi))`, with the clamp taken from the
/// manifest's own quantization limits.
fn derived_requant(
    manifest: &acceptance_support::Manifest,
    layer: &ManifestLayer,
) -> (Vec<i64>, u32, (i64, i64)) {
    let (m0, shift) = serialize_requant(&layer.requant);
    let act_qmax = manifest.act_qmax as i64;
    let output_qmax = manifest.output_qmax as i64;
    let clamp = if layer.out_unsigned {
        (0, output_qmax)
    } else if layer.out_relu {
        (0, act_qmax)
    } else {
        (-act_qmax, act_qmax)
    };
    (m0, shift, clamp)
}

#[test]
fn acceptance_graph_lowers_with_independently_derived_quantization() {
    let fixture = Fixture::load("graph");
    let artifact = frontend::import_manifest(&fixture.dir).expect("import the acceptance manifest");
    let (plan, report) = lower_graph(
        artifact.graph(),
        InputShape::new(GRAPH_SHAPE.0, GRAPH_SHAPE.1, GRAPH_SHAPE.2)
            .expect("acceptance input shape"),
    )
    .expect("lower the acceptance graph");
    plan.validate()
        .expect("the lowered plan is a valid planner input");

    // The operator sequence, matched by prefix so an index-based naming scheme
    // for the elementwise operators is not pinned.
    let names: Vec<&str> = plan.ops.iter().map(|op| op.name.as_str()).collect();
    assert_eq!(
        names.len(),
        7,
        "one primitive operator per graph operator: {names:?}"
    );
    assert_eq!(names[0], "stem");
    assert_eq!(names[1], "dw");
    assert_eq!(names[2], "mid");
    assert!(
        names[3].starts_with("add."),
        "the residual lowers to {names:?}"
    );
    assert!(
        names[4].starts_with("upsample_nearest2x."),
        "the upsample lowers to {names:?}"
    );
    assert_eq!(names[5], "head");
    assert_eq!(names[6], "quantize_u8_output");

    for layer_name in ["stem", "dw", "mid", "head"] {
        let layer = fixture.manifest.layer(layer_name);
        let op = plan
            .ops
            .iter()
            .find(|op| op.name == layer_name)
            .unwrap_or_else(|| panic!("{layer_name} is not in the lowered plan"));
        let OpDetail::Conv(detail) = &op.detail else {
            panic!("{layer_name} must lower to a convolution");
        };

        // Per-channel requantization, derived here from the manifest floats.
        let (m0, shift, (lo, hi)) = derived_requant(&fixture.manifest, layer);
        let requant = &detail.requant;
        assert_eq!(requant.shift, shift as i32, "{layer_name}: shift");
        assert_eq!(
            requant.m0,
            *m0.iter().max().expect("non-empty multipliers"),
            "{layer_name}: lowering carries the largest channel multiplier as m0"
        );
        assert_eq!(
            requant.per_channel_multipliers, m0,
            "{layer_name}: per-channel multipliers must be the derived fixed-point vector"
        );
        assert_eq!(
            (requant.saturate_lo, requant.saturate_hi),
            (lo as i32, hi as i32),
            "{layer_name}: clamp bounds must follow the manifest activation"
        );
        assert_eq!(
            requant.rounding,
            Rounding::HalfAwayFromZero,
            "{layer_name}: rounding mode"
        );
        assert!(
            requant.multiplier_error_ppm.is_some(),
            "{layer_name}: the fixed-point approximation error must be recorded"
        );

        // The weight tensor is named by the shared convention and has the
        // (out_channels, channels_per_group, kh, kw) shape the blob uses.
        let weight = plan.tensor(detail.weights).expect("weight tensor");
        assert_eq!(weight.name, tessera::lower::weight_tensor_name(layer_name));
        let channels_per_group = if layer_name == "dw" {
            1
        } else {
            layer.in_channels / layer.groups.max(1)
        };
        assert_eq!(
            (
                weight.shape.n,
                weight.shape.c,
                weight.shape.h,
                weight.shape.w
            ),
            (
                layer.out_channels,
                channels_per_group,
                layer.kernel[0],
                layer.kernel[1]
            ),
            "{layer_name}: weight tensor shape"
        );
        assert_eq!(
            weight.elem,
            ElemType::I8,
            "{layer_name}: weight element type"
        );
        assert_eq!(
            report.weight_tensors.get(layer_name).map(String::as_str),
            Some(weight.name.as_str()),
            "{layer_name}: the report names the same weight tensor"
        );

        // A non-zero int32 bias must survive; an all-zero bias is the identity
        // and is dropped with a note.
        let has_bias = layer.bias_int32.iter().any(|value| *value != 0);
        assert_eq!(
            detail.bias.is_some(),
            has_bias,
            "{layer_name}: bias presence must follow the declared int32 bias"
        );
        if let Some(bias) = detail.bias {
            let tensor = plan.tensor(bias).expect("bias tensor");
            assert_eq!(
                tensor.shape.n, layer.out_channels,
                "{layer_name}: bias length"
            );
            assert_eq!(
                tensor.elem,
                ElemType::I32,
                "{layer_name}: bias element type"
            );
        }

        assert_eq!(detail.groups, layer.groups, "{layer_name}: groups");
        assert_eq!(detail.stride, layer.stride, "{layer_name}: stride");
        let expected_padding = if layer_name == "stem" || layer_name == "dw" {
            Padding::Reflect
        } else {
            Padding::Zero {
                top: 0,
                bottom: 0,
                left: 0,
                right: 0,
            }
        };
        assert_eq!(
            detail.padding, expected_padding,
            "{layer_name}: padding rule"
        );
        assert_eq!(
            detail.activation.kind,
            if layer.out_unsigned {
                ActivationKind::UnsignedClamp
            } else if layer.out_relu {
                ActivationKind::Relu
            } else {
                ActivationKind::None
            },
            "{layer_name}: activation kind"
        );
    }

    // The depthwise layer keeps one input channel per group.
    let dw = plan.ops.iter().find(|op| op.name == "dw").expect("dw");
    let OpDetail::Conv(dw_detail) = &dw.detail else {
        panic!("dw must lower to a convolution");
    };
    assert_eq!(dw_detail.groups, 4);

    // The residual add: the branch factor is point_scales[1] / point_scales[3],
    // serialized independently here.
    let add = plan
        .ops
        .iter()
        .find(|op| op.name.starts_with("add."))
        .expect("the residual add lowers");
    let factor = fixture.manifest.point_scales[1] / fixture.manifest.point_scales[3];
    let (add_m0, add_shift) = serialize_requant(&[factor]);
    let OpDetail::Elementwise(add_detail) = &add.detail else {
        panic!("the residual must lower to an elementwise operator");
    };
    let add_requant = add_detail.requant.as_ref().expect("the branch requantizes");
    assert_eq!(add_requant.m0, add_m0[0], "residual branch multiplier");
    assert_eq!(add_requant.shift, add_shift as i32, "residual branch shift");
    assert_eq!(
        (add_requant.saturate_lo, add_requant.saturate_hi),
        (-127, 127),
        "the branch operand is clamped to the symmetric activation range"
    );

    // The residual carries a 9-bit value: the plan must not model it as i8.
    let add_output = plan.tensor(add.outputs[0]).expect("residual output tensor");
    assert!(
        matches!(add_output.elem, ElemType::I9 | ElemType::I16),
        "the residual output must widen past i8, got {:?}",
        add_output.elem
    );

    // The upsample duplicates a 2x2 tile and preserves the widened element type.
    let upsample = plan
        .ops
        .iter()
        .find(|op| op.name.starts_with("upsample_nearest2x."))
        .expect("the upsample lowers");
    let OpDetail::Upsample(upsample_detail) = &upsample.detail else {
        panic!("the upsample must lower to an upsample operator");
    };
    assert_eq!(upsample_detail.scale, 2);
    let up_output = plan.tensor(upsample.outputs[0]).expect("upsample output");
    assert_eq!((up_output.shape.h, up_output.shape.w), (10, 10));
    assert_eq!(
        up_output.elem, add_output.elem,
        "duplication changes no code"
    );

    // The unsigned head feeds the output clamp, and the clamp produces the image.
    let head = plan.ops.iter().find(|op| op.name == "head").expect("head");
    let head_output = plan.tensor(head.outputs[0]).expect("head output");
    assert_eq!(
        head_output.elem,
        ElemType::U8,
        "the unsigned head writes u8 codes"
    );
    let quantize = plan
        .ops
        .iter()
        .find(|op| op.name == "quantize_u8_output")
        .expect("the output clamp");
    let OpDetail::Elementwise(quantize_detail) = &quantize.detail else {
        panic!("the output clamp is elementwise");
    };
    assert_eq!(
        quantize_detail.activation.kind,
        ActivationKind::UnsignedClamp,
        "the image is clamped to [0, output_qmax]"
    );
    assert_eq!(
        quantize.inputs, head.outputs,
        "the clamp reads the head's tensor"
    );
    assert!(
        quantize_detail.requant.is_none(),
        "the output clamp does not requantize"
    );
}

#[test]
fn acceptance_graph_plans_feasibly_with_one_step_per_operator() {
    let fixture = Fixture::load("graph");
    let artifact = frontend::import_manifest(&fixture.dir).expect("import");
    let (plan, _) = lower_graph(
        artifact.graph(),
        InputShape::new(GRAPH_SHAPE.0, GRAPH_SHAPE.1, GRAPH_SHAPE.2).expect("shape"),
    )
    .expect("lower");

    let layout = planner::solve(
        &plan,
        &target(),
        &policy("policy.toml"),
        &provenance(&artifact.provenance().manifest_sha256),
    )
    .expect("solve the acceptance graph");
    assert_eq!(
        layout.status,
        PlanStatus::Feasible,
        "the acceptance graph must be feasible under the acceptance policy: {:?}",
        layout.rejections.counts
    );
    layout.verify().expect("the selected layout re-verifies");
    layout
        .validate()
        .expect("the layout satisfies its own invariants");

    let execution: ExecutionPlan = layout.execution_plan().expect("projection");
    assert_eq!(
        execution.steps.len(),
        plan.ops.len(),
        "fusion is disabled in the acceptance policy, so every operator is one step"
    );
    for (step, op) in execution.steps.iter().zip(&plan.ops) {
        assert_eq!(step.op, op.name, "the schedule preserves the lowered order");
    }
    for step in &execution.steps {
        for name in [&step.input_buffer, &step.output_buffer] {
            let binding = execution
                .buffers
                .iter()
                .find(|buffer| &buffer.name == name)
                .unwrap_or_else(|| panic!("step {} buffer {name} is undeclared", step.op));
            assert_ne!(
                binding.placement,
                tessera::layout::MemoryPlacement::Ddr,
                "step {} must stream {name} through an on-chip buffer",
                step.op
            );
        }
    }

    // The stem step carries the per-channel multipliers the reference derives.
    let stem_layer = fixture.manifest.layer("stem");
    let (stem_m0, stem_shift, (stem_lo, stem_hi)) = derived_requant(&fixture.manifest, stem_layer);
    let stem = execution
        .steps
        .iter()
        .find(|step| step.op == "stem")
        .expect("stem is scheduled");
    let stem_requant = stem.requant.as_ref().expect("stem requantizes");
    assert_eq!(stem_requant.per_channel_multipliers, stem_m0);
    assert_eq!(stem_requant.m0, *stem_m0.iter().max().expect("multipliers"));
    assert_eq!(stem_requant.shift, stem_shift as i32);
    assert_eq!(
        (stem_requant.saturate_lo, stem_requant.saturate_hi),
        (stem_lo as i32, stem_hi as i32)
    );

    // The residual step declares a code interval that leaves the i8 domain and
    // contains every code the reference produced: a 9-bit activation is plan
    // data, not an emitter detail.
    let frames = run_fixture(&fixture);
    let residual = frames
        .iter()
        .flat_map(|frame| frame.ops[3].plane.codes.iter().copied())
        .collect::<Vec<_>>();
    let observed = CodeRange::new(
        *residual.iter().min().expect("codes"),
        *residual.iter().max().expect("codes"),
    );
    let add_step = execution
        .steps
        .iter()
        .find(|step| step.op.starts_with("add."))
        .expect("the residual is scheduled");
    let declared = add_step
        .out_code_range
        .unwrap_or_else(|| panic!("the residual step must declare its output code range"));
    assert!(
        !declared.fits_i8(),
        "the residual must be declared wider than i8, got {declared:?}"
    );
    assert!(
        declared.contains(observed),
        "the declared residual interval {declared:?} must contain every observed code {observed:?}"
    );
    let upsample = execution
        .steps
        .iter()
        .find(|step| step.op.starts_with("upsample_nearest2x."))
        .expect("the upsample is scheduled");
    assert_eq!(
        upsample.upsample_scale,
        Some(2),
        "the upsample factor is plan data"
    );
    if let (Some(residual_range), Some(upsample_range)) =
        (add_step.out_code_range, upsample.out_code_range)
    {
        assert_eq!(
            upsample_range, residual_range,
            "duplication changes no code, so the interval must be preserved"
        );
    }
}

#[test]
fn acceptance_graph_is_infeasible_when_the_frame_budget_is_impossible() {
    let fixture = Fixture::load("graph");
    let artifact = frontend::import_manifest(&fixture.dir).expect("import");
    let (plan, _) = lower_graph(
        artifact.graph(),
        InputShape::new(GRAPH_SHAPE.0, GRAPH_SHAPE.1, GRAPH_SHAPE.2).expect("shape"),
    )
    .expect("lower");

    let layout = planner::solve(
        &plan,
        &target(),
        &policy("policy_impossible.toml"),
        &provenance("acceptance-impossible"),
    )
    .expect("an impossible budget is an infeasible layout, not an error");
    assert_eq!(
        layout.status,
        PlanStatus::Infeasible,
        "a one-cycle frame budget cannot hold the acceptance graph"
    );
    assert!(
        layout.selection.is_none(),
        "an infeasible layout selects nothing"
    );
    let rejections = &layout.rejections;
    assert!(
        rejections.rejected_candidates > 0,
        "an infeasible layout must explain what it rejected"
    );
    assert!(
        !rejections.counts.is_empty(),
        "an infeasible layout must categorise its rejections"
    );
    // The category names are the planner's own vocabulary and are being
    // extended, so the suite checks that they are present and accounted for
    // rather than pinning the whole set.
    for (category, count) in &rejections.counts {
        assert!(
            !category.trim().is_empty(),
            "a rejection category must be named"
        );
        assert!(*count > 0, "category {category:?} is counted but zero");
    }
    let counted: u64 = rejections.counts.values().sum();
    assert_eq!(
        counted, rejections.rejected_candidates,
        "every rejected candidate is counted exactly once: {:?}",
        rejections.counts
    );
    assert!(
        rejections.counts.contains_key("ddr_read_budget"),
        "one cycle buys 9 bytes of reads and the graph input alone is 75, so the read budget \
         is the binding constraint: {:?}",
        rejections.counts
    );
    layout
        .validate()
        .expect("the infeasible layout is a valid artifact");
}

#[test]
fn acceptance_target_with_no_usable_dsp_is_infeasible_not_an_error() {
    let fixture = Fixture::load("graph");
    let artifact = frontend::import_manifest(&fixture.dir).expect("import");
    let (plan, _) = lower_graph(
        artifact.graph(),
        InputShape::new(GRAPH_SHAPE.0, GRAPH_SHAPE.1, GRAPH_SHAPE.2).expect("shape"),
    )
    .expect("lower");

    let exhausted =
        Ti60Target::from_toml_file(fixture_dir("").join("target_impossible.toml")).expect("target");
    let layout = planner::solve(
        &plan,
        &exhausted,
        &policy("policy.toml"),
        &provenance("acceptance-no-dsp"),
    )
    .expect("a fully reserved DSP budget is an infeasible layout, not an error");
    assert_eq!(layout.status, PlanStatus::Infeasible);
    assert!(
        layout.rejections.counts.contains_key("dsp_budget"),
        "the exhausted resource must be named: {:?}",
        layout.rejections.counts
    );
    for sample in &layout.rejections.samples {
        assert!(
            !sample.reason.to_string().is_empty(),
            "every rejection sample must carry a reason"
        );
    }
    layout
        .validate()
        .expect("the infeasible layout is a valid artifact");
}

#[test]
fn stale_provenance_is_rejected_after_the_solved_input_changes() {
    let out = stage_dir("stale-provenance");
    let out_arg = out.display().to_string();
    let graph = fixture_dir("graph");

    acceptance_support::run_ok(&[
        "import".into(),
        graph.display().to_string(),
        "--input-shape".into(),
        "1,5,5,3".into(),
        "--out".into(),
        out_arg.clone(),
    ]);
    acceptance_support::run_ok(&[
        "solve".into(),
        "--graph".into(),
        out.join(stages::GRAPH_FILE).display().to_string(),
        "--target".into(),
        target_path().display().to_string(),
        "--policy".into(),
        fixture_dir("").join("policy.toml").display().to_string(),
        "--out".into(),
        out_arg.clone(),
    ]);
    stages::verify_chain(&StageDir::new(&out)).expect("the untouched chain verifies");

    // A single byte of whitespace changes the artifact the layout was solved
    // from, so the recorded hash no longer describes it.
    let artifact = out.join(stages::GRAPH_FILE);
    let mut text = std::fs::read_to_string(&artifact).expect("read the staged graph");
    text.push('\n');
    std::fs::write(&artifact, text).expect("rewrite the staged graph");

    let error = stages::verify_chain(&StageDir::new(&out))
        .expect_err("a stale solved-from hash must be rejected");
    let text = error.to_string();
    assert!(
        text.contains("solved-from input hash"),
        "the failure must name the stale check: {text}"
    );
    assert!(
        text.contains("graph.tgraph"),
        "the failure must name the changed artifact: {text}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// The MLIR-execution leg needs per-element evidence, not one aggregate. This
/// builds the oracle document for the shipped `fixtures/w8` from the suite's own
/// reference, for the stimulus that `tests/mlir_lowering_execution.rs` uses.
fn w8_execution_oracle(manifest: &Manifest, weights: &[u8], stimulus: &[i64]) -> serde_json::Value {
    let input = plane_from_nhwc(stimulus, 3, 6, 6);
    let (carry_ops, carry_image) = run_frame_with(manifest, weights, &input, ResidualCarry::Carry);
    let (wrap_ops, _) = run_frame_with(manifest, weights, &input, ResidualCarry::WrapToI8);

    let tensor = |name: &str, elem: &str, shape: (u32, u32, u32), codes: Vec<i64>| {
        let sum: i64 = codes.iter().sum();
        serde_json::json!({
            "name": name,
            "elem": elem,
            "shape": [shape.0, shape.1, shape.2],
            "sum": sum,
            "min": codes.iter().min().copied(),
            "max": codes.iter().max().copied(),
            "outside_i8": codes.iter().filter(|code| **code < -128 || **code > 127).count(),
            "codes": codes,
        })
    };

    let mut tensors = vec![tensor("input", "i8", (3, 6, 6), stimulus.to_vec())];
    for (op, wrapped) in carry_ops.iter().zip(&wrap_ops) {
        let mut value = tensor(
            &op.name,
            op.elem,
            op.plane.shape(),
            plane_to_nhwc(&op.plane),
        );
        if op.name == "add" {
            value.as_object_mut().expect("object").insert(
                "codes_if_residual_wrapped_to_i8".to_string(),
                serde_json::json!(plane_to_nhwc(&wrapped.plane)),
            );
        }
        tensors.push(value);
    }
    tensors.push(tensor(
        "image",
        "u8",
        carry_image.shape(),
        plane_to_nhwc(&carry_image),
    ));

    let head = manifest.layer("head");
    let filter: Vec<i64> = weights[head.weight_offset..head.weight_offset + head.weight_count]
        .iter()
        .map(|byte| *byte as i8 as i64)
        .collect();
    let rows: Vec<Vec<i64>> = (0..3).map(|f| filter[f * 4..f * 4 + 4].to_vec()).collect();
    let dead: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.iter().all(|code| *code <= 0))
        .map(|(index, _)| index)
        .collect();

    serde_json::json!({
        "format": "tessera.acceptance_execution_oracle.v1",
        "source": "fixtures/w8 (manifest.json + weights.bin)",
        "purpose": "per-element expected codes for the MLIR execution leg; derived by the acceptance suite's independent scalar reference, never from an emitted IR artifact",
        "stimulus": {
            "order": "NHWC",
            "shape": [1, 6, 6, 3],
            "generator": "code = ((hi*13 + wi*17 + ci*19 + 29) % 255) - 127, hi in 0..6, wi in 0..6, ci in 0..3",
            "codes": stimulus,
        },
        "derivation": [
            "acc = bias_int32[oc] + sum over taps of (activation code * weight code), int32",
            "code = clamp((acc*M0 + (acc*M0 >= 0 ? 2^(shift-1) : 2^(shift-1)-1)) >> shift, lo, hi)",
            "M0/shift derived from the manifest f32 requant vectors; clamp per activation (relu [0,127], unsigned head [0,255])",
            "residual add: sum = clamp_sym(requant(branch)) + trunk, not clamped (carried as i16)",
            "reflect padding (edge not repeated); nearest 2x upsample duplicates each code",
        ],
        "tensor_order": "NHWC, the order the emitted module and the execution driver use",
        "head_filter_rows": rows,
        "head_rows_that_are_all_non_positive": dead,
        "image_degenerate_reason": "a head row with no positive coefficient over an always-nonnegative activation is identically zero; the image is flat while such rows exist",
        "tensors": tensors,
    })
}

/// Regenerates the w8 execution oracle when `TESSERA_ACCEPTANCE_REGEN=1`, and
/// otherwise proves it is exactly reproducible from the fixture. A fixture change
/// (for example re-derived head weights) fails this test with the regeneration
/// command, so the committed oracle cannot drift silently.
#[test]
fn w8_execution_oracle_is_reproducible_from_the_fixture() {
    let oracle_path = fixture_dir("").join("w8_execution_oracle.json");
    let checked_in: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&oracle_path).expect("read the w8 execution oracle"),
    )
    .expect("parse the w8 execution oracle");

    let w8 = fixtures().join("w8");
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(w8.join("manifest.json")).expect("read the w8 manifest"),
    )
    .expect("parse the w8 manifest");
    let weights = std::fs::read(w8.join(&manifest.weight_file)).expect("read the w8 blob");
    let stimulus: Vec<i64> = checked_in["stimulus"]["codes"]
        .as_array()
        .expect("stimulus codes")
        .iter()
        .map(|value| value.as_i64().expect("code"))
        .collect();
    let rebuilt = w8_execution_oracle(&manifest, &weights, &stimulus);

    regenerate_or_compare(
        &oracle_path,
        &checked_in,
        &rebuilt,
        "the w8 execution oracle",
        "TESSERA_ACCEPTANCE_REGEN=1 cargo test -p tessera --test acceptance_pipeline \
         w8_execution_oracle -- --nocapture",
    );

    // Independent of the file: the tensors a comparison should use must discriminate.
    let ops = run_frame(&manifest, &weights, &plane_from_nhwc(&stimulus, 3, 6, 6)).0;
    let add = ops.iter().find(|op| op.name == "add").expect("the add");
    assert!(
        add.plane.codes.iter().any(|code| *code != 0),
        "the residual must be nonzero, or the MLIR comparison proves nothing"
    );
    assert!(
        add.plane.outside_i8() > 0,
        "the residual must leave the i8 domain, so the carry is exercised"
    );
    let upsample = ops
        .iter()
        .find(|op| op.name == "upsample_nearest2x")
        .expect("the upsample");
    assert!(upsample.plane.outside_i8() > 0);
    let wrapped = run_frame_with(
        &manifest,
        &weights,
        &plane_from_nhwc(&stimulus, 3, 6, 6),
        ResidualCarry::WrapToI8,
    )
    .1;
    let carried = run_frame_with(
        &manifest,
        &weights,
        &plane_from_nhwc(&stimulus, 3, 6, 6),
        ResidualCarry::Carry,
    )
    .1;
    assert_ne!(
        wrapped.codes, carried.codes,
        "wrapping the residual must change the image, or the discriminator is inert"
    );

    // The image may only be flat while the head has a dead row; when the fixture
    // is fixed, a varying image is required instead.
    let dead = rebuilt["head_rows_that_are_all_non_positive"]
        .as_array()
        .expect("dead rows")
        .len();
    let mut codes = carried.codes.clone();
    codes.sort_unstable();
    codes.dedup();
    if dead > 0 {
        assert_eq!(
            codes.len(),
            1,
            "a head with {dead} dead row(s) must give a flat image; if the fixture changed, \
             regenerate the oracle and require a varying image here"
        );
        eprintln!(
            "note: fixtures/w8 has {dead} head row(s) with no positive coefficient, so its image \
             is a constant and cannot serve as an image-level acceptance signal; the w8 oracle \
             compares the residual and upsampled tensors instead"
        );
    } else {
        assert!(
            codes.len() >= 4,
            "with no dead head row the image must vary, got {} distinct codes",
            codes.len()
        );
    }
}

#[test]
fn real_ink_dw_imports_and_lowers_at_a_small_input() {
    // The real exported model is large, so it stays outside the crate: the test
    // resolves an external directory and reports a skip instead of embedding a
    // copy or hard-coding a machine path.
    let dir = std::env::var("TESSERA_INK_DW_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| fixture_dir("").join("..").join("ink_dw"));
    if !dir.join("manifest.json").exists() {
        eprintln!(
            "SKIP: no real ink_dw export at {} (set TESSERA_INK_DW_DIR); nothing was verified",
            dir.display()
        );
        return;
    }

    let artifact = frontend::import_manifest(&dir).expect("import the real ink_dw export");
    let layers = artifact.graph().layers();
    let depthwise = layers
        .iter()
        .filter(|layer| layer.kind == tessera::graph::LayerKind::DepthwiseConv2d)
        .count();
    assert!(
        depthwise > 0,
        "the real model must contain depthwise layers"
    );
    assert!(
        layers.len() >= 20,
        "the real model is a deep network, got {} layers",
        layers.len()
    );
    let ops = artifact.graph().ops();
    assert!(
        ops.iter()
            .any(|op| matches!(op, tessera::graph::GraphOp::Add { .. })),
        "the real model must contain a residual/skip add"
    );
    assert!(
        ops.iter()
            .any(|op| matches!(op, tessera::graph::GraphOp::UpsampleNearest2x { .. })),
        "the real model must contain a nearest upsample"
    );

    // Small spatial extent: the manifest carries no resolution, so the caller
    // supplies one. 40x20 is the bit-exact reference size used by the exporter.
    let shape = InputShape::new(3, 20, 40).expect("40x20x3");
    let (plan, report) = lower_graph(artifact.graph(), shape).expect("lower the real ink_dw model");
    plan.validate()
        .expect("the lowered real graph is a valid planner input");
    assert_eq!(
        report.weight_tensors.len(),
        layers.len(),
        "every layer contributes one named weight tensor"
    );

    let resolver = GraphWeightResolver::load(&artifact, &dir).expect("resolve the weight blob");
    let mut resolved_bytes = 0usize;
    for layer in layers {
        let op = plan
            .ops
            .iter()
            .find(|op| op.name == layer.name.as_str())
            .unwrap_or_else(|| panic!("{} is missing from the lowered plan", layer.name));
        let OpDetail::Conv(detail) = &op.detail else {
            panic!("{} must lower to a convolution", layer.name);
        };
        assert_eq!(
            detail.requant.per_channel_multipliers.len(),
            layer.out_channels.get() as usize,
            "{}: one multiplier per output channel",
            layer.name
        );
        let weight = plan.tensor(detail.weights).expect("weight tensor");
        assert_eq!(
            weight.name,
            tessera::lower::weight_tensor_name(layer.name.as_str())
        );
        let channels_per_group = if layer.kind == tessera::graph::LayerKind::DepthwiseConv2d {
            1
        } else {
            layer.in_channels.get() / layer.groups.get().max(1)
        };
        let reference = TensorRef {
            id: weight.id,
            name: weight.name.clone(),
            shape: tessera::layout::TensorShape::new(
                layer.out_channels.get(),
                channels_per_group,
                layer.kernel.h.get(),
                layer.kernel.w.get(),
            ),
            elem: ElemType::I8,
        };
        let bytes = resolver
            .tensor_bytes(&reference)
            .unwrap_or_else(|error| panic!("{}: weight bytes: {error}", layer.name));
        assert_eq!(
            bytes.len(),
            layer.weight_count,
            "{}: the resolved slices must tile the blob exactly",
            layer.name
        );
        resolved_bytes += bytes.len();
    }
    assert_eq!(
        resolved_bytes,
        artifact.provenance().weight_bytes,
        "the resolved slices must cover the whole weight blob exactly once"
    );
}
