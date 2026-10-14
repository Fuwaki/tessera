//! The MLIR stage as a computational stage: emitted, optimized, analyzed and
//! lowered into a plan the layout search actually consumes.
//!
//! These tests run the real modules against the checked-in W8 fixture — a
//! manifest with a reflect-padded 3x3 convolution, a depthwise convolution, a
//! pointwise convolution, a residual `add`, a nearest-neighbour upsample, a bias
//! and an unsigned head — and assert the properties the pipeline promises:
//!
//! * the emitted module is accepted by MLIR (`parse` + `verify`) and the pinned
//!   pass pipeline runs on it;
//! * the optimized module is analyzed into operations, buffers and loop nests,
//!   and every structural operator is still attributed to its layer or graph op;
//! * the primitives are lowered from that artifact and *verified against* the
//!   analysis, so a changed or dropped structure fails the lowering;
//! * the lowered graph solves, proving the saved optimized artifact drives the
//!   constraint solver;
//! * residual `add` codes stay widened: the sum of two `[-127, 127]` points is
//!   carried as `[-254, 254]` (nine signed bits) and is never narrowed.
//!
//! The structural and semantic tests need the `melior` feature; they report a
//! skip when it is absent instead of claiming a result they did not observe.

use std::path::{Path, PathBuf};

use tessera::frontend;
use tessera::graph::GraphArtifact;
use tessera::layout::{ElemType, ExecutionPlan, PlanStatus};
use tessera::lower::{InputShape, lower_mlir_artifact};
use tessera::mlir::{self, ElementType, MlirArtifact, MlirOptions, MlirStage, SourceGraph};
use tessera::planner::{self, Policy, ProvenanceInput};
use tessera::primitive::{self, PrimitiveArtifact};
use tessera::target::Ti60Target;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-mlir-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn w8_artifact() -> GraphArtifact {
    frontend::import_manifest(&fixtures().join("w8")).expect("import the W8 fixture")
}

/// W8 fixture input: 6x6 pixels with three channels.
fn w8_shape() -> InputShape {
    InputShape::new(3, 6, 6).expect("6x6x3 input")
}

fn emit_w8() -> MlirArtifact {
    let artifact = w8_artifact();
    mlir::lower(
        artifact.graph(),
        SourceGraph::from(&artifact),
        &MlirOptions::new(w8_shape().to_nhwc()),
    )
    .expect("emit the W8 module")
}

fn optimize_w8() -> MlirArtifact {
    mlir::optimize(&emit_w8()).expect("the pinned pipeline must optimize the W8 module")
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("load target")
}

fn policy(name: &str) -> Policy {
    Policy::from_toml_file(fixtures().join("policies").join(name)).expect("load policy")
}

fn provenance(hash: &str) -> ProvenanceInput {
    ProvenanceInput {
        graph_sha256: hash.to_string(),
        target_sha256: "target-sha".to_string(),
        policy_sha256: "policy-sha".to_string(),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

// ---------------------------------------------------------------------------
// Emission: the module MLIR 23 is asked to accept
// ---------------------------------------------------------------------------

#[test]
fn emitted_w8_module_describes_every_operator_with_a_stable_tag() {
    let artifact = emit_w8();
    assert_eq!(artifact.stage, MlirStage::Tensor);
    assert_eq!(artifact.input_shape, vec![1, 6, 6, 3]);
    assert!(
        artifact.analysis.is_none(),
        "the tensor stage is not analyzed"
    );

    // The emitted text is exactly the analyzed text: the artifact records the
    // hash of the module it stores.
    let checks = artifact.verify(Some(&w8_artifact())).expect("verify");
    for check in &checks {
        assert!(check.ok, "{}: {}", check.check, check.detail);
    }

    // Reflect padding, depthwise, pointwise, add, upsample, unsigned head.
    let text = &artifact.module;
    assert!(text.contains("%stem__padded = tensor.pad %input low[0, 1, 1, 0]"));
    assert!(text.contains("linalg.conv_2d_nhwc_fhwc"));
    assert!(text.contains("linalg.depthwise_conv_2d_nhwc_hwc"));
    assert!(text.contains("%upper = arith.constant 255 : i64"));
    for tag in [
        "conv#0",
        "pad#0",
        "epilogue#0",
        "conv#2",
        "pad#2",
        "epilogue#2",
        "conv#4",
        "epilogue#4",
        "add#6",
        "upsample#7",
        "conv#8",
        "epilogue#8",
    ] {
        assert!(
            text.contains(&format!("loc(\"tessera:{tag}\")")),
            "the module must tag {tag} so the optimized IR can be attributed"
        );
    }
}

/// A residual `add` output must stay widened.
///
/// The sum of two saturated points needs nine bits, so the emitted module must
/// carry it in `i16` and must never truncate it back to a byte, even though the
/// convolution reading it may be the named or the generalized form. This is a
/// static check of the emitter's own numerical contract; the emitted IR is also
/// executed against the reference in `mlir_lowering_execution.rs`, which is
/// where the behavior itself is proven.
#[test]
fn residual_add_codes_are_widened_and_a_wide_conv_widens_them_explicitly() {
    let artifact = emit_w8();
    let text = &artifact.module;

    // The sum of two saturated points is carried in i16 and is never truncated
    // back to a byte: `trunci i32 -> i16` keeps every code the reference keeps.
    assert!(
        text.contains("%sum = arith.trunci %sum32 : i32 to i16"),
        "the add sum must not be narrowed to i8"
    );
    assert!(
        !text.contains("%sum8 = arith.trunci %sum32 : i32 to i8"),
        "an i8 sum would wrap the reference's [-254, 254] range"
    );

    // The tensor table records the exact interval, in the emitter's vocabulary.
    let add_output = artifact
        .tensors
        .iter()
        .find(|tensor| tensor.element == ElementType::I16)
        .expect("the add output is transported in i16");
    // The trunk is a relu point (`[0, 127]`) and the branch is the same relu
    // point requantized by a positive factor and clamped, so it stays in
    // `[0, 127]`: the unclamped sum spans `[0, 254]` — nine signed bits, and
    // nine-bit arithmetic is what a backend must instantiate.
    let range = add_output.code_range.expect("an interval");
    assert_eq!((range.lo, range.hi), (0, 254));
    assert_eq!(range.signed_bits(), 9, "nine signed bits, not sixteen");
    assert_eq!((add_output.shape[1], add_output.shape[2]), (6, 6));

    // The convolution that reads the widened point is an explicit generic
    // convolution: the named linalg form takes one element type for both the
    // activation and the filter, which cannot express i16 codes x i8 weights.
    assert!(
        text.contains("linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 1 + d4, d2 * 1 + d5, d6)>"),
        "the head convolution must be an explicit generic over the widened codes"
    );
    assert!(text.contains("%activation32 = arith.extsi %activation : i16 to i32"));
    assert!(text.contains("%weight32 = arith.extsi %weight : i8 to i32"));

    // The accumulator legality check ran: the operator records an i32 accumulator
    // bound it proved.
    let accumulators: Vec<&tessera::mlir::TensorBinding> = artifact
        .tensors
        .iter()
        .filter(|tensor| tensor.element == ElementType::I32)
        .collect();
    assert!(
        !accumulators.is_empty(),
        "accumulators are named by the module"
    );
    for accumulator in accumulators {
        let range = accumulator.code_range.expect("an accumulator bound");
        assert!(range.lo <= 0 && range.hi >= 0);
        assert!(
            range.hi <= i32::MAX as i64,
            "the i32 accumulator datapath must be provably sufficient"
        );
    }
}

// ---------------------------------------------------------------------------
// Optimization: MLIR parses, verifies and transforms the emitted text
// ---------------------------------------------------------------------------

#[test]
fn the_optimized_w8_module_analyzes_into_loops_buffers_and_fused_groups() {
    if !mlir::melior_available() {
        eprintln!(
            "SKIP: this build has no `melior` feature, so MLIR cannot parse or optimize the module"
        );
        return;
    }
    let optimized = optimize_w8();
    assert_eq!(optimized.stage, MlirStage::Planning);
    assert!(optimized.provenance.melior.is_some());
    let analysis = optimized
        .analyzed()
        .expect("the optimized module is analyzed");

    // The pipeline really ran: loops and buffers exist that the tensor-stage text
    // does not contain.
    assert!(analysis.counts.loops > 0, "{analysis:?}");
    assert!(analysis.counts.buffers > 0);
    assert_eq!(
        analysis.counts.dialects.get("affine").copied().unwrap_or(0) > 0,
        true
    );
    assert!(analysis.counts.unrecognized_tags.is_empty());

    // Every structural operator is still attributed in the optimized IR.
    for operator in &optimized.operators {
        match operator.kind {
            tessera::mlir::OperatorKind::Relu | tessera::mlir::OperatorKind::QuantizeU8 => {
                assert_eq!(analysis.tag_count(&operator.tag), 0);
            }
            _ => assert!(
                analysis.tag_count(&operator.tag) > 0,
                "operator {} ({:?}) lost its operations",
                operator.index,
                operator.kind
            ),
        }
    }

    // The residual add and the pointwise convolution that consumes it are fused
    // into one loop nest, and the nest records the operations it performs.
    let fused = analysis.fused_nests();
    assert!(!fused.is_empty(), "affine-loop-fusion must fuse something");
    let nest = fused
        .iter()
        .find(|nest| nest.operator_indices.len() > 1)
        .expect("a nest with more than one operator");
    let add = optimized
        .operators
        .iter()
        .find(|operator| operator.kind == tessera::mlir::OperatorKind::Add)
        .expect("the fixture has a residual add");
    assert!(
        nest.operator_indices.contains(&add.index),
        "the residual add must be attributed to a fused nest"
    );
    assert!(
        nest.operations.values().sum::<usize>() > 0,
        "a nest records the operations it performs"
    );

    // The reflect padding materialized as a padded buffer, exactly the shape the
    // layer declares: 6x6 input, one pixel of reflection, three channels.
    let padded = analysis.buffers_with(&[1, 8, 8, 3], "i8");
    assert!(
        !padded.is_empty(),
        "the padding of `stem` must materialize: {:?}",
        analysis.buffers
    );
    assert_eq!(padded[0].bytes, Some(8 * 8 * 3));
}

#[test]
fn optimization_refuses_without_melior_instead_of_pretending() {
    let emitted = emit_w8();
    match mlir::optimize(&emitted) {
        Ok(optimized) => {
            assert!(mlir::melior_available());
            assert!(optimized.analysis.is_some());
        }
        Err(error) => {
            assert!(!mlir::melior_available());
            assert!(error.to_string().contains("melior"), "{error}");
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive lowering: driven by the optimized artifact, verified against it
// ---------------------------------------------------------------------------

fn lowered_w8() -> (MlirArtifact, PrimitiveArtifact) {
    let optimized = optimize_w8();
    let primitives = primitive::lower(&optimized).expect("lower the optimized artifact");
    (optimized, primitives)
}

#[test]
fn the_optimized_artifact_lowers_to_primitives_that_keep_every_binding() {
    if !mlir::melior_available() {
        eprintln!("SKIP: primitive lowering is driven by the structured analysis");
        return;
    }
    let (optimized, primitives) = lowered_w8();
    assert_eq!(primitives.format, primitive::PRIMITIVE_FORMAT);
    assert_eq!(primitives.input_shape, vec![1, 6, 6, 3]);
    primitives
        .plan
        .validate()
        .expect("the lowered primitive graph is valid");
    primitives.check(&optimized).expect("provenance holds");

    // The operator set and order are the manifest's, minus the relu markers.
    let names: Vec<&str> = primitives
        .plan
        .ops
        .iter()
        .map(|op| op.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "stem",
            "dw",
            "mid",
            "add.3",
            "upsample_nearest2x.3",
            "head",
            "quantize_u8_output"
        ]
    );

    // Weights, bias and per-channel requantization survive the round trip through
    // the module and the optimizer.
    let stem = primitives
        .plan
        .ops
        .iter()
        .find(|op| op.name == "stem")
        .expect("stem");
    let tessera::planner::OpDetail::Conv(stem_detail) = &stem.detail else {
        panic!("stem must lower to a convolution");
    };
    assert_eq!(stem_detail.requant.per_channel_multipliers.len(), 4);
    assert!(stem_detail.requant.multiplier_error_ppm.is_some());
    assert!(stem_detail.bias.is_some(), "stem declares a non-zero bias");
    assert_eq!(
        primitives
            .plan
            .tensor(stem_detail.weights)
            .expect("stem weights")
            .name,
        "weights:stem"
    );
    let mid = primitives
        .plan
        .ops
        .iter()
        .find(|op| op.name == "mid")
        .expect("mid");
    let tessera::planner::OpDetail::Conv(mid_detail) = &mid.detail else {
        panic!("mid must lower to a convolution");
    };
    assert!(mid_detail.bias.is_none(), "an all-zero bias is an identity");

    // Padding is carried as a rule, not as a shape, and reflect survives.
    let stem_padding = stem_detail.padding;
    assert!(
        matches!(stem_padding, tessera::planner::Padding::Reflect),
        "{stem_padding:?}"
    );
}

#[test]
fn widened_codes_reach_the_planner_as_nine_bit_values() {
    if !mlir::melior_available() {
        eprintln!("SKIP: primitive lowering is driven by the structured analysis");
        return;
    }
    let (_, primitives) = lowered_w8();
    let add = primitives
        .plan
        .ops
        .iter()
        .find(|op| op.name == "add.3")
        .expect("add.3");
    let output = primitives
        .plan
        .tensor(add.outputs[0])
        .expect("the add output tensor");
    assert_eq!(
        output.elem,
        ElemType::I9,
        "a nine-bit value in a sixteen-bit word, not a byte"
    );
    assert_eq!(output.elem.bits(), 9);
    assert_eq!(output.elem.storage_bits(), 16);
    let range = output.code_range.expect("the exact interval");
    assert_eq!((range.lo, range.hi), (0, 254));

    // The plan says the sum is not clamped to a byte, and the interval is
    // persisted next to the transport type.
    let tessera::planner::OpDetail::Elementwise(detail) = &add.detail else {
        panic!("add must lower to an elementwise op");
    };
    assert_eq!(detail.activation.clamp_lo, 0);
    assert_eq!(detail.activation.clamp_hi, 254);
    let interval = primitives
        .interval(&output.name)
        .expect("the artifact records the interval");
    assert_eq!(
        (interval.lo, interval.hi, interval.signed_bits),
        (0, 254, 9)
    );
    assert_eq!(interval.mlir_element, "i16");

    // The upsampled point carries the same widened codes, and the unsigned head
    // still consumes them: the fixture's add feeds upsample then head.
    let upsample = primitives
        .plan
        .ops
        .iter()
        .find(|op| op.name == "upsample_nearest2x.3")
        .expect("upsample");
    let upsampled = primitives
        .plan
        .tensor(upsample.outputs[0])
        .expect("upsample output");
    assert_eq!(upsampled.elem, ElemType::I9);
    let head = primitives
        .plan
        .ops
        .iter()
        .find(|op| op.name == "head")
        .expect("head");
    assert_eq!(head.inputs, upsample.outputs);
    let image = primitives
        .plan
        .tensor(head.outputs[0])
        .expect("head output");
    assert_eq!(image.elem, ElemType::U8);
    assert_eq!(
        image.code_range,
        Some(tessera::layout::CodeRange::new(0, 255))
    );
}

#[test]
fn a_saved_optimized_artifact_drives_the_layout_search() {
    if !mlir::melior_available() {
        eprintln!("SKIP: the optimize stage needs Melior");
        return;
    }
    let (optimized, primitives) = lowered_w8();

    // Persist and reload both artifacts, then solve from what was read back.
    let dir = temp_dir("solve");
    let primitives_path = dir.join("primitives.json");
    primitives.save(&primitives_path).expect("save primitives");
    let reloaded = PrimitiveArtifact::load(&primitives_path).expect("load primitives");
    assert_eq!(reloaded, primitives);
    reloaded.check(&optimized).expect("reloaded provenance");
    let plan_path = dir.join("primitive.json");
    std::fs::write(&plan_path, reloaded.plan_json().expect("plan json")).expect("write plan");

    // `tessera solve --primitive` path: the plan parses as a PlanInput.
    let text = std::fs::read_to_string(&plan_path).expect("read plan");
    let plan = tessera::stages::parse_primitive(&text).expect("the plan is a valid PlanInput");

    let layout = planner::solve(
        &plan,
        &target(),
        &policy("explore.toml"),
        &provenance(&optimized.provenance.source_content_hash),
    )
    .expect("solving the lowered plan");
    assert_eq!(layout.status, PlanStatus::Feasible);
    layout.verify().expect("the layout re-verifies");
    let execution: ExecutionPlan = layout.execution_plan().expect("projection");
    assert_eq!(
        execution.steps.len(),
        plan.ops.len(),
        "every lowered operator is scheduled"
    );

    // The schedule carries the interval, which is what sizes the datapath.
    let add_step = execution
        .steps
        .iter()
        .find(|step| step.op == "add.3")
        .expect("add.3 is scheduled");
    // The trunk arrives as a saturated relu point, the sum leaves widened.
    let in_range = add_step
        .in_code_range
        .expect("the input interval is recorded");
    assert_eq!((in_range.lo, in_range.hi), (0, 127));
    let out_range = add_step
        .out_code_range
        .expect("the output interval is recorded");
    assert_eq!((out_range.lo, out_range.hi), (0, 254));
    assert_eq!(out_range.signed_bits(), 9);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fusion_and_buffers_of_the_optimized_module_are_persisted() {
    if !mlir::melior_available() {
        eprintln!("SKIP: primitive lowering is driven by the structured analysis");
        return;
    }
    let (_, primitives) = lowered_w8();
    assert!(!primitives.groups.is_empty());
    let fused = primitives.fused_groups();
    assert!(!fused.is_empty(), "the optimizer fused loop nests");
    let group = fused
        .iter()
        .find(|group| group.members.len() > 1)
        .expect("a group with more than one operator");
    let names: Vec<&str> = group
        .members
        .iter()
        .map(|member| member.name.as_str())
        .collect();
    assert!(
        names.contains(&"add.3"),
        "the residual add is fused into a nest: {names:?}"
    );
    for member in &group.members {
        assert!(
            !member.roles.is_empty(),
            "roles name the operations: {member:?}"
        );
    }

    // Buffers come from the optimized module, not from a shape convention. The
    // fixture's buffers are all statically shaped, so the allocation total is
    // known; a dynamically shaped buffer would make it unknown instead.
    assert_eq!(
        primitives.dynamic_buffers(),
        0,
        "the fixture's buffers are statically shaped"
    );
    let allocated = primitives
        .allocated_bytes()
        .expect("a statically shaped module has a known allocation total");
    assert!(
        allocated >= 3 * 8 * 8,
        "the padded inputs alone exceed the naive input size, found {allocated}"
    );
    assert!(
        primitives
            .buffers
            .iter()
            .any(|buffer| buffer.shape == vec![1, 8, 8, 3] && buffer.element == "i8"),
        "the padded stem input is materialized: {:?}",
        primitives.buffers
    );
    assert!(
        primitives
            .buffers
            .iter()
            .any(|buffer| buffer.element == "i32"),
        "accumulators are explicitly i32"
    );
}

// ---------------------------------------------------------------------------
// The verification is load-bearing
// ---------------------------------------------------------------------------

#[test]
fn primitive_lowering_rejects_a_tensor_stage_artifact() {
    let emitted = emit_w8();
    let error = lower_mlir_artifact(&emitted).expect_err("the tensor stage must be refused");
    assert!(
        error.to_string().contains("analysis") || error.to_string().contains("module"),
        "{error}"
    );
}

#[test]
fn primitive_lowering_rejects_an_analysis_that_lost_an_operator() {
    if !mlir::melior_available() {
        eprintln!("SKIP: primitive lowering is driven by the structured analysis");
        return;
    }
    let optimized = optimize_w8();
    let mut tampered = optimized.clone();
    let analysis = tampered.analysis.as_mut().expect("analysis");
    // Drop every trace of the depthwise convolutions from the *recorded*
    // analysis — its tags, its operations and its loop nest — while leaving the
    // recorded module hash untouched, which is exactly the tamper the
    // self-declared hash used to accept.
    analysis
        .tags
        .retain(|tag, _| !tag.starts_with("conv#2") && !tag.starts_with("pad#2"));
    analysis
        .nests
        .retain(|nest| !nest.operator_indices.contains(&2));
    assert_eq!(
        tampered.analysis.as_ref().unwrap().module_hash,
        optimized.provenance.module_hash,
        "the tamper must keep the recorded module hash to be meaningful"
    );
    let error = lower_mlir_artifact(&tampered).expect_err("a lost operator must fail the lowering");
    assert!(
        error.to_string().contains("analysis"),
        "the error must be the re-derived analysis mismatch: {error}"
    );
}

#[test]
fn primitive_lowering_rejects_a_padding_that_did_not_materialize() {
    if !mlir::melior_available() {
        eprintln!("SKIP: primitive lowering is driven by the structured analysis");
        return;
    }
    let optimized = optimize_w8();
    let mut tampered = optimized.clone();
    let analysis = tampered.analysis.as_mut().expect("analysis");
    // Pretend the optimizer lost the padded buffer of `stem`.
    analysis
        .buffers
        .retain(|buffer| buffer.shape != vec![1, 8, 8, 3]);
    let error =
        lower_mlir_artifact(&tampered).expect_err("a missing padded buffer must fail the lowering");
    assert!(
        error.to_string().contains("analysis"),
        "the error must be the re-derived analysis mismatch: {error}"
    );
}

#[test]
fn artifact_verification_detects_a_tampered_module() {
    let mut artifact = emit_w8();
    artifact.module.push_str("// tampered\n");
    // The recorded hash is recomputed from the text, so a module whose bytes
    // changed under a stale hash is rejected at validation and by `check`.
    assert!(
        artifact.validate_header().is_err(),
        "the stored text no longer hashes to the recorded module hash"
    );
    assert!(
        artifact.verify(None).is_err(),
        "verify must recompute the module hash rather than trust the recorded one"
    );
    assert!(artifact.check(None).is_err());
}
