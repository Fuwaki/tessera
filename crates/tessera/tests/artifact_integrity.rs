//! Integrity of the audit artifacts.
//!
//! The persisted optimized module and the primitive graph lowered from it carry
//! evidence — a structured analysis, activation intervals, fusion groups,
//! buffers, provenance. This suite drives the *tamper* path the review found: it
//! rewrites that evidence while keeping the recorded hashes, and asserts the
//! artifact layer refuses it instead of trusting its own metadata.
//!
//! The structural checks need the `melior` feature (the analysis is re-derived
//! with Melior); the textual ones run everywhere. A build without Melior must
//! report analysis verification as impossible rather than as verified.

use std::path::{Path, PathBuf};

use tessera::frontend;
use tessera::graph::GraphArtifact;
use tessera::mlir::{
    self, ArtifactMetrics, ElementType, MeliorProvenance, MlirArtifact, MlirOptions,
    MlirProvenance, MlirStage, PINNED_MELIOR_VERSION, SourceGraph, TensorBinding,
    analysis_verification_available, content_hash,
};
use tessera::primitive::{self, PrimitiveArtifact};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-integrity-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn w8_artifact() -> GraphArtifact {
    frontend::import_manifest(&fixtures().join("w8")).expect("import the W8 fixture")
}

fn emit_w8() -> MlirArtifact {
    let artifact = w8_artifact();
    mlir::lower(
        artifact.graph(),
        SourceGraph::from(&artifact),
        &MlirOptions::new([1, 6, 6, 3]),
    )
    .expect("emit the W8 module")
}

fn optimize_w8() -> MlirArtifact {
    mlir::optimize(&emit_w8()).expect("the pinned pipeline must optimize the W8 module")
}

fn lowered(optimized: &MlirArtifact) -> PrimitiveArtifact {
    primitive::lower(optimized).expect("lower the optimized module")
}

// ---------------------------------------------------------------------------
// Textual integrity: recomputed hashes, no Melior needed
// ---------------------------------------------------------------------------

#[test]
fn a_module_that_no_longer_hashes_its_text_is_rejected() {
    // An artifact is only as trustworthy as the module it stores. Editing that
    // text while keeping the recorded hash must be rejected when the artifact is
    // validated, and by `verify` and `check` too.
    let optimized = MlirArtifact {
        format: tessera::mlir::MLIR_ARTIFACT_FORMAT.to_string(),
        schema_version: tessera::mlir::MLIR_ARTIFACT_SCHEMA_VERSION,
        function: "tessera_graph".to_string(),
        stage: MlirStage::Tensor,
        module: "module {\n}\n".to_string(),
        input_shape: vec![1, 4, 4, 3],
        arguments: Vec::new(),
        tensors: Vec::new(),
        layers: Vec::new(),
        operators: Vec::new(),
        output: TensorBinding {
            ssa_name: "out".to_string(),
            shape: vec![1, 1, 1, 1],
            element: ElementType::I8,
            point: None,
            scale: None,
            code_range: None,
        },
        metrics: ArtifactMetrics {
            operators: 0,
            layers: 0,
            tensors: 0,
            macs: 0,
            requantizations: 0,
            structural: None,
        },
        analysis: None,
        provenance: MlirProvenance {
            source_format: "tessera.graph".to_string(),
            source_schema_version: 1,
            source_content_hash: content_hash("graph"),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            melior: None,
            module_hash: content_hash("module {\n}\n"),
        },
    };
    assert!(optimized.validate_header().is_ok());

    let mut tampered = optimized.clone();
    tampered.module.push_str("// edited after the fact\n");
    assert!(
        tampered.validate_header().is_err(),
        "validate must recompute the module hash, not trust the recorded one"
    );
    assert!(tampered.verify(None).is_err());
    assert!(tampered.check(None).is_err());
}

// ---------------------------------------------------------------------------
// Structural integrity: the analysis is re-derived, not trusted
// ---------------------------------------------------------------------------

#[test]
fn a_tampered_recording_of_a_module_is_refused() {
    if !analysis_verification_available() {
        eprintln!("SKIP: re-deriving the analysis needs the `melior` feature");
        return;
    }
    let optimized = optimize_w8();
    let recorded = optimized.provenance.module_hash.clone();

    // Every one of these rewrites the *recorded* analysis while leaving the
    // recorded module hash untouched — the exact tamper a self-declared hash
    // cannot see.
    let mut tamps: Vec<(&str, MlirArtifact)> = Vec::new();

    let mut dropped_tag = optimized.clone();
    dropped_tag
        .analysis
        .as_mut()
        .unwrap()
        .tags
        .retain(|tag, _| !tag.starts_with("conv#2"));
    tamps.push(("a dropped tag", dropped_tag));

    let mut dropped_nest = optimized.clone();
    dropped_nest
        .analysis
        .as_mut()
        .unwrap()
        .nests
        .retain(|nest| !nest.operator_indices.contains(&2));
    tamps.push(("a dropped loop nest", dropped_nest));

    let mut bumped_operations = optimized.clone();
    {
        let analysis = bumped_operations.analysis.as_mut().unwrap();
        let count = analysis.operations.values_mut().next().unwrap();
        *count += 1;
    }
    tamps.push(("a rewritten operation count", bumped_operations));

    let mut dropped_buffer = optimized.clone();
    dropped_buffer
        .analysis
        .as_mut()
        .unwrap()
        .buffers
        .retain(|buffer| buffer.shape != vec![1, 8, 8, 3]);
    tamps.push(("a dropped padded buffer", dropped_buffer));

    let mut lying_counts = optimized.clone();
    {
        let analysis = lying_counts.analysis.as_mut().unwrap();
        analysis.counts.loops += 7;
    }
    tamps.push(("a rewritten loop count", lying_counts));

    for (what, tampered) in tamps {
        assert_eq!(
            tampered.analysis.as_ref().unwrap().module_hash,
            recorded,
            "{what}: the tamper must keep the module hash"
        );
        // The link check passes; the re-derivation must not.
        let checks = tampered
            .verify(None)
            .expect("the header itself is still valid");
        assert!(
            checks
                .iter()
                .any(|check| !check.ok && check.check.contains("re-derived")),
            "{what}: the re-derived analysis check must fail: {checks:?}"
        );
        assert!(tampered.check(None).is_err(), "{what}: check must fail");
        assert!(
            tampered.analyzed().is_err(),
            "{what}: lowering must not use the tampered analysis"
        );
        assert!(
            primitive::lower(&tampered).is_err(),
            "{what}: the lowered plan must not be driven by the tampered analysis"
        );
    }
}

#[test]
fn a_rewritten_plan_interval_is_refused_by_check() {
    if !analysis_verification_available() {
        eprintln!("SKIP: re-lowering needs the `melior` feature");
        return;
    }
    let optimized = optimize_w8();
    let primitives = lowered(&optimized);
    primitives
        .check(&optimized)
        .expect("the honest artifact re-checks");

    // Narrow the residual add's interval from nine signed bits to eight. The
    // codes still fit the container, so nothing but a re-lowering notices.
    let mut tampered = primitives.clone();
    let interval = tampered
        .intervals
        .iter_mut()
        .find(|interval| interval.signed_bits == 9)
        .expect("the widened add interval is recorded");
    let name = interval.tensor.clone();
    assert_eq!((interval.lo, interval.hi), (0, 254));
    interval.hi = 127;
    interval.signed_bits = 8;

    let error = tampered
        .check(&optimized)
        .expect_err("a rewritten interval must fail the check");
    assert!(
        error.to_string().contains("intervals") || error.to_string().contains(&name),
        "the error must name the rewritten evidence: {error}"
    );

    // The same tamper survives a save/load round trip and is caught there too:
    // the artifact cannot launder its plan through the filesystem.
    let dir = temp_dir("interval");
    let path = dir.join("primitives.json");
    tampered.save(&path).expect("save the tampered artifact");
    let reloaded = PrimitiveArtifact::load(&path).expect("the header is still valid");
    assert_eq!(reloaded, tampered);
    assert!(
        reloaded.check(&optimized).is_err(),
        "a reloaded artifact must be re-checked, not believed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn weights_and_bias_carry_no_activation_interval() {
    if !analysis_verification_available() {
        eprintln!("SKIP: lowering needs the structured analysis");
        return;
    }
    let optimized = optimize_w8();
    let primitives = lowered(&optimized);

    // The plan still describes the weight and bias tensors, correctly typed.
    let weights = primitives
        .plan
        .tensors
        .iter()
        .find(|tensor| tensor.name == "weights:stem")
        .expect("stem weights are in the plan");
    assert!(
        matches!(
            weights.elem,
            tessera::layout::ElemType::I8 | tessera::layout::ElemType::I4
        ),
        "the weight tensor is typed by its packing, not as a 32-bit value: {:?}",
        weights.elem
    );
    let bias = primitives
        .plan
        .tensors
        .iter()
        .find(|tensor| tensor.name == "bias:stem")
        .expect("the stem bias is in the plan");
    assert_eq!(bias.elem, tessera::layout::ElemType::I32);

    // Neither is published as an activation interval, and no interval claims a
    // 32-bit width for a value that never travels as one.
    assert!(
        primitives.interval("weights:stem").is_none(),
        "a weight tensor has no activation interval"
    );
    assert!(
        primitives.interval("bias:stem").is_none(),
        "a bias tensor has no activation interval"
    );
    assert!(!primitives.intervals.is_empty(), "activations are recorded");
    for interval in &primitives.intervals {
        assert!(
            !interval.tensor.starts_with("weights:") && !interval.tensor.starts_with("bias:"),
            "{interval:?} is not an activation"
        );
        assert!(
            interval.signed_bits <= 9,
            "{interval:?} claims a wide activation where the lowering bounded it narrow"
        );
    }
}

#[test]
fn a_dynamic_allocation_is_unknown_not_zero_and_is_refused_in_planning() {
    if !analysis_verification_available() {
        eprintln!("SKIP: analyzing a module needs the `melior` feature");
        return;
    }
    let module = "module {\n  func.func @dyn(%n: index) {\n    %0 = memref.alloc(%n) : \
                  memref<4x?xi8>\n    return\n  }\n}\n";
    let analysis =
        mlir::analyze(module, PINNED_MELIOR_VERSION, &mlir::default_pipeline()).expect("analyze");

    // The allocation is recorded with an unknown shape and an unknown size; its
    // statically known extent is not published as if it were the shape, and the
    // byte total is reported as unknown rather than as zero.
    let buffer = analysis
        .buffers
        .iter()
        .find(|buffer| buffer.element == "i8")
        .expect("the allocation is recorded");
    assert!(
        buffer.shape.is_empty(),
        "a dynamic allocation has no static shape: {buffer:?}"
    );
    assert_eq!(buffer.bytes, None);
    assert_eq!(analysis.counts.dynamic_buffers, 1);
    assert_eq!(analysis.allocated_bytes(), None);
    assert_eq!(analysis.structural().allocated_bytes, 0);

    // Planning refuses a module whose allocation size it cannot determine.
    let artifact = planning_artifact(module, analysis);
    let error = artifact
        .analyzed()
        .expect_err("planning must refuse an unknown allocation size");
    assert!(
        error.to_string().contains("dynamically shaped"),
        "the refusal must say why: {error}"
    );
}

// ---------------------------------------------------------------------------
// Capability: verification is impossible without Melior, and says so
// ---------------------------------------------------------------------------

#[test]
fn analysis_verification_capability_matches_the_build() {
    assert_eq!(analysis_verification_available(), mlir::melior_available());

    // A planning-stage module whose analysis was never re-derived must not be
    // reported as verified. In a Melior build the fake analysis is refuted; in a
    // build without it the refutation is impossible and is reported as such.
    let module = "module {\n  func.func @f() {\n    return\n  }\n}\n";
    let artifact = planning_artifact(
        module,
        mlir::ModuleAnalysis {
            format: tessera::mlir::ANALYSIS_FORMAT.to_string(),
            schema_version: tessera::mlir::ANALYSIS_SCHEMA_VERSION,
            module_hash: content_hash(module),
            melior_version: PINNED_MELIOR_VERSION.to_string(),
            pipeline: mlir::default_pipeline(),
            counts: tessera::mlir::AnalysisCounts {
                operations: 0,
                dialects: Default::default(),
                loops: 0,
                buffers: 0,
                dynamic_buffers: 0,
                buffer_bytes: 0,
                tagged_operations: 0,
                unrecognized_tags: Vec::new(),
            },
            operations: Default::default(),
            tags: Default::default(),
            buffers: Vec::new(),
            nests: Vec::new(),
        },
    );

    let checks = artifact.verify(None).expect("the header is valid");
    let analysis_check = checks
        .iter()
        .find(|check| check.check.contains("re-derived"))
        .expect("the re-derivation check is always reported");
    if mlir::melior_available() {
        // The empty analysis is not the analysis of this module.
        assert!(!analysis_check.ok, "{analysis_check:?}");
    } else {
        assert!(
            !analysis_check.ok && analysis_check.detail.contains("cannot be re-derived"),
            "without Melior the check must fail loudly, not pass: {analysis_check:?}"
        );
        assert!(artifact.analyzed().is_err());
    }
    assert!(artifact.check(None).is_err());
}

/// Build a planning-stage artifact for `module` carrying `analysis`.
fn planning_artifact(module: &str, analysis: mlir::ModuleAnalysis) -> MlirArtifact {
    MlirArtifact {
        format: tessera::mlir::MLIR_ARTIFACT_FORMAT.to_string(),
        schema_version: tessera::mlir::MLIR_ARTIFACT_SCHEMA_VERSION,
        function: "tessera_graph".to_string(),
        stage: MlirStage::Planning,
        module: module.to_string(),
        input_shape: vec![1, 4, 4, 3],
        arguments: Vec::new(),
        tensors: Vec::new(),
        layers: Vec::new(),
        operators: Vec::new(),
        output: TensorBinding {
            ssa_name: "out".to_string(),
            shape: vec![1, 1, 1, 1],
            element: ElementType::I8,
            point: None,
            scale: None,
            code_range: None,
        },
        metrics: ArtifactMetrics {
            operators: 0,
            layers: 0,
            tensors: 0,
            macs: 0,
            requantizations: 0,
            structural: None,
        },
        analysis: Some(analysis),
        provenance: MlirProvenance {
            source_format: "tessera.graph".to_string(),
            source_schema_version: 1,
            source_content_hash: content_hash("graph"),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            melior: Some(MeliorProvenance {
                melior_version: PINNED_MELIOR_VERSION.to_string(),
                pipeline: mlir::default_pipeline(),
            }),
            module_hash: content_hash(module),
        },
    }
}
