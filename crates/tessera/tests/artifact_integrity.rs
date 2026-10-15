//! Integrity of the audit artifacts.
//!
//! The persisted optimized MLIR module carries
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
    content_hash,
};

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
    }
}

// ---------------------------------------------------------------------------
// A fabricated analysis is refuted by re-derivation
// ---------------------------------------------------------------------------

#[test]
fn a_fabricated_analysis_is_refuted() {
    // A planning-stage module whose analysis was never re-derived must not be
    // reported as verified: re-deriving it from the text refutes it.
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
    // The empty analysis is not the analysis of this module.
    assert!(!analysis_check.ok, "{analysis_check:?}");

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
