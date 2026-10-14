//! MLIR lowering layer: Graph IR -> tensor-level MLIR -> planning primitives.
//!
//! Boundaries (ADR-0001, ADR-0002, concept book §5.2):
//!
//! ```text
//! Graph IR
//!   -> emit_module        tensor-level linalg/arith/tensor text      (this crate, no MLIR needed)
//!   -> optimize           bufferized memref/affine planning text     (Melior, `melior` feature)
//!   -> MlirArtifact       versioned, inspectable, resumable artifact
//! ```
//!
//! * [`emit_module`] is pure Rust and always available. It writes deterministic
//!   MLIR text for the validated v1 operator subset and rejects everything else
//!   with a typed error instead of guessing.
//! * [`optimize`] runs the pinned pass pipeline through Melior 0.28.2:
//!   `linalg-generalize-named-ops`,
//!   `one-shot-bufferize{bufferize-function-boundaries}`,
//!   `convert-linalg-to-affine-loops`,
//!   `affine-loop-fusion{mode=producer}`, `canonicalize`, `cse`. Tensor-level
//!   fusion is deliberately *not* in the pipeline: for
//!   `depthwise -> requant -> pointwise` it moves the requantization into the
//!   pointwise channel reduction and can repeat it once per output channel
//!   (`nngen/tools/mlir_optimizer/README.md:58-73`).
//! * Both steps are exposed separately so the CLI can stop after emission, and
//!   [`optimize`] consumes a saved [`MlirArtifact`] rather than an in-memory
//!   graph, so `tessera optimize` never re-imports or re-emits.
//!
//! Compile-time optionality: the whole Melior dependency sits behind the
//! `melior` feature. Without it the crate still emits, persists and validates
//! artifacts, and [`optimize`] returns [`MlirError::MeliorUnavailable`].
//!
//! Required dependency (added by integration, not by this module):
//!
//! ```toml
//! [dependencies]
//! melior = { version = "=0.28.2", optional = true }
//!
//! [features]
//! melior = ["dep:melior"]
//! ```
//!
//! Melior 0.28.2 pulls `mlir-sys` 230.0.2, which requires an LLVM/MLIR 23.x
//! installation discoverable through `MLIR_SYS_230_PREFIX` (and
//! `TABLEGEN_230_PREFIX` when ODS-generated dialect bindings are ever enabled).

pub mod analysis;
mod artifact;
mod emit;
#[cfg(test)]
mod fixtures;
mod pipeline;
pub mod quant;

pub use analysis::{
    ANALYSIS_FORMAT, ANALYSIS_SCHEMA_VERSION, AnalysisCounts, AnalyzedBuffer, LoopNest,
    ModuleAnalysis, OpTag, TAG_PREFIX, analysis_verification_available, analyze,
};

pub use artifact::{
    ArgumentRole, ArtifactMetrics, BiasContribution, CodeRange, DEFAULT_FUNCTION_NAME, ElementType,
    FilterStorageLayout, LayerBinding, LayerKind, MLIR_ARTIFACT_FORMAT,
    MLIR_ARTIFACT_SCHEMA_VERSION, MeliorProvenance, MlirArgument, MlirArtifact, MlirError,
    MlirFilterLayout, MlirOptions, MlirProvenance, MlirStage, OperatorBinding, OperatorKind,
    PaddingSpec, PipelineStage, ProvenanceCheck, Quantization, StructuralMetrics, TensorBinding,
    WeightPacking, WeightReference, content_hash, memref_type, tensor_type,
};
pub use emit::{
    EmittedModule, TAG_ADD, TAG_CONV, TAG_EPILOGUE, TAG_PAD, TAG_UPSAMPLE, emit_module, emit_text,
};
pub use pipeline::{
    OptimizationOutcome, PINNED_MELIOR_VERSION, default_pipeline, melior_available,
    render_pipeline, run_pipeline,
};
pub use quant::{
    ClampKind, ClampSpec, MAX_MULTIPLIER, MAX_SHIFT, QuantError, RequantSpec, RoundingMode,
    round_half_away, serialize_requant, serialize_scalar_requant,
};

use crate::graph::Graph;

/// Identity of the graph artifact an MLIR artifact descends from.
///
/// Provenance is data, not a lookup: the caller passes the values recorded in
/// the persisted graph artifact (`tessera.graph`), so this layer never guesses a
/// hash or re-reads an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceGraph<'a> {
    /// Graph artifact format, e.g. `tessera.graph`.
    pub format: &'a str,
    /// Graph artifact schema version.
    pub schema_version: u32,
    /// Hash identifying the exact graph artifact contents.
    pub content_hash: &'a str,
}

impl<'a> SourceGraph<'a> {
    /// Provenance reference for a graph artifact.
    pub const fn new(format: &'a str, schema_version: u32, content_hash: &'a str) -> Self {
        SourceGraph {
            format,
            schema_version,
            content_hash,
        }
    }
}

impl<'a> From<&'a crate::graph::GraphArtifact> for SourceGraph<'a> {
    /// Provenance of a persisted graph artifact.
    ///
    /// The manifest SHA-256 recorded by the frontend pins the graph's contents,
    /// so it is the identity used for the source-content hash; the artifact's own
    /// format and schema version are carried verbatim.
    fn from(artifact: &'a crate::graph::GraphArtifact) -> Self {
        SourceGraph {
            format: &artifact.format,
            schema_version: artifact.schema_version,
            content_hash: &artifact.provenance.manifest_sha256,
        }
    }
}

/// Emit and persist the tensor-level MLIR artifact for `graph`.
pub fn lower(
    graph: &Graph,
    source: SourceGraph<'_>,
    options: &MlirOptions,
) -> Result<MlirArtifact, MlirError> {
    let emitted = emit::emit_module(graph, options)?;
    let module_hash = content_hash(&emitted.module);
    Ok(MlirArtifact {
        format: MLIR_ARTIFACT_FORMAT.to_string(),
        schema_version: MLIR_ARTIFACT_SCHEMA_VERSION,
        function: options.function_name.clone(),
        stage: MlirStage::Tensor,
        module: emitted.module,
        input_shape: emitted.input_shape,
        arguments: emitted.arguments,
        tensors: emitted.tensors,
        layers: emitted.layers,
        operators: emitted.operators,
        output: emitted.output,
        metrics: emitted.metrics,
        analysis: None,
        provenance: MlirProvenance {
            source_format: source.format.to_string(),
            source_schema_version: source.schema_version,
            source_content_hash: source.content_hash.to_string(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            melior: None,
            module_hash,
        },
    })
}

/// Emit and then optimize in one call.
pub fn lower_optimized(
    graph: &Graph,
    source: SourceGraph<'_>,
    options: &MlirOptions,
) -> Result<MlirArtifact, MlirError> {
    let artifact = lower(graph, source, options)?;
    optimize(&artifact)
}

/// Run the pinned pipeline over an emitted artifact.
///
/// The input artifact's module text is parsed and verified first, then the pinned
/// pipeline runs, then the planning-stage artifact is rebuilt with Melior
/// provenance and structural metrics of the *new* text. The graph-level
/// semantics, argument list and operator bindings are carried over unchanged,
/// because the pipeline may only change representation, never meaning.
pub fn optimize(artifact: &MlirArtifact) -> Result<MlirArtifact, MlirError> {
    artifact.validate_header()?;
    if artifact.stage != MlirStage::Tensor {
        return Err(MlirError::Pipeline {
            detail: format!(
                "artifact is already at stage {}; optimize consumes a tensor-stage module",
                artifact.stage
            ),
        });
    }
    let stages = pipeline::default_pipeline();
    let outcome = pipeline::run_pipeline(&artifact.module, &stages)?;
    // The analysis is the optimizer's real output: the planning stage is driven
    // from the loop and buffer structure of the text that comes out of the pass
    // pipeline, and the artifact carries that structure so a later stage (or a
    // reader) does not have to re-derive it or re-run MLIR.
    let analysis = analysis::analyze(&outcome.module, PINNED_MELIOR_VERSION, &stages)?;
    let structural = analysis.structural();
    let mut optimized = artifact.clone();
    optimized.stage = MlirStage::Planning;
    optimized.provenance.melior = Some(MeliorProvenance {
        melior_version: PINNED_MELIOR_VERSION.to_string(),
        pipeline: stages,
    });
    optimized.provenance.module_hash = content_hash(&outcome.module);
    optimized.metrics.structural = Some(structural);
    optimized.module = outcome.module;
    optimized.analysis = Some(analysis);
    Ok(optimized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::artifact::{ArtifactMetrics, MlirProvenance};

    fn tensor_stage_artifact(module: &str) -> MlirArtifact {
        MlirArtifact {
            format: MLIR_ARTIFACT_FORMAT.to_string(),
            schema_version: MLIR_ARTIFACT_SCHEMA_VERSION,
            function: DEFAULT_FUNCTION_NAME.to_string(),
            stage: MlirStage::Tensor,
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
                code_range: Some(CodeRange::new(-127, 127)),
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
                module_hash: content_hash(module),
            },
        }
    }

    const TRIVIAL_MODULE: &str = "module {\n  func.func @f() {\n    return\n  }\n}\n";

    #[test]
    fn optimization_is_gated_and_reports_absence_truthfully() {
        let artifact = tensor_stage_artifact(TRIVIAL_MODULE);
        let result = optimize(&artifact);
        if melior_available() {
            assert!(result.is_ok(), "{result:?}");
            let optimized = result.unwrap();
            assert_eq!(optimized.stage, MlirStage::Planning);
            assert!(optimized.metrics.structural.is_some());
            assert!(optimized.provenance.melior.is_some());
            assert_eq!(
                optimized.provenance.module_hash,
                content_hash(&optimized.module)
            );
        } else {
            match result {
                Err(MlirError::MeliorUnavailable { detail }) => {
                    assert!(
                        detail.contains("#[cfg(feature = \"melior\")]")
                            || detail.contains("melior 0.28.2")
                    );
                }
                other => panic!("expected MeliorUnavailable, got {other:?}"),
            }
        }
    }

    #[test]
    fn lowering_persists_and_roundtrips_a_graph_artifact() {
        let graph = crate::mlir::fixtures::sample_graph();
        let options = crate::mlir::fixtures::sample_options();
        let source = SourceGraph::new(
            crate::graph::GRAPH_FORMAT,
            crate::graph::GRAPH_SCHEMA_VERSION,
            "manifest-sha256",
        );
        let artifact = super::lower(&graph, source, &options).expect("lower");
        assert_eq!(artifact.format, MLIR_ARTIFACT_FORMAT);
        assert_eq!(artifact.stage, MlirStage::Tensor);
        assert_eq!(artifact.function, DEFAULT_FUNCTION_NAME);
        assert_eq!(artifact.layers.len(), 3);
        assert!(artifact.metrics.structural.is_none());
        assert_eq!(artifact.provenance.melior, None);
        assert_eq!(
            artifact.provenance.module_hash,
            content_hash(&artifact.module)
        );

        let json = artifact.to_json().expect("serialize");
        let parsed = MlirArtifact::from_json(&json).expect("parse");
        assert_eq!(parsed, artifact);

        // The persisted module is exactly the emitted text, so `tessera optimize`
        // never needs the graph again.
        let direct = crate::mlir::emit_text(&graph, &options).expect("emit");
        assert_eq!(artifact.module, direct);
    }

    #[test]
    fn source_graph_can_be_taken_from_a_graph_artifact() {
        let artifact = crate::graph::GraphArtifact::new(
            crate::mlir::fixtures::sample_graph(),
            crate::graph::Provenance {
                source_format: crate::graph::MANIFEST_FORMAT.to_string(),
                importer_version: "tessera-frontend/0.1.0".to_string(),
                manifest_sha256: "abc123".to_string(),
                weights_sha256: "def456".to_string(),
                weight_file: "weights.bin".to_string(),
                weight_bytes: 132,
            },
        );
        match artifact {
            Ok(artifact) => {
                let source = SourceGraph::from(&artifact);
                assert_eq!(source.format, crate::graph::GRAPH_FORMAT);
                assert_eq!(source.schema_version, crate::graph::GRAPH_SCHEMA_VERSION);
                assert_eq!(source.content_hash, "abc123");
            }
            Err(error) => {
                // The fixture is built from public fields and is expected to
                // validate; if it does not, the fixture must be fixed rather than
                // the conversion bypassed.
                panic!("fixture graph must validate: {error}");
            }
        }
    }

    /// End-to-end: a real manifest string imported by the frontend lowers to MLIR.
    const HEAD_ONLY_MANIFEST: &str = r#"{
  "format": "fpga_st_train.int8.v1",
  "arch": "tiny",
  "bits": { "acts": 8, "weights": 8 },
  "act_qmax": 127.0,
  "output_qmax": 255.0,
  "weight_qmax": 127.0,
  "points": ["image"],
  "point_scales": [0.007874016],
  "graph": [
    { "op": "conv", "layer": "head", "point": 0, "scale": 0.007874016 },
    { "op": "quantize_u8_output", "scale": 0.003921569 }
  ],
  "layers": [
    {
      "name": "head",
      "op": "conv2d",
      "in_channels": 3,
      "out_channels": 2,
      "groups": 1,
      "act_bits": 8,
      "weight_bits": 8,
      "act_zero_point": 0,
      "out_relu": false,
      "out_unsigned": true,
      "kernel": [1, 1],
      "stride": [1, 1],
      "reflect_pad": 0,
      "in_point": 0,
      "out_point": 1,
      "in_scale": 0.007874016,
      "out_scale": 0.003921569,
      "weight_scale": [0.001, 0.002],
      "bias": [0.0, 0.0],
      "bias_int32": [1, -2],
      "requant": [0.01, 0.02],
      "weight_offset": 0,
      "weight_count": 6,
      "weight_byte_offset": 0,
      "weight_bytes": 6
    }
  ],
  "weight_file": "weights.bin",
  "weight_packing": "i8-x1",
  "weight_bytes": 6
}"#;

    #[test]
    fn imported_manifest_lowers_to_mlir_end_to_end() {
        let weights = vec![0u8; 6];
        let artifact =
            crate::frontend::import_manifest_str(HEAD_ONLY_MANIFEST, &weights, "fixture.json")
                .expect("import");
        let options = MlirOptions::new([1, 4, 4, 3]);
        let mlir = lower(artifact.graph(), SourceGraph::from(&artifact), &options).expect("lower");

        assert_eq!(mlir.layers.len(), 1);
        assert_eq!(mlir.layers[0].layer, "head");
        assert!(mlir.layers[0].quantization.clamp.unsigned);
        assert_eq!(mlir.layers[0].quantization.clamp.upper, 255);
        assert_eq!(mlir.layers[0].stride, [1, 1]);
        assert_eq!(mlir.layers[0].filter.mlir_layout, MlirFilterLayout::Fhwc);
        assert_eq!(mlir.output.shape, vec![1, 4, 4, 2]);
        assert_eq!(mlir.output.element, ElementType::I8);
        assert!(mlir.module.contains("linalg.conv_2d_nhwc_fhwc"));
        assert!(mlir.module.contains("%upper = arith.constant 255 : i64"));
        assert_eq!(mlir.metrics.operators, 2);
        assert_eq!(mlir.provenance.source_format, crate::graph::GRAPH_FORMAT);
        assert_eq!(
            mlir.provenance.source_schema_version,
            crate::graph::GRAPH_SCHEMA_VERSION
        );
        // The manifest identity is carried through as the source content hash.
        assert_eq!(mlir.provenance.source_content_hash.len(), 64);
        assert_eq!(
            mlir.provenance.source_content_hash,
            crate::frontend::sha256_hex(HEAD_ONLY_MANIFEST.as_bytes())
        );
    }

    #[test]
    fn optimization_refuses_a_non_tensor_stage() {
        let mut artifact = tensor_stage_artifact(TRIVIAL_MODULE);
        artifact.stage = MlirStage::Planning;
        assert!(matches!(
            optimize(&artifact),
            Err(MlirError::Pipeline { .. })
        ));
    }
}
