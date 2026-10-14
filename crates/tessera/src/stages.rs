//! Persistent stage directory: file names, deterministic IO and provenance
//! validation across the whole pipeline.
//!
//! Every stage writes one artifact into a *stage directory*, and later stages
//! read the earlier artifacts instead of re-running earlier work
//! (ADR-0002). This module owns the names and the cross-stage consistency
//! checks so the CLI and the tests share one definition:
//!
//! ```text
//! <dir>/graph.tgraph        GraphArtifact          (JSON)
//! <dir>/mlir.json           MlirArtifact, tensor   (JSON)
//! <dir>/module.mlir         emitted MLIR text
//! <dir>/optimized.json      MlirArtifact, planning (JSON, requires `melior`)
//! <dir>/optimized.mlir      optimized MLIR text
//! <dir>/target.toml         copy of the target the layout was solved against
//! <dir>/policy.toml         copy of the policy the layout was solved against
//! <dir>/primitive.json      lowered PlanInput            (JSON)
//! <dir>/layout.tlayout      LayoutArtifact               (TOML)
//! <dir>/plan.tplan          ExecutionPlan                (TOML)
//! <dir>/rtl/                RTL bundle
//! ```
//!
//! Later stages copy their *inputs* into the directory so that the provenance
//! hashes recorded by the layout can be re-checked later without the original
//! files. [`verify_chain`] is that re-check.

use crate::frontend::{load_artifact, sha256_hex};
use crate::graph::GraphArtifact;
use crate::layout::{ExecutionPlan, LayoutArtifact};
use crate::mlir::MlirArtifact;
use crate::planner::PlanInput;
use crate::primitive::PrimitiveArtifact;
use crate::rtl::RtlMetadata;
use std::fmt;
use std::path::{Path, PathBuf};

/// Persisted graph artifact.
pub const GRAPH_FILE: &str = "graph.tgraph";
/// Emitted tensor-level MLIR artifact.
pub const MLIR_FILE: &str = "mlir.json";
/// Emitted tensor-level MLIR module text.
pub const MODULE_FILE: &str = "module.mlir";
/// Optimized planning-stage MLIR artifact.
pub const OPTIMIZED_FILE: &str = "optimized.json";
/// Optimized MLIR module text.
pub const OPTIMIZED_MODULE_FILE: &str = "optimized.mlir";
/// Copy of the target file a layout was solved against.
pub const TARGET_FILE: &str = "target.toml";
/// Copy of the policy file a layout was solved against.
pub const POLICY_FILE: &str = "policy.toml";
/// Lowered planner input (`PlanInput`, the planner's own contract).
pub const PRIMITIVE_FILE: &str = "primitive.json";
/// Lowered primitive graph with its evidence (`tessera.primitive`).
pub const PRIMITIVES_FILE: &str = "primitives.json";
/// Concrete input activation shape recorded during `import` as `N,H,W,C`.
pub const INPUT_SHAPE_FILE: &str = "input_shape.txt";
/// Solved layout artifact.
pub const LAYOUT_FILE: &str = "layout.tlayout";
/// Backend-ready execution plan projected from the layout.
pub const PLAN_FILE: &str = "plan.tplan";
/// Directory holding the emitted RTL bundle.
pub const RTL_DIR: &str = "rtl";
/// RTL bundle metadata file inside [`RTL_DIR`].
pub const RTL_METADATA_FILE: &str = "rtl_bundle.json";
/// Simulation working directory.
pub const SIM_DIR: &str = "sim";

/// A compiler stage directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDir {
    root: PathBuf,
}

impl StageDir {
    /// Wrap a directory path (it is not created until something is written).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        StageDir { root: root.into() }
    }

    /// Root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path of a named entry inside the directory.
    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }

    /// Whether a named entry exists.
    pub fn exists(&self, name: &str) -> bool {
        self.path(name).exists()
    }

    /// Create the directory (and parents).
    pub fn ensure(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.root)
    }

    /// Write deterministic text, creating the directory and a trailing newline.
    pub fn write_text(&self, name: &str, text: &str) -> std::io::Result<PathBuf> {
        self.ensure()?;
        let path = self.path(name);
        let mut body = text.to_string();
        if !body.ends_with('\n') {
            body.push('\n');
        }
        std::fs::write(&path, body)?;
        Ok(path)
    }

    /// Read a named entry as UTF-8 text.
    pub fn read_text(&self, name: &str) -> std::io::Result<String> {
        std::fs::read_to_string(self.path(name))
    }

    /// Copy an external file into the directory under `name`.
    pub fn copy_in(&self, source: impl AsRef<Path>, name: &str) -> std::io::Result<PathBuf> {
        let source = source.as_ref();
        let destination = self.path(name);
        if source == destination {
            return Ok(destination);
        }
        self.ensure()?;
        std::fs::copy(source, &destination)?;
        Ok(destination)
    }

    /// SHA-256 of a named entry, for provenance recording.
    pub fn sha256(&self, name: &str) -> Result<String, ChainError> {
        sha256_of(self.path(name))
    }
}

/// SHA-256 of an arbitrary file, for provenance recording.
pub fn sha256_of(path: impl AsRef<Path>) -> Result<String, ChainError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| ChainError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    Ok(sha256_hex(&bytes))
}

/// Kind of a persisted artifact, detected from its `format` tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// `tessera.graph`
    Graph,
    /// `tessera.mlir`
    Mlir,
    /// `tessera.layout`
    Layout,
    /// `tessera.execution_plan`
    Plan,
    /// `tessera.rtl_bundle`
    RtlBundle,
    /// A lowered primitive graph (`PlanInput`), which has no format tag.
    Primitive,
    /// A lowered primitive graph with its evidence (`tessera.primitive`).
    PrimitiveArtifact,
}

impl ArtifactKind {
    /// Stable lowercase name.
    pub const fn as_str(self) -> &'static str {
        match self {
            ArtifactKind::Graph => "graph",
            ArtifactKind::Mlir => "mlir",
            ArtifactKind::Layout => "layout",
            ArtifactKind::Plan => "execution_plan",
            ArtifactKind::RtlBundle => "rtl_bundle",
            ArtifactKind::Primitive => "primitive",
            ArtifactKind::PrimitiveArtifact => "primitive_artifact",
        }
    }
}

/// Detect an artifact kind from its text contents.
///
/// JSON artifacts are read from their `format` field; TOML artifacts from their
/// `format` key. A `PlanInput` JSON document has no format tag, so it is
/// recognised by its required `tensors`/`ops`/`inputs`/`outputs` fields.
pub fn detect_kind(text: &str) -> Option<ArtifactKind> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') {
        let value: serde_json::Value = serde_json::from_str(text).ok()?;
        if let Some(format) = value.get("format").and_then(|v| v.as_str()) {
            return match format {
                crate::graph::GRAPH_FORMAT => Some(ArtifactKind::Graph),
                crate::mlir::MLIR_ARTIFACT_FORMAT => Some(ArtifactKind::Mlir),
                crate::rtl::RTL_BUNDLE_FORMAT => Some(ArtifactKind::RtlBundle),
                crate::primitive::PRIMITIVE_FORMAT => Some(ArtifactKind::PrimitiveArtifact),
                _ => None,
            };
        }
        let has = |key: &str| value.get(key).is_some();
        if has("tensors") && has("ops") && has("inputs") && has("outputs") {
            return Some(ArtifactKind::Primitive);
        }
        return None;
    }
    let parsed = crate::target::toml_lite::parse_toml(text).ok()?;
    let format = parsed.get("format").and_then(|v| v.as_str())?;
    match format {
        crate::layout::LAYOUT_FORMAT => Some(ArtifactKind::Layout),
        crate::layout::PLAN_FORMAT => Some(ArtifactKind::Plan),
        crate::rtl::RTL_BUNDLE_FORMAT => Some(ArtifactKind::RtlBundle),
        _ => None,
    }
}

/// One checked link in the provenance chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainCheck {
    /// Stage name.
    pub stage: &'static str,
    /// What was checked.
    pub check: String,
    /// Whether it held.
    pub ok: bool,
    /// Detail, present when the check failed or when it is informational.
    pub detail: String,
}

/// A chain failure: an artifact is unreadable or a provenance link is broken.
#[derive(Debug)]
pub enum ChainError {
    /// Filesystem failure.
    Io { path: String, detail: String },
    /// A required stage is absent.
    Missing { stage: &'static str, path: String },
    /// An artifact is present but malformed.
    Malformed { stage: &'static str, detail: String },
    /// Provenance does not link two stages.
    Provenance { detail: String },
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainError::Io { path, detail } => write!(f, "io: {path}: {detail}"),
            ChainError::Missing { stage, path } => {
                write!(f, "missing {stage} artifact: no file at {path}")
            }
            ChainError::Malformed { stage, detail } => {
                write!(f, "malformed {stage} artifact: {detail}")
            }
            ChainError::Provenance { detail } => write!(f, "provenance mismatch: {detail}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Load and validate every graph artifact in `dir` (if present).
pub fn load_graph(dir: &StageDir) -> Result<Option<GraphArtifact>, ChainError> {
    let path = dir.path(GRAPH_FILE);
    if !path.exists() {
        return Ok(None);
    }
    load_artifact(&path)
        .map(Some)
        .map_err(|error| ChainError::Malformed {
            stage: GRAPH_FILE,
            detail: error.to_string(),
        })
}

/// Load and validate the MLIR artifact in `dir` (tensor or planning stage).
pub fn load_mlir(dir: &StageDir, name: &str) -> Result<Option<MlirArtifact>, ChainError> {
    let path = dir.path(name);
    if !path.exists() {
        return Ok(None);
    }
    MlirArtifact::load(&path)
        .map(Some)
        .map_err(|error| ChainError::Malformed {
            stage: "mlir",
            detail: error.to_string(),
        })
}

/// Load and validate the layout artifact in `dir`.
pub fn load_layout(dir: &StageDir) -> Result<Option<LayoutArtifact>, ChainError> {
    let path = dir.path(LAYOUT_FILE);
    if !path.exists() {
        return Ok(None);
    }
    LayoutArtifact::load(&path)
        .map(Some)
        .map_err(|error| ChainError::Malformed {
            stage: LAYOUT_FILE,
            detail: error.to_string(),
        })
}

/// Load and validate the execution plan in `dir`.
pub fn load_plan(dir: &StageDir) -> Result<Option<ExecutionPlan>, ChainError> {
    let path = dir.path(PLAN_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|error| ChainError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    ExecutionPlan::from_toml_str(&text)
        .map(Some)
        .map_err(|error| ChainError::Malformed {
            stage: PLAN_FILE,
            detail: error.to_string(),
        })
}

/// Load the RTL bundle metadata in `dir`, when a bundle was emitted.
pub fn load_rtl_metadata(dir: &StageDir) -> Result<Option<RtlMetadata>, ChainError> {
    let path = dir.path(RTL_DIR).join(RTL_METADATA_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|error| ChainError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| ChainError::Malformed {
            stage: RTL_METADATA_FILE,
            detail: error.to_string(),
        })
}

/// Load a lowered primitive graph written by `tessera solve`.
///
/// Both forms are accepted: the plain planner contract (`primitive.json`) and the
/// evidence-carrying `tessera.primitive` artifact (`primitives.json`), whose
/// `plan` is what the planner consumes.
pub fn load_primitive(dir: &StageDir) -> Result<Option<PlanInput>, ChainError> {
    for name in [PRIMITIVE_FILE, PRIMITIVES_FILE] {
        let path = dir.path(name);
        if !path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&path).map_err(|error| ChainError::Io {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        return parse_primitive_stage(&text, name).map(Some);
    }
    Ok(None)
}

/// Load the evidence-carrying primitive artifact, when one was written.
pub fn load_primitive_artifact(dir: &StageDir) -> Result<Option<PrimitiveArtifact>, ChainError> {
    let path = dir.path(PRIMITIVES_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).map_err(|error| ChainError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let artifact = PrimitiveArtifact::from_json(&text).map_err(|error| ChainError::Malformed {
        stage: PRIMITIVES_FILE,
        detail: error.to_string(),
    })?;
    artifact
        .validate_header()
        .map_err(|error| ChainError::Malformed {
            stage: PRIMITIVES_FILE,
            detail: error.to_string(),
        })?;
    Ok(Some(artifact))
}

/// Parse a primitive document (either form) and validate it.
pub fn parse_primitive(text: &str) -> Result<PlanInput, ChainError> {
    parse_primitive_stage(text, PRIMITIVE_FILE)
}

fn parse_primitive_stage(text: &str, stage: &'static str) -> Result<PlanInput, ChainError> {
    if detect_kind(text) == Some(ArtifactKind::PrimitiveArtifact) {
        let artifact =
            PrimitiveArtifact::from_json(text).map_err(|error| ChainError::Malformed {
                stage,
                detail: error.to_string(),
            })?;
        return Ok(artifact.plan);
    }
    let plan: PlanInput = serde_json::from_str(text).map_err(|error| ChainError::Malformed {
        stage,
        detail: error.to_string(),
    })?;
    plan.validate()
        .map_err(|detail| ChainError::Malformed { stage, detail })?;
    Ok(plan)
}

/// Re-check every persisted artifact in `dir` and every provenance link
/// between them.
///
/// The checks are the ones the pipeline promises:
/// * each present artifact re-validates against its own schema;
/// * `mlir.json` descends from `graph.tgraph` (same source content hash);
/// * `optimized.json` descends from the same graph and records Melior
///   provenance;
/// * `layout.tlayout` names the graph, target and policy hashes of the copies
///   in the directory;
/// * `plan.tplan` carries exactly the layout's provenance;
/// * `rtl/rtl_bundle.json` carries the plan's provenance and its per-file
///   digests match the files on disk.
///
/// Absent stages are not failures — a user may stop after any stage. At least
/// one artifact must be present.
pub fn verify_chain(dir: &StageDir) -> Result<Vec<ChainCheck>, ChainError> {
    let mut checks = Vec::new();
    let graph = load_graph(dir)?;
    let mlir = load_mlir(dir, MLIR_FILE)?;
    let optimized = load_mlir(dir, OPTIMIZED_FILE)?;
    let layout = load_layout(dir)?;
    let plan = load_plan(dir)?;
    let primitives = load_primitive_artifact(dir)?;
    let rtl = load_rtl_metadata(dir)?;

    if let Some(graph) = &graph {
        checks.push(ChainCheck {
            stage: "graph",
            check: format!("{} schema {}", graph.format, graph.schema_version),
            ok: true,
            detail: format!(
                "{} layers, {} ops, {} points, {} weight bytes",
                graph.metrics().layer_count,
                graph.metrics().graph_op_count,
                graph.metrics().point_count,
                graph.provenance().weight_bytes
            ),
        });
    }

    // The MLIR artifacts own their integrity checks: the module text is
    // re-hashed against the recorded hash, the recorded source graph is compared
    // with the staged one, and the embedded analysis is re-derived and compared
    // field by field (which needs the `melior` feature; a build without it reports
    // that check as failed rather than trusting the metadata). They are run here
    // rather than re-implemented, so `verify` and the artifact agree on what
    // integrity means and a gap is reported once.
    for artifact in [&mlir, &optimized].into_iter().flatten() {
        match artifact.verify(graph.as_ref()) {
            Ok(provenance) => {
                for check in provenance {
                    checks.push(ChainCheck {
                        stage: "mlir",
                        check: check.check,
                        ok: check.ok,
                        detail: check.detail,
                    });
                }
            }
            Err(error) => checks.push(ChainCheck {
                stage: "mlir",
                check: "artifact integrity".to_string(),
                ok: false,
                detail: error.to_string(),
            }),
        }
    }
    if let (Some(mlir), Some(optimized)) = (&mlir, &optimized) {
        let ok = optimized.provenance.melior.is_some()
            && optimized.provenance.source_content_hash == mlir.provenance.source_content_hash;
        checks.push(ChainCheck {
            stage: "optimized",
            check: "Melior pipeline provenance".to_string(),
            ok,
            detail: match &optimized.provenance.melior {
                Some(melior) => format!(
                    "melior {} ran {} passes, module hash {}",
                    melior.melior_version,
                    melior.pipeline.len(),
                    short(&optimized.provenance.module_hash)
                ),
                None => "optimized artifact records no Melior provenance".to_string(),
            },
        });
    }

    if let Some(layout) = &layout {
        // The layout records the hash of the artifact it was solved from: the
        // graph artifact, the optimized MLIR artifact, or a primitive graph. A
        // stage directory may hold several of them (a solve can follow an
        // import), so the hash decides which one the layout descends from
        // instead of a fixed preference order.
        let candidates = [
            GRAPH_FILE,
            OPTIMIZED_FILE,
            MLIR_FILE,
            PRIMITIVE_FILE,
            PRIMITIVES_FILE,
        ];
        let mut matched: Option<&'static str> = None;
        let mut present: Vec<(&'static str, String)> = Vec::new();
        for file in candidates {
            if !dir.exists(file) {
                continue;
            }
            let actual = dir.sha256(file)?;
            if actual == layout.provenance.graph_sha256 {
                matched = Some(file);
                break;
            }
            present.push((file, actual));
        }
        match matched {
            Some(file) => checks.push(ChainCheck {
                stage: "layout",
                check: "solved-from input hash".to_string(),
                ok: true,
                detail: format!("{file} matches {}", short(&layout.provenance.graph_sha256)),
            }),
            None => checks.push(ChainCheck {
                stage: "layout",
                check: "solved-from input hash".to_string(),
                ok: false,
                detail: if present.is_empty() {
                    format!(
                        "none of {} is in the directory, so the layout's input cannot be re-checked",
                        candidates.join(", ")
                    )
                } else {
                    format!(
                        "the layout records {}; {}",
                        short(&layout.provenance.graph_sha256),
                        present
                            .iter()
                            .map(|(file, hash)| format!("{file} hashes to {}", short(hash)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
            }),
        }
        for (file, recorded, what) in [
            (TARGET_FILE, &layout.provenance.target_sha256, "target"),
            (POLICY_FILE, &layout.provenance.policy_sha256, "policy"),
        ] {
            if !dir.exists(file) {
                checks.push(ChainCheck {
                    stage: "layout",
                    check: format!("{what} input copy present"),
                    ok: false,
                    detail: format!("{file} is not in the stage directory"),
                });
                continue;
            }
            let actual = dir.sha256(file)?;
            let ok = &actual == recorded;
            checks.push(ChainCheck {
                stage: "layout",
                check: format!("{what} hash"),
                ok,
                detail: if ok {
                    format!("{} matches {}", file, short(recorded))
                } else {
                    format!(
                        "{file} hashes to {} but the layout records {}",
                        short(&actual),
                        short(recorded)
                    )
                },
            });
        }
    }

    if let (Some(primitives), Some(optimized)) = (&primitives, &optimized) {
        let linked = primitives.check(optimized);
        checks.push(ChainCheck {
            stage: "primitives",
            check: "descends from the optimized module".to_string(),
            ok: linked.is_ok(),
            detail: match linked {
                Ok(()) => format!(
                    "{} operators, {} fusion groups ({} fused), {} buffers, {} intervals, module \
                     {}",
                    primitives.plan.ops.len(),
                    primitives.groups.len(),
                    primitives.fused_groups().len(),
                    primitives.buffers.len(),
                    primitives.intervals.len(),
                    short(&primitives.provenance.mlir_module_hash)
                ),
                Err(error) => error.to_string(),
            },
        });
    } else if let Some(primitives) = &primitives {
        // Standalone: the optimized module is not in this directory, so only the
        // artifact's own consistency can be checked, and that is stated.
        let own = primitives.validate_header();
        checks.push(ChainCheck {
            stage: "primitives",
            check: "artifact self-consistency".to_string(),
            ok: own.is_ok(),
            detail: match own {
                Ok(()) => format!(
                    "no optimized module in the directory; checked format, schema and plan only \
                     (module {} is trusted from the producer)",
                    short(&primitives.provenance.mlir_module_hash)
                ),
                Err(error) => error.to_string(),
            },
        });
    }

    // The numerical contract the whole chain is built on: every requantization in
    // the plan uses the documented half-away-from-zero fixed-point datapath, and the bound on
    // how far the serialized M0/shift is from the manifest's f32 coefficients is
    // recorded rather than assumed. The exactness baseline is the compiled
    // fixed-point IR, never the training f32.
    let mut rounds: Vec<(&str, crate::layout::Rounding, Option<u32>)> = Vec::new();
    if let Some(plan) = &plan {
        for step in &plan.steps {
            if let Some(requant) = &step.requant {
                rounds.push((
                    step.op.as_str(),
                    requant.rounding,
                    requant.multiplier_error_ppm,
                ));
            }
        }
    } else if let Some(layout) = &layout {
        for step in &layout.schedule {
            if let Some(requant) = &step.requant {
                rounds.push((
                    step.op.as_str(),
                    requant.rounding,
                    requant.multiplier_error_ppm,
                ));
            }
        }
    }
    if !rounds.is_empty() {
        let off_contract: Vec<&str> = rounds
            .iter()
            .filter(|(_, rounding, _)| *rounding != crate::layout::Rounding::HalfAwayFromZero)
            .map(|(op, _, _)| *op)
            .collect();
        let bounded = rounds.iter().filter(|(_, _, ppm)| ppm.is_some()).count();
        let worst = rounds.iter().filter_map(|(_, _, ppm)| *ppm).max();
        checks.push(ChainCheck {
            stage: "layout",
            check: "semantic contract".to_string(),
            ok: off_contract.is_empty(),
            detail: if off_contract.is_empty() {
                format!(
                    "half-away-from-zero fixed-point requantization on {} step(s); {bounded} \
                     carry an \
                     approximation bound, worst {} ppm against the manifest f32 (the exactness \
                     baseline is the compiled fixed-point IR)",
                    rounds.len(),
                    match worst {
                        Some(ppm) => ppm.to_string(),
                        None => "no".to_string(),
                    }
                )
            } else {
                format!(
                    "steps {:?} do not use the documented half-away-from-zero rounding, so the \
                     plan is not \
                     the datapath this compiler claims to emit",
                    off_contract
                )
            },
        });
    }

    if let (Some(layout), Some(plan)) = (&layout, &plan) {
        let ok = plan.provenance == layout.provenance;
        checks.push(ChainCheck {
            stage: "plan",
            check: "inherits layout provenance".to_string(),
            ok,
            detail: format!(
                "{} steps, {} weight tensors, {} buffers",
                plan.steps.len(),
                plan.weights.len(),
                plan.buffers.len()
            ),
        });
    }

    if let (Some(plan), Some(rtl)) = (&plan, &rtl) {
        let provenance_ok = rtl.provenance.target == plan.provenance.target_name
            && rtl.provenance.input_manifest_sha256 == plan.provenance.graph_sha256
            && rtl.provenance.planner_config_sha256 == plan.provenance.policy_sha256;
        checks.push(ChainCheck {
            stage: "rtl",
            check: "carries plan provenance".to_string(),
            ok: provenance_ok,
            detail: format!(
                "{} files, kernel {}, bundle digest {}",
                rtl.files.len(),
                rtl.kernel_id,
                short(&rtl.bundle_digest)
            ),
        });
        let dir_rtl = dir.path(RTL_DIR);
        for file in &rtl.files {
            let path = dir_rtl.join(&file.name);
            if !path.exists() {
                checks.push(ChainCheck {
                    stage: "rtl",
                    check: format!("{} present", file.name),
                    ok: false,
                    detail: "recorded in metadata but missing on disk".to_string(),
                });
                continue;
            }
            let text = std::fs::read_to_string(&path).map_err(|error| ChainError::Io {
                path: path.display().to_string(),
                detail: error.to_string(),
            })?;
            let digest = format!("{:016x}", crate::rtl::fnv1a64(text.as_bytes()));
            let ok = digest == file.fnv1a64 && text.len() == file.bytes;
            checks.push(ChainCheck {
                stage: "rtl",
                check: format!("{} digest", file.name),
                ok,
                detail: if ok {
                    format!("{} bytes, fnv1a64 {digest}", file.bytes)
                } else {
                    format!(
                        "expected {} bytes / {}, found {} / {digest}",
                        file.bytes,
                        file.fnv1a64,
                        text.len()
                    )
                },
            });
        }
    }

    if let Some(rtl) = &rtl {
        if let Some(full) = &rtl.full_graph {
            // The allocator may reuse words, so "no two slots overlap" is the wrong
            // invariant; the right one is that slots whose *live ranges* overlap do
            // not overlap in address. This is checked from the persisted artifact
            // rather than trusted, because a sum-of-live-words allocation can hand
            // two simultaneously live slots the same words.
            let mut conflicts = Vec::new();
            for (index, a) in full.slots.iter().enumerate() {
                for b in full.slots.iter().skip(index + 1) {
                    let both_live = a.live[0] <= b.live[1] && b.live[0] <= a.live[1];
                    if !both_live {
                        continue;
                    }
                    let disjoint =
                        a.offset + a.elements <= b.offset || b.offset + b.elements <= a.offset;
                    if !disjoint {
                        conflicts.push(format!(
                            "{} [{}, {}) and {} [{}, {}) are both live over steps {}-{} / {}-{}",
                            a.name,
                            a.offset,
                            a.offset + a.elements,
                            b.name,
                            b.offset,
                            b.offset + b.elements,
                            a.live[0],
                            a.live[1],
                            b.live[0],
                            b.live[1]
                        ));
                    }
                }
            }
            checks.push(ChainCheck {
                stage: "rtl",
                check: "simultaneous address disjoint".to_string(),
                ok: conflicts.is_empty(),
                detail: if conflicts.is_empty() {
                    format!(
                        "{} arena slot(s) checked pairwise; no two simultaneously live slots share \
                         an address",
                        full.slots.len()
                    )
                } else {
                    format!(
                        "{} overlapping pair(s): {}",
                        conflicts.len(),
                        conflicts.join("; ")
                    )
                },
            });
            // The emitted high-water mark may be smaller than the plan's budget (it
            // may reuse words) but never larger, or the plan under-budgets the
            // structure that was built from it.
            if let Some(plan) = &plan {
                let budget = plan.engine.arena_words;
                let high_water = full.arena_words;
                checks.push(ChainCheck {
                    stage: "rtl",
                    check: "arena high-water within budget".to_string(),
                    ok: budget == 0 || high_water <= budget,
                    detail: if budget == 0 {
                        format!("emitted {high_water} word(s); the plan declares no budget")
                    } else {
                        format!(
                            "emitted {high_water} words against a plan budget of {budget} \
                             (without reuse: {})",
                            full.arena_words_without_reuse
                        )
                    },
                });
            }
        }
    }

    if checks.is_empty() {
        return Err(ChainError::Missing {
            stage: "any",
            path: dir.root().display().to_string(),
        });
    }
    if let Some(bad) = checks.iter().find(|check| !check.ok) {
        return Err(ChainError::Provenance {
            detail: format!("{}: {}", bad.check, bad.detail),
        });
    }
    Ok(checks)
}

fn short(hash: &str) -> String {
    if hash.len() <= 12 {
        hash.to_string()
    } else {
        format!("{}…", &hash[..12])
    }
}
