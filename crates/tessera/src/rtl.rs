//! Tessera RTL emission.
//!
//! The only emit path is the CIRCT hybrid kernel ([`circt_lower`]): solved
//! tessera decisions → CIRCT (`hw`/`comb`/`sv`) in the same Melior context →
//! in-process ExportVerilog. The DUT datapath is a single integer
//! `*_sched_core`. Solved `tessera.engine` / bind ops are mapping decisions
//! recorded in the dialect and reflected on `engines_busy` as a one-hot of the
//! engine that owns the core's current op — not as separate per-engine FSMs.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::layout::{ElemType as LayoutElemType, TensorRef};

pub mod circt_ctx;
pub mod circt_data;
pub mod circt_datapath;
pub mod circt_lower;

pub use circt_lower::{
    CIRCT_KERNEL_ID, CirctSummary, lower_solved_tessera, lower_solved_tessera_text,
    lower_solved_tessera_text_with_content, lower_solved_tessera_with_content,
};

/// Bundle format tag written into `rtl_bundle.json`.
pub const RTL_BUNDLE_FORMAT: &str = "tessera.rtl_bundle";
/// Schema version of the RTL bundle metadata document.
pub const RTL_BUNDLE_SCHEMA_VERSION: u32 = 1;
/// Alias of [`CIRCT_KERNEL_ID`] for callers that still name `RTL_KERNEL_ID`.
pub const RTL_KERNEL_ID: &str = CIRCT_KERNEL_ID;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RtlError {
    UnsupportedElementType {
        layer: String,
        tensor: String,
        element_type: ElementType,
    },
    UnsupportedQuantization {
        layer: String,
        detail: String,
    },
    WeightsNotResident {
        layer: String,
        tensor: String,
        detail: String,
    },
    UnsupportedLayout {
        detail: String,
    },
    ShapeMismatch {
        layer: String,
        what: String,
        expected: u64,
        actual: u64,
    },
    InvalidKernelParam {
        layer: String,
        field: &'static str,
        detail: String,
    },
    MissingProvenance {
        field: &'static str,
    },
    Serialize {
        context: &'static str,
        source: serde_json::Error,
    },
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl fmt::Display for RtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RtlError::UnsupportedElementType {
                layer,
                tensor,
                element_type,
            } => write!(
                f,
                "rtl: step {layer}: tensor {tensor} has element type {element_type}, which \
                 {RTL_KERNEL_ID} cannot consume (supported: i8, and i4 packed as i4-x2 weights)"
            ),
            RtlError::UnsupportedQuantization { layer, detail } => {
                write!(f, "rtl: step {layer}: unsupported requantization: {detail}")
            }
            RtlError::WeightsNotResident {
                layer,
                tensor,
                detail,
            } => write!(
                f,
                "rtl: step {layer}: weights tensor {tensor} is not ROM-resident: {detail}"
            ),
            RtlError::UnsupportedLayout { detail } => {
                write!(f, "rtl: unsupported layout: {detail}")
            }
            RtlError::ShapeMismatch {
                layer,
                what,
                expected,
                actual,
            } => write!(
                f,
                "rtl: step {layer}: {what} length mismatch: expected {expected}, got {actual}"
            ),
            RtlError::InvalidKernelParam {
                layer,
                field,
                detail,
            } => write!(f, "rtl: step {layer}: invalid {field}: {detail}"),
            RtlError::MissingProvenance { field } => {
                write!(f, "rtl: missing provenance field `{field}`")
            }
            RtlError::Serialize { context, source } => {
                write!(f, "rtl: {context}: {source}")
            }
            RtlError::Io { context, source } => write!(f, "rtl: {context}: {source}"),
        }
    }
}

impl Error for RtlError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RtlError::Serialize { source, .. } => Some(source),
            RtlError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Element type of a stream or tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    I8,
    I9,
    I16,
    I4,
    U8,
    I32,
    F32,
}

impl From<LayoutElemType> for ElementType {
    fn from(value: LayoutElemType) -> Self {
        match value {
            LayoutElemType::I8 => ElementType::I8,
            LayoutElemType::I9 => ElementType::I9,
            LayoutElemType::I16 => ElementType::I16,
            LayoutElemType::I4 => ElementType::I4,
            LayoutElemType::U8 => ElementType::U8,
            LayoutElemType::I32 => ElementType::I32,
            LayoutElemType::F32 => ElementType::F32,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ElementType::I8 => "i8",
            ElementType::I9 => "i9",
            ElementType::I16 => "i16",
            ElementType::I4 => "i4",
            ElementType::U8 => "u8",
            ElementType::I32 => "i32",
            ElementType::F32 => "f32",
        })
    }
}

/// Supplies stored weight bytes for a tensor referenced by the plan / dialect.
pub trait WeightResolver {
    fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError>;
}

/// Tile geometry recorded in bundle metadata (from solved engines' simd×pe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileGeometry {
    pub simd: u32,
    pub noc: u32,
    pub k: u32,
    pub fold: u32,
}

/// Plan-level context recorded in bundle metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanContext {
    pub plan_format: String,
    pub plan_schema_version: u32,
    pub execution_mode: String,
    pub granularity: String,
    pub engine: String,
    pub step_index: u32,
    pub step_count: u32,
    pub estimated_cycles: u64,
    pub spatial_in: [u32; 2],
    pub spatial_out: [u32; 2],
}

/// One output channel's requantization parameters (metadata / legacy helpers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRequant {
    pub multiplier: u32,
    pub shift: u8,
    pub lo: i16,
    pub hi: i16,
}

/// Provenance required for a reproducible RTL bundle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub compiler_version: String,
    pub git_revision: String,
    pub dirty: bool,
    pub graph_schema_version: u32,
    pub plan_schema_version: u32,
    pub target: String,
    pub target_revision: String,
    pub input_manifest_sha256: String,
    pub weights_sha256: String,
    pub planner_config_sha256: String,
}

impl Provenance {
    pub fn validate(&self) -> Result<(), RtlError> {
        let fields: [(&'static str, &str); 7] = [
            ("compiler_version", &self.compiler_version),
            ("git_revision", &self.git_revision),
            ("target", &self.target),
            ("target_revision", &self.target_revision),
            ("input_manifest_sha256", &self.input_manifest_sha256),
            ("weights_sha256", &self.weights_sha256),
            ("planner_config_sha256", &self.planner_config_sha256),
        ];
        for (name, value) in fields {
            if value.trim().is_empty() {
                return Err(RtlError::MissingProvenance { field: name });
            }
        }
        if self.plan_schema_version == 0 {
            return Err(RtlError::MissingProvenance {
                field: "plan_schema_version",
            });
        }
        Ok(())
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Provenance {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            git_revision: "unknown".to_string(),
            dirty: false,
            graph_schema_version: 0,
            plan_schema_version: 1,
            target: "unspecified".to_string(),
            target_revision: "unspecified".to_string(),
            input_manifest_sha256: "unspecified".to_string(),
            weights_sha256: "unspecified".to_string(),
            planner_config_sha256: "unspecified".to_string(),
        }
    }
}

/// `ceil(log2(value))`, at least 1.
pub fn bits_for(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    u32::BITS - (value - 1).leading_zeros()
}

/// Deterministic identifier fragment: lowercase `[a-z0-9_]`, leading letter.
pub fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() || out.as_bytes()[0].is_ascii_digit() {
        out.insert(0, 'm');
    }
    out
}

/// Deterministic PRNG used for stimulus frames.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_i8(&mut self) -> i8 {
        (self.next_u64() as i8).wrapping_add(1).saturating_sub(1)
    }
}

/// Options for CIRCT hybrid emission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlConfig {
    pub module_prefix: String,
    pub emit_testbench: bool,
    pub emit_memory_images: bool,
    pub fixture_windows: usize,
    pub fixture_seed: u64,
}

impl Default for RtlConfig {
    fn default() -> Self {
        RtlConfig {
            module_prefix: "tessera".to_string(),
            emit_testbench: true,
            emit_memory_images: true,
            fixture_windows: 16,
            fixture_seed: 0x5eed_1234_abcd_0001,
        }
    }
}

impl RtlConfig {
    pub fn validate(&self) -> Result<(), RtlError> {
        if self.module_prefix.trim().is_empty() {
            return Err(RtlError::InvalidKernelParam {
                layer: "<config>".to_string(),
                field: "module_prefix",
                detail: "must be non-empty".to_string(),
            });
        }
        if self.emit_testbench && self.fixture_windows == 0 {
            return Err(RtlError::InvalidKernelParam {
                layer: "<config>".to_string(),
                field: "fixture_windows",
                detail: "a testbench needs at least one stimulus window".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RtlFileKind {
    SystemVerilog,
    Testbench,
    MemoryImage,
    Script,
    Metadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlFile {
    pub name: String,
    pub kind: RtlFileKind,
    pub contents: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub name: String,
    pub kind: RtlFileKind,
    pub bytes: usize,
    pub fnv1a64: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RtlMetadata {
    pub format: String,
    pub schema_version: u32,
    pub kernel_id: String,
    pub provenance: Provenance,
    pub plan: PlanContext,
    pub top_module: String,
    pub kernel_module: String,
    pub layer: String,
    pub tile: TileGeometry,
    pub in_channels: u64,
    pub out_channels: u64,
    pub accumulator_width: u32,
    pub max_abs_accumulator: u64,
    pub kernel_cycles_per_window: u64,
    pub requantization: Vec<ChannelRequant>,
    pub requant_multiplier_error_ppm: Option<u32>,
    pub config: RtlConfig,
    pub files: Vec<FileDigest>,
    pub supported_operators: Vec<String>,
    pub supported_interface: Vec<String>,
    pub unsupported: Vec<String>,
    pub reuse_candidates: Vec<String>,
    pub reproduce: Vec<String>,
    pub bundle_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtlBundle {
    pub metadata: RtlMetadata,
    files: Vec<RtlFile>,
}

impl RtlBundle {
    pub fn files(&self) -> &[RtlFile] {
        &self.files
    }

    pub fn file(&self, name: &str) -> Option<&RtlFile> {
        self.files.iter().find(|f| f.name == name)
    }

    pub fn verilog_sources(&self) -> Vec<&str> {
        let mut sources: Vec<&RtlFile> = self
            .files
            .iter()
            .filter(|f| matches!(f.kind, RtlFileKind::SystemVerilog | RtlFileKind::Testbench))
            .collect();
        let rank = |f: &RtlFile| match f.kind {
            RtlFileKind::SystemVerilog => 0u8,
            RtlFileKind::Testbench => 1,
            _ => 2,
        };
        sources.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.name.cmp(&b.name)));
        sources.into_iter().map(|f| f.name.as_str()).collect()
    }

    pub fn write_to(&self, dir: &Path) -> Result<Vec<PathBuf>, RtlError> {
        fs::create_dir_all(dir).map_err(|source| RtlError::Io {
            context: format!("creating {}", dir.display()),
            source,
        })?;
        let mut written = Vec::with_capacity(self.files.len());
        for file in &self.files {
            let path = dir.join(&file.name);
            fs::write(&path, &file.contents).map_err(|source| RtlError::Io {
                context: format!("writing {}", path.display()),
                source,
            })?;
            written.push(path);
        }
        Ok(written)
    }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone)]
pub enum SimulationOutcome {
    Passed { log: String },
    Failed { log: String, detail: String },
    ToolUnavailable { tool: String, reason: String },
}

impl SimulationOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, SimulationOutcome::Passed { .. })
    }

    pub fn log(&self) -> &str {
        match self {
            SimulationOutcome::Passed { log } | SimulationOutcome::Failed { log, .. } => log,
            SimulationOutcome::ToolUnavailable { reason, .. } => reason,
        }
    }
}

pub fn verilator_version() -> Option<String> {
    let binary = std::env::var("TESSERA_VERILATOR").unwrap_or_else(|_| "verilator".to_string());
    let output = Command::new(&binary).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Some(text.lines().next().unwrap_or("verilator").trim().to_string())
}

/// Run the bundle's self-checking testbench under Verilator when available.
pub fn simulate_with_verilator(
    bundle: &RtlBundle,
    work_dir: &Path,
) -> Result<SimulationOutcome, RtlError> {
    let binary = std::env::var("TESSERA_VERILATOR").unwrap_or_else(|_| "verilator".to_string());
    let version = match verilator_version() {
        Some(version) => version,
        None => {
            return Ok(SimulationOutcome::ToolUnavailable {
                tool: binary,
                reason: "verilator not found on PATH (set TESSERA_VERILATOR to override)"
                    .to_string(),
            });
        }
    };

    if bundle
        .files()
        .iter()
        .all(|f| f.kind != RtlFileKind::Testbench)
    {
        return Err(RtlError::UnsupportedLayout {
            detail: "bundle has no testbench; enable RtlConfig::emit_testbench".to_string(),
        });
    }

    bundle.write_to(work_dir)?;
    let sources: Vec<String> = bundle
        .verilog_sources()
        .into_iter()
        .map(|name| name.to_string())
        .collect();
    let obj_dir = work_dir.join("obj_dir");
    let top = bundle.metadata.top_module.clone();

    let mut command = Command::new(&binary);
    command
        .current_dir(work_dir)
        .arg("--binary")
        .arg("--timing")
        .arg("-Wno-fatal")
        .arg("-Wno-DECLFILENAME")
        .arg("--top-module")
        .arg(&top)
        .arg("--Mdir")
        .arg(&obj_dir)
        .arg("-o")
        .arg(format!("{top}_sim"));
    for source in &sources {
        command.arg(source);
    }
    let compile = command.output().map_err(|source| RtlError::Io {
        context: format!("running {binary}"),
        source,
    })?;
    let mut log = format!("--- verilator ({version}) ---\n");
    log.push_str(&String::from_utf8_lossy(&compile.stdout));
    log.push_str(&String::from_utf8_lossy(&compile.stderr));
    if !compile.status.success() {
        return Ok(SimulationOutcome::Failed {
            log,
            detail: format!("verilator exited with {}", compile.status),
        });
    }

    let sim_path = obj_dir.join(format!("{top}_sim"));
    let run = Command::new(&sim_path)
        .current_dir(work_dir)
        .output()
        .map_err(|source| RtlError::Io {
            context: format!("running {}", sim_path.display()),
            source,
        })?;
    log.push_str("--- simulation ---\n");
    log.push_str(&String::from_utf8_lossy(&run.stdout));
    log.push_str(&String::from_utf8_lossy(&run.stderr));

    if run.status.success() && log.contains("TESSA_RTL_PASS") {
        Ok(SimulationOutcome::Passed { log })
    } else {
        Ok(SimulationOutcome::Failed {
            log,
            detail: format!("simulator exited with {}", run.status),
        })
    }
}
