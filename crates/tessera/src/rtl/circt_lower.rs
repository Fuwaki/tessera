//! Tessera dialect decisions → CIRCT (`hw` / `comb` / `sv`) in the same Melior
//! context, then in-process [`super::circt_ctx::export_verilog`].
//!
//! Sole datapath: the integer `*_sched_core` (ROM + MAC + requant + add/upsample)
//! bit-exact vs [`super::circt_data::evaluate_integer`]. Solved `tessera.engine`
//! binds appear only as `engines_busy` one-hots decoded from the core's current
//! op — there are no per-engine stub FSMs.
//!
//! Sequential state uses `sv.reg` + `sv.alwaysff` so ExportVerilog needs no
//! prior `--lower-seq-to-sv` pass.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use melior::ir::{Module, operation::OperationLike};

use crate::dialect::read::{BufferOp, EngineOp, OperatorOp, SegmentOp, TesseraModule};
use crate::dialect::{self, verify_module};

use super::circt_ctx::{self, load_circt_dialects};
use super::circt_data::{ResolvedOp, evaluate_integer, resolve_ops, stimulus_frame};
use super::{
    FileDigest, PlanContext, Provenance, RtlBundle, RtlConfig, RtlError, RtlFile, RtlFileKind,
    RtlMetadata, TileGeometry, RTL_BUNDLE_FORMAT, RTL_BUNDLE_SCHEMA_VERSION, bits_for, fnv1a64,
    slug,
};

/// Stable identifier of the CIRCT hybrid kernel family.
pub const CIRCT_KERNEL_ID: &str = "tessera.hybrid.circt.v1";

/// Default K for rotating cross-segment banks (matches the dialect solver).
pub const KBANK_K: u64 = 2;

/// Summary for the CIRCT hybrid backend.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CirctSummary {
    pub engines: usize,
    pub segments: usize,
    pub local_arenas: usize,
    pub kbanks: usize,
    pub binds: usize,
    pub initiation_interval: u64,
    pub latency: u64,
    pub mac_lanes_total: u64,
}

/// Lower a *solved* tessera module to SystemVerilog via CIRCT in `context`.
pub fn lower_solved_tessera(
    context: &melior::Context,
    tessera: &Module,
    config: &RtlConfig,
) -> Result<(RtlBundle, CirctSummary), RtlError> {
    lower_solved_tessera_with_content(context, tessera, config, None, None)
}

/// Lower with graph + weights so ROMs and the bit-exact testbench can be filled.
pub fn lower_solved_tessera_with_content(
    context: &melior::Context,
    tessera: &Module,
    config: &RtlConfig,
    graph: Option<&crate::graph::Graph>,
    weights: Option<&dyn super::WeightResolver>,
) -> Result<(RtlBundle, CirctSummary), RtlError> {
    config.validate()?;
    load_circt_dialects(context);
    let model = verify_module(tessera).map_err(|e| RtlError::UnsupportedLayout {
        detail: format!("tessera module failed verification before CIRCT lowering: {e}"),
    })?;
    if !model.has_decisions() {
        return Err(RtlError::UnsupportedLayout {
            detail: "CIRCT hybrid lowering needs solved decisions; run `tessera solve --tessera` first"
                .into(),
        });
    }
    let mut plan = HybridPlan::from_model(&model)?;
    if let (Some(graph), Some(weights)) = (graph, weights) {
        plan.resolved = resolve_ops(&model, graph, weights)?;
        plan.results = model.results.clone();
        if let Some(arg) = model.arguments.first() {
            plan.in_shape = [
                arg.shape.get(1).copied().unwrap_or(1) as u32,
                arg.shape.get(2).copied().unwrap_or(1) as u32,
                arg.shape.get(3).copied().unwrap_or(1) as u32,
            ];
        }
    }
    let summary = plan.summary();
    let prefix = slug(&config.module_prefix);
    let dut_name = format!("{prefix}_hybrid_circt");
    let circt_ir = emit_circt_ir(&plan, &dut_name)?;
    let circt_module = match Module::parse(context, &circt_ir) {
        Some(m) => m,
        None => {
            let _ = std::fs::write("/tmp/tessera_circt_fail.mlir", &circt_ir);
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "failed to parse generated CIRCT IR ({} bytes); dumped /tmp/tessera_circt_fail.mlir",
                    circt_ir.len()
                ),
            });
        }
    };
    if !circt_module.as_operation().verify() {
        return Err(RtlError::UnsupportedLayout {
            detail: "generated CIRCT module failed MLIR verification".into(),
        });
    }
    let sv = circt_ctx::export_verilog(&circt_module)?;
    if !sv.contains(dut_name.as_str()) {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "ExportVerilog output missing top `{dut_name}` ({} bytes)",
                sv.len()
            ),
        });
    }
    package_bundle(plan, summary, dut_name, sv, config)
}

/// Register tessera + CIRCT, parse solved MLIR text, lower.
pub fn lower_solved_tessera_text(
    solved_mlir: &str,
    config: &RtlConfig,
) -> Result<(RtlBundle, CirctSummary), RtlError> {
    let context = dialect::register_in_fresh_context().map_err(|e| RtlError::UnsupportedLayout {
        detail: format!("register tessera dialect: {e}"),
    })?;
    load_circt_dialects(&context);
    let module = dialect::load_tessera_module(&context, solved_mlir).map_err(|e| {
        RtlError::UnsupportedLayout {
            detail: format!("parse solved tessera module: {e}"),
        }
    })?;
    lower_solved_tessera_with_content(&context, &module, config, None, None)
}

/// Like [`lower_solved_tessera_text`] but fills ROMs / bit-exact TB from graph+weights.
pub fn lower_solved_tessera_text_with_content(
    solved_mlir: &str,
    config: &RtlConfig,
    graph: &crate::graph::Graph,
    weights: &dyn super::WeightResolver,
) -> Result<(RtlBundle, CirctSummary), RtlError> {
    let context = dialect::register_in_fresh_context().map_err(|e| RtlError::UnsupportedLayout {
        detail: format!("register tessera dialect: {e}"),
    })?;
    load_circt_dialects(&context);
    let module = dialect::load_tessera_module(&context, solved_mlir).map_err(|e| {
        RtlError::UnsupportedLayout {
            detail: format!("parse solved tessera module: {e}"),
        }
    })?;
    lower_solved_tessera_with_content(&context, &module, config, Some(graph), Some(weights))
}


fn package_bundle(
    plan: HybridPlan,
    summary: CirctSummary,
    dut_name: String,
    sv: String,
    config: &RtlConfig,
) -> Result<(RtlBundle, CirctSummary), RtlError> {
    let tb_name = format!("{dut_name}_tb");
    let mut files: Vec<RtlFile> = Vec::new();
    files.push(RtlFile {
        name: format!("{dut_name}.sv"),
        kind: RtlFileKind::SystemVerilog,
        contents: sv,
    });
    if config.emit_testbench {
        let (stim, expected, tb) = emit_bitexact_artifacts(&dut_name, &tb_name, &plan, config)?;
        files.push(RtlFile {
            name: "stim_input.hex".into(),
            kind: RtlFileKind::MemoryImage,
            contents: stim,
        });
        files.push(RtlFile {
            name: "expected.hex".into(),
            kind: RtlFileKind::MemoryImage,
            contents: expected,
        });
        files.push(RtlFile {
            name: format!("{tb_name}.sv"),
            kind: RtlFileKind::Testbench,
            contents: tb,
        });
        files.push(RtlFile {
            name: "run_verilator.sh".into(),
            kind: RtlFileKind::Script,
            contents: emit_run_script(&dut_name, &tb_name),
        });
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    let digests: Vec<FileDigest> = files
        .iter()
        .map(|f| FileDigest {
            name: f.name.clone(),
            kind: f.kind,
            bytes: f.contents.len(),
            fnv1a64: format!("{:016x}", fnv1a64(f.contents.as_bytes())),
        })
        .collect();
    let mut digest_input = String::new();
    for d in &digests {
        let _ = writeln!(digest_input, "{} {} {}", d.name, d.bytes, d.fnv1a64);
    }
    let bundle_digest = format!("{:016x}", fnv1a64(digest_input.as_bytes()));
    let mut provenance = Provenance::default();
    provenance.compiler_version = env!("CARGO_PKG_VERSION").into();
    provenance.git_revision = "mainline-hybrid".into();
    provenance.plan_schema_version = 1;
    provenance.graph_schema_version = 1;
    provenance.target = "circt-hybrid".into();
    provenance.target_revision = CIRCT_KERNEL_ID.into();
    provenance.input_manifest_sha256 = "tessera-dialect-solved".into();
    provenance.weights_sha256 = "fnv1a64:pending".into();
    provenance.planner_config_sha256 = format!(
        "ii={};latency={};engines={}",
        summary.initiation_interval, summary.latency, summary.engines
    );
    provenance.validate()?;
    let metadata = RtlMetadata {
        format: RTL_BUNDLE_FORMAT.into(),
        schema_version: RTL_BUNDLE_SCHEMA_VERSION,
        kernel_id: CIRCT_KERNEL_ID.into(),
        provenance,
        plan: PlanContext {
            plan_format: "tessera.dialect".into(),
            plan_schema_version: 1,
            execution_mode: "hybrid".into(),
            granularity: "frame".into(),
            engine: plan.engines.first().map(|e| e.name.clone()).unwrap_or_else(|| "none".into()),
            step_index: 0,
            step_count: plan.ops.len() as u32,
            estimated_cycles: summary.latency.max(summary.initiation_interval),
            spatial_in: [0, 0],
            spatial_out: [0, 0],
        },
        top_module: if config.emit_testbench { tb_name } else { dut_name.clone() },
        kernel_module: dut_name,
        layer: plan.ops.iter().map(|o| o.name.clone()).collect::<Vec<_>>().join(","),
        tile: TileGeometry {
            simd: plan.engines.first().map(|e| e.simd as u32).unwrap_or(1),
            noc: plan.engines.first().map(|e| e.pe as u32).unwrap_or(1),
            k: 1,
            fold: 1,
        },
        in_channels: plan.ops.first().and_then(|o| o.operand_types.first()).and_then(|t| t.shape.get(3).copied()).unwrap_or(0),
        out_channels: plan.ops.last().map(|o| o.output.shape.get(3).copied().unwrap_or(0)).unwrap_or(0),
        accumulator_width: 32,
        max_abs_accumulator: 0,
        kernel_cycles_per_window: summary.initiation_interval,
        requantization: Vec::new(),
        requant_multiplier_error_ppm: None,
        config: config.clone(),
        files: digests,
        supported_operators: vec!["hybrid CIRCT: engines (simd×pe), arenas, K-banks, frame-pipeline".into()],
        supported_interface: vec![
            "clk, rst".into(),
            "start / done".into(),
            format!("engines_busy [{}:0]", plan.engines.len().saturating_sub(1)),
            "ld_en/ld_addr/ld_data".into(),
            "rd_addr/rd_data".into(),
        ],
        unsupported: vec![
            "hybrid engine slots are bind annotations on sched_core (no per-engine FSMs)".into(),
        ],
        reuse_candidates: vec!["CIRCT hw/comb/sv → ExportVerilog (circt-sys)".into()],
        reproduce: vec!["emit: lower_solved_tessera".into(), "simulate: run_verilator.sh".into()],
        bundle_digest,
    };
    let metadata_json = serde_json::to_string_pretty(&metadata).map_err(|source| RtlError::Serialize {
        context: "serializing CIRCT hybrid bundle metadata",
        source,
    })?;
    files.push(RtlFile { name: "rtl_bundle.json".into(), kind: RtlFileKind::Metadata, contents: metadata_json });
    Ok((RtlBundle { metadata, files }, summary))
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EngineInst {
    name: String,
    mode: String,
    simd: u64,
    pe: u64,
    ops: Vec<u32>,
}

#[derive(Debug, Clone)]
struct LocalArena {
    segment: u64,
    words: u64,
}

#[derive(Debug, Clone)]
struct KBank {
    name: String,
    words: u64,
    banks: u64,
    producer: Option<u32>,
}

#[derive(Debug, Clone)]
struct HybridPlan {
    ops: Vec<OperatorOp>,
    engines: Vec<EngineInst>,
    segments: Vec<SegmentOp>,
    arenas: Vec<LocalArena>,
    kbanks: Vec<KBank>,
    binds: Vec<(u32, String, u64, u64)>,
    initiation_interval: u64,
    latency: u64,
    /// Filled when graph+weights are supplied.
    resolved: Vec<ResolvedOp>,
    results: Vec<u32>,
    /// (H, W, C) of the graph input.
    in_shape: [u32; 3],
}

impl HybridPlan {
    fn from_model(model: &TesseraModule) -> Result<Self, RtlError> {
        let schedule = model.schedule.as_ref().ok_or_else(|| RtlError::UnsupportedLayout {
            detail: "solved module has no tessera.schedule".into(),
        })?;
        if model.segments.is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: "solved module has no tessera.segment ops".into(),
            });
        }
        if model.engines.is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: "solved module has no tessera.engine ops".into(),
            });
        }

        let mut engines: Vec<EngineInst> = model
            .engines
            .iter()
            .map(|e: &EngineOp| EngineInst {
                name: e.name.clone(),
                mode: e.mode.clone(),
                simd: e.simd.max(1),
                pe: e.pe.max(1),
                ops: Vec::new(),
            })
            .collect();
        let eng_index: BTreeMap<String, usize> = engines
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name.clone(), i))
            .collect();

        let mut binds = Vec::new();
        for b in &model.binds {
            binds.push((b.op, b.engine.clone(), b.segment, b.step));
            let Some(&ei) = eng_index.get(&b.engine) else {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!("bind references unknown engine {}", b.engine),
                });
            };
            engines[ei].ops.push(b.op);
        }
        for e in &mut engines {
            e.ops.sort_by_key(|op| {
                binds
                    .iter()
                    .find(|(o, _, _, _)| o == op)
                    .map(|(_, _, _, step)| *step)
                    .unwrap_or(0)
            });
            e.ops.dedup();
        }

        let mut arena_words: BTreeMap<u64, u64> = BTreeMap::new();
        for seg in &model.segments {
            arena_words.insert(seg.index, 1);
        }
        let mut kbanks = Vec::new();
        let mut seen_kbank = BTreeSet::new();
        for buf in &model.buffers {
            if buf.residence == "local" || buf.residence == "arena" {
                let seg = infer_buffer_segment(buf, &model.binds);
                let entry = arena_words.entry(seg).or_insert(1);
                *entry = (*entry).max(buf.base.saturating_add(buf.words).max(1));
            } else {
                let key = buf.producer;
                if seen_kbank.insert(key) {
                    let name = match key {
                        None => "kbank_in".to_string(),
                        Some(p) => format!("kbank_t{p}"),
                    };
                    kbanks.push(KBank {
                        name,
                        words: buf.words.max(1),
                        banks: buf.banks.max(KBANK_K),
                        producer: key,
                    });
                } else if let Some(existing) = kbanks.iter_mut().find(|k| k.producer == key) {
                    existing.words = existing.words.max(buf.words.max(1));
                    existing.banks = existing.banks.max(buf.banks.max(KBANK_K));
                }
            }
        }
        if !kbanks.iter().any(|k| k.producer.is_none()) {
            let in_words = model
                .arguments
                .first()
                .map(|t| t.elements())
                .unwrap_or(1)
                .max(1);
            kbanks.insert(
                0,
                KBank {
                    name: "kbank_in".into(),
                    words: in_words,
                    banks: KBANK_K,
                    producer: None,
                },
            );
        }
        if let Some(&out_op) = model.results.first() {
            if !kbanks.iter().any(|k| k.producer == Some(out_op)) {
                let words = model
                    .operator(out_op)
                    .map(|o| o.output.elements())
                    .unwrap_or(1)
                    .max(1);
                kbanks.push(KBank {
                    name: format!("kbank_t{out_op}"),
                    words,
                    banks: KBANK_K,
                    producer: Some(out_op),
                });
            }
        }

        let arenas: Vec<LocalArena> = arena_words
            .into_iter()
            .map(|(segment, words)| LocalArena { segment, words })
            .collect();

        Ok(HybridPlan {
            ops: model.operators.clone(),
            engines,
            segments: model.segments.clone(),
            arenas,
            kbanks,
            binds,
            initiation_interval: schedule.initiation_interval.max(1),
            latency: schedule.latency.max(1),
            resolved: Vec::new(),
            results: model.results.clone(),
            in_shape: [1, 1, 1],
        })
    }

    fn summary(&self) -> CirctSummary {
        CirctSummary {
            engines: self.engines.len(),
            segments: self.segments.len(),
            local_arenas: self.arenas.len(),
            kbanks: self.kbanks.len(),
            binds: self.binds.len(),
            initiation_interval: self.initiation_interval,
            latency: self.latency,
            mac_lanes_total: self.engines.iter().map(|e| e.simd * e.pe).sum(),
        }
    }

    fn max_arena_words(&self) -> u64 {
        self.arenas.iter().map(|a| a.words).max().unwrap_or(1).max(1)
    }
}

fn infer_buffer_segment(buf: &BufferOp, binds: &[crate::dialect::read::BindOp]) -> u64 {
    if let Some(p) = buf.producer {
        if let Some(b) = binds.iter().find(|b| b.op == p) {
            return b.segment;
        }
    }
    binds.first().map(|b| b.segment).unwrap_or(0)
}

fn emit_circt_ir(plan: &HybridPlan, dut_name: &str) -> Result<String, RtlError> {
    if plan.resolved.is_empty() {
        return Err(RtlError::UnsupportedLayout {
            detail: "CIRCT hybrid emit needs graph+weights (resolved ops); pass --graph <graph.tgraph>"
                .into(),
        });
    }
    let mut ir = String::new();
    let _ = writeln!(
        ir,
        "// Tessera hybrid CIRCT — sole datapath=sched_core; {} eng binds, {} seg, II={}",
        plan.engines.len(),
        plan.segments.len(),
        plan.initiation_interval
    );
    emit_mac_lane(&mut ir);
    // Map each resolved op (in schedule order) to its hybrid engine index.
    let eng_index: std::collections::BTreeMap<String, usize> = plan
        .engines
        .iter()
        .enumerate()
        .map(|(i, e)| (e.name.clone(), i))
        .collect();
    let mut op_engines: Vec<usize> = Vec::with_capacity(plan.resolved.len());
    for rop in &plan.resolved {
        let eng = plan
            .binds
            .iter()
            .find(|(op, _, _, _)| *op == rop.index)
            .and_then(|(_, name, _, _)| eng_index.get(name).copied())
            .unwrap_or(0);
        op_engines.push(eng);
    }
    super::circt_datapath::emit_sched_core(
        &mut ir,
        &plan.resolved,
        plan.in_shape,
        &plan.results,
        &op_engines,
        plan.engines.len().max(1),
        dut_name,
    )?;
    emit_top(&mut ir, plan, dut_name);
    Ok(ir)
}

fn emit_mac_lane(ir: &mut String) {
    // i16 activation × i8 weight → i32 accumulate (acts may exceed i8 after unclamped add).
    ir.push_str(concat!(
        "hw.module @tessera_mac_lane(in %a : i16, in %b : i8, in %acc : i32, out o : i32) {
",
        "  %as = comb.extract %a from 15 : (i16) -> i1
",
        "  %ar = comb.replicate %as : (i1) -> i16
",
        "  %ae = comb.concat %ar, %a : i16, i16
",
        "  %bs = comb.extract %b from 7 : (i8) -> i1
",
        "  %br = comb.replicate %bs : (i1) -> i24
",
        "  %be = comb.concat %br, %b : i24, i8
",
        "  %p = comb.mul %ae, %be : i32
",
        "  %s = comb.add %acc, %p : i32
",
        "  hw.output %s : i32
",
        "}

",
    ));
}

fn emit_top(ir: &mut String, plan: &HybridPlan, dut_name: &str) {
    let in_words = u64::from(plan.in_shape[0])
        * u64::from(plan.in_shape[1])
        * u64::from(plan.in_shape[2]);
    let out_words = plan
        .resolved
        .iter()
        .find(|o| plan.results.first() == Some(&o.index))
        .map(|o| o.out_words())
        .or_else(|| plan.resolved.last().map(|o| o.out_words()))
        .unwrap_or(1);
    let in_aw = bits_for(in_words.max(1) as u32).max(1);
    let out_aw = bits_for(out_words.max(1) as u32).max(1);
    super::circt_datapath::emit_sched_top(
        ir,
        plan.engines.len().max(1),
        in_aw,
        out_aw,
        dut_name,
    );
}

fn emit_bitexact_artifacts(
    dut: &str,
    tb: &str,
    plan: &HybridPlan,
    config: &RtlConfig,
) -> Result<(String, String, String), RtlError> {
    let [ih, iw, ic] = plan.in_shape;
    // in_shape stored as [H,W,C] in lower_solved_tessera_with_content
    let input = stimulus_frame(ic, ih, iw, config.fixture_seed);
    let golden = evaluate_integer(&plan.resolved, &input, &plan.results)?;
    let mut stim = String::new();
    for &v in &input {
        let _ = writeln!(stim, "{:04x}", (v as i16) as u16);
    }
    let mut expected = String::new();
    for &v in &golden {
        let _ = writeln!(expected, "{:04x}", (v as i16) as u16);
    }
    let neng = plan.engines.len().max(1);
    let in_words = input.len();
    let out_words = golden.len();
    let in_aw = bits_for(in_words.max(1) as u32).max(1);
    let out_aw = bits_for(out_words.max(1) as u32).max(1);
    let busy_w = if neng == 1 { 1 } else { neng };
    // Worst-case cycles: sum over ops of out_words * (inner+2) + overhead
    let mut beats: u64 = 64;
    for op in &plan.resolved {
        let inner = match op.kind {
            crate::dialect::read::OpKind::Conv2d | crate::dialect::read::OpKind::PointwiseConv2d => {
                let icg = (op.ic / op.groups.max(1)).max(1);
                u64::from(icg) * u64::from(op.kernel[0].max(1)) * u64::from(op.kernel[1].max(1))
            }
            crate::dialect::read::OpKind::DepthwiseConv2d => {
                u64::from(op.kernel[0].max(1)) * u64::from(op.kernel[1].max(1))
            }
            _ => 1,
        };
        beats += op.out_words() * (inner + 3);
    }
    let timeout = beats.saturating_mul(4).max(10_000);
    let tb_sv = format!(
        r#"// Bit-exact TB: CIRCT hybrid sched_core vs circt_data::evaluate_integer
`timescale 1ns/1ps
module {tb};
  logic clk = 0, rst = 1, start = 0, done, ld_en = 0;
  logic [{busy_w}-1:0] engines_busy;
  logic [{in_aw}-1:0] ld_addr = '0;
  logic signed [15:0] ld_data = '0;
  logic [{out_aw}-1:0] rd_addr = '0;
  logic signed [15:0] rd_data;
  logic signed [15:0] stim [0:{in_words}-1];
  logic signed [15:0] exp_mem [0:{out_words}-1];
  int mism = 0, checks = 0, cyc = 0, i;
  string stim_path = "stim_input.hex";
  string exp_path = "expected.hex";

  {dut} dut (
    .clk(clk), .rst(rst), .start(start), .done(done),
    .engines_busy(engines_busy),
    .ld_en(ld_en), .ld_addr(ld_addr), .ld_data(ld_data),
    .rd_addr(rd_addr), .rd_data(rd_data)
  );

  initial forever #5 clk = ~clk;
  always @(posedge clk) begin
    cyc <= cyc + 1;
    if (cyc > {timeout}) begin
      $display("TESSA_RTL_TIMEOUT cycles=%0d", cyc);
      $display("TESSA_RTL_FAIL");
      $finish;
    end
  end

  initial begin
    $readmemh(stim_path, stim);
    $readmemh(exp_path, exp_mem);
    repeat (4) @(negedge clk);
    rst = 0;
    // Load stimulus into sched_core input region.
    for (i = 0; i < {in_words}; i = i + 1) begin
      @(negedge clk);
      ld_en = 1; ld_addr = i[{in_aw}-1:0]; ld_data = stim[i];
    end
    @(negedge clk); ld_en = 0;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait (done === 1'b1);
    @(negedge clk);
    for (i = 0; i < {out_words}; i = i + 1) begin
      rd_addr = i[{out_aw}-1:0];
      @(negedge clk);
      checks = checks + 1;
      if (rd_data !== exp_mem[i]) begin
        mism = mism + 1;
        if (mism <= 12)
          $display("TESSA_RTL_MISMATCH i=%0d expected=%0d actual=%0d", i, $signed(exp_mem[i]), $signed(rd_data));
      end
    end
    $display("TESSA_RTL_RESULT module={tb} checks=%0d mismatches=%0d cycles=%0d", checks, mism, cyc);
    if (mism == 0) $display("TESSA_RTL_PASS");
    else $display("TESSA_RTL_FAIL");
    $finish;
  end
endmodule
"#
    );
    let _ = (dut,);
    Ok((stim, expected, tb_sv))
}

fn emit_run_script(dut: &str, tb: &str) -> String {
    format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nverilator --binary --timing -Wno-fatal --top-module {tb} -o {tb}_sim {dut}.sv {tb}.sv\n./obj_dir/{tb}_sim\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{linalg_to_tessera, register_in_fresh_context};
    use crate::frontend;
    use crate::lower::InputShape;
    use crate::mlir::{self, MlirOptions};
    use crate::planner::{Policy, ProvenanceInput};
    use crate::solver::MappingBounds;
    use crate::solver::dialect::solve_tessera;
    use crate::target::Ti60Target;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }

    #[test]
    fn w8_solved_lowers_through_circt_export_verilog() {
        let shape = InputShape::new(3, 6, 6).expect("shape");
        let artifact = frontend::import_manifest(&fixtures().join("w8")).expect("import");
        let linalg = mlir::emit_text(artifact.graph(), &MlirOptions::new(shape.to_nhwc()))
            .expect("emit");
        let context = register_in_fresh_context().expect("dialect");
        load_circt_dialects(&context);
        let module = linalg_to_tessera(&context, &linalg).expect("convert");
        let target =
            Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("target");
        let policy =
            Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("policy");
        let provenance = ProvenanceInput {
            graph_sha256: artifact.provenance().manifest_sha256.clone(),
            target_sha256: "t".into(),
            policy_sha256: "p".into(),
            compiler_version: env!("CARGO_PKG_VERSION").into(),
        };
        let bounds = MappingBounds::hybrid_for(&target, &policy).expect("bounds");
        let solved = solve_tessera(&context, &module, &target, &policy, &provenance, &bounds)
            .expect("solve");
        assert!(solved.decisions.is_some());
        let mut cfg = RtlConfig::default();
        cfg.module_prefix = "w8".into();
        cfg.emit_testbench = true;
        let (bundle, summary) = lower_solved_tessera(&context, &module, &cfg).expect("circt lower");
        eprintln!(
            "w8 CIRCT: engines={} segments={} kbanks={} arenas={} lanes={} II={}",
            summary.engines,
            summary.segments,
            summary.kbanks,
            summary.local_arenas,
            summary.mac_lanes_total,
            summary.initiation_interval
        );
        assert!(summary.engines >= 1);
        assert!(summary.segments >= 1);
        assert!(summary.kbanks >= 1);
        let sv = bundle
            .file("w8_hybrid_circt.sv")
            .expect("dut sv")
            .contents
            .clone();
        assert!(sv.contains("module w8_hybrid_circt") || sv.contains("w8_hybrid_circt"), "{sv}");
        assert!(sv.contains("tessera_mac_lane") || sv.contains("mac_lane"), "mac lane missing");
        assert!(
            bundle.metadata.kernel_id == CIRCT_KERNEL_ID,
            "{}",
            bundle.metadata.kernel_id
        );
    }
}
