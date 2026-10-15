//! Mapped multi-engine RTL backend (schema 2).
//!
//! `lower_mapped` is the RTL side of the *mapped* organization the solver
//! produces. The plan's [`HardwareMapping`] is the authority: which engine
//! instances exist, which operator runs on which of them, where each tensor
//! lives (a word offset in the shared arena, or a FIFO between two engines),
//! and how long the critical path is. This module turns that structure into
//! SystemVerilog, and refuses anything it cannot implement faithfully.
//!
//! # Generated structure
//!
//! * **One module per engine instance**, with `simd x pe x tap_parallel`
//!   multiplier lanes instantiated as explicit `..._mul_lane` module instances.
//!   The lane count is structurally visible in the emitted text instead of
//!   being hidden inside one behavioural `*`.
//! * A **shared activation arena** in the top module, addressed by the mapped
//!   word offsets ([`BufferResidence::offset_words`]). Every engine reads and
//!   writes that one memory, which is exactly what makes an arena residence
//!   shared between two engines.
//! * A **ready/valid FIFO of `depth_elements` words** for every
//!   [`PipelineLink`], sitting between the producer and consumer instances.
//!   The producer buffers its output in a local RAM in its own natural order and
//!   flushes it through the FIFO in tensor order; the consumer drains the FIFO
//!   into a local RAM of its own before it computes. Both engines are busy at
//!   the same time for the duration of the transfer.
//! * A **sequencer** that issues one `start` pulse per schedule step. A step
//!   whose producer lives on another engine waits for that producer's `done`
//!   *unless* a pipeline link licenses overlap, in which case it starts one
//!   cycle after the producer starts and takes its words out of the FIFO. Two
//!   engines therefore overlap only where the plan says they may.
//!
//! # Numeric contract
//!
//! Identical to [`crate::rtl::reference`]: `i64` accumulation of
//! `bias + sum(i8 x i8)`, then `(acc * M0 + round) >>> shift` with the
//! sign-dependent rounding offset (`2^(shift-1)` for a non-negative product,
//! `2^(shift-1) - 1` otherwise), then the step's saturation clamp. The
//! testbench's expected codes are computed in Rust with that same formula, so a
//! matching simulation is independent evidence and never an echo of the RTL.
//!
//! The emitted accumulator register, and the accumulator operand of the requant
//! `acc * M0` multiply, are [`MAPPED_ACC_BITS`] bits wide: the width every
//! engine instance of the mapping declares as `accumulator_bits`, which is also
//! what the target model charges the requant multiplier for. The Rust reference
//! evaluation accumulates in `i64`, so the two agree exactly whenever the
//! accumulated `bias + sum(i8 x i8)` fits the declared width; a plan declaring
//! any other width is rejected rather than emitted.
//!
//! # Supported subset
//!
//! Pointwise 1x1 convolutions with `stride == [1, 1]`, `groups == 1`,
//! `tap_parallel == 1`, zero padding, signed `i8` weights and activations,
//! half-away-from-zero requantization with `M0 < 2^17`, and saturation bounds
//! that fit signed int8. Any other operator, element type, quantization form,
//! residence, or engine shape is rejected with an [`RtlError`] that names the
//! step and the construct.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use super::{
    PlanContext, Provenance, RTL_BUNDLE_FORMAT, RTL_BUNDLE_SCHEMA_VERSION, RtlBundle, RtlConfig,
    RtlError, RtlFile, RtlFileKind, RtlMetadata, TileGeometry, WeightResolver, bits_for, fnv1a64,
    slug,
};
use crate::graph::Graph;
use crate::layout::{
    ActivationKind, BufferKind, ElemType, EngineRole, ExecutionPlan, HardwareMapping, OpKind,
    REQUANT_MAX_MULTIPLIER, Requant, Rounding, ScheduleStep, TensorId,
};

/// Stable identifier of the mapped multi-engine kernel family.
pub const MAPPED_KERNEL_ID: &str = "tessera.mapped.i8i8.v1";
/// Activation word width of the emitted arena and stream ports.
pub const MAPPED_WORD_BITS: u32 = 16;
/// Accumulator width of the emitted MAC arrays.
///
/// This is the width the mapping declares on every engine instance
/// ([`crate::layout::EngineInstance::accumulator_bits`]) and the width the
/// target model charges the requant multiplier for
/// ([`crate::target::Ti60Target::map_requant`]: two DSP48 chunks of an 18-bit
/// multiplier operand). The emitted accumulator register and the accumulator
/// operand of the requant `acc * M0` multiply are exactly this wide, and
/// [`lower_mapped`] refuses a plan that declares anything else.
pub const MAPPED_ACC_BITS: u32 = 32;

/// Most-significant bit index of the emitted accumulator register.
const MAPPED_ACC_MSB: u32 = MAPPED_ACC_BITS - 1;

// ---------------------------------------------------------------------------
// Machine-readable summary
// ---------------------------------------------------------------------------

/// One generated engine instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedEngineSummary {
    /// Instance name.
    pub name: String,
    /// `shared` or `dedicated`.
    pub role: String,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Instantiated multiply-accumulate lanes (`simd x pe x tap`).
    pub lanes: u64,
    /// Operators bound to the instance, in run order.
    pub operators: Vec<String>,
}

/// One generated pipeline cut.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedLinkSummary {
    /// Producing engine.
    pub producer_engine: String,
    /// Consuming engine.
    pub consumer_engine: String,
    /// Tensor crossing the cut.
    pub tensor: String,
    /// Words crossing.
    pub words: u64,
    /// FIFO depth in words.
    pub depth_elements: u32,
}

/// Machine-readable summary of the generated mapped structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedSummary {
    /// Engine instances, in the mapping's order.
    pub engines: Vec<MappedEngineSummary>,
    /// Pipeline cuts.
    pub links: Vec<MappedLinkSummary>,
    /// Words in the shared activation arena.
    pub arena_words: u64,
    /// Address width of the shared activation arena.
    pub arena_address_bits: u32,
    /// Total instantiated multiplier lanes (`sum(simd x pe x tap)`).
    pub multiplier_lanes: u64,
    /// Whether any pipeline link was emitted.
    pub pipelined: bool,
    /// Critical-path frame length recorded by the plan.
    pub makespan_cycles: u64,
    /// Sum of the schedule steps' own cycle estimates.
    pub schedule_cycles: u64,
    /// Semantic notes about generated structure that differs from the plan's
    /// estimate.
    pub notes: Vec<String>,
}

impl MappedSummary {
    /// Whether the structure contains a ready/valid FIFO between two engines.
    pub fn pipelined(&self) -> bool {
        !self.links.is_empty()
    }

    /// Instantiated multiplier lanes of one engine, by name.
    pub fn lanes_of(&self, engine: &str) -> Option<u64> {
        self.engines
            .iter()
            .find(|e| e.name == engine)
            .map(|e| e.lanes)
    }
}

/// Emits the mapped RTL bundle for a schema-2 [`ExecutionPlan`].
///
/// The plan must carry a [`HardwareMapping`]; a schema-1 plan belongs to
/// [`super::SystemVerilogBackend::lower_execution_plan`] or to the full-graph
/// backend. `graph` is used to check that the plan was solved for the given
/// graph (every step name must name a graph layer); all numeric data comes from
/// the plan.
pub fn lower_mapped(
    plan: &ExecutionPlan,
    graph: &Graph,
    weights: &dyn WeightResolver,
    config: &RtlConfig,
) -> Result<RtlBundle, RtlError> {
    config.validate()?;
    if plan.mapping.is_none() {
        return Err(RtlError::UnsupportedLayout {
            detail: "the mapped backend needs a schema-2 plan with a hardware mapping; this plan \
                     carries none. Solve the layout with the mapping solver, or emit the schema-1 \
                     plan with the single-kernel / full-graph backend"
                .to_string(),
        });
    }
    if plan.engine.accumulator_bits != MAPPED_ACC_BITS {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "plan engine {} declares accumulator_bits = {}, but the emitted MAC array \
                 accumulates in {MAPPED_ACC_BITS} bits and the requant multiplier takes that same \
                 {MAPPED_ACC_BITS}-bit accumulator as its operand; the plan's datapath and the \
                 emitted one must agree",
                plan.engine.name, plan.engine.accumulator_bits
            ),
        });
    }
    let mut model = build_model(plan, graph, weights)?;
    model.frames = build_frames(&model, config);
    let expected = evaluate(&model)?;
    let provenance = plan_provenance(plan, &model)?;
    provenance.validate()?;

    let dut_name = format!("{}_mapped_v2", slug(&config.module_prefix));
    let tb_name = format!("{dut_name}_tb");
    let top_module = if config.emit_testbench {
        tb_name.clone()
    } else {
        dut_name.clone()
    };

    let mut files: Vec<RtlFile> = Vec::new();
    files.push(RtlFile {
        name: format!("{dut_name}.sv"),
        kind: RtlFileKind::SystemVerilog,
        contents: emit_dut(&model, &dut_name, &provenance, &config.note()),
    });
    if config.emit_testbench {
        files.push(RtlFile {
            name: format!("{tb_name}.sv"),
            kind: RtlFileKind::Testbench,
            contents: emit_testbench(&model, &dut_name, &tb_name, &provenance, &expected),
        });
        files.push(RtlFile {
            name: "run_verilator.sh".to_string(),
            kind: RtlFileKind::Script,
            contents: emit_run_script(&dut_name, &tb_name),
        });
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));

    let digests: Vec<super::FileDigest> = files
        .iter()
        .map(|f| super::FileDigest {
            name: f.name.clone(),
            kind: f.kind,
            bytes: f.contents.len(),
            fnv1a64: format!("{:016x}", fnv1a64(f.contents.as_bytes())),
        })
        .collect();
    let mut digest_input = String::new();
    for digest in &digests {
        let _ = writeln!(
            digest_input,
            "{} {} {}",
            digest.name, digest.bytes, digest.fnv1a64
        );
    }
    let bundle_digest = format!("{:016x}", fnv1a64(digest_input.as_bytes()));

    let summary = summarize(&model);
    let widest = model
        .engines
        .iter()
        .max_by_key(|engine| engine.lanes)
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: "a mapped plan must instantiate at least one engine".to_string(),
        })?;
    let metadata = RtlMetadata {
        format: RTL_BUNDLE_FORMAT.to_string(),
        schema_version: RTL_BUNDLE_SCHEMA_VERSION,
        kernel_id: MAPPED_KERNEL_ID.to_string(),
        provenance,
        plan: PlanContext {
            plan_format: plan.format.clone(),
            plan_schema_version: plan.schema_version,
            execution_mode: plan.execution_mode.as_str().to_string(),
            granularity: "frame".to_string(),
            engine: widest.name.clone(),
            step_index: 0,
            step_count: plan.steps.len() as u32,
            estimated_cycles: summary.makespan_cycles,
            spatial_in: [model.in_spatial.0, model.in_spatial.1],
            spatial_out: [model.out_spatial.0, model.out_spatial.1],
        },
        top_module,
        kernel_module: dut_name.clone(),
        layer: model
            .ops
            .iter()
            .map(|op| op.name.clone())
            .collect::<Vec<_>>()
            .join(","),
        tile: TileGeometry {
            simd: widest.simd,
            noc: widest.pe,
            k: 1,
            fold: 1,
        },
        in_channels: u64::from(model.in_channels),
        out_channels: u64::from(model.out_channels),
        accumulator_width: MAPPED_ACC_BITS,
        max_abs_accumulator: model.ops.iter().map(|op| op.max_abs_acc).max().unwrap_or(0),
        kernel_cycles_per_window: summary.makespan_cycles,
        requantization: Vec::new(),
        requant_multiplier_error_ppm: None,
        config: config.clone(),
        files: digests,
        supported_operators: vec![format!(
            "pointwise_dot: 1x1 conv, groups=1, stride=1, tap_parallel=1, zero padding, \
             i8xi8 symmetric, half-away-from-zero requantization [{MAPPED_KERNEL_ID}]"
        )],
        supported_interface: vec![
            "clk, rst (synchronous, active-high reset)".to_string(),
            "start / done: one frame per start pulse".to_string(),
            format!(
                "engines_busy [{}:0]: one busy bit per engine instance, so overlap between two \
                 engines is observable",
                model.engines.len().saturating_sub(1)
            ),
            "ld_en/ld_addr/ld_data: stimulus load port into the shared activation arena"
                .to_string(),
            "rd_addr/rd_data: combinational read port of the shared activation arena".to_string(),
        ],
        unsupported: unsupported_summary(),
        reuse_candidates: vec![
            "nngen/examples/st_stream/rtl/sts_lib.v: sts_rq (bit-identical fx requantization)"
                .to_string(),
            "nngen/examples/st_stream/rtl/sts_lib.v: sts_fifo (ready/valid FIFO between engine \
             instances, matching the emitted pipeline links)"
                .to_string(),
        ],
        reproduce: vec![
            format!(
                "emit: tessera RTL backend mapped (lower_mapped) -> {dut_name}.sv + {tb_name}.sv + \
                 rtl_bundle.json"
            ),
            format!(
                "simulate: verilator --binary --timing -Wno-fatal --top-module {tb_name} -o \
                 {tb_name}_sim {dut_name}.sv {tb_name}.sv"
            ),
            "verify: the testbench prints TESSA_RTL_PASS with mismatches=0".to_string(),
        ],
        bundle_digest,
        full_graph: None,
        mapped: Some(summary),
    };

    let metadata_json =
        serde_json::to_string_pretty(&metadata).map_err(|source| RtlError::Serialize {
            context: "serializing bundle metadata",
            source,
        })?;
    files.push(RtlFile {
        name: "rtl_bundle.json".to_string(),
        kind: RtlFileKind::Metadata,
        contents: metadata_json,
    });

    Ok(RtlBundle { metadata, files })
}

fn unsupported_summary() -> Vec<String> {
    vec![
        "operators other than a 1x1 pointwise convolution with stride [1,1] and groups 1"
            .to_string(),
        "kernel tap parallelism (tap_parallel != 1)".to_string(),
        "nonzero padding".to_string(),
        "activation or weight element types other than signed i8".to_string(),
        "requantization other than half-away-from-zero with M0 in [0, 2^17)".to_string(),
        "saturation bounds outside signed int8".to_string(),
        "unsigned output quantization".to_string(),
        "tensor residences other than Arena (shared arena) and Fifo (pipeline link)".to_string(),
        "more than one pipeline link in or out of a single engine, or a link whose producer and \
         consumer are the same engine"
            .to_string(),
        "a plan with no hardware mapping (schema 1), or with no single stimulus / comparison \
         tensor"
            .to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Resolved model
// ---------------------------------------------------------------------------

/// One scheduled operator, resolved into what the engine datapath needs.
struct MappedOp {
    /// Operator name.
    name: String,
    /// Engine instance index.
    engine: usize,
    /// Position within the engine's operator list.
    op_index: usize,
    /// Input channels.
    ic: u32,
    /// Output channels.
    oc: u32,
    /// Input height.
    ih: u32,
    /// Input width.
    iw: u32,
    /// Output height.
    oh: u32,
    /// Output width.
    ow: u32,
    /// Reduction beats (`ic / simd`).
    k: u32,
    /// Output-channel blocks (`oc / pe`).
    fold: u32,
    /// Weights, `(out_channel, in_channel)` row-major.
    weights: Vec<i8>,
    /// Bias per output channel.
    bias: Vec<i64>,
    /// Requantization multiplier per output channel.
    m0: Vec<i64>,
    /// Requantization shift.
    shift: u32,
    /// Clamp low.
    lo: i64,
    /// Clamp high.
    hi: i64,
    /// Worst-case `|accumulator|` over this operator's code range.
    max_abs_acc: u64,
    /// Tensor this operator reads.
    in_tensor: TensorId,
    /// Tensor this operator writes.
    out_tensor: TensorId,
    /// Arena word offset of the input tensor (0 when it arrives over a link).
    in_base: u64,
    /// Arena word offset of the output tensor (0 when it leaves over a link).
    out_base: u64,
    /// Words of the input tensor.
    in_words: u64,
    /// Words of the output tensor.
    out_words: u64,
    /// Pipeline link feeding this operator, when its input arrives over one.
    link_in: Option<usize>,
    /// Whether this operator drains that link (the first consumer does).
    drains: bool,
    /// Pipeline link this operator writes, when its output leaves over one.
    link_out: Option<usize>,
}

/// One engine instance and the operators bound to it.
struct MappedEngine {
    /// Instance index.
    index: usize,
    /// Instance name.
    name: String,
    /// Shared or dedicated.
    role: EngineRole,
    /// Input-channel lanes.
    simd: u32,
    /// Output-channel lanes.
    pe: u32,
    /// Kernel tap parallelism.
    tap: u32,
    /// Instantiated lanes.
    lanes: u64,
    /// Operator indices, in run order.
    ops: Vec<usize>,
    /// Link this engine produces.
    link_out: Option<usize>,
    /// Link this engine consumes.
    link_in: Option<usize>,
    /// Words of the produced link tensor.
    link_out_words: u64,
    /// Words of the consumed link tensor.
    link_in_words: u64,
    /// Weight ROM depth.
    weight_total: u64,
    /// Bias ROM depth.
    bias_total: u64,
    /// Multiplier ROM depth.
    m0_total: u64,
    /// Worst-case `|accumulator|` over the engine's operators.
    max_abs_acc: u64,
}

/// One pipeline cut between two engines.
struct MappedLink {
    /// Link index.
    index: usize,
    /// Producer engine index.
    producer: usize,
    /// Consumer engine index.
    consumer: usize,
    /// Tensor crossing the cut.
    tensor: TensorId,
    /// Tensor name.
    tensor_name: String,
    /// Words crossing.
    words: u64,
    /// FIFO depth in words.
    depth: u32,
}

/// One schedule step, with the steps it must wait for.
struct MappedStage {
    /// Operator index.
    op: usize,
    /// Engine index.
    engine: usize,
    /// Stages that must have finished before this one starts.
    barriers: Vec<usize>,
    /// Stages that must merely have started before this one starts (a pipeline
    /// link licenses overlap, and the FIFO carries the data).
    overlaps: Vec<usize>,
}

/// The whole resolved mapped structure.
struct MappedModel {
    /// Operators in schedule order.
    ops: Vec<MappedOp>,
    /// Engine instances.
    engines: Vec<MappedEngine>,
    /// Pipeline cuts.
    links: Vec<MappedLink>,
    /// Schedule steps.
    stages: Vec<MappedStage>,
    /// Words in the shared arena.
    arena_words: u64,
    /// Stimulus tensor.
    in_tensor: TensorId,
    /// Words of the stimulus tensor.
    in_words: u64,
    /// Arena offset of the stimulus tensor.
    in_base: u64,
    /// Channels of the stimulus tensor.
    in_channels: u32,
    /// Comparison tensor.
    out_tensor: TensorId,
    /// Words of the comparison tensor.
    out_words: u64,
    /// Arena offset of the comparison tensor.
    out_base: u64,
    /// Channels of the comparison tensor.
    out_channels: u32,
    /// Spatial extent of the stimulus.
    in_spatial: (u32, u32),
    /// Spatial extent of the comparison tensor.
    out_spatial: (u32, u32),
    /// Sum of the steps' own cycle estimates.
    schedule_cycles: u64,
    /// Critical path recorded by the plan.
    makespan_cycles: u64,
    /// Testbench stimulus, one frame of input codes per entry.
    frames: Vec<Vec<i64>>,
}

impl MappedModel {
    fn arena_address_bits(&self) -> u32 {
        bits_for(u32::try_from(self.arena_words.max(1)).unwrap_or(u32::MAX))
    }
}

// ---------------------------------------------------------------------------
// Model construction: fail closed on everything the emitted datapath cannot do
// ---------------------------------------------------------------------------

fn unsupported(layer: &str, detail: impl Into<String>) -> RtlError {
    RtlError::UnsupportedLayout {
        detail: format!("mapped backend: step {layer}: {}", detail.into()),
    }
}

fn build_model(
    plan: &ExecutionPlan,
    graph: &Graph,
    weights: &dyn WeightResolver,
) -> Result<MappedModel, RtlError> {
    let mapping = plan
        .mapping
        .as_ref()
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: "the mapped backend needs a schema-2 plan with a hardware mapping".to_string(),
        })?;
    if plan.steps.is_empty() {
        return Err(RtlError::UnsupportedLayout {
            detail: "the mapped backend needs at least one schedule step".to_string(),
        });
    }
    let engines = build_engines(mapping)?;
    let mut digest: Vec<u8> = Vec::new();

    // Addressable tensor residences. A pipeline link tensor has a Fifo
    // residence and is *not* placed in the arena; every other activation has an
    // Arena residence, including the graph input and the comparison tensor.
    let mut residences: BTreeMap<TensorId, &crate::layout::BufferResidence> = BTreeMap::new();
    for residence in &mapping.buffers {
        if residences.insert(residence.tensor, residence).is_some() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "tensor {} has more than one residence in the mapping",
                    residence.tensor
                ),
            });
        }
    }

    let mut links = build_links(mapping, &engines, &tensor_names(plan))?;

    // Bindings by operator name.
    let mut bindings: BTreeMap<&str, &crate::layout::OpEngineBinding> = BTreeMap::new();
    for binding in &mapping.bindings {
        if bindings.insert(binding.op.as_str(), binding).is_some() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!("operator {} has more than one binding", binding.op),
            });
        }
    }

    let mut ops: Vec<MappedOp> = Vec::with_capacity(plan.steps.len());
    for step in plan.steps.iter() {
        if graph.layer_named(&step.op).is_none() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "step {} names no layer of the supplied graph; the plan and the graph must \
                     describe the same network",
                    step.op
                ),
            });
        }
        let binding = bindings
            .get(step.op.as_str())
            .ok_or_else(|| unsupported(&step.op, "the mapping has no binding for this operator"))?;
        let engine_index = engines
            .iter()
            .position(|engine| engine.name == binding.engine)
            .ok_or_else(|| {
                unsupported(
                    &step.op,
                    format!(
                        "binding names engine {:?}, which the mapping does not instantiate",
                        binding.engine
                    ),
                )
            })?;
        if step.engine != binding.engine {
            return Err(unsupported(
                &step.op,
                format!(
                    "step.engine {:?} and binding.engine {:?} disagree",
                    step.engine, binding.engine
                ),
            ));
        }
        let engine = &engines[engine_index];
        let op = build_op(
            step,
            binding,
            engine,
            weights,
            &residences,
            &links,
            &mut digest,
        )?;
        ops.push(op);
    }

    // Engine operator lists, in schedule order, and each operator's position
    // inside its engine (which is what the engine's `op_sel` port selects).
    let mut engines = engines;
    for index in 0..ops.len() {
        let engine_index = ops[index].engine;
        let position = engines[engine_index].ops.len();
        engines[engine_index].ops.push(index);
        ops[index].op_index = position;
    }
    for (index, engine) in engines.iter_mut().enumerate() {
        if engine.ops.is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} is instantiated but no schedule step is bound to it",
                    index
                ),
            });
        }
        engine.weight_total = engine
            .ops
            .iter()
            .map(|op| ops[*op].weights.len() as u64)
            .sum();
        engine.bias_total = engine.ops.iter().map(|op| ops[*op].bias.len() as u64).sum();
        engine.m0_total = engine.ops.iter().map(|op| ops[*op].m0.len() as u64).sum();
    }

    // Resolve the pipeline ends to concrete engines and operator positions.
    for (index, op) in ops.iter_mut().enumerate() {
        if let Some(link) = op.link_in {
            if op.engine != links[link].consumer {
                return Err(unsupported(
                    &op.name,
                    format!(
                        "pipeline link {} enters engine {} but this step runs on engine {}",
                        link, links[link].consumer, op.engine
                    ),
                ));
            }
        }
        if let Some(link) = op.link_out {
            if op.engine != links[link].producer {
                return Err(unsupported(
                    &op.name,
                    format!(
                        "pipeline link {} leaves engine {} but this step runs on engine {}",
                        link, links[link].producer, op.engine
                    ),
                ));
            }
        }
        let _ = index;
    }
    // The producer's tensor and the consumer's tensor are the same tensor, so
    // their word counts must agree; the FIFO carries exactly that many words,
    // and the summary reports the producer step's output element count.
    for index in 0..links.len() {
        let link = &links[index];
        let producer_words: Vec<u64> = ops
            .iter()
            .filter(|op| op.link_out == Some(link.index))
            .map(|op| op.out_words)
            .collect();
        let consumer_words: Vec<u64> = ops
            .iter()
            .filter(|op| op.link_in == Some(link.index))
            .map(|op| op.in_words)
            .collect();
        if producer_words.is_empty() || consumer_words.is_empty() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "pipeline link {} on tensor {} has no producer or no consumer step",
                    link.index, link.tensor_name
                ),
            });
        }
        if producer_words[0] != consumer_words[0] {
            return Err(RtlError::ShapeMismatch {
                layer: link.tensor_name.clone(),
                what: "pipeline tensor words".to_string(),
                expected: producer_words[0],
                actual: consumer_words[0],
            });
        }
        links[index].words = producer_words[0];
    }

    // Engines that produce or consume a link, and the operator position of the
    // drain (the first consumer in schedule order).
    let mut engines = engines;
    for link in &links {
        if engines[link.producer].link_out.is_some() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} produces more than one pipeline tensor; the emitted engine has one \
                     stream-out port",
                    engines[link.producer].name
                ),
            });
        }
        if engines[link.consumer].link_in.is_some() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} consumes more than one pipeline tensor; the emitted engine has one \
                     stream-in port",
                    engines[link.consumer].name
                ),
            });
        }
        engines[link.producer].link_out = Some(link.index);
        engines[link.consumer].link_in = Some(link.index);
        engines[link.producer].link_out_words = link.words;
        engines[link.consumer].link_in_words = link.words;
        // Exactly the first consumer drains; later consumers on the same engine
        // read the link RAM the drain filled.
        let mut consumers: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.link_in == Some(link.index))
            .map(|(index, _)| index)
            .collect();
        consumers.sort_unstable();
        for (position, consumer) in consumers.iter().enumerate() {
            ops[*consumer].drains = position == 0;
        }
    }

    for engine in engines.iter_mut() {
        engine.max_abs_acc = engine
            .ops
            .iter()
            .map(|op| ops[*op].max_abs_acc)
            .max()
            .unwrap_or(0);
    }

    // Stimulus and comparison tensors: produced by no step / consumed by no
    // step. Exactly one of each, so the generated testbench has one load stream
    // and one comparison stream.
    let produced: Vec<TensorId> = ops.iter().map(|op| op.out_tensor).collect();
    let consumed: Vec<TensorId> = ops.iter().map(|op| op.in_tensor).collect();
    let mut inputs: Vec<TensorId> = Vec::new();
    for op in &ops {
        if !produced.contains(&op.in_tensor) && !inputs.contains(&op.in_tensor) {
            inputs.push(op.in_tensor);
        }
    }
    let mut outputs: Vec<TensorId> = Vec::new();
    for op in ops.iter().rev() {
        if !consumed.contains(&op.out_tensor) && !outputs.contains(&op.out_tensor) {
            outputs.push(op.out_tensor);
            break;
        }
    }
    if inputs.len() != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "the mapped backend emits one stimulus stream, but the plan has {} graph input \
                 tensors",
                inputs.len()
            ),
        });
    }
    if outputs.len() != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: "the mapped backend emits one comparison stream, but the plan has no final \
                     tensor that no step consumes"
                .to_string(),
        });
    }
    let in_tensor = inputs[0];
    let out_tensor = outputs[0];
    let input_op = ops
        .iter()
        .find(|op| op.in_tensor == in_tensor)
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: "the stimulus tensor is not read by any step".to_string(),
        })?;
    let output_op = ops
        .iter()
        .find(|op| op.out_tensor == out_tensor)
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: "the comparison tensor is not written by any step".to_string(),
        })?;
    let (in_words, in_base, in_channels, in_spatial) = (
        input_op.in_words,
        input_op.in_base,
        input_op.ic,
        (input_op.ih, input_op.iw),
    );
    let (out_words, out_base, out_channels, out_spatial) = (
        output_op.out_words,
        output_op.out_base,
        output_op.oc,
        (output_op.oh, output_op.ow),
    );

    // Stages: a step waits for the producers of its inputs, unless a pipeline
    // link licenses overlap.
    let drainer: BTreeMap<usize, usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.drains)
        .filter_map(|(stage, op)| op.link_in.map(|link| (link, stage)))
        .collect();
    let mut stages = Vec::with_capacity(ops.len());
    for (stage, op) in ops.iter().enumerate() {
        let mut barriers = Vec::new();
        let mut overlaps = Vec::new();
        for (other_stage, other) in ops.iter().enumerate() {
            if other_stage == stage {
                continue;
            }
            if other.out_tensor != op.in_tensor {
                continue;
            }
            if op.link_in.is_some() && op.drains && other.link_out == op.link_in {
                overlaps.push(other_stage);
            } else {
                barriers.push(other_stage);
            }
        }
        // A later consumer on the same engine re-reads the link RAM the first
        // consumer filled, so it must wait for that drain to have happened.
        if let Some(link) = op.link_in {
            if !op.drains {
                if let Some(drain_stage) = drainer.get(&link) {
                    if *drain_stage != stage && !barriers.contains(drain_stage) {
                        barriers.push(*drain_stage);
                    }
                }
            }
        }
        barriers.sort_unstable();
        stages.push(MappedStage {
            op: stage,
            engine: op.engine,
            barriers,
            overlaps,
        });
    }

    // Arena size: the high-water word offset over all Arena residences.
    let arena_words = mapping
        .buffers
        .iter()
        .filter(|residence| residence.kind == BufferKind::Arena)
        .map(|residence| residence.offset_words.saturating_add(residence.words))
        .max()
        .unwrap_or(0)
        .max(in_words.max(out_words));

    let schedule_cycles = plan
        .steps
        .iter()
        .map(|step| step.estimated_cycles)
        .sum::<u64>();

    Ok(MappedModel {
        ops,
        engines,
        links,
        stages,
        arena_words,
        in_tensor,
        in_words,
        in_base,
        in_channels,
        out_tensor,
        out_words,
        out_base,
        out_channels,
        in_spatial,
        out_spatial,
        schedule_cycles,
        makespan_cycles: mapping.makespan_cycles,
        frames: Vec::new(),
    })
}

fn build_engines(mapping: &HardwareMapping) -> Result<Vec<MappedEngine>, RtlError> {
    if mapping.engines.is_empty() {
        return Err(RtlError::UnsupportedLayout {
            detail: "a mapped plan must instantiate at least one engine".to_string(),
        });
    }
    let mut engines = Vec::with_capacity(mapping.engines.len());
    let mut seen: Vec<&str> = Vec::new();
    for (index, instance) in mapping.engines.iter().enumerate() {
        if seen.contains(&instance.name.as_str()) {
            return Err(RtlError::UnsupportedLayout {
                detail: format!("engine name {} is not unique", instance.name),
            });
        }
        seen.push(instance.name.as_str());
        if instance.simd == 0 || instance.pe == 0 {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} declares zero lanes (simd {}, pe {})",
                    instance.name, instance.simd, instance.pe
                ),
            });
        }
        if instance.tap_parallel != 1 {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} declares tap_parallel {}; the emitted pointwise array has one tap, \
                     so the planner must assign tap_parallel = 1",
                    instance.name, instance.tap_parallel
                ),
            });
        }
        if instance.accumulator_bits != MAPPED_ACC_BITS {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} declares accumulator_bits = {}, but the emitted MAC array \
                     accumulates in {MAPPED_ACC_BITS} bits and the requant multiplier takes that \
                     same {MAPPED_ACC_BITS}-bit accumulator as its operand; the plan's datapath \
                     and the emitted one must agree",
                    instance.name, instance.accumulator_bits
                ),
            });
        }
        if instance.simd.saturating_mul(instance.pe) > 64 {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "engine {} declares {} lanes; the emitted unrolled multiplier array is limited \
                     to 64 lanes per instance",
                    instance.name,
                    instance.simd.saturating_mul(instance.pe)
                ),
            });
        }
        engines.push(MappedEngine {
            index,
            name: instance.name.clone(),
            role: instance.role,
            simd: instance.simd,
            pe: instance.pe,
            tap: instance.tap_parallel,
            lanes: instance.lanes(),
            ops: Vec::new(),
            link_out: None,
            link_in: None,
            link_out_words: 0,
            link_in_words: 0,
            weight_total: 0,
            bias_total: 0,
            m0_total: 0,
            max_abs_acc: 0,
        });
    }
    Ok(engines)
}

/// Names of the plan's tensors, by id.
///
/// The plan's tensor table is the authority on a tensor's name: a summary that
/// named a pipeline tensor by its numeric id would tell the reader nothing
/// about which tensor crosses the cut.
fn tensor_names(plan: &ExecutionPlan) -> BTreeMap<TensorId, &str> {
    plan.tensors
        .iter()
        .map(|tensor| (tensor.id, tensor.name.as_str()))
        .collect()
}

fn build_links(
    mapping: &HardwareMapping,
    engines: &[MappedEngine],
    tensors: &BTreeMap<TensorId, &str>,
) -> Result<Vec<MappedLink>, RtlError> {
    let mut links = Vec::with_capacity(mapping.pipeline.len());
    for (index, link) in mapping.pipeline.iter().enumerate() {
        let producer = engines
            .iter()
            .position(|engine| engine.name == link.producer_engine)
            .ok_or_else(|| RtlError::UnsupportedLayout {
                detail: format!(
                    "pipeline link {index} names producer engine {}, which the mapping does not \
                     instantiate",
                    link.producer_engine
                ),
            })?;
        let consumer = engines
            .iter()
            .position(|engine| engine.name == link.consumer_engine)
            .ok_or_else(|| RtlError::UnsupportedLayout {
                detail: format!(
                    "pipeline link {index} names consumer engine {}, which the mapping does not \
                     instantiate",
                    link.consumer_engine
                ),
            })?;
        if producer == consumer {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "pipeline link {index} connects engine {} to itself; a pipeline cut must \
                     cross between two engine instances",
                    link.producer_engine
                ),
            });
        }
        if link.depth_elements == 0 {
            return Err(RtlError::UnsupportedLayout {
                detail: format!("pipeline link {index} has depth_elements 0"),
            });
        }
        // The plan's tensor table is the authority on the tensor's name: the
        // numeric id is a fallback for a plan that does not carry the tensor.
        let tensor_name = tensors
            .get(&link.tensor)
            .map(|name| (*name).to_string())
            .unwrap_or_else(|| link.tensor.to_string());
        links.push(MappedLink {
            index,
            producer,
            consumer,
            tensor: link.tensor,
            tensor_name,
            words: 0,
            depth: link.depth_elements,
        });
    }
    Ok(links)
}

#[allow(clippy::too_many_arguments)]
fn build_op(
    step: &ScheduleStep,
    binding: &crate::layout::OpEngineBinding,
    engine: &MappedEngine,
    weights: &dyn WeightResolver,
    residences: &BTreeMap<TensorId, &crate::layout::BufferResidence>,
    links: &[MappedLink],
    digest: &mut Vec<u8>,
) -> Result<MappedOp, RtlError> {
    let layer = step.op.clone();

    match step.op_kind {
        OpKind::PointwiseConv2d => {}
        OpKind::Conv2d if step.kernel == [1, 1] && step.groups == 1 => {}
        other => {
            return Err(RtlError::UnsupportedOperator {
                layer,
                op: super::full::planned_op(other),
                supported: "pointwise_dot: PointwiseConv2d, or Conv2d with kernel [1,1], \
                            groups=1 and stride [1,1]",
            });
        }
    }
    if step.kernel != [1, 1] {
        return Err(unsupported(
            &layer,
            format!("kernel {:?} is not a 1x1 pointwise kernel", step.kernel),
        ));
    }
    if step.stride != [1, 1] {
        return Err(unsupported(
            &layer,
            format!(
                "stride {:?} is not [1, 1]; the emitted array computes one output pixel per tap \
                 sweep and implements no strided addressing",
                step.stride
            ),
        ));
    }
    if step.groups != 1 {
        return Err(unsupported(
            &layer,
            format!("groups {} is not 1", step.groups),
        ));
    }
    if step.tap_parallel != 1 || binding.tap_parallel != 1 || engine.tap != 1 {
        return Err(unsupported(
            &layer,
            format!(
                "tap_parallel is {} in the step, {} in the binding and {} on the engine; a 1x1 \
                 kernel has a single tap and the emitted array instantiates one",
                step.tap_parallel, binding.tap_parallel, engine.tap
            ),
        ));
    }
    if step.simd != binding.simd || step.pe != binding.pe {
        return Err(unsupported(
            &layer,
            format!(
                "the step folds simd {} x pe {} but its binding folds simd {} x pe {}",
                step.simd, step.pe, binding.simd, binding.pe
            ),
        ));
    }
    if step.simd != engine.simd || step.pe != engine.pe {
        return Err(unsupported(
            &layer,
            format!(
                "the step folds {} lanes but engine {} instantiates {}; all operators on one \
                 engine share that engine's shape",
                binding.lanes(),
                engine.name,
                engine.lanes
            ),
        ));
    }
    if step.cost.lanes != binding.lanes() {
        return Err(unsupported(
            &layer,
            format!(
                "step.cost.lanes is {} but the binding instantiates {} lanes",
                step.cost.lanes,
                binding.lanes()
            ),
        ));
    }
    if let Some(padding) = &step.padding {
        if padding.top != 0
            || padding.bottom != 0
            || padding.left != 0
            || padding.right != 0
            || padding.reflect
        {
            return Err(unsupported(
                &layer,
                "the step pads its input; the emitted pointwise datapath reads the input plane at \
                 the output coordinate and implements no border rule",
            ));
        }
    }
    if step.in_spatial != step.out_spatial {
        return Err(unsupported(
            &layer,
            format!(
                "input spatial {:?} differs from output spatial {:?}; a pointwise 1x1 stride-1 \
                 kernel preserves the spatial extent",
                step.in_spatial, step.out_spatial
            ),
        ));
    }
    if step.inputs.len() != 1 || step.outputs.len() != 1 {
        return Err(unsupported(
            &layer,
            format!(
                "the step has {} inputs and {} outputs; the emitted operator consumes exactly one \
                 activation tensor and produces exactly one",
                step.inputs.len(),
                step.outputs.len()
            ),
        ));
    }
    if step.simd == 0 || step.in_channels % step.simd != 0 {
        return Err(unsupported(
            &layer,
            format!(
                "in_channels {} is not divisible by simd {}",
                step.in_channels, step.simd
            ),
        ));
    }
    if step.out_channels % step.pe != 0 {
        return Err(unsupported(
            &layer,
            format!(
                "out_channels {} is not divisible by pe {}",
                step.out_channels, step.pe
            ),
        ));
    }

    let weight_ref = step.weights.as_ref().ok_or_else(|| {
        unsupported(
            &layer,
            "a pointwise step without a weight tensor is malformed",
        )
    })?;
    let expected_weights = u64::from(step.out_channels) * u64::from(step.in_channels);
    if weight_ref.shape.n != step.out_channels || weight_ref.shape.c != step.in_channels {
        return Err(RtlError::ShapeMismatch {
            layer,
            what: format!(
                "weight tensor {} shape (n=out_channels, c=in_channels)",
                weight_ref.name
            ),
            expected: expected_weights,
            actual: u64::from(weight_ref.shape.n) * u64::from(weight_ref.shape.c),
        });
    }
    if weight_ref.elem != ElemType::I8 {
        return Err(unsupported(
            &layer,
            format!(
                "weight tensor {} has element type {:?}; the emitted multiplier lanes take signed \
                 i8 weights",
                weight_ref.name, weight_ref.elem
            ),
        ));
    }
    let raw = weights.tensor_bytes(weight_ref)?;
    digest.extend_from_slice(&raw);
    if raw.len() as u64 != expected_weights {
        return Err(RtlError::ShapeMismatch {
            layer,
            what: format!("weight tensor {} bytes", weight_ref.name),
            expected: expected_weights,
            actual: raw.len() as u64,
        });
    }
    let weights_i8: Vec<i8> = raw.iter().map(|byte| *byte as i8).collect();

    // Optional bias. The plan's bias tensor is i32 codes and the emitted ROM is
    // as wide as the accumulator ([`MAPPED_ACC_BITS`] bits), so every bias code
    // is held exactly and the accumulation stays inside the declared width.
    let mut bias = vec![0i64; step.out_channels as usize];
    if let Some(bias_ref) = &step.bias {
        if bias_ref.elem != ElemType::I32 {
            return Err(unsupported(
                &layer,
                format!(
                    "bias tensor {} has element type {:?}; the emitted bias ROM holds i32 codes",
                    bias_ref.name, bias_ref.elem
                ),
            ));
        }
        let bytes = weights.tensor_bytes(bias_ref)?;
        digest.extend_from_slice(&bytes);
        if bytes.len() as u64 != u64::from(step.out_channels) * 4 {
            return Err(RtlError::ShapeMismatch {
                layer,
                what: format!("bias tensor {} bytes", bias_ref.name),
                expected: u64::from(step.out_channels) * 4,
                actual: bytes.len() as u64,
            });
        }
        for (channel, chunk) in bytes.chunks_exact(4).enumerate() {
            bias[channel] = i64::from(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
    }

    let (m0, shift, lo, hi) = requant_of(step)?;
    let m0 = if m0.len() == 1 {
        vec![m0[0]; step.out_channels as usize]
    } else {
        m0
    };

    // Worst-case accumulator magnitude: the clamp bounds are already inside
    // int8, so the input codes are bounded by 127.
    let input_bound = 127i64;
    let mut max_abs_acc = bias.iter().map(|value| value.abs()).max().unwrap_or(0);
    for (channel, channel_bias) in bias.iter().enumerate() {
        let mut sum = channel_bias.abs();
        for ic in 0..step.in_channels as usize {
            sum += i64::from(weights_i8[channel * step.in_channels as usize + ic].unsigned_abs())
                * input_bound;
        }
        max_abs_acc = max_abs_acc.max(sum);
    }

    let in_tensor = step.inputs[0];
    let out_tensor = step.outputs[0];
    let in_words =
        u64::from(step.in_channels) * u64::from(step.in_spatial[0]) * u64::from(step.in_spatial[1]);
    let out_words = u64::from(step.out_channels)
        * u64::from(step.out_spatial[0])
        * u64::from(step.out_spatial[1]);

    let link_in = links
        .iter()
        .find(|link| link.tensor == in_tensor)
        .map(|link| link.index);
    let link_out = links
        .iter()
        .find(|link| link.tensor == out_tensor)
        .map(|link| link.index);

    let (in_base, out_base) = (
        residence_of(&layer, in_tensor, link_in.is_some(), in_words, residences)?,
        residence_of(
            &layer,
            out_tensor,
            link_out.is_some(),
            out_words,
            residences,
        )?,
    );

    Ok(MappedOp {
        name: layer,
        engine: engine.index,
        op_index: 0,
        ic: step.in_channels,
        oc: step.out_channels,
        ih: step.in_spatial[0],
        iw: step.in_spatial[1],
        oh: step.out_spatial[0],
        ow: step.out_spatial[1],
        k: step.in_channels / step.simd,
        fold: step.out_channels / step.pe,
        weights: weights_i8,
        bias,
        m0,
        shift,
        lo,
        hi,
        max_abs_acc: max_abs_acc as u64,
        in_tensor,
        out_tensor,
        in_base,
        out_base,
        in_words,
        out_words,
        link_in,
        drains: false,
        link_out,
    })
}

/// Arena word offset of one tensor, or 0 when the tensor crosses a pipeline.
fn residence_of(
    layer: &str,
    tensor: TensorId,
    pipelined: bool,
    words: u64,
    residences: &BTreeMap<TensorId, &crate::layout::BufferResidence>,
) -> Result<u64, RtlError> {
    let residence = residences.get(&tensor).ok_or_else(|| {
        unsupported(
            layer,
            format!("the mapping gives tensor {} no residence", tensor),
        )
    })?;
    if residence.words != words {
        return Err(RtlError::ShapeMismatch {
            layer: layer.to_string(),
            what: format!("residence words of tensor {tensor}"),
            expected: words,
            actual: residence.words,
        });
    }
    match residence.kind {
        BufferKind::Fifo if pipelined => Ok(0),
        BufferKind::Fifo => Err(unsupported(
            layer,
            format!(
                "tensor {} is placed in a FIFO but no pipeline link crosses it",
                tensor
            ),
        )),
        BufferKind::Arena if pipelined => Err(unsupported(
            layer,
            format!(
                "tensor {} crosses a pipeline link and must not also be placed in the arena",
                tensor
            ),
        )),
        BufferKind::Arena => Ok(residence.offset_words),
        BufferKind::Register => Err(unsupported(
            layer,
            format!(
                "tensor {} is register-resident; the emitted organization reads and writes every \
                 activation through the shared arena or a pipeline FIFO",
                tensor
            ),
        )),
    }
}

/// The step's requantization, in the emitted datapath's form.
fn requant_of(step: &ScheduleStep) -> Result<(Vec<i64>, u32, i64, i64), RtlError> {
    let layer = step.op.clone();
    let requant: &Requant = step.requant.as_ref().ok_or_else(|| {
        RtlError::UnsupportedQuantization {
            layer: layer.clone(),
            detail: format!(
                "step has no requantization contract (`Requant: None`); {MAPPED_KERNEL_ID} never \
                 infers one"
            ),
        }
    })?;
    if requant.rounding != Rounding::HalfAwayFromZero {
        return Err(RtlError::UnsupportedQuantization {
            layer,
            detail: format!(
                "rounding {:?} is not reproducible by the emitted datapath; supported: \
                 HalfAwayFromZero",
                requant.rounding
            ),
        });
    }
    if !(0..=REQUANT_MAX_MULTIPLIER).contains(&requant.m0) {
        return Err(RtlError::UnsupportedQuantization {
            layer,
            detail: format!(
                "m0 {} does not fit the 17-bit multiplier ROM (representable range [0, {}])",
                requant.m0, REQUANT_MAX_MULTIPLIER
            ),
        });
    }
    if !(0..=31).contains(&requant.shift) {
        return Err(RtlError::UnsupportedQuantization {
            layer,
            detail: format!(
                "shift {} is outside the emitted range 0..=31",
                requant.shift
            ),
        });
    }
    let channels = step.out_channels as usize;
    if !requant.per_channel_multipliers.is_empty()
        && requant.per_channel_multipliers.len() != channels
    {
        return Err(RtlError::ShapeMismatch {
            layer,
            what: "per-channel requantization multipliers".to_string(),
            expected: channels as u64,
            actual: requant.per_channel_multipliers.len() as u64,
        });
    }
    for (index, multiplier) in requant.per_channel_multipliers.iter().enumerate() {
        if !(0..=REQUANT_MAX_MULTIPLIER).contains(multiplier) {
            return Err(RtlError::UnsupportedQuantization {
                layer,
                detail: format!(
                    "channel {index}: multiplier {multiplier} does not fit the 17-bit multiplier \
                     ROM (representable range [0, {}])",
                    REQUANT_MAX_MULTIPLIER
                ),
            });
        }
    }
    let (mut lo, mut hi) = step.activation.effective_clamp(Some(requant));
    match step.activation.kind {
        ActivationKind::None => {}
        ActivationKind::Relu => lo = lo.max(0),
        ActivationKind::Relu6 => {
            lo = lo.max(0);
            hi = hi.min(6);
        }
        ActivationKind::UnsignedClamp => {
            return Err(RtlError::UnsupportedQuantization {
                layer,
                detail: "activation kind UnsignedClamp produces an unsigned output range, which \
                         the signed int8 datapath cannot represent"
                    .to_string(),
            });
        }
    }
    if lo > hi {
        return Err(RtlError::UnsupportedQuantization {
            layer,
            detail: format!(
                "the requantization saturate range and the activation clamp do not intersect \
                 (lo {lo} > hi {hi})"
            ),
        });
    }
    if lo < -128 || hi > 127 {
        return Err(RtlError::UnsupportedQuantization {
            layer,
            detail: format!(
                "clamp range [{lo}, {hi}] does not fit the signed int8 activation stream the \
                 emitted arena carries; the planner must emit QuantizeOutput for wider ranges"
            ),
        });
    }
    let multipliers = if requant.per_channel_multipliers.is_empty() {
        vec![requant.m0]
    } else {
        requant.per_channel_multipliers.clone()
    };
    Ok((
        multipliers,
        requant.shift as u32,
        i64::from(lo),
        i64::from(hi),
    ))
}

fn plan_provenance(plan: &ExecutionPlan, model: &MappedModel) -> Result<Provenance, RtlError> {
    let source = &plan.provenance;
    let mut digest_input: Vec<u8> = Vec::new();
    for op in &model.ops {
        digest_input.extend_from_slice(&op.weights.iter().map(|w| *w as u8).collect::<Vec<u8>>());
        for value in &op.bias {
            digest_input.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(Provenance {
        compiler_version: source.compiler_version.clone(),
        git_revision: source
            .options
            .get("git_revision")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        dirty: source
            .options
            .get("dirty")
            .map(|v| v == "true")
            .unwrap_or(false),
        graph_schema_version: source
            .options
            .get("graph_schema_version")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        plan_schema_version: plan.schema_version,
        target: source.target_name.clone(),
        target_revision: source.target_revision.clone(),
        input_manifest_sha256: source.graph_sha256.clone(),
        weights_sha256: format!("fnv1a64:{:016x}", fnv1a64(&digest_input)),
        planner_config_sha256: source.policy_sha256.clone(),
    })
}

fn summarize(model: &MappedModel) -> MappedSummary {
    let mut notes = Vec::new();
    notes.push(format!(
        "shared activation arena: {} words x {MAPPED_WORD_BITS} bits, addressed by the mapping's \
         word offsets",
        model.arena_words
    ));
    if model.links.is_empty() {
        notes.push(
            "no pipeline link: every consumer engine starts only after its producer engine has \
             finished, so the engine arrays never overlap"
                .to_string(),
        );
    } else {
        notes.push(format!(
            "{} pipeline link(s): the consumer starts one cycle after the producer starts and \
             receives its words through the ready/valid FIFO, so both engines are busy in the same \
             cycle",
            model.links.len()
        ));
    }
    if model.makespan_cycles < model.schedule_cycles {
        notes.push(format!(
            "plan makespan {} cycles against {} cycles of serial step estimates",
            model.makespan_cycles, model.schedule_cycles
        ));
    }
    let engines = model
        .engines
        .iter()
        .map(|engine| MappedEngineSummary {
            name: engine.name.clone(),
            role: engine.role.as_str().to_string(),
            simd: engine.simd,
            pe: engine.pe,
            tap_parallel: engine.tap,
            lanes: engine.lanes,
            operators: engine
                .ops
                .iter()
                .map(|op| model.ops[*op].name.clone())
                .collect(),
        })
        .collect();
    let links = model
        .links
        .iter()
        .map(|link| MappedLinkSummary {
            producer_engine: model.engines[link.producer].name.clone(),
            consumer_engine: model.engines[link.consumer].name.clone(),
            tensor: link.tensor_name.clone(),
            words: link.words,
            depth_elements: link.depth,
        })
        .collect();
    MappedSummary {
        engines,
        links,
        arena_words: model.arena_words,
        arena_address_bits: model.arena_address_bits(),
        multiplier_lanes: model.engines.iter().map(|engine| engine.lanes).sum(),
        pipelined: !model.links.is_empty(),
        makespan_cycles: model.makespan_cycles,
        schedule_cycles: model.schedule_cycles,
        notes,
    }
}

// ---------------------------------------------------------------------------
// Testbench stimulus and the Rust reference evaluation
// ---------------------------------------------------------------------------

/// The exact integer datapath of the emitted RTL, evaluated in Rust.
///
/// Deliberately a second implementation, written from the resolved model rather
/// than from the emitted text: the testbench's expected codes must be evidence
/// about the RTL, not a copy of it.
fn evaluate_pointwise(op: &MappedOp, input: &[i64]) -> Vec<i64> {
    let ic = op.ic as usize;
    let ih = op.ih as usize;
    let iw = op.iw as usize;
    let oc = op.oc as usize;
    let mut out = vec![0i64; oc * ih * iw];
    for channel in 0..oc {
        for y in 0..ih {
            for x in 0..iw {
                let mut acc = op.bias[channel];
                for input_channel in 0..ic {
                    let code = input[(input_channel * ih + y) * iw + x];
                    acc += code * i64::from(op.weights[channel * ic + input_channel]);
                }
                let multiplier = op.m0[channel];
                let product = acc * multiplier;
                let rounding = if op.shift == 0 {
                    0
                } else if product >= 0 {
                    1i64 << (op.shift - 1)
                } else {
                    (1i64 << (op.shift - 1)) - 1
                };
                let value = (product + rounding) >> op.shift;
                out[(channel * ih + y) * iw + x] = value.clamp(op.lo, op.hi);
            }
        }
    }
    out
}

fn build_frames(model: &MappedModel, config: &RtlConfig) -> Vec<Vec<i64>> {
    let mut generator = super::Xorshift64::new(config.fixture_seed);
    let mut frames = Vec::with_capacity(config.fixture_windows);
    for _ in 0..config.fixture_windows {
        let frame: Vec<i64> = (0..model.in_words)
            .map(|_| i64::from(generator.next_i8()).clamp(-127, 127))
            .collect();
        frames.push(frame);
    }
    frames
}

/// Expected codes of the comparison tensor, one frame at a time.
fn evaluate(model: &MappedModel) -> Result<Vec<Vec<i64>>, RtlError> {
    let mut out = Vec::with_capacity(model.frames.len());
    for frame in &model.frames {
        let mut values: BTreeMap<TensorId, Vec<i64>> = BTreeMap::new();
        values.insert(model.in_tensor, frame.clone());
        for op in &model.ops {
            let input = values.get(&op.in_tensor).ok_or_else(|| {
                unsupported(
                    &op.name,
                    format!(
                        "the reference evaluation has no value for input tensor {}",
                        op.in_tensor
                    ),
                )
            })?;
            let result = evaluate_pointwise(op, input);
            values.insert(op.out_tensor, result);
        }
        let final_values =
            values
                .get(&model.out_tensor)
                .ok_or_else(|| RtlError::UnsupportedLayout {
                    detail: format!(
                        "the reference evaluation never produced tensor {}",
                        model.out_tensor
                    ),
                })?;
        out.push(final_values.clone());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

fn header(provenance: &Provenance, note: &str, what: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated by tessera {MAPPED_KERNEL_ID}. Do not edit by hand."
    );
    let _ = writeln!(
        out,
        "// {what}; one module per engine instance over a shared activation arena."
    );
    let _ = writeln!(
        out,
        "// compiler_version: {}  git_revision: {}  dirty: {}",
        provenance.compiler_version, provenance.git_revision, provenance.dirty
    );
    let _ = writeln!(
        out,
        "// target: {} ({})  plan_schema_version: {}  graph_schema_version: {}",
        provenance.target,
        provenance.target_revision,
        provenance.plan_schema_version,
        provenance.graph_schema_version
    );
    let _ = writeln!(
        out,
        "// input_manifest_sha256: {}  weights_digest: {}  planner_config_sha256: {}",
        provenance.input_manifest_sha256,
        provenance.weights_sha256,
        provenance.planner_config_sha256
    );
    let _ = writeln!(out, "// config: {note}");
    out
}

fn sv_signed(bits: u32, value: i64) -> String {
    if value < 0 {
        format!("-{bits}'sd{}", value.unsigned_abs())
    } else {
        format!("{bits}'sd{value}")
    }
}

fn emit_dut(model: &MappedModel, dut_name: &str, provenance: &Provenance, note: &str) -> String {
    let mut s = header(provenance, note, "mapped multi-engine organization");
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s);
    s.push_str(&emit_mul_lane(model, dut_name));
    s.push_str(&emit_fifo(model, dut_name));
    for (index, engine) in model.engines.iter().enumerate() {
        s.push_str(&emit_engine(model, engine, index, dut_name));
    }
    s.push_str(&emit_top(model, dut_name));
    let _ = writeln!(s, "`default_nettype wire");
    s
}

fn mul_lane_module(model: &MappedModel, dut_name: &str) -> String {
    let _ = model;
    format!("{dut_name}_mul_lane")
}

fn fifo_module(model: &MappedModel, dut_name: &str) -> String {
    let _ = model;
    format!("{dut_name}_fifo")
}

fn engine_module(model: &MappedModel, index: usize, dut_name: &str) -> String {
    let engine = &model.engines[index];
    format!("{dut_name}_engine{index}_{}", slug(&engine.name))
}

fn top_module(model: &MappedModel, dut_name: &str) -> String {
    let _ = model;
    dut_name.to_string()
}

/// One multiplier lane. Instantiated once per `simd x pe` lane, so the
/// multiplier count of an engine is visible in the elaborated hierarchy.
fn emit_mul_lane(_model: &MappedModel, dut_name: &str) -> String {
    let name = mul_lane_module(_model, dut_name);
    let mut s = String::new();
    let _ = writeln!(s, "// One signed i8 x i8 multiplier lane.");
    let _ = writeln!(s, "module {name} (");
    let _ = writeln!(s, "    input  logic signed [15:0] a,");
    let _ = writeln!(s, "    input  logic signed [7:0]  b,");
    let _ = writeln!(s, "    output logic signed [23:0] p");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    assign p = a * b;");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s);
    s
}

/// The ready/valid FIFO of one pipeline link.
fn emit_fifo(model: &MappedModel, dut_name: &str) -> String {
    let name = fifo_module(model, dut_name);
    let max_depth = model.links.iter().map(|link| link.depth).max().unwrap_or(1);
    let depth_bits = bits_for(max_depth);
    let mut s = String::new();
    let _ = writeln!(s, "// Ready/valid FIFO between two engine instances.");
    let _ = writeln!(s, "module {name} #(");
    let _ = writeln!(s, "    parameter int DEPTH = {max_depth}");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  logic                clk,");
    let _ = writeln!(s, "    input  logic                rst,");
    let _ = writeln!(s, "    input  logic                push,");
    let _ = writeln!(s, "    input  logic signed [15:0]  din,");
    let _ = writeln!(s, "    output logic                full,");
    let _ = writeln!(s, "    input  logic                pop,");
    let _ = writeln!(s, "    output logic signed [15:0]  dout,");
    let _ = writeln!(s, "    output logic                empty");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    localparam int PTRW = {depth_bits};");
    let _ = writeln!(s, "    logic signed [15:0] mem [0:DEPTH-1];");
    let _ = writeln!(s, "    logic [PTRW-1:0] wptr;");
    let _ = writeln!(s, "    logic [PTRW-1:0] rptr;");
    let _ = writeln!(s, "    logic [PTRW:0]   count;");
    let _ = writeln!(s, "    assign empty = (count == 0);");
    let _ = writeln!(s, "    assign full  = (count == DEPTH[PTRW:0]);");
    let _ = writeln!(s, "    assign dout  = mem[rptr];");
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (rst) begin");
    let _ = writeln!(s, "            wptr <= '0; rptr <= '0; count <= '0;");
    let _ = writeln!(s, "        end else begin");
    let _ = writeln!(s, "            if (push && !full) begin");
    let _ = writeln!(s, "                mem[wptr] <= din;");
    let _ = writeln!(s, "                wptr <= wptr + PTRW'(1);");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "            if (pop && !empty) begin");
    let _ = writeln!(s, "                rptr <= rptr + PTRW'(1);");
    let _ = writeln!(s, "            end");
    let _ = writeln!(
        s,
        "            count <= count + (push && !full) - (pop && !empty);"
    );
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s);
    s
}

/// One engine instance: the MAC array, its ROMs, its sequencer and its
/// optional link RAMs.
fn emit_engine(model: &MappedModel, engine: &MappedEngine, index: usize, dut_name: &str) -> String {
    let name = engine_module(model, index, dut_name);
    let mul = mul_lane_module(model, dut_name);
    let aw = model.arena_address_bits();
    let simd = engine.simd as usize;
    let pe = engine.pe as usize;
    let op_bits = bits_for(engine.ops.len() as u32);
    let mut s = String::new();

    let _ = writeln!(
        s,
        "// Engine instance {} ({}, {} lanes = simd {} x pe {} x tap {}).",
        index,
        engine.role.as_str(),
        engine.lanes,
        engine.simd,
        engine.pe,
        engine.tap
    );
    let _ = writeln!(
        s,
        "// Instantiates {} multiplier lane instances ({mul}).",
        engine.lanes
    );
    let _ = writeln!(s, "module {name} #(");
    let _ = writeln!(s, "    parameter int AW = {aw}");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  logic                clk,");
    let _ = writeln!(s, "    input  logic                rst,");
    let _ = writeln!(s, "    input  logic                start,");
    let _ = writeln!(s, "    input  logic [{op_bits}-1:0] op_sel,");
    let _ = writeln!(s, "    output logic                busy,");
    let _ = writeln!(s, "    output logic                done,");
    for lane in 0..simd {
        let _ = writeln!(s, "    output logic [AW-1:0]       ar_addr_{lane},");
        let _ = writeln!(s, "    input  logic signed [15:0]  ar_data_{lane},");
    }
    let _ = writeln!(s, "    output logic                aw_en,");
    let _ = writeln!(s, "    output logic [AW-1:0]       aw_addr,");
    let _ = writeln!(s, "    output logic signed [15:0]  aw_data,");
    let _ = writeln!(s, "    input  logic                li_valid,");
    let _ = writeln!(s, "    output logic                li_ready,");
    let _ = writeln!(s, "    input  logic signed [15:0]  li_data,");
    let _ = writeln!(s, "    output logic                lo_valid,");
    let _ = writeln!(s, "    input  logic                lo_ready,");
    let _ = writeln!(s, "    output logic signed [15:0]  lo_data");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    // ---- geometry ----");
    let _ = writeln!(s, "    localparam int SIMD = {simd};");
    let _ = writeln!(s, "    localparam int PE   = {pe};");
    let _ = writeln!(s, "    localparam int NOP  = {};", engine.ops.len());
    let _ = writeln!(
        s,
        "    localparam int WTOT = {};",
        engine
            .ops
            .iter()
            .map(|op| model.ops[*op].weights.len() as u64)
            .sum::<u64>()
            .max(1)
    );
    let _ = writeln!(
        s,
        "    localparam int BTOT = {};",
        engine
            .ops
            .iter()
            .map(|op| model.ops[*op].bias.len() as u64)
            .sum::<u64>()
            .max(1)
    );
    let _ = writeln!(
        s,
        "    localparam int MTOT = {};",
        engine
            .ops
            .iter()
            .map(|op| model.ops[*op].m0.len() as u64)
            .sum::<u64>()
            .max(1)
    );
    let weight_total: u64 = engine
        .ops
        .iter()
        .map(|op| model.ops[*op].weights.len() as u64)
        .sum();
    let bias_total: u64 = engine
        .ops
        .iter()
        .map(|op| model.ops[*op].bias.len() as u64)
        .sum();
    let m0_total: u64 = engine
        .ops
        .iter()
        .map(|op| model.ops[*op].m0.len() as u64)
        .sum();

    // ---- state machine ----
    let _ = writeln!(s, "    // ---- sequencer ----");
    let _ = writeln!(s, "    localparam logic [2:0] S_IDLE  = 3'd0;");
    let _ = writeln!(s, "    localparam logic [2:0] S_DESC  = 3'd1;");
    let _ = writeln!(s, "    localparam logic [2:0] S_DRAIN = 3'd2;");
    let _ = writeln!(s, "    localparam logic [2:0] S_RUN   = 3'd3;");
    let _ = writeln!(s, "    localparam logic [2:0] S_WRITE = 3'd4;");
    let _ = writeln!(s, "    localparam logic [2:0] S_FLUSH = 3'd5;");
    let _ = writeln!(s, "    localparam logic [2:0] S_DONE  = 3'd6;");
    let _ = writeln!(s, "    logic [2:0] state;");
    let _ = writeln!(s, "    logic done_r;");
    let _ = writeln!(
        s,
        "    logic [31:0] d_ic, d_oc, d_ih, d_iw, d_plane, d_fold, d_k;"
    );
    let _ = writeln!(
        s,
        "    logic [31:0] d_in_base, d_out_base, d_in_words, d_out_words, d_wbase, d_bbase, \
         d_m0base;"
    );
    let _ = writeln!(s, "    logic signed [63:0] d_lo, d_hi;");
    let _ = writeln!(s, "    logic [4:0]  d_shift;");
    let _ = writeln!(s, "    logic        d_link_in, d_drain, d_link_out;");
    let _ = writeln!(
        s,
        "    logic [31:0] ocb, yy, xx, kb, pe_i, drain_i, flush_i;"
    );
    let _ = writeln!(s, "    logic signed [{MAPPED_ACC_MSB}:0] acc [0:PE-1];");
    let _ = writeln!(s);
    let _ = writeln!(s, "    // ---- constant contents ----");
    let _ = writeln!(s, "    logic signed [7:0]  w_rom [0:WTOT-1];");
    let _ = writeln!(s, "    logic signed [{MAPPED_ACC_MSB}:0] b_rom [0:BTOT-1];");
    let _ = writeln!(s, "    logic [16:0]        m0_rom [0:MTOT-1];");
    let _ = writeln!(s, "    initial begin");
    let mut weight_index = 0usize;
    for op in &engine.ops {
        let op = &model.ops[*op];
        let _ = writeln!(
            s,
            "        // op {} ({}): weights (out_channel, in_channel), row-major",
            op.op_index, op.name
        );
        for weight in &op.weights {
            let _ = writeln!(
                s,
                "        w_rom[{weight_index}] = {};",
                sv_signed(8, i64::from(*weight))
            );
            weight_index += 1;
        }
    }
    let mut bias_index = 0usize;
    for op in &engine.ops {
        let op = &model.ops[*op];
        for value in &op.bias {
            let _ = writeln!(
                s,
                "        b_rom[{bias_index}] = {};",
                sv_signed(MAPPED_ACC_BITS, *value)
            );
            bias_index += 1;
        }
    }
    let mut m0_index = 0usize;
    for op in &engine.ops {
        let op = &model.ops[*op];
        for value in &op.m0 {
            let _ = writeln!(
                s,
                "        m0_rom[{m0_index}] = 17'd{};",
                (*value).unsigned_abs()
            );
            m0_index += 1;
        }
    }
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // ROM depths emitted above: {weight_total} weights, {bias_total} biases, {m0_total} \
         multipliers."
    );

    // ---- multiplier lanes ----
    let _ = writeln!(
        s,
        "    // A lane product needs 24 bits (signed i8 x i8 with room to spare); the"
    );
    let _ = writeln!(
        s,
        "    // accumulator is {MAPPED_ACC_BITS}-bit, so every product is sign-extended explicitly."
    );
    let _ = writeln!(
        s,
        "    function automatic logic signed [{MAPPED_ACC_MSB}:0] sext_acc(input logic signed \
         [23:0] v);"
    );
    s.push_str(&format!(
        "        begin\n            sext_acc = {{{{8{{v[23]}}}}, v}};\n        end\n"
    ));
    let _ = writeln!(s, "    endfunction");
    let _ = writeln!(
        s,
        "    // ---- multiplier lanes ({} x {} = {}) ----",
        simd,
        pe,
        simd * pe
    );
    for lane in 0..simd {
        let _ = writeln!(s, "    logic signed [15:0] act_lane_{lane};");
    }
    for gp in 0..pe {
        for gs in 0..simd {
            let _ = writeln!(s, "    logic signed [7:0]  lane_wt_p{gp}_s{gs};");
            let _ = writeln!(s, "    logic signed [23:0] lane_prod_p{gp}_s{gs};");
        }
    }
    for lane in 0..simd {
        let address = if engine.link_in.is_some() {
            format!("link_in_ram_0[(kb*SIMD + {lane})*d_plane + yy*d_iw + xx]")
        } else {
            format!("16'sd0")
        };
        if engine.link_in.is_some() {
            let _ = writeln!(
                s,
                "    assign act_lane_{lane} = d_link_in ? {address} : ar_data_{lane};"
            );
        } else {
            let _ = writeln!(s, "    assign act_lane_{lane} = ar_data_{lane};");
        }
    }
    for gp in 0..pe {
        for gs in 0..simd {
            let _ = writeln!(
                s,
                "    assign lane_wt_p{gp}_s{gs} = w_rom[d_wbase + (ocb*PE + {gp})*d_ic + kb*SIMD + \
                 {gs}];"
            );
            let _ = writeln!(
                s,
                "    {mul} u_lane_p{gp}_s{gs} (.a(act_lane_{gs}), .b(lane_wt_p{gp}_s{gs}), \
                 .p(lane_prod_p{gp}_s{gs}));"
            );
        }
    }
    let _ = writeln!(s);

    // ---- link RAMs ----
    if engine.link_in.is_some() {
        let _ = writeln!(
            s,
            "    logic signed [15:0] link_in_ram_0 [0:{}];",
            engine.link_in_words.max(1) - 1
        );
    }
    if engine.link_out.is_some() {
        let _ = writeln!(
            s,
            "    logic signed [15:0] link_out_ram_0 [0:{}];",
            engine.link_out_words.max(1) - 1
        );
    }

    // ---- requantization ----
    let _ = writeln!(
        s,
        "    // ---- requantization: (acc*M0 + round) >>> shift, clamp ----"
    );
    let _ = writeln!(
        s,
        "    // The multiplier's accumulator operand is the {MAPPED_ACC_BITS}-bit `acc` register \
         (the width the plan declares); M0 < 2^17, so the exact product fits 64 signed bits."
    );
    let _ = writeln!(s, "    logic signed [63:0] rq_product [0:PE-1];");
    let _ = writeln!(s, "    logic signed [63:0] rq_rounded [0:PE-1];");
    let _ = writeln!(s, "    logic signed [63:0] rq_value [0:PE-1];");
    let _ = writeln!(s, "    logic signed [15:0] rq_word [0:PE-1];");
    let _ = writeln!(s, "    always_comb begin");
    let _ = writeln!(
        s,
        "        for (int lane = 0; lane < PE; lane = lane + 1) begin"
    );
    let _ = writeln!(
        s,
        "            rq_product[lane] = acc[lane] * $signed({{1'b0, m0_rom[d_m0base + ocb*PE + \
         lane]}});"
    );
    let _ = writeln!(
        s,
        "            rq_rounded[lane] = rq_product[lane] + ((d_shift == 5'd0) ? 64'sd0 : \
         ((rq_product[lane] >= 0) ? (64'sd1 << (d_shift - 5'd1)) : ((64'sd1 << (d_shift - 5'd1)) \
         - 64'sd1)));"
    );
    let _ = writeln!(
        s,
        "            rq_value[lane] = rq_rounded[lane] >>> d_shift;"
    );
    let _ = writeln!(s, "            if (rq_value[lane] > d_hi) begin");
    let _ = writeln!(s, "                rq_word[lane] = d_hi[15:0];");
    let _ = writeln!(s, "            end else if (rq_value[lane] < d_lo) begin");
    let _ = writeln!(s, "                rq_word[lane] = d_lo[15:0];");
    let _ = writeln!(s, "            end else begin");
    let _ = writeln!(s, "                rq_word[lane] = rq_value[lane][15:0];");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);

    // ---- outputs ----
    let _ = writeln!(s, "    assign busy = (state != S_IDLE);");
    let _ = writeln!(s, "    assign done = done_r;");
    let _ = writeln!(s, "    assign aw_en   = (state == S_WRITE) && !d_link_out;");
    let _ = writeln!(
        s,
        "    assign aw_addr = AW'(d_out_base + (ocb*PE + pe_i)*d_plane + yy*d_iw + xx);"
    );
    let _ = writeln!(s, "    assign aw_data = rq_word[pe_i];");
    if engine.link_in.is_some() {
        let _ = writeln!(s, "    assign li_ready = (state == S_DRAIN) && d_drain;");
    } else {
        let _ = writeln!(s, "    assign li_ready = 1'b0;");
    }
    if engine.link_out.is_some() {
        let _ = writeln!(s, "    assign lo_valid = (state == S_FLUSH) && d_link_out;");
        let _ = writeln!(s, "    assign lo_data  = link_out_ram_0[flush_i];");
    } else {
        let _ = writeln!(s, "    assign lo_valid = 1'b0;");
        let _ = writeln!(s, "    assign lo_data  = 16'sd0;");
    }
    for lane in 0..simd {
        let _ = writeln!(
            s,
            "    assign ar_addr_{lane} = AW'(d_in_base + (kb*SIMD + {lane})*d_plane + yy*d_iw + \
             xx);"
        );
    }
    let _ = writeln!(s);

    // ---- FSM ----
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (rst) begin");
    let _ = writeln!(s, "            state <= S_IDLE;");
    let _ = writeln!(s, "            done_r <= 1'b0;");
    let _ = writeln!(
        s,
        "            kb <= 32'd0; ocb <= 32'd0; yy <= 32'd0; xx <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "            pe_i <= 32'd0; drain_i <= 32'd0; flush_i <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "            d_ic <= 32'd0; d_oc <= 32'd0; d_ih <= 32'd0; d_iw <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "            d_plane <= 32'd0; d_fold <= 32'd0; d_k <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "            d_in_base <= 32'd0; d_out_base <= 32'd0; d_in_words <= 32'd0; d_out_words <= \
         32'd0;"
    );
    let _ = writeln!(
        s,
        "            d_wbase <= 32'd0; d_bbase <= 32'd0; d_m0base <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "            d_lo <= 64'sd0; d_hi <= 64'sd0; d_shift <= 5'd0;"
    );
    let _ = writeln!(
        s,
        "            d_link_in <= 1'b0; d_drain <= 1'b0; d_link_out <= 1'b0;"
    );
    for gp in 0..pe {
        let _ = writeln!(s, "            acc[{gp}] <= {MAPPED_ACC_BITS}'sd0;");
    }
    let _ = writeln!(s, "        end else begin");
    let _ = writeln!(s, "            done_r <= 1'b0;");
    let _ = writeln!(s, "            case (state)");
    let _ = writeln!(s, "                S_IDLE: begin");
    let _ = writeln!(s, "                    if (start) begin");
    let _ = writeln!(s, "                        state <= S_DESC;");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");

    // descriptor
    let _ = writeln!(s, "                S_DESC: begin");
    let _ = writeln!(s, "                    case (op_sel)");
    for (position, op_index) in engine.ops.iter().enumerate() {
        let op = &model.ops[*op_index];
        let wbase = weight_base(model, engine, position);
        let bbase = bias_base(model, engine, position);
        let m0base = m0_base(model, engine, position);
        let _ = writeln!(s, "                        {op_bits}'d{position}: begin");
        let _ = writeln!(
            s,
            "                            d_ic <= 32'd{}; d_oc <= 32'd{}; d_ih <= 32'd{}; d_iw <= \
             32'd{};",
            op.ic, op.oc, op.ih, op.iw
        );
        let _ = writeln!(
            s,
            "                            d_plane <= 32'd{}; d_fold <= 32'd{}; d_k <= 32'd{};",
            u64::from(op.oh) * u64::from(op.ow),
            op.fold,
            op.k
        );
        let _ = writeln!(
            s,
            "                            d_in_base <= 32'd{}; d_out_base <= 32'd{}; d_in_words <= \
             32'd{}; d_out_words <= 32'd{};",
            op.in_base, op.out_base, op.in_words, op.out_words
        );
        let _ = writeln!(
            s,
            "                            d_wbase <= 32'd{wbase}; d_bbase <= 32'd{bbase}; d_m0base \
             <= 32'd{m0base};"
        );
        let _ = writeln!(
            s,
            "                            d_lo <= {}; d_hi <= {}; d_shift <= 5'd{};",
            sv_signed(64, op.lo),
            sv_signed(64, op.hi),
            op.shift
        );
        let _ = writeln!(
            s,
            "                            d_link_in <= 1'b{}; d_drain <= 1'b{}; d_link_out <= \
             1'b{};",
            u8::from(op.link_in.is_some()),
            u8::from(op.drains),
            u8::from(op.link_out.is_some())
        );
        let _ = writeln!(s, "                        end");
    }
    let _ = writeln!(s, "                        default: ;");
    let _ = writeln!(s, "                    endcase");
    // The operator that takes its input over a pipeline link drains the FIFO
    // into its local RAM before it computes; every other operator starts
    // straight in the reduction state.
    let drains_positions: Vec<usize> = engine
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| model.ops[**op].drains)
        .map(|(position, _)| position)
        .collect();
    if drains_positions.is_empty() {
        let _ = writeln!(s, "                    state <= S_RUN;");
    } else {
        let conditions = drains_positions
            .iter()
            .map(|position| format!("(op_sel == {op_bits}'d{position})"))
            .collect::<Vec<_>>()
            .join(" || ");
        let _ = writeln!(s, "                    if ({conditions}) begin");
        let _ = writeln!(s, "                        state <= S_DRAIN;");
        let _ = writeln!(s, "                    end else begin");
        let _ = writeln!(s, "                        state <= S_RUN;");
        let _ = writeln!(s, "                    end");
    }
    let _ = writeln!(
        s,
        "                    ocb <= 32'd0; yy <= 32'd0; xx <= 32'd0; kb <= 32'd0;"
    );
    let _ = writeln!(
        s,
        "                    pe_i <= 32'd0; drain_i <= 32'd0; flush_i <= 32'd0;"
    );
    for gp in 0..pe {
        let _ = writeln!(s, "                    acc[{gp}] <= {MAPPED_ACC_BITS}'sd0;");
    }
    let _ = writeln!(s, "                end");

    // drain
    let _ = writeln!(s, "                S_DRAIN: begin");
    let _ = writeln!(s, "                    if (li_valid) begin");
    if engine.link_in.is_some() {
        let _ = writeln!(
            s,
            "                        link_in_ram_0[drain_i] <= li_data;"
        );
    }
    let _ = writeln!(s, "                        drain_i <= drain_i + 32'd1;");
    let _ = writeln!(
        s,
        "                        if (drain_i + 32'd1 == d_in_words) begin"
    );
    let _ = writeln!(s, "                            drain_i <= 32'd0;");
    let _ = writeln!(s, "                            state <= S_RUN;");
    let _ = writeln!(s, "                        end");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");

    // run
    let _ = writeln!(s, "                S_RUN: begin");
    for gp in 0..pe {
        let lane_sum: Vec<String> = (0..simd)
            .map(|gs| format!("sext_acc(lane_prod_p{gp}_s{gs})"))
            .collect();
        let sum = lane_sum.join(" + ");
        let _ = writeln!(s, "                    if (kb == 32'd0) begin");
        let _ = writeln!(
            s,
            "                        acc[{gp}] <= b_rom[d_bbase + ocb*PE + {gp}] + ({sum});"
        );
        let _ = writeln!(s, "                    end else begin");
        let _ = writeln!(
            s,
            "                        acc[{gp}] <= acc[{gp}] + ({sum});"
        );
        let _ = writeln!(s, "                    end");
    }
    let _ = writeln!(s, "                    if (kb + 32'd1 >= d_k) begin");
    let _ = writeln!(s, "                        kb <= 32'd0;");
    let _ = writeln!(s, "                        state <= S_WRITE;");
    let _ = writeln!(s, "                    end else begin");
    let _ = writeln!(s, "                        kb <= kb + 32'd1;");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");

    // write
    let _ = writeln!(s, "                S_WRITE: begin");
    if engine.link_out.is_some() {
        let _ = writeln!(
            s,
            "                    if (d_link_out) link_out_ram_0[(ocb*PE + pe_i)*d_plane + yy*d_iw \
             + xx] <= rq_word[pe_i];"
        );
    }
    // The write phase produces one output word per cycle. `pe_i` walks the
    // engine's output-channel lanes; once the lanes of one `(oc block, y, x)`
    // position are written the position advances and the engine returns to
    // S_RUN for the next accumulation. Only the last position of the last
    // channel block retires the operator.
    let _ = writeln!(s, "                    pe_i <= pe_i + 32'd1;");
    let _ = writeln!(s, "                    if (pe_i + 32'd1 >= 32'd{pe}) begin");
    let _ = writeln!(s, "                        pe_i <= 32'd0;");
    let _ = writeln!(s, "                        if (xx + 32'd1 >= d_iw) begin");
    let _ = writeln!(s, "                            xx <= 32'd0;");
    let _ = writeln!(
        s,
        "                            if (yy + 32'd1 >= d_ih) begin"
    );
    let _ = writeln!(s, "                                yy <= 32'd0;");
    let _ = writeln!(s, "                                ocb <= ocb + 32'd1;");
    let _ = writeln!(
        s,
        "                                if (ocb + 32'd1 >= d_fold) begin"
    );
    if engine.link_out.is_some() {
        let _ = writeln!(s, "                                    ocb <= 32'd0;");
        let _ = writeln!(
            s,
            "                                    state <= d_link_out ? S_FLUSH : S_DONE;"
        );
    } else {
        let _ = writeln!(s, "                                    state <= S_DONE;");
    }
    let _ = writeln!(s, "                                end else begin");
    let _ = writeln!(s, "                                    state <= S_RUN;");
    let _ = writeln!(s, "                                end");
    let _ = writeln!(s, "                            end else begin");
    let _ = writeln!(s, "                                yy <= yy + 32'd1;");
    let _ = writeln!(s, "                                state <= S_RUN;");
    let _ = writeln!(s, "                            end");
    let _ = writeln!(s, "                        end else begin");
    let _ = writeln!(s, "                            xx <= xx + 32'd1;");
    let _ = writeln!(s, "                            state <= S_RUN;");
    let _ = writeln!(s, "                        end");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");

    // flush
    if engine.link_out.is_some() {
        let _ = writeln!(s, "                S_FLUSH: begin");
        let _ = writeln!(s, "                    if (lo_ready) begin");
        let _ = writeln!(s, "                        flush_i <= flush_i + 32'd1;");
        let _ = writeln!(
            s,
            "                        if (flush_i + 32'd1 >= d_out_words) begin"
        );
        let _ = writeln!(s, "                            flush_i <= 32'd0;");
        let _ = writeln!(s, "                            state <= S_DONE;");
        let _ = writeln!(s, "                        end");
        let _ = writeln!(s, "                    end");
        let _ = writeln!(s, "                end");
    }

    // done
    let _ = writeln!(s, "                S_DONE: begin");
    let _ = writeln!(s, "                    done_r <= 1'b1;");
    let _ = writeln!(s, "                    state <= S_IDLE;");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                default: state <= S_IDLE;");
    let _ = writeln!(s, "            endcase");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s);
    s
}

fn weight_base(model: &MappedModel, engine: &MappedEngine, position: usize) -> u64 {
    engine.ops[..position]
        .iter()
        .map(|op| model.ops[*op].weights.len() as u64)
        .sum()
}

fn bias_base(model: &MappedModel, engine: &MappedEngine, position: usize) -> u64 {
    engine.ops[..position]
        .iter()
        .map(|op| model.ops[*op].bias.len() as u64)
        .sum()
}

fn m0_base(model: &MappedModel, engine: &MappedEngine, position: usize) -> u64 {
    engine.ops[..position]
        .iter()
        .map(|op| model.ops[*op].m0.len() as u64)
        .sum()
}

fn emit_top(model: &MappedModel, dut_name: &str) -> String {
    let name = top_module(model, dut_name);
    let fifo = fifo_module(model, dut_name);
    let aw = model.arena_address_bits();
    let neng = model.engines.len();
    let nlink = model.links.len();
    let nstg = model.stages.len();
    // One more code than there are stages: `cur` counts up to NSTG to retire
    // the last stage.
    let stg_bits = bits_for((nstg + 1) as u32);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "// Mapped top: one shared activation arena, {neng} engine instance(s), {nlink} pipeline \
         link(s)."
    );
    let _ = writeln!(s, "module {name} #(");
    let _ = writeln!(s, "    parameter int AW = {aw}");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  logic                clk,");
    let _ = writeln!(s, "    input  logic                rst,");
    let _ = writeln!(s, "    input  logic                start,");
    let _ = writeln!(s, "    output logic                done,");
    let _ = writeln!(s, "    output logic [{neng}-1:0] engines_busy,");
    let _ = writeln!(s, "    input  logic                ld_en,");
    let _ = writeln!(s, "    input  logic [AW-1:0]       ld_addr,");
    let _ = writeln!(s, "    input  logic signed [15:0]  ld_data,");
    let _ = writeln!(s, "    input  logic [AW-1:0]       rd_addr,");
    let _ = writeln!(s, "    output logic signed [15:0]  rd_data");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    localparam int NW = {};", model.arena_words.max(1));
    let _ = writeln!(s, "    // ---- shared activation arena ----");
    let _ = writeln!(s, "    logic signed [15:0] arena [0:NW-1];");
    let _ = writeln!(s, "    assign rd_data = arena[rd_addr];");
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (ld_en) arena[ld_addr] <= ld_data;");
    for index in 0..neng {
        let _ = writeln!(
            s,
            "        if (e{index}_aw_en) arena[e{index}_aw_addr] <= e{index}_aw_data;"
        );
    }
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);

    for (index, engine) in model.engines.iter().enumerate() {
        let module = engine_module(model, index, dut_name);
        let simd = engine.simd as usize;
        let op_bits = bits_for(engine.ops.len() as u32);
        let _ = writeln!(s, "    // ---- engine {index}: {} ----", engine.name);
        let _ = writeln!(s, "    logic e{index}_start;");
        let _ = writeln!(s, "    logic [{op_bits}-1:0] e{index}_opsel;");
        let _ = writeln!(s, "    logic e{index}_busy;");
        let _ = writeln!(s, "    logic e{index}_done;");
        for lane in 0..simd {
            let _ = writeln!(s, "    logic [AW-1:0] e{index}_ar_addr_{lane};");
            let _ = writeln!(s, "    logic signed [15:0] e{index}_ar_data_{lane};");
        }
        let _ = writeln!(s, "    logic e{index}_aw_en;");
        let _ = writeln!(s, "    logic [AW-1:0] e{index}_aw_addr;");
        let _ = writeln!(s, "    logic signed [15:0] e{index}_aw_data;");
        let _ = writeln!(s, "    logic e{index}_li_valid;");
        let _ = writeln!(s, "    logic e{index}_li_ready;");
        let _ = writeln!(s, "    logic signed [15:0] e{index}_li_data;");
        let _ = writeln!(s, "    logic e{index}_lo_valid;");
        let _ = writeln!(s, "    logic e{index}_lo_ready;");
        let _ = writeln!(s, "    logic signed [15:0] e{index}_lo_data;");
        let _ = writeln!(s, "    {module} u_engine{index} (");
        let _ = writeln!(s, "        .clk(clk), .rst(rst), .start(e{index}_start),");
        let _ = writeln!(s, "        .op_sel(e{index}_opsel),");
        let _ = writeln!(s, "        .busy(e{index}_busy), .done(e{index}_done),");
        for lane in 0..simd {
            let _ = writeln!(
                s,
                "        .ar_addr_{lane}(e{index}_ar_addr_{lane}), .ar_data_{lane}(e{index}_ar_data_{lane}),"
            );
        }
        let _ = writeln!(
            s,
            "        .aw_en(e{index}_aw_en), .aw_addr(e{index}_aw_addr), .aw_data(e{index}_aw_data),"
        );
        let _ = writeln!(
            s,
            "        .li_valid(e{index}_li_valid), .li_ready(e{index}_li_ready), .li_data(e{index}_li_data),"
        );
        let _ = writeln!(
            s,
            "        .lo_valid(e{index}_lo_valid), .lo_ready(e{index}_lo_ready), .lo_data(e{index}_lo_data));"
        );
        for lane in 0..simd {
            let _ = writeln!(
                s,
                "    assign e{index}_ar_data_{lane} = arena[e{index}_ar_addr_{lane}];"
            );
        }
        let _ = writeln!(s);
    }

    for (index, link) in model.links.iter().enumerate() {
        let producer = link.producer;
        let consumer = link.consumer;
        let _ = writeln!(
            s,
            "    // ---- pipeline link {index}: {} -> {} ({} words, depth {}) ----",
            model.engines[producer].name,
            model.engines[consumer].name,
            link.tensor_name,
            link.depth
        );
        let _ = writeln!(s, "    logic l{index}_empty;");
        let _ = writeln!(s, "    logic l{index}_full;");
        let _ = writeln!(s, "    {fifo} #(.DEPTH({})) u_fifo{index} (", link.depth);
        let _ = writeln!(s, "        .clk(clk), .rst(rst),");
        let _ = writeln!(
            s,
            "        .push(e{producer}_lo_valid), .din(e{producer}_lo_data),"
        );
        let _ = writeln!(s, "        .full(l{index}_full),");
        let _ = writeln!(
            s,
            "        .pop(e{consumer}_li_valid && e{consumer}_li_ready), .dout(e{consumer}_li_data),"
        );
        let _ = writeln!(s, "        .empty(l{index}_empty));");
        let _ = writeln!(s, "    assign e{producer}_lo_ready = !l{index}_full;");
        let _ = writeln!(s, "    assign e{consumer}_li_valid = !l{index}_empty;");
        let _ = writeln!(s);
    }
    // Engines without a link tie their stream ports off explicitly.
    for (index, engine) in model.engines.iter().enumerate() {
        if engine.link_in.is_none() {
            let _ = writeln!(s, "    assign e{index}_li_valid = 1'b0;");
            let _ = writeln!(s, "    assign e{index}_li_data = 16'sd0;");
        }
        if engine.link_out.is_none() {
            let _ = writeln!(s, "    assign e{index}_lo_ready = 1'b1;");
        }
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "    // ---- sequencer ----");
    let _ = writeln!(s, "    localparam int NSTG = {nstg};");
    let _ = writeln!(s, "    logic [NSTG-1:0] st_started;");
    let _ = writeln!(s, "    logic [NSTG-1:0] st_done;");
    let _ = writeln!(s, "    logic [NSTG-1:0] st_hit;");
    let _ = writeln!(s, "    logic [{stg_bits}-1:0] cur;");
    let _ = writeln!(s, "    logic run;");
    for index in 0..neng {
        let _ = writeln!(s, "    logic [{stg_bits}-1:0] stage_of{index};");
    }
    let _ = writeln!(
        s,
        "    assign engines_busy = {{{}}};",
        (0..neng)
            .rev()
            .map(|index| format!("e{index}_busy"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // `done` reads the latched per-stage completion bits, not `run`: `run` is
    // cleared the cycle the last stage finishes, and a one-cycle `done` pulse
    // would be a race for the testbench.
    let _ = writeln!(s, "    assign done = (&st_done);");
    // A stage's done bit is set only while its own engine reports done for that
    // stage, so a later stage on the same engine can never inherit an earlier
    // stage's completion.
    let hits: Vec<String> = model
        .stages
        .iter()
        .enumerate()
        .map(|(stage, info)| {
            format!(
                "(e{}_done && (stage_of{} == {stg_bits}'d{stage}))",
                info.engine, info.engine
            )
        })
        .collect();
    let _ = writeln!(
        s,
        "    assign st_hit = {{{}}};",
        hits.iter().rev().cloned().collect::<Vec<_>>().join(", ")
    );
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        if (rst) begin");
    let _ = writeln!(
        s,
        "            cur <= '0; run <= 1'b0; st_started <= '0; st_done <= '0;"
    );
    for index in 0..neng {
        let _ = writeln!(
            s,
            "            e{index}_start <= 1'b0; e{index}_opsel <= '0;"
        );
        let _ = writeln!(s, "            stage_of{index} <= '0;");
    }
    let _ = writeln!(s, "        end else begin");
    for index in 0..neng {
        let _ = writeln!(s, "            e{index}_start <= 1'b0;");
    }
    let _ = writeln!(s, "            st_done <= st_done | st_hit;");
    let _ = writeln!(s, "            if (start) begin");
    let _ = writeln!(
        s,
        "                cur <= '0; run <= 1'b1; st_started <= '0; st_done <= '0;"
    );
    for index in 0..neng {
        let _ = writeln!(s, "                stage_of{index} <= '0;");
    }
    let _ = writeln!(s, "            end else if (run) begin");
    let _ = writeln!(s, "                if (&(st_done | st_hit)) begin");
    let _ = writeln!(s, "                    run <= 1'b0;");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                case (cur)");
    for (stage, info) in model.stages.iter().enumerate() {
        let op = &model.ops[info.op];
        let engine = info.engine;
        let mut conditions = vec![format!("!e{engine}_busy")];
        for barrier in &info.barriers {
            conditions.push(format!("st_done[{barrier}]"));
        }
        for other in &info.overlaps {
            conditions.push(format!("st_started[{other}]"));
        }
        let condition = conditions.join(" && ");
        let _ = writeln!(
            s,
            "                    // stage {stage}: {} on engine {engine} (op {}), wait for {}",
            op.name,
            op.op_index,
            if info.barriers.is_empty() && info.overlaps.is_empty() {
                "nothing".to_string()
            } else {
                format!("{:?}/{:?}", info.barriers, info.overlaps)
            }
        );
        let _ = writeln!(s, "                    {stg_bits}'d{stage}: begin");
        let _ = writeln!(s, "                        if ({condition}) begin");
        let _ = writeln!(s, "                            e{engine}_start <= 1'b1;");
        let _ = writeln!(
            s,
            "                            e{engine}_opsel <= {}'d{};",
            bits_for(model.engines[engine].ops.len() as u32),
            op.op_index
        );
        let _ = writeln!(
            s,
            "                            st_started[{stage}] <= 1'b1;"
        );
        let _ = writeln!(
            s,
            "                            stage_of{engine} <= {stg_bits}'d{stage};"
        );
        let _ = writeln!(
            s,
            "                            cur <= {stg_bits}'d{};",
            stage + 1
        );
        let _ = writeln!(s, "                        end");
        let _ = writeln!(s, "                    end");
    }
    let _ = writeln!(s, "                    default: ;");
    let _ = writeln!(s, "                endcase");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s);
    s
}

fn emit_run_script(dut_name: &str, tb_name: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "#!/bin/sh");
    let _ = writeln!(s, "# Reproduces the checked-in bundle.");
    let _ = writeln!(s, "set -eu");
    let _ = writeln!(s, "cd \"$(dirname \"$0\")\"");
    let _ = writeln!(
        s,
        "verilator --binary --timing -Wno-fatal -Wno-DECLFILENAME --top-module {tb_name} -o \
         {tb_name}_sim {dut_name}.sv {tb_name}.sv"
    );
    let _ = writeln!(s, "./obj_dir/{tb_name}_sim");
    s
}

fn emit_testbench(
    model: &MappedModel,
    dut_name: &str,
    tb_name: &str,
    provenance: &Provenance,
    expected: &[Vec<i64>],
) -> String {
    let aw = model.arena_address_bits();
    let neng = model.engines.len();
    let nlink = model.links.len();
    let frames = model.frames.len().max(1);
    let ni = model.in_words;
    let no = model.out_words;
    let timeout = (model.arena_words)
        .saturating_mul(8)
        .saturating_add(
            ni.saturating_add(no)
                .saturating_mul(frames as u64)
                .saturating_mul(4),
        )
        .saturating_mul(4)
        .saturating_add(50_000);
    let busy_sum = (0..neng)
        .map(|index| format!("engines_busy[{index}]"))
        .collect::<Vec<_>>()
        .join(" + ");
    let mut s = header(provenance, "self-checking testbench", "mapped testbench");
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s, "module {tb_name};");
    let _ = writeln!(s, "    localparam int AW = {aw};");
    let _ = writeln!(s, "    localparam int NF = {frames};");
    let _ = writeln!(s, "    localparam int NI = {ni};");
    let _ = writeln!(s, "    localparam int NO = {no};");
    let _ = writeln!(s, "    localparam int IN_BASE = {};", model.in_base);
    let _ = writeln!(s, "    localparam int OUT_BASE = {};", model.out_base);
    let _ = writeln!(s, "    localparam int NENG = {neng};");
    let _ = writeln!(s, "    localparam int NLINK = {nlink};");
    let _ = writeln!(s, "    localparam int TIMEOUT = {timeout};");
    let _ = writeln!(s, "    logic clk = 1'b0;");
    let _ = writeln!(s, "    logic rst = 1'b1;");
    let _ = writeln!(s, "    logic start = 1'b0;");
    let _ = writeln!(s, "    logic done;");
    let _ = writeln!(s, "    logic [{neng}-1:0] engines_busy;");
    let _ = writeln!(s, "    logic ld_en = 1'b0;");
    let _ = writeln!(s, "    logic [AW-1:0] ld_addr = '0;");
    let _ = writeln!(s, "    logic signed [15:0] ld_data = 16'sd0;");
    let _ = writeln!(s, "    logic [AW-1:0] rd_addr = '0;");
    let _ = writeln!(s, "    logic signed [15:0] rd_data;");
    let _ = writeln!(s, "    logic signed [15:0] stim [0:NF*NI-1];");
    let _ = writeln!(s, "    logic signed [15:0] exp_mem [0:NF*NO-1];");
    let _ = writeln!(s, "    int mism = 0;");
    let _ = writeln!(s, "    int checks = 0;");
    let _ = writeln!(s, "    int cyc = 0;");
    let _ = writeln!(s, "    int overlap_cycles = 0;");
    let _ = writeln!(s, "    int frame = 0;");
    let _ = writeln!(s, "    int i = 0;");
    let _ = writeln!(s, "    logic signed [15:0] got;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    {dut_name} dut (");
    let _ = writeln!(
        s,
        "        .clk(clk), .rst(rst), .start(start), .done(done),"
    );
    let _ = writeln!(s, "        .engines_busy(engines_busy),");
    let _ = writeln!(
        s,
        "        .ld_en(ld_en), .ld_addr(ld_addr), .ld_data(ld_data),"
    );
    let _ = writeln!(s, "        .rd_addr(rd_addr), .rd_data(rd_data));");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        forever #5 clk = ~clk;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // Stimulus and expected codes, computed in Rust with the same"
    );
    let _ = writeln!(s, "    // integer formula the datapath implements.");
    let _ = writeln!(s, "    initial begin");
    for (index, frame) in model.frames.iter().enumerate() {
        for (offset, code) in frame.iter().enumerate() {
            let _ = writeln!(
                s,
                "        stim[{}] = {};",
                index * frame.len() + offset,
                sv_signed(16, *code)
            );
        }
    }
    for (index, frame) in expected.iter().enumerate() {
        for (offset, code) in frame.iter().enumerate() {
            let _ = writeln!(
                s,
                "        exp_mem[{}] = {};",
                index * frame.len() + offset,
                sv_signed(16, *code)
            );
        }
    }
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        cyc <= cyc + 1;");
    let _ = writeln!(s, "        if (({busy_sum}) > 1) begin");
    let _ = writeln!(s, "            overlap_cycles <= overlap_cycles + 1;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "        if (cyc > TIMEOUT) begin");
    let _ = writeln!(
        s,
        "            $display(\"TESSA_RTL_TIMEOUT cycles=%0d\", cyc);"
    );
    let _ = writeln!(s, "            $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "            $finish;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        repeat (4) @(negedge clk);");
    let _ = writeln!(s, "        rst = 1'b0;");
    let _ = writeln!(
        s,
        "        for (frame = 0; frame < NF; frame = frame + 1) begin"
    );
    let _ = writeln!(s, "            for (i = 0; i < NI; i = i + 1) begin");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "                ld_en = 1'b1;");
    let _ = writeln!(s, "                ld_addr = AW'(IN_BASE + i);");
    let _ = writeln!(s, "                ld_data = stim[frame*NI + i];");
    let _ = writeln!(s, "            end");
    let _ = writeln!(
        s,
        "            @(negedge clk); ld_en = 1'b0; ld_addr = '0; ld_data = 16'sd0;"
    );
    let _ = writeln!(s, "            @(negedge clk); start = 1'b1;");
    let _ = writeln!(s, "            @(negedge clk); start = 1'b0;");
    let _ = writeln!(s, "            while (!done) @(negedge clk);");
    let _ = writeln!(s, "            for (i = 0; i < NO; i = i + 1) begin");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "                rd_addr = AW'(OUT_BASE + i);");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "                got = rd_data;");
    let _ = writeln!(s, "                checks = checks + 1;");
    let _ = writeln!(
        s,
        "                if (got !== exp_mem[frame*NO + i]) begin"
    );
    let _ = writeln!(s, "                    mism = mism + 1;");
    let _ = writeln!(
        s,
        "                    if (mism < 12) $display(\"TESSA_RTL_MISMATCH frame=%0d idx=%0d \
         expected=%0d actual=%0d\", frame, i, exp_mem[frame*NO + i], got);"
    );
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "        end");
    let _ = writeln!(
        s,
        "        $display(\"TESSA_RTL_PIPELINE engines=%0d links=%0d overlap_cycles=%0d\", NENG, \
         NLINK, overlap_cycles);"
    );
    let _ = writeln!(
        s,
        "        $display(\"TESSA_RTL_RESULT module={tb_name} frames=%0d checks=%0d mismatches=%0d \
         cycles=%0d\", NF, checks, mism, cyc);"
    );
    let _ = writeln!(s, "        if (mism == 0) $display(\"TESSA_RTL_PASS\");");
    let _ = writeln!(s, "        else $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "        $finish;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s, "`default_nettype wire");
    s
}

// ---------------------------------------------------------------------------
// Emission tests
// ---------------------------------------------------------------------------

/// Structural and simulation tests of the emitted organization.
///
/// These build a [`MappedModel`] directly, with no solver and no plan, so the
/// generated datapath can be exercised (and checked against the Rust reference
/// evaluation) independently of how a mapping was found. The geometry mirrors
/// the four-layer pointwise chain the mapping tests use: 4 -> 4 channels over a
/// 2x2 plane, split across two engine instances with and without a pipeline
/// link.
#[cfg(test)]
mod emission_tests {
    use super::*;

    fn weights_for(seed: u64, count: usize) -> Vec<i8> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as i8
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn synthetic_op(
        engine: usize,
        op_index: usize,
        name: &str,
        simd: u32,
        pe: u32,
        in_base: u64,
        out_base: u64,
        in_tensor: TensorId,
        out_tensor: TensorId,
        seed: u64,
    ) -> MappedOp {
        let (ic, oc, ih, iw) = (4u32, 4u32, 2u32, 2u32);
        let weights = weights_for(seed, (oc * ic) as usize);
        let m0: Vec<i64> = (0..oc as usize)
            .map(|channel| 1 + (channel as i64 * 3))
            .collect();
        let mut max_abs_acc = 0i64;
        for channel in 0..oc as usize {
            let mut sum = 0i64;
            for ic_index in 0..ic as usize {
                sum += i64::from(weights[channel * ic as usize + ic_index].unsigned_abs()) * 127;
            }
            max_abs_acc = max_abs_acc.max(sum);
        }
        MappedOp {
            name: name.to_string(),
            engine,
            op_index,
            ic,
            oc,
            ih,
            iw,
            oh: ih,
            ow: iw,
            k: ic / simd,
            fold: oc / pe,
            weights,
            bias: vec![0; oc as usize],
            m0,
            shift: 3,
            lo: -128,
            hi: 127,
            max_abs_acc: max_abs_acc as u64,
            in_tensor,
            out_tensor,
            in_base,
            out_base,
            in_words: u64::from(ic) * u64::from(ih) * u64::from(iw),
            out_words: u64::from(oc) * u64::from(ih) * u64::from(iw),
            link_in: None,
            drains: false,
            link_out: None,
        }
    }

    /// The four-layer chain over two engines. With `pipelined`, the tensor
    /// crossing the engine boundary travels through a FIFO and the consumer is
    /// allowed to overlap its producer; without it the consumer waits.
    fn synthetic(pipelined: bool) -> MappedModel {
        let words = 16u64;
        let base = |index: u64| index * words;
        let (t_in, t0, t1, t2, t_out) = (
            TensorId(0),
            TensorId(1),
            TensorId(2),
            TensorId(3),
            TensorId(4),
        );
        let mut ops = vec![
            synthetic_op(0, 0, "pw0", 1, 1, base(0), base(1), t_in, t0, 0x11),
            synthetic_op(0, 1, "pw1", 1, 1, base(1), base(2), t0, t1, 0x22),
            synthetic_op(1, 0, "pw2", 2, 2, base(2), base(3), t1, t2, 0x33),
            synthetic_op(1, 1, "pw3", 2, 2, base(3), base(4), t2, t_out, 0x44),
        ];
        let mut links = Vec::new();
        if pipelined {
            ops[1].link_out = Some(0);
            ops[1].out_base = 0;
            ops[2].link_in = Some(0);
            ops[2].drains = true;
            ops[2].in_base = 0;
            links.push(MappedLink {
                index: 0,
                producer: 0,
                consumer: 1,
                tensor: t1,
                tensor_name: "pw1.out".to_string(),
                words,
                // Shallower than the tensor, so the producer really stalls on
                // the ready/valid handshake.
                depth: 4,
            });
        }
        let engines = vec![
            MappedEngine {
                index: 0,
                name: "shared0".to_string(),
                role: EngineRole::Shared,
                simd: 1,
                pe: 1,
                tap: 1,
                lanes: 1,
                ops: vec![0, 1],
                link_out: pipelined.then_some(0),
                link_in: None,
                link_out_words: if pipelined { words } else { 0 },
                link_in_words: 0,
                weight_total: 32,
                bias_total: 8,
                m0_total: 8,
                max_abs_acc: ops[0].max_abs_acc.max(ops[1].max_abs_acc),
            },
            MappedEngine {
                index: 1,
                name: "shared1".to_string(),
                role: EngineRole::Shared,
                simd: 2,
                pe: 2,
                tap: 1,
                lanes: 4,
                ops: vec![2, 3],
                link_out: None,
                link_in: pipelined.then_some(0),
                link_out_words: 0,
                link_in_words: if pipelined { words } else { 0 },
                weight_total: 32,
                bias_total: 8,
                m0_total: 8,
                max_abs_acc: ops[2].max_abs_acc.max(ops[3].max_abs_acc),
            },
        ];
        let stages = vec![
            MappedStage {
                op: 0,
                engine: 0,
                barriers: vec![],
                overlaps: vec![],
            },
            MappedStage {
                op: 1,
                engine: 0,
                barriers: vec![0],
                overlaps: vec![],
            },
            MappedStage {
                op: 2,
                engine: 1,
                barriers: if pipelined { vec![] } else { vec![1] },
                overlaps: if pipelined { vec![1] } else { vec![] },
            },
            MappedStage {
                op: 3,
                engine: 1,
                barriers: vec![2],
                overlaps: vec![],
            },
        ];
        let mut frames = Vec::new();
        for frame in 0..2u64 {
            frames.push(
                weights_for(0xabc + frame, words as usize)
                    .into_iter()
                    .map(i64::from)
                    .collect(),
            );
        }
        MappedModel {
            ops,
            engines,
            links,
            stages,
            arena_words: base(4) + words,
            in_tensor: t_in,
            in_words: words,
            in_base: base(0),
            in_channels: 4,
            out_tensor: t_out,
            out_words: words,
            out_base: base(4),
            out_channels: 4,
            in_spatial: (2, 2),
            out_spatial: (2, 2),
            schedule_cycles: 96,
            makespan_cycles: 64,
            frames,
        }
    }

    fn bundle_for(model: &MappedModel, config: &RtlConfig) -> RtlBundle {
        let expected = evaluate(model).expect("the reference evaluation runs");
        let dut_name = "tessera_mapped_v2".to_string();
        let tb_name = "tessera_mapped_v2_tb".to_string();
        let provenance = Provenance {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            git_revision: "test".to_string(),
            dirty: false,
            graph_schema_version: 1,
            plan_schema_version: 1,
            target: "ti60f225".to_string(),
            target_revision: "test".to_string(),
            input_manifest_sha256: "ab".repeat(32),
            weights_sha256: "cd".repeat(32),
            planner_config_sha256: "ef".repeat(32),
        };
        let files = vec![
            RtlFile {
                name: format!("{dut_name}.sv"),
                kind: RtlFileKind::SystemVerilog,
                contents: emit_dut(model, &dut_name, &provenance, "synthetic"),
            },
            RtlFile {
                name: format!("{tb_name}.sv"),
                kind: RtlFileKind::Testbench,
                contents: emit_testbench(model, &dut_name, &tb_name, &provenance, &expected),
            },
        ];
        let metadata = RtlMetadata {
            format: RTL_BUNDLE_FORMAT.to_string(),
            schema_version: RTL_BUNDLE_SCHEMA_VERSION,
            kernel_id: MAPPED_KERNEL_ID.to_string(),
            provenance,
            plan: PlanContext::direct("synthetic"),
            top_module: tb_name,
            kernel_module: dut_name,
            layer: "pw0,pw1,pw2,pw3".to_string(),
            tile: TileGeometry {
                simd: 2,
                noc: 2,
                k: 1,
                fold: 1,
            },
            in_channels: 4,
            out_channels: 4,
            accumulator_width: MAPPED_ACC_BITS,
            max_abs_accumulator: 0,
            kernel_cycles_per_window: model.makespan_cycles,
            requantization: vec![],
            requant_multiplier_error_ppm: None,
            config: config.clone(),
            files: vec![],
            supported_operators: vec![],
            supported_interface: vec![],
            unsupported: vec![],
            reuse_candidates: vec![],
            reproduce: vec![],
            bundle_digest: String::new(),
            full_graph: None,
            mapped: Some(summarize(model)),
        };
        RtlBundle { metadata, files }
    }

    fn simulate(tag: &str, pipelined: bool, expect_overlap: bool) {
        let model = synthetic(pipelined);
        let summary = summarize(&model);
        assert_eq!(summary.engines.len(), 2);
        assert_eq!(summary.multiplier_lanes, 5, "1 + 4 lane instances");
        assert_eq!(summary.links.len(), usize::from(pipelined));
        let config = RtlConfig {
            module_prefix: "tessera".to_string(),
            emit_testbench: true,
            emit_memory_images: false,
            fixture_windows: model.frames.len(),
            fixture_seed: 1,
        };
        let bundle = bundle_for(&model, &config);
        let dut = bundle
            .file("tessera_mapped_v2.sv")
            .expect("the DUT source is emitted")
            .contents
            .clone();
        assert_eq!(
            dut.matches(" u_lane_").count(),
            5,
            "one multiplier per lane: 1 on engine 0, 4 on engine 1"
        );
        assert_eq!(dut.matches(" u_engine").count(), 2);
        assert_eq!(dut.matches(" u_fifo").count(), usize::from(pipelined));
        if super::super::verilator_version().is_none() {
            eprintln!("SKIP {tag}: verilator unavailable; the mapped RTL is not simulated");
            return;
        }
        let dir =
            std::env::temp_dir().join(format!("tessera_mapped_unit_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        match super::super::simulate_with_verilator(&bundle, &dir).expect("the scaffold runs") {
            SimulationOutcome::Passed { log } => {
                let line = log
                    .lines()
                    .find(|line| line.contains("TESSA_RTL_PIPELINE"))
                    .expect("the testbench reports its pipeline structure");
                let overlap: usize = line
                    .split_whitespace()
                    .find_map(|token| token.strip_prefix("overlap_cycles="))
                    .expect("overlap_cycles is reported")
                    .parse()
                    .expect("overlap_cycles is numeric");
                if expect_overlap {
                    assert!(overlap > 0, "{tag}: expected overlap, log:\n{log}");
                } else {
                    assert_eq!(overlap, 0, "{tag}: unexpected overlap, log:\n{log}");
                }
                let _ = std::fs::remove_dir_all(&dir);
            }
            SimulationOutcome::ToolUnavailable { tool, reason } => {
                eprintln!("SKIP {tag}: {tool} unavailable ({reason})");
            }
            SimulationOutcome::Failed { log, detail } => {
                panic!("{tag} failed: {detail}\n{log}");
            }
        }
    }

    use super::super::SimulationOutcome;

    #[test]
    fn two_engine_chain_without_a_link_matches_the_reference() {
        simulate("synthetic_serial", false, false);
    }

    #[test]
    fn two_engine_chain_with_a_link_matches_the_reference() {
        simulate("synthetic_pipeline", true, true);
    }
}

// ---------------------------------------------------------------------------
// Plan-level tests
// ---------------------------------------------------------------------------

/// Tests that need a real schema-2 plan: the mapping the solver finds, projected
/// onto an execution plan, is the authority `lower_mapped` must agree with, so
/// the emitted datapath and the reported structure have to match it. The
/// synthetic model above cannot show that, because it is built by hand.
#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::planner::{Policy, ProvenanceInput};
    use crate::solver::{
        ChainFixture, MappingBounds, SpatialShape, pointwise_chain4, solve_mapping,
    };
    use crate::target::Ti60Target;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }

    fn target() -> Ti60Target {
        Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("load the Ti60")
    }

    fn policy() -> Policy {
        Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("load the policy")
    }

    fn provenance() -> ProvenanceInput {
        ProvenanceInput {
            graph_sha256: "graph-sha256".to_string(),
            target_sha256: "target-sha256".to_string(),
            policy_sha256: "policy-sha256".to_string(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    struct FixtureWeights<'a> {
        weights: &'a BTreeMap<String, Vec<i8>>,
    }

    impl WeightResolver for FixtureWeights<'_> {
        fn tensor_bytes(&self, tensor: &crate::layout::TensorRef) -> Result<Vec<u8>, RtlError> {
            self.weights
                .get(&tensor.name)
                .map(|codes| codes.iter().map(|code| *code as u8).collect())
                .ok_or_else(|| RtlError::WeightsNotResident {
                    layer: tensor.name.clone(),
                    tensor: tensor.name.clone(),
                    detail: "the fixture carries no weight tensor with this name".to_string(),
                })
        }
    }

    /// Two shared engines with exactly one pipeline cut: organization 3 of the
    /// mapping tests.
    fn two_shared_engines_with_one_cut() -> MappingBounds {
        MappingBounds {
            min_engines: 2,
            max_engines: 2,
            min_lanes_per_engine: 1,
            max_lanes_per_engine: 4,
            min_pipeline_links: 1,
            max_pipeline_links: 1,
            min_dedicated_engines: 0,
            min_shared_engines: 2,
            max_arena_words: 1 << 20,
            shapes: vec![
                SpatialShape {
                    simd: 1,
                    pe: 1,
                    tap_parallel: 1,
                },
                SpatialShape {
                    simd: 1,
                    pe: 2,
                    tap_parallel: 1,
                },
                SpatialShape {
                    simd: 2,
                    pe: 1,
                    tap_parallel: 1,
                },
                SpatialShape {
                    simd: 1,
                    pe: 4,
                    tap_parallel: 1,
                },
                SpatialShape {
                    simd: 2,
                    pe: 2,
                    tap_parallel: 1,
                },
                SpatialShape {
                    simd: 4,
                    pe: 1,
                    tap_parallel: 1,
                },
            ],
        }
    }

    /// Solve the chain for one organization and project it onto a plan.
    fn solve(bounds: &MappingBounds) -> (ChainFixture, ExecutionPlan) {
        let fixture = pointwise_chain4();
        let artifact = solve_mapping(&fixture.input, &target(), &policy(), &provenance(), bounds)
            .expect("the mapping solve succeeds");
        let plan = artifact
            .execution_plan()
            .expect("a validated layout projects onto an execution plan");
        assert!(
            plan.mapping.is_some(),
            "a schema-2 plan carries the mapping it was solved for"
        );
        (fixture, plan)
    }

    fn lower(fixture: &ChainFixture, plan: &ExecutionPlan) -> RtlBundle {
        let resolver = FixtureWeights {
            weights: &fixture.weights,
        };
        lower_mapped(plan, &fixture.graph, &resolver, &RtlConfig::default())
            .expect("the mapped backend emits the solved chain")
    }

    fn dut_source(bundle: &RtlBundle) -> String {
        bundle
            .files()
            .iter()
            .find(|file| file.name.ends_with("_mapped_v2.sv"))
            .expect("the bundle carries the mapped DUT source")
            .contents
            .clone()
    }

    /// The emitted accumulator register must be exactly as wide as the plan
    /// declares the engine's accumulator, and the requant multiplier must take
    /// that register as its accumulator operand. The target model charges the
    /// multiplier for `accumulator_bits` (two DSP48 chunks of an 18-bit
    /// operand), so an emitted datapath of any other width would contradict the
    /// artifact's own cost model.
    #[test]
    fn the_emitted_accumulator_is_the_declared_width() {
        let (fixture, plan) = solve(&two_shared_engines_with_one_cut());
        let mapping = plan.mapping.as_ref().expect("a schema-2 plan");
        let declared: Vec<u32> = mapping
            .engines
            .iter()
            .map(|engine| engine.accumulator_bits)
            .collect();
        assert!(
            declared.iter().all(|bits| *bits == MAPPED_ACC_BITS),
            "the plan declares {MAPPED_ACC_BITS}-bit accumulators, saw {declared:?}"
        );
        assert_eq!(plan.engine.accumulator_bits, MAPPED_ACC_BITS);

        let dut = dut_source(&lower(&fixture, &plan));
        let acc = format!("logic signed [{}:0] acc [0:PE-1];", MAPPED_ACC_BITS - 1);
        assert!(
            dut.contains(&acc),
            "every engine accumulates in a {MAPPED_ACC_BITS}-bit register:\n{dut}"
        );
        assert!(
            !dut.contains("logic signed [63:0] acc"),
            "no engine accumulates in a width the plan does not declare"
        );
        assert!(
            dut.contains("rq_product[lane] = acc[lane] * $signed("),
            "the requant multiplier takes the {MAPPED_ACC_BITS}-bit accumulator as its operand"
        );
        assert!(
            dut.contains(&format!("{MAPPED_ACC_BITS}'sd0")),
            "the accumulator reset uses its own width"
        );
    }

    /// A pipeline link summary must name the tensor crossing the cut the way the
    /// plan names it, and must report the words that really cross: the producer
    /// step's output element count.
    #[test]
    fn a_pipeline_link_summary_names_the_plan_tensor_and_its_words() {
        let (fixture, plan) = solve(&two_shared_engines_with_one_cut());
        let bundle = lower(&fixture, &plan);
        let summary = bundle
            .metadata
            .mapped
            .as_ref()
            .expect("the mapped backend records its structure");
        assert_eq!(summary.links.len(), 1, "these bounds ask for one cut");
        let link = &summary.links[0];

        let tensor = plan
            .tensors
            .iter()
            .find(|tensor| tensor.name == link.tensor)
            .unwrap_or_else(|| {
                panic!(
                    "the link summary names the plan's tensor, saw {:?}",
                    link.tensor
                )
            });
        let producer = plan
            .steps
            .iter()
            .find(|step| step.outputs.contains(&tensor.id))
            .expect("the cut tensor is produced by a step of the plan");
        let expected = u64::from(producer.out_channels)
            * u64::from(producer.out_spatial[0])
            * u64::from(producer.out_spatial[1]);
        assert!(expected > 0, "a cut tensor carries at least one word");
        assert_eq!(
            link.words, expected,
            "the link carries the producer step {}'s {expected} output words",
            producer.op
        );
    }

    /// A plan whose engine declares an accumulator width the emitted datapath
    /// does not implement must be refused, and the refusal must name the engine
    /// and both widths.
    #[test]
    fn an_engine_accumulator_width_that_differs_is_rejected() {
        let declared = MAPPED_ACC_BITS * 2;
        let (fixture, mut plan) = solve(&two_shared_engines_with_one_cut());
        let engine_name = {
            let mapping = plan
                .mapping
                .as_mut()
                .expect("a schema-2 plan carries the mapping");
            let engine = mapping.engines.first_mut().expect("at least one engine");
            engine.accumulator_bits = declared;
            engine.name.clone()
        };
        let resolver = FixtureWeights {
            weights: &fixture.weights,
        };
        let error = lower_mapped(&plan, &fixture.graph, &resolver, &RtlConfig::default())
            .expect_err("the emitted MAC array cannot accumulate in a wider register");
        let message = error.to_string();
        assert!(!engine_name.is_empty(), "the engine has a name to report");
        assert!(
            message.contains(&engine_name),
            "the rejection names engine {engine_name}: {message}"
        );
        assert!(
            message.contains(&format!("accumulator_bits = {declared}")),
            "the rejection names the declared width {declared}: {message}"
        );
        assert!(
            message.contains(&format!("{MAPPED_ACC_BITS} bits")),
            "the rejection names the emitted width {MAPPED_ACC_BITS}: {message}"
        );

        // The plan's engine summary is the same declaration, so it is held to the
        // same width.
        let (fixture, mut plan) = solve(&two_shared_engines_with_one_cut());
        plan.engine.accumulator_bits = declared;
        let resolver = FixtureWeights {
            weights: &fixture.weights,
        };
        let error = lower_mapped(&plan, &fixture.graph, &resolver, &RtlConfig::default())
            .expect_err("the plan's engine summary must agree with the emitted datapath");
        let message = error.to_string();
        assert!(!plan.engine.name.is_empty(), "the plan engine has a name");
        assert!(
            message.contains(&plan.engine.name),
            "the rejection names the plan's engine {}: {message}",
            plan.engine.name
        );
        assert!(
            message.contains(&format!("accumulator_bits = {declared}")),
            "the rejection names the declared width {declared}: {message}"
        );
    }
}
