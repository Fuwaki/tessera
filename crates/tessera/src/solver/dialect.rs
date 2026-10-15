//! The solver on the tessera dialect.
//!
//! [`problem_from_tessera`] reads the operator ops of a tessera module into the
//! solver's in-memory problem; [`solve_tessera`] solves it with CP-SAT and
//! writes the decisions back into the same module as `tessera.engine`,
//! `tessera.bind`, `tessera.buffer`, `tessera.segment` and `tessera.schedule`
//! ops, then re-verifies the module.
//!
//! Segments are derived from the solved engine assignment: the operators of
//! one shared engine form a shared segment, and every maximal run of
//! consecutive dedicated operators forms a dedicated segment. A tensor whose
//! producer and consumers sit in one segment lives in that segment's local
//! arena (1 bank); a tensor that crosses segments, or the graph input and
//! output, lives in a K-bank rotating buffer (K = 2, so the producer segment
//! can write frame f+1 while the consumer reads frame f).

use std::collections::HashMap;

use melior::Context;
use melior::ir::attribute::{
    ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, BlockLike, Identifier, Location, Module, RegionLike};

use crate::dialect::read::{OpKind as DOp, OperatorOp, Source, TesseraModule};
use crate::layout::{
    Activation, ActivationKind, CodeRange, ElemType, EngineRole, LayoutArtifact, OpKind,
    PlanStatus, Requant, Rounding, TensorId, TensorShape,
};
use crate::lower::elem_for_range;
use crate::planner::{
    ConvDetail, ElementwiseDetail, OpDetail, Padding, PlanInput, Policy, PrimitiveOp,
    ProvenanceInput, TensorDesc, UpsampleDetail,
};
use crate::target::Ti60Target;

use super::{MappingBounds, SolverError};

/// K of a cross-segment rotating buffer.
pub const CROSS_SEGMENT_BANKS: u64 = 2;

fn invalid(detail: impl Into<String>) -> SolverError {
    SolverError::InvalidInput { detail: detail.into() }
}

fn narrow(v: u64, what: &str) -> Result<u32, SolverError> {
    u32::try_from(v).map_err(|_| invalid(format!("{what} {v} does not fit in 32 bits")))
}

fn nhwc(shape: &[u64], what: &str) -> Result<TensorShape, SolverError> {
    match shape {
        [n, h, w, c] => Ok(TensorShape::new(
            narrow(*n, what)?,
            narrow(*c, what)?,
            narrow(*h, what)?,
            narrow(*w, what)?,
        )),
        other => Err(invalid(format!("{what}: {other:?} is not NHWC"))),
    }
}

fn activation_for(clamp: [i64; 2]) -> Activation {
    let kind = match clamp {
        [0, 255] => ActivationKind::UnsignedClamp,
        [0, _] => ActivationKind::Relu,
        _ => ActivationKind::None,
    };
    Activation { kind, clamp_lo: clamp[0] as i32, clamp_hi: clamp[1] as i32 }
}

fn plan_kind(kind: DOp) -> OpKind {
    match kind {
        DOp::Conv2d => OpKind::Conv2d,
        DOp::PointwiseConv2d => OpKind::PointwiseConv2d,
        DOp::DepthwiseConv2d => OpKind::DepthwiseConv2d,
        DOp::Add => OpKind::Add,
        DOp::Upsample => OpKind::NearestUpsample2x,
        DOp::Relu => OpKind::Activation,
        DOp::Requant => OpKind::Requant,
        DOp::Quantize => OpKind::QuantizeOutput,
    }
}

/// The solver problem of a tessera module: one plan op per operator op, one
/// plan tensor per SSA edge and weight ref, named after the operator ops.
pub fn problem_from_tessera(module: &TesseraModule) -> Result<PlanInput, SolverError> {
    let mut tensors: Vec<TensorDesc> = Vec::new();
    let mut push = |name: String, shape: TensorShape, elem: ElemType, range: Option<CodeRange>| {
        let id = TensorId::new(tensors.len() as u32);
        tensors.push(TensorDesc { id, name, shape, elem, code_range: range });
        id
    };
    let input_ty = module.arguments.first().ok_or_else(|| invalid("graph has no input"))?;
    let [lo, hi] = module.input_range.unwrap_or([-127, 127]);
    let (elem, range) = elem_for_range(lo, hi, false);
    let input = push("input".into(), nhwc(&input_ty.shape, "graph input")?, elem, Some(range));

    let mut produced: HashMap<u32, TensorId> = HashMap::new();
    let mut ops = Vec::new();
    for op in &module.operators {
        let activation_of = |source: Source| -> Result<TensorId, SolverError> {
            match source {
                Source::Arg(0) => Ok(input),
                Source::Op(i) => produced
                    .get(&i)
                    .copied()
                    .ok_or_else(|| invalid(format!("operator #{} reads #{i} before it exists", op.index))),
                Source::Arg(n) => Err(invalid(format!("operator #{} reads weight arg {n} as an activation", op.index))),
            }
        };
        let inputs: Vec<TensorId> =
            op.activations().map(|(s, _)| activation_of(s)).collect::<Result<_, _>>()?;
        let unsigned = op.kind == DOp::Quantize || op.clamp == [0, 255];
        let (elem, range) = elem_for_range(op.range[0], op.range[1], unsigned);
        let out_name = if op.kind == DOp::Quantize { "output.u8".to_string() } else { format!("{}__out", op.name) };
        let elem = if op.kind == DOp::Quantize { ElemType::U8 } else { elem };
        let shape = nhwc(&op.output.shape, &op.name)?;
        let detail = match op.kind {
            k if k.is_conv() => {
                let w = &op.operand_types[1].shape;
                let wshape = if k == DOp::DepthwiseConv2d {
                    TensorShape::new(narrow(w[2], "filter")?, 1, narrow(w[0], "filter")?, narrow(w[1], "filter")?)
                } else {
                    TensorShape::new(narrow(w[0], "filter")?, narrow(w[3], "filter")?, narrow(w[1], "filter")?, narrow(w[2], "filter")?)
                };
                let weights = push(format!("weights:{}", op.name), wshape, ElemType::I8, None);
                let oc = shape.c;
                let bias = push(format!("bias:{}", op.name), TensorShape::new(oc, 1, 1, 1), ElemType::I32, None);
                OpDetail::Conv(ConvDetail {
                    weights,
                    bias: Some(bias),
                    groups: narrow(op.groups, "groups")?,
                    stride: [narrow(op.stride[0], "stride")?, narrow(op.stride[1], "stride")?],
                    padding: if op.reflect {
                        Padding::Reflect
                    } else {
                        let p = narrow(op.pad, "pad")?;
                        Padding::Zero { top: p, bottom: p, left: p, right: p }
                    },
                    requant: requant(op)?,
                    activation: activation_for(op.clamp),
                })
            }
            DOp::Add => OpDetail::Elementwise(ElementwiseDetail {
                requant: Some(requant(op)?),
                activation: Activation { kind: ActivationKind::None, clamp_lo: op.range[0] as i32, clamp_hi: op.range[1] as i32 },
            }),
            DOp::Upsample => OpDetail::Upsample(UpsampleDetail { scale: narrow(op.scale, "scale")? }),
            DOp::Quantize => OpDetail::Elementwise(ElementwiseDetail {
                requant: None,
                activation: Activation { kind: ActivationKind::UnsignedClamp, clamp_lo: 0, clamp_hi: op.clamp[1] as i32 },
            }),
            DOp::Relu | DOp::Requant => {
                return Err(invalid(format!("standalone {} is not mapped yet; the frontend folds it", op.kind.name())));
            }
            _ => unreachable!(),
        };
        let output = push(out_name, shape, elem, Some(range));
        produced.insert(op.index, output);
        ops.push(PrimitiveOp {
            name: op.name.clone(),
            kind: plan_kind(op.kind),
            inputs,
            outputs: vec![output],
            detail,
        });
    }
    let outputs = module
        .results
        .iter()
        .map(|r| produced.get(r).copied().ok_or_else(|| invalid("graph result is not produced")))
        .collect::<Result<_, _>>()?;
    Ok(PlanInput { name: module.function.clone(), tensors, ops, inputs: vec![input], outputs })
}

fn requant(op: &OperatorOp) -> Result<Requant, SolverError> {
    let m0 = op.multiplier.iter().copied().max().ok_or_else(|| invalid(format!("{} has no multiplier", op.name)))?;
    Ok(Requant {
        m0,
        shift: op.shift as i32,
        rounding: Rounding::HalfAwayFromZero,
        saturate_lo: op.clamp[0] as i32,
        saturate_hi: op.clamp[1] as i32,
        per_channel_multipliers: if op.kind.is_conv() { op.multiplier.clone() } else { Vec::new() },
        multiplier_error_ppm: None,
    })
}

/// Decisions derived from a solved layout, in dialect terms.
#[derive(Debug, Clone, PartialEq)]
pub struct Decisions {
    pub engines: Vec<crate::dialect::read::EngineOp>,
    pub binds: Vec<crate::dialect::read::BindOp>,
    pub buffers: Vec<crate::dialect::read::BufferOp>,
    pub segments: Vec<crate::dialect::read::SegmentOp>,
    pub schedule: crate::dialect::read::ScheduleOp,
}

/// Turn a feasible solved layout into dialect decisions.
pub fn decisions_from_layout(
    module: &TesseraModule,
    plan: &PlanInput,
    layout: &LayoutArtifact,
) -> Result<Decisions, SolverError> {
    use crate::dialect::read::{BindOp, BufferOp, EngineOp, ScheduleOp, SegmentOp};
    let mapping = layout.mapping.as_ref().ok_or_else(|| invalid("layout has no hardware mapping"))?;
    let index_of: HashMap<&str, u32> = module.operators.iter().map(|o| (o.name.as_str(), o.index)).collect();
    let _role: HashMap<&str, EngineRole> = mapping.engines.iter().map(|e| (e.name.as_str(), e.role)).collect();
    let engines: Vec<EngineOp> = mapping
        .engines
        .iter()
        .map(|e| EngineOp {
            name: e.name.clone(),
            mode: e.role.as_str().to_string(),
            simd: u64::from(e.simd),
            pe: u64::from(e.pe),
            vectors: u64::from(e.tap_parallel),
        })
        .collect();
    let binding: HashMap<&str, &crate::layout::OpEngineBinding> =
        mapping.bindings.iter().map(|b| (b.op.as_str(), b)).collect();

    // Segments come from CP-SAT (via the layout mapping), not from a post-hoc
    // walk of the binding list: that walk produced interleaved shared segments
    // on W8 and a producer/consumer cycle between segments.
    let mut segments: Vec<SegmentOp> = Vec::new();
    let mut segment_of_op: HashMap<u32, u64> = HashMap::new();
    let mut binds = Vec::new();
    if mapping.segments.is_empty() {
        return Err(invalid(
            "layout mapping has no CP-SAT segments; refuse post-hoc derivation",
        ));
    }
    for seg in &mapping.segments {
        let op_indices: Vec<u32> = seg
            .ops
            .iter()
            .map(|name| {
                index_of
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| invalid(format!("segment names unknown op {name}")))
            })
            .collect::<Result<_, _>>()?;
        for &idx in &op_indices {
            segment_of_op.insert(idx, u64::from(seg.index));
        }
        segments.push(SegmentOp {
            index: u64::from(seg.index),
            mode: seg.mode.clone(),
            ops: op_indices,
            engines: seg.engines.clone(),
            cycles: seg.cycles,
        });
    }
    for (step, op) in plan.ops.iter().enumerate() {
        let b = binding
            .get(op.name.as_str())
            .ok_or_else(|| invalid(format!("{} is not bound", op.name)))?;
        let index = *index_of
            .get(op.name.as_str())
            .ok_or_else(|| invalid(format!("{} is not a tessera op", op.name)))?;
        let seg = *segment_of_op
            .get(&index)
            .ok_or_else(|| invalid(format!("operator #{} is in no CP-SAT segment", index)))?;
        binds.push(BindOp {
            op: index,
            engine: b.engine.clone(),
            segment: seg,
            step: step as u64,
        });
    }

    // Buffer residences: K-bank when the producer and a consumer sit in
    // different segments (or the tensor is the graph input/output); local
    // otherwise. Bank counts come from the CP-SAT residence when available.
    let step_of: HashMap<u32, u64> = binds.iter().map(|b| (b.op, b.step)).collect();
    let mut buffers = Vec::new();
    let mut offsets: HashMap<&'static str, u64> = HashMap::new();
    let producers: Vec<Option<u32>> = std::iter::once(None)
        .chain(module.operators.iter().map(|o| Some(o.index)))
        .collect();
    for producer in producers {
        let consumers: Vec<u32> = module
            .operators
            .iter()
            .filter(|o| {
                o.activations().any(|(s, _)| match (s, producer) {
                    (Source::Arg(0), None) => true,
                    (Source::Op(i), Some(p)) => i == p,
                    _ => false,
                })
            })
            .map(|o| o.index)
            .collect();
        let words = match producer {
            None => module.arguments[0].elements(),
            Some(p) => module.operator(p).map(|o| o.output.elements()).unwrap_or(0),
        };
        let is_result = producer.map(|p| module.results.contains(&p)).unwrap_or(false);
        let segs: Vec<Option<u64>> = producer
            .iter()
            .map(|p| segment_of_op.get(p).copied())
            .chain(consumers.iter().map(|c| segment_of_op.get(c).copied()))
            .collect();
        let local = producer.is_some()
            && !is_result
            && !consumers.is_empty()
            && segs.iter().all(|s| s.is_some() && *s == segs[0]);
        let (residence, banks): (&'static str, u64) =
            if local { ("local", 1) } else { ("kbank", CROSS_SEGMENT_BANKS) };
        let first = producer.and_then(|p| step_of.get(&p).copied()).unwrap_or(0);
        let last = consumers
            .iter()
            .filter_map(|c| step_of.get(c).copied())
            .max()
            .unwrap_or(first)
            .max(first);
        let base = offsets.entry(residence).or_insert(0);
        buffers.push(BufferOp {
            producer,
            residence: residence.into(),
            banks,
            base: *base,
            words: words.max(1),
            first_step: first,
            last_step: last,
        });
        *base += words.max(1) * banks;
    }
    let ii = segments.iter().map(|s| s.cycles).max().unwrap_or(1);
    let latency = segments.iter().map(|s| s.cycles).sum::<u64>().max(ii);
    Ok(Decisions {
        engines,
        binds,
        buffers,
        schedule: ScheduleOp {
            segments: segments.iter().map(|s| s.index).collect(),
            initiation_interval: ii,
            latency,
        },
        segments,
    })
}

/// Append the decision ops to the graph function, before its terminator.
pub fn write_decisions(context: &Context, module: &Module, d: &Decisions) -> Result<(), SolverError> {
    let func = module.body().first_operation().ok_or_else(|| invalid("module has no function"))?;
    let block = func
        .region(0)
        .ok()
        .and_then(|r| r.first_block())
        .ok_or_else(|| invalid("function has no body"))?;
    let terminator = block.terminator().ok_or_else(|| invalid("function has no terminator"))?;
    let loc = Location::unknown(context);
    let i64t = IntegerType::new(context, 64).into();
    let int = |v: i64| -> Attribute { IntegerAttribute::new(i64t, v).into() };
    let s = |v: &str| -> Attribute { StringAttribute::new(context, v).into() };
    let arr = |v: &[i64]| -> Attribute { DenseI64ArrayAttribute::new(context, v).into() };
    let id = |n: &str| Identifier::new(context, n);
    let mut emit = |name: &str, attrs: Vec<(&str, Attribute)>| -> Result<(), SolverError> {
        let attrs: Vec<_> = attrs.into_iter().map(|(k, v)| (id(k), v)).collect();
        let op = OperationBuilder::new(name, loc)
            .add_attributes(&attrs)
            .build()
            .map_err(|e| invalid(format!("{name}: {e}")))?;
        block.insert_operation_before(terminator, op);
        Ok(())
    };
    for e in &d.engines {
        emit("tessera.engine", vec![("sym_name", s(&e.name)), ("mode", s(&e.mode)), ("simd", int(e.simd as i64)), ("pe", int(e.pe as i64)), ("vectors", int(e.vectors as i64))])?;
    }
    for b in &d.binds {
        emit("tessera.bind", vec![("op", int(b.op as i64)), ("engine", s(&b.engine)), ("segment", int(b.segment as i64)), ("step", int(b.step as i64))])?;
    }
    for b in &d.buffers {
        emit("tessera.buffer", vec![
            ("producer", int(b.producer.map(i64::from).unwrap_or(-1))), ("residence", s(&b.residence)), ("banks", int(b.banks as i64)),
            ("base", int(b.base as i64)), ("words", int(b.words as i64)), ("first_step", int(b.first_step as i64)), ("last_step", int(b.last_step as i64)),
        ])?;
    }
    for seg in &d.segments {
        let ops: Vec<i64> = seg.ops.iter().map(|o| i64::from(*o)).collect();
        let engines: Vec<Attribute> = seg.engines.iter().map(|e| s(e)).collect();
        emit("tessera.segment", vec![
            ("index", int(seg.index as i64)), ("mode", s(&seg.mode)), ("ops", arr(&ops)),
            ("engines", ArrayAttribute::new(context, &engines).into()), ("cycles", int(seg.cycles as i64)),
        ])?;
    }
    let order: Vec<i64> = d.schedule.segments.iter().map(|v| *v as i64).collect();
    emit("tessera.schedule", vec![("segments", arr(&order)), ("initiation_interval", int(d.schedule.initiation_interval as i64)), ("latency", int(d.schedule.latency as i64))])?;
    Ok(())
}

/// Outcome of solving a tessera module.
#[derive(Debug)]
pub struct DialectSolve {
    /// The solver's full report (status, resources, cycles, mapping).
    pub layout: LayoutArtifact,
    /// The decisions written into the module (`None` when infeasible).
    pub decisions: Option<Decisions>,
}

/// Read a tessera module, solve it, and write the decisions back into it.
pub fn solve_tessera(
    context: &Context,
    module: &Module,
    target: &Ti60Target,
    policy: &Policy,
    provenance: &ProvenanceInput,
    bounds: &MappingBounds,
) -> Result<DialectSolve, SolverError> {
    let model = crate::dialect::read(module).map_err(|e| invalid(e.to_string()))?;
    if model.has_decisions() {
        return Err(invalid("module already carries solver decisions"));
    }
    crate::dialect::check(&model).map_err(|e| invalid(e.to_string()))?;
    let plan = problem_from_tessera(&model)?;
    let layout = super::solve_mapping(&plan, target, policy, provenance, bounds)?;
    if layout.status != PlanStatus::Feasible {
        return Ok(DialectSolve { layout, decisions: None });
    }
    let decisions = decisions_from_layout(&model, &plan, &layout)?;
    write_decisions(context, module, &decisions)?;
    crate::dialect::verify_module(module).map_err(|e| SolverError::Check { detail: e.to_string() })?;
    Ok(DialectSolve { layout, decisions: Some(decisions) })
}
