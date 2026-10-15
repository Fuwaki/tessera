//! Semantic verifier for a tessera mapping module.
//!
//! IRDL only checks local operand/result shapes. This module re-checks the
//! solver's decisions the same way `crate::solver::check` does: every operator
//! is bound exactly once, engine lanes divide the channels they cover, arena
//! residences do not overlap while live, and the recorded initiation interval
//! matches the critical path of the segment pipeline.

use std::collections::{HashMap, HashSet};
use std::fmt;

use super::read::{OpKind, Source, TesseraModule, read};

/// Dialect-level error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DialectError {
    /// IRDL registration or parse failure.
    Irdl { detail: String },
    /// linalg -> tessera conversion failed.
    Convert { detail: String },
    /// A mapping module failed the semantic verifier.
    Verify { detail: String },
}

impl fmt::Display for DialectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DialectError::Irdl { detail } => write!(f, "tessera dialect IRDL: {detail}"),
            DialectError::Convert { detail } => write!(f, "tessera conversion: {detail}"),
            DialectError::Verify { detail } => write!(f, "tessera dialect verify: {detail}"),
        }
    }
}

impl std::error::Error for DialectError {}

/// Verify a tessera module in place: read it and run [`check`].
pub fn verify_module(module: &melior::ir::Module) -> Result<TesseraModule, DialectError> {
    use melior::ir::operation::OperationLike;
    if !module.as_operation().verify() {
        return Err(fail("the module does not pass MLIR/IRDL verification"));
    }
    let model = read(module)?;
    check(&model)?;
    Ok(model)
}

fn fail(detail: impl Into<String>) -> DialectError {
    DialectError::Verify { detail: detail.into() }
}

macro_rules! ensure {
    ($cond:expr, $($fmt:tt)*) => {
        if !$cond {
            return Err(fail(format!($($fmt)*)));
        }
    };
}

fn conv_out(input: u64, kernel: u64, stride: u64, pad: u64) -> Option<u64> {
    (input + 2 * pad).checked_sub(kernel).map(|span| span / stride + 1)
}

/// Semantic checks IRDL cannot express.
///
/// Operator ops: unique indices, operands produced before use, NHWC shapes
/// consistent with kernel/stride/pad/groups, weight refs shaped for the op,
/// per-channel bias/M0, ordered clamps, add operand shapes, upsample scale.
///
/// Decision ops (when any are present): every operator bound exactly once to
/// a declared engine, engines with positive simd/pe/V and a known mode,
/// segments partitioning the operators with a mode matching their engines,
/// buffers with a known residence and bank count (local = 1, kbank >= 2),
/// non-overlapping word ranges for simultaneously live buffers of the same
/// residence, and a schedule ordering every segment once with an initiation
/// interval no shorter than the slowest segment.
pub fn check(model: &TesseraModule) -> Result<(), DialectError> {
    ensure!(!model.operators.is_empty(), "module has no operator ops");
    ensure!(!model.arguments.is_empty(), "graph has no input argument");
    let mut seen = HashSet::new();
    for op in &model.operators {
        let at = format!("{} `{}` (#{})", op.kind.name(), op.name, op.index);
        ensure!(seen.insert(op.index), "{at}: duplicate operator index");
        for (source, _) in op.activations() {
            match source {
                Source::Arg(0) => {}
                Source::Arg(n) => ensure!(false, "{at}: activation operand is weight argument {n}"),
                Source::Op(p) => ensure!(seen.contains(&p) && p != op.index, "{at}: uses #{p} before it is produced"),
            }
        }
        for (source, _) in op.operands.iter().zip(&op.operand_types).skip(op.activations().count()) {
            ensure!(matches!(source, Source::Arg(n) if *n > 0), "{at}: weight operands must be weight arguments");
        }
        ensure!(op.clamp[0] <= op.clamp[1], "{at}: clamp {:?} is empty", op.clamp);
        ensure!(op.output.shape.len() == 4, "{at}: output is not NHWC");
        let input = &op.operand_types[0];
        ensure!(input.shape.len() == 4, "{at}: input is not NHWC");
        let [n, h, w, c] = [input.shape[0], input.shape[1], input.shape[2], input.shape[3]];
        let out = &op.output.shape;
        match op.kind {
            k if k.is_conv() => {
                ensure!(op.operands.len() == 4, "{at}: expects input, weight, bias, multiplier");
                ensure!(op.stride[0] > 0 && op.stride[1] > 0 && op.groups > 0, "{at}: zero stride or groups");
                let oh = conv_out(h, op.kernel[0], op.stride[0], op.pad);
                let ow = conv_out(w, op.kernel[1], op.stride[1], op.pad);
                ensure!(out[0] == n && Some(out[1]) == oh && Some(out[2]) == ow,
                    "{at}: output {out:?} does not follow from input {:?}, kernel {:?}, stride {:?}, pad {}",
                    input.shape, op.kernel, op.stride, op.pad);
                ensure!(!op.reflect || op.pad > 0, "{at}: reflect padding without pad");
                let oc = out[3];
                let weight = &op.operand_types[1].shape;
                if k == OpKind::DepthwiseConv2d {
                    ensure!(op.groups == c && oc == c, "{at}: depthwise needs groups = channels");
                    ensure!(weight == &vec![op.kernel[0], op.kernel[1], c], "{at}: depthwise filter {weight:?}");
                } else {
                    ensure!(c % op.groups == 0 && oc % op.groups == 0, "{at}: groups do not divide channels");
                    ensure!(weight == &vec![oc, op.kernel[0], op.kernel[1], c / op.groups], "{at}: filter {weight:?}");
                    if k == OpKind::PointwiseConv2d {
                        ensure!(op.kernel == [1, 1] && op.stride == [1, 1] && op.pad == 0, "{at}: pointwise must be 1x1/s1/p0");
                    }
                }
                for (slot, what) in [(2, "bias"), (3, "multiplier")] {
                    ensure!(op.operand_types[slot].shape == vec![oc], "{at}: {what} is not per output channel");
                }
            }
            OpKind::Add => {
                ensure!(op.operands.len() == 2, "{at}: expects trunk and branch");
                ensure!(op.operand_types[1].shape == input.shape && out == &input.shape, "{at}: operand shapes differ");
                ensure!(op.multiplier.len() == 1 || op.multiplier.len() as u64 == c, "{at}: multiplier is neither scalar nor per channel");
            }
            OpKind::Upsample => {
                ensure!(op.scale > 0, "{at}: zero scale");
                ensure!(out == &vec![n, h * op.scale, w * op.scale, c], "{at}: output {out:?} is not input x{}", op.scale);
            }
            _ => ensure!(out[..3] == input.shape[..3], "{at}: elementwise op changes the shape"),
        }
    }
    for r in &model.results {
        ensure!(seen.contains(r), "graph returns unknown operator #{r}");
    }
    if model.has_decisions() {
        check_decisions(model)?;
    }
    Ok(())
}

fn check_decisions(model: &TesseraModule) -> Result<(), DialectError> {
    let mut engines = HashMap::new();
    for e in &model.engines {
        ensure!(matches!(e.mode.as_str(), "dedicated" | "shared"), "engine {}: mode {:?}", e.name, e.mode);
        ensure!(e.simd > 0 && e.pe > 0 && e.vectors > 0, "engine {}: zero lanes", e.name);
        ensure!(engines.insert(e.name.as_str(), e).is_none(), "engine {} declared twice", e.name);
    }
    let mut bound = HashMap::new();
    for b in &model.binds {
        ensure!(model.operator(b.op).is_some(), "bind of unknown operator #{}", b.op);
        ensure!(engines.contains_key(b.engine.as_str()), "bind #{} to undeclared engine {}", b.op, b.engine);
        ensure!(bound.insert(b.op, b).is_none(), "operator #{} bound twice", b.op);
    }
    for op in &model.operators {
        ensure!(bound.contains_key(&op.index), "operator #{} is not bound", op.index);
    }
    let mut segment_of = HashMap::new();
    let mut segments = HashMap::new();
    for s in &model.segments {
        ensure!(matches!(s.mode.as_str(), "dedicated" | "shared"), "segment {}: mode {:?}", s.index, s.mode);
        ensure!(segments.insert(s.index, s).is_none(), "segment {} declared twice", s.index);
        ensure!(!s.ops.is_empty(), "segment {} is empty", s.index);
        for e in &s.engines {
            let engine = engines.get(e.as_str()).ok_or_else(|| fail(format!("segment {}: unknown engine {e}", s.index)))?;
            ensure!(engine.mode == s.mode, "segment {} is {} but engine {e} is {}", s.index, s.mode, engine.mode);
        }
        for op in &s.ops {
            ensure!(segment_of.insert(*op, s.index).is_none(), "operator #{op} is in two segments");
            let b = bound.get(op).ok_or_else(|| fail(format!("segment {}: operator #{op} unbound", s.index)))?;
            ensure!(b.segment == s.index, "operator #{op} bound to segment {} but listed in {}", b.segment, s.index);
            ensure!(s.engines.contains(&b.engine), "operator #{op} runs on {} outside segment {}", b.engine, s.index);
        }
        if s.mode == "shared" {
            ensure!(s.engines.len() == 1, "shared segment {} must have one engine", s.index);
        } else {
            let mut used = HashSet::new();
            for op in &s.ops {
                ensure!(used.insert(bound[op].engine.as_str()), "dedicated segment {} reuses an engine", s.index);
            }
        }
    }
    for op in &model.operators {
        ensure!(segment_of.contains_key(&op.index), "operator #{} is in no segment", op.index);
    }
    for (i, b) in model.buffers.iter().enumerate() {
        match b.residence.as_str() {
            "local" => ensure!(b.banks == 1, "local buffer {i} has {} banks", b.banks),
            "kbank" => ensure!(b.banks >= 2, "kbank buffer {i} has {} banks", b.banks),
            "offchip" => {}
            other => ensure!(false, "buffer {i}: residence {other:?}"),
        }
        if let Some(p) = b.producer {
            ensure!(model.operator(p).is_some(), "buffer {i} of unknown operator #{p}");
        }
        ensure!(b.first_step <= b.last_step && b.words > 0, "buffer {i}: empty range");
        for (j, o) in model.buffers.iter().enumerate().skip(i + 1) {
            let live = b.first_step <= o.last_step && o.first_step <= b.last_step;
            let share = b.base < o.base + o.words * o.banks && o.base < b.base + b.words * b.banks;
            ensure!(b.residence != o.residence || b.residence == "offchip" || !live || !share,
                "buffers {i} and {j} ({}) overlap while live", b.residence);
        }
    }
    let schedule = model.schedule.as_ref().ok_or_else(|| fail("decisions without tessera.schedule"))?;
    let mut order = schedule.segments.clone();
    order.sort_unstable();
    let mut declared: Vec<u64> = segments.keys().copied().collect();
    declared.sort_unstable();
    ensure!(order == declared, "schedule orders segments {:?} but {declared:?} exist", schedule.segments);
    let slowest = model.segments.iter().map(|s| s.cycles).max().unwrap_or(0);
    ensure!(schedule.initiation_interval >= slowest, "II {} is shorter than segment cycles {slowest}", schedule.initiation_interval);
    ensure!(schedule.latency >= schedule.initiation_interval, "latency below II");
    Ok(())
}
