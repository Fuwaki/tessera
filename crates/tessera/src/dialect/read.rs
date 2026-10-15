//! Read a tessera module into plain Rust data.
//!
//! The verifier and the solver both consume this view; it is a read of the
//! MLIR module, not a separate IR — every field comes from an op operand,
//! result type or attribute, and decisions are written back as ops
//! ([`super::write`]).

use std::collections::HashMap;

use melior::ir::attribute::{
    ArrayAttribute, BoolAttribute, DenseI64ArrayAttribute, IntegerAttribute, StringAttribute,
};
use melior::ir::block::BlockArgument;
use melior::ir::operation::{OperationLike, OperationRef, OperationResult};
use melior::ir::r#type::{IntegerType, RankedTensorType, ShapedTypeLike, DimSize};
use melior::ir::{Attribute, BlockLike, Module, RegionLike, Type, Value, ValueLike};

use super::DialectError;

fn err(detail: impl Into<String>) -> DialectError {
    DialectError::Verify { detail: detail.into() }
}

/// Where an operator operand comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    /// Function argument (0 = graph input; others are weight refs).
    Arg(usize),
    /// Result of the operator with this `index`.
    Op(u32),
}

/// A static tensor type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTy {
    pub shape: Vec<u64>,
    pub bits: u32,
}

impl TensorTy {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Operator op kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Conv2d,
    PointwiseConv2d,
    DepthwiseConv2d,
    Add,
    Upsample,
    Relu,
    Requant,
    Quantize,
}

impl OpKind {
    pub fn is_conv(self) -> bool {
        matches!(self, OpKind::Conv2d | OpKind::PointwiseConv2d | OpKind::DepthwiseConv2d)
    }
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Conv2d => "conv2d",
            OpKind::PointwiseConv2d => "pointwise_conv2d",
            OpKind::DepthwiseConv2d => "depthwise_conv2d",
            OpKind::Add => "add",
            OpKind::Upsample => "nearest_upsample2x",
            OpKind::Relu => "relu",
            OpKind::Requant => "requant",
            OpKind::Quantize => "quantize",
        }
    }
}

/// One `tessera.<operator>` op.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatorOp {
    pub index: u32,
    pub name: String,
    pub kind: OpKind,
    pub operands: Vec<Source>,
    pub operand_types: Vec<TensorTy>,
    pub output: TensorTy,
    pub kernel: [u64; 2],
    pub stride: [u64; 2],
    pub pad: u64,
    pub reflect: bool,
    pub groups: u64,
    pub shift: u64,
    pub clamp: [i64; 2],
    pub multiplier: Vec<i64>,
    pub scale: u64,
    /// Exact inclusive code interval of the result.
    pub range: [i64; 2],
}

impl OperatorOp {
    /// Activation operands (graph input or other operators).
    pub fn activations(&self) -> impl Iterator<Item = (Source, &TensorTy)> {
        let n = match self.kind {
            OpKind::Add => 2,
            _ => 1,
        };
        self.operands.iter().copied().zip(self.operand_types.iter()).take(n)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOp {
    pub name: String,
    pub mode: String,
    pub simd: u64,
    pub pe: u64,
    pub vectors: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindOp {
    pub op: u32,
    pub engine: String,
    pub segment: u64,
    pub step: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferOp {
    /// Producing operator index; `None` is the graph input.
    pub producer: Option<u32>,
    pub residence: String,
    pub banks: u64,
    pub base: u64,
    pub words: u64,
    pub first_step: u64,
    pub last_step: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentOp {
    pub index: u64,
    pub mode: String,
    pub ops: Vec<u32>,
    pub engines: Vec<String>,
    pub cycles: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleOp {
    pub segments: Vec<u64>,
    pub initiation_interval: u64,
    pub latency: u64,
}

/// Everything in the graph function of a tessera module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TesseraModule {
    pub function: String,
    /// Code interval of the graph input (`tessera.input_range`).
    pub input_range: Option<[i64; 2]>,
    pub arguments: Vec<TensorTy>,
    pub operators: Vec<OperatorOp>,
    /// Operator index of each graph result.
    pub results: Vec<u32>,
    pub engines: Vec<EngineOp>,
    pub binds: Vec<BindOp>,
    pub buffers: Vec<BufferOp>,
    pub segments: Vec<SegmentOp>,
    pub schedule: Option<ScheduleOp>,
}

impl TesseraModule {
    pub fn operator(&self, index: u32) -> Option<&OperatorOp> {
        self.operators.iter().find(|op| op.index == index)
    }
    pub fn has_decisions(&self) -> bool {
        !self.engines.is_empty() || !self.binds.is_empty() || self.schedule.is_some()
    }
}

pub(crate) fn tensor_ty(ty: Type) -> Result<TensorTy, DialectError> {
    let t = RankedTensorType::try_from(ty).map_err(|_| err(format!("{ty} is not a ranked tensor")))?;
    let mut shape = Vec::new();
    for i in 0..t.rank() {
        match t.dim_size(i).map_err(|e| err(e.to_string()))? {
            DimSize::Static(n) => shape.push(n),
            DimSize::Dynamic => return Err(err(format!("{ty} has a dynamic dimension"))),
        }
    }
    let bits = IntegerType::try_from(t.element())
        .map_err(|_| err(format!("{ty} is not an integer tensor")))?
        .width();
    Ok(TensorTy { shape, bits })
}

fn attr<'c>(op: &OperationRef<'c, '_>, name: &str) -> Result<Attribute<'c>, DialectError> {
    op.attribute(name).map_err(|_| {
        err(format!("{} has no `{name}` attribute", op.name().as_string_ref().as_str().unwrap_or("?")))
    })
}
fn int(op: &OperationRef, name: &str) -> Result<i64, DialectError> {
    IntegerAttribute::try_from(attr(op, name)?)
        .map(|a| a.value())
        .map_err(|_| err(format!("`{name}` is not an integer")))
}
fn uint(op: &OperationRef, name: &str) -> Result<u64, DialectError> {
    let v = int(op, name)?;
    u64::try_from(v).map_err(|_| err(format!("`{name}` = {v} is negative")))
}
fn string(op: &OperationRef, name: &str) -> Result<String, DialectError> {
    StringAttribute::try_from(attr(op, name)?)
        .map(|a| a.value().to_string())
        .map_err(|_| err(format!("`{name}` is not a string")))
}
fn array(op: &OperationRef, name: &str) -> Result<Vec<i64>, DialectError> {
    let a = DenseI64ArrayAttribute::try_from(attr(op, name)?)
        .map_err(|_| err(format!("`{name}` is not an i64 array")))?;
    (0..a.len()).map(|i| a.element(i).map_err(|e| err(e.to_string()))).collect()
}
fn pair_u(op: &OperationRef, name: &str) -> Result<[u64; 2], DialectError> {
    match array(op, name)?.as_slice() {
        [a, b] if *a >= 0 && *b >= 0 => Ok([*a as u64, *b as u64]),
        other => Err(err(format!("`{name}` must be two non-negative values, got {other:?}"))),
    }
}
fn bool_attr(op: &OperationRef, name: &str) -> Result<bool, DialectError> {
    let a = attr(op, name)?;
    if let Ok(b) = BoolAttribute::try_from(a) {
        return Ok(b.value());
    }
    IntegerAttribute::try_from(a)
        .map(|i| i.value() != 0)
        .map_err(|_| err(format!("`{name}` is not a bool")))
}
fn strings(op: &OperationRef, name: &str) -> Result<Vec<String>, DialectError> {
    let a = ArrayAttribute::try_from(attr(op, name)?)
        .map_err(|_| err(format!("`{name}` is not an array")))?;
    (0..a.len())
        .map(|i| {
            let e = a.element(i).map_err(|e| err(e.to_string()))?;
            StringAttribute::try_from(e)
                .map(|s| s.value().to_string())
                .map_err(|_| err(format!("`{name}` holds a non-string")))
        })
        .collect()
}
fn index_u32(v: i64, what: &str) -> Result<u32, DialectError> {
    u32::try_from(v).map_err(|_| err(format!("{what} {v} out of range")))
}

/// Read the (single) graph function of a tessera module.
pub fn read(module: &Module) -> Result<TesseraModule, DialectError> {
    let func = module
        .body()
        .first_operation()
        .filter(|op| op.name().as_string_ref().as_str() == Ok("func.func"))
        .ok_or_else(|| err("module has no graph function"))?;
    let mut out = TesseraModule {
        function: string(&func, "sym_name")?,
        input_range: if func.has_attribute("tessera.input_range") {
            match array(&func, "tessera.input_range")?.as_slice() {
                [lo, hi] => Some([*lo, *hi]),
                _ => return Err(err("tessera.input_range must have two values")),
            }
        } else {
            None
        },
        ..TesseraModule::default()
    };
    let entry = func
        .region(0)
        .ok()
        .and_then(|r| r.first_block())
        .ok_or_else(|| err("graph function has no body"))?;
    for i in 0..entry.argument_count() {
        out.arguments.push(tensor_ty(entry.argument(i).map_err(|e| err(e.to_string()))?.r#type())?);
    }
    let entry_key = entry.to_raw().ptr as usize;
    let mut producer: HashMap<usize, u32> = HashMap::new();
    let source = |value: Value, producer: &HashMap<usize, u32>| -> Result<Source, DialectError> {
        if let Ok(arg) = BlockArgument::try_from(value) {
            if arg.owner().to_raw().ptr as usize == entry_key {
                return Ok(Source::Arg(arg.argument_number()));
            }
            return Err(err("operand is a nested block argument"));
        }
        let owner = OperationResult::try_from(value)
            .map_err(|_| err("operand is not an op result"))?
            .owner();
        producer
            .get(&(owner.to_raw().ptr as usize))
            .copied()
            .map(Source::Op)
            .ok_or_else(|| err("operand is not produced by a tessera operator op"))
    };

    let mut cursor = entry.first_operation();
    while let Some(op) = cursor {
        cursor = op.next_in_block();
        let name = op.name().as_string_ref().as_str().unwrap_or("").to_string();
        let kind = match name.as_str() {
            "tessera.conv" => match string(&op, "kind")?.as_str() {
                "conv2d" => Some(OpKind::Conv2d),
                "pointwise_conv2d" => Some(OpKind::PointwiseConv2d),
                "depthwise_conv2d" => Some(OpKind::DepthwiseConv2d),
                other => return Err(err(format!("unknown conv kind {other:?}"))),
            },
            "tessera.add" => Some(OpKind::Add),
            "tessera.upsample" => Some(OpKind::Upsample),
            "tessera.relu" => Some(OpKind::Relu),
            "tessera.requant" => Some(OpKind::Requant),
            "tessera.quantize" => Some(OpKind::Quantize),
            _ => None,
        };
        if let Some(kind) = kind {
            let index = if op.has_attribute("index") {
                index_u32(int(&op, "index")?, "operator index")?
            } else {
                out.operators.len() as u32
            };
            let mut operands = Vec::new();
            let mut operand_types = Vec::new();
            for i in 0..op.operand_count() {
                let v = op.operand(i).map_err(|e| err(e.to_string()))?;
                operands.push(source(v, &producer)?);
                operand_types.push(tensor_ty(v.r#type())?);
            }
            let output = tensor_ty(op.result(0).map_err(|e| err(e.to_string()))?.r#type())?;
            let has = |n: &str| op.has_attribute(n);
            let operator = OperatorOp {
                index,
                name: if has("name") { string(&op, "name")? } else { format!("{}#{index}", kind.name()) },
                kind,
                operands,
                operand_types,
                output,
                kernel: if has("kernel") { pair_u(&op, "kernel")? } else { [1, 1] },
                stride: if has("stride") { pair_u(&op, "stride")? } else { [1, 1] },
                pad: if has("pad") { uint(&op, "pad")? } else { 0 },
                reflect: if has("reflect") { bool_attr(&op, "reflect")? } else { false },
                groups: if has("groups") { uint(&op, "groups")? } else { 1 },
                shift: if has("shift") { uint(&op, "shift")? } else { 0 },
                clamp: if has("clamp") {
                    match array(&op, "clamp")?.as_slice() {
                        [lo, hi] => [*lo, *hi],
                        other => return Err(err(format!("clamp must have two values, got {other:?}"))),
                    }
                } else {
                    [i64::MIN, i64::MAX]
                },
                multiplier: if has("multiplier") { array(&op, "multiplier")? } else { Vec::new() },
                scale: if has("scale") { uint(&op, "scale")? } else { 1 },
                range: [0, 0],
            };
            let mut operator = operator;
            operator.range = if has("range") {
                match array(&op, "range")?.as_slice() {
                    [lo, hi] => [*lo, *hi],
                    other => return Err(err(format!("range must have two values, got {other:?}"))),
                }
            } else {
                operator.clamp
            };
            producer.insert(op.to_raw().ptr as usize, index);
            out.operators.push(operator);
            continue;
        }
        match name.as_str() {
            "tessera.engine" => out.engines.push(EngineOp {
                name: string(&op, "sym_name")?,
                mode: string(&op, "mode")?,
                simd: uint(&op, "simd")?,
                pe: uint(&op, "pe")?,
                vectors: uint(&op, "vectors")?,
            }),
            "tessera.bind" => out.binds.push(BindOp {
                op: index_u32(int(&op, "op")?, "bind op")?,
                engine: string(&op, "engine")?,
                segment: uint(&op, "segment")?,
                step: uint(&op, "step")?,
            }),
            "tessera.buffer" => out.buffers.push(BufferOp {
                producer: match int(&op, "producer")? {
                    -1 => None,
                    p => Some(index_u32(p, "buffer producer")?),
                },
                residence: string(&op, "residence")?,
                banks: uint(&op, "banks")?,
                base: uint(&op, "base")?,
                words: uint(&op, "words")?,
                first_step: uint(&op, "first_step")?,
                last_step: uint(&op, "last_step")?,
            }),
            "tessera.segment" => out.segments.push(SegmentOp {
                index: uint(&op, "index")?,
                mode: string(&op, "mode")?,
                ops: array(&op, "ops")?
                    .into_iter()
                    .map(|v| index_u32(v, "segment op"))
                    .collect::<Result<_, _>>()?,
                engines: strings(&op, "engines")?,
                cycles: uint(&op, "cycles")?,
            }),
            "tessera.schedule" => {
                if out.schedule.is_some() {
                    return Err(err("more than one tessera.schedule"));
                }
                out.schedule = Some(ScheduleOp {
                    segments: array(&op, "segments")?.into_iter().map(|v| v.max(0) as u64).collect(),
                    initiation_interval: uint(&op, "initiation_interval")?,
                    latency: uint(&op, "latency")?,
                });
            }
            "func.return" => {
                for i in 0..op.operand_count() {
                    match source(op.operand(i).map_err(|e| err(e.to_string()))?, &producer)? {
                        Source::Op(index) => out.results.push(index),
                        Source::Arg(_) => return Err(err("graph returns a function argument")),
                    }
                }
            }
            other => return Err(err(format!("unexpected op {other:?} in a tessera module"))),
        }
    }
    Ok(out)
}
