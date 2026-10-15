//! Tensor-level linalg -> tessera dialect, in process.
//!
//! The emitter tags the final linalg op of every operator with a discardable
//! `tessera.op` dictionary (kind, kernel, stride, padding, quantization).
//! This pass:
//!
//! 1. runs graph-level tensor optimization (`canonicalize`, `cse`) on the
//!    linalg module;
//! 2. walks the graph function and, for every annotated op, finds its
//!    operands by a depth-first search over the def-chains of the ops that
//!    implement it (including values captured inside nested regions). The
//!    search stops at another annotated op (an activation edge) or at a
//!    function argument (argument 0 is the graph input, an activation edge;
//!    every other argument is a weight ref: filter, bias, M0);
//! 3. builds one `tessera.<op>` per annotated op with OperationBuilder in the
//!    same context, with tensor edges as SSA values, into a fresh function of
//!    the same signature. The linalg implementation is dropped.

use std::collections::{HashMap, HashSet};

use melior::Context;
use melior::dialect::func;
use melior::ir::attribute::{
    DictionaryAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::block::BlockArgument;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef, OperationResult};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type,
    Value, ValueLike,
};
use melior::pass::PassManager;
use melior::utility::{parse_pass_pipeline, register_all_passes};

use super::DialectError;

/// Discardable attribute carrying an operator's semantics on its final op.
pub const ANNOTATION: &str = "tessera.op";

/// Graph-level tensor optimization run before conversion.
pub const TENSOR_PIPELINE: &str = "builtin.module(canonicalize,cse)";

fn err(detail: impl Into<String>) -> DialectError {
    DialectError::Convert { detail: detail.into() }
}

/// Where a DFS over def-chains ended.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Leaf {
    /// Result of an annotated op (key: raw pointer of the op).
    Op(usize),
    /// Function argument by number.
    Arg(usize),
}

/// Parse tensor-level linalg text, optimize it, and convert it to a module
/// holding only `tessera.*` operator ops (plus `func.func`/`func.return`).
/// `context` must have the stock dialects and `tessera` loaded.
pub fn linalg_to_tessera<'c>(
    context: &'c Context,
    linalg: &str,
) -> Result<Module<'c>, DialectError> {
    let mut source =
        Module::parse(context, linalg).ok_or_else(|| err("could not parse the linalg module"))?;
    register_all_passes();
    let manager = PassManager::new(context);
    parse_pass_pipeline(manager.as_operation_pass_manager(), TENSOR_PIPELINE)
        .map_err(|e| err(format!("tensor pipeline: {e}")))?;
    manager
        .run(&mut source)
        .map_err(|e| err(format!("tensor pipeline failed: {e}")))?;

    let body = source.body();
    let graph = body
        .first_operation()
        .filter(|op| op.name().as_string_ref().as_str() == Ok("func.func"))
        .ok_or_else(|| err("module has no graph function"))?;
    let name = graph
        .attribute("sym_name")
        .map_err(|_| err("graph function has no name"))?;
    let name = StringAttribute::try_from(name).map_err(|_| err("bad function name"))?;
    let fty = graph
        .attribute("function_type")
        .map_err(|_| err("graph function has no type"))?;
    let fty = TypeAttribute::try_from(fty).map_err(|_| err("bad function type"))?;
    let fty = FunctionType::try_from(fty.value()).map_err(|_| err("bad function type"))?;
    let entry = graph
        .region(0)
        .ok()
        .and_then(|r| r.first_block())
        .ok_or_else(|| err("graph function has no body"))?;

    // Annotated ops in program order.
    let mut annotated: Vec<OperationRef> = Vec::new();
    let mut cursor = entry.first_operation();
    while let Some(op) = cursor {
        if op.has_attribute(ANNOTATION) {
            annotated.push(op);
        }
        cursor = op.next_in_block();
    }
    if annotated.is_empty() {
        return Err(err("no op carries a tessera.op annotation"));
    }
    let annotated_keys: HashSet<usize> =
        annotated.iter().map(|op| op.to_raw().ptr as usize).collect();
    let entry_key = entry.to_raw().ptr as usize;

    let location = Location::unknown(context);
    let module = Module::new(location);
    let inputs: Vec<(Type, Location)> = (0..fty.input_count())
        .map(|i| (fty.input(i).expect("input"), location))
        .collect();
    let region = Region::new();
    let block = region.append_block(Block::new(&inputs));
    let mut produced: HashMap<usize, Value> = HashMap::new();

    for op in &annotated {
        let leaves = operand_leaves(*op, &annotated_keys, entry_key)?;
        let dict = op
            .attribute(ANNOTATION)
            .ok()
            .and_then(|a| DictionaryAttribute::try_from(a).ok())
            .ok_or_else(|| err("tessera.op is not a dictionary"))?;
        let kind = dict
            .element_by_name("kind")
            .and_then(|a| StringAttribute::try_from(a).ok())
            .map(|s| s.value().to_string())
            .ok_or_else(|| err("tessera.op has no kind"))?;
        let (op_name, extra): (&str, Vec<(Identifier, melior::ir::Attribute)>) =
            match kind.as_str() {
                "conv2d" | "pointwise_conv2d" | "depthwise_conv2d" => ("tessera.conv", vec![]),
                "add" => ("tessera.add", vec![]),
                "nearest_upsample2x" => ("tessera.upsample", vec![]),
                other => return Err(err(format!("unknown operator kind {other:?}"))),
            };
        let mut operands: Vec<Value> = Vec::new();
        for leaf in &leaves {
            operands.push(match leaf {
                Leaf::Arg(n) => block.argument(*n).map_err(|e| err(e.to_string()))?.into(),
                Leaf::Op(k) => *produced
                    .get(k)
                    .ok_or_else(|| err("annotated op used before it was produced"))?,
            });
        }
        let expected = if op_name == "tessera.conv" {
            4
        } else if op_name == "tessera.add" {
            2
        } else {
            1
        };
        if operands.len() != expected {
            return Err(err(format!(
                "{op_name} ({kind}) resolved {} operands, expected {expected}",
                operands.len()
            )));
        }
        let mut attributes = extra;
        for i in 0..dict.len() {
            let (id, value) = dict.element(i).map_err(|e| err(e.to_string()))?;
            if op_name != "tessera.conv" && id.as_string_ref().as_str() == Ok("kind") {
                continue;
            }
            attributes.push((id, value));
        }
        let result_type = op.result(0).map_err(|e| err(e.to_string()))?.r#type();
        let new = OperationBuilder::new(op_name, op.location())
            .add_operands(&operands)
            .add_results(&[result_type])
            .add_attributes(&attributes)
            .build()
            .map_err(|e| err(format!("{op_name}: {e}")))?;
        let new = block.append_operation(new);
        produced.insert(
            op.to_raw().ptr as usize,
            new.result(0).map_err(|e| err(e.to_string()))?.into(),
        );
    }

    // The graph result must be an annotated op's result.
    let terminator = entry.terminator().ok_or_else(|| err("graph has no return"))?;
    let mut results = Vec::new();
    for i in 0..terminator.operand_count() {
        let value = terminator.operand(i).map_err(|e| err(e.to_string()))?;
        let owner = OperationResult::try_from(value)
            .map_err(|_| err("graph returns a function argument"))?
            .owner();
        results.push(
            *produced
                .get(&(owner.to_raw().ptr as usize))
                .ok_or_else(|| err("graph returns a value no tessera op produces"))?,
        );
    }
    // The unsigned output step is folded into the head epilogue in linalg;
    // in the dialect it is an explicit `tessera.quantize` on the graph result.
    if let Some(quantize) = graph
        .attribute("tessera.quantize_output")
        .ok()
        .and_then(|a| DictionaryAttribute::try_from(a).ok())
    {
        let [value] = results.as_slice() else {
            return Err(err("tessera.quantize_output on a graph with several results"));
        };
        let mut attributes = Vec::new();
        for i in 0..quantize.len() {
            attributes.push(quantize.element(i).map_err(|e| err(e.to_string()))?);
        }
        let op = OperationBuilder::new("tessera.quantize", location)
            .add_operands(&[*value])
            .add_results(&[value.r#type()])
            .add_attributes(&attributes)
            .build()
            .map_err(|e| err(format!("tessera.quantize: {e}")))?;
        let op = block.append_operation(op);
        results = vec![op.result(0).map_err(|e| err(e.to_string()))?.into()];
    }
    block.append_operation(func::r#return(&results, location));
    let mut facts = Vec::new();
    if let Ok(range) = graph.attribute("tessera.input_range") {
        facts.push((Identifier::new(context, "tessera.input_range"), range));
    }
    module.body().append_operation(func::func(
        context,
        name,
        TypeAttribute::new(fty.into()),
        region,
        &facts,
        location,
    ));
    if !module.as_operation().verify() {
        return Err(err("converted tessera module failed verification"));
    }
    Ok(module)
}

/// DFS over everything an annotated op is computed from, returning the
/// leaves in first-visit order.
fn operand_leaves(
    root: OperationRef,
    annotated: &HashSet<usize>,
    entry: usize,
) -> Result<Vec<Leaf>, DialectError> {
    let mut leaves = Vec::new();
    let mut seen_leaves = HashSet::new();
    let mut seen_ops = HashSet::new();
    visit_op(root, root, annotated, entry, &mut leaves, &mut seen_leaves, &mut seen_ops)?;
    Ok(leaves)
}

#[allow(clippy::too_many_arguments)]
fn visit_op(
    root: OperationRef,
    op: OperationRef,
    annotated: &HashSet<usize>,
    entry: usize,
    leaves: &mut Vec<Leaf>,
    seen_leaves: &mut HashSet<Leaf>,
    seen_ops: &mut HashSet<usize>,
) -> Result<(), DialectError> {
    if !seen_ops.insert(op.to_raw().ptr as usize) {
        return Ok(());
    }
    for i in 0..op.operand_count() {
        let value = op.operand(i).map_err(|e| err(e.to_string()))?;
        visit_value(root, value, annotated, entry, leaves, seen_leaves, seen_ops)?;
    }
    // Values captured by nested regions are dependencies too.
    for r in 0..op.region_count() {
        let Ok(region) = op.region(r) else { continue };
        let mut block = region.first_block();
        while let Some(b) = block {
            let mut inner = b.first_operation();
            while let Some(o) = inner {
                visit_op(root, o, annotated, entry, leaves, seen_leaves, seen_ops)?;
                inner = o.next_in_block();
            }
            block = b.next_in_region();
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn visit_value(
    root: OperationRef,
    value: Value,
    annotated: &HashSet<usize>,
    entry: usize,
    leaves: &mut Vec<Leaf>,
    seen_leaves: &mut HashSet<Leaf>,
    seen_ops: &mut HashSet<usize>,
) -> Result<(), DialectError> {
    if let Ok(arg) = BlockArgument::try_from(value) {
        // Block arguments of nested regions are local to the op.
        if arg.owner().to_raw().ptr as usize == entry {
            let leaf = Leaf::Arg(arg.argument_number());
            if seen_leaves.insert(leaf) {
                leaves.push(leaf);
            }
        }
        return Ok(());
    }
    let owner = OperationResult::try_from(value)
        .map_err(|_| err("value is neither a block argument nor an op result"))?
        .owner();
    let key = owner.to_raw().ptr as usize;
    if key != root.to_raw().ptr as usize && annotated.contains(&key) {
        let leaf = Leaf::Op(key);
        if seen_leaves.insert(leaf) {
            leaves.push(leaf);
        }
        return Ok(());
    }
    // Values defined inside the root's own regions are already covered.
    visit_op(root, owner, annotated, entry, leaves, seen_leaves, seen_ops)
}

/// Load a module that is either annotated tensor-level linalg or already
/// tessera ops, returning a tessera module in `context`.
pub fn load_tessera_module<'c>(
    context: &'c Context,
    text: &str,
) -> Result<Module<'c>, DialectError> {
    if text.contains("tessera.conv")
        || text.contains("\"tessera.conv\"")
        || text.contains("'tessera.conv'")
    {
        let module = Module::parse(context, text)
            .ok_or_else(|| err("could not parse the tessera module"))?;
        if !module.as_operation().verify() {
            return Err(err("tessera module failed IRDL verification"));
        }
        return Ok(module);
    }
    if text.contains(ANNOTATION) {
        return linalg_to_tessera(context, text);
    }
    Err(err(
        "module is neither annotated linalg (tessera.op) nor a tessera dialect module",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::register_in_fresh_context;
    use melior::ir::operation::OperationLike;

    fn op_names(module: &Module) -> Vec<String> {
        let func = module.body().first_operation().expect("func");
        let block = func.region(0).unwrap().first_block().unwrap();
        let mut names = Vec::new();
        let mut cursor = block.first_operation();
        while let Some(op) = cursor {
            names.push(op.name().as_string_ref().as_str().unwrap().to_string());
            cursor = op.next_in_block();
        }
        names
    }

    #[test]
    fn every_tessera_op_is_created_and_verified_in_process() {
        let context = register_in_fresh_context().expect("tessera dialect");
        let text = include_str!("../../dialects/tests/ops.mlir");
        let module = Module::parse(&context, text).expect("parse tessera ops");
        assert!(module.as_operation().verify());
        // A missing required attribute must be rejected by the IRDL definition.
        let broken = text.replace("kind = \"conv2d\", ", "");
        let rejected = Module::parse(&context, &broken)
            .map(|m| m.as_operation().verify())
            .unwrap_or(false);
        assert!(!rejected, "conv without kind must not verify");
    }

    #[test]
    fn sample_graph_converts_to_tessera_ops_with_ssa_edges() {
        let graph = crate::mlir::fixtures::sample_graph();
        let options = crate::mlir::fixtures::sample_options();
        let linalg = crate::mlir::emit_text(&graph, &options).expect("emit");
        let context = register_in_fresh_context().expect("tessera dialect");
        let module = linalg_to_tessera(&context, &linalg).expect("convert");
        let names = op_names(&module);
        assert!(names.iter().all(|n| n.starts_with("tessera.") || n == "func.return"), "{names:?}");
        assert!(names.iter().any(|n| n == "tessera.conv"), "{names:?}");
        let text = module.as_operation().to_string();
        assert!(!text.contains("linalg."), "{text}");
        eprintln!("{text}");
    }
}

#[cfg(test)]
mod verify_tests {
    use crate::dialect::{check, register_in_fresh_context, verify_module, linalg_to_tessera};
    use melior::ir::Module;

    #[test]
    fn hand_written_module_with_decisions_verifies() {
        let context = register_in_fresh_context().expect("dialect");
        let text = include_str!("../../dialects/tests/ops.mlir");
        let module = Module::parse(&context, text).expect("parse");
        let model = verify_module(&module).expect("verify");
        assert_eq!(model.operators.len(), 3);
        assert_eq!(model.binds.len(), 3);
        // Rules IRDL cannot see are caught by the Rust verifier.
        let mut bad = model.clone();
        bad.binds.pop();
        assert!(check(&bad).is_err(), "unbound operator");
        let mut bad = model.clone();
        bad.buffers[0].banks = 1;
        assert!(check(&bad).is_err(), "kbank with one bank");
        let mut bad = model.clone();
        bad.schedule.as_mut().unwrap().initiation_interval = 10;
        assert!(check(&bad).is_err(), "II below segment cycles");
    }

    #[test]
    fn converted_sample_graph_verifies_and_bad_edits_are_rejected() {
        let graph = crate::mlir::fixtures::sample_graph();
        let options = crate::mlir::fixtures::sample_options();
        let linalg = crate::mlir::emit_text(&graph, &options).expect("emit");
        let context = register_in_fresh_context().expect("dialect");
        let module = linalg_to_tessera(&context, &linalg).expect("convert");
        let model = verify_module(&module).expect("verify converted module");
        assert_eq!(model.input_range, Some([-127, 127]));
        let last = model.operators.last().unwrap();
        assert_eq!(last.kind, crate::dialect::read::OpKind::Quantize);
        assert_eq!(model.results, vec![last.index]);
        assert!(model.operators.iter().filter(|o| o.kind.is_conv()).all(|o| !o.multiplier.is_empty()));
        let mut bad = model.clone();
        bad.operators[0].stride = [2, 2];
        assert!(check(&bad).is_err(), "stride inconsistent with output shape");
        let mut bad = model.clone();
        bad.operators[0].clamp = [5, 1];
        assert!(check(&bad).is_err(), "empty clamp");
        let mut bad = model.clone();
        bad.operators.swap(0, 1);
        bad.operators.swap(1, 2);
        assert!(check(&bad).is_err(), "use before definition");
    }
}
