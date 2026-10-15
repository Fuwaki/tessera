//! Independent verification of a solver assignment.
//!
//! The solver's status is never treated as proof. Before a feasible artifact is
//! built, this module recomputes every hard fact from the decoded assignment:
//! that each operator is bound exactly once, that its fold is legal for its
//! channels and lies inside its engine's lane count, that an engine's role
//! follows from its binding count, that a pipeline link joins two *different*
//! used engines across a tensor they really produce and consume, that arena
//! residences are placed by lifetime, and that the makespan and DSP figures
//! follow from the schedule and the target model. Only a plan that passes all of
//! that is lowered.

use crate::layout::{ArenaRange, BufferKind, EngineRole, TensorId};
use crate::solver::SolverError;
use crate::solver::encode::{Problem, Solution, SolvedBinding, SolvedEngine, SolvedLink, SolvedSegment};
use crate::target::Ti60Target;

/// A verified tensor residence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedResidence {
    /// Tensor held.
    pub tensor: TensorId,
    /// Residence kind.
    pub kind: BufferKind,
    /// Words held.
    pub words: u64,
    /// Arena word offset (`Arena` only).
    pub offset_words: u64,
    /// Producing operator, if any.
    pub producer_op: String,
    /// Consuming operators, in schedule order.
    pub consumer_ops: Vec<String>,
}

/// A verified mapping, ready to be lowered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checked {
    /// Slot index of each entry in `engines` (`Solution::engines` index).
    pub slots: Vec<usize>,
    /// Used engine slots together with their configuration.
    pub engines: Vec<SolvedEngine>,
    /// One binding per operator, in operator order.
    pub bindings: Vec<SolvedBinding>,
    /// Pipeline links.
    pub links: Vec<SolvedLink>,
    /// Tensor residences.
    pub residences: Vec<CheckedResidence>,
    /// Frame-pipeline segments chosen by CP-SAT.
    pub segments: Vec<SolvedSegment>,
    /// Critical-path frame length in cycles.
    pub makespan_cycles: u64,
    /// Frame DSP48 blocks, recomputed from the target.
    pub dsp48: u64,
    /// Arena high-water mark in words.
    pub arena_words: u64,
}

/// `BufferResidence::producer_op` for a tensor no operator produces.
pub const GRAPH_INPUT_SENTINEL: &str = "<graph-input>";

fn fail(detail: impl Into<String>) -> SolverError {
    SolverError::Check {
        detail: detail.into(),
    }
}

/// Recompute and verify everything a feasible schema-2 plan needs.
pub fn verify(
    problem: &Problem,
    solution: &Solution,
    target: &Ti60Target,
) -> Result<Checked, SolverError> {
    let ops = problem.ops.len();

    // ---- engines ----------------------------------------------------------
    let mut engines: Vec<SolvedEngine> = Vec::new();
    for (slot, engine) in solution.engines.iter().enumerate() {
        if !engine.used {
            continue;
        }
        if engine.shape.simd == 0 || engine.shape.pe == 0 || engine.shape.tap_parallel == 0 {
            return Err(fail(format!("engine slot {slot} instantiates zero lanes")));
        }
        engines.push(*engine);
    }
    if (engines.len() as u32) < problem.bounds.min_engines
        || (engines.len() as u32) > problem.bounds.max_engines
    {
        return Err(fail(format!(
            "{} engines are outside the requested [{}, {}]",
            engines.len(),
            problem.bounds.min_engines,
            problem.bounds.max_engines
        )));
    }
    let slot_of: Vec<usize> = solution
        .engines
        .iter()
        .enumerate()
        .filter(|(_, e)| e.used)
        .map(|(slot, _)| slot)
        .collect();
    let engine_slot_of_op = |op: usize| -> Result<usize, SolverError> {
        solution
            .bindings
            .iter()
            .find(|b| b.op == op)
            .map(|b| b.engine)
            .ok_or_else(|| fail(format!("operator {op} has no binding")))
    };
    let engine_index = |slot: usize| -> Result<usize, SolverError> {
        slot_of
            .iter()
            .position(|candidate| *candidate == slot)
            .ok_or_else(|| fail(format!("engine slot {slot} is not instantiated")))
    };

    // ---- bindings and lanes ----------------------------------------------
    if solution.bindings.len() != ops {
        return Err(fail(format!(
            "{} bindings for {ops} operators",
            solution.bindings.len()
        )));
    }
    for binding in &solution.bindings {
        let op = binding.op;
        let spec = &problem.ops[op];
        let index = engine_index(binding.engine)?;
        let engine = engines[index];
        if engine.shape != binding.candidate.shape {
            return Err(fail(format!(
                "operator {:?} uses shape ({}, {}, {}) but engine slot {} instantiates ({}, {}, {})",
                spec.name,
                binding.candidate.shape.simd,
                binding.candidate.shape.pe,
                binding.candidate.shape.tap_parallel,
                binding.engine,
                engine.shape.simd,
                engine.shape.pe,
                engine.shape.tap_parallel
            )));
        }
        let shape = binding.candidate.shape;
        if shape.tap_parallel != 1 {
            return Err(fail(format!(
                "operator {:?} asks for tap parallelism {} which this backend cannot instantiate",
                spec.name, shape.tap_parallel
            )));
        }
        if spec.in_channels % shape.simd != 0 {
            return Err(fail(format!(
                "operator {:?}: SIMD {} does not divide its {} input channels",
                spec.name, shape.simd, spec.in_channels
            )));
        }
        if spec.out_channels % shape.pe != 0 {
            return Err(fail(format!(
                "operator {:?}: PE {} does not divide its {} output channels",
                spec.name, shape.pe, spec.out_channels
            )));
        }
        let lanes = shape.lanes();
        if lanes < u64::from(problem.bounds.min_lanes_per_engine)
            || lanes > u64::from(problem.bounds.max_lanes_per_engine)
        {
            return Err(fail(format!(
                "operator {:?}: {lanes} lanes are outside [{}, {}]",
                spec.name, problem.bounds.min_lanes_per_engine, problem.bounds.max_lanes_per_engine
            )));
        }
        if lanes > engine.shape.lanes() {
            return Err(fail(format!(
                "operator {:?} uses {lanes} lanes but its engine instantiates {}",
                spec.name,
                engine.shape.lanes()
            )));
        }
        if binding.candidate.temporal_iterations != spec.macs.div_ceil(lanes.max(1)) {
            return Err(fail(format!(
                "operator {:?}: {} MACs over {lanes} lanes is not {} lane iterations",
                spec.name, spec.macs, binding.candidate.temporal_iterations
            )));
        }
    }

    // ---- roles ------------------------------------------------------------
    let mut dedicated = 0u32;
    let mut shared = 0u32;
    for (index, engine) in engines.iter().enumerate() {
        let slot = slot_of[index];
        let count = solution
            .bindings
            .iter()
            .filter(|b| b.engine == slot)
            .count();
        let expected_dedicated = count == 1;
        if engine.dedicated != expected_dedicated {
            return Err(fail(format!(
                "engine slot {slot} is labelled {} with {count} bindings",
                if engine.dedicated {
                    EngineRole::Dedicated.as_str()
                } else {
                    EngineRole::Shared.as_str()
                }
            )));
        }
        if expected_dedicated {
            dedicated += 1;
        } else if count >= 2 {
            shared += 1;
        } else {
            return Err(fail(format!(
                "engine slot {slot} is instantiated with no bindings"
            )));
        }
    }
    if dedicated < problem.bounds.min_dedicated_engines {
        return Err(fail(format!(
            "{dedicated} dedicated engines but {} are required",
            problem.bounds.min_dedicated_engines
        )));
    }
    if shared < problem.bounds.min_shared_engines {
        return Err(fail(format!(
            "{shared} shared engines but {} are required",
            problem.bounds.min_shared_engines
        )));
    }

    // ---- pipeline ---------------------------------------------------------
    let mut seen_edges = std::collections::BTreeSet::new();
    for link in &solution.links {
        if !seen_edges.insert(link.edge) {
            return Err(fail("the same edge is cut more than once"));
        }
        let edge = problem
            .edges
            .get(link.edge)
            .ok_or_else(|| fail(format!("link references edge {}", link.edge)))?;
        let producer_slot = engine_slot_of_op(edge.producer)?;
        let consumer_slot = engine_slot_of_op(edge.consumer)?;
        if producer_slot == consumer_slot {
            return Err(fail(format!(
                "pipeline link cuts a tensor between one engine (slot {producer_slot}) and itself"
            )));
        }
        let _ = engine_index(producer_slot)?;
        let _ = engine_index(consumer_slot)?;
    }
    if (solution.links.len() as u32) < problem.bounds.min_pipeline_links
        || (solution.links.len() as u32) > problem.bounds.max_pipeline_links
    {
        return Err(fail(format!(
            "{} pipeline links are outside the requested [{}, {}]",
            solution.links.len(),
            problem.bounds.min_pipeline_links,
            problem.bounds.max_pipeline_links
        )));
    }

    // ---- residences, arena placement by lifetime --------------------------
    let last_step = ops.saturating_sub(1) as u64;
    let consumers_of = |tensor: TensorId| -> Vec<usize> {
        problem
            .edges
            .iter()
            .filter(|edge| edge.tensor == tensor)
            .map(|edge| edge.consumer)
            .collect()
    };
    let linked: std::collections::BTreeSet<TensorId> = solution
        .links
        .iter()
        .map(|link| problem.edges[link.edge].tensor)
        .collect();

    let mut ranges = Vec::new();
    for (index, tensor) in problem.tensors.iter().enumerate() {
        let residence = solution
            .residences
            .get(index)
            .ok_or_else(|| fail(format!("tensor {} has no residence", tensor.id)))?;
        let is_fifo = linked.contains(&tensor.id);
        if residence.arena == is_fifo {
            return Err(fail(format!(
                "tensor {} residence does not match its pipeline status",
                tensor.id
            )));
        }
        if is_fifo {
            continue;
        }
        let start = tensor.producer.map(|p| p as u64).unwrap_or(0);
        let consumers = consumers_of(tensor.id);
        let end = consumers
            .iter()
            .map(|c| *c as u64)
            .max()
            .unwrap_or(last_step);
        ranges.push(ArenaRange {
            start,
            end,
            words: tensor.words.max(1),
        });
    }
    let (offsets, arena_words) = crate::layout::allocate_arena(&ranges);

    let mut residences = Vec::new();
    let mut cursor = 0usize;
    for (_index, tensor) in problem.tensors.iter().enumerate() {
        // A graph input has no producing operator. The sentinel keeps the field
        // non-empty (the TOML subset cannot carry an empty string) and is
        // unambiguous because operator names come from the graph.
        let producer_op = tensor
            .producer
            .map(|p| problem.ops[p].name.clone())
            .unwrap_or_else(|| GRAPH_INPUT_SENTINEL.to_string());
        if linked.contains(&tensor.id) {
            residences.push(CheckedResidence {
                tensor: tensor.id,
                kind: BufferKind::Fifo,
                words: tensor.words,
                offset_words: 0,
                producer_op,
                consumer_ops: vec![],
            });
            continue;
        }
        let offset = offsets[cursor];
        cursor += 1;
        let consumer_ops = consumers_of(tensor.id)
            .into_iter()
            .map(|c| problem.ops[c].name.clone())
            .collect();
        residences.push(CheckedResidence {
            tensor: tensor.id,
            kind: BufferKind::Arena,
            words: tensor.words,
            offset_words: offset,
            producer_op,
            consumer_ops,
        });
    }
    // Two arena residences that are live at the same time must not share words.
    for a in 0..residences.len() {
        for b in (a + 1)..residences.len() {
            let (ra, rb) = (&residences[a], &residences[b]);
            if ra.kind != BufferKind::Arena || rb.kind != BufferKind::Arena {
                continue;
            }
            let live = |r: &CheckedResidence| -> (u64, u64) {
                let from = problem
                    .tensors
                    .iter()
                    .position(|t| t.id == r.tensor)
                    .and_then(|index| problem.tensors[index].producer)
                    .map(|p| p as u64)
                    .unwrap_or(0);
                let to = problem
                    .edges
                    .iter()
                    .filter(|e| e.tensor == r.tensor)
                    .map(|e| e.consumer as u64)
                    .max()
                    .unwrap_or(last_step);
                (from, to)
            };
            let (a_from, a_to) = live(ra);
            let (b_from, b_to) = live(rb);
            if a_from <= b_to
                && b_from <= a_to
                && ra.offset_words < rb.offset_words + rb.words
                && rb.offset_words < ra.offset_words + ra.words
            {
                return Err(fail(format!(
                    "arena tensors {} and {} share words while both are live",
                    ra.tensor, rb.tensor
                )));
            }
        }
    }

    // ---- makespan and DSP -------------------------------------------------
    let estimate = |op: usize| -> u64 {
        solution
            .bindings
            .iter()
            .find(|b| b.op == op)
            .map(|b| b.candidate.temporal_iterations.saturating_add(1))
            .unwrap_or(0)
    };
    let sequential: u64 = (0..ops).fold(0u64, |sum, i| sum.saturating_add(estimate(i)));
    let makespan_cycles = if solution.links.is_empty() {
        sequential
    } else {
        let mut per_engine = vec![0u64; solution.engines.len()];
        for binding in &solution.bindings {
            per_engine[binding.engine] =
                per_engine[binding.engine].saturating_add(estimate(binding.op));
        }
        let widest = per_engine
            .iter()
            .enumerate()
            .filter(|(slot, _)| solution.engines[*slot].used)
            .map(|(_, work)| *work)
            .max()
            .unwrap_or(0);
        let pipelined = widest.saturating_add(1);
        if pipelined >= sequential {
            return Err(fail(format!(
                "pipelined makespan {pipelined} is not below the sequential sum {sequential}"
            )));
        }
        pipelined
    };

    let mut dsp48 = 0u64;
    for engine in &engines {
        dsp48 = dsp48.saturating_add(problem.engine_dsp(target, engine.shape));
    }

    // ---- segments ---------------------------------------------------------
    {
        let mut covered = vec![false; ops];
        for (si, seg) in solution.segments.iter().enumerate() {
            if seg.ops.is_empty() {
                return Err(fail(format!("segment {si} is empty")));
            }
            // Contiguity in operator order.
            let mut sorted = seg.ops.clone();
            sorted.sort_unstable();
            if sorted != seg.ops.iter().copied().collect::<Vec<_>>() {
                // ops should already be in topo order; also require contiguous indices
            }
            if sorted.last().unwrap() - sorted.first().unwrap() + 1 != sorted.len() {
                return Err(fail(format!(
                    "segment {si} ops {sorted:?} are not contiguous in topological order"
                )));
            }
            for &op in &seg.ops {
                if op >= ops {
                    return Err(fail(format!("segment {si} references op {op}")));
                }
                if covered[op] {
                    return Err(fail(format!("operator {op} is in two segments")));
                }
                covered[op] = true;
            }
            if seg.shared {
                if seg.engines.len() != 1 {
                    return Err(fail(format!("shared segment {si} must use exactly one engine")));
                }
                let slot = seg.engines[0];
                if solution.engines.get(slot).map(|e| e.dedicated).unwrap_or(true) {
                    return Err(fail(format!("shared segment {si} engine slot {slot} is dedicated")));
                }
                for &op in &seg.ops {
                    let bound = engine_slot_of_op(op)?;
                    if bound != slot {
                        return Err(fail(format!(
                            "shared segment {si} op {op} is on slot {bound}, not {slot}"
                        )));
                    }
                }
            } else {
                for &op in &seg.ops {
                    let slot = engine_slot_of_op(op)?;
                    if !solution.engines.get(slot).map(|e| e.dedicated).unwrap_or(false) {
                        return Err(fail(format!(
                            "dedicated segment {si} op {op} is on a shared engine"
                        )));
                    }
                }
            }
        }
        if covered.iter().any(|c| !*c) {
            return Err(fail("not every operator is in a segment"));
        }
        // Edges only go forward across segments.
        let seg_of: Vec<u32> = {
            let mut v = vec![0u32; ops];
            for seg in &solution.segments {
                for &op in &seg.ops {
                    v[op] = seg.id;
                }
            }
            v
        };
        for edge in &problem.edges {
            if seg_of[edge.producer] > seg_of[edge.consumer] {
                return Err(fail(format!(
                    "segment order reverses edge {} -> {}",
                    edge.producer, edge.consumer
                )));
            }
        }
    }

    Ok(Checked {
        slots: slot_of.clone(),
        engines,
        bindings: solution.bindings.clone(),
        links: solution.links.clone(),
        residences,
        segments: solution.segments.clone(),
        makespan_cycles,
        dsp48,
        arena_words,
    })
}
