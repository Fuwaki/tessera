//! Encoding of the hardware-mapping search as a [`ConstraintModel`].
//!
//! The model is a genuine scheduling/binding problem, not a lookup of known
//! answers. Its variables are:
//!
//! * `use.{e}` — whether engine slot `e` is instantiated;
//! * `shape.{e}.{s}` — which spatial shape engine `e` instantiates;
//! * `bind.{i}.{e}` — which engine operator `i` runs on;
//! * `cand.{i}.{s}` — which fold candidate operator `i` uses;
//! * `start.{i}`, `dur.{i}` — its schedule position and cost;
//! * `makespan` — the frame length;
//! * `ded.{e}`/`shared.{e}` — its role, which is a *consequence* of its binding
//!   count;
//! * `link.{k}` — whether a pipeline cut overlaps producer and consumer;
//! * `arena.{t}`, `live.{t}`, `dem.{t}` — whether a tensor lives in the arena
//!   and for how long.
//!
//! From those it constrains: one candidate and one engine per operator, engine
//! shape agreement, engine lane bounds and count, dedicated/shared binding
//! counts, dependency order (relaxed only across a chosen pipeline link),
//! `NoOverlap` of the operators sharing an engine, and a `Cumulative` arena
//! whose demand is the resident tensors' word counts.
//!
//! The objective is lexicographic and expressed as a list: minimize the
//! makespan, then the total DSP48, then the engine count, then the total lane
//! count.

use std::collections::BTreeMap;

use crate::layout::{OpKind, TensorId};
use crate::solver::model::{Constraint, DemandInterval, Interval, Rel, Term, Var};
use crate::solver::model::{ConstraintModel, Sense};
use crate::solver::{Candidate, MappingBounds, SolverError, SpatialShape};
use crate::target::Ti60Target;

/// One operator of the problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSpec {
    /// Operator name.
    pub name: String,
    /// Operator kind.
    pub kind: OpKind,
    /// Multiply-accumulate products for the whole frame.
    pub macs: u64,
    /// Input channels.
    pub in_channels: u32,
    /// Output channels.
    pub out_channels: u32,
    /// Legal fold candidates.
    pub candidates: Vec<Candidate>,
}

/// A producer→consumer tensor crossing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeSpec {
    /// Producer operator index.
    pub producer: usize,
    /// Consumer operator index.
    pub consumer: usize,
    /// Tensor crossing the edge.
    pub tensor: TensorId,
}

/// A tensor that may occupy the arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Tensor identity.
    pub id: TensorId,
    /// Words (elements) held.
    pub words: u64,
    /// Operator index producing it, if any.
    pub producer: Option<usize>,
}

/// Everything the encoder needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// Operators.
    pub ops: Vec<OpSpec>,
    /// Producer→consumer crossings.
    pub edges: Vec<EdgeSpec>,
    /// Arena-eligible tensors.
    pub tensors: Vec<TensorSpec>,
    /// Every distinct spatial shape any operator may use.
    pub shapes: Vec<SpatialShape>,
    /// Caller bounds.
    pub bounds: MappingBounds,
    /// Activation operand width in bits.
    pub activation_bits: u32,
    /// Accumulator width in bits.
    pub accumulator_bits: u32,
    /// Weight operand width in bits.
    pub weight_bits: u32,
}

/// A solved engine slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedEngine {
    /// Whether the slot is instantiated.
    pub used: bool,
    /// Instantiated shape (meaningful only when `used`).
    pub shape: SpatialShape,
    /// Whether the engine is dedicated.
    pub dedicated: bool,
}

/// A solved operator binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolvedBinding {
    /// Operator index.
    pub op: usize,
    /// Engine slot.
    pub engine: usize,
    /// Chosen candidate.
    pub candidate: Candidate,
    /// Schedule start.
    pub start: i64,
}

/// A solved pipeline link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedLink {
    /// Edge index.
    pub edge: usize,
}

/// A solved tensor residence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolvedResidence {
    /// Whether the tensor is arena-resident (false when pipeline-crossed).
    pub arena: bool,
    /// Banks: 1 = local to a segment, 2 = K-bank rotating across segments.
    pub banks: u32,
}

/// One frame-pipeline segment chosen by CP-SAT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolvedSegment {
    /// Dense segment id from 0.
    pub id: u32,
    /// Operator indices in topological order.
    pub ops: Vec<usize>,
    /// Engine slots used by those operators.
    pub engines: Vec<usize>,
    /// Whether the segment uses a shared engine.
    pub shared: bool,
    /// Cycles = sum (shared) or max (dedicated) of binding durations.
    pub cycles: u64,
}

/// The decoded solver assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    /// Engine slots.
    pub engines: Vec<SolvedEngine>,
    /// Bindings, one per operator.
    pub bindings: Vec<SolvedBinding>,
    /// Chosen pipeline links.
    pub links: Vec<SolvedLink>,
    /// Per-tensor arena presence, in `Problem::tensors` order.
    pub residences: Vec<SolvedResidence>,
    /// Frame-pipeline segments, ordered by dense id.
    pub segments: Vec<SolvedSegment>,
    /// Frame length from the scheduler.
    pub makespan: i64,
}

impl Problem {
    /// Total DSP48 for one engine of `shape`, from the real target model.
    pub fn engine_dsp(&self, target: &Ti60Target, shape: SpatialShape) -> u64 {
        let lanes = crate::target::MacLanes::new(shape.lanes());
        let mac = target
            .map_mac(self.activation_bits, self.weight_bits, lanes)
            .map(|m| m.dsp48.get())
            .unwrap_or(0);
        let requant = target.map_requant(lanes, self.accumulator_bits).get();
        mac.saturating_add(requant)
    }

    /// Horizon no schedule can exceed: every operator run back to back.
    fn horizon(&self) -> i64 {
        self.ops
            .iter()
            .map(|op| {
                op.candidates
                    .iter()
                    .map(|c| c.temporal_iterations)
                    .max()
                    .unwrap_or(0)
            })
            .fold(0u64, |sum, d| sum.saturating_add(d))
            .saturating_add(1)
            .max(1) as i64
    }
}


fn seg_var(i: usize) -> String {
    format!("seg.{i}")
}
fn cross_var(k: usize) -> String {
    format!("cross.{k}")
}
fn kbank_var(t: usize) -> String {
    format!("kbank.{t}")
}

fn used(e: usize) -> String {
    format!("use.{e}")
}
fn shape_var(e: usize, s: usize) -> String {
    format!("shape.{e}.{s}")
}
fn bind_var(i: usize, e: usize) -> String {
    format!("bind.{i}.{e}")
}
fn cand_var(i: usize, s: usize) -> String {
    format!("cand.{i}.{s}")
}
fn start_var(i: usize) -> String {
    format!("start.{i}")
}
fn dur_var(i: usize) -> String {
    format!("dur.{i}")
}
fn link_var(k: usize) -> String {
    format!("link.{k}")
}
fn arena_var(t: usize) -> String {
    format!("arena.{t}")
}
fn live_var(t: usize) -> String {
    format!("live.{t}")
}
fn dem_var(t: usize) -> String {
    format!("dem.{t}")
}

/// Build the constraint model.
pub fn encode(problem: &Problem, target: &Ti60Target) -> ConstraintModel {
    let mut model = ConstraintModel::new();
    let engines = problem.bounds.max_engines.max(1) as usize;
    let ops = problem.ops.len();
    let horizon = problem.horizon();
    let max_dur = problem
        .ops
        .iter()
        .flat_map(|op| op.candidates.iter().map(|c| c.temporal_iterations))
        .max()
        .unwrap_or(1)
        .max(1);
    let max_lanes = problem
        .shapes
        .iter()
        .map(|s| s.lanes())
        .max()
        .unwrap_or(1)
        .max(1);
    let max_dsp = problem
        .shapes
        .iter()
        .map(|s| problem.engine_dsp(target, *s))
        .max()
        .unwrap_or(0)
        .max(1);

    // ---- variables ------------------------------------------------------
    for e in 0..engines {
        model.var(Var::boolean(used(e)));
    }
    for e in 0..engines {
        for s in 0..problem.shapes.len() {
            model.var(Var::boolean(shape_var(e, s)));
        }
        model.var(Var::integer(format!("englanes.{e}"), 0, max_lanes as i64));
        model.var(Var::boolean(format!("ded.{e}")));
        model.var(Var::boolean(format!("shared.{e}")));
    }
    for i in 0..ops {
        for e in 0..engines {
            model.var(Var::boolean(bind_var(i, e)));
        }
        for s in 0..problem.shapes.len() {
            model.var(Var::boolean(cand_var(i, s)));
        }
        model.var(Var::integer(start_var(i), 0, horizon));
        model.var(Var::integer(dur_var(i), 1, max_dur as i64));
    }
    model.var(Var::integer("makespan", 0, horizon));
    model.var(Var::integer("engines_used", 0, engines as i64));
    model.var(Var::integer(
        "total_lanes",
        0,
        (engines as u64 * max_lanes) as i64,
    ));
    model.var(Var::integer(
        "total_dsp",
        0,
        (engines as u64 * max_dsp) as i64,
    ));
    for k in 0..problem.edges.len() {
        model.var(Var::boolean(link_var(k)));
    }
    model.var(Var::integer("zero", 0, 0));
    model.var(Var::integer(
        "arena_capacity",
        problem.bounds.max_arena_words as i64,
        problem.bounds.max_arena_words as i64,
    ));
    for (t, tensor) in problem.tensors.iter().enumerate() {
        model.var(Var::boolean(arena_var(t)));
        model.var(Var::integer(live_var(t), 1, horizon));
        model.var(Var::integer(
            dem_var(t),
            tensor.words as i64,
            tensor.words as i64,
        ));
    }

    // ---- one candidate and one engine per operator -----------------------
    for i in 0..ops {
        let allowed = allowed_shapes(problem, i);
        model.constrain(Constraint::ExactlyOne {
            vars: allowed.iter().map(|s| cand_var(i, *s)).collect(),
        });
        // A shape the operator cannot legally use is never selected.
        for s in 0..problem.shapes.len() {
            if !allowed.contains(&s) {
                model.constrain(Constraint::Linear {
                    terms: vec![(cand_var(i, s), 1)],
                    rel: Rel::Eq,
                    bound: 0,
                });
            }
        }
        model.constrain(Constraint::ExactlyOne {
            vars: (0..engines).map(|e| bind_var(i, e)).collect(),
        });
    }

    // ---- engine shape, lane bound, usage ---------------------------------
    for e in 0..engines {
        // One shape per engine, and only when the engine is used.
        let mut terms: Vec<Term> = (0..problem.shapes.len())
            .map(|s| (shape_var(e, s), 1))
            .collect();
        terms.push((used(e), -1));
        model.constrain(Constraint::Linear {
            terms,
            rel: Rel::Eq,
            bound: 0,
        });
        // englanes = sum(shape * lanes)
        let mut terms: Vec<Term> = vec![(format!("englanes.{e}"), 1)];
        for s in 0..problem.shapes.len() {
            terms.push((shape_var(e, s), -(problem.shapes[s].lanes() as i64)));
        }
        model.constrain(Constraint::Linear {
            terms,
            rel: Rel::Eq,
            bound: 0,
        });
        // min_lanes*use <= englanes <= max_lanes*use
        model.constrain(Constraint::Linear {
            terms: vec![
                (format!("englanes.{e}"), 1),
                (used(e), -(problem.bounds.min_lanes_per_engine as i64)),
            ],
            rel: Rel::Ge,
            bound: 0,
        });
        model.constrain(Constraint::Linear {
            terms: vec![
                (format!("englanes.{e}"), 1),
                (used(e), -(problem.bounds.max_lanes_per_engine as i64)),
            ],
            rel: Rel::Le,
            bound: 0,
        });
        // ded + shared == use
        model.constrain(Constraint::Linear {
            terms: vec![
                (format!("ded.{e}"), 1),
                (format!("shared.{e}"), 1),
                (used(e), -1),
            ],
            rel: Rel::Eq,
            bound: 0,
        });
        // bind => use, and use => some binding
        for i in 0..ops {
            model.constrain(Constraint::Linear {
                terms: vec![(bind_var(i, e), 1), (used(e), -1)],
                rel: Rel::Le,
                bound: 0,
            });
        }
        let mut terms: Vec<Term> = (0..ops).map(|i| (bind_var(i, e), 1)).collect();
        terms.push((used(e), -1));
        model.constrain(Constraint::Linear {
            terms,
            rel: Rel::Ge,
            bound: 0,
        });
        // A dedicated engine serves exactly one operator; a shared engine at
        // least two.
        let binds: Vec<Term> = (0..ops).map(|i| (bind_var(i, e), 1)).collect();
        model.constrain(Constraint::ImplyLinear {
            when: format!("ded.{e}"),
            terms: binds.clone(),
            rel: Rel::Eq,
            bound: 1,
        });
        model.constrain(Constraint::ImplyLinear {
            when: format!("shared.{e}"),
            terms: binds,
            rel: Rel::Ge,
            bound: 2,
        });
    }

    // ---- shared shape agreement ------------------------------------------
    // If operator `i` is bound to engine `e` and uses shape `s`, engine `e`
    // instantiates shape `s`: `bind + cand - shape <= 1`.
    for i in 0..ops {
        for e in 0..engines {
            for s in 0..problem.shapes.len() {
                model.constrain(Constraint::Linear {
                    terms: vec![
                        (bind_var(i, e), 1),
                        (cand_var(i, s), 1),
                        (shape_var(e, s), -1),
                    ],
                    rel: Rel::Le,
                    bound: 1,
                });
            }
        }
    }

    // ---- engine count ----------------------------------------------------
    let use_terms: Vec<Term> = (0..engines).map(|e| (used(e), 1)).collect();
    model.constrain(Constraint::Linear {
        terms: use_terms.clone(),
        rel: Rel::Ge,
        bound: problem.bounds.min_engines as i64,
    });
    model.constrain(Constraint::Linear {
        terms: use_terms.clone(),
        rel: Rel::Le,
        bound: problem.bounds.max_engines as i64,
    });
    model.constrain(Constraint::Linear {
        terms: {
            let mut t: Vec<Term> = vec![("engines_used".into(), 1)];
            t.extend(use_terms.iter().map(|(n, c)| (n.clone(), -*c)));
            t
        },
        rel: Rel::Eq,
        bound: 0,
    });
    // Role counts.
    model.constrain(Constraint::Linear {
        terms: (0..engines).map(|e| (format!("ded.{e}"), 1)).collect(),
        rel: Rel::Ge,
        bound: problem.bounds.min_dedicated_engines as i64,
    });
    model.constrain(Constraint::Linear {
        terms: (0..engines).map(|e| (format!("shared.{e}"), 1)).collect(),
        rel: Rel::Ge,
        bound: problem.bounds.min_shared_engines as i64,
    });

    // ---- duration and makespan -------------------------------------------
    for i in 0..ops {
        let mut terms: Vec<Term> = vec![(dur_var(i), 1)];
        for (s, candidate) in problem.ops[i].candidates_shape_iter(problem) {
            terms.push((cand_var(i, s), -(candidate.temporal_iterations as i64)));
        }
        model.constrain(Constraint::Linear {
            terms,
            rel: Rel::Eq,
            bound: 0,
        });
        model.constrain(Constraint::Linear {
            terms: vec![(start_var(i), 1), (dur_var(i), 1), ("makespan".into(), -1)],
            rel: Rel::Le,
            bound: 0,
        });
    }

    // ---- dependency order, relaxed across a chosen link ------------------
    for (k, edge) in problem.edges.iter().enumerate() {
        let p = edge.producer;
        let c = edge.consumer;
        // Without a link the consumer waits for the producer to finish.
        model.constrain(Constraint::Linear {
            terms: vec![
                (start_var(c), 1),
                (start_var(p), -1),
                (dur_var(p), -1),
                (link_var(k), horizon),
            ],
            rel: Rel::Ge,
            bound: 0,
        });
        // With a link the consumer may start one cycle after the producer does.
        model.constrain(Constraint::ImplyLinear {
            when: link_var(k),
            terms: vec![(start_var(c), 1), (start_var(p), -1)],
            rel: Rel::Ge,
            bound: 1,
        });
        // A link needs two distinct engines.
        for e in 0..engines {
            model.constrain(Constraint::Linear {
                terms: vec![(link_var(k), 1), (bind_var(p, e), 1), (bind_var(c, e), 1)],
                rel: Rel::Le,
                bound: 2,
            });
        }
    }
    let links_terms: Vec<Term> = (0..problem.edges.len()).map(|k| (link_var(k), 1)).collect();
    if !links_terms.is_empty() {
        model.constrain(Constraint::Linear {
            terms: links_terms.clone(),
            rel: Rel::Ge,
            bound: problem.bounds.min_pipeline_links as i64,
        });
        model.constrain(Constraint::Linear {
            terms: links_terms,
            rel: Rel::Le,
            bound: problem.bounds.max_pipeline_links as i64,
        });
    } else if problem.bounds.min_pipeline_links > 0 {
        // A link is demanded but no crossing exists: the model is infeasible.
        model.constrain(Constraint::Linear {
            terms: vec![("zero".into(), 1)],
            rel: Rel::Ge,
            bound: 1,
        });
    }

    // ---- frame-pipeline segments and K-bank residence --------------------
    //
    // Each operator gets a segment id. Data edges only go forward in segment
    // id, operators that share an engine share a segment, and a segment's
    // operators form a contiguous block in topological order. A tensor that
    // crosses segments (or is the graph input) lives in a K-bank buffer;
    // tensors internal to a segment are local (1 bank).
    let max_seg = (ops.max(1) - 1) as i64;
    for i in 0..ops {
        model.var(Var::integer(seg_var(i), 0, max_seg));
    }
    for k in 0..problem.edges.len() {
        model.var(Var::boolean(cross_var(k)));
    }
    for t_i in 0..problem.tensors.len() {
        model.var(Var::boolean(kbank_var(t_i)));
    }
    // Edge: seg_p <= seg_c, and cross <=> seg_c > seg_p.
    for (k, edge) in problem.edges.iter().enumerate() {
        let p = edge.producer;
        let c = edge.consumer;
        model.constrain(Constraint::Linear {
            terms: vec![(seg_var(c), 1), (seg_var(p), -1)],
            rel: Rel::Ge,
            bound: 0,
        });
        model.constrain(Constraint::Linear {
            terms: vec![(seg_var(c), 1), (seg_var(p), -1), (cross_var(k), -1)],
            rel: Rel::Ge,
            bound: 0,
        });
        model.constrain(Constraint::Linear {
            terms: vec![
                (seg_var(c), 1),
                (seg_var(p), -1),
                (cross_var(k), -max_seg.max(1)),
            ],
            rel: Rel::Le,
            bound: 0,
        });
    }
    // Same engine => same segment.
    let m = max_seg.max(1);
    for e in 0..engines {
        for i in 0..ops {
            for j in (i + 1)..ops {
                model.constrain(Constraint::Linear {
                    terms: vec![
                        (seg_var(i), 1),
                        (seg_var(j), -1),
                        (bind_var(i, e), m),
                        (bind_var(j, e), m),
                    ],
                    rel: Rel::Le,
                    bound: 2 * m,
                });
                model.constrain(Constraint::Linear {
                    terms: vec![
                        (seg_var(j), 1),
                        (seg_var(i), -1),
                        (bind_var(i, e), m),
                        (bind_var(j, e), m),
                    ],
                    rel: Rel::Le,
                    bound: 2 * m,
                });
            }
        }
    }
    // Segment contiguity and |seg_i-seg_j| auxiliaries. gap must equal the
    // absolute difference (pos/neg split + AtMostOne), otherwise a loose lower
    // bound lets shared-segment ownership stay vacuous when segs are equal.
    for i in 0..ops {
        for j in (i + 1)..ops {
            let gap = format!("segabs.{i}.{j}");
            let gap_pos = format!("segpos.{i}.{j}");
            let gap_neg = format!("segneg.{i}.{j}");
            let pos_nz = format!("segposnz.{i}.{j}");
            let neg_nz = format!("segnenz.{i}.{j}");
            model.var(Var::integer(gap.clone(), 0, m));
            model.var(Var::integer(gap_pos.clone(), 0, m));
            model.var(Var::integer(gap_neg.clone(), 0, m));
            model.var(Var::boolean(pos_nz.clone()));
            model.var(Var::boolean(neg_nz.clone()));
            model.constrain(Constraint::AtMostOne {
                vars: vec![pos_nz.clone(), neg_nz.clone()],
            });
            // seg_i - seg_j = gap_pos - gap_neg
            model.constrain(Constraint::Linear {
                terms: vec![
                    (seg_var(i), 1),
                    (seg_var(j), -1),
                    (gap_pos.clone(), -1),
                    (gap_neg.clone(), 1),
                ],
                rel: Rel::Eq,
                bound: 0,
            });
            // gap = gap_pos + gap_neg
            model.constrain(Constraint::Linear {
                terms: vec![
                    (gap.clone(), 1),
                    (gap_pos.clone(), -1),
                    (gap_neg.clone(), -1),
                ],
                rel: Rel::Eq,
                bound: 0,
            });
            // gap_pos <= m*pos_nz, gap_neg <= m*neg_nz, and each is at least its flag
            model.constrain(Constraint::Linear {
                terms: vec![(gap_pos.clone(), 1), (pos_nz.clone(), -m)],
                rel: Rel::Le,
                bound: 0,
            });
            model.constrain(Constraint::Linear {
                terms: vec![(gap_neg.clone(), 1), (neg_nz.clone(), -m)],
                rel: Rel::Le,
                bound: 0,
            });
            model.constrain(Constraint::Linear {
                terms: vec![(gap_pos.clone(), 1), (pos_nz.clone(), -1)],
                rel: Rel::Ge,
                bound: 0,
            });
            model.constrain(Constraint::Linear {
                terms: vec![(gap_neg.clone(), 1), (neg_nz.clone(), -1)],
                rel: Rel::Ge,
                bound: 0,
            });
            for k in (i + 1)..j {
                model.constrain(Constraint::Linear {
                    terms: vec![(seg_var(i), 1), (seg_var(k), -1), (gap.clone(), -m)],
                    rel: Rel::Le,
                    bound: 0,
                });
                model.constrain(Constraint::Linear {
                    terms: vec![(seg_var(k), 1), (seg_var(i), -1), (gap.clone(), -m)],
                    rel: Rel::Le,
                    bound: 0,
                });
            }
        }
    }
    // Shared engine in a segment owns every operator of that segment.
    for e in 0..engines {
        for i in 0..ops {
            for j in 0..ops {
                if i == j {
                    continue;
                }
                let gap = if i < j {
                    format!("segabs.{i}.{j}")
                } else {
                    format!("segabs.{j}.{i}")
                };
                // bind_i + shared - gap <= 1 + bind_j
                // When gap=0 and both antecedents hold, bind_j is forced.
                model.constrain(Constraint::Linear {
                    terms: vec![
                        (bind_var(i, e), 1),
                        (format!("shared.{e}"), 1),
                        (gap, -1),
                        (bind_var(j, e), -1),
                    ],
                    rel: Rel::Le,
                    bound: 1,
                });
            }
        }
    }
    // K-bank: graph-input tensors always; produced tensors when any outgoing
    // edge crosses segments. Local otherwise. Align with the legacy arena xor
    // link: a cross is a pipeline cut for that edge.
    for (t_i, tensor) in problem.tensors.iter().enumerate() {
        match tensor.producer {
            None => {
                model.constrain(Constraint::Linear {
                    terms: vec![(kbank_var(t_i), 1)],
                    rel: Rel::Eq,
                    bound: 1,
                });
            }
            Some(_) => {
                let crosses: Vec<usize> = problem
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.tensor == tensor.id)
                    .map(|(k, _)| k)
                    .collect();
                if crosses.is_empty() {
                    // No consumer edge: treat as output-side K-bank.
                    model.constrain(Constraint::Linear {
                        terms: vec![(kbank_var(t_i), 1)],
                        rel: Rel::Eq,
                        bound: 1,
                    });
                } else {
                    // kbank >= each cross; kbank <= sum(cross)
                    for &k in &crosses {
                        model.constrain(Constraint::Linear {
                            terms: vec![(kbank_var(t_i), 1), (cross_var(k), -1)],
                            rel: Rel::Ge,
                            bound: 0,
                        });
                    }
                    let mut terms: Vec<Term> = vec![(kbank_var(t_i), 1)];
                    for &k in &crosses {
                        terms.push((cross_var(k), -1));
                    }
                    model.constrain(Constraint::Linear {
                        terms,
                        rel: Rel::Le,
                        bound: 0,
                    });
                }
            }
        }
        // Tie the legacy link bit to cross for the same edge, so arena / link
        // accounting stays consistent: link_k == cross_k.
        for (k, edge) in problem.edges.iter().enumerate() {
            if edge.tensor == tensor.id {
                model.constrain(Constraint::Linear {
                    terms: vec![(link_var(k), 1), (cross_var(k), -1)],
                    rel: Rel::Eq,
                    bound: 0,
                });
            }
        }
    }

    // ---- one operator at a time per engine -------------------------------
    for e in 0..engines {
        let intervals: Vec<Interval> = (0..ops)
            .map(|i| Interval {
                start: start_var(i),
                length: dur_var(i),
                present: bind_var(i, e),
            })
            .collect();
        model.constrain(Constraint::NoOverlap { intervals });
    }

    // ---- arena -----------------------------------------------------------
    for (t, tensor) in problem.tensors.iter().enumerate() {
        // Either the tensor lives in the arena, or exactly one link cuts it.
        let mut terms: Vec<Term> = vec![(arena_var(t), 1)];
        for (k, edge) in problem.edges.iter().enumerate() {
            if edge.tensor == tensor.id {
                terms.push((link_var(k), 1));
            }
        }
        model.constrain(Constraint::Linear {
            terms,
            rel: Rel::Eq,
            bound: 1,
        });
        let start_ref = match tensor.producer {
            Some(p) => start_var(p),
            None => "zero".to_string(),
        };
        // live = makespan - start_ref
        model.constrain(Constraint::Linear {
            terms: vec![
                (live_var(t), 1),
                ("makespan".into(), -1),
                (start_ref.clone(), 1),
            ],
            rel: Rel::Eq,
            bound: 0,
        });
        model.constrain(Constraint::Cumulative {
            capacity: "arena_capacity".into(),
            intervals: vec![DemandInterval {
                start: start_ref,
                length: live_var(t),
                present: arena_var(t),
                demand: dem_var(t),
            }],
        });
    }

    // ---- objective -------------------------------------------------------
    model.constrain(Constraint::Linear {
        terms: {
            let mut t: Vec<Term> = vec![("total_lanes".into(), 1)];
            for e in 0..engines {
                t.push((format!("englanes.{e}"), -1));
            }
            t
        },
        rel: Rel::Eq,
        bound: 0,
    });
    model.constrain(Constraint::Linear {
        terms: {
            let mut t: Vec<Term> = vec![("total_dsp".into(), 1)];
            for e in 0..engines {
                for s in 0..problem.shapes.len() {
                    let dsp = problem.engine_dsp(target, problem.shapes[s]) as i64;
                    t.push((shape_var(e, s), -dsp));
                }
            }
            t
        },
        rel: Rel::Eq,
        bound: 0,
    });

    model.objective("makespan", Sense::Min);
    model.objective("total_dsp", Sense::Min);
    model.objective("engines_used", Sense::Min);
    model.objective("total_lanes", Sense::Min);
    model
}

/// Indices into `Problem::shapes` that operator `i` may use.
fn allowed_shapes(problem: &Problem, i: usize) -> Vec<usize> {
    problem.ops[i]
        .candidates
        .iter()
        .filter_map(|candidate| problem.shapes.iter().position(|s| *s == candidate.shape))
        .collect()
}

impl OpSpec {
    /// The operator's candidates paired with their index in `Problem::shapes`.
    fn candidates_shape_iter<'a>(
        &'a self,
        problem: &'a Problem,
    ) -> impl Iterator<Item = (usize, &'a Candidate)> + 'a {
        self.candidates.iter().filter_map(move |candidate| {
            problem
                .shapes
                .iter()
                .position(|s| *s == candidate.shape)
                .map(|index| (index, candidate))
        })
    }
}

/// Decode a solver assignment into a [`Solution`].
pub fn decode(problem: &Problem, values: &BTreeMap<String, i64>) -> Result<Solution, SolverError> {
    let engines_n = problem.bounds.max_engines.max(1) as usize;
    let get = |name: &str| -> Result<i64, SolverError> {
        values
            .get(name)
            .copied()
            .ok_or_else(|| SolverError::Protocol {
                detail: format!("the solver assignment is missing {name:?}"),
            })
    };
    let truthy = |name: &str| -> Result<bool, SolverError> { Ok(get(name)? != 0) };

    let mut engines = Vec::with_capacity(engines_n);
    for e in 0..engines_n {
        let is_used = truthy(&used(e))?;
        let mut shape = SpatialShape {
            simd: 1,
            pe: 1,
            tap_parallel: 1,
        };
        for s in 0..problem.shapes.len() {
            if truthy(&shape_var(e, s))? {
                shape = problem.shapes[s];
            }
        }
        engines.push(SolvedEngine {
            used: is_used,
            shape,
            dedicated: truthy(&format!("ded.{e}"))?,
        });
    }

    let mut bindings = Vec::with_capacity(problem.ops.len());
    for i in 0..problem.ops.len() {
        let engine = (0..engines_n)
            .find(|e| truthy(&bind_var(i, *e)).unwrap_or(false))
            .ok_or_else(|| SolverError::Protocol {
                detail: format!("operator {} has no engine binding", problem.ops[i].name),
            })?;
        let shape_index = (0..problem.shapes.len())
            .find(|s| truthy(&cand_var(i, *s)).unwrap_or(false))
            .ok_or_else(|| SolverError::Protocol {
                detail: format!("operator {} has no fold candidate", problem.ops[i].name),
            })?;
        let candidate = problem.ops[i]
            .candidates
            .iter()
            .find(|c| c.shape == problem.shapes[shape_index])
            .cloned()
            .ok_or_else(|| SolverError::Protocol {
                detail: format!(
                    "operator {} was assigned a shape it does not support",
                    problem.ops[i].name
                ),
            })?;
        bindings.push(SolvedBinding {
            op: i,
            engine,
            candidate,
            start: get(&start_var(i))?,
        });
    }

    let mut links = Vec::new();
    for k in 0..problem.edges.len() {
        if truthy(&link_var(k))? {
            links.push(SolvedLink { edge: k });
        }
    }

    let mut residences = Vec::with_capacity(problem.tensors.len());
    for t in 0..problem.tensors.len() {
        let kbank = truthy(&kbank_var(t))?;
        residences.push(SolvedResidence {
            arena: truthy(&arena_var(t))?,
            banks: if kbank { 2 } else { 1 },
        });
    }

    let mut raw_seg = Vec::with_capacity(problem.ops.len());
    for i in 0..problem.ops.len() {
        raw_seg.push(get(&seg_var(i))? as u32);
    }
    // Densify segment ids and build SolvedSegment entries.
    let mut unique = raw_seg.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut segments = Vec::with_capacity(unique.len());
    for (dense, &raw) in unique.iter().enumerate() {
        let op_idxs: Vec<usize> = raw_seg
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == raw)
            .map(|(i, _)| i)
            .collect();
        let mut eng = Vec::new();
        let mut shared = false;
        let mut cycles_shared = 0u64;
        let mut cycles_ded = 0u64;
        for &i in &op_idxs {
            let b = bindings
                .iter()
                .find(|b| b.op == i)
                .expect("binding for segmented op");
            if !eng.contains(&b.engine) {
                eng.push(b.engine);
            }
            if !engines[b.engine].dedicated {
                shared = true;
            }
            let dur = b.candidate.temporal_iterations.max(1);
            cycles_shared = cycles_shared.saturating_add(dur);
            cycles_ded = cycles_ded.max(dur);
        }
        segments.push(SolvedSegment {
            id: dense as u32,
            ops: op_idxs,
            engines: eng,
            shared,
            cycles: if shared { cycles_shared } else { cycles_ded },
        });
    }

    Ok(Solution {
        engines,
        bindings,
        links,
        residences,
        segments,
        makespan: get("makespan")?,
    })
}
