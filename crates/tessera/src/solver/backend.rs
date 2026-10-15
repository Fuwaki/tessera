//! The CP-SAT backend: the one adapter between the solver-agnostic constraint
//! vocabulary and Google OR-Tools.
//!
//! [`ConstraintModel`] describes variables, constraints and a lexicographic
//! objective list without naming a solver library. This module is the single
//! place that turns that description into a CP-SAT model, solves it *in
//! process* and hands the assignment back; nothing else in the compiler names
//! a solver crate, spawns an interpreter or carries a second solver.
//!
//! Two properties are contractual:
//!
//! * **Determinism** — the search is pinned to `random_seed = 1` and a single
//!   worker, so the same program always yields the same assignment.
//! * **Honest status** — only `INFEASIBLE` becomes [`Outcome::Infeasible`].
//!   Every objective stage must prove its optimum, so a stage that stops at
//!   `FEASIBLE` without proving it, and any `UNKNOWN` or `MODEL_INVALID`
//!   status, is a backend failure: freezing an unproven value, or reporting
//!   "no organization fits", would be a silently wrong plan.
//!
//! Objectives are optimized hierarchically: the first is optimized to
//! optimality, fixed at that optimum, then the next is optimized, and so on,
//! which realizes a lexicographic objective list on a single-objective solver.

use std::collections::BTreeMap;

use cpsat_rs::prelude::{
    BoolVar, BoundedLinearExpr, CpModel, CpSolver, CpSolverStatus, IntVar, IntervalVar, LinearExpr,
    SatParameters, SolveResponse,
};

use super::SolverError;
use super::model::{Constraint, ConstraintModel, Rel, Sense, Term, Var};

/// The solver's answer for one constraint program.
pub(crate) enum Outcome {
    /// A feasible assignment, one entry per declared variable.
    Solved(BTreeMap<String, i64>),
    /// No assignment satisfies the program.
    Infeasible,
}

/// Solve `program` with CP-SAT and return its assignment.
///
/// # Errors
///
/// [`SolverError::Backend`] when the program cannot be translated, when the
/// backend cannot run it, or when a stage stops without an answer the compiler
/// may act on: `UNKNOWN`, `MODEL_INVALID`, and an objective stage that stops at
/// `FEASIBLE` without proving its optimum.
pub(crate) fn solve(program: &ConstraintModel) -> Result<Outcome, SolverError> {
    let mut built = Built::new(program)?;
    built.constrain(program)?;

    // Pinned search: a fixed seed and one worker make the assignment a function
    // of the program alone.
    let parameters = SatParameters::default()
        .with_random_seed(1)
        .with_num_workers(1);

    let mut response: Option<SolveResponse> = None;
    for objective in &program.objectives {
        let var = built.int(&objective.var, "objective")?;
        match objective.sense {
            Sense::Min => built.model.minimize(var),
            Sense::Max => built.model.maximize(var),
        }
        let solved = run(&built.model, &parameters)?;
        // Only a *proven* optimum may be frozen: fixing a value the solver did
        // not prove would constrain every later objective to a wrong answer.
        // A pure feasibility program (below) has no such stage.
        if !solved.is_optimal() {
            return match solved.status() {
                CpSolverStatus::Infeasible => Ok(Outcome::Infeasible),
                other => Err(SolverError::Backend {
                    detail: format!(
                        "the CP-SAT backend did not prove an optimum for the objective {:?} \
                         (status {other:?}); freezing an unproven value would constrain the \
                         remaining objectives to a wrong answer",
                        objective.var
                    ),
                }),
            };
        }
        // Fix this objective at its optimum before the next one is optimized.
        built.model.add(LinearExpr::from(var).eq(solved.value(var)));
        response = Some(solved);
    }
    if response.is_none() {
        // A pure feasibility program: there is no objective list to walk.
        let solved = run(&built.model, &parameters)?;
        if !solved.is_feasible() {
            return status(solved.status());
        }
        response = Some(solved);
    }
    let response = response.expect("a feasible response was stored above");

    let values = built
        .variables
        .iter()
        .map(|(name, declaration)| (name.clone(), response.value(declaration.handle.int())))
        .collect();
    Ok(Outcome::Solved(values))
}

/// Run the solver, mapping an FFI or decode failure to a backend error.
fn run(model: &CpModel, parameters: &SatParameters) -> Result<SolveResponse, SolverError> {
    CpSolver::solve_with_params(model, parameters).map_err(|error| SolverError::Backend {
        detail: format!("the CP-SAT backend could not run the constraint program: {error}"),
    })
}

/// Turn a status that carries no feasible solution into an outcome.
///
/// `INFEASIBLE` is a result. `UNKNOWN` and `MODEL_INVALID` mean the backend did
/// not decide the program, and answering "infeasible" there would report a
/// bound set as unsatisfiable without ever showing the solver a reason.
fn status(status: CpSolverStatus) -> Result<Outcome, SolverError> {
    match status {
        CpSolverStatus::Infeasible => Ok(Outcome::Infeasible),
        other => Err(SolverError::Backend {
            detail: format!(
                "the CP-SAT backend stopped with the status {other:?} without deciding the \
                 constraint program"
            ),
        }),
    }
}

/// A malformed program the backend refuses to run.
fn malformed(detail: impl Into<String>) -> SolverError {
    SolverError::Backend {
        detail: detail.into(),
    }
}

/// A declared variable, remembered as the handle its domain requires.
#[derive(Debug, Clone, Copy)]
enum Handle {
    /// Boolean variable.
    Boolean(BoolVar),
    /// Bounded integer variable.
    Integer(IntVar),
}

impl Handle {
    /// The variable, as arithmetic.
    fn int(self) -> IntVar {
        match self {
            Handle::Boolean(var) => var.as_int_var(),
            Handle::Integer(var) => var,
        }
    }
}

/// A declared variable and the domain it was declared with.
#[derive(Debug, Clone, Copy)]
struct Declaration {
    /// Solver handle.
    handle: Handle,
    /// Declared inclusive lower bound.
    lb: i64,
    /// Declared inclusive upper bound.
    ub: i64,
}

/// A program being translated, with its variables already declared.
struct Built {
    /// The CP-SAT model under construction.
    model: CpModel,
    /// Declared variables by name.
    variables: BTreeMap<String, Declaration>,
}

impl Built {
    /// Declare every variable of `program`.
    fn new(program: &ConstraintModel) -> Result<Self, SolverError> {
        let mut built = Self {
            model: CpModel::new(),
            variables: BTreeMap::new(),
        };
        for var in &program.variables {
            let name = var.name();
            if built.variables.contains_key(name) {
                return Err(malformed(format!("duplicate variable name {name:?}")));
            }
            let declaration = match var {
                Var::Bool { .. } => Declaration {
                    handle: Handle::Boolean(built.model.new_bool_var(name)),
                    lb: 0,
                    ub: 1,
                },
                Var::Int { lb, ub, .. } => {
                    if lb > ub {
                        return Err(malformed(format!(
                            "variable {name:?} has the empty integer domain {lb}..={ub}"
                        )));
                    }
                    Declaration {
                        handle: Handle::Integer(built.model.new_int_var(*lb..=*ub, name)),
                        lb: *lb,
                        ub: *ub,
                    }
                }
            };
            built.variables.insert(name.to_string(), declaration);
        }
        Ok(built)
    }

    /// Translate every constraint of `program`.
    fn constrain(&mut self, program: &ConstraintModel) -> Result<(), SolverError> {
        for constraint in &program.constraints {
            match constraint {
                Constraint::Linear { terms, rel, bound } => {
                    let expression = self.expression(terms, "linear")?;
                    self.model.add(bounded(expression, *rel, *bound));
                }
                Constraint::ExactlyOne { vars } => {
                    let literals = self.literals(vars, "exactly_one")?;
                    if literals.is_empty() {
                        return Err(malformed("exactly_one needs at least one literal"));
                    }
                    self.model.add_exactly_one(&literals);
                }
                Constraint::AtMostOne { vars } => {
                    let literals = self.literals(vars, "at_most_one")?;
                    self.model.add_at_most_one(&literals);
                }
                Constraint::ImplyLinear {
                    when,
                    terms,
                    rel,
                    bound,
                } => {
                    let literal = self.literal(when, "imply_linear")?;
                    let expression = self.expression(terms, "imply_linear")?;
                    self.model.add(bounded(expression, *rel, *bound));
                    // OR-Tools' native half-reification: the constraint just
                    // appended holds exactly when `when` is true. The builder
                    // has no enforced-add, so reach through the proto hatch
                    // instead of approximating the implication with a big-M.
                    let constraints = &mut self.model.raw_proto_mut().constraints;
                    let appended = constraints
                        .last_mut()
                        .expect("the constraint was appended above");
                    appended.enforcement_literal.push(literal.index());
                }
                Constraint::NoOverlap { intervals } => {
                    let mut built = Vec::with_capacity(intervals.len());
                    for (index, interval) in intervals.iter().enumerate() {
                        built.push(self.optional_interval(
                            &interval.start,
                            &interval.length,
                            &interval.present,
                            index,
                            "interval",
                        )?);
                    }
                    self.model.add_no_overlap(&built);
                }
                Constraint::Cumulative {
                    capacity,
                    intervals,
                } => {
                    let capacity = self.int(capacity, "cumulative")?;
                    let mut built = Vec::with_capacity(intervals.len());
                    let mut demands = Vec::with_capacity(intervals.len());
                    for (index, interval) in intervals.iter().enumerate() {
                        built.push(self.optional_interval(
                            &interval.start,
                            &interval.length,
                            &interval.present,
                            index,
                            "cumulative",
                        )?);
                        demands.push(LinearExpr::from(self.int(&interval.demand, "cumulative")?));
                    }
                    self.model.add_cumulative(&built, &demands, capacity);
                }
            }
        }
        Ok(())
    }

    /// Build an optional interval named `{prefix}_{index}`.
    ///
    /// An interval needs an end, so the end is materialized as its own bounded
    /// variable with the bounds implied by its two operands — never a guessed
    /// horizon — and linked to them. `present` is the interval's presence
    /// literal: an absent interval is ignored by `no_overlap` and `cumulative`.
    fn optional_interval(
        &mut self,
        start_name: &str,
        length_name: &str,
        present_name: &str,
        index: usize,
        prefix: &str,
    ) -> Result<IntervalVar, SolverError> {
        let start = self.int(start_name, prefix)?;
        let length = self.int(length_name, prefix)?;
        let present = self.literal(present_name, prefix)?;
        let (start_lo, start_hi) = self.domain(start_name, prefix)?;
        let (length_lo, length_hi) = self.domain(length_name, prefix)?;
        let end_lo = start_lo.checked_add(length_lo);
        let end_hi = start_hi.checked_add(length_hi);
        let (Some(end_lo), Some(end_hi)) = (end_lo, end_hi) else {
            return Err(malformed(format!(
                "{prefix} interval {index} spans {start_name:?} + {length_name:?}, whose declared \
                 bounds overflow"
            )));
        };
        let end = self
            .model
            .new_int_var(end_lo..=end_hi, format!("end_{prefix}_{index}"));
        self.model
            .add((LinearExpr::from(start) + length - end).eq(0));
        Ok(self.model.new_optional_interval_var(
            start,
            length,
            end,
            present,
            format!("{prefix}_{index}"),
        ))
    }

    /// A linear expression from `terms`.
    fn expression(&self, terms: &[Term], what: &str) -> Result<LinearExpr, SolverError> {
        let mut expression = LinearExpr::constant(0);
        for (name, coefficient) in terms {
            expression.add_term(self.int(name, what)?, *coefficient);
        }
        Ok(expression)
    }

    /// The literals named by `names`.
    fn literals(&self, names: &[String], what: &str) -> Result<Vec<BoolVar>, SolverError> {
        names.iter().map(|name| self.literal(name, what)).collect()
    }

    /// A declared variable, as arithmetic.
    fn int(&self, name: &str, what: &str) -> Result<IntVar, SolverError> {
        Ok(self.declaration(name, what)?.handle.int())
    }

    /// A declared variable, as a literal.
    fn literal(&self, name: &str, what: &str) -> Result<BoolVar, SolverError> {
        match self.declaration(name, what)?.handle {
            Handle::Boolean(var) => Ok(var),
            Handle::Integer(_) => Err(malformed(format!(
                "{what} needs the literal {name:?}, which is not a Boolean variable"
            ))),
        }
    }

    /// The declared domain of a variable.
    fn domain(&self, name: &str, what: &str) -> Result<(i64, i64), SolverError> {
        let declaration = self.declaration(name, what)?;
        Ok((declaration.lb, declaration.ub))
    }

    /// A declaration, or a named error for a reference to nothing.
    fn declaration(&self, name: &str, what: &str) -> Result<Declaration, SolverError> {
        self.variables
            .get(name)
            .copied()
            .ok_or_else(|| malformed(format!("{what} references the unknown variable {name:?}")))
    }
}

/// A bounded expression for `expression rel bound`.
fn bounded(expression: LinearExpr, rel: Rel, bound: i64) -> BoundedLinearExpr {
    match rel {
        Rel::Le => expression.le(bound),
        Rel::Eq => expression.eq(bound),
        Rel::Ge => expression.ge(bound),
    }
}
