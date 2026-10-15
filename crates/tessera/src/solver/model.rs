//! Solver-agnostic constraint vocabulary.
//!
//! This module is the *only* place the compiler describes a constraint
//! problem. It is deliberately solver-independent: a [`ConstraintModel`] is a
//! plain data structure that serializes to the JSON document below, and no
//! type here names a solver library, a model object or a wire schema. That
//! keeps the constraint language retargetable without touching the encoding
//! that produces it: `src/solver/backend.rs` is the one adapter that turns a
//! model into a CP-SAT program.
//!
//! The document is:
//!
//! ```json
//! {
//!   "variables": [{"name":"a","domain":"bool"},
//!                 {"name":"t","domain":"int","lb":0,"ub":10}],
//!   "constraints": [
//!     {"op":"linear","terms":[["a",1]],"rel":"<=","bound":1},
//!     {"op":"exactly_one","vars":["a"]},
//!     {"op":"at_most_one","vars":["a"]},
//!     {"op":"imply_linear","when":"a","terms":[["t",1]],"rel":"==","bound":0},
//!     {"op":"no_overlap","intervals":[{"start":"s","length":"L","present":"p"}]},
//!     {"op":"cumulative","capacity":"cap",
//!      "intervals":[{"start":"s","length":"L","present":"p","demand":"d"}]}
//!   ],
//!   "objectives": [{"var":"makespan","sense":"min"}]
//! }
//! ```
//!
//! Objectives are optimized hierarchically: the backend optimizes the first to
//! optimality, fixes it, then optimizes the next, and so on. That is how a
//! lexicographic objective is expressed on top of a single-objective solver.

use serde::{Deserialize, Serialize};

/// A named decision variable.
///
/// Serializes as `{"name": ..., "domain": "bool"}` or
/// `{"name": ..., "domain": "int", "lb": ..., "ub": ...}`. Modelled as an enum
/// rather than a struct with a flattened tag, because serde cannot deserialize
/// a flattened internally-tagged enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "domain", rename_all = "snake_case")]
pub enum Var {
    /// Boolean variable.
    Bool {
        /// Unique name.
        name: String,
    },
    /// Bounded integer variable.
    Int {
        /// Unique name.
        name: String,
        /// Inclusive lower bound.
        lb: i64,
        /// Inclusive upper bound.
        ub: i64,
    },
}

impl Var {
    /// Boolean variable.
    pub fn boolean(name: impl Into<String>) -> Self {
        Var::Bool { name: name.into() }
    }

    /// Bounded integer variable.
    pub fn integer(name: impl Into<String>, lb: i64, ub: i64) -> Self {
        Var::Int {
            name: name.into(),
            lb,
            ub,
        }
    }

    /// Variable name.
    pub fn name(&self) -> &str {
        match self {
            Var::Bool { name } | Var::Int { name, .. } => name,
        }
    }
}

/// Comparison relation in a linear constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rel {
    /// `sum <= bound`
    #[serde(rename = "<=")]
    Le,
    /// `sum == bound`
    #[serde(rename = "==")]
    Eq,
    /// `sum >= bound`
    #[serde(rename = ">=")]
    Ge,
}

/// One `(variable, coefficient)` term of a linear expression.
pub type Term = (String, i64);

/// An optional interval, referenced by variable name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    /// Start-time variable.
    pub start: String,
    /// Length variable.
    pub length: String,
    /// Presence literal (a Boolean variable).
    pub present: String,
}

/// An optional interval with a demand, for a cumulative constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemandInterval {
    /// Start-time variable.
    pub start: String,
    /// Length variable.
    pub length: String,
    /// Presence literal (a Boolean variable).
    pub present: String,
    /// Demand variable (a resource amount).
    pub demand: String,
}

/// A single constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Constraint {
    /// `sum(terms) rel bound`.
    Linear {
        /// Terms.
        terms: Vec<Term>,
        /// Relation.
        rel: Rel,
        /// Right-hand side.
        bound: i64,
    },
    /// Exactly one of `vars` is true.
    ExactlyOne {
        /// Literals.
        vars: Vec<String>,
    },
    /// At most one of `vars` is true.
    AtMostOne {
        /// Literals.
        vars: Vec<String>,
    },
    /// `when` implies `sum(terms) rel bound`.
    ImplyLinear {
        /// Antecedent literal.
        when: String,
        /// Terms.
        terms: Vec<Term>,
        /// Relation.
        rel: Rel,
        /// Right-hand side.
        bound: i64,
    },
    /// No two present intervals overlap.
    NoOverlap {
        /// Intervals.
        intervals: Vec<Interval>,
    },
    /// Present intervals never exceed `capacity` at any instant.
    Cumulative {
        /// Capacity variable.
        capacity: String,
        /// Intervals with demands.
        intervals: Vec<DemandInterval>,
    },
}

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sense {
    /// Minimize.
    Min,
    /// Maximize.
    Max,
}

/// One objective in the lexicographic hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    /// Variable optimized.
    pub var: String,
    /// Direction.
    pub sense: Sense,
}

/// A complete constraint program.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConstraintModel {
    /// Variables, in declaration order.
    pub variables: Vec<Var>,
    /// Constraints.
    pub constraints: Vec<Constraint>,
    /// Lexicographic objectives.
    pub objectives: Vec<Objective>,
}

impl ConstraintModel {
    /// Empty program.
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            constraints: Vec::new(),
            objectives: Vec::new(),
        }
    }

    /// Declare a variable.
    pub fn var(&mut self, var: Var) {
        self.variables.push(var);
    }

    /// Add a constraint.
    pub fn constrain(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add an objective, in priority order.
    pub fn objective(&mut self, var: impl Into<String>, sense: Sense) {
        self.objectives.push(Objective {
            var: var.into(),
            sense,
        });
    }

    /// Serialize to the inspection JSON document.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

impl Default for ConstraintModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: a linear constraint from owned parts.
pub fn linear(terms: Vec<Term>, rel: Rel, bound: i64) -> Constraint {
    Constraint::Linear { terms, rel, bound }
}
