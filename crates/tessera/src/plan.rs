//! Lowered primitive graph and planning policy.
//!
//! Re-exports the typed contract from [`crate::planner`] while that module is
//! being narrowed to the CP-SAT-only entry point. New code should import from
//! here.

pub use crate::planner::{
    POLICY_FORMAT, POLICY_SCHEMA_VERSION, Policy, PolicyConstraints, PolicyError, SearchPolicy,
    FusionPolicy, MemoryPolicy, OperatorOverride, Hints, Padding, ConvDetail, ElementwiseDetail,
    UpsampleDetail, OpDetail, TensorDesc, PrimitiveOp, PlanInput, PrimitiveGraph, PlannerError,
    ProvenanceInput,
};
