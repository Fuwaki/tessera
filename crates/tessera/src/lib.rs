//! Tessera: a hardware-aware compiler for quantized neural-network inference.
//!
//! The compiler is a sequence of inspectable, versioned stages. This crate exposes each
//! stage as a module so that later stages consume an artifact rather than an opaque
//! in-memory state:
//!
//! ```text
//! manifest.json  --frontend-->  Graph IR (graph.tgraph)
//!                               --mlir--> optimized graph (mlir.json / optimized.json)
//!                               --target/plan--> layout (layout.tlayout, plan.tplan)
//!                               --rtl--> SystemVerilog (rtl/)
//! ```
//!
//! Stage modules are added behind this file; adding a stage must not change the meaning
//! of the graph IR in [`graph`].
//!
//! * [`graph`] — typed, target-independent graph IR, validation, metrics and the
//!   persisted [`graph::GraphArtifact`].
//! * [`frontend`] — `fpga_st_train.int8.v1` manifest import, packing checks, provenance
//!   hashing and artifact save/load.
//! * [`mlir`] — deterministic Graph IR to MLIR emission, the pinned Melior pass
//!   pipeline, the structured analysis of an optimized module, and the persisted
//!   [`mlir::MlirArtifact`].
//! * [`lower`] — Graph IR *and* optimized MLIR artifacts to the planner's typed
//!   primitive graph, including requantization serialization, per-channel
//!   multiplier preservation and the exact code interval of every activation.
//! * [`primitive`] — the persisted primitive graph (`tessera.primitive`): the
//!   planner input, the optimizer's fusion groups and buffers, and provenance.
//! * [`target`] — declarative Ti60 target descriptions.
//! * [`planner`] — layout search over execution mode, folding, fusion, placement and
//!   granularity, producing a [`layout::LayoutArtifact`].
//! * [`layout`] — the persisted layout, its validation, and the backend-ready
//!   [`layout::ExecutionPlan`] projection.
//! * [`weights`] — weight-byte resolution for the RTL backend, including the concrete
//!   resolver over a persisted [`graph::GraphArtifact`].
//! * [`rtl`] — SystemVerilog emission for the supported pointwise-dot envelope, plus
//!   the Verilator simulation scaffold.
//! * [`stages`] — the persistent stage directory, artifact detection and the
//!   cross-stage provenance checks used by `tessera verify`.

pub mod frontend;
pub mod graph;
pub mod layout;
pub mod lower;
pub mod mlir;
pub mod planner;
pub mod primitive;
pub mod rtl;
pub mod solver;
pub mod stages;
pub mod target;
pub mod weights;
