# Tessera quality standard

This document defines engineering rules for Tessera. It is normative for new code and design changes. A feature is incomplete when it can generate plausible RTL but cannot explain, validate, and reproduce its dataflow and resource decisions.

## 1. Core quality bar

Every compiler feature MUST have:

1. a typed semantic model;
2. an explicit legality/invariant check;
3. a deterministic lowering or planning result;
4. a software reference or executable semantic check;
5. an RTL simulation path when RTL is affected;
6. a report that separates estimated values from measured values.

The default development loop is:

```text
model → typed graph → MLIR → hardware plan → RTL
          ↓             ↓          ↓
       semantic ref  plan check  RTL sim
                                  ↓
                              synthesis report
```

Generation and simulation are not optional post-processing. They are first-class compiler stages and must be available from the initial supported operator set.

### 1.1 Persistent stage artifacts

The compiler MUST treat intermediate artifacts as part of the product, not temporary debug output. Import, optimization, lowering, constraint solving, layout materialization, RTL emission, and verification MUST have inspectable boundaries.

Each persisted artifact MUST include:

- a format name and schema version;
- input artifact hashes;
- compiler/pass/target revisions;
- relevant options and search seed;
- deterministic content or a documented normalization rule;
- diagnostics and metrics produced at that stage.

Stage commands MUST validate artifact compatibility before consuming it. A user MUST be able to rerun layout search with a different hardware constraint or policy file without repeating graph import and MLIR optimization. RTL emission MUST consume a validated layout and MUST NOT silently change the selected layout.

The repository SHOULD provide inspection commands for graph metrics, MLIR/lowered IR, constraints, rejected candidates, layout resources, schedule steps, and verification results. A successful stage is a reusable result even when a later stage fails.

## 2. Rust modeling rules

### 2.1 Prefer types over conventions

Rust types MUST represent domain distinctions that affect correctness. Do not encode them only in field names, comments, or unvalidated integers.

Use newtypes and enums for:

- tensor identifiers;
- operator identifiers;
- layer names;
- channel counts and spatial dimensions;
- bit widths;
- cycle counts;
- byte counts;
- DSP/RAM resource counts;
- clock frequency;
- frame rate;
- buffer identifiers;
- engine identifiers;
- memory placement;
- execution granularity;
- rounding and saturation modes.

Example:

```rust
pub struct TensorId(u32);
pub struct DspBlocks(u32);
pub struct Ram10Blocks(u32);
pub struct FrameCycles(u64);

pub enum Placement {
    Register,
    Fifo(FifoSpec),
    LineBuffer(LineBufferSpec),
    Bram(BramSpec),
    Ddr(DdrRegion),
}

pub enum ExecutionMode {
    Dedicated,
    Streaming,
    SharedTile,
    Hybrid,
}
```

A raw `u32` MAY be used internally for an index only when no domain invariant is attached. Public compiler APIs SHOULD expose domain types.

### 2.2 Make invalid states difficult to construct

Constructors MUST validate invariants at the boundary:

- dimensions are non-zero;
- channel folding divides the channel count;
- tap parallelism is supported by the operator;
- bit widths are supported by the target;
- tile dimensions include valid halo limits;
- buffer sizes are consistent with element type and layout;
- resource budgets are non-negative;
- a plan references existing tensors, engines, and buffers.

Use `Result` for recoverable user/model/target errors. Use assertions only for internal invariants that represent a programmer error after validated construction.

### 2.3 Preserve units

Do not mix:

- bytes and bits;
- logical elements and physical RAM blocks;
- MAC operations and DSP blocks;
- cycles and nanoseconds;
- per-tile and per-frame quantities.

Conversions MUST be explicit and named. A target model MUST expose the conversion from logical storage to physical resource usage rather than hiding it in arithmetic scattered across passes.

### 2.4 Ownership and mutation

IR transformations SHOULD use explicit ownership boundaries:

- immutable graph views for analysis;
- owned transformation results for rewrites;
- explicit rewrite transactions or replacement maps;
- no global mutable compiler state;
- no hidden mutation through shared caches.

A pass MUST declare whether it preserves, rewrites, or invalidates analyses.

## 3. Dataflow representation

### 3.1 Separate semantics from realization

The graph MUST represent what the model computes. It MUST NOT prematurely encode whether execution is:

- pixel, row, tile, or frame based;
- spatially expanded or temporally reused;
- BRAM or DDR resident;
- dedicated or shared-engine based.

Those choices belong to Hardware IR and the Execution Plan.

```text
Graph IR:
  Conv → Requant → Add

Hardware IR:
  tile(80×60)
  shared_engine(conv0)
  bram(buffer_a)
  ddr(buffer_skip)
  schedule(read → dw → pw → add → write)
```

### 3.2 Tensors and edges are first-class

A dataflow edge MUST be represented as a tensor/value with:

- stable identity;
- producer;
- consumers and fan-out count;
- shape;
- element type;
- layout;
- quantization contract;
- padding/halo requirements;
- lifetime once a hardware plan exists.

Fan-out MUST be explicit. A fusion pass MUST NOT remove or overwrite a producer value when another consumer still requires it.

### 3.3 Quantization is semantic data

Quantization attributes MUST remain explicit through lowering:

- accumulator type;
- multiplier or `M0`;
- shift;
- rounding mode;
- saturation range;
- activation clamp;
- per-channel versus scalar parameters;
- padding mode.

The compiler MUST NOT infer these from arbitrary lowered arithmetic when they can be represented directly. Any fusion that changes the placement of requantization MUST prove equivalent semantics or reject the rewrite.

### 3.4 Hardware streams are distinct from graph edges

A graph edge means a value dependency. A hardware stream means a transport protocol and rate contract. A stream type MUST make explicit:

- element width;
- lane count;
- valid/ready protocol;
- packet or frame boundary behavior;
- producer and consumer rate;
- backpressure behavior;
- required FIFO or buffering.

The compiler MUST NOT assume that connecting two graph values automatically creates a legal stream.

## 4. Open/closed design principle

Tessera MUST be open for new targets, operators, execution modes, and RTL backends without requiring invasive changes to existing graph passes.

It SHOULD be closed for modification in stable core semantics: adding a Ti60 target MUST NOT alter the meaning of `Conv`, and adding a new backend MUST NOT change the manifest importer.

### 4.1 Extension points

Use explicit traits or registries for:

```rust
trait TargetModel {
    fn validate(&self, plan: &ExecutionPlan) -> Result<(), TargetError>;
    fn map_operator(&self, op: &MappedOp) -> ResourceUse;
    fn map_buffer(&self, buffer: &BufferSpec) -> ResourceUse;
    fn estimate(&self, schedule: &Schedule) -> Cost;
}

trait Backend {
    fn lower(&self, plan: &ExecutionPlan) -> Result<RtlArtifact, BackendError>;
}

trait Frontend {
    fn import(&self, input: &Input) -> Result<Graph, FrontendError>;
}
```

The exact traits may evolve, but responsibilities MUST remain separated:

- frontend: input format to Graph IR;
- optimization: legal semantic rewrites;
- planner: graph to Execution Plan;
- target: resource and cost rules;
- backend: Execution Plan to artifacts.

### 4.2 Avoid abstraction without a second implementation

Do not create generic interfaces merely to look extensible. An extension point is justified when at least one of these is true:

- two implementations already exist;
- the boundary is a planned compatibility contract;
- the boundary isolates target-specific correctness risk;
- the boundary is required for testing with a reference implementation.

## 5. MLIR rules

Melior/MLIR is the optimization and progressive-lowering substrate, not the source of truth for every hardware decision.

MLIR passes MAY handle:

- graph canonicalization;
- tensor and layout rewrites;
- producer/consumer fusion;
- bufferization;
- loop tiling, interchange, and fusion;
- affine access analysis;
- dead value and copy elimination.

Tessera-owned Hardware IR MUST retain information that generic MLIR does not guarantee to preserve:

- engine binding;
- DSP packing;
- RAM banking and ports;
- FIFO rate contracts;
- DMA and DDR placement;
- ping-pong ownership;
- static schedule steps;
- target resource reservations;
- RTL module selection.

Every MLIR transformation that changes numerical order, buffer lifetime, or data movement MUST have a focused equivalence check.

## 6. Planning and resource correctness

A plan MUST distinguish:

```text
estimated resources  ≠  synthesized resources
estimated cycles     ≠  measured cycles
peak capacity        ≠ port/bank feasibility
ideal DMA overlap    ≠ sustained DDR throughput
```

Hard constraints MUST be checked before RTL generation:

- DSP budget;
- RAM block budget;
- logic budget when modeled;
- clock/frame budget;
- DDR bandwidth;
- buffer lifetime and port conflicts;
- stream rate and FIFO depth;
- operator and layout legality.

A planner MUST fail explicitly when no candidate is feasible. It MUST NOT silently fall back to an unbounded or obviously over-budget configuration.

Plans MUST be deterministic for fixed:

- graph;
- target description;
- compiler version;
- planner options;
- random seed, if search is stochastic.

## 7. RTL generation quality

Generated RTL MUST be reproducible and inspectable:

- stable module and signal naming;
- stable instance ordering;
- emitted plan metadata;
- model and weight hashes;
- target and compiler version;
- no hidden dependence on current working directory;
- no generated source committed unless explicitly designated as a fixture.

Every RTL backend MUST expose:

```text
lower(plan) → RTL sources + memory images + metadata
```

The backend MUST reject plans it cannot faithfully implement. It MUST NOT silently reinterpret unsupported placement, rounding, lane count, or memory protocol.

Status at v0.1: the emitted backend supports a single streaming pointwise-dot step (signed int8
activations, ROM-resident weights, register/FIFO activation interfaces, no bias) and whole models
through a static-schedule FSM over a lifetime-reused activation arena with one shared MAC array.
One requantization contract serves both: a 17-bit multiplier with **half-away-from-zero** rounding
(`2^(shift-1)` offset for non-negative products, `2^(shift-1)-1` for negative ones). The exactness
baseline is the compiled fixed-point IR, not the training `f32` coefficients, which are approximated
with a per-step recorded bound. Every plan outside the envelope is rejected with the step name and
the unsupported construct ([ADR-0003](docs/adr/0003-stream-interfaces-and-the-v1-rtl-envelope.md)).

## 8. Verification is a built-in feature

The minimum supported operator must have all of the following before it is considered implemented:

1. Graph IR construction test.
2. Reference evaluator test.
3. MLIR lowering/verification test when the operator enters MLIR.
4. Plan legality and resource accounting test.
5. RTL generation test.
6. RTL simulation against the reference.
7. A failure diagnostic that identifies tensor, layer, coordinates, and expected/actual values.

Tests MUST target observable behavior and invariants. Avoid tests that merely assert source text, module count, non-empty output, or forwarding of configuration.

The standard differential chain is:

```text
manifest
  → Rust reference
  → MLIR lowered reference
  → execution-plan simulator
  → RTL simulator
```

The existing NNgen `ng.eval`, `stref`, and Verilator outputs are temporary external oracles during migration and SHOULD remain in regression fixtures until Rust owns the equivalent semantics.

## 9. Error handling and diagnostics

User-facing compiler errors MUST include:

- stage;
- model/layer/operator identity;
- relevant shape and type;
- violated constraint;
- selected or attempted configuration;
- actionable remediation where possible.

Example:

```text
planning layer enc.0.pw: no legal fold satisfies the Ti60 budget
  input channels: 32
  output channels: 64
  DSP budget: 96
  required frame cycles: 12,481,024
  frame budget: 10,000,000
  suggestions: enlarge tile, enable DW→PW fusion, or increase MAC DSP budget
```

Avoid `unwrap()` in compiler paths. Use typed errors for expected failures and preserve source context.

## 10. Documentation and compatibility

Every new operator, target, pass, or backend MUST document:

- semantic contract;
- supported layouts and types;
- resource model;
- unsupported combinations;
- verification coverage;
- generated artifacts.

Versioned serialized formats MUST include a schema version. Changes that alter plan interpretation require an ADR or migration note.

## 11. Review checklist

Before merging a compiler feature:

- [ ] Domain invariants are represented by types or validated constructors.
- [ ] Graph semantics are independent of hardware realization.
- [ ] Fan-out, lifetime, layout, and quantization are explicit.
- [ ] Target-specific logic is behind a target boundary.
- [ ] The planner rejects infeasible configurations.
- [ ] Estimates are labeled and separated from measurements.
- [ ] RTL generation is deterministic.
- [ ] Reference and RTL simulation cover the changed path.
- [ ] Generated artifacts include provenance metadata.
- [ ] Documentation and ADRs are updated for architectural decisions.

## 12. Definition of done

A Tessera feature is done only when a user can:

```text
provide graph
  → inspect optimized IR
  → inspect selected execution plan
  → inspect resource/cycle report
  → generate RTL
  → run simulation
  → compare output with reference
```

A compiler that only emits Verilog is a code generator. Tessera is complete only when it explains and verifies why that Verilog is legal for the selected target.
