# Tessera concept book

## 1. Purpose

Tessera is a Rust-first compiler for quantized neural-network inference on resource-constrained FPGAs. It accepts a model graph, preserves its numerical contract, searches hardware realizations under explicit target constraints, and emits inspectable RTL artifacts.

The project exists to bridge a gap between two useful but structurally different approaches:

- dedicated graph-to-RTL generation, where each operator receives a specialized implementation;
- streaming/dataflow generation, where layers are connected with ready/valid channels and folding parameters.

Tessera treats both as hardware backends of one compiler, not as one code generator with hidden special cases.

## 2. Goals

### Required

- Rust as the primary implementation language.
- Melior/MLIR for graph optimization and progressive lowering.
- Explicit preservation of quantization, padding, layout, and rounding semantics.
- A target-independent graph representation.
- A target model that describes DSP, RAM, logic, clock, and DDR constraints.
- A hardware-aware planner that can reject infeasible designs before RTL emission.
- Multiple execution models: dedicated, streaming, shared-tile, and hybrid.
- A stable execution-plan format between planning and RTL generation.
- Deterministic RTL generation, verified bit-exactly against the compiled
  fixed-point contract (the training `f32` requantization is approximated, with
  the per-step error bound recorded).
- Synthesis and board measurements usable to calibrate cost models.

### Recommended

- Keep ONNX and manifest import separate from target-specific code.
- Keep the existing `nngen` implementation as a correctness and regression oracle during migration.
- Generate plans before generating RTL.
- Make resource models and RTL templates replaceable.
- Report estimates separately from measured synthesis results.

### Explicitly not a first goal

- A universal ONNX implementation.
- A generic FPGA vendor abstraction with no tested target.
- Replacing every existing NNgen feature.
- Inferring exact post-place-and-route timing from static IR alone.
- CPU-driven dispatch for every layer.

## 3. Compiler as a persistent pipeline

Tessera is not a single command that transforms a model directly into RTL. It is a sequence of inspectable, resumable stages. Every stage consumes a versioned artifact and produces a versioned artifact that can be viewed, validated, compared, and reused by later stages.

```text
model description
    ↓ import
graph artifact
    ↓ MLIR optimize / progressively lower
optimized graph artifact
    ↓ target constraints + layout search
layout artifact
    ↓ plan validation / schedule materialization
execution-plan artifact
    ↓ RTL lowering
RTL artifact bundle
    ↓ simulation / synthesis / board validation
measurement artifacts
```

The compiler MUST allow a user to stop after any successful stage. A later command MUST be able to consume a saved artifact without rerunning earlier stages when the artifact's provenance and schema are compatible.

The stages are:

1. **Import** — parse ONNX or manifest into target-independent Graph IR.
2. **Graph optimization** — canonicalize, fuse, eliminate copies, and preserve numerical semantics through MLIR.
3. **Progressive lowering** — expose tensors, buffers, loops, and primitive operations while retaining Tessera hardware-relevant attributes.
4. **Constraint loading** — parse a user-editable target and policy description.
5. **Layout search** — explore execution mode, folding, fusion, tile/row granularity, memory placement, and engine assignment.
6. **Layout materialization** — save the selected layout and resource/cycle/bandwidth report.
7. **RTL lowering** — convert only a validated layout into RTL, memories, schedule tables, and driver metadata.
8. **Verification and measurement** — run semantic simulation, RTL simulation, synthesis, and optional board validation.

The term **layout** means the hardware organization and placement of computation/data, not FPGA physical placement and routing. Physical placement remains a downstream Efinity/toolchain result.

### 3.1 Stage commands

The CLI exposes explicit stage commands rather than hiding the pipeline behind one opaque invocation ([`cli.md`](cli.md) is the command reference; `--help` is authoritative):

```text
tessera import    manifest.json|dir --input-shape N,H,W,C --out build/
tessera emit-mlir build/graph.tgraph --out build/
tessera optimize  build/mlir.json --out build/          # requires the `melior` feature
tessera solve     --graph build/graph.tgraph --target ti60.toml --policy p.toml --out build/
tessera solve     --primitive primitive.json --target ti60.toml --policy p.toml --out build/
tessera inspect   build/graph.tgraph | build/mlir.json | build/layout.tlayout | build/rtl/rtl_bundle.json
tessera emit      build/layout.tlayout --graph build/graph.tgraph --out build/
tessera simulate  build/layout.tlayout --weights DIR --out build/
tessera verify    build/
```

Each command validates its input artifact before work begins and writes provenance into its output. A stage directory keeps every artifact, plus copies of the target and policy a layout was solved against, so `verify` can re-check the whole chain later without the original paths. A failed later stage does not invalidate earlier artifacts.

### 3.2 Metrics are first-class artifacts

Graph and layout inspection MUST be available without generating RTL. Metrics MAY include:

- operator and tensor counts;
- shape and dtype summaries;
- fan-out and critical-path information;
- estimated MACs and activation bytes;
- fusion candidates and rejected legality reasons;
- buffer lifetimes and peak live bytes;
- candidate DSP/RAM/logic use;
- estimated cycles and DDR traffic;
- constraint slack and bottlenecks.

Reports MUST identify whether a value is static analysis, target-model estimate, synthesis measurement, or board measurement.

### 3.3 Iterative compilation

The user MUST be able to modify only the target constraints or planning policy and rerun layout search against the same optimized graph. Conversely, graph optimization MUST be reusable across multiple target descriptions. Layout search MUST be reusable across multiple RTL emission attempts when the layout schema and backend contract remain compatible.

The cache key for a stage is the content hash of:

```text
input artifact + relevant options + target/policy + compiler/pass versions
```

No stage may rely on an implicit current working directory or mutable global state.

## 4. Compiler contracts

The boundaries are deliberate:

- **Graph artifact** owns model semantics and is target-independent.
- **Optimized graph artifact** records MLIR transformations and their legality/provenance.
- **Lowered IR artifact** exposes buffers, loops, and primitive operations without selecting a complete FPGA layout.
- **Constraint description** owns user-visible hardware budgets and search policy.
- **Layout artifact** owns the selected hardware organization, placement, schedule, and estimates.
- **Execution Plan** is the backend-ready portion of a layout and is reproducible before code generation.
- **RTL artifact bundle** owns module binding, cycle-level implementation, memory images, and interfaces.
- **Measurement artifacts** own simulator, synthesis, and board observations; they never silently overwrite estimates.

```text
Graph IR
  → Optimized Graph IR
  → Lowered Primitive IR
  → Layout IR
  → Execution Plan
  → RTL IR / SystemVerilog
```

The persistent file format for each boundary is part of the public compiler contract. It MUST be versioned independently from in-memory Rust types.


## 5. IR layers

### 5.1 Graph IR

The graph IR represents operators and tensors without committing to an execution architecture:

- tensor shape and element type;
- layout and alignment;
- graph dependencies and fan-out;
- convolution groups, kernel, stride, and padding;
- quantization scale, multiplier, shift, rounding, and saturation;
- weights by stable symbol and external storage reference.

Initial operators:

- `conv2d`;
- `depthwise_conv2d`;
- `pointwise_conv2d`;
- `requant` and activation;
- elementwise `add`;
- nearest-neighbor upsample.

### 5.2 MLIR layer

Tessera will lower Graph IR to a domain-aware MLIR representation. MLIR passes may progressively expose:

```text
operator graph → tensor operations → buffers → loops → target lowering
```

Early optimization candidates:

- producer/consumer fusion;
- dead tensor and redundant copy elimination;
- tile and halo analysis;
- buffer lifetime analysis;
- loop interchange, tiling, and fusion;
- preserving requantization at the correct computation boundary;
- legality checks for fan-out and residual paths.

Generic `linalg`, `tensor`, `memref`, `affine`, and `scf` are reusable implementation layers. Hardware-specific semantics must remain explicit in a Tessera dialect or hardware IR rather than being reconstructed from arbitrary low-level loops.

### 5.3 Hardware IR

Hardware IR describes a realization:

- engines and operator bindings;
- spatial and temporal folding;
- row, pixel, tile, or frame execution;
- line buffers, FIFOs, scratchpads, BRAM, and DDR buffers;
- ping-pong ownership and DMA operations;
- fusion boundaries;
- schedule steps and dependencies;
- resource reservations.

### 5.4 Execution Plan

The execution plan is a versioned, serializable artifact. It must include enough information to reproduce RTL and reports:

- target identity and resource budget;
- selected execution mode;
- tile dimensions and halo policy;
- engine definitions;
- SIMD/PE/tap configuration;
- buffer placement and lifetime;
- DMA and schedule steps;
- estimated cycles, DSP, RAM, logic, and DDR traffic;
- compiler and model hashes.

## 6. Target model

A target model is a plugin, not a global constant table. It exposes:

```text
resource_budget()
map_mac(widths, packing)
map_ram(width, depth, ports, banking)
estimate_operator(op, configuration)
estimate_schedule(schedule)
validate(plan)
```

The Ti60 model must represent at least:

- available and reserved DSP blocks;
- EFX DSP packing for supported operand widths;
- RAM10/XLR capacity and reservation;
- physical RAM width/depth rounding;
- port and bank constraints;
- clock and frame-cycle budget;
- DDR peak/effective bandwidth;
- non-ML accelerator resource consumers.

Static estimates are not treated as proof. Synthesis and place-and-route reports are measurements used to calibrate the target model.

## 7. Planner

The planner solves a constrained design-space problem. A candidate may vary:

- execution mode;
- tile or row shape;
- fusion choice;
- engine assignment;
- SIMD, PE, tap, and spatial parallelism;
- buffer placement;
- weight residency;
- ping-pong depth;
- overlap of compute and DMA.

Hard constraints include:

```text
DSP_used ≤ DSP_budget
RAM_used ≤ RAM_budget
logic_used ≤ logic_budget
DDR_traffic ≤ bandwidth_budget
cycles ≤ frame_budget
all graph dependencies are satisfied
all selected divisors and layouts are legal
```

Objectives are configurable. Examples:

- minimize frame cycles;
- minimize DSP while meeting a frame budget;
- minimize DDR traffic;
- minimize BRAM;
- produce a Pareto frontier.

The planner's own search is candidate enumeration, pruning, and branch-and-bound.
The joint mapping solve (`src/solver/`) is a real constraint program: it encodes
engines, bindings, folds, schedule, pipeline cuts and arena placement into a
solver-agnostic model and hands it to CP-SAT in process, which is the "after the
cost model and plan schema stabilize" step described here.

## 8. RTL backends

Backends consume the same Execution Plan but may emit different structures:

### Dedicated backend

Generates operator-specialized FSM/stream hardware. This preserves the useful NNgen-style baseline.

### Streaming backend

Generates ready/valid layers from the existing `st_stream` primitive library: windows, depthwise, MVAU, requant, FIFO, and ROM.

### Shared-tile backend

Generates a configurable MAC engine, tile buffers, line buffers, DMA, ping-pong control, and a static schedule engine. This is the primary new backend.

### Implemented in v1

The emitted backend (`tessera.pw_dot.i8i8.v1`) is a narrow, streaming pointwise-dot kernel with a ready/valid interface, a folded weight ROM and a per-channel requantization ROM. It emits **exactly one kernel**: a single pointwise step with signed int8 activations, no bias, and streamed activation buffers. Everything outside that envelope is rejected by step name, with the unsupported construct named — a multi-step plan, a depthwise or non-1x1 kernel, `add`, standalone requant/activation, upsampling, unsigned output, bias terms, shared-tile/hybrid engines, non-stream activation buffers, or requantization outside the 17-bit multiplier field. A layout that the backend cannot emit is a planning result to be revisited, never a silently altered plan (ADR-0003).

For this reason a whole-model layout (for example the W8 example's seven steps) solves successfully and is then rejected at `emit` with the full step list. The narrow envelope is reachable end to end from a pointwise plan, which `fixtures/pointwise` demonstrates, and simulation is verified against the Rust reference model when Verilator is available.

The backends share semantics and plan validation, not necessarily RTL implementation details.

## 9. Verification

Every compiler stage needs an executable oracle:

```text
Graph reference
  ↔ MLIR lowered reference
  ↔ plan simulator
  ↔ RTL simulation
  ↔ synthesis resource report
  ↔ board telemetry
```

Initial acceptance criteria:

1. Rust manifest import reproduces graph shapes and quantization metadata.
2. Rust/MLIR reference matches existing `ng.eval` on deterministic vectors.
3. Plan validation rejects resource or cycle overflow.
4. Generated RTL is bit-exact on supported operators *against the compiled
   fixed-point contract*: `(acc*M0 + round) >> shift`, half-away-from-zero. The training
   `f32` factors are approximated by the serialized `M0`/`shift`, and the
   worst-case relative error of that approximation is an artifact field, not a
   hidden rounding.
5. Efinity synthesis reports are parsed and compared with estimates.
6. A board run confirms output and schedule counters for one fixed model.

## 10. Project structure

Planned modules:

```text
crates/tessera/
  src/
    graph/       Graph IR and semantic validation
    frontend/    manifest and later ONNX import
    mlir/        Melior context, dialects, and passes
    hardware/    Hardware IR and execution plan
    target/      Target model implementations
    planner/     Candidate generation and constrained search
    rtl/         RTL IR and SystemVerilog emission
    verify/      reference, plan, RTL, and report comparison
    cli/         user-facing commands
```

The current repository begins as one crate to keep iteration cheap; modules may become workspace crates once interfaces stabilize.

## 11. Initial milestones

### M0: repository and contracts

- Rust workspace and CLI.
- Versioned plan schema.
- Manifest parsing for the current export.
- Golden vector import.

### M1: semantic graph

- Graph IR for the initial operator set.
- Shape, layout, quantization, and padding validation.
- Reference evaluator for deterministic small tensors.

### M2: MLIR lowering

- Manifest → Tessera MLIR.
- Stable quantization attributes.
- DW→requant→PW fusion legality and equivalence check.
- Buffer and loop lowering.

### M3: Ti60 planner

- Target resource model.
- Candidate folding and buffer placement.
- Cycle, RAM, DSP, and DDR estimates.
- Feasibility and Pareto report.

### M4: first RTL backend

- Shared tile kernel for one DW→PW block.
- BRAM ping-pong and fixed schedule.
- SystemVerilog generation.
- Verilator test against the fixed-point reference.

### M5: closed loop

- Full initial model.
- Efinity synthesis report parser.
- Cost-model calibration.
- Board integration without per-layer CPU scheduling.

### Status at v0.1

M1-M3 are implemented: the graph IR and manifest frontend, deterministic MLIR emission (with in-process optimization behind the optional `melior` feature), the Ti60 target model and layout search, and the persisted stage artifacts with cross-stage provenance verification. M4 is implemented for a deliberately narrow envelope — one streaming pointwise-dot kernel with a Verilator test against the fixed-point reference — and rejects the rest of the plan space by name, so M5 (whole-model RTL, synthesis reports, board measurement) remains open. DMA scheduling and the shared-tile backend are not implemented.

## 12. Non-goals and risks

### Risks

- Cost model may disagree with physical mapping.
- Fusion can accidentally repeat requantization inside an output-channel reduction.
- Buffer placement can be legal by capacity but impossible by port/bank topology.
- DDR overlap assumptions may fail under arbitration.
- MLIR version and Melior bindings may impose ecosystem friction.
- Generality can expand scope faster than verification can keep up.

### Mitigations

- Keep estimates and measurements separate.
- Start with one target and one model family.
- Make unsupported combinations fail explicitly.
- Validate every lowering with deterministic vectors.
- Preserve old NNgen/st_stream paths as temporary oracles.
- Add target and backend interfaces before adding more operators.

## 13. Naming and positioning

Tessera is intentionally neutral: it describes composition of tiles and does not lock the project to Efinix, one dataflow style, or one IR. The public description should be:

> Tessera is an open, extensible compiler that maps quantized neural-network graphs to resource-constrained FPGA execution plans and RTL.

It should not claim to be the first graph-to-RTL compiler. Its value is extensibility, inspectable plans, target plugins, open RTL, and a synthesis/board feedback loop.
