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
- Deterministic RTL generation and bit-exact verification.
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

## 3. Compiler contract

```text
Model frontend
  → Graph IR
  → MLIR graph/tensor/buffer/loop optimization
  → Hardware IR
  → Target-aware design-space search
  → Execution Plan
  → RTL backend
  → simulation / synthesis / board validation
```

The boundaries are deliberate:

- **Graph IR** owns model semantics.
- **MLIR** owns legal graph and loop transformations.
- **Hardware IR** owns tiles, engines, buffers, streams, DMA, and schedules.
- **Target model** owns resource and cost rules.
- **Execution Plan** is the reproducible compiler output before code generation.
- **RTL backend** owns module binding and cycle-level implementation.

## 4. IR layers

### 4.1 Graph IR

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

### 4.2 MLIR layer

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

### 4.3 Hardware IR

Hardware IR describes a realization:

- engines and operator bindings;
- spatial and temporal folding;
- row, pixel, tile, or frame execution;
- line buffers, FIFOs, scratchpads, BRAM, and DDR buffers;
- ping-pong ownership and DMA operations;
- fusion boundaries;
- schedule steps and dependencies;
- resource reservations.

### 4.4 Execution Plan

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

## 5. Target model

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

## 6. Planner

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

The first solver should use candidate enumeration, pruning, and branch-and-bound. A CP-SAT solver may be introduced after the cost model and plan schema stabilize.

## 7. RTL backends

Backends consume the same Execution Plan but may emit different structures:

### Dedicated backend

Generates operator-specialized FSM/stream hardware. This preserves the useful NNgen-style baseline.

### Streaming backend

Generates ready/valid layers from the existing `st_stream` primitive library: windows, depthwise, MVAU, requant, FIFO, and ROM.

### Shared-tile backend

Generates a configurable MAC engine, tile buffers, line buffers, DMA, ping-pong control, and a static schedule engine. This is the primary new backend.

The backends share semantics and plan validation, not necessarily RTL implementation details.

## 8. Verification

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
4. Generated RTL is bit-exact on supported operators.
5. Efinity synthesis reports are parsed and compared with estimates.
6. A board run confirms output and schedule counters for one fixed model.

## 9. Project structure

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

## 10. Initial milestones

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
- Verilator bit-exact test.

### M5: closed loop

- Full initial model.
- Efinity synthesis report parser.
- Cost-model calibration.
- Board integration without per-layer CPU scheduling.

## 11. Non-goals and risks

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

## 12. Naming and positioning

Tessera is intentionally neutral: it describes composition of tiles and does not lock the project to Efinix, one dataflow style, or one IR. The public description should be:

> Tessera is an open, extensible compiler that maps quantized neural-network graphs to resource-constrained FPGA execution plans and RTL.

It should not claim to be the first graph-to-RTL compiler. Its value is extensibility, inspectable plans, target plugins, open RTL, and a synthesis/board feedback loop.
