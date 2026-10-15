# Tessera

**An extensible hardware-aware compiler for quantized neural networks on FPGAs.**

Tessera compiles a quantized model graph into a resource-constrained hardware execution plan and, for a supported subset, synthesizable RTL. It is designed for open, inspectable FPGA backends rather than encrypted accelerator IP or CPU-driven layer orchestration.

> Working name: **Tessera** — a composition of small tiles into a complete architecture.

## Design thesis

Tessera is an explicit, persistent compilation pipeline rather than a one-shot model-to-RTL command:

```text
manifest.json  --import-->  graph.tgraph
               --emit-mlir-->  mlir.json / module.mlir
               --optimize-->   optimized.json / optimized.mlir    (opt-in `melior` feature)
               --solve-->      layout.tlayout + plan.tplan
               --emit-->       rtl/  (SystemVerilog, testbench, memory images)
               --simulate-->   Verilator result
```

Every stage produces an inspectable, versioned artifact in a stage directory, together with the copies of its inputs needed to re-check provenance later. Users can inspect graph and layout metrics, change hardware constraints, compare multiple layouts, and emit RTL only after selecting a validated layout. MLIR owns graph/tensor/buffer/loop transformations; Tessera-owned planning and backends own target resources, schedules, memory placement, and RTL emission.

## Initial target

The first target is Efinix Titanium Ti60, using the existing `nngen` / `fpga_st_train` manifest contract as a compatibility and correctness oracle. The architecture does not encode Ti60 assumptions into graph optimization APIs.

## Current status

- **[CLI reference](docs/cli.md)** — commands, stage directory contents, and the examples below.
- [Developer setup](docs/developer-setup.md) — default build, the optional `melior`/LLVM 23 toolchain, Verilator, fixture regeneration.
- [Concept book](docs/concept-book.md), [hardware constraint language](docs/constraint-language.md), [mapping solver](docs/solver.md), [versioning standard](VERSIONING.md), [quality standard](QUALITY.md).
- ADRs: [0001](docs/adr/0001-rust-mlir-boundary.md) Rust/Melior boundary, [0002](docs/adr/0002-persisted-compiler-stages.md) persisted stages, [0003](docs/adr/0003-stream-interfaces-and-the-v1-rtl-envelope.md) stream interfaces and the v1 RTL envelope.

Implemented end to end:

- **Frontend** (`src/graph.rs`, `src/frontend.rs`) — typed, target-independent graph IR for `fpga_st_train.int8.v1`: manifest import, i8-x1/i4-x2 packing checks, validation, metrics and the versioned `tessera.graph` artifact.
- **MLIR** (`src/mlir*.rs`) — deterministic tensor-level MLIR emission for the v1 operator subset, plus the pinned Melior 0.28.2 pass pipeline behind the optional `melior` feature.
- **Lowering** (`src/lower.rs`) — Graph IR to the planner's typed primitive graph, including fixed-point requantization serialization that preserves per-output-channel multipliers.
- **Planner** (`src/target.rs`, `src/planner.rs`, `src/layout.rs`) — Ti60 target model and layout search over execution mode, granularity, folding, fusion and memory placement, with rejected-candidate reporting and a validated `tessera.layout` artifact.
- **Mapping solver** (`src/solver/`) — a joint solve of engine count, operator binding, fold, schedule, pipeline cuts and arena placement through a solver-agnostic constraint model and an in-process CP-SAT backend (OR-Tools C++ through `cpsat-rs`), with independent verification and schema-2 lowering. See [docs/solver.md](docs/solver.md).
- **RTL** (`src/rtl.rs`, `src/rtl/full.rs`, `src/rtl/mapped.rs`, `src/weights.rs`) — SystemVerilog emission with a self-checking testbench and a Verilator simulation path, for three envelopes: a single streaming pointwise-dot step, whole models through a static-schedule FSM over an activation arena with one shared MAC array, and schema-2 mapped plans with per-engine arrays, time-multiplexing and optional pipeline FIFOs. Anything outside them is rejected by step name.
- **Verification** (`src/stages.rs`) — artifact detection, cross-stage provenance re-checks, and deterministic output checks.

The W8 example solves to a valid seven-step shared-tile layout under
`fixtures/policies/emit.toml`, then emits and simulates through the full-graph
backend. Open work: DMA scheduling and frame overlap, synthesis reports and board
measurement (see [concept book](docs/concept-book.md) §8 and §11).

## Quick start

The default `cargo build` links the native CP-SAT backend, so it needs the
OR-Tools C++ 9.15 and `protoc` prerequisites described in
[Developer setup](docs/developer-setup.md) before the build below. Everything
after that uses checked-in fixtures and needs no external model files:

```sh
cargo build --release

# Solve, emit, verify and simulate the supported pointwise path.
./target/release/tessera solve --primitive fixtures/pointwise/plan_input.json \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/streaming.toml --out build/pointwise
./target/release/tessera emit build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise
./target/release/tessera verify build/pointwise
./target/release/tessera simulate build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise

# Run the whole-model path up to the layout, and observe the named RTL rejection.
./target/release/tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
./target/release/tessera emit-mlir build/w8/graph.tgraph --out build/w8
./target/release/tessera solve --graph build/w8/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/explore.toml --out build/w8
./target/release/tessera inspect build/w8/layout.tlayout
```

## Development

The default build links the native CP-SAT backend, so it needs OR-Tools C++ 9.15
and `protoc`; `ORTOOLS_PREFIX` (default search: the usual prefixes, then
`/opt/ortools`) selects the installation, and its `lib` must be on the loader
path for anything that links the backend — the mapping-solver tests today (see
[developer setup](docs/developer-setup.md)):

```sh
export ORTOOLS_PREFIX=/opt/ortools
export LD_LIBRARY_PATH="$ORTOOLS_PREFIX/lib:$LD_LIBRARY_PATH"

cargo fmt --all --check
cargo check --all-targets
cargo test  --all-targets
cargo test  -p tessera --features melior   # requires LLVM/MLIR 23.x, see docs/developer-setup.md
cargo run   -p tessera --example generate_fixtures   # regenerate fixtures/ deterministically
```

## License

Apache-2.0.
