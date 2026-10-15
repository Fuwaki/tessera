# Tessera

**An extensible hardware-aware compiler for quantized neural networks on FPGAs.**

Tessera compiles a quantized model graph into a resource-constrained hardware
execution plan and synthesizable RTL. It is designed for open, inspectable FPGA
backends rather than encrypted accelerator IP or CPU-driven layer orchestration.

> Working name: **Tessera** — a composition of small tiles into a complete architecture.

## Design thesis

Tessera is an explicit, persistent compilation pipeline rather than a one-shot
model-to-RTL command:

```text
manifest.json
    --import-->     graph.tgraph
    --emit-mlir-->  mlir.json / module.mlir   (linalg / tensor)
    --optimize-->   optimized.json            (opt-in `melior` feature)
    --solve-->      tessera.mlir + solved.mlir  (tessera dialect + CP-SAT)
                    (tessera.engine / bind / buffer / segment / schedule)
    --emit-->       rtl/   (CIRCT hw/comb/sv → in-process ExportVerilog)
    --simulate-->   Verilator bit-exact check vs integer reference
```

Every stage produces an inspectable, versioned artifact in a stage directory,
together with the copies of its inputs needed to re-check provenance later.
Users can inspect graph and layout metrics, change hardware constraints, compare
multiple layouts, and emit RTL only after a solved dialect module is available.
MLIR owns graph/tensor/buffer transformations; Tessera-owned planning and
backends own target resources, schedules, memory placement, and RTL emission.

## Initial target

The first target is Efinix Titanium Ti60, using the existing `nngen` /
`fpga_st_train` manifest contract as a compatibility and correctness oracle.
The architecture does not encode Ti60 assumptions into graph optimization APIs.

## Current status

- **[CLI reference](docs/cli.md)** — commands, stage directory contents, and the examples below.
- [Developer setup](docs/developer-setup.md) — default build, the optional `melior`/LLVM 23 toolchain, CIRCT, Verilator, fixture regeneration.
- [Concept book](docs/concept-book.md), [hardware constraint language](docs/constraint-language.md), [mapping solver](docs/solver.md), [versioning standard](VERSIONING.md), [quality standard](QUALITY.md).
- ADRs: [0001](docs/adr/0001-rust-mlir-boundary.md) Rust/Melior boundary, [0002](docs/adr/0002-persisted-compiler-stages.md) persisted stages, [0003](docs/adr/0003-stream-interfaces-and-the-v1-rtl-envelope.md) stream interfaces (historical; RTL path superseded by CIRCT hybrid).

Implemented end to end:

- **Frontend** (`src/graph.rs`, `src/frontend.rs`) — typed, target-independent graph IR for `fpga_st_train.int8.v1`: manifest import, i8-x1/i4-x2 packing checks, validation, metrics and the versioned `tessera.graph` artifact.
- **MLIR** (`src/mlir*.rs`) — deterministic tensor-level MLIR emission for the v1 operator subset, plus the pinned Melior 0.28.2 pass pipeline behind the optional `melior` feature.
- **Tessera dialect + mapping solver** (`src/dialect/`, `src/solver/`) — operators and decision ops (`tessera.engine` / `bind` / `buffer` / `segment` / `schedule`); joint CP-SAT solve (OR-Tools through `cpsat-rs`) writes decisions back into the same Melior module. See [docs/solver.md](docs/solver.md).
- **RTL** (`src/rtl.rs`, `src/rtl/circt_*.rs`) — CIRCT hybrid path only (`tessera.hybrid.circt.v1`): solved dialect → CIRCT `hw`/`comb`/`sv` in the same Melior context → in-process ExportVerilog. The DUT datapath is a single integer `*_sched_core` (weight/requant ROMs, i16×i8 MAC, integer requant, add/upsample). Solved `tessera.engine` / bind ops are **mapping decisions** in the dialect; they surface on the DUT as `engines_busy` (one-hot of the engine that owns the core's current op), not as separate per-engine FSMs. Real per-engine microarchitecture is deferred.
- **Verification** (`src/stages.rs`, Verilator TB) — artifact detection, cross-stage provenance re-checks, and bit-exact simulation against the compiled fixed-point contract `(acc*M0 + round) >>> shift`.

Open work: per-engine datapath FSMs, DMA scheduling and frame overlap, synthesis reports and board measurement (see [concept book](docs/concept-book.md) §8).

## Quick start

The default `cargo build` links the native CP-SAT backend, so it needs the
OR-Tools C++ 9.15 and `protoc` prerequisites described in
[Developer setup](docs/developer-setup.md). CIRCT ExportVerilog needs the pinned
CIRCT install (`~/.local/share/circt23` by default). Everything after that uses
checked-in fixtures:

```sh
cargo build --release

# Import → MLIR → solve on tessera dialect → CIRCT emit → Verilator.
./target/release/tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
./target/release/tessera emit-mlir build/w8/graph.tgraph --out build/w8
# Convert / annotate to tessera dialect (see docs/cli.md), then:
./target/release/tessera solve --tessera build/w8/tessera.mlir \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8
./target/release/tessera emit --tessera build/w8/solved.mlir \
    --graph build/w8/graph.tgraph --out build/w8
./target/release/tessera verify build/w8
```

`--graph` is required for content-filled ROMs and the bit-exact testbench.
Without it, emit still produces structure but cannot resolve weight/requant
bytes from the manifest.

## Development

```sh
export ORTOOLS_PREFIX=/opt/ortools
export LD_LIBRARY_PATH="$ORTOOLS_PREFIX/lib:$LD_LIBRARY_PATH"

cargo fmt --all --check
cargo check --all-targets
cargo test  --all-targets
cargo test  -p tessera --features melior   # requires LLVM/MLIR 23.x
cargo run   -p tessera --example generate_fixtures
```

## License

Apache-2.0.
