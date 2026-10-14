# Tessera

**An extensible hardware-aware compiler for quantized neural networks on FPGAs.**

Tessera compiles a quantized model graph into a resource-constrained hardware execution plan and, eventually, synthesizable RTL. It is designed for open, inspectable FPGA backends rather than encrypted accelerator IP or CPU-driven layer orchestration.

> Working name: **Tessera** — a composition of small tiles into a complete architecture.

## Design thesis

Tessera is an explicit, persistent compilation pipeline rather than a one-shot model-to-RTL command:

```text
model description
  → graph.tgraph
  → optimized.tgraph / lowered.tir
  → target.toml + policy.toml
  → layout.tlayout
  → RTL + weights + reports
  → simulation / synthesis / board measurements
```

Every stage produces an inspectable, versioned artifact. Users can inspect graph metrics, change hardware constraints, compare multiple layouts, and emit RTL only after selecting a validated layout. MLIR owns graph/tensor/buffer/loop transformations; Tessera-owned planning and backends own target resources, schedules, memory placement, and RTL emission.

## Initial target

The first target is Efinix Titanium Ti60, using the existing `nngen` / `fpga_st_train` manifest contract as a compatibility and correctness oracle. The architecture must not encode Ti60 assumptions into graph optimization APIs.

## Current status

- Workspace and CLI skeleton created.
- Concept book: [`docs/concept-book.md`](docs/concept-book.md).
- Initial architecture decision: [`docs/adr/0001-rust-mlir-boundary.md`](docs/adr/0001-rust-mlir-boundary.md).
- Hardware constraint language: [`docs/constraint-language.md`](docs/constraint-language.md).
- Quality standard: [`QUALITY.md`](QUALITY.md).
- Multi-stage artifact decision: [`docs/adr/0002-persisted-compiler-stages.md`](docs/adr/0002-persisted-compiler-stages.md).
- No model import, MLIR lowering, planner, or RTL backend is implemented yet.

## First milestone

Support the existing W8A8 manifest for depthwise 3×3, pointwise 1×1, requantization, add, and nearest upsample; produce a deterministic execution plan; then generate and verify one shared-tile SystemVerilog backend.

## License

Apache-2.0.
