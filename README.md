# Tessera

**An extensible hardware-aware compiler for quantized neural networks on FPGAs.**

Tessera compiles a quantized model graph into a resource-constrained hardware execution plan and, eventually, synthesizable RTL. It is designed for open, inspectable FPGA backends rather than encrypted accelerator IP or CPU-driven layer orchestration.

> Working name: **Tessera** — a composition of small tiles into a complete architecture.

## Design thesis

```text
ONNX / manifest
      ↓
Graph IR
      ↓
MLIR optimization and progressive lowering
      ↓
Hardware-aware planning
      ↓
Execution Plan
      ↓
RTL + weights + runtime artifacts
```

MLIR owns graph, tensor, buffer, and loop transformations. Tessera-owned hardware IR and backends own FPGA resources, cycle models, memory placement, scheduling, and RTL emission.

## Initial target

The first target is Efinix Titanium Ti60, using the existing `nngen` / `fpga_st_train` manifest contract as a compatibility and correctness oracle. The architecture must not encode Ti60 assumptions into graph optimization APIs.

## Current status

- Workspace and CLI skeleton created.
- Concept book: [`docs/concept-book.md`](docs/concept-book.md).
- Initial architecture decision: [`docs/adr/0001-rust-mlir-boundary.md`](docs/adr/0001-rust-mlir-boundary.md).
- Quality standard: [`QUALITY.md`](QUALITY.md).
- Version-control standard: [`VERSIONING.md`](VERSIONING.md).
- No model import, MLIR lowering, planner, or RTL backend is implemented yet.

## First milestone

Support the existing W8A8 manifest for depthwise 3×3, pointwise 1×1, requantization, add, and nearest upsample; produce a deterministic execution plan; then generate and verify one shared-tile SystemVerilog backend.

## License

Apache-2.0.
