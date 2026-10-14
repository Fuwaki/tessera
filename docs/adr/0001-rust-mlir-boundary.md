# ADR-0001: Rust owns the compiler shell; Melior owns progressive IR optimization

## Status

Accepted

## Context

The existing workspace combines Python and Rust experiments. NNgen and `st_stream` provide useful model semantics, RTL templates, and validation artifacts, but their backends use incompatible execution models. The new compiler needs explicit hardware planning, resource constraints, and extensible RTL generation.

Melior/MLIR already provides mature infrastructure for SSA IR, pattern rewrites, tensor/buffer transformations, and progressively lower-level operations. It does not provide the Ti60 resource model, execution-plan search, or the required RTL backend.

## Decision

Build Tessera as a Rust-first compiler. Use Melior/MLIR for graph, tensor, buffer, and loop optimization and progressive lowering. Keep hardware-specific scheduling, resource modeling, execution plans, and RTL generation in Tessera-owned Rust modules.

The initial frontend accepts the existing manifest contract. ONNX import is a later frontend that targets the same Graph IR. NNgen and `st_stream` remain compatibility and correctness oracles during migration.

## Consequences

### Positive

- MLIR transformations can be reused without making MLIR encode all FPGA details.
- Rust types can enforce plan and target invariants.
- Ti60-specific constraints remain replaceable target data rather than graph-level assumptions.
- Dedicated, streaming, and shared-tile backends can consume one execution-plan contract.
- Existing Python implementations remain useful for differential verification.

### Negative

- A conversion boundary between Rust Graph IR, MLIR, Hardware IR, and RTL IR must be maintained.
- Melior/MLIR dependency and version management remain non-trivial.
- The project must implement its own target-aware planner and RTL backend.
- Initial functionality will be narrower than the existing NNgen frontend.

## Rejected alternatives

### Replace NNgen in Python

Rejected because it preserves the current language split and makes the incompatible backend assumptions harder to isolate.

### Use generic MLIR directly as RTL input

Rejected because generic tensor/loop IR does not preserve enough explicit information about DSP packing, RAM banking, DMA, FIFO, and static schedule choices.

### Fork FINN or hls4ml as the primary implementation

Rejected for now because their architecture and vendor/tool assumptions do not match the intended open Ti60 shared-tile backend. Their design patterns remain useful references.

## Verification obligation

The Rust/MLIR pipeline must compare against existing `ng.eval`, `stref`, and RTL vectors before replacing any current backend path.
