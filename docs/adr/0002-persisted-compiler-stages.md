# ADR-0002: Persisted multi-stage compiler artifacts

## Status

Accepted

## Context

A model-to-RTL compiler has multiple independently changing concerns: model import and semantics, MLIR optimization, progressive lowering, target constraints, hardware layout search, RTL generation, and verification. A single command that performs all stages makes intermediate graph quality, resource decisions, and failed assumptions difficult to inspect. It also forces users to repeat expensive or unchanged work when they only want to adjust constraints or compare layouts.

Tessera must support an iterative workflow in which users can inspect optimized graphs, examine graph metrics, edit target/policy constraints, solve multiple layouts, and generate RTL only from a chosen layout.

## Decision

Treat Tessera as a persistent, staged compiler. Each stage consumes and produces a versioned, inspectable artifact:

```text
model description
  → graph artifact
  → optimized graph artifact
  → lowered primitive artifact
  → user target/policy files
  → layout artifact
  → RTL artifact bundle
  → simulation/synthesis/board result artifacts
```

Users may stop, inspect, compare, and resume at every stage. A new constraint or policy file must allow layout search to rerun without repeating import and optimization. RTL emission consumes a validated saved layout and may not silently re-solve it.

Target facts and optimization policy are separate inputs. The solver emits a versioned layout containing the selected mode, engine bindings, folds, buffers, schedule, resource estimates, rejected-candidate summary, and provenance. The word “layout” refers to logical hardware organization, not physical FPGA placement and routing.

TOML is the initial proposed user syntax for target and policy files; schemas remain versioned and may evolve based on implementation experience.

## Consequences

### Positive

- Optimized graphs and layouts can be audited and reused.
- Users can compare execution organizations against the same graph.
- Failures are localized to a stage without discarding earlier results.
- RTL generation becomes a deterministic lowering of a selected layout.
- Resource estimates and measured synthesis/board results remain separate artifacts.

### Negative

- Every persisted artifact needs a schema, provenance, compatibility check, and inspection tooling.
- Cache invalidation must account for compiler passes, target revisions, and policy inputs.
- Stable serialization becomes a compatibility commitment.
- More explicit commands add CLI surface compared with a single one-shot flow.

## Alternatives considered

### One-shot model-to-RTL command

Rejected as the only workflow because it hides intermediate decisions and makes constraint iteration expensive. A convenience one-shot command may later orchestrate the same explicit stages.

### Keep intermediate states only in memory

Rejected because it prevents review, reuse, graph metric inspection, and reproducible comparisons.

### Let RTL emission modify the layout

Rejected because it makes solver output non-reproducible and mixes planning decisions with backend implementation. Unsupported layouts must fail validation or return to the solver.

## Verification obligations

- Each artifact declares schema and provenance.
- A saved optimized graph can be solved against at least two policies without re-import.
- A layout can be inspected without RTL emission.
- RTL emission from the same layout is deterministic.
- Incompatible or unsupported artifacts fail with actionable diagnostics.
