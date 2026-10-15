# Tessera artifacts

Every stage of the compiler writes one versioned artifact into a stage
directory, and every artifact is inspectable, re-validatable and attributable.
This document lists what a full run produces, what each file proves, how to
reproduce it, and — explicitly — what it does not prove.

## What a run produces

| File | Format | Produced by | What it is |
| --- | --- | --- | --- |
| `graph.tgraph` | `tessera.graph` (JSON) | `import`, `solve --graph` | Typed, target-independent graph plus import provenance (manifest and weight digests) |
| `<weight-file>` | raw blob | `import`, `solve --graph` | Copy of the manifest weight blob, checked against its recorded SHA-256 |
| `input_shape.txt` | `N,H,W,C` | `import --input-shape`, `solve` | The concrete resolution the manifest cannot carry |
| `mlir.json` | `tessera.mlir` (JSON) | `emit-mlir` | Tensor-stage module, arguments, layer/operator bindings, structural metrics |
| `module.mlir` | MLIR text | `emit-mlir` | The emitted module, byte-for-byte deterministic for a graph and shape |
| `optimized.json` | `tessera.mlir` (JSON) | `optimize` | Planning-stage module with the pinned pipeline's provenance and the structured `analysis` (operations, loop nests, buffers, `loc` tags) |
| `optimized.mlir` | MLIR text | `optimize` | The module after the pinned pass pipeline |
| `primitive.json` | `PlanInput` (JSON) | `solve` | The planner's contract: tensors (with element type and exact code interval), operators, graph I/O |
| `primitives.json` | `tessera.primitive` (JSON) | `solve --mlir` | The same plan plus its evidence: code intervals per activation tensor, loop groups mapped to operators (with fused nests), materialized buffers, and MLIR provenance |
| `target.toml`, `policy.toml` | `tessera.target` / `tessera.policy` | `solve` | Copies of the constraints the layout was solved against, so the recorded hashes can be re-checked later |
| `layout.tlayout` | `tessera.layout` (TOML) | `solve` | The selected organization, schedule, buffers, resource estimate and slack, per-mode rejection summary, search statistics, `certainty`, and provenance |
| `solved.mlir` | Tessera dialect (MLIR text) | `solve --tessera` | Dialect with engine / bind / buffer / segment / schedule decisions |
| `rtl/` | bundle | `emit --tessera` | CIRCT ExportVerilog SystemVerilog, self-checking testbench, bundle metadata (`rtl_bundle.json`), reproduce script |
| `sim/` | Verilator work dir | `simulate` | Compiled model, simulator log, and the trace the testbench dumps |

The **code interval** recorded next to each activation tensor is what sizes the
datapath: a residual `add` produces codes outside `[-128, 127]`, so it is carried
as nine-bit values in a sixteen-bit word rather than clamped back to int8. That
interval is computed by the lowering (from the graph) or read from the MLIR
analysis (from the optimized module), never inferred from the storage container.
Only activation tensors have one: a weight or bias tensor is described by its
plan entry (name, shape, element type) and carries no interval, so a 4-bit weight
can never be read as a 32-bit datapath width. A dynamically shaped allocation
similarly has no statically known size, and the module's byte total is then
reported as unknown rather than as a partial sum presented as the whole.

The **`certainty`** field distinguishes a proven result from a truncated one:
`optimal` and `proven_infeasible` mean the declared space was covered; `best_effort`
and `inconclusive` mean the candidate budget stopped the search. `PlanStatus`
says what was found; `PlanCertainty` says how much of the space it was proven
over. A truncated search is never presented as a property of the target.

## Reproducing the two end-to-end runs

Both runs use only committed fixtures. The synthetic W8 model exercises the whole
operator set; the real `ink_dw` model exercises a trained 29-layer network.

```sh
cargo build --release
TESSERA=./target/release/tessera

# 1. Synthetic W8: 3x3 conv, depthwise 3x3, 1x1 convs, residual add, nearest
#    upsample, unsigned head, non-zero int32 bias on the stem.
$TESSERA import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
$TESSERA emit-mlir build/w8/graph.tgraph --out build/w8
$TESSERA solve --graph build/w8/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8
$TESSERA emit --tessera build/w8/solved.mlir --graph build/w8/graph.tgraph --out build/w8
$TESSERA simulate build/w8/layout.tlayout --graph build/w8/graph.tgraph --out build/w8
$TESSERA verify build/w8

# 2. The same chain through the optimized module (needs the `melior` feature and
#    LLVM/MLIR 23.x, see developer-setup.md).
$TESSERA optimize build/w8/mlir.json --out build/w8
$TESSERA solve --mlir build/w8/optimized.json \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8-mlir
$TESSERA inspect build/w8-mlir/primitives.json
$TESSERA verify build/w8-mlir

# 3. The real ink_dw model, at a resolution that fits the Ti60 reference budget.
$TESSERA import fixtures/ink_dw --input-shape 1,16,16,3 --out build/ink
$TESSERA solve --graph build/ink/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/ink
$TESSERA emit --tessera build/ink/solved.mlir --graph build/ink/graph.tgraph --out build/ink
$TESSERA verify build/ink
```

`fixtures/policies/emit.toml` is the policy shaped for the CIRCT hybrid solve
(frame granularity, lane bounds the CP-SAT mapping respects). The generic
`streaming.toml` and `explore.toml` policies are deliberately unconstrained, so
the difference between "the best mapping the solver can find" and "a mapping the
`*_sched_core` path can emit" stays visible instead of being hidden by the fixture.

## What `verify` does and does not prove

`tessera verify <dir>` re-checks, for every artifact present:

* that each artifact re-validates against its own schema;
* that `mlir.json` descends from the staged graph artifact (source format,
  schema and manifest hash);
* that `optimized.json` records the Melior pipeline that produced it;
* that `primitives.json` descends from the optimized module it names;
* that `layout.tlayout` names the hash of the staged input it was solved from
  (graph, optimized module or primitive graph) and of the staged target and
  policy copies — editing either after solving is detected;
* that `plan.tplan` carries exactly the layout's provenance;
* that every file in `rtl/` matches its recorded size and checksum.

It does **not** prove numerical correctness. Simulation against the Rust
reference does, and an independent scalar reference (`tessera::rtl::reference`)
judges the emitted trace; the reference's own divergence from the manifest `f32`
contract is reported as `semantic_notes` rather than hidden.

Every number in a layout is a *model estimate*, not a measurement. Cycles and
resources are labelled as estimates; the generated structure's own numbers
(arena words and width, generated unstalled cycles) are printed separately by
`emit` so the two are never confused. Synthesis and board measurements are not
implemented, so nothing here claims them.

## Measured results

These numbers come from the committed snapshots under `fixtures/snapshots/`, so
each one can be re-read from the repository rather than taken on trust. The
planner's figures are model estimates; the generated figures come from the emitted
structure, which is why both are printed together.

**Synthetic W8, seven steps** (`fixtures/policies/emit.toml`, input `1,6,6,3`):

| Quantity | Value | Source |
| --- | --- | --- |
| organization | `shared_tile`, frame granularity, 7 steps, `simd = pe = tap_parallel = 1` | plan |
| planner resources | 3 DSP48, 7 RAM10, 2500 XLR, 10 869 frame cycles | estimate |
| activation arena | 1404 words x 16 bits = 2808 bytes (~3 RAM10 of 10 240 bits), budget 1404, without reuse 2124 | generated vs plan |
| MACs | 1 lane, 7488 MACs per frame | generated |
| cycles | 9770 unstalled generated vs 10 785 plan estimate | generated vs estimate |
| RTL simulation | CIRCT `*_sched_core`, bit-exact vs integer ref (0 mismatches) | Verilator |
| busy one-hot | `engines_busy` reflects bind decisions (no per-engine FSMs) | emitted DUT |

**Real ink_dw, 29 layers / 61 graph ops / 37 steps** (`fixtures/policies/emit.toml`,
input `1,8,8,3`):

| Quantity | Value | Source |
| --- | --- | --- |
| organization | `shared_tile`, frame granularity, 37 steps, single lane | plan |
| planner resources | 4 DSP48, 69 RAM10 of 88, 2500 XLR, 491 486 frame cycles | estimate |
| activation arena | 4608 words x 16 bits, budget 4608, without reuse 21 056 | generated vs plan |
| MACs | 1 lane, 399 104 MACs per frame | generated |
| cycles | 407 114 unstalled generated vs 491 424 plan estimate | generated vs estimate |
| RTL simulation | CIRCT hybrid path when solved + emitted | Verilator |
| busy one-hot | bind → `engines_busy` | emitted DUT |

**Resolution sweep** for the same real model (limits: 152 DSP48, 88 RAM10,
40 568 XLR; every result is a complete or proven-infeasible search, never
truncated):

| input `N,H,W,C` | status | DSP48 | RAM10 / 88 | frame cycles |
| --- | --- | --- | --- | --- |
| `1,8,8,3` | feasible | 4 | 69 | 491 486 |
| `1,16,8,3` | feasible | 4 | 78 | 980 574 |
| `1,16,16,3` | proven infeasible | - | 96 | - |
| `1,40,20,3` (the exporter's reference size) | proven infeasible | - | 173 | - |
| `1,480,640,3` | proven infeasible | - | - | exceeds the frame-cycle budget |

Above 16x8 the blocker is the activation arena, which is frame-sized, on top of the
resolution-independent 49 RAM10 blocks the 50 144-byte weight blob occupies. No
budget was raised and no rejection was hidden: the whole-model RTL evidence above
comes from the resolutions that do fit.

Two properties are checked rather than asserted:

* **the arena allocator is validated from the artifact.** `verify` derives each
  slot's live range from the bundle and fails when two simultaneously live slots
  overlap in address, so the value of reuse is not taken on faith; the same check
  holds the emitted high-water mark at or below the budget the plan was accepted
  with. Both bundles satisfy it, and the emitted high-water mark equals the plan's
  budget exactly (1404 and 4608 words) — with the no-reuse figure printed beside it
  so the two stay distinguishable;
* **the numerics are named, not implied.** `solve` and `inspect` print
  `tessera.fixed_point.requant.half_away_from_zero.v1` with the multiplier width
  and the worst recorded `Requant::multiplier_error_ppm` against the manifest `f32`
  coefficients, `verify` re-checks that every step uses that rounding, and the
  bundle reports how many steps differ from the manifest `f32` with the worst
  per-step code distance. The exactness baseline is the compiled fixed-point IR.

## Review bundles

`emit` writes the bundle into `<stage>/rtl`: the SystemVerilog DUT and
self-checking testbench, the memory images (weights, requantization, stimulus,
expected codes), the reference step trace, a `run_verilator.sh` reproduce script,
and `rtl_bundle.json` with the plan, provenance, per-file digests, the generated
structure and the arena slots.

Two complete stage directories are committed as review fixtures, so the numbers
above can be re-read and re-checked from the repository without an FPGA toolchain:

| Snapshot | Contents | Size | Checks |
| --- | --- | --- | --- |
| `fixtures/snapshots/w8/` | graph, weights, emitted and optimized MLIR, primitive graph, target and policy copies, layout, plan, `rtl/` bundle, `sim/sim_outputs.txt` | 272 KB | `tessera verify` ok, 19 checks |
| `fixtures/snapshots/ink_dw/` | the same for the real 29-layer model at 1x8x8x3 | 3.1 MB | `tessera verify` ok, 16 checks |

`tessera verify fixtures/snapshots/<name>` re-checks the module text hash, the
primitive-to-module link, the recorded input/target/policy hashes, the semantic
contract, arena disjointness and the arena high-water mark, and every generated
file digest — so a committed snapshot is as checkable as a freshly emitted bundle.
The Verilator object directory is not committed (it is the simulator's build
output, not the compiler's); the per-step trace and the expected codes it was
compared against are.
