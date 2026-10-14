# Tessera CLI

Tessera is a persistent, staged compiler: every command consumes a versioned
artifact and writes a versioned artifact into a **stage directory**. Any stage
can be inspected, compared, or rerun without repeating earlier work, and a
failed later stage never invalidates an earlier artifact (ADR-0002).

```text
manifest.json --import--> graph.tgraph --emit-mlir--> mlir.json
              --solve--> layout.tlayout + plan.tplan --emit--> rtl/
```

Run `tessera <command> --help` for the authoritative flags; this document
mirrors them.

## Stage directory

`--out <DIR>` names the stage directory. Commands create it and never delete
anything in it:

| File | Produced by | Contents |
| --- | --- | --- |
| `graph.tgraph` | `import`, `solve --graph` | `tessera.graph` artifact (JSON) |
| `<weight-file>` | `import`, `solve --graph` | copy of the manifest weight blob, verified against its recorded SHA-256 |
| `input_shape.txt` | `import --input-shape` | concrete `N,H,W,C` input resolution |
| `mlir.json` | `emit-mlir` | `tessera.mlir` artifact, tensor stage (JSON) |
| `module.mlir` | `emit-mlir` | emitted MLIR text |
| `optimized.json` | `optimize` | `tessera.mlir` artifact, planning stage (JSON) |
| `optimized.mlir` | `optimize` | optimized MLIR text |
| `primitive.json` | `solve` | the lowered planner graph (`PlanInput`) |
| `primitives.json` | `solve --mlir` | the `tessera.primitive` artifact: plan, code intervals, fusion groups, buffers, MLIR provenance |
| `target.toml`, `policy.toml` | `solve` | copies of the constraint inputs the layout was solved against |
| `layout.tlayout` | `solve` | `tessera.layout` artifact (TOML) |
| `plan.tplan` | `solve`, `emit`, `simulate` | `tessera.execution_plan` projection (TOML) |
| `rtl/` | `emit`, `simulate` | the RTL bundle (SystemVerilog, testbench, memory images, `rtl_bundle.json`) |
| `sim/` | `simulate` | Verilator work directory |

`--out` is a convenience, not a database: every command also accepts an
artifact path directly, so a user may keep artifacts wherever they like.

## Commands

### `import`

```sh
tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
```

Imports a `fpga_st_train.int8.v1` manifest (a `manifest.json` path or the
directory holding it), validates it, and writes `graph.tgraph`. The weight blob
named by the manifest is copied into the stage directory and re-checked against
the digest the frontend recorded, so later stages never depend on the manifest's
original location.

`--input-shape N,H,W,C` (NHWC, batch 1) is optional here but required before
`solve` or `emit-mlir`: the manifest contract carries no spatial extents.
Recording it at import time means later stages do not have to repeat it.

### `emit-mlir`

```sh
tessera emit-mlir build/w8/graph.tgraph --out build/w8
```

Emits deterministic tensor-level MLIR (`linalg`/`arith`/`tensor`) for the
validated v1 operator subset and writes both the artifact and the module text.
This stage is pure Rust and always available. Unsupported operators are
rejected by name; nothing is silently dropped.

### `optimize`

```sh
tessera optimize build/w8/mlir.json --out build/w8
```

Runs the pinned pass pipeline through Melior 0.28.2:
`linalg-generalize-named-ops`,
`one-shot-bufferize{bufferize-function-boundaries}`,
`convert-linalg-to-affine-loops`, `affine-loop-fusion{mode=producer}`,
`canonicalize`, `cse`.

Requires the optional `melior` feature and an LLVM/MLIR 23.x installation; see
[developer-setup.md](developer-setup.md). A build without the feature refuses to
run the stage and says so — it never pretends the module was optimized.

### `solve`

```sh
tessera solve --graph build/w8/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/explore.toml --out build/w8

tessera solve --primitive fixtures/pointwise/plan_input.json \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/streaming.toml --out build/pointwise
```

Lowers the input to the planner's typed primitive graph, copies the target and
policy into the stage directory, runs the layout search, and writes
`layout.tlayout` plus, when the plan is feasible, `plan.tplan`. Exactly one
input is selected:

* `--graph` lowers the manifest graph directly (convolution layers, `add`,
  nearest upsample and the unsigned output quantization). The manifest's
  per-channel requantization is serialized to fixed-point `(M0, shift)` and
  carried verbatim in the layout.
* `--mlir` lowers the **optimized module** instead: `tessera::primitive::lower`
  builds the plan from the module's own operations, loop nests and materialized
  buffers and verifies it against the module's analysis, so this path cannot
  re-derive a plan behind MLIR's back. It writes `primitive.json` (the planner's
  contract), `primitives.json` (the `tessera.primitive` artifact, with code
  intervals, fusion groups, buffers and MLIR provenance) and records the input
  resolution the module was emitted for.
* `--primitive` solves a primitive graph (either form) directly, which is how
  the pointwise fixture reaches the narrow RTL envelope without a manifest.
* An **infeasible** or **inconclusive** search is written, reported, and the
  command exits non-zero: no candidate is ever chosen silently.

`solve` also prints the numerical contract it produces: the identifier
`tessera.fixed_point.requant.half_away_from_zero.v1`, the half-away-from-zero rounding and multiplier width, and
the worst recorded step error of the serialized `M0`/`shift` against the
manifest's `f32` coefficients. The exactness baseline is the **compiled
fixed-point IR** — the training `f32` requantization factors are approximated,
with that bound recorded per step rather than described as bit-exact.

The search report distinguishes a proven result from a truncated one: `solve`
prints the layout's `certainty` (`optimal`, `best_effort`, `proven_infeasible`,
`inconclusive`) with the candidate/expansion counts, and reports per-execution-mode
candidates, feasible candidates and the first rejection reason. A search that hit
its candidate budget is reported as INCONCLUSIVE, never as "no feasible layout".

### `inspect`

```sh
tessera inspect build/w8/graph.tgraph
tessera inspect build/w8/layout.tlayout
tessera inspect build/pointwise/rtl/rtl_bundle.json
tessera inspect build/w8/graph.tgraph --raw
```

Prints a summary of any persisted artifact — graph metrics, MLIR operators and
structural metrics, the lowered primitive graph, layout resources/schedule/
rejection counts, execution-plan steps, or RTL bundle files. `--raw` prints the
artifact's own serialization. An unrecognised file is reported, never guessed
at.

### `emit`

```sh
tessera emit build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise

tessera emit build/w8/layout.tlayout \
    --graph build/w8/graph.tgraph --out build/w8
```

Projects the validated layout onto an execution plan and emits the RTL bundle.
`--weights <DIR>` resolves `<tensor-name>.bin` blobs for primitive fixtures;
`--graph <graph.tgraph>` resolves weight bytes from the manifest blob through
the layer ranges the frontend recorded, re-checked against the artifact's
SHA-256.

Emission fails closed, and dispatches on the plan's shape:

* **One pointwise step** goes to the single-kernel backend: signed int8
  activations, ROM-resident weights, streamed activation buffers, one
  requantization contract, no bias.
* **Any other plan** goes to the full-graph backend, which emits one
  static-schedule FSM over an activation arena with a single shared MAC array.
  It requires a `shared_tile` frame schedule whose steps ask for `tap_parallel
  == 1` and at most 64 MAC lanes, and it refuses a streaming- or
  dedicated-labelled plan by name rather than relabelling it. Because it resolves
  bias from the model, it needs `--graph`, not `--weights`.

Anything outside both envelopes is reported by step name, for example:

```text
error: rtl: unsupported layout: v1 emits exactly one kernel, but the plan has 7
steps [stem(Conv2d), dw(DepthwiseConv2d), ...]; the supported envelope is a
single pointwise step
```

For the full-graph path `emit` prints the generated structure next to the
planner's estimate — arena words x word width, the arena's RAM10 equivalent, MAC
lanes, MACs per frame, and both the unstalled generated cycle count and the
plan's cycle estimate — so a resource estimate is never mistaken for a
measurement, and a `semantic_notes` entry is shown rather than hidden.

### `simulate`

```sh
tessera simulate build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise
```

Emits the bundle (deterministically, byte-for-byte identical to `emit`) and runs
the generated self-checking testbench with Verilator against the Rust reference
model. When Verilator is not installed the command prints `simulation: SKIP` and
exits 0 — it never reports a pass it did not observe. Set `TESSERA_VERILATOR` to
point at a specific binary.

### `verify`

```sh
tessera verify build/w8
tessera verify build/w8 --format json
```

Re-checks every artifact present in a stage directory and every provenance link
between them:

* each artifact re-validates against its own schema;
* `mlir.json` descends from the graph artifact recorded beside it (format,
  schema and manifest hash);
* `optimized.json` records the Melior pipeline that produced it;
* `layout.tlayout` names the hashes of the staged input, target and policy
  copies — so editing `target.toml` after solving is detected;
* `plan.tplan` carries exactly the layout's provenance;
* every file in `rtl/` matches its recorded size and digest.

Missing stages are not failures: a user may stop after any stage. At least one
artifact must be present.

## Current scope

Supported today, end to end:

* `fpga_st_train.int8.v1` manifest import (i8-x1 and i4-x2 packing), including
  the reference operator set: 3x3 convolution, depthwise 3x3, pointwise 1x1,
  fused relu, residual `add`, nearest 2x upsample, unsigned output quantization.
* Deterministic MLIR emission for that subset; in-process optimization behind the
  `melior` feature; and lowering the *optimized module* (not the graph) into the
  planner's primitive graph, verified against the module's own analysis.
* Ti60-family target model and layout search with per-mode rejection reports, an
  explicit certainty classification, and exact activation code intervals that
  size the datapath (a residual sum is carried at nine bits, not clamped to
  int8).
* RTL emission and simulation for a **single pointwise step** (streaming
  interface, ROM weights) and for **whole models** through the full-graph
  backend: one static-schedule FSM over an activation arena with a single shared
  MAC array, `shared_tile`/frame plans, bias and weights resolved from the
  manifest.

Not supported yet (each fails with a named error rather than a wrong result):

* Hybrid architectures, DMA scheduling, ping-pong frame overlap, and plans whose
  steps want more than `tap_parallel == 1` or 64 MAC lanes on the full-graph
  engine.
* Asymmetric quantization, activation bit widths other than 8, and weight
  widths outside {4, 8}.
* Synthesis and board measurement.

## Examples

Everything below runs from the repository root with checked-in fixtures and no
external model files:

```sh
# The supported narrow path: solve, emit, verify, simulate.
tessera solve --primitive fixtures/pointwise/plan_input.json \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/streaming.toml --out build/pointwise
tessera emit build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise
tessera verify build/pointwise
tessera simulate build/pointwise/layout.tlayout \
    --weights fixtures/pointwise/weights --out build/pointwise

# The whole-model path: import, emit MLIR, solve, emit RTL and simulate.
tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
tessera emit-mlir build/w8/graph.tgraph --out build/w8
tessera solve --graph build/w8/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8
tessera inspect build/w8/layout.tlayout
tessera emit build/w8/layout.tlayout --graph build/w8/graph.tgraph --out build/w8
tessera simulate build/w8/layout.tlayout --graph build/w8/graph.tgraph --out build/w8

# The same chain through the optimized MLIR module instead of the graph.
tessera optimize build/w8/mlir.json --out build/w8            # needs the `melior` feature
tessera solve --mlir build/w8/optimized.json \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8-mlir
tessera inspect build/w8-mlir/primitives.json
```

`build/` and `out/` are ignored by git; `fixtures/` is committed and
regenerated deterministically by

```sh
cargo run -p tessera --example generate_fixtures
```

### Committed review snapshots

`fixtures/snapshots/{w8,ink_dw}/` are complete stage directories produced by the
commands above (graph, weights, emitted and optimized MLIR, primitive graph,
target and policy copies, layout, plan, the `rtl/` bundle and the per-step trace).
They are committed so the emitted structure can be inspected and re-verified
without an FPGA toolchain:

```sh
tessera verify fixtures/snapshots/w8
tessera inspect fixtures/snapshots/ink_dw/rtl/rtl_bundle.json
```

The generator does not touch them: it writes only the fixtures listed above. See
[artifacts.md](artifacts.md) for what each snapshot contains and the measured
numbers behind it.

### Real-model fixture

`fixtures/ink_dw/` is a **snapshot of a real trained model**, not generated: the
`tiny_dw` W8A8 manifest and weight blob from
`st_work/fst/runs/ink_dw/quant/w8a8` (manifest sha256
`e2daffcbb02c5729a6195b35b10bdb9b3fadbe3fd6109626c2d22992ff11dc6e`, weights
sha256 `caf1d7da0064e587d158f7e39fd7eb77f15a028cac0d0c489fd3e5ac77a5190b`). It
covers conv 3×3, depthwise 3×3 with stride 2, reflect padding, four residual
`add`s, two nearest upsamples and an unsigned head: 29 layers, 61 graph ops,
50 144 weight bytes, `i8-x1`.

The manifest carries no spatial extents, so the input resolution is chosen by the
caller. The reference flow in `nngen/examples/st_stream` runs this model at
`--res 640x480`, which its planner parses as `width x height`; in Tessera's
`N,H,W,C` order that is `--input-shape 1,480,640,3`. The exporter's reference size,
40×20, is `--input-shape 1,20,40,3`, which is what the acceptance tests
use for fast runs.

```sh
tessera import fixtures/ink_dw --input-shape 1,20,40,3 --out build/ink
tessera solve --graph build/ink/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/explore.toml --out build/ink
```

The whole model is planned with `fixtures/policies/emit.toml`, the policy shaped
to the emitted full-graph engine (shared-tile frame schedule, `tap_parallel == 1`,
at most 64 MAC lanes per convolution step). The resolution you pick is a budget
decision, not a modelling one: the activation arena is frame-sized, while the
50 144 weight bytes are resolution-independent, so the same 29-layer model can fit
at a smaller input and not at the reference 40x20. `solve` reports its result
with a certainty and per-mode rejections, and `emit` reports the arena's RAM10
equivalent next to the planner's estimate, so an infeasible resolution is an
explicit, readable outcome rather than a hidden rounding. See ADR-0003 and
[verification.md](verification.md).
