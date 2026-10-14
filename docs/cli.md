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

Lowers the input (a graph artifact, or an already-lowered primitive graph) to
the planner's typed primitive graph, copies the target and policy into the stage
directory, runs the layout search, and writes `layout.tlayout` plus, when the
plan is feasible, `plan.tplan`.

* `--graph` lowers the manifest graph: convolution layers, `add`, nearest
  upsample and the unsigned output quantization. The manifest's per-channel
  requantization is serialized to fixed-point `(M0, shift)` and carried
  verbatim in the layout.
* `--primitive` solves a primitive graph directly, which is how the pointwise
  fixture reaches the narrow RTL envelope without a manifest.
* An **infeasible** layout is written, its rejection categories are printed, and
  the command exits non-zero: no candidate is ever chosen silently.

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

Emission fails closed. The backend of this build emits **one kernel**: a single
pointwise step with signed int8 activations, ROM-resident weights, streaming
activation buffers, one requantization contract, and no bias. Anything else is
reported by step name, for example:

```text
error: rtl: unsupported layout: v1 emits exactly one kernel, but the plan has 7
steps [stem(Conv2d), dw(DepthwiseConv2d), ...]; the supported envelope is a
single pointwise step
```

Note that with the present envelope `emit --graph` always fails, and for a
precise reason rather than an oversight: a valid manifest graph must end in an
unsigned output quantization (`quantize_u8_output`), so every manifest-derived
plan has at least two steps. `--graph` is the correct resolver interface for
graph plans and takes over as soon as a backend accepts multi-step streaming
plans; today the successful path is a single-step primitive plan plus
`--weights`. The rejection names the steps, so the user can see exactly what the
backend is missing.

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
  `melior` feature.
* Ti60-family target model and layout search over execution mode, granularity,
  folding, fusion and memory placement, with rejected-candidate reporting.
* RTL emission and simulation for a **single pointwise step** with signed int8
  activations, ROM weights and streamed activation interfaces.

Not supported yet (each fails with a named error rather than a wrong result):

* RTL for multi-step plans, depthwise or non-1x1 kernels, `add`, standalone
  requant/activation, upsampling, unsigned output, bias terms, shared-tile or
  hybrid engines, and non-stream activation buffers. This is why the example W8
  model solves to a valid layout but is rejected at `emit`: its plan has seven
  steps.
* Asymmetric quantization, activation bit widths other than 8, and weight
  widths outside {4, 8}.
* DMA scheduling, synthesis and board measurement.

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

# The whole-model path: import, emit MLIR, solve, and observe the clear RTL rejection.
tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
tessera emit-mlir build/w8/graph.tgraph --out build/w8
tessera solve --graph build/w8/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/explore.toml --out build/w8
tessera inspect build/w8/layout.tlayout
tessera emit build/w8/layout.tlayout --graph build/w8/graph.tgraph --out build/w8
```

`build/` and `out/` are ignored by git; `fixtures/` is committed and
regenerated deterministically by

```sh
cargo run -p tessera --example generate_fixtures
```
