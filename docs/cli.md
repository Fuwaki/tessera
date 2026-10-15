# Tessera CLI

Tessera is a persistent, staged compiler: every command consumes a versioned
artifact and writes a versioned artifact into a **stage directory**. Any stage
can be inspected, compared, or rerun without repeating earlier work, and a
failed later stage never invalidates an earlier artifact (ADR-0002).

```text
manifest.json --import--> graph.tgraph --emit-mlir--> mlir.json / module.mlir
              --solve --tessera--> tessera.mlir + solved.mlir (+ layout)
              --emit --tessera--> rtl/   (CIRCT ExportVerilog, *_sched_core)
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
| `tessera.mlir` | `solve --tessera` | Tessera dialect module fed to CP-SAT |
| `solved.mlir` | `solve --tessera` | Dialect with `tessera.engine` / `bind` / `buffer` / `segment` / `schedule` |
| `layout.tlayout` | `solve` | `tessera.layout` artifact (TOML) beside the dialect solve |
| `rtl/` | `emit --tessera` | CIRCT hybrid RTL bundle (SystemVerilog, testbench, `rtl_bundle.json`) |
| `sim/` | Verilator work dir | Simulation logs when a TB is run |

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
tessera solve --tessera build/w8/tessera.mlir \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8
```

Reads a Tessera dialect module (or annotated tensor-level linalg, converted in
process). CP-SAT jointly decides engine count, operator binding, fold, schedule,
pipeline cuts and arena placement, then writes those decisions back into the
**same** Melior module as `tessera.engine` / `bind` / `buffer` / `segment` /
`schedule`. The stage directory receives `tessera.mlir`, `solved.mlir`, copies
of the target and policy, and a `layout.tlayout` summary.

`tessera.engine` / bind are mapping decisions for later RTL reflection
(`engines_busy` one-hot); they are not a promise of per-engine FSMs in the
emitted DUT.


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
tessera emit --tessera build/w8/solved.mlir \
    --graph build/w8/graph.tgraph --out build/w8
```

CIRCT hybrid path only (`tessera.hybrid.circt.v1`):

1. Parse the solved Tessera dialect in Melior.
2. Lower mapping + ops into CIRCT `hw` / `comb` / `sv` in the same context.
3. Run in-process ExportVerilog into `rtl/`.

The DUT datapath is a single integer `*_sched_core` (ROMs, i16×i8 MAC, requant,
add/upsample). Solved engines/binds appear as `engines_busy` (one-hot of the
engine that owns the core's current op) — not as separate per-engine modules.

`--graph <graph.tgraph>` resolves weight and requant bytes from the manifest
blob so ROMs and the self-checking testbench are content-filled. Without
`--graph`, structure can still emit but content-filled / bit-exact emission
requires resolved ops from the graph.

Layout-only `emit <layout.tlayout>` is retired on this branch and fails with a
pointer to `--tessera`.


### `simulate`

```sh
# Prefer running the emitted self-checking TB after a CIRCT emit, e.g.:
#   sh build/w8/rtl/run_verilator.sh
# or Verilator against rtl/<module>_tb.sv with the hex stimuli beside it.
```

The CIRCT bundle's testbench compares DUT outputs to vectors from the integer
evaluator (`(acc*M0 + round) >>> shift`). When Verilator is not installed the
helper prints a skip rather than a false pass. Set `TESSERA_VERILATOR` to point
at a specific binary.

The legacy `tessera simulate <layout.tlayout>` entry still exists in the CLI
but routes through the retired layout backends and will fail closed with the
same CIRCT-path guidance as layout-only `emit`.


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

Supported today, end to end on the CIRCT hybrid path:

* `fpga_st_train.int8.v1` manifest import (i8-x1 and i4-x2 packing), including
  the reference operator set: 3x3 convolution, depthwise 3x3, pointwise 1x1,
  fused relu, residual `add`, nearest 2x upsample, unsigned output quantization.
* Deterministic MLIR emission for that subset; in-process optimization behind the
  `melior` feature.
* Tessera dialect + in-process CP-SAT mapping solve writing engine / bind /
  buffer / segment / schedule decisions into `solved.mlir`.
* RTL emission via CIRCT ExportVerilog with a single `*_sched_core` datapath;
  `engines_busy` one-hot reflects bind decisions; Verilator bit-exact check
  against the integer contract when `--graph` fills content.

Not supported yet (each fails with a named error rather than a wrong result):

* Per-engine microarchitecture FSMs (engine/bind remain mapping + busy one-hot).
* DMA scheduling, ping-pong frame overlap.
* Asymmetric quantization, activation bit widths other than 8, and weight
  widths outside {4, 8}.
* Synthesis and board measurement.


## Examples

Everything below runs from the repository root with checked-in fixtures and no
external model files. CIRCT must be available for `emit --tessera`
(see [developer-setup.md](developer-setup.md)):

```sh
# Import and emit tensor MLIR.
tessera import fixtures/w8 --input-shape 1,6,6,3 --out build/w8
tessera emit-mlir build/w8/graph.tgraph --out build/w8

# Solve on a tessera dialect module (annotated linalg is converted in process).
tessera solve --tessera build/w8/tessera.mlir \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/policies/emit.toml --out build/w8

# CIRCT hybrid emit with content-filled ROMs + bit-exact TB.
tessera emit --tessera build/w8/solved.mlir \
    --graph build/w8/graph.tgraph --out build/w8

tessera verify build/w8
tessera inspect build/w8/rtl/rtl_bundle.json
```
