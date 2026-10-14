# Verification and acceptance

This document defines how Tessera's full-graph pipeline is accepted, what the
acceptance fixtures are, and how every expected value in them was derived. It is
the contract the acceptance suite enforces:

| Artifact | Checked by | Commands |
| --- | --- | --- |
| `fixtures/acceptance/**` | `cargo test -p tessera --test acceptance_vectors` | §6.1 |
| manifest → graph → lowered plan | `cargo test -p tessera --test acceptance_pipeline` | §6.2 |
| CLI stages, empty-DSP target, impossible budget, stale provenance | `cargo test -p tessera --test acceptance_pipeline`, `--test acceptance_cli` | §6.2, §6.3 |
| emitted RTL vs the independent reference | `cargo test -p tessera --test acceptance_cli` (or `scripts/acceptance_full_graph.sh`) | §5, §6.3 |
| the whole gate, driven like a user | `scripts/acceptance_full_graph.sh` | §6.4 |

**Status of this document.** This document *defines* the gate; it records no
result. The gate — `scripts/acceptance_full_graph.sh --strict` plus
`cargo test` on the three acceptance targets — is executed once, at the final
integration wave, over the whole tree. No acceptance result, pass or failure, is
claimed anywhere in this document.

## 1. What "verified" means here

An acceptance value is only worth as much as its independence. In this suite:

* **Expected codes come from `manifest.json` + `weights.bin` + the input tensor,
  never from the compiler and never from the emitted RTL.** The scalar reference
  in `crates/tessera/tests/acceptance_support/mod.rs` reads the manifest with its
  own schema, decodes the weight blob itself, and evaluates the graph itself.
* **The frozen vectors in `expected.json` are re-derived on every run.** They are
  a drift detector (an edited manifest, weight blob or vector file fails the
  test), not an oracle that the reference trusts.
* **The fixed-point parameters are re-derived, then cross-checked.** `M0`, the
  shift and the clamp for every operator are computed from the manifest's `f32`
  requantization vectors by this suite, and `acceptance_pipeline.rs` then asserts
  that lowering produced exactly those numbers. Agreement is therefore evidence
  about the compiler, not a restatement of it.
* **The RTL is judged by the suite's own stimulus.** The suite writes
  `stim_input.hex` itself, runs the compiled simulation with `+NOCHECK`, reads the
  trace the testbench dumps, and compares it against the reference. The generated
  testbench's built-in self-check is *not* the acceptance criterion: it compares
  the RTL against the compiler's own Rust reference, which is not an independent
  observation.
* **Nothing unavailable is presented as verified.** A gate that cannot run (no
  simulator, a plan outside the emitted envelope, the MLIR planning stage
  unavailable, no real model to import) reports `SKIP`/`UNCOV` with its reason —
  and in the acceptance configuration it is a **failure**: `scripts/acceptance_full_graph.sh
  --strict` exports `TESSERA_ACCEPTANCE_STRICT=1`, which makes the acceptance
  tests panic instead of skipping, and treats an uncovered gate, a refused
  full-graph emission, an unavailable `optimize`, or missing simulated
  residual/unsigned/bias coverage as a failure. The narrow pointwise envelope is
  a *regression* path only: its absence is reported loudly but does not gate the
  full-graph acceptance.

What is *not* in scope: synthesis results, timing closure, absolute cycle counts
on hardware, and model accuracy. The suite checks the integer contract the
hardware model is defined by, not the network's usefulness.

## 2. The integer datapath the fixtures encode

The scalar reference implements the datapath the emitted RTL is defined by, not
the floating-point reference in `fpga_st_train/src/intref.rs`:

```text
acc        = bias_int32[oc] + sum over taps of (activation code * weight code)   // int32
product    = acc * M0[oc]                                                       // signed
offset     = product >= 0 ? 2^(shift-1) : 2^(shift-1) - 1                       // shift == 0 -> 0
code       = clamp((product + offset) >> shift, lo, hi)                         // arithmetic >>
```

with, per operator:

```text
max_abs    = max over channels of |requant[ch]|
shift      = clamp(floor(log2((2^17 - 1) / max_abs)), 0, 31)
M0[ch]     = round_half_away(requant[ch] * 2^shift)
lo, hi     = [0, output_qmax]   for an unsigned head
             [0, act_qmax]      for a relu layer
             [-act_qmax, act_qmax] otherwise
```

and, per operator family:

* **convolution** (`conv2d`, `depthwise_conv2d`, `1x1`): reflection padding first
  (the edge is *not* repeated), then a valid convolution, stride 1 in every
  fixture. Weight order in the blob is `(out_channel, channel_per_group, kh, kw)`;
  a depthwise layer has one weight channel per output channel.
* **residual `add`**: `sum = clamp(requant(branch * factor)) + trunk`, with
  `factor = point_scales[source] / point_scales[destination]` and the branch
  clamped to `[-act_qmax, act_qmax]`. **The sum is not clamped**: the result is a
  9-bit value in `[-254, 254]` and the following operators must read it at that
  width. The branch requantization `M0`/shift is serialized from `factor` by the
  same rule as a layer.
* **nearest 2x upsample**: every code is duplicated into a 2x2 block; the code
  interval is unchanged.
* **unsigned output clamp**: `quantize_u8_output` clamps to `[0, output_qmax]`
  and feeds the image; the image value is `code * quantize_scale`.

### 2.1 Documented divergences (accepted, not silently passed)

| Divergence | Where | How the suite treats it |
| --- | --- | --- |
| The fixed-point path is not bit-identical to `intref.rs`'s `acc * requant` in `f32`. | `mlir/quant.rs` records the worst per-channel relative error; lowering stores it as `Requant::multiplier_error_ppm`. | The suite asserts the **integer** path (that is the hardware contract) and checks that an error bound is recorded. It never claims the two paths agree code for code. |
| `mlir/emit.rs::write_add` truncates the residual sum to `i8` with `arith.trunci`, while the oracle carries the 9-bit value. | MLIR artifact only; the RTL backend implements the carry and records it in `semantic_notes`. | The suite asserts the carry at every level it can observe (plan tensor element type, layout `out_code_range`, RTL trace labelled `i16`). The MLIR wrap is reported as a divergence, never counted as a pass. |

## 3. Acceptance fixtures

`fixtures/acceptance/` is hand-authored and deliberately *not* produced by
`examples/generate_fixtures.rs`. Each directory is self-contained: it holds
`manifest.json`, the weight blob, the input, and the frozen expected vectors.

### 3.1 `fixtures/acceptance/rounding` — exact rounding and saturation

Two 1x1 operators on a 2x2x4 input, with every number small enough to check by
hand. `rnd` is an identity convolution (`weight = 1` on `oc == ic`, 0 elsewhere)
with `bias_int32 = [0, 0, 2000, -2000]`, and its per-channel requantization is
exactly `[0.25, 0.125, 0.25, 0.25]`, so `shift = 18` and
`M0 = [65536, 32768, 65536, 65536]`. The layer is signed (`[-127, 127]`).

Frame 0 drives the input positions `[-2,4,0,0]`, `[2,4,0,0]`, `[0,4,0,0]`,
`[-2,2,0,0]`; `head` selects channels 1..3 with a requantization of exactly
`1.0` (`shift = 16`, `M0 = 65536`), so the unsigned clamp is a passthrough probe.

| Input code | Channel | `acc` | `acc * M0 >> shift` | `rnd` code | Why |
| --- | --- | --- | --- | --- | --- |
| `-2` | 0 | -2 | -0.5 | **-1** | half away from zero at 0.25, negative |
| `4` | 1 | 4 | 0.5 | **1** | half away from zero at 0.125, positive |
| `2000` (bias) | 2 | 2000 | 500 | **127** | saturates at the signed clamp |
| `-2000` (bias) | 3 | -2000 | -500 | **-127** | saturates at the signed clamp |
| `2` | 0 | 2 | 0.5 | **1** | frame 1, half away from zero again |
| `-4` | 1 | -4 | -0.5 | **-1** | frame 1, negative half away from zero |
| `-1` | 0 | -1 | -0.25 | **0** | rounds toward zero |
| `4` | 1 | 4 | 0.5 | **1** | quarter- and half-LSB boundaries together |

`head` then produces, for frame 0,
`[[1,1,1,0], [127,127,127,127], [0,0,0,0]]` in `(c, h, w)` order: channel 0 is
`rnd` channel 1 unchanged, channel 1 is `rnd` channel 2, channel 2 is `rnd`
channel 3 clamped to zero by the unsigned clamp. Frame 1 exercises the other
signs. `acceptance_vectors.rs` asserts that table cell by cell.

### 3.2 `fixtures/acceptance/graph` — the whole pipeline

A 5x5x3 input through the same operator mix as the reference model, with exact
power-of-two point scales so that every requantization factor is exact in `f32`:

| Layer | Op | Kernel | Pad | Groups | in → out | in/out scale | weight scale | requant | `M0` | shift | clamp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stem` | `conv2d` (relu) | 3x3 | reflect 1 | 1 | 0 → 1 | 0.0078125 → 0.015625 | `[0.02, 0.04, 0.06, 0.08]` | `[0.01, 0.02, 0.03, 0.04]` | `[20972, 41943, 62915, 83886]` | 21 | `[0, 127]` |
| `dw` | `depthwise_conv2d` (relu) | 3x3 | reflect 1 | 4 | 1 → 2 | 0.015625 → 0.03125 | `[0.1, 0.2, 0.3, 0.4]` | `[0.05, 0.1, 0.15, 0.2]` | `[26214, 52429, 78643, 104858]` | 19 | `[0, 127]` |
| `mid` | `conv2d` (relu) | 1x1 | — | 1 | 2 → 3 | 0.03125 → 0.015625 | `[0.006, 0.012, 0.018, 0.024]` | `[0.012, 0.024, 0.036, 0.048]` | `[25166, 50332, 75497, 100663]` | 21 | `[0, 127]` |
| `add` | residual | — | — | — | 3 ← 1 | factor 1.0 | — | `[1.0]` | `[65536]` | 16 | branch `[-127, 127]`, sum unclamped |
| `head` | `conv2d` (unsigned) | 1x1 | — | 1 | 3 → image | 0.015625 → 1/255 | `[0.08, 0.1, 0.06]` | `[0.31875, 0.39844, 0.23906]` | `[83558, 104448, 62669]` | 18 | `[0, 255]` |

`bias_int32` is `[20000, -10, 20, 100]` for `stem` (non-zero, and channel 0
saturates the relu clamp), all zero for `dw` (the identity is dropped with a
note), `[0, 0, 50, -20]` for `mid`, and zero for `head`.

Two manifest conventions these fixtures must keep, because the frontend enforces
them: a layer with `out_relu: true` uses the `conv2d+relu` /
`depthwise_conv2d+relu` operator vocabulary (a plain `conv2d` with
`out_relu: true` is rejected), and a depthwise layer declares `groups` equal to
its input and output channel counts. The weight blob is contiguous in layer
order, one `i8` code per byte (`i8-x1`), and its length equals `weight_bytes`.

Two frames are driven. Coverage the suite asserts on the recomputed values:

* **reflection padding is real**: recomputing `stem` with zero padding yields a
  different tensor, so the border is actually exercised;
* **9-bit residual**: the `add` output leaves `[-128, 127]` in both frames
  (observed ranges `[15, 189]` and `[2, 248]`, inside the static `[-127, 254]`),
  and the trace must label that step `i16`;
* **nearest upsample**: every output code equals the input code at `(y/2, x/2)`,
  and the interval is preserved;
* **unsigned head**: clamp hits both ends across the frames (`0` and `255` are
  both present);
* **relu clamp**: `stem` produces both `0` and `127`;
* **heterogeneous per-channel multipliers**: every layer's requantization vector
  has at least three distinct values, so ignoring the channel index fails;
* **input quantization**: the codes equal
  `clamp(round_half_away(pixel / 255 / point_scales[0]), -127, 127)`, checked
  against the source image stored in `input.json`.

### 3.3 Policies and target

| File | Purpose | Deterministic expectation |
| --- | --- | --- |
| `policy.toml` | The acceptance search: the explore policy's space with fusion disabled, so the schedule has exactly one step per operator. | The acceptance graph is `feasible`, with 7 steps. |
| `policy_streaming.toml` | Streaming/dedicated, frame granularity — the shape the narrow RTL envelope is defined for. | Feasible if the search space allows it; the RTL run tries this policy first. |
| `policy_impossible.toml` | The same search space as `policy.toml` with `max_frame_cycles = 1`. | `Infeasible`; one cycle buys 9 bytes of reads at 900 MB/s and 100 MHz while the graph input alone is 75 bytes, so `ddr_read_budget` must appear among the rejection categories, and the suite asserts it does. The CLI exits non-zero and still writes `layout.tlayout`, whose `status` must be `infeasible`. The suite does **not** pin the whole category set: the planner's rejection vocabulary is extended as architecture models land, and only structural invariants (non-empty counts, each count positive, `sum(counts) == rejected_candidates`, `validate()` passes) are required. |
| `target_impossible.toml` | `reserved.dsp48 == budget.dsp48`. | `Infeasible` (not a reservation error), with `dsp_budget` among the rejection categories. |

### 3.4 The w8 execution oracle (MLIR leg)

`fixtures/acceptance/w8_execution_oracle.json` holds, for the exact stimulus that
`tests/mlir_lowering_execution.rs` uses, the per-element expected codes of every
tensor of the shipped `fixtures/w8` in NHWC order (the order the emitted module
and the execution driver use), with per-tensor fingerprints (sum, min, max, codes
outside `i8`) and, for the residual, the codes an `i8`-narrowed residual would
produce. It is derived by the suite's independent reference from `manifest.json`
plus `weights.bin` — never from an emitted artifact.

`acceptance_pipeline.rs::w8_execution_oracle_is_reproducible_from_the_fixture`
rebuilds the whole document from `manifest.json` plus `weights.bin` on every run
and fails when the checked-in copy differs, so the oracle cannot drift silently:
a deliberate fixture change (for example re-derived head weights) fails with the
one command that refreshes it,

```sh
TESSERA_ACCEPTANCE_REGEN=1 cargo test -p tessera --test acceptance_pipeline \
    w8_execution_oracle -- --nocapture
```

The same test asserts, independently of the file, that the tensors a comparison
should use are non-zero and carry codes outside `i8`, and that the `i8`-wrapped
residual differs from the carried one (so that transport rule is a real
discriminator, not an inert one). The oracle is therefore also a *per-element*
comparison target for the MLIR execution leg instead of one aggregate number.

*Why this exists:* a per-tensor oracle is what makes a divergence in the
execution leg attributable in one step. A mismatch that appears only in the head's
output while `add` and `upsample` match element for element localizes the defect
to the head, not to the depthwise filter layout or the residual transport; an
accumulator that is read before it is written appears exactly that way, and it is
also non-deterministic between runs, which is why the leg must be compared per
tensor and run more than once rather than judged by one aggregate number.

Two facts the file records, because they decide how w8 may be judged:

* **Two of w8's three head rows have no positive coefficient**
  (`[-108,-101,-94,-87]` and `[-80,-73,-66,-59]` over an activation that is never
  negative). Those channels are identically zero for *every* stimulus, and the
  third is zero for the stimulus the execution test uses, so the image is a
  constant. The suite consequently never judges w8 by its image: it compares the
  residual and upsampled tensors, which are non-zero and contain 27 and 108 codes
  outside `i8`. Once the fixture's head is fixed, the test automatically requires
  a varying image instead of accepting a flat one.
* A fixture whose output is constant, or reaches only its clamp ends, is
  rejected by
  `per_tensor_fingerprints_are_stable_and_the_image_is_not_degenerate`: an
  output-only comparison would otherwise pass while proving nothing.

### 3.5 The real model

`TESSERA_INK_DW_DIR`, or `fixtures/ink_dw/` when that snapshot exists, points at a
real exported `fpga_st_train.int8.v1` directory (`manifest.json` + its weight
blob). The suite keeps that model **outside** the crate: no large blob is
embedded, no machine path is hard-coded, and an absent model reports a skip
naming the path it looked in. Import and lowering are checked at
`--input-shape 1,20,40,3` (the exporter's reference size), which
exercises real depthwise layers, real residual adds, a nearest upsample and an
unsigned head.

*Not covered for the real model*: RTL emission. It is a 29-layer graph; the
acceptance gate for it is the full-graph backend, which the acceptance graph
already gates.

## 4. Stimulus and trace contract

The RTL comparison relies on two files the emitted design owns:

* `stim_input.hex` — all frames concatenated, frame-major, one two-hex-digit
  byte per line (two's complement for signed codes), `(c, h, w)` order inside a
  frame. The suite writes its own, so the stimulus is never the compiler's seed.
* `sim_outputs.txt` — `format tessera.sim_trace.v1`, with, per frame,
  `frame <n> input <c> <h> <w>`, one `frame <n> step <i> <op> <elem> <c> <h> <w>`
  block per emitted schedule step, `frame <n> output <c> <h> <w>`,
  `frame <n> cycles <n>` and `frame <n> stalls <in> <out>`. Values are plain
  signed decimals, one per line, `(c, h, w)` order. `elem` is `i8`, `i16` when the
  step's stored codes leave the signed 8-bit domain, and `u8` for an unsigned
  clamp.

The suite invokes the compiled simulation binary directly with `+STIM=<path>`,
`+TRACE=<path>` and `+NOCHECK`, then parses the trace itself
(`acceptance_support::parse_trace`). If the format tag is not the documented one,
the parse fails rather than guessing. The suite writes its stimulus and trace
outside `sim/` (into the stage directory), so a re-emit cannot clobber them; if
the requested trace is absent it also looks in the testbench's default location
(`<stage>/sim/sim_outputs.txt`).

Two consequences worth stating plainly:

* **The generated testbench's built-in self-check is never acceptance evidence.**
  It compares the RTL against the compiler's own Rust reference. If the emitted
  testbench does not implement the plusargs and the trace, the gate reports an
  uncovered gate (fatal under `--strict`) naming the missing trace, and the
  suite says so explicitly in the reason it prints.
* **A trace produced from the wrong stimulus cannot pass.** Every comparison
  first asserts the DUT read back exactly the suite's input codes, so a
  testbench that ignores `+STIM` fails at the input block instead of silently
  comparing someone else's vectors.

## 5. What the RTL run requires from the pipeline

For each of the acceptance and pointwise fixtures the RTL test:

1. runs `solve` (trying `policy_streaming.toml` first, then `policy.toml`);
2. runs `emit`; a refusal is accepted **only** if it is a fail-closed refusal of
   the documented form (a plan shape the backend does not implement, or a
   placement it does not emit), and the run then reports `UNCOV` (uncovered),
   never a pass. Any other emit failure is a test failure;
3. runs `simulate` to produce a compiled binary, and then runs that binary with
   the suite's own stimulus and trace;
4. asserts the DUT read back exactly the suite's input codes;
5. asserts one trace step per operator, in plan order, matched through the
   `step_match` prefixes recorded in `expected.json` (so an operator naming
   change cannot silently weaken the comparison);
6. compares every code of every step, and the image, against the reference;
7. asserts the residual step is labelled `i16` and contains a value outside the
   signed 8-bit domain (in **every** frame);
8. asserts the simulated unsigned clamp reaches both ends (`0` and `255` across
   the frames), so the unsigned path is exercised, not merely present;
9. asserts the simulated `stem` equals the *biased* reference while the same
   fixture with `bias_int32` zeroed provably differs, so the non-zero int32 bias
   demonstrably reached the datapath;
10. reports a refusal as `UNCOV` (fatal in strict mode) and never as a pass.

## 6. Acceptance command list

### 6.1 Vectors

```sh
cargo test -p tessera --test acceptance_vectors
```

Re-derives both fixtures and compares them with `expected.json`, including the
hand table of §3.1, the coverage list of §3.2, and a control that a single
perturbed code is detected.

### 6.2 Pipeline

```sh
cargo test -p tessera --test acceptance_pipeline
```

Imports and lowers the acceptance graph, checks every derived `M0`/shift/clamp
and tensor shape against lowering, asserts the residual is modelled wider than
i8, solves the graph (feasible, one step per operator, non-DDR stream buffers,
residual code interval declared and containing every observed code), forces
`Infeasible` under the impossible budget and the empty-DSP target, rejects a
stale solved-from hash, and (when a real export is present) imports and lowers it.

### 6.3 CLI and RTL

```sh
cargo test -p tessera --test acceptance_cli -- --nocapture
```

Runs the CLI (feasible layout, verify, infeasible budget, empty DSP, stale
provenance) and, when the bundle can be simulated, compares the trace against the
reference. `--nocapture` prints the skip reasons.

### 6.4 The whole gate

```sh
scripts/acceptance_full_graph.sh           # development: notes uncovered gates
scripts/acceptance_full_graph.sh --strict  # THE acceptance run
```

`--strict` is the acceptance configuration and must be used at the final wave. It

* exports `TESSERA_ACCEPTANCE_STRICT=1`, so the acceptance tests fail (panic)
  rather than skip when a required gate is unavailable — full-graph `emit` being
  refused, no simulation evidence, and missing simulated residual/unsigned/bias
  coverage are failures, not notes;
* requires the MLIR planning stage (§6.5);
* turns every remaining `UNCOV` into a non-zero exit.

Without `--strict` the script still runs everything, prints each uncovered gate
with its reason, and exits 0 only if nothing failed, saying loudly that the run
was not a full acceptance. Either way it exits 1 on any `FAIL`.

### 6.5 The MLIR planning stage

```sh
tessera emit-mlir build/acceptance/graph.tgraph --out build/acceptance
tessera optimize build/acceptance/mlir.json --out build/acceptance
```

`emit-mlir` is pure Rust and always available. `optimize` needs the pinned Melior
pipeline (`--features melior` against LLVM/MLIR 23.x, see
[developer-setup.md](developer-setup.md)); under `--strict` the gate requires it
and the acceptance run must therefore be made with that build. When it succeeds
the script requires `optimized.json` to exist: a stage that reports success
without writing its artifact is a failure.

The planning stage is verified by **re-derivation**, not by its own claims.
`tessera verify` recomputes the stored module's hash and re-derives the structured
analysis — operations, `loc` tags, loop nests, fusion decisions, buffers — from
the saved text with Melior, and it refuses a `primitives.json` until the optimized
module is re-lowered and the plan, activation intervals, fusion groups, buffers
and lowering notes are compared field by field. A rewritten analysis that keeps
the recorded `module_hash`, or an edited plan that keeps its provenance, is
therefore caught. A build without the `melior` feature cannot re-derive the
analysis at all, so it reports that check as unverifiable and fails rather than
passing on metadata it never confirmed.

### 6.6 Manual equivalent, stage by stage

```sh
tessera import fixtures/acceptance/graph --input-shape 1,5,5,3 --out build/acceptance
tessera solve --graph build/acceptance/graph.tgraph \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/acceptance/policy.toml --out build/acceptance
tessera verify build/acceptance
tessera emit build/acceptance/layout.tlayout --graph build/acceptance/graph.tgraph \
    --out build/acceptance
tessera simulate build/acceptance/layout.tlayout --graph build/acceptance/graph.tgraph \
    --out build/acceptance

# Impossible budgets, stale provenance:
# (a solve into a *different* stage directory needs the input shape again: the
# recorded one is looked up in the stage directory the command writes to.)
tessera solve --graph build/acceptance/graph.tgraph --input-shape 1,5,5,3 \
    --target fixtures/targets/ti60f225.toml \
    --policy fixtures/acceptance/policy_impossible.toml --out build/impossible
tessera solve --graph build/acceptance/graph.tgraph --input-shape 1,5,5,3 \
    --target fixtures/acceptance/target_impossible.toml \
    --policy fixtures/acceptance/policy.toml --out build/no-dsp
printf '\n' >> build/acceptance/graph.tgraph && tessera verify build/acceptance
```

Driving the emitted binary with the suite's own vectors (the flow
`acceptance_cli.rs` automates):

```sh
tessera emit build/acceptance/layout.tlayout --graph build/acceptance/graph.tgraph \
    --out build/acceptance
tessera simulate build/acceptance/layout.tlayout --graph build/acceptance/graph.tgraph \
    --out build/acceptance       # compiles the binary and runs it once
cd build/acceptance/sim
./obj_dir/tessera_acceptance_graph_tb_sim \
    +STIM=/abs/path/stim_input.hex +TRACE=/abs/path/sim_outputs.txt +NOCHECK
```

`stim_input.hex` must contain every frame concatenated, frame-major, one
two-hex-digit byte per code, `(c, h, w)` order; the expected values are the
`expected.json` vectors of the same fixture.

## 7. Regenerating the frozen vectors

`expected.json` is a frozen second opinion. It is reproduced from the fixture's
own manifest, weight blob and input by `acceptance_vectors.rs` on every run; if
the manifest or blob changes, that test fails until the vectors are regenerated
deliberately. When regenerating, the derivation must remain §2 and §3 — no value
may be taken from an artifact produced by the compiler under test.

## 8. Known gaps at this revision

These are the gaps as they stand in the revision that introduced this suite.
They are deliberately listed so that the acceptance run can be read against a
known baseline: **update this section when the listed fixes land**, rather than
leaving a stale claim in place.

* The full-graph RTL gate (emit -> simulate -> trace comparison) is only
  satisfied once a backend can emit the seven-step plan. Until then the
  acceptance run fails it loudly under `--strict`, so the gap can never be
  mistaken for a pass. The RTL bundle's refusal, not the suite, defines when it
  becomes coverable.
* `fixtures/w8`'s image is uninformative (two head rows have no positive
  coefficient), so image-level claims about that fixture are not evidence. The
  fixture owner should re-derive its head weights; the suite will then require a
  varying image in place of a flat one.
* The narrow pointwise envelope is a regression path only. As of this revision
  the planner selects a BRAM activation placement for the shipped pointwise plan
  and the backend only emits register/FIFO streamed activations, so that path is
  refused; `tests/cli.rs` and `tests/pipeline.rs` assert it, `acceptance_cli.rs`
  reports it as a regression.
* Fusion (`depthwise_requant_pointwise`, `short_residual`) is disabled in the
  acceptance policies, so a fused schedule is not covered by the trace
  comparison.
* Only `i8` activations and `i8-x1` weight packing are covered; `i4-x2` (w4)
  needs its own vectors.
* Batch is 1, strides are 1, and the fixtures use power-of-two point scales; a
  non-dyadic scale is covered only through the manifest of the real model.
* Placements other than the streaming/FIFO interface (line buffer, BRAM, shared
  tile, hybrid) are not compared against the reference: they are not emittable
  in the v1 envelope.
* The suite verifies the integer contract and the emitted RTL's behaviour under
  Verilator. It says nothing about synthesis, timing or board-level results.
