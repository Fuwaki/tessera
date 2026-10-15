# ADR-0003: Engine interfaces are streams; the v1 RTL envelope is one pointwise step

## Status

Superseded (RTL envelope)

The stream-interface buffer rule in Decision (1) still informs how layouts name
engine-facing buffers. Decision (2)–(3) described the retired `pw_dot` / layout
backends. On `mainline-hybrid` the emit path is CIRCT hybrid
(`tessera.hybrid.circt.v1`): solved Tessera dialect → CIRCT → ExportVerilog with
a single `*_sched_core` datapath; `tessera.engine` / bind are mapping decisions
reflected as `engines_busy` one-hot, not per-engine FSMs. See the concept book
§8 and `docs/cli.md`.

## Context

The layout artifact records, for every schedule step, an input buffer and an
output buffer, and every buffer carries a memory placement (`register`, `fifo`,
`line_buffer`, `bram`, `ddr`). The RTL backend for a streaming engine has
ready/valid ports and therefore requires the activations a step consumes and
produces to be on-chip streams.

The planner originally named a step's buffer after the graph port it touched, so
a step reading a graph input was bound directly to a `ddr` buffer, and a
step producing a graph output was bound directly to a `ddr` buffer. That is
accurate about where the *frame* lives, but wrong about the *engine*: a
ready/valid port is not a DDR interface, and a step bound directly to DDR would
require a DMA inside the generated kernel, which v1 does not emit. As a result
no planner-derived plan could ever be emitted, and the only way to reach the RTL
backend was to bypass the planner — which defeats the point of a persisted plan.

## Decision

1. **A step's interface buffers describe the engine, not the memory it talks
   to.** For dedicated and streaming execution modes, a step that reads a graph
   input (or writes a graph output) is bound to a `fifo` buffer named
   `stream.<op>.in` / `stream.<op>.out`. The external `ddr` port buffers
   (`in.<tensor>`, `out.<tensor>`) remain in the buffer table with their
   producers and consumers, because they are the frame endpoints the DMA and the
   DDR traffic model are about. Shared-tile and hybrid modes keep the previous
   binding, where the intermediate placement (BRAM or DDR) is the engine's own
   storage.

2. **The v1 RTL backend emits exactly one pointwise step and fails closed on
   everything else.** Supported: one step, `PointwiseConv2d` (or `Conv2d` with a
   1x1 kernel and `groups == 1`), `tap_parallel == 1`, no bias, signed int8
   activation/weight/output elements, weights resident in ROM, activation
   buffers in register or FIFO, one requantization contract with half-away-from-zero
   rounding inside the 17-bit multiplier field, and an activation clamp inside
   int8. The backend enumerates the unsupported constructs in its error, naming
   the step (`unsupported layout: v1 emits exactly one kernel, but the plan has 7
   steps [stem(Conv2d), ...]`).

3. **A rejected layout is returned to the user, never adjusted.** `emit` does
   not re-solve, does not drop steps, does not narrow a clamp, and does not
   substitute a different mode. The rejected-candidate summary and the selected
   plan stay in the artifact so the user can change the target, the policy, or
   the graph, and solve again.

## Consequences

### Positive

- A planner-produced layout is emittable: the CLI reaches the backend through
  the artifacts rather than around them, and the emitted kernel has no implicit
  DMA.
- The DDR model stays meaningful (traffic is computed from tensors), while the
  engine interface stays physically honest.
- Unsupported plans fail with a name and a reason instead of producing RTL that
  would not match the model.

### Negative

- The buffer table carries both the port buffer and the stream buffer for graph
  I/O, which is one more entry per boundary to read.
- `emit` refuses most of the plan space until the wider backends land, so the
  whole-model example stops at the layout stage.

## Alternatives considered

### Let the RTL backend treat a `ddr` step buffer as streamed

Rejected: it would make the backend accept a placement it cannot implement, and
the generated kernel would silently lack the DMA that placement implies.

### Keep the DDR binding and add a separate DMA stage

Rejected for v1: it moves the problem into a component that does not exist yet,
and the plan would still not describe the engine interface.

### Emit a multi-step kernel by inlining every step

Rejected: the pointwise-dot kernel implements one operator. Inlining depthwise,
add, upsample and unsigned quantization into it would be a different backend with
its own verification needs, not a relaxation of this one.

## Verification obligations

- `cargo test -p tessera --test pipeline` proves that every scheduled step of
  the whole-model layout resolves to a non-DDR interface buffer, that the
  whole-model plan is rejected naming its steps, and that the pointwise fixture
  emits and matches the Rust reference model under Verilator.
- `cargo test -p tessera --test cli` proves the same through the command line,
  including that RTL emission is byte-for-byte reproducible.
