# Tessera hardware constraint language

This document defines the first user-facing shape of Tessera's hardware and planning constraints. It is intentionally declarative: users describe budgets, legal implementation choices, and optimization priorities; the planner chooses a layout. The language describes a target and a policy, not RTL.

## 1. File roles

Use separate files so the same optimized graph can be solved repeatedly:

```text
optimized.tgraph       model-derived, target-independent
 ti60.toml             target capacities and mapping rules
 policy.toml           user priorities and legal execution choices
 layout.tlayout        solver output, versioned and reproducible
```

A target file describes what the platform has. A policy file describes what the user wants to permit or optimize. Keeping them separate allows one graph to be retargeted without re-importing or re-optimizing it.

## 2. Example target description

The initial syntax is TOML because it is readable, diffable, and supported by the Rust ecosystem. The schema is versioned independently from the Rust structs.

```toml
format = "tessera.target"
schema_version = 1
name = "ti60f225"
revision = "board-2026-10"

[clock]
core_hz = 100_000_000

[budget]
dsp48 = 120
ram10 = 80
xlr = 40_000
ddr_read_bytes_per_s = 900_000_000
ddr_write_bytes_per_s = 700_000_000

[reserved]
dsp48 = 8
ram10 = 168
xlr = 20_232

[mac]
# Physical mapping rules, not generic arithmetic assumptions.
int8xint8_per_dsp48 = 2
int9xint8_per_dsp48 = 1
requant_lane_dsp48 = 1

[[ram.primitive]]
name = "ram10"
capacity_bits = 10_240
max_width_bits = 40
max_depth = 1024
read_ports = 1
write_ports = 1

[[stream.primitive]]
name = "efx_dsp48"
valid_modes = ["int8_mac_dual", "int9_mac_single"]
```

The target model MUST validate values and convert logical buffers to physical resources, including width/depth rounding, banking, ports, and duplication.

## 3. Example planning policy

```toml
format = "tessera.policy"
schema_version = 1

[objective]
primary = "minimize_cycles"
secondary = ["minimize_ddr_bytes", "minimize_ram10"]

[constraints]
max_frame_cycles = 10_000_000
require_bit_exact = true
allow_cpu_layer_dispatch = false

[search]
execution_modes = ["streaming", "shared_tile", "hybrid"]
granularities = ["row", "tile"]
tile_widths = [40, 80, 160]
tile_heights = [40, 60, 80]
max_candidates = 100_000
seed = 1

[fusion]
allow = ["depthwise_requant_pointwise", "short_residual"]
forbid = ["fanout_value", "cross_ddr_boundary"]

[memory]
placements = ["line_buffer", "bram", "ddr"]
allow_ping_pong = true
max_ping_pong_buffers = 2

[[operator_override]]
pattern = "depthwise_conv2d"
allowed_simd = [1, 2, 4, 8, 16, 32]
allowed_tap_parallel = [1, 3, 9]

[[operator_override]]
pattern = "pointwise_conv2d"
allowed_simd = [1, 2, 4, 8, 16, 32]
allowed_pe = [1, 2, 4, 8, 16]
```

An override restricts or prioritizes candidates. It MUST NOT bypass legality checks. Unknown operators, modes, placements, or target features MUST produce an error rather than being ignored.

## 4. Constraint semantics

Constraints fall into three classes:

### Hard constraints

A plan is invalid when it violates one:

```text
DSP use <= target available DSP
RAM use <= target available RAM
DDR traffic <= sustained target bandwidth
frame cycles <= policy max_frame_cycles
folding factors divide operator dimensions
required ports and stream rates are satisfiable
```

### Soft objectives

A valid plan may be ranked by:

```text
cycles
DDR bytes
RAM blocks
DSP blocks
logic estimate
number of schedule transitions
```

The objective order or weighted form MUST be recorded in the layout artifact.

### Hints

Hints affect candidate ordering but cannot make an invalid plan valid:

```toml
[hint]
prefer_fusion = true
prefer_bram_over_ddr = true
prefer_shared_engines = true
```

## 5. Solver output

A solver writes a layout artifact rather than directly writing RTL:

```text
layout.tlayout
  ├── selected execution mode
  ├── engine definitions and resource reservations
  ├── tile/row geometry and halo policy
  ├── operator-to-engine bindings
  ├── folding factors
  ├── buffer placement and lifetimes
  ├── DMA and ping-pong schedule
  ├── hard-constraint proof/report
  ├── objective score and rejected alternatives
  └── provenance hashes
```

Example summary:

```toml
format = "tessera.layout"
schema_version = 1
status = "feasible"

[provenance]
graph_sha256 = "..."
target_sha256 = "..."
policy_sha256 = "..."
compiler_version = "0.1.0"

[resources.estimate]
dsp48 = 112
ram10 = 76
frame_cycles = 9_866_272
ddr_read_bytes_per_frame = 123_456_789
ddr_write_bytes_per_frame = 98_765_432

[resources.slack]
dsp48 = 8
ram10 = 4
frame_cycles = 133_728

[[schedule]]
step = 0
op = "enc.0.dw"
engine = "shared_conv_0"
mode = "tile"
tile = [80, 60]
simd = 8
tap_parallel = 3
input_buffer = "ping"
output_buffer = "dw_scratch"

[[schedule]]
step = 1
op = "enc.0.pw"
engine = "shared_conv_0"
mode = "tile"
simd = 16
pe = 8
input_buffer = "dw_scratch"
output_buffer = "pong"
```

`status = "feasible"` means the target model accepted the plan. It does not claim that synthesis or place-and-route will succeed; measured results are attached later as separate artifacts.

## 6. Iteration workflow

Users can iterate without repeating earlier work:

```sh
# Import and optimize once.
tessera import model.json --out graph.tgraph
tessera optimize graph.tgraph --out optimized.tgraph

# Explore several hardware assumptions and policies.
tessera solve optimized.tgraph --target ti60.toml --policy fast.toml --out fast.tlayout
tessera solve optimized.tgraph --target ti60.toml --policy small.toml --out small.tlayout

tessera inspect fast.tlayout --resources --schedule

tessera inspect small.tlayout --resources --schedule

# Emit only a selected validated layout.
tessera emit fast.tlayout --backend shared-tile --out build/fast
```

A layout can be regenerated with a different backend only when that backend declares support for the layout features. Otherwise the compiler must report the missing capability.

## 7. Design rules

- Constraint files are data, not executable code.
- Target facts and user preferences are separate.
- Estimates are explicit and never presented as synthesis measurements.
- Every rejected candidate SHOULD have a reason category.
- The solver MUST be deterministic for a fixed seed and inputs.
- A layout is immutable once emitted; iteration creates another artifact.
- RTL emission consumes a validated layout and must not silently re-solve it.
- If RTL emission needs a different choice, it fails and sends the user back to planning.
