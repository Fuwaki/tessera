// Tessera dialect.
//
// Two layers live in one module:
//  * operator ops (`conv`, `add`, `upsample`, `relu`, `requant`, `quantize`)
//    produced from annotated tensor-level linalg; tensor edges are SSA values
//    and weights/bias/multipliers are function arguments (weight refs);
//  * decision ops (`engine`, `bind`, `buffer`, `segment`, `schedule`) written
//    by the CP-SAT solver next to the operator ops.
// Every operand/attribute slot gets its own constraint value: IRDL unifies
// slots that share one SSA constraint.
// Cross-op legality is checked by the Rust verifier in `src/dialect/verify.rs`.
module {
  irdl.dialect @tessera {
    // conv2d / pointwise_conv2d / depthwise_conv2d fused with bias, requant
    // (multiply by per-channel M0, round, shift) and the activation clamp.
    irdl.operation @conv {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      %c3 = irdl.base "!builtin.tensor"
      %c4 = irdl.base "!builtin.tensor"
      %c5 = irdl.base "!builtin.tensor"
      %c6 = irdl.base "#builtin.string"
      %c7 = irdl.base "#builtin.string"
      %c8 = irdl.base "#builtin.integer"
      %c9 = irdl.base "#builtin.dense_array"
      %c10 = irdl.base "#builtin.dense_array"
      %c11 = irdl.base "#builtin.integer"
      %c12 = irdl.base "#builtin.integer"
      %c13 = irdl.base "#builtin.integer"
      %c14 = irdl.base "#builtin.integer"
      %c15 = irdl.base "#builtin.dense_array"
      %c16 = irdl.base "#builtin.dense_array"
      irdl.operands(input: %c1, weight: %c2, bias: %c3, multiplier: %c4)
      irdl.results(output: %c5)
      irdl.attributes {
        "kind" = %c6, "name" = %c7, "index" = %c8, "kernel" = %c9, "stride" = %c10,
        "pad" = %c11, "reflect" = %c12, "groups" = %c13, "shift" = %c14, "clamp" = %c15,
        "multiplier" = %c16
      }
    }

    // trunk + clamp(requant(branch)); the sum itself is not clamped.
    irdl.operation @add {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      %c3 = irdl.base "!builtin.tensor"
      %c4 = irdl.base "#builtin.integer"
      %c5 = irdl.base "#builtin.dense_array"
      %c6 = irdl.base "#builtin.integer"
      %c7 = irdl.base "#builtin.dense_array"
      %c8 = irdl.base "#builtin.dense_array"
      irdl.operands(trunk: %c1, branch: %c2)
      irdl.results(output: %c3)
      irdl.attributes {"index" = %c4, "multiplier" = %c5, "shift" = %c6, "clamp" = %c7, "range" = %c8}
    }

    // Nearest-neighbour upsampling by an integer scale.
    irdl.operation @upsample {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      %c3 = irdl.base "#builtin.integer"
      %c4 = irdl.base "#builtin.integer"
      %c5 = irdl.base "#builtin.dense_array"
      irdl.operands(input: %c1)
      irdl.results(output: %c2)
      irdl.attributes {"index" = %c3, "scale" = %c4, "range" = %c5}
    }

    // Standalone activation / requant / output quantization. The frontend
    // folds these into `conv`/`add` clamps; they exist so an unfused graph is
    // still expressible.
    irdl.operation @relu {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      irdl.operands(input: %c1)
      irdl.results(output: %c2)
    }
    irdl.operation @requant {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      %c3 = irdl.base "!builtin.tensor"
      %c4 = irdl.base "#builtin.integer"
      %c5 = irdl.base "#builtin.dense_array"
      irdl.operands(input: %c1, multiplier: %c2)
      irdl.results(output: %c3)
      irdl.attributes {"shift" = %c4, "clamp" = %c5}
    }
    irdl.operation @quantize {
      %c1 = irdl.base "!builtin.tensor"
      %c2 = irdl.base "!builtin.tensor"
      %c3 = irdl.base "#builtin.dense_array"
      %c4 = irdl.base "#builtin.integer"
      irdl.operands(input: %c1)
      irdl.results(output: %c2)
      irdl.attributes {"clamp" = %c3, "index" = %c4}
    }

    // ---- solver decisions ----

    // One engine instance: simd x pe MAC lanes, V vectors per step.
    // mode = "dedicated" | "shared".
    irdl.operation @engine {
      %c1 = irdl.base "#builtin.string"
      %c2 = irdl.base "#builtin.string"
      %c3 = irdl.base "#builtin.integer"
      %c4 = irdl.base "#builtin.integer"
      %c5 = irdl.base "#builtin.integer"
      irdl.attributes {"sym_name" = %c1, "mode" = %c2, "simd" = %c3, "pe" = %c4, "vectors" = %c5}
    }

    // Operator `op` (its `index`) runs on `engine` in schedule step `step`.
    irdl.operation @bind {
      %c1 = irdl.base "#builtin.integer"
      %c2 = irdl.base "#builtin.string"
      %c3 = irdl.base "#builtin.integer"
      %c4 = irdl.base "#builtin.integer"
      irdl.attributes {"op" = %c1, "engine" = %c2, "segment" = %c3, "step" = %c4}
    }

    // Residence of the tensor produced by operator `producer` (-1 = graph
    // input). residence = "local" (segment arena) | "kbank" (K rotating banks
    // shared across segments) | "offchip".
    irdl.operation @buffer {
      %c1 = irdl.base "#builtin.integer"
      %c2 = irdl.base "#builtin.string"
      %c3 = irdl.base "#builtin.integer"
      %c4 = irdl.base "#builtin.integer"
      %c5 = irdl.base "#builtin.integer"
      %c6 = irdl.base "#builtin.integer"
      %c7 = irdl.base "#builtin.integer"
      irdl.attributes {
        "producer" = %c1, "residence" = %c2, "banks" = %c3, "base" = %c4, "words" = %c5,
        "first_step" = %c6, "last_step" = %c7
      }
    }

    // A frame-pipeline segment: operators `ops` (indices) on `engines`.
    irdl.operation @segment {
      %c1 = irdl.base "#builtin.integer"
      %c2 = irdl.base "#builtin.string"
      %c3 = irdl.base "#builtin.dense_array"
      %c4 = irdl.base "#builtin.array"
      %c5 = irdl.base "#builtin.integer"
      irdl.attributes {"index" = %c1, "mode" = %c2, "ops" = %c3, "engines" = %c4, "cycles" = %c5}
    }

    // Frame pipeline: segment order and initiation interval in cycles.
    irdl.operation @schedule {
      %c1 = irdl.base "#builtin.dense_array"
      %c2 = irdl.base "#builtin.integer"
      %c3 = irdl.base "#builtin.integer"
      irdl.attributes {"segments" = %c1, "initiation_interval" = %c2, "latency" = %c3}
    }
  }
}
