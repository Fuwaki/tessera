# Developer setup

## Default build

The default build has no non-Rust dependencies:

```sh
cargo build
cargo test
```

Everything except the `optimize` stage works without MLIR: manifest import,
MLIR *emission*, graph lowering, layout search, RTL emission and Verilator
simulation are all implemented in this repository.

## Optional MLIR optimization (`melior` feature)

The `optimize` stage runs MLIR passes in process through
[Melior](https://github.com/mlir-rs/melior), pinned to exactly `0.28.2`:

```toml
melior = { version = "=0.28.2", optional = true }

[features]
default = []
melior = ["dep:melior"]
```

The pin is exact because both the pass names and the Melior API used in
`crates/tessera/src/mlir/pipeline.rs` are release specific. Melior 0.28.2 pulls
`mlir-sys` 230.0.2, which requires an **LLVM/MLIR 23.x** installation:

```sh
export MLIR_SYS_230_PREFIX=/path/to/llvm-23
# only needed if ODS-generated dialect bindings are ever enabled:
export TABLEGEN_230_PREFIX=/path/to/llvm-23

cargo check -p tessera --features melior
cargo test  -p tessera --features melior
cargo run   -p tessera --features melior -- optimize build/w8/mlir.json --out build/w8
```

`MLIR_SYS_230_PREFIX` must point at a prefix whose `bin/llvm-config` reports
version 23.x. The build script fails with, for example:

```text
failed to find correct version (23.x.x) of llvm-config (found 22.1.8)
```

That failure is the honest outcome of a missing toolchain: the feature is wired
correctly, the environment is not. No stage fakes a success it did not observe —
without the feature, `optimize` (and `tessera optimize`) reports
`MeliorUnavailable` and names the pin.

## Verilator (optional)

`tessera simulate` and the `rtl_verilator` integration tests use Verilator when
it is installed and skip transparently when it is not:

```sh
verilator --version
TESSERA_VERILATOR=/path/to/verilator cargo test -p tessera --test rtl_verilator
```

A skipped simulation is reported as `simulation: SKIP` and is never counted as a
verified result.

## Fixtures

`fixtures/` holds the committed examples: a synthetic W8 manifest with its
weight blob, a single-step pointwise primitive graph with its weight blob, and
the reference Ti60 target plus two policies. Regenerate them all from the same
Rust types the compiler loads:

```sh
cargo run -p tessera --example generate_fixtures
```

The generator is deterministic, so regeneration produces no diff.
