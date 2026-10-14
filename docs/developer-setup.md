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
`mlir-sys` 230.0.2, which requires an **LLVM/MLIR 23.x** installation. Both
build scripts are driven by environment variables whose names are versioned:
`MLIR_SYS_230_PREFIX` (read by `mlir-sys`) and `TABLEGEN_230_PREFIX` (read by
`tblgen`, which `melior-macro` runs). They must both point at the same LLVM 23
prefix, and that prefix's `bin/llvm-config` must report `23.x`:

```sh
export MLIR_SYS_230_PREFIX=/home/fuwaki/.local/share/mlir23   # llvm-config 23.1.3
export TABLEGEN_230_PREFIX=/home/fuwaki/.local/share/mlir23
"$MLIR_SYS_230_PREFIX/bin/llvm-config" --version                # -> 23.1.3

cargo check -p tessera --features melior
cargo test  -p tessera --features melior
cargo run   -p tessera --features melior -- optimize build/w8/mlir.json --out build/w8
```

On this workstation that prefix is the LLVM/MLIR 23.1.3 install at
`/home/fuwaki/.local/share/mlir23` (it carries `lib/cmake/mlir` and
`bin/llvm-tblgen`). No `LD_LIBRARY_PATH` is needed: `mlir-sys` links the static
MLIR archives by default. If a shared link is requested
(`MLIR_SYS_LINK_SHARED=1`), add `$MLIR_SYS_230_PREFIX/lib` to `LD_LIBRARY_PATH`
when *running* the compiler.

`MLIR_SYS_230_PREFIX` must point at a prefix whose `bin/llvm-config` reports
version 23.x. Pointing it at anything else fails loudly, for example:

```text
failed to find correct version (23.x.x) of llvm-config (found 22.1.8)
```

That failure is the honest outcome of a wrong or missing toolchain, not a wiring
bug: the feature resolves, downloads and builds melior 0.28.2, `mlir-sys`
230.0.2 and `tblgen` 0.10.0, and stops exactly where the environment is wrong.
Verified on this workstation with a standalone crate that depends only on
`melior = "=0.28.2"`: with both variables set it builds and
`Context::new()` links and runs. Without the feature, `optimize` (and
`tessera optimize`) reports `MeliorUnavailable` and names the pin rather than
faking a result.

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
