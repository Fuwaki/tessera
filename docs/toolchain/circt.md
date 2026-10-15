# CIRCT toolchain

Tessera lowers the `tessera` hardware dialect into CIRCT (`hw` / `seq` / `comb`,
and `esi` when a stream boundary needs it), then runs CIRCT's ExportVerilog to
produce SystemVerilog.

## Installation on this machine

| Item | Value |
| --- | --- |
| Prefix | `~/.local/share/circt23` |
| CIRCT commit | `847b0af5a` (`Bump LLVM (#10756)`, branch `llvm23-align`) |
| Linked LLVM/MLIR | `~/.local/share/mlir23` (conda-forge `mlir=23.1.3` / `llvmdev=23.1.3`) |
| Build dir | `/data/toolchains/circt-build` |
| Source dir | `/data/toolchains/circt-src` |

Configure (standalone against the existing LLVM 23.1.3 install):

```sh
cmake -G Ninja /data/toolchains/circt-src \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local/share/circt23 \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_DIR=$HOME/.local/share/mlir23/lib/cmake/llvm \
  -DMLIR_DIR=$HOME/.local/share/mlir23/lib/cmake/mlir \
  -DCIRCT_BINDINGS_PYTHON_ENABLED=OFF \
  -DCIRCT_SLANG_FRONTEND_ENABLED=OFF

ninja -C /data/toolchains/circt-build circt-opt
ninja -C /data/toolchains/circt-build install
```

CIRCT `main` tracks LLVM 24 and cannot link the Melior 0.28.2 / mlir-sys 230
stack. The pinned commit is the last LLVM-23-era bump that still configures
cleanly against `mlir=23.1.3`.

Runtime: put `$HOME/.local/share/circt23/bin` on `PATH`, and keep
`$HOME/.local/share/mlir23/lib` on `LD_LIBRARY_PATH` (CIRCT tools link LLVM
dynamically from that prefix).

## Verified on this machine

- `circt-opt` built and installed from commit `847b0af5a` against `mlir=23.1.3`.
- Smoke: `comb.add` inside `hw.module` lowers through `--export-verilog`.
