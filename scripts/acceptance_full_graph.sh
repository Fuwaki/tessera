#!/bin/sh
# Acceptance gate for the full-graph Tessera pipeline.
#
# Run this once, after every module owner has landed. It drives the real CLI
# over the checked-in acceptance fixtures, re-derives every expected value with
# the independent scalar reference in `tests/acceptance_support`, and compares
# the RTL's own trace against those values.
#
# Usage:
#   scripts/acceptance_full_graph.sh [--strict] [--features <list>]
#
#   --strict         the acceptance gate: treat "UNCOV" (a gate that could not
#                    run, e.g. no simulator is installed or the MLIR planning
#                    stage is unavailable) as a failure instead of a note, and
#                    require a simulated PASS rather than a skip. It also
#                    exports TESSERA_ACCEPTANCE_STRICT=1, so the acceptance tests
#                    themselves fail on an uncovered gate instead of skipping it.
#   --features LIST  cargo features to build the CLI and the tests with.
#                    Default: `melior` under --strict, nothing otherwise, because
#                    only the `melior` build can run `optimize`, and the strict
#                    chain insists on the optimized-MLIR path.
#
# Environment:
#   CARGO                   cargo binary (default: cargo)
#   TESSERA_VERILATOR       simulator binary (default: verilator)
#   TESSERA_INK_DW_DIR      real ink_dw export to import at a small resolution
#                           (default: <fixtures>/ink_dw if that directory exists)
#   TESSERA_ACCEPTANCE_OUT  stage directory to use (default: a fresh temp dir)
#   TESSERA_ACCEPTANCE_KEEP set to 1 to keep the temporary stage directory
#
# The MLIR toolchain is not discovered here: set MLIR_SYS_230_PREFIX and
# TABLEGEN_230_PREFIX to an LLVM/MLIR 23.x prefix before invoking this script
# when it builds with `--features melior` (see docs/developer-setup.md).
#
# The CP-SAT backend links OR-Tools dynamically. When ORTOOLS_PREFIX names the
# installation, its `lib` is prepended to the loader path below; a binary that
# links the backend — the mapping-solver tests, or the CLI once it solves a
# mapping — cannot start without it (see docs/developer-setup.md).
#
# Nothing here uses Python: the runtime smoke path is the Rust binary plus this
# shell script.

set -u

if [ -n "${ORTOOLS_PREFIX:-}" ]; then
    LD_LIBRARY_PATH="$ORTOOLS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FIXTURES="$ROOT/fixtures"
ACCEPTANCE="$FIXTURES/acceptance"
CARGO=${CARGO:-cargo}
TARGET="$FIXTURES/targets/ti60f225.toml"
STRICT=no
FEATURES=""
FEATURES_GIVEN=no
INK_INPUT_SHAPE="1,8,8,3"

while [ $# -gt 0 ]; do
    case "$1" in
        --strict) STRICT=yes; shift ;;
        --features)
            if [ $# -lt 2 ]; then
                echo "--features needs a value" >&2
                exit 64
            fi
            FEATURES="$2"
            FEATURES_GIVEN=yes
            shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 64 ;;
    esac
done

# The strict gate is the optimized-MLIR gate, so it needs the feature that can
# run `optimize`; a caller can override it explicitly.
FEATURE_ARGS=""
if [ "$FEATURES_GIVEN" = "yes" ]; then
    if [ -n "$FEATURES" ]; then
        FEATURE_ARGS="--features $FEATURES"
    fi
elif [ "$STRICT" = "yes" ]; then
    FEATURE_ARGS="--features melior"
fi
echo "cargo features: ${FEATURE_ARGS:-none}"
if [ -n "$FEATURE_ARGS" ]; then
    if [ -z "${MLIR_SYS_230_PREFIX:-}" ] || [ -z "${TABLEGEN_230_PREFIX:-}" ]; then
        echo "note: MLIR_SYS_230_PREFIX / TABLEGEN_230_PREFIX are unset; the build will fail" \
            "unless they point at an LLVM/MLIR 23.x prefix (docs/developer-setup.md)"
    fi
fi

# Stage directories live under /var/tmp, not /tmp: a full-graph bundle plus a
# Verilator build is hundreds of megabytes, and /tmp is often a small tmpfs.
STAGE_ROOT=${TESSERA_ACCEPTANCE_ROOT:-${TMPDIR:-/var/tmp}}
WORK=${TESSERA_ACCEPTANCE_OUT:-$(mktemp -d "$STAGE_ROOT/tessera-acceptance.XXXXXX")}
mkdir -p "$WORK"

if [ "$STRICT" = "yes" ]; then
    # The tests fail (rather than skip) on an uncovered gate.
    export TESSERA_ACCEPTANCE_STRICT=1
fi

PASSED=0
FAILED=0
SKIPPED=0
UNCOVERED=0

pass() {
    PASSED=$((PASSED + 1))
    echo "PASS  $1"
}

fail() {
    FAILED=$((FAILED + 1))
    echo "FAIL  $1"
    shift
    for line in "$@"; do echo "      $line"; done
}

skip() {
    SKIPPED=$((SKIPPED + 1))
    echo "SKIP  $1"
    shift
    for line in "$@"; do echo "      $line"; done
}

uncovered() {
    if [ "$STRICT" = "yes" ]; then
        fail "$@"
        return
    fi
    UNCOVERED=$((UNCOVERED + 1))
    echo "UNCOV $1"
    shift
    for line in "$@"; do echo "      $line"; done
}

section() {
    echo
    echo "== $1"
}

# Runs a command, capturing both streams into OUT and ERR in this shell.
# Returns the command's own status.
run_capture() {
    shift # the label is informational only; argv follows
    OUT=$("$@" 2>"$WORK/stderr.txt")
    status=$?
    ERR=$(cat "$WORK/stderr.txt")
    return $status
}

tessera() {
    label=$1
    shift
    # shellcheck disable=SC2086  # FEATURE_ARGS is a deliberate word list
    run_capture "$label" "$CARGO" run -q -p tessera $FEATURE_ARGS -- "$@"
}

section "1. independent reference against the frozen vectors"
# shellcheck disable=SC2086
if run_capture vectors "$CARGO" test -p tessera $FEATURE_ARGS --test acceptance_vectors; then
    pass "vectors: the frozen expectations equal the independently recomputed values"
else
    fail "vectors: the reference and the frozen vectors disagree" "$OUT" "$ERR"
fi

section "2. import, lowering and planning (library acceptance)"
# shellcheck disable=SC2086
if run_capture pipeline "$CARGO" test -p tessera $FEATURE_ARGS --test acceptance_pipeline; then
    pass "pipeline: lowering, planner and provenance match the reference"
else
    fail "pipeline: library-level acceptance failed" "$OUT" "$ERR"
fi

section "3. CLI acceptance and the RTL trace"
# shellcheck disable=SC2086
if run_capture cli "$CARGO" test -p tessera $FEATURE_ARGS --test acceptance_cli -- --nocapture; then
    if echo "$OUT$ERR" | grep -q "SKIP: the full acceptance graph is not emittable yet"; then
        uncovered "rtl: the full-graph RTL path is not emittable yet" \
            "the suite compared nothing for the full graph; see the SKIP reason above"
    else
        pass "rtl: the emitted RTL trace matches the independent reference"
    fi
    if echo "$OUT$ERR" | grep -q "SKIP: no pointwise simulation evidence"; then
        uncovered "rtl: the pointwise envelope produced no simulation evidence"
    fi
else
    fail "rtl: CLI acceptance failed" "$OUT" "$ERR"
fi

# The optimized-MLIR leg is part of the CLI suite only when it is built; say so
# explicitly instead of letting a missing feature look like a pass.
if [ -z "$FEATURE_ARGS" ]; then
    uncovered "rtl: the optimized-MLIR acceptance leg needs --features melior" \
        "rerun with --strict (which defaults to it) or pass --features melior"
else
    pass "rtl: the optimized-MLIR acceptance leg ran with the pinned feature"
fi

section "4. end-to-end CLI: import -> emit-mlir -> optimize -> solve --mlir"
GRAPH_STAGE="$WORK/graph"
if tessera import import "$ACCEPTANCE/graph" --input-shape 1,5,5,3 --out "$GRAPH_STAGE"; then
    pass "import: the acceptance graph imports"
else
    fail "import: the acceptance graph did not import" "$OUT" "$ERR"
fi

if tessera emit-mlir emit-mlir "$GRAPH_STAGE/graph.tgraph" --out "$GRAPH_STAGE"; then
    case "$OUT" in
        *operators*) pass "emit-mlir: the tensor stage emitted the acceptance graph" ;;
        *) fail "emit-mlir: the tensor stage reported no operator summary" "$OUT" ;;
    esac
else
    fail "emit-mlir: the tensor stage failed" "$OUT" "$ERR"
fi

MLIR_STAGE="$GRAPH_STAGE"
if tessera optimize optimize "$GRAPH_STAGE/mlir.json" --out "$GRAPH_STAGE"; then
    if [ -f "$GRAPH_STAGE/optimized.json" ]; then
        pass "optimize: the pinned Melior pipeline wrote the planning-stage artifact"
    else
        fail "optimize: reported success but wrote no optimized.json" "$OUT"
    fi
else
    case "$OUT$ERR" in
        *melior*|*MLIR_SYS*)
            fail "optimize: the acceptance gate requires the Melior planning stage" "$OUT" "$ERR" ;;
        *) fail "optimize: the planning stage failed" "$OUT" "$ERR" ;;
    esac
fi

# The plan that is emitted and simulated below must come from the optimized
# module, not from a graph re-lowering that merely runs `optimize` alongside.
if [ -f "$GRAPH_STAGE/optimized.json" ]; then
    if tessera solve-mlir solve --mlir "$GRAPH_STAGE/optimized.json" --target "$TARGET" \
        --policy "$ACCEPTANCE/policy.toml" --out "$GRAPH_STAGE"; then
        case "$OUT" in
            *"layout: feasible"*) pass "solve --mlir: the optimized module is planned" ;;
            *) fail "solve --mlir: no feasible layout from the optimized module" "$OUT" ;;
        esac
    else
        fail "solve --mlir: solve failed on the optimized module" "$OUT" "$ERR"
    fi
    if [ -f "$GRAPH_STAGE/primitives.json" ]; then
        pass "solve --mlir: the primitive artifact carries its evidence"
    else
        fail "solve --mlir: no primitives.json was written" "$OUT"
    fi
    if tessera verify verify "$GRAPH_STAGE"; then
        case "$OUT" in
            *"verify: ok"*) pass "verify: the optimized stage chain verifies" ;;
            *) fail "verify: the optimized stage chain did not verify" "$OUT" ;;
        esac
    else
        fail "verify: verify exited non-zero on the optimized chain" "$OUT" "$ERR"
    fi
else
    uncovered "solve --mlir: no optimized artifact to plan from" \
        "run with --strict and a working MLIR 23 toolchain"
fi

# The same staged constraints must also plan the graph directly: layout search
# is reusable across inputs without repeating import (ADR-0002).
if tessera solve-graph solve --graph "$GRAPH_STAGE/graph.tgraph" --target "$TARGET" \
    --policy "$ACCEPTANCE/policy.toml" --out "$WORK/graph-derived"; then
    case "$OUT" in
        *"layout: feasible"*) pass "solve --graph: the same constraints also plan the graph" ;;
        *) fail "solve --graph: no feasible layout for the graph" "$OUT" ;;
    esac
else
    fail "solve --graph: solve failed for the graph" "$OUT" "$ERR"
fi

if tessera solve-impossible solve --graph "$GRAPH_STAGE/graph.tgraph" --input-shape 1,5,5,3 \
    --target "$TARGET" --policy "$ACCEPTANCE/policy_impossible.toml" --out "$WORK/impossible"; then
    fail "infeasible-budget: an impossible frame budget was accepted" "$OUT"
else
    case "$OUT$ERR" in
        *INFEASIBLE*|*infeasible*) pass "infeasible-budget: reported as infeasible, and named" ;;
        *) fail "infeasible-budget: failed without explaining itself" "$OUT" "$ERR" ;;
    esac
fi

if tessera solve-no-dsp solve --graph "$GRAPH_STAGE/graph.tgraph" --input-shape 1,5,5,3 \
    --target "$ACCEPTANCE/target_impossible.toml" \
    --policy "$ACCEPTANCE/policy.toml" --out "$WORK/no-dsp"; then
    fail "infeasible-target: an exhausted DSP budget was accepted" "$OUT"
else
    case "$OUT$ERR" in
        *dsp_budget*) pass "infeasible-target: the exhausted resource is named" ;;
        *) fail "infeasible-target: the exhausted resource was not named" "$OUT" "$ERR" ;;
    esac
fi

# Provenance staleness is checked in its own stage directory, so that the
# layout there descends from the graph artifact being perturbed.
STALE_STAGE="$WORK/stale"
if tessera stale-solve solve --graph "$GRAPH_STAGE/graph.tgraph" --target "$TARGET" \
    --policy "$ACCEPTANCE/policy.toml" --out "$STALE_STAGE"; then
    cp "$STALE_STAGE/graph.tgraph" "$WORK/graph.tgraph.orig"
    printf '\n' >>"$STALE_STAGE/graph.tgraph"
    if tessera verify-stale verify "$STALE_STAGE"; then
        fail "stale-provenance: verify accepted an edited solved-from artifact" "$OUT"
    else
        case "$OUT$ERR" in
            *"solved-from input hash"*)
                pass "stale-provenance: an edited solved-from artifact is rejected" ;;
            *) fail "stale-provenance: verify failed without naming the stale hash" "$OUT" "$ERR" ;;
        esac
    fi
    cp "$WORK/graph.tgraph.orig" "$STALE_STAGE/graph.tgraph"
else
    fail "stale-provenance: could not stage the check" "$OUT" "$ERR"
fi

section "5. emitted RTL for the optimized plan, through the command line"
LAYOUT="$MLIR_STAGE/layout.tlayout"
if tessera emit emit "$LAYOUT" --graph "$GRAPH_STAGE/graph.tgraph" --out "$GRAPH_STAGE"; then
    pass "emit: the acceptance plan emits an RTL bundle"
    if tessera simulate simulate "$LAYOUT" --graph "$GRAPH_STAGE/graph.tgraph" \
        --out "$GRAPH_STAGE"; then
        case "$OUT$ERR" in
            *"simulation: PASS"*)
                pass "simulate: the generated design reports PASS"
                case "$OUT$ERR" in
                    *mismatches=0*) pass "simulate: the trace reports zero mismatches" ;;
                    *) fail "simulate: PASS without a mismatch count" "$OUT" ;;
                esac
                if [ -f "$GRAPH_STAGE/sim/sim_outputs.txt" ]; then
                    pass "simulate: the per-step trace is on disk"
                else
                    fail "simulate: PASS without a per-step trace" \
                        "no sim/sim_outputs.txt in $GRAPH_STAGE; a PASS that cannot be re-read is not evidence"
                fi ;;
            *"simulation: SKIP"*)
                uncovered "simulate: no simulator was available" "$OUT" ;;
            *) fail "simulate: the simulation reported no result" "$OUT" "$ERR" ;;
        esac
    else
        fail "simulate: simulate exited non-zero" "$OUT" "$ERR"
    fi
else
    fail "emit: the acceptance plan did not emit" "$OUT" "$ERR"
fi

section "6. real ink_dw export: full chain on the stock weights"
INK=${TESSERA_INK_DW_DIR:-$FIXTURES/ink_dw}
if [ ! -f "$INK/manifest.json" ]; then
    skip "ink_dw: no exported model to import" \
        "looked in $INK; set TESSERA_INK_DW_DIR to a directory holding manifest.json and its weight blob"
else
    INK_STAGE="$WORK/ink"
    if tessera ink-import import "$INK" --input-shape "$INK_INPUT_SHAPE" --out "$INK_STAGE"; then
        pass "ink_dw: the real 29-layer export imports at $INK_INPUT_SHAPE"
    else
        fail "ink_dw: the real export did not import" "$OUT" "$ERR"
    fi
    if tessera ink-solve solve --graph "$INK_STAGE/graph.tgraph" --target "$TARGET" \
        --policy "$FIXTURES/policies/emit.toml" --out "$INK_STAGE"; then
        case "$OUT" in
            *"layout: feasible"*) pass "ink_dw: the real model reaches a feasible layout" ;;
            *) fail "ink_dw: no feasible layout for the real model" "$OUT" ;;
        esac
    else
        fail "ink_dw: solve failed for the real model" "$OUT" "$ERR"
    fi
    if [ -f "$INK_STAGE/layout.tlayout" ]; then
        if tessera ink-verify verify "$INK_STAGE"; then
            case "$OUT" in
                *"verify: ok"*) pass "ink_dw: the real stage chain verifies" ;;
                *) fail "ink_dw: the real stage chain did not verify" "$OUT" ;;
            esac
        else
            fail "ink_dw: verify failed for the real model" "$OUT" "$ERR"
        fi
        if tessera ink-emit emit "$INK_STAGE/layout.tlayout" --graph "$INK_STAGE/graph.tgraph" \
            --out "$INK_STAGE"; then
            pass "ink_dw: the real model emits an RTL bundle"
            if tessera ink-simulate simulate "$INK_STAGE/layout.tlayout" \
                --graph "$INK_STAGE/graph.tgraph" --out "$INK_STAGE"; then
                case "$OUT$ERR" in
                    *"simulation: PASS"*)
                        pass "ink_dw: the real model simulates"
                        case "$OUT$ERR" in
                            *mismatches=0*) pass "ink_dw: the real trace reports zero mismatches" ;;
                            *) fail "ink_dw: PASS without a mismatch count" "$OUT" ;;
                        esac
                        if [ -f "$INK_STAGE/sim/sim_outputs.txt" ]; then
                            pass "ink_dw: the real per-step trace is on disk"
                        else
                            fail "ink_dw: PASS without a per-step trace" \
                                "no sim/sim_outputs.txt in $INK_STAGE"
                        fi ;;
                    *"simulation: SKIP"*)
                        uncovered "ink_dw: no simulator was available for the real model" "$OUT" ;;
                    *) fail "ink_dw: the simulation reported no result" "$OUT" "$ERR" ;;
                esac
                TRACE="$INK_STAGE/sim/sim_outputs.txt"
                if [ -f "$TRACE" ]; then
                    echo "      ink_dw trace: $TRACE ($(wc -c <"$TRACE" | tr -d ' ') bytes)"
                fi
            else
                fail "ink_dw: simulate exited non-zero" "$OUT" "$ERR"
            fi
        else
            fail "ink_dw: the real model did not emit" "$OUT" "$ERR"
        fi
    fi
fi

section "summary"
echo "passed=$PASSED failed=$FAILED skipped=$SKIPPED uncovered=$UNCOVERED"
echo "stage directory: $WORK"

if [ "${TESSERA_ACCEPTANCE_KEEP:-0}" != "1" ] && [ "${TESSERA_ACCEPTANCE_OUT:-}" = "" ]; then
    rm -rf "$WORK"
fi

if [ "$FAILED" -ne 0 ]; then
    echo "acceptance: FAILED ($FAILED failing check(s))"
    exit 1
fi
if [ "$UNCOVERED" -ne 0 ]; then
    echo "acceptance: PASSED WITH $UNCOVERED UNCOVERED GATE(S) — this is not full acceptance"
    exit 0
fi
echo "acceptance: PASSED"
exit 0
