# Hardware-mapping solver

`tessera::solver` solves the *whole* hardware mapping jointly, where
`tessera::planner` enumerates one execution organization at a time:

- how many MAC engine instances to instantiate,
- which operator runs on which instance,
- what spatial fold each operator uses,
- when each operator runs,
- whether two engines overlap across a tensor through a pipeline FIFO,
- where every intermediate activation lives.

The result is a schema-2 `tessera.layout` artifact whose `mapping` field records
the engines, the operator bindings, the pipeline and the buffer residences.
Nothing in the solve is a lookup of a known answer: the four organizations the
tests exercise are four different `MappingBounds` on one input graph.

## Stages

```text
PlanInput + MappingBounds
   --candidates-->  legal (simd, pe, tap=1) folds per operator
   --encode-->      a ConstraintModel (solver-agnostic)
   --backend-->     a CP-SAT assignment, solved in process
   --check-->       an independent, recomputed verification
   --lower-->       a schema-2 LayoutArtifact
```

| Module | Responsibility |
| --- | --- |
| `candidates.rs` | Legal folds: `simd` divides input channels, `pe` divides output channels, `tap_parallel == 1`, lane product inside the per-engine bounds, shape enumerated by the caller. |
| `model.rs` | The constraint vocabulary (`Var`, `Constraint`, `Objective`, `ConstraintModel`) and its JSON document. Solver-independent: it names no solver library, model object or wire schema. |
| `encode.rs` | The real scheduling/binding/buffer program, plus decoding of the assignment. |
| `backend.rs` | The only module that names a solver crate. Translates every constraint into CP-SAT, pins `random_seed = 1` with one worker, solves in process and returns the assignment. |
| `check.rs` | Recomputes bindings, lane divisibility, roles, pipeline endpoints, arena liveness, the makespan formula and the DSP figure. |
| `lower.rs` | Fills the schedule, buffers, costs, requant and selection fields and attaches the `HardwareMapping`. |

## The constraint model

Variables:

| Variable | Meaning |
| --- | --- |
| `use.{e}` | engine slot `e` is instantiated |
| `shape.{e}.{s}` | engine slot `e` instantiates shape `s` |
| `bind.{i}.{e}` | operator `i` runs on engine slot `e` |
| `cand.{i}.{s}` | operator `i` uses fold `s` |
| `start.{i}`, `dur.{i}` | schedule position and lane sweeps |
| `makespan` | frame length |
| `ded.{e}`, `shared.{e}` | role, a consequence of the binding count |
| `link.{k}` | edge `k` is cut by a pipeline FIFO |
| `arena.{t}`, `live.{t}`, `dem.{t}` | tensor residence, lifetime and demand |

Constraints: exactly one candidate and one engine per operator; `bind ⇒ use`
and `use ⇒ some bind`; one shape per engine, with `bind ∧ cand ⇒ shape` so every
operator sharing an engine shares its shape; engine lane bounds; engine-count and
role-count bounds; `ded ⇒ Σbind == 1` and `shared ⇒ Σbind ≥ 2`; `dur` from the
chosen candidate; dependency order, relaxed *only* on an edge that carries a
chosen link (`start_c ≥ start_p + 1` with a link, `start_c ≥ start_p + dur_p`
without); `NoOverlap` over the operators bound to each engine; and a `Cumulative`
arena whose demand is each resident tensor's word count, with pipeline-crossed
tensors excluded.

Objective, hierarchical and lexicographic: minimize the makespan, then the total
DSP48, then the engine count, then the total lane count. The backend realizes
this by optimizing each objective to optimality and constraining it to that
optimum before moving on.

## The backend boundary

`ConstraintModel` is a plain, solver-agnostic description: variables with a
Boolean or bounded-integer domain, six constraint forms, and a lexicographic
objective list. `src/solver/backend.rs` is the only module that names a solver
crate, and it translates each constraint directly rather than re-encoding it:

| `Constraint` | CP-SAT |
| --- | --- |
| `Linear` | `add(expression rel bound)` |
| `ExactlyOne` / `AtMostOne` | `add_exactly_one` / `add_at_most_one` |
| `ImplyLinear` | the linear constraint carrying the antecedent as its native `enforcement_literal` (half-reification), not a bound-derived big-M |
| `NoOverlap` | `add_no_overlap` over optional intervals |
| `Cumulative` | `add_cumulative` over the same intervals, with one demand expression per interval and the capacity expression |

Every optional interval materializes its end as `start + length` with the bounds
implied by its two operands — never a guessed horizon — and carries its presence
literal, so an absent interval is simply ignored by `no_overlap` and
`cumulative`.

The JSON form of a model (documented in `model.rs`) survives as an inspection
format: `encode_model` returns the model, `to_json` prints it, and a test proves
the description never names the backend implementation. It is not a subprocess
protocol — there is no interpreter, no `TESSERA_CPSAT_PYTHON`, and no second
solver in the repository.

The search is pinned to `random_seed = 1` and `num_search_workers = 1`, so the
same program always yields the same assignment — a second solve of the same
input is byte-identical, which `tests/mapping_solver.rs` asserts.

Statuses are reported honestly. Only CP-SAT's `INFEASIBLE` becomes
`PlanStatus::Infeasible`. Every objective stage must *prove* its optimum, so a
stage that stops at `FEASIBLE` without proving it — and any `UNKNOWN` or
`MODEL_INVALID` status — is a `SolverError::Backend` failure rather than a value
frozen as if it were optimal; a pure feasibility program (no objectives) may
accept the first feasible assignment. A program that cannot be translated —
duplicate or unknown variable names, an empty `exactly_one`, a literal that was
not declared Boolean, an empty integer domain — is refused before the solver
runs, for the same reason.

Building the backend needs an OR-Tools C++ 9.15 installation and `protoc`, and
running it needs that installation's `lib` on the loader path; see
[developer setup](developer-setup.md).

Budget and DMA modelling are out of scope: the mapping solve charges no DDR
traffic and does not schedule DMA. `Resources.ddr_*` are zero, and the frame
budget is enforced by the artifact's own validation rather than by the model.

## Independent verification

`check.rs` never trusts the solver's status. Before lowering it recomputes:

- one binding per operator, on an instantiated engine whose shape is the
  operator's shape;
- lane divisibility (`simd | in_channels`, `pe | out_channels`) and the
  per-engine lane bounds;
- the role of every engine from its binding count, against the requested
  minimum dedicated/shared counts;
- that every pipeline link joins two *different* used engines across a tensor one
  really produces and the other really consumes;
- arena placement by tensor lifetime (`layout::allocate_arena`), and that no two
  simultaneously live residences share a word;
- the makespan: the sequential sum when nothing is pipelined, otherwise
  `max(engine work) + 1`, strictly below the sequential sum;
- the DSP48 figure from `Ti60Target::map_mac` and `Ti60Target::map_requant`.

Only then does `lower.rs` build the artifact, and `solve_mapping` validates it
before returning. A checked mapping that no longer fits the target budget is
reported as infeasible rather than returned.

## Schema 2

`LAYOUT_SCHEMA_VERSION_V2 = 2` adds `LayoutArtifact::mapping` (and
`ExecutionPlan::mapping`). Version 1 requires `mapping` to be `None`; version 2
requires it to be present. On top of the v1 per-step cost rules, a schema-2
artifact must satisfy:

- every schedule step has exactly one binding with the same operator name, the
  engine exists, and the step's engine is that engine;
- binding lanes equal the step's `cost.lanes` and the step's `simd × pe × tap`,
  and never exceed the engine's lanes;
- a dedicated engine carries exactly one binding, a shared engine at least two;
- every pipeline endpoint is a used engine, `depth_elements ≥ 1`, and the tensor
  is produced on the producer engine and consumed on the consumer engine;
- arena residences that are live at the same time do not overlap in word offsets,
  and non-arena residences carry no arena offset;
- with no pipeline, `makespan_cycles` equals the sum of the step estimates; with
  a pipeline it is strictly smaller and equals `resources.frame_cycles`.

The v1 rule "every step's lanes ≤ `selection.engine.lanes`" is skipped for
schema 2: `selection.engine` is only the widest-instance summary, so it is not a
global lane ceiling. Version 1 keeps that rule unchanged.

## Four organizations from four bounds

On the checked-in four-layer pointwise chain (`pointwise_chain4()`, each
operator 64 MACs over a 1×2×2×4 activation) against `ti60f225.toml`:

| Bounds | Result | Sequential | Pipelined |
| --- | --- | --- | --- |
| 1 engine, lanes = 1 | one shared engine, 4 ops, 3 DSP48 | 260 cycles | — |
| 1 engine, lanes = 4 | one shared engine, 4 ops, 10 DSP48 | 68 cycles | — |
| 2 engines, exactly 1 pipeline link, 2 shared engines | two shared engines of 2 ops each, 20 DSP48 | 68 cycles | 35 cycles |
| 2 engines, ≥ 1 dedicated, ≥ 1 shared | one dedicated + one shared engine, 20 DSP48 | 68 cycles | — |

The DSP48 figures come from the target model: `lanes / 2` blocks for the int8
dual-packed MAC path plus `lanes × ceil(32 / 18)` blocks for the 32-bit by 18-bit
requant multiplier.

## Scope

This solver handles the same operator envelope as the mapped RTL backend:
convolutions with `tap_parallel = 1`. Non-convolution operators are an error, not
a silently approximated plan. There is no DDR/DMA modelling, no synthesis flow
and no other FPGA vendor.
