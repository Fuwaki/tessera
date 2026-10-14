# Tessera version-control standard

This document defines how Tessera versions source code, compiler behavior, IRs, execution plans, generated RTL, target descriptions, and compatibility contracts. Version control is part of compiler correctness: a plan or RTL artifact must be attributable to the exact inputs and rules that produced it.

## 1. Principles

- Every reproducible artifact MUST identify the compiler, target model, input graph, weights, and planner configuration that produced it.
- Semantic changes MUST be versioned explicitly; users MUST NOT infer compatibility from a Git commit timestamp.
- Generated RTL MUST be reproducible from committed source and inputs. Large generated outputs SHOULD remain build artifacts unless they are required golden fixtures.
- A change that cannot be explained as a reviewable diff MUST NOT be hidden in generated files or untracked local state.
- The repository MUST remain buildable from a clean checkout at every release tag.

## 2. Version scheme

Tessera uses Semantic Versioning for the public compiler and artifact contracts:

```text
MAJOR.MINOR.PATCH
```

### MAJOR

Increment MAJOR for incompatible changes to:

- Graph IR semantics;
- Hardware IR semantics;
- Execution Plan interpretation;
- serialized manifest or plan schemas without a migration path;
- RTL interface contracts;
- numerical behavior, rounding, padding, or layout semantics;
- public CLI behavior that invalidates existing automation.

### MINOR

Increment MINOR for backward-compatible additions:

- new operators;
- new target models;
- new execution modes;
- new planner objectives;
- new backend capabilities;
- new optional plan fields;
- new CLI commands or optional flags.

### PATCH

Increment PATCH for compatible corrections:

- bug fixes that preserve the documented contract;
- diagnostics;
- performance improvements with equivalent output and plan meaning;
- documentation and test-only changes;
- build and dependency maintenance that does not change generated semantics.

If a change alters generated RTL or plan selection for the same input, it is not automatically a PATCH. Review the numerical, resource, and schedule contracts first.

## 3. Versioned artifacts

Tessera has independent version domains:

| Artifact | Required version |
|---|---|
| Compiler CLI | SemVer release version |
| Graph IR / MLIR dialect | dialect/IR schema version |
| Hardware IR | hardware IR schema version |
| Execution Plan | `plan_schema_version` |
| Target model | target model revision |
| RTL interface | interface version or explicit compatibility contract |
| Weight/manifest input | manifest schema version and content hash |
|

A generated artifact MUST include provenance similar to:

```json
{
  "compiler_version": "0.1.0",
  "git_revision": "<commit>",
  "dirty": false,
  "graph_schema_version": 1,
  "plan_schema_version": 1,
  "target": "ti60f225",
  "target_revision": "2026-10-14",
  "input_manifest_sha256": "...",
  "weights_sha256": "...",
  "planner_config_sha256": "..."
}
```

A dirty working tree MUST be recorded as `dirty: true`; release generation MUST reject a dirty tree unless an explicit override is supplied.

## 4. Schema evolution

Serialized formats MUST be explicit and self-describing:

```json
{
  "format": "tessera.execution_plan",
  "schema_version": 1
}
```

Rules:

- Adding an optional field is compatible.
- Removing or changing the meaning of a field is incompatible.
- Enum additions require consumers to reject unknown values safely or declare a forward-compatible policy.
- Units MUST be encoded in field names or schema documentation; do not change `cycles` to mean nanoseconds.
- A schema migration MUST be deterministic and tested against fixtures.
- Old plans MUST NOT be silently reinterpreted as new plans.

When compatibility is required, provide an explicit migration:

```text
plan v1 → migrate → plan v2 → validate → backend
```

## 5. Git repository rules

### 5.1 Branches

- `main` contains releasable code.
- `develop` MAY contain integrated work when the project uses a development branch.
- Feature branches SHOULD be short-lived and named by intent, for example:
  - `feat/manifest-import`;
  - `feat/ti60-target-model`;
  - `fix/requant-rounding`;
  - `docs/execution-plan`.
- A branch MUST NOT mix unrelated compiler, target, and cleanup changes without a documented reason.

### 5.2 Commits

Each commit SHOULD represent one coherent change. A commit MUST:

- build or clearly document why it cannot build yet;
- include tests for behavior changes when reachable;
- avoid generated build directories, local caches, credentials, and machine-specific paths;
- preserve the ability to identify the affected IR/backend contract.

Recommended subject format:

```text
<area>: <imperative summary>
```

Examples:

```text
manifest: import W8A8 depthwise layers
mlir: preserve requant boundaries during fusion
planner: reject RAM bank overflow
rtl: emit shared tile schedule
```

Do not use commits that only say `misc`, `changes`, or `update` for architectural work.

### 5.3 History rewriting

- Do not rewrite shared branches.
- Force-push requires explicit coordination and MUST use `--force-with-lease`.
- Release tags MUST be immutable.
- If an architectural decision changes, add or supersede an ADR; do not silently rewrite the rationale in place.

## 6. Generated files and golden fixtures

Generated files fall into three categories:

### Reproducible build artifacts

Examples:

- generated Verilog;
- ROM images;
- plan reports;
- synthesis output;
- temporary simulator directories.

These SHOULD be ignored by Git and regenerated in CI or release jobs.

### Small golden fixtures

Small deterministic RTL, plan, or MLIR fixtures MAY be committed when they are required to detect regressions. Each fixture MUST include:

- the command that generated it;
- input and weight hashes;
- expected tool versions where relevant;
- a test that consumes it.

### Release deliverables

Board handoff packages MAY be committed or attached to a release when they are intentionally distributed artifacts. They MUST include a manifest of source revisions and hashes.

Never commit a generated file merely because it is convenient to inspect locally.

## 7. Dependency and toolchain versions

- `Cargo.lock` MUST be committed for binary/compiler work.
- Rust toolchain policy MUST be recorded in `rust-toolchain.toml` or release documentation.
- Melior and MLIR versions MUST be pinned as a compatible pair.
- Target vendor tool versions MUST be recorded for synthesis results.
- A dependency update MUST state whether it can alter numerical behavior, IR, generated RTL, or resource estimates.
- CI SHOULD exercise a clean checkout rather than relying on local caches.

## 8. Reproducible compiler outputs

The compiler MUST provide a reproducibility command or equivalent metadata path:

```text
input graph + weights + target + options + compiler revision
    → identical plan
    → identical RTL text or normalized RTL hash
```

If output intentionally contains nondeterminism, the source MUST be documented and a stable normalized representation MUST be available for comparison.

Search algorithms MUST accept an explicit seed. Tie-breaking MUST be deterministic, preferably by stable operator ID, candidate key, and lexical name.

## 9. Compatibility testing

Every release SHOULD validate a matrix covering:

```text
compiler version
× graph/manifest schema
× plan schema
× target model revision
× backend
× simulator
```

At minimum:

- old supported plans are validated or rejected with a clear migration error;
- supported manifests reproduce reference outputs;
- generated RTL is bit-exact on golden vectors;
- plan resource totals remain within declared budgets;
- synthesis reports are attributable to the exact release and target revision.

## 10. Release process

A release MUST follow this sequence:

1. Verify the working tree is clean.
2. Run formatting, unit tests, reference checks, RTL simulation, and required synthesis checks.
3. Generate plans and artifacts from pinned inputs.
4. Record compiler, target, toolchain, graph, and weight hashes.
5. Review public API, IR, plan, and RTL interface changes.
6. Update the changelog and migration notes.
7. Create an annotated, immutable Git tag.
8. Publish source and only the documented release artifacts.
9. Verify the release from a fresh checkout.

A release MUST NOT be made from an uncommitted local modification or an unrecorded target-tool installation.

## 11. Changelog categories

Changes SHOULD be grouped as:

- **Added** — new operators, targets, passes, or backends;
- **Changed** — behavior or performance-model changes;
- **Fixed** — corrected behavior;
- **Compatibility** — schema, interface, or toolchain changes;
- **Generated output** — deliberate RTL/resource changes;
- **Known limitations** — unsupported or unverified cases.

Numerical or schedule changes MUST mention whether they are:

- bit-exact semantic changes;
- plan-only changes;
- RTL-structure changes;
- measured synthesis changes;
- estimated-model changes.

## 12. Definition of version-control done

A change is ready to merge only when:

- [ ] The correct version domain is identified.
- [ ] Schema and migration impact are documented.
- [ ] Provenance fields are preserved or updated.
- [ ] Generated artifacts are classified correctly.
- [ ] The commit is focused and reproducible.
- [ ] Relevant reference, plan, RTL, and synthesis checks are recorded.
- [ ] The changelog or ADR is updated when the public contract changes.
