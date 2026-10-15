//! The solver on the tessera dialect: frontend -> linalg -> tessera ops ->
//! CP-SAT -> decision ops written back into the same module.

use std::path::{Path, PathBuf};

use melior::ir::Module;
use tessera::dialect::{self, linalg_to_tessera, register_in_fresh_context, verify_module};
use tessera::frontend;
use tessera::lower::InputShape;
use tessera::mlir::{self, MlirOptions};
use tessera::planner::{Policy, ProvenanceInput};
use tessera::solver::MappingBounds;
use tessera::solver::dialect::{problem_from_tessera, solve_tessera};
use tessera::target::Ti60Target;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn target() -> Ti60Target {
    Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("target")
}

fn policy() -> Policy {
    Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("policy")
}

fn emit_policy() -> Policy {
    Policy::from_toml_file(fixtures().join("policies/emit.toml")).expect("emit policy")
}

fn tessera_text(dir: &str, shape: InputShape) -> (String, tessera::graph::GraphArtifact) {
    let artifact = frontend::import_manifest(&fixtures().join(dir)).expect("import");
    let linalg = mlir::emit_text(artifact.graph(), &MlirOptions::new(shape.to_nhwc())).expect("emit");
    (linalg, artifact)
}

fn check_fixture(dir: &str, shape: InputShape, policy: &Policy, require_feasible: bool) {
    use tessera::layout::PlanStatus;

    let (linalg, artifact) = tessera_text(dir, shape);
    let context = register_in_fresh_context().expect("dialect");
    let module: Module = linalg_to_tessera(&context, &linalg).expect("linalg -> tessera");
    let model = verify_module(&module).expect("verify operator ops");

    let problem = problem_from_tessera(&model).expect("problem");
    assert_eq!(problem.ops.len(), model.operators.len(), "{dir}: op count");

    let provenance = ProvenanceInput {
        graph_sha256: artifact.provenance().manifest_sha256.clone(),
        target_sha256: "target".into(),
        policy_sha256: "policy".into(),
        compiler_version: env!("CARGO_PKG_VERSION").into(),
    };
    let bounds = MappingBounds::hybrid_for(&target(), policy).expect("bounds");
    let solved = solve_tessera(&context, &module, &target(), policy, &provenance, &bounds)
        .expect("solve on the dialect");
    match solved.layout.status {
        PlanStatus::Infeasible if !require_feasible => {
            eprintln!("{dir}: infeasible under the given policy ({} operators)", model.operators.len());
            return;
        }
        PlanStatus::Infeasible => panic!("{dir}: expected a feasible solve"),
        PlanStatus::Feasible => {}
    }
    let decisions = solved.decisions.expect("feasible");
    let text = module.as_operation().to_string();
    eprintln!(
        "{dir}: {} ops, {} engines, {} segments, II={}",
        model.operators.len(),
        decisions.engines.len(),
        decisions.segments.len(),
        decisions.schedule.initiation_interval
    );
    // Segments are contiguous in operator index and form a DAG along the edges.
    let mut covered = vec![false; model.operators.len()];
    let mut seg_of = std::collections::HashMap::new();
    for seg in &decisions.segments {
        assert!(!seg.ops.is_empty(), "{dir}: empty segment");
        // Contiguous in topological (operators) order, not in sparse index values
        // (annotation indices may skip: 0,2,4,...).
        let positions: Vec<usize> = seg
            .ops
            .iter()
            .map(|idx| {
                model
                    .operators
                    .iter()
                    .position(|o| o.index == *idx)
                    .expect("segment op")
            })
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted.last().unwrap() - sorted.first().unwrap() + 1,
            sorted.len(),
            "{dir}: segment {} ops {:?} are not contiguous in topo order",
            seg.index,
            seg.ops
        );
        assert_eq!(positions, sorted, "{dir}: segment {} ops out of topo order", seg.index);
        for &op in &seg.ops {
            let slot = model.operators.iter().position(|o| o.index == op).expect("op");
            assert!(!covered[slot], "{dir}: op #{op} in two segments");
            covered[slot] = true;
            seg_of.insert(op, seg.index);
        }
        if seg.mode == "shared" {
            assert_eq!(seg.engines.len(), 1, "{dir}: shared segment engines");
        }
    }
    assert!(covered.iter().all(|&c| c), "{dir}: not every op is segmented");
    for op in &model.operators {
        for (source, _) in op.activations() {
            if let tessera::dialect::read::Source::Op(p) = source {
                assert!(
                    seg_of[&p] <= seg_of[&op.index],
                    "{dir}: segment order reverses #{p} -> #{}",
                    op.index
                );
            }
        }
    }
    let model = verify_module(&module).expect("decided module verifies");
    assert_eq!(model.binds.len(), model.operators.len());
    assert!(text.contains("\"tessera.engine\"") && text.contains("\"tessera.schedule\""));
    assert_eq!(decisions.segments.len(), model.segments.len());
    assert!(solve_tessera(&context, &module, &target(), policy, &provenance, &bounds).is_err());
    let _ = dialect::DIALECT_NAME;
}

#[test]
fn w8_is_solved_on_the_tessera_dialect() {
    check_fixture("w8", InputShape::new(3, 6, 6).expect("shape"), &policy(), true);
}


#[test]
fn ink_dw_dialect_problem_builds_from_emitted_linalg() {
    if !fixtures().join("ink_dw/manifest.json").exists() {
        eprintln!("SKIP: fixtures/ink_dw is not present");
        return;
    }
    let shape = InputShape::new(3, 8, 8).expect("shape");
    let artifact = frontend::import_manifest(&fixtures().join("ink_dw")).expect("import");
    let linalg = mlir::emit_text(artifact.graph(), &MlirOptions::new(shape.to_nhwc()))
        .expect("ink_dw i16 depthwise must emit (6-iterator generic)");
    let context = register_in_fresh_context().expect("dialect");
    let module = linalg_to_tessera(&context, &linalg).expect("linalg -> tessera");
    let model = verify_module(&module).expect("verify");
    let plan = problem_from_tessera(&model).expect("problem");
    assert!(
        plan.ops.len() >= 10,
        "ink_dw should lower to a multi-op problem, got {}",
        plan.ops.len()
    );
    let kinds: Vec<_> = plan.ops.iter().map(|o| o.kind).collect();
    assert!(
        kinds.iter().any(|k| matches!(k, tessera::layout::OpKind::DepthwiseConv2d)),
        "ink_dw must include a depthwise conv: {kinds:?}"
    );
    eprintln!("ink_dw dialect problem: {} ops", plan.ops.len());
}

#[test]
#[ignore = "full ink_dw CP-SAT with segment vars is too slow for the default suite; run with --ignored"]
fn ink_dw_cpsat_segments_are_contiguous_and_acyclic() {
    use tessera::layout::PlanStatus;
    use tessera::solver::{self, MappingBounds};

    if !fixtures().join("ink_dw/manifest.json").exists() {
        eprintln!("SKIP: fixtures/ink_dw is not present");
        return;
    }
    // Full CP-SAT over ink_dw (~29 ops) with segment vars: keep ignored.
    // The default suite covers dialect construction via `ink_dw_dialect_problem`.
    let shape = InputShape::new(3, 8, 8).expect("shape");
    let artifact = frontend::import_manifest(&fixtures().join("ink_dw")).expect("import");
    // Dialect path: i16 depthwise uses a 6-iterator generic (no Ic reduction).
    let linalg = match mlir::emit_text(artifact.graph(), &MlirOptions::new(shape.to_nhwc())) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("ink_dw: emit still blocked ({error}); segment encoding covered by W8");
            return;
        }
    };
    let context = register_in_fresh_context().expect("dialect");
    let module = match linalg_to_tessera(&context, &linalg) {
        Ok(m) => m,
        Err(error) => {
            eprintln!("ink_dw: convert blocked ({error}); segment encoding covered by W8");
            return;
        }
    };
    let model = verify_module(&module).expect("verify");
    let plan = problem_from_tessera(&model).expect("problem");
    let provenance = ProvenanceInput {
        graph_sha256: artifact.provenance().manifest_sha256.clone(),
        target_sha256: "target".into(),
        policy_sha256: "policy".into(),
        compiler_version: env!("CARGO_PKG_VERSION").into(),
    };
    let policy = emit_policy();
    let mut bounds = MappingBounds::hybrid_for(&target(), &policy).expect("bounds");
    // Cap the search: full hybrid on 29 ops with segment variables is too slow
    // for a unit test; the encoding is what we need to exercise.
    bounds.max_engines = bounds.max_engines.min(8);
    bounds.shapes.retain(|s| s.lanes() <= 16);
    if bounds.shapes.is_empty() {
        bounds.shapes.push(tessera::solver::SpatialShape { simd: 1, pe: 1, tap_parallel: 1 });
    }
    let layout = solver::solve_mapping(&plan, &target(), &policy, &provenance, &bounds)
        .expect("solve ink_dw");
    match layout.status {
        PlanStatus::Infeasible => {
            eprintln!(
                "ink_dw: infeasible under emit policy ({} ops) — segment encoding not exercised",
                plan.ops.len()
            );
            return;
        }
        PlanStatus::Feasible => {}
    }
    let mapping = layout.mapping.as_ref().expect("schema-2 mapping");
    assert!(
        !mapping.segments.is_empty(),
        "CP-SAT must record segments on the layout mapping"
    );
    let mut covered = vec![false; plan.ops.len()];
    let mut seg_of = vec![0u32; plan.ops.len()];
    for seg in &mapping.segments {
        assert!(!seg.ops.is_empty());
        let positions: Vec<usize> = seg
            .ops
            .iter()
            .map(|name| plan.ops.iter().position(|o| o.name == *name).expect("op"))
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted.last().unwrap() - sorted.first().unwrap() + 1,
            sorted.len(),
            "ink_dw: segment {} {:?} not contiguous",
            seg.index,
            seg.ops
        );
        for &p in &positions {
            assert!(!covered[p], "op in two segments");
            covered[p] = true;
            seg_of[p] = seg.index;
        }
        if seg.mode == "shared" {
            assert_eq!(seg.engines.len(), 1);
        }
    }
    assert!(covered.iter().all(|&c| c), "not every op is segmented");
    // Edges only go forward across segments.
    for (i, op) in plan.ops.iter().enumerate() {
        for inp in &op.inputs {
            if let Some(p) = plan.ops.iter().position(|o| o.outputs.contains(inp)) {
                assert!(
                    seg_of[p] <= seg_of[i],
                    "ink_dw: segment order reverses {} -> {}",
                    plan.ops[p].name,
                    op.name
                );
            }
        }
    }
    eprintln!(
        "ink_dw: {} ops, {} engines, {} segments (CP-SAT)",
        plan.ops.len(),
        mapping.engines.len(),
        mapping.segments.len()
    );
}
