//! Tessera command-line interface.
//!
//! The CLI is a thin, explicit driver over the persisted stages
//! (ADR-0002): every subcommand reads versioned artifacts from disk and writes
//! versioned artifacts back, and no subcommand re-runs an earlier stage
//! implicitly. See `docs/cli.md` for the worked examples this help text
//! mirrors.
//!
//! Failure policy: a stage that cannot do exactly what was asked reports why and
//! exits non-zero. In particular `emit` never picks a different layout, and it
//! never weakens a quantization contract to fit the backend's supported
//! envelope.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

use tessera::graph::GraphArtifact;
use tessera::layout::LayoutArtifact;
use tessera::lower::{InputShape, lower_graph};
use tessera::mlir::{self, MlirArtifact, MlirOptions, SourceGraph};
use tessera::planner::{self, Policy, ProvenanceInput};
use tessera::primitive;
use tessera::rtl::{self, RtlConfig, SimulationOutcome, SystemVerilogBackend};
use tessera::stages::{self, ArtifactKind, ChainCheck, StageDir};
use tessera::target::Ti60Target;
use tessera::weights::{GraphWeightResolver, NamedBlobResolver};

/// Identifier of the numerical contract this compiler implements.
///
/// The generated datapath computes `(acc * M0 + round) >> shift` with a 17-bit
/// multiplier and a rounding offset of `2^(shift-1)` for non-negative products and
/// `2^(shift-1) - 1` for negative ones, i.e. **round half away from zero**. That is
/// not the same rule as "half up" (toward positive infinity): the two disagree
/// exactly at negative ties, where this datapath rounds away from zero. What it
/// does *not* reproduce is the
/// training `f32` requantization factor: that is approximated by the serialized
/// `M0`/`shift`, and every step records the worst-case relative error it incurred
/// (`Requant::multiplier_error_ppm`). Reports say so instead of calling the result
/// bit-exact, because the two claims are different.
const SEMANTIC_CONTRACT: &str = "tessera.fixed_point.requant.half_away_from_zero.v1";

#[derive(Debug, Parser)]
#[command(
    name = "tessera",
    version,
    about = "Compile quantized graphs into hardware execution plans and RTL",
    long_about = "Tessera is a persistent, staged compiler: each subcommand consumes and \
                  produces versioned artifacts in a stage directory, so any stage can be \
                  inspected, compared and resumed without repeating earlier work."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Import a `fpga_st_train.int8.v1` manifest into a persisted graph artifact.
    Import(ImportArgs),
    /// Emit tensor-level MLIR text from a persisted graph artifact.
    EmitMlir(EmitMlirArgs),
    /// Run the pinned MLIR pass pipeline over an emitted artifact.
    Optimize(OptimizeArgs),
    /// Solve a hardware layout for a graph artifact or a lowered primitive graph.
    Solve(SolveArgs),
    /// Inspect a persisted artifact (graph, MLIR, primitive, layout or plan).
    Inspect(InspectArgs),
    /// Project a validated layout onto an execution plan and emit its RTL bundle.
    Emit(EmitArgs),
    /// Emit the RTL bundle and simulate it with Verilator (skipped when absent).
    Simulate(SimulateArgs),
    /// Re-check every persisted artifact and provenance link in a stage directory.
    Verify(VerifyArgs),
}

#[derive(Debug, Args)]
struct ImportArgs {
    /// Path to `manifest.json` or to the directory that contains it.
    manifest: PathBuf,
    /// Stage directory to write `graph.tgraph` into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Input activation shape as `N,H,W,C`. Recorded for later stages.
    #[arg(long, value_name = "N,H,W,C")]
    input_shape: Option<String>,
}

#[derive(Debug, Args)]
struct EmitMlirArgs {
    /// Persisted graph artifact (`graph.tgraph`).
    graph: PathBuf,
    /// Stage directory to write `mlir.json` and `module.mlir` into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Input activation shape as `N,H,W,C` (defaults to the recorded shape).
    #[arg(long, value_name = "N,H,W,C")]
    input_shape: Option<String>,
    /// Emitted MLIR function name.
    #[arg(long, value_name = "NAME", default_value = mlir::DEFAULT_FUNCTION_NAME)]
    function: String,
}

#[derive(Debug, Args)]
struct OptimizeArgs {
    /// Tensor-stage MLIR artifact (`mlir.json`).
    mlir: PathBuf,
    /// Stage directory to write `optimized.json` and `optimized.mlir` into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

#[derive(Debug, Args)]
struct SolveArgs {
    /// Persisted graph artifact to lower and solve for.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["primitive", "mlir"])]
    graph: Option<PathBuf>,
    /// Lowered primitive graph (`PlanInput` JSON, or a `tessera.primitive` artifact).
    #[arg(long, value_name = "FILE", conflicts_with_all = ["graph", "mlir"])]
    primitive: Option<PathBuf>,
    /// Optimized MLIR artifact to lower and solve for (`optimized.json`).
    ///
    /// The plan is derived from the optimized module's own operations, loop
    /// nests and materialized buffers, so this path cannot re-derive a plan from
    /// the graph behind MLIR's back.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["graph", "primitive"])]
    mlir: Option<PathBuf>,
    /// Target description (TOML).
    #[arg(long, value_name = "FILE")]
    target: PathBuf,
    /// Planning policy (TOML).
    #[arg(long, value_name = "FILE")]
    policy: PathBuf,
    /// Stage directory to write the layout, plan and input copies into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Input activation shape as `N,H,W,C`, required for `--graph`.
    #[arg(long, value_name = "N,H,W,C")]
    input_shape: Option<String>,
}

#[derive(Debug, Args)]
struct InspectArgs {
    /// Artifact file to inspect.
    artifact: PathBuf,
    /// Print the artifact's own serialized form instead of a summary.
    #[arg(long)]
    raw: bool,
}

#[derive(Debug, Args)]
struct EmitArgs {
    /// Validated layout artifact (`layout.tlayout`).
    layout: PathBuf,
    /// Stage directory to write `plan.tplan` and `rtl/` into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Graph artifact to resolve weight bytes from (with its weight blob).
    #[arg(long, value_name = "FILE", conflicts_with = "weights")]
    graph: Option<PathBuf>,
    /// Directory of `<tensor-name>.bin` weight blobs for primitive fixtures.
    #[arg(long, value_name = "DIR", conflicts_with = "graph")]
    weights: Option<PathBuf>,
    /// Generated module name prefix.
    #[arg(long, value_name = "PREFIX", default_value = "tessera")]
    module_prefix: String,
    /// Stimulus windows in the generated self-checking testbench.
    #[arg(long, value_name = "N", default_value_t = 16)]
    fixture_windows: usize,
    /// Testbench stimulus seed.
    #[arg(long, value_name = "SEED", default_value_t = 0x5eed_1234_abcd_0001)]
    fixture_seed: u64,
    /// Do not emit the self-checking testbench.
    #[arg(long)]
    no_testbench: bool,
}

#[derive(Debug, Args)]
struct SimulateArgs {
    /// Validated layout artifact (`layout.tlayout`).
    layout: PathBuf,
    /// Stage directory to write `plan.tplan`, `rtl/` and `sim/` into.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Graph artifact to resolve weight bytes from (with its weight blob).
    #[arg(long, value_name = "FILE", conflicts_with = "weights")]
    graph: Option<PathBuf>,
    /// Directory of `<tensor-name>.bin` weight blobs for primitive fixtures.
    #[arg(long, value_name = "DIR", conflicts_with = "graph")]
    weights: Option<PathBuf>,
    /// Generated module name prefix.
    #[arg(long, value_name = "PREFIX", default_value = "tessera")]
    module_prefix: String,
    /// Stimulus windows in the generated self-checking testbench.
    #[arg(long, value_name = "N", default_value_t = 16)]
    fixture_windows: usize,
    /// Testbench stimulus seed.
    #[arg(long, value_name = "SEED", default_value_t = 0x5eed_1234_abcd_0001)]
    fixture_seed: u64,
}

#[derive(Debug, Args)]
struct VerifyArgs {
    /// Stage directory to check.
    dir: PathBuf,
    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            for cause in error.chain().skip(1) {
                eprintln!("  caused by: {cause}");
            }
            std::process::ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<std::process::ExitCode> {
    match cli.command {
        Command::Import(args) => cmd_import(args),
        Command::EmitMlir(args) => cmd_emit_mlir(args),
        Command::Optimize(args) => cmd_optimize(args),
        Command::Solve(args) => cmd_solve(args),
        Command::Inspect(args) => cmd_inspect(args),
        Command::Emit(args) => cmd_emit(args),
        Command::Simulate(args) => cmd_simulate(args),
        Command::Verify(args) => cmd_verify(args),
    }
}

fn ok() -> Result<std::process::ExitCode> {
    Ok(std::process::ExitCode::SUCCESS)
}

// ---------------------------------------------------------------------------
// import
// ---------------------------------------------------------------------------

fn cmd_import(args: ImportArgs) -> Result<std::process::ExitCode> {
    let artifact = tessera::frontend::import_manifest(&args.manifest)
        .with_context(|| format!("importing manifest {}", args.manifest.display()))?;
    let dir = StageDir::new(args.out.clone());
    dir.ensure()
        .with_context(|| format!("creating stage directory {}", dir.root().display()))?;
    let path = dir.path(stages::GRAPH_FILE);
    tessera::frontend::save_artifact(&artifact, &path)
        .with_context(|| format!("writing {}", path.display()))?;

    // Copy the weight blob next to the artifact so the stage directory is
    // self-contained: `emit` and `verify` must not depend on the manifest's
    // original location.
    let manifest_dir = if args.manifest.is_dir() {
        args.manifest.clone()
    } else {
        args.manifest
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };
    let weight_file = artifact.provenance().weight_file.clone();
    let destination = copy_weight_blob(&dir, &manifest_dir, &weight_file)?;
    let digest = stages::sha256_of(&destination)?;
    if digest != artifact.provenance().weights_sha256 {
        bail!(
            "copied weight blob {} does not match the recorded digest (expected {}, found {})",
            destination.display(),
            artifact.provenance().weights_sha256,
            digest
        );
    }
    println!("  weight blob: {}", destination.display());

    if let Some(shape) = &args.input_shape {
        let parsed = InputShape::parse_nhwc(shape)?;
        write_input_shape(&dir, parsed)?;
    }

    let metrics = artifact.metrics();
    println!(
        "imported {} ({} layers, {} graph ops, {} points)",
        args.manifest.display(),
        metrics.layer_count,
        metrics.graph_op_count,
        metrics.point_count
    );
    println!(
        "  weights: {} codes, {} bytes, packing {} ({})",
        metrics.weight_codes,
        metrics.weight_bytes,
        artifact.graph().packing().as_str(),
        artifact.provenance().weight_file
    );
    println!(
        "  manifest sha256: {}",
        artifact.provenance().manifest_sha256
    );
    println!("  graph artifact: {}", path.display());
    ok()
}

// ---------------------------------------------------------------------------
// emit-mlir
// ---------------------------------------------------------------------------

fn cmd_emit_mlir(args: EmitMlirArgs) -> Result<std::process::ExitCode> {
    let artifact = load_graph_file(&args.graph)?;
    let dir = StageDir::new(args.out.clone());
    let shape = resolve_input_shape(&dir, args.input_shape.as_deref(), &args.graph)?;
    write_input_shape(&dir, shape)?;

    let options = MlirOptions::new(shape.to_nhwc()).with_function_name(&args.function);
    options
        .validate()
        .map_err(|error| anyhow!("invalid MLIR options: {error}"))?;
    let emitted = mlir::lower(artifact.graph(), SourceGraph::from(&artifact), &options)
        .map_err(|error| anyhow!("emitting MLIR: {error}"))?;

    let json = dir.path(stages::MLIR_FILE);
    emitted
        .save(&json)
        .with_context(|| format!("writing {}", json.display()))?;
    let module = dir.path(stages::MODULE_FILE);
    emitted
        .save_module(&module)
        .with_context(|| format!("writing {}", module.display()))?;

    println!(
        "emitted {} ({} operators, {} arguments, {} layers) for input shape {}",
        args.function,
        emitted.operators.len(),
        emitted.arguments.len(),
        emitted.layers.len(),
        format_shape(shape)
    );
    println!("  module hash: {}", emitted.provenance.module_hash);
    println!("  artifact: {}", json.display());
    println!("  module:   {}", module.display());
    ok()
}

// ---------------------------------------------------------------------------
// optimize
// ---------------------------------------------------------------------------

fn cmd_optimize(args: OptimizeArgs) -> Result<std::process::ExitCode> {
    if !mlir::melior_available() {
        bail!(
            "this build has no `melior` feature, so it cannot run MLIR passes; rebuild with \
             `--features melior` against melior {} and an LLVM/MLIR 23.x installation \
             (see docs/developer-setup.md)",
            mlir::PINNED_MELIOR_VERSION
        );
    }
    let artifact = MlirArtifact::load(&args.mlir)
        .with_context(|| format!("loading {}", args.mlir.display()))?;
    let optimized = mlir::optimize(&artifact)
        .map_err(|error| anyhow!("running the pinned MLIR pipeline: {error}"))?;

    let dir = StageDir::new(args.out.clone());
    let json = dir.path(stages::OPTIMIZED_FILE);
    optimized
        .save(&json)
        .with_context(|| format!("writing {}", json.display()))?;
    let module = dir.path(stages::OPTIMIZED_MODULE_FILE);
    optimized
        .save_module(&module)
        .with_context(|| format!("writing {}", module.display()))?;

    let stages_run = optimized
        .provenance
        .melior
        .as_ref()
        .map(|p| p.pipeline.len())
        .unwrap_or(0);
    println!(
        "optimized {} with melior {} ({} passes)",
        args.mlir.display(),
        mlir::PINNED_MELIOR_VERSION,
        stages_run
    );
    if let Some(structural) = &optimized.metrics.structural {
        println!(
            "  structural: {} linalg ops, {} affine loops, {} allocations ({} bytes)",
            structural.linalg_ops,
            structural.affine_loops,
            structural.memref_allocs,
            structural.allocated_bytes
        );
    }
    println!("  module hash: {}", optimized.provenance.module_hash);
    println!("  artifact: {}", json.display());
    ok()
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

fn cmd_solve(args: SolveArgs) -> Result<std::process::ExitCode> {
    let dir = StageDir::new(args.out.clone());
    dir.ensure()
        .with_context(|| format!("creating stage directory {}", dir.root().display()))?;

    // Always materialize the input artifact into the stage directory, and
    // record the hash of the *staged copy* so `verify` can re-check it later
    // without the original path.
    let (input, notes, primitives_json, source_label, source_file) =
        match (&args.graph, &args.primitive, &args.mlir) {
            (Some(graph), None, None) => {
                let artifact = load_graph_file(graph)?;
                // Resolve the resolution from the directory the graph lives in (or
                // from this stage directory) *before* staging a copy: the copy lands
                // in the output directory, which need not hold a recorded shape.
                let shape = resolve_input_shape(&dir, args.input_shape.as_deref(), graph)?;
                let staged = materialize(&dir, graph, stages::GRAPH_FILE)?;
                copy_weight_blob(
                    &dir,
                    graph.parent().unwrap_or_else(|| Path::new(".")),
                    artifact.provenance().weight_file.as_str(),
                )?;
                write_input_shape(&dir, shape)?;
                let (plan, report) =
                    lower_graph(artifact.graph(), shape).map_err(|error| anyhow!("{error}"))?;
                (
                    plan,
                    report.notes,
                    None,
                    graph.display().to_string(),
                    stages::GRAPH_FILE,
                )
            }
            (None, Some(primitive), None) => {
                let text = std::fs::read_to_string(primitive)
                    .with_context(|| format!("reading {}", primitive.display()))?;
                let plan = stages::parse_primitive(&text).map_err(|error| anyhow!("{error}"))?;
                // A full `tessera.primitive` artifact is preserved verbatim, evidence and
                // all, so `verify` can link it to the module it descends from; a bare
                // `PlanInput` is re-serialized canonically.
                let is_artifact =
                    stages::detect_kind(&text) == Some(ArtifactKind::PrimitiveArtifact);
                if is_artifact {
                    let staged = materialize(&dir, primitive, stages::PRIMITIVES_FILE)?;
                    let artifact = primitive::PrimitiveArtifact::from_json(&text)
                        .map_err(|error| anyhow!("{error}"))?;
                    println!(
                        "  primitive artifact: {} ({} operators, {} fusion groups ({} fused), {} \
                     buffers) staged verbatim",
                        staged.display(),
                        artifact.plan.ops.len(),
                        artifact.groups.len(),
                        artifact.fused_groups().len(),
                        artifact.buffers.len()
                    );
                }
                let canonical = serde_json::to_string_pretty(&plan)
                    .context("serializing the primitive graph")?;
                dir.write_text(stages::PRIMITIVE_FILE, &canonical)
                    .with_context(|| {
                        format!("writing {}", dir.path(stages::PRIMITIVE_FILE).display())
                    })?;
                (
                    plan,
                    Vec::new(),
                    None,
                    primitive.display().to_string(),
                    stages::PRIMITIVE_FILE,
                )
            }
            (None, None, Some(mlir)) => {
                let artifact = MlirArtifact::load(mlir)
                    .with_context(|| format!("loading MLIR artifact {}", mlir.display()))?;
                let staged = materialize(&dir, mlir, stages::OPTIMIZED_FILE)?;
                // Integrity before consumption: the module text is re-hashed, the staged
                // graph is compared, and the embedded analysis is re-derived and compared
                // field by field. A build without the `melior` feature cannot re-derive it,
                // so the check fails there instead of trusting metadata it cannot confirm.
                artifact
                    .check(None)
                    .map_err(|error| anyhow!("MLIR artifact integrity: {error}"))?;
                println!(
                    "  analysis integrity: {}",
                    if mlir::analysis_verification_available() {
                        "re-derived from the module text and matched field by field"
                    } else {
                        "checked by the artifact (this build can re-derive it)"
                    }
                );
                let lowered = primitive::lower(&artifact)
                    .map_err(|error| anyhow!("lowering the optimized module: {error}"))?;
                let input_tensor = lowered
                    .plan
                    .tensor(lowered.plan.inputs[0])
                    .ok_or_else(|| anyhow!("the lowered plan has no input tensor"))?;
                write_input_shape(&dir, {
                    InputShape::new(
                        input_tensor.shape.c,
                        input_tensor.shape.h,
                        input_tensor.shape.w,
                    )
                    .map_err(|error| anyhow!("{error}"))?
                })?;
                let primitives = lowered.to_json().map_err(|error| anyhow!("{error}"))?;
                println!(
                    "lowered {} into {} tensors / {} operators, {} fusion groups ({} fused), {} \
                 buffers ({} allocated, {} dynamically shaped)",
                    staged.display(),
                    lowered.plan.tensors.len(),
                    lowered.plan.ops.len(),
                    lowered.groups.len(),
                    lowered.fused_groups().len(),
                    lowered.buffers.len(),
                    match lowered.allocated_bytes() {
                        Some(bytes) => format!("{bytes} bytes"),
                        None => "unknown".to_string(),
                    },
                    lowered.dynamic_buffers()
                );
                (
                    lowered.plan,
                    lowered.notes.clone(),
                    Some(primitives),
                    mlir.display().to_string(),
                    stages::OPTIMIZED_FILE,
                )
            }
            (None, None, None) => bail!(
                "select an input with --graph <graph.tgraph>, --mlir <optimized.json> or \
             --primitive <file>"
            ),
            _ => unreachable!("clap rejects more than one solve input"),
        };
    let source_hash = dir.sha256(source_file)?;

    let primitive_path = dir.path(stages::PRIMITIVE_FILE);
    if source_file != stages::PRIMITIVE_FILE {
        let primitive_json = serde_json::to_string_pretty(&input)
            .context("serializing the lowered primitive graph")?;
        dir.write_text(stages::PRIMITIVE_FILE, &primitive_json)
            .with_context(|| format!("writing {}", primitive_path.display()))?;
    }
    if let Some(primitives) = &primitives_json {
        // The evidence-carrying artifact (`tessera.primitive`) is written next to
        // the planner's own input so `verify` can link it to the optimized module.
        dir.write_text(stages::PRIMITIVES_FILE, primitives)
            .with_context(|| format!("writing {}", dir.path(stages::PRIMITIVES_FILE).display()))?;
    }

    dir.copy_in(&args.target, stages::TARGET_FILE)
        .with_context(|| format!("copying {}", args.target.display()))?;
    dir.copy_in(&args.policy, stages::POLICY_FILE)
        .with_context(|| format!("copying {}", args.policy.display()))?;
    let target_hash = dir.sha256(stages::TARGET_FILE)?;
    let policy_hash = dir.sha256(stages::POLICY_FILE)?;

    let target = Ti60Target::from_toml_file(dir.path(stages::TARGET_FILE))
        .map_err(|error| anyhow!("loading target: {error}"))?;
    let policy = Policy::from_toml_file(dir.path(stages::POLICY_FILE))
        .map_err(|error| anyhow!("loading policy: {error}"))?;

    for note in &notes {
        println!("  lower: {note}");
    }
    println!(
        "lowered {source_label}: {} tensors, {} operators (input {} -> output {})",
        input.tensors.len(),
        input.ops.len(),
        input.inputs[0],
        input.outputs[0]
    );

    let provenance = ProvenanceInput {
        graph_sha256: source_hash,
        target_sha256: target_hash,
        policy_sha256: policy_hash,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let artifact = planner::solve(&input, &target, &policy, &provenance)
        .map_err(|error| anyhow!("planning failed: {error}"))?;
    let layout_path = dir.path(stages::LAYOUT_FILE);
    artifact
        .save(&layout_path)
        .with_context(|| format!("writing {}", layout_path.display()))?;

    let certainty = artifact.certainty;
    let search = &artifact.search;
    println!(
        "  search: certainty {certainty:?}, {} candidates explored, {} expansions, \
         {} truncated candidate(s) of a {}-expansion share, truncated {}",
        search.explored,
        search.expanded_nodes,
        search.truncated_candidates,
        search.per_candidate_budget,
        search.truncated
    );

    // What the compiler's correctness claim actually is. The generated datapath is
    // exact against the *compiled* fixed-point contract; the training `f32`
    // requantization factors are approximated by the serialized 17-bit `M0` and the
    // shift, and the worst-case relative error of that approximation is recorded
    // per step rather than hidden behind the word "exact".
    let mut worst_ppm: Option<u32> = None;
    let mut off_contract = Vec::new();
    for step in &artifact.schedule {
        if let Some(requant) = &step.requant {
            if requant.rounding != tessera::layout::Rounding::HalfAwayFromZero {
                off_contract.push(step.op.clone());
            }
            if let Some(ppm) = requant.multiplier_error_ppm {
                worst_ppm = Some(worst_ppm.map_or(ppm, |current: u32| current.max(ppm)));
            }
        }
    }
    if off_contract.is_empty() {
        println!(
            "  semantic contract: {} (half-away-from-zero requantization, {}-bit multiplier); \
             the exactness baseline is the compiled fixed-point IR, not the training f32",
            SEMANTIC_CONTRACT,
            rtl::MULTIPLIER_BITS
        );
    }
    match worst_ppm {
        Some(ppm) => println!(
            "  requantization approximation: worst recorded {ppm} ppm relative to the manifest's \
             f32 coefficients ({} step(s) carry a bound)",
            artifact
                .schedule
                .iter()
                .filter(|step| step
                    .requant
                    .as_ref()
                    .is_some_and(|r| r.multiplier_error_ppm.is_some()))
                .count()
        ),
        None => println!(
            "  requantization approximation: no step recorded a bound, so the manifest f32 \
             divergence is unreported for this plan"
        ),
    }
    for op in &off_contract {
        println!("  warning: step {op} does not use the documented half-away-from-zero rounding");
    }

    match artifact.status {
        tessera::layout::PlanStatus::Feasible => {
            let plan = artifact
                .execution_plan()
                .map_err(|error| anyhow!("projecting the execution plan: {error}"))?;
            let plan_path = dir.path(stages::PLAN_FILE);
            write_plan(&plan_path, &plan)?;
            let resources = artifact
                .resources
                .expect("validated feasible artifact has a resource estimate");
            println!(
                "layout: feasible, mode {}, granularity {}, {} steps",
                artifact
                    .selection
                    .as_ref()
                    .map(|s| s.execution_mode.as_str())
                    .unwrap_or("?"),
                artifact
                    .selection
                    .as_ref()
                    .map(|s| format!("{:?}", s.granularity))
                    .unwrap_or_else(|| "?".to_string()),
                plan.steps.len()
            );
            println!(
                "  resources: {} DSP48, {} RAM10, {} XLR, {} frame cycles, {} read / {} write DDR bytes",
                resources.dsp48,
                resources.ram10,
                resources.xlr,
                resources.frame_cycles,
                resources.ddr_read_bytes_per_frame,
                resources.ddr_write_bytes_per_frame
            );
            for step in &plan.steps {
                println!(
                    "  step {}: {} ({:?}) simd {} pe {} tap {} in_buffer {} out_buffer {}",
                    step.step,
                    step.op,
                    step.op_kind,
                    step.simd,
                    step.pe,
                    step.tap_parallel,
                    step.input_buffer,
                    step.output_buffer
                );
            }
            println!("  layout: {}", layout_path.display());
            println!("  plan:   {}", plan_path.display());
            ok()
        }
        tessera::layout::PlanStatus::Infeasible => {
            // A truncated search has not proved that nothing is feasible, so it
            // must not be reported as a property of the target.
            let conclusive = certainty == tessera::layout::PlanCertainty::ProvenInfeasible;
            println!(
                "layout: {}, no candidate satisfied the target and policy",
                if conclusive {
                    "INFEASIBLE"
                } else {
                    "INCONCLUSIVE"
                }
            );
            for (mode, summary) in &artifact.rejections.by_mode {
                println!(
                    "  mode {mode}: {} candidate(s), {} feasible",
                    summary.candidates, summary.feasible
                );
                for (category, count) in &summary.counts {
                    println!("    rejected {category}: {count}");
                }
                if let Some(reason) = &summary.first_reason {
                    println!("    first reason: {reason}");
                }
            }
            for (category, count) in &artifact.rejections.counts {
                println!("  rejected {category}: {count}");
            }
            for sample in artifact.rejections.samples.iter().take(5) {
                println!(
                    "  e.g. {} / {:?}: {}",
                    sample.execution_mode.as_str(),
                    sample.granularity,
                    sample.reason
                );
            }
            println!("  layout: {}", layout_path.display());
            if conclusive {
                bail!(
                    "no feasible layout under {} and {} (the search covered the declared space); \
                     the rejected-candidate summary is in {}",
                    dir.path(stages::TARGET_FILE).display(),
                    dir.path(stages::POLICY_FILE).display(),
                    layout_path.display()
                );
            }
            bail!(
                "layout INCONCLUSIVE under {} and {}: the candidate budget stopped the search \
                 ({} truncated candidate(s)) without proving infeasibility; the rejected-candidate \
                 summary is in {}",
                dir.path(stages::TARGET_FILE).display(),
                dir.path(stages::POLICY_FILE).display(),
                search.truncated_candidates,
                layout_path.display()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

fn cmd_inspect(args: InspectArgs) -> Result<std::process::ExitCode> {
    let text = std::fs::read_to_string(&args.artifact)
        .with_context(|| format!("reading {}", args.artifact.display()))?;
    if args.raw {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
        return ok();
    }
    let kind = stages::detect_kind(&text).ok_or_else(|| {
        anyhow!(
            "{} is not a recognised tessera artifact (no matching format tag)",
            args.artifact.display()
        )
    })?;
    println!("{}: {}", args.artifact.display(), kind.as_str());
    match kind {
        ArtifactKind::Graph => {
            let artifact = tessera::frontend::load_artifact(&args.artifact)
                .map_err(|error| anyhow!("{error}"))?;
            let metrics = artifact.metrics();
            println!(
                "  source: {} ({})",
                artifact.provenance().source_format,
                artifact.graph().arch()
            );
            println!(
                "  manifest sha256: {}",
                artifact.provenance().manifest_sha256
            );
            println!(
                "  weights sha256:  {}",
                artifact.provenance().weights_sha256
            );
            println!(
                "  layers: {} ({} conv, {} depthwise, {} pointwise, {} relu)",
                metrics.layer_count,
                metrics.conv_layer_count,
                metrics.depthwise_layer_count,
                metrics.pointwise_layer_count,
                metrics.relu_layer_count
            );
            println!(
                "  ops: {} ({} add, {} upsample), points: {}",
                metrics.graph_op_count,
                metrics.add_count,
                metrics.upsample_count,
                metrics.point_count
            );
            println!(
                "  weights: {} codes / {} bytes, packing {}",
                metrics.weight_codes,
                metrics.weight_bytes,
                artifact.graph().packing().as_str()
            );
            println!("  macs per output pixel: {}", metrics.macs_per_output_pixel);
            for layer in &metrics.per_layer {
                println!(
                    "    {}: {} macs/pixel, {} weight codes",
                    layer.name, layer.macs_per_pixel, layer.weight_codes
                );
            }
        }
        ArtifactKind::Mlir => {
            let artifact =
                MlirArtifact::load(&args.artifact).map_err(|error| anyhow!("{error}"))?;
            println!(
                "  stage: {}, function: {}",
                artifact.stage.as_str(),
                artifact.function
            );
            println!(
                "  operators: {}, arguments: {}, layers: {}",
                artifact.operators.len(),
                artifact.arguments.len(),
                artifact.layers.len()
            );
            println!(
                "  source: {} ({})",
                artifact.provenance.source_format, artifact.provenance.source_schema_version
            );
            println!(
                "  source content hash: {}",
                artifact.provenance.source_content_hash
            );
            println!("  module hash: {}", artifact.provenance.module_hash);
            match &artifact.provenance.melior {
                Some(melior) => println!(
                    "  melior {}: {} passes",
                    melior.melior_version,
                    melior.pipeline.len()
                ),
                None => println!("  melior: not run (tensor stage)"),
            }
            if let Some(structural) = &artifact.metrics.structural {
                println!(
                    "  structural: {} linalg ops, {} affine loops, {} allocations ({} bytes)",
                    structural.linalg_ops,
                    structural.affine_loops,
                    structural.memref_allocs,
                    structural.allocated_bytes
                );
            }
        }
        ArtifactKind::Primitive => {
            let plan = stages::parse_primitive(&text).map_err(|error| anyhow!("{error}"))?;
            println!("  graph: {}", plan.name);
            println!(
                "  tensors: {}, ops: {}, inputs: {:?}, outputs: {:?}",
                plan.tensors.len(),
                plan.ops.len(),
                plan.inputs,
                plan.outputs
            );
            for op in &plan.ops {
                println!("    {}: {:?}", op.name, op.kind);
            }
        }
        ArtifactKind::PrimitiveArtifact => {
            let artifact = primitive::PrimitiveArtifact::from_json(&text)
                .map_err(|error| anyhow!("{error}"))?;
            println!("  graph: {}", artifact.plan.name);
            println!(
                "  tensors: {}, operators: {}, inputs: {:?}, outputs: {:?}",
                artifact.plan.tensors.len(),
                artifact.plan.ops.len(),
                artifact.plan.inputs,
                artifact.plan.outputs
            );
            let allocated = match artifact.allocated_bytes() {
                Some(bytes) => format!("{bytes} bytes"),
                None => format!(
                    "unknown over {} dynamic allocation(s)",
                    artifact.dynamic_buffers()
                ),
            };
            println!(
                "  loop groups: {} ({} fused), buffers: {} ({allocated}), intervals: {}",
                artifact.groups.len(),
                artifact.fused_groups().len(),
                artifact.buffers.len(),
                artifact.intervals.len()
            );
            println!(
                "  source module hash: {} (melior {})",
                artifact.provenance.mlir_module_hash,
                artifact
                    .provenance
                    .melior_version
                    .as_deref()
                    .unwrap_or("not recorded")
            );
            for op in &artifact.plan.ops {
                println!("    {}: {:?}", op.name, op.kind);
            }
            for note in &artifact.notes {
                println!("  note: {note}");
            }
        }
        ArtifactKind::Layout => {
            let artifact =
                LayoutArtifact::load(&args.artifact).map_err(|error| anyhow!("{error}"))?;
            println!(
                "  status: {}",
                if artifact.status == tessera::layout::PlanStatus::Feasible {
                    "feasible"
                } else {
                    "infeasible"
                }
            );
            println!(
                "  target: {} ({})",
                artifact.provenance.target_name, artifact.provenance.target_revision
            );
            println!("  graph sha256:  {}", artifact.provenance.graph_sha256);
            println!("  target sha256: {}", artifact.provenance.target_sha256);
            println!("  policy sha256: {}", artifact.provenance.policy_sha256);
            println!(
                "  search: {} explored, {} expanded, {} truncated candidate(s) of a {}-expansion \
                 share, truncated {}",
                artifact.search.explored,
                artifact.search.expanded_nodes,
                artifact.search.truncated_candidates,
                artifact.search.per_candidate_budget,
                artifact.search.truncated
            );
            println!("  certainty: {:?}", artifact.certainty);
            // The exactness baseline is the compiled fixed-point contract; the
            // training f32 coefficients are approximated by M0/shift, and the
            // recorded bound is what a reader should judge that approximation by.
            let worst = artifact
                .schedule
                .iter()
                .filter_map(|step| step.requant.as_ref()?.multiplier_error_ppm)
                .max();
            println!(
                "  semantic contract: {SEMANTIC_CONTRACT}; exactness baseline is the compiled \
                 fixed-point IR, not the training f32{}",
                match worst {
                    Some(ppm) => format!(", worst recorded requantization error {ppm} ppm"),
                    None => ", no requantization error bound recorded".to_string(),
                }
            );
            if let Some(resources) = artifact.resources {
                println!(
                    "  resources: {} DSP48, {} RAM10, {} XLR, {} cycles, {} read / {} write DDR bytes",
                    resources.dsp48,
                    resources.ram10,
                    resources.xlr,
                    resources.frame_cycles,
                    resources.ddr_read_bytes_per_frame,
                    resources.ddr_write_bytes_per_frame
                );
            }
            for step in &artifact.schedule {
                println!(
                    "    step {}: {} ({:?}) simd {} pe {} tap {} mode {:?}",
                    step.step,
                    step.op,
                    step.op_kind,
                    step.simd,
                    step.pe,
                    step.tap_parallel,
                    step.mode
                );
            }
            println!(
                "  rejections: {} candidates across {} categories",
                artifact.rejections.rejected_candidates,
                artifact.rejections.counts.len()
            );
        }
        ArtifactKind::Plan => {
            let plan = tessera::layout::ExecutionPlan::from_toml_str(&text)
                .map_err(|error| anyhow!("{error}"))?;
            println!(
                "  mode: {}, {} steps",
                plan.execution_mode.as_str(),
                plan.steps.len()
            );
            println!(
                "  weights: {} tensors, buffers: {}",
                plan.weights.len(),
                plan.buffers.len()
            );
            for step in &plan.steps {
                println!("    step {}: {} {:?}", step.step, step.op, step.op_kind);
            }
        }
        ArtifactKind::RtlBundle => {
            let metadata: rtl::RtlMetadata =
                serde_json::from_str(&text).context("parsing RTL bundle metadata")?;
            println!(
                "  kernel: {}, top module: {}",
                metadata.kernel_id, metadata.top_module
            );
            println!("  layer: {}, tile {:?}", metadata.layer, metadata.tile);
            println!("  files: {}", metadata.files.len());
            for file in &metadata.files {
                println!("    {} ({:?}), {} bytes", file.name, file.kind, file.bytes);
            }
        }
    }
    ok()
}

// ---------------------------------------------------------------------------
// emit / simulate
// ---------------------------------------------------------------------------

fn emission_input(
    layout_path: &Path,
    out: &Path,
    graph: &Option<PathBuf>,
    weights: &Option<PathBuf>,
    module_prefix: &str,
    fixture_windows: usize,
    fixture_seed: u64,
    emit_testbench: bool,
) -> Result<(LayoutArtifact, StageDir, RtlConfig)> {
    if graph.is_none() && weights.is_none() {
        bail!("weight bytes are required: pass --graph <graph.tgraph> or --weights <DIR>");
    }
    let layout = LayoutArtifact::load(layout_path)
        .with_context(|| format!("loading layout {}", layout_path.display()))?;
    layout
        .verify()
        .map_err(|error| anyhow!("layout {} is not usable: {error}", layout_path.display()))?;
    let config = RtlConfig {
        module_prefix: module_prefix.to_string(),
        emit_testbench,
        emit_memory_images: true,
        fixture_windows,
        fixture_seed,
    };
    config
        .validate()
        .map_err(|error| anyhow!("invalid RTL options: {error}"))?;
    Ok((layout, StageDir::new(out.to_path_buf()), config))
}

fn emit_bundle(
    layout: &LayoutArtifact,
    graph: &Option<PathBuf>,
    weights: &Option<PathBuf>,
    dir: &StageDir,
    config: &RtlConfig,
) -> Result<rtl::RtlBundle> {
    let plan = layout
        .execution_plan()
        .map_err(|error| anyhow!("projecting the execution plan: {error}"))?;
    write_plan(&dir.path(stages::PLAN_FILE), &plan)?;

    // Dispatch by plan shape, never by preference: a single pointwise step goes
    // to the single-kernel backend unchanged; anything else needs the full-graph
    // backend *and* the graph artifact, because bias comes from the model and
    // cannot be recovered from weight bytes alone.
    let bundle = if plan.steps.len() > 1 {
        match graph {
            Some(graph_path) => {
                let artifact = load_graph_file(graph_path)?;
                let resolver = graph_resolver(&artifact, graph_path)?;
                println!(
                    "  full-graph backend: {} steps, one static-schedule engine over an \
                     activation arena",
                    plan.steps.len()
                );
                SystemVerilogBackend
                    .lower_full_graph(&plan, artifact.graph(), &resolver, config)
                    .map_err(|error| anyhow!("{error}"))?
            }
            None => bail!(
                "the plan has {} steps, so it needs the full-graph backend, which resolves \
                 weights *and* bias from the graph artifact: pass --graph <graph.tgraph> \
                 instead of --weights <DIR>",
                plan.steps.len()
            ),
        }
    } else {
        match (graph, weights) {
            (Some(graph_path), _) => {
                let artifact = load_graph_file(graph_path)?;
                let resolver = graph_resolver(&artifact, graph_path)?;
                SystemVerilogBackend
                    .lower_execution_plan(&plan, &resolver, config)
                    .map_err(|error| anyhow!("{error}"))?
            }
            (None, Some(weights_dir)) => {
                let resolver =
                    NamedBlobResolver::from_dir(weights_dir).map_err(|error| anyhow!("{error}"))?;
                SystemVerilogBackend
                    .lower_execution_plan(&plan, &resolver, config)
                    .map_err(|error| anyhow!("{error}"))?
            }
            (None, None) => unreachable!("checked by the caller"),
        }
    };
    let rtl_dir = dir.path(stages::RTL_DIR);
    let written = bundle
        .write_to(&rtl_dir)
        .map_err(|error| anyhow!("writing {}: {error}", rtl_dir.display()))?;
    println!(
        "emitted kernel {} ({} files) for step {}",
        bundle.metadata.kernel_id,
        written.len(),
        bundle.metadata.layer
    );
    println!(
        "  tile: simd {} k {} noc {} fold {}",
        bundle.metadata.tile.simd,
        bundle.metadata.tile.k,
        bundle.metadata.tile.noc,
        bundle.metadata.tile.fold
    );
    println!(
        "  max |accumulator| {}, {} cycles per window",
        bundle.metadata.max_abs_accumulator, bundle.metadata.kernel_cycles_per_window
    );
    if let Some(full) = &bundle.metadata.full_graph {
        // The planner's RAM10 figure and the backend's arena are different
        // measurements; both are printed so the difference is visible, and the
        // plan estimate is never presented as the generated structure's size.
        let arena_macroblocks = full
            .arena_words
            .saturating_mul(u64::from(full.word_width))
            .div_ceil(8)
            .div_ceil(1280);
        println!(
            "  full graph: input {} -> output {}, arena {} words x {} bits ({} bytes, \
             ~{arena_macroblocks} RAM10 of 10240 bits, planner reported {} RAM10 total), \
             {} MAC lanes, {} MACs/frame",
            full.input.label(),
            full.output.label(),
            full.arena_words,
            full.word_width,
            full.arena_words.saturating_mul(u64::from(full.word_width)) / 8,
            layout.resources.map_or(0, |r| r.ram10),
            full.mac_lanes,
            full.total_macs
        );
        println!(
            "  cycles: {} unstalled generated, plan estimate {}",
            full.cycles_unstalled, full.plan_cycles
        );
        for note in &full.semantic_notes {
            println!("  note: {note}");
        }
    }
    println!("  bundle digest: {}", bundle.metadata.bundle_digest);
    for file in bundle.files() {
        println!("    {} ({:?})", file.name, file.kind);
    }
    Ok(bundle)
}

fn cmd_emit(args: EmitArgs) -> Result<std::process::ExitCode> {
    let (layout, dir, config) = emission_input(
        &args.layout,
        &args.out,
        &args.graph,
        &args.weights,
        &args.module_prefix,
        args.fixture_windows,
        args.fixture_seed,
        !args.no_testbench,
    )?;
    emit_bundle(&layout, &args.graph, &args.weights, &dir, &config)?;
    ok()
}

fn cmd_simulate(args: SimulateArgs) -> Result<std::process::ExitCode> {
    let (layout, dir, config) = emission_input(
        &args.layout,
        &args.out,
        &args.graph,
        &args.weights,
        &args.module_prefix,
        args.fixture_windows,
        args.fixture_seed,
        true,
    )?;
    let bundle = emit_bundle(&layout, &args.graph, &args.weights, &dir, &config)?;
    // Verilator runs with the work directory as its cwd and receives `--Mdir`
    // as a path, so the directory must be created and absolute: a relative path
    // would resolve differently inside the child process.
    let sim_dir = dir.path(stages::SIM_DIR);
    std::fs::create_dir_all(&sim_dir).with_context(|| format!("creating {}", sim_dir.display()))?;
    let sim_dir = std::fs::canonicalize(&sim_dir)
        .with_context(|| format!("resolving {}", sim_dir.display()))?;
    match rtl::simulate_with_verilator(&bundle, &sim_dir).map_err(|error| anyhow!("{error}"))? {
        SimulationOutcome::Passed { log } => {
            println!("simulation: PASS");
            print!("{log}");
            ok()
        }
        SimulationOutcome::Failed { log, detail } => {
            println!("simulation: FAIL ({detail})");
            print!("{log}");
            bail!("generated RTL did not match the reference model")
        }
        SimulationOutcome::ToolUnavailable { tool, reason } => {
            println!("simulation: SKIP ({reason})");
            println!("  the emitted RTL is textually verified but NOT simulated");
            println!("  tool: {tool}");
            ok()
        }
    }
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: VerifyArgs) -> Result<std::process::ExitCode> {
    let dir = StageDir::new(args.dir.clone());
    match stages::verify_chain(&dir) {
        Ok(checks) => {
            print_checks(&checks, args.format)?;
            println!("verify: ok ({} checks)", checks.len());
            ok()
        }
        Err(error) => {
            bail!("{error}");
        }
    }
}

fn print_checks(checks: &[ChainCheck], format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Text => {
            for check in checks {
                println!(
                    "{:<4} {:<10} {}: {}",
                    if check.ok { "ok" } else { "FAIL" },
                    check.stage,
                    check.check,
                    check.detail
                );
            }
        }
        OutputFormat::Json => {
            let value = serde_json::json!(
                checks
                    .iter()
                    .map(|check| serde_json::json!({
                        "stage": check.stage,
                        "check": check.check,
                        "ok": check.ok,
                        "detail": check.detail,
                    }))
                    .collect::<Vec<_>>()
            );
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Weight resolver for a graph artifact, reading its blob from beside it.
fn graph_resolver<'a>(
    artifact: &'a GraphArtifact,
    graph_path: &Path,
) -> Result<GraphWeightResolver<'a>> {
    let base = graph_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    GraphWeightResolver::load(artifact, base).map_err(|error| anyhow!("{error}"))
}

fn load_graph_file(path: &Path) -> Result<GraphArtifact> {
    tessera::frontend::load_artifact(path)
        .map_err(|error| anyhow!("loading graph artifact {}: {error}", path.display()))
}

/// Copy `source` into the stage directory as `name`, unless it is already the
/// same file, and return the staged path.
fn materialize(dir: &StageDir, source: &Path, name: &str) -> Result<PathBuf> {
    let destination = dir.path(name);
    dir.ensure()
        .with_context(|| format!("creating stage directory {}", dir.root().display()))?;
    let same_file = match (
        std::fs::canonicalize(source),
        std::fs::canonicalize(&destination),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_file {
        std::fs::copy(source, &destination).with_context(|| {
            format!("copying {} to {}", source.display(), destination.display())
        })?;
    }
    Ok(destination)
}

/// Copy the weight blob named by a graph artifact next to the staged artifact.
///
/// The file name comes from the artifact's provenance and `base` is the
/// directory the artifact was read from, so a stage directory stays
/// self-contained: `emit` and `verify` never depend on where the manifest was
/// imported from.
fn copy_weight_blob(dir: &StageDir, base: &Path, weight_file: &str) -> Result<PathBuf> {
    let source = base.join(weight_file);
    let destination = dir.path(weight_file);
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let same_file = match (
        std::fs::canonicalize(&source),
        std::fs::canonicalize(&destination),
    ) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_file {
        std::fs::copy(&source, &destination).with_context(|| {
            format!(
                "copying weight blob {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(destination)
}

fn write_input_shape(dir: &StageDir, shape: InputShape) -> Result<()> {
    let text = format!(
        "{},{},{},{}\n",
        shape.to_nhwc()[0],
        shape.to_nhwc()[1],
        shape.to_nhwc()[2],
        shape.to_nhwc()[3]
    );
    let path = dir.path(stages::INPUT_SHAPE_FILE);
    dir.write_text(stages::INPUT_SHAPE_FILE, text.trim_end())
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Resolve the input shape from the flag, the stage directory, or fail.
///
/// The manifest contract carries no spatial extents, so one concrete input shape
/// is required by both the MLIR emitter and the planner. There is deliberately
/// no default: guessing a resolution would silently change every shape in the
/// graph.
fn resolve_input_shape(
    dir: &StageDir,
    flag: Option<&str>,
    graph_path: &Path,
) -> Result<InputShape> {
    if let Some(text) = flag {
        return InputShape::parse_nhwc(text).map_err(|error| anyhow!("{error}"));
    }
    let recorded = dir.path(stages::INPUT_SHAPE_FILE);
    if recorded.exists() {
        let text = std::fs::read_to_string(&recorded)
            .with_context(|| format!("reading {}", recorded.display()))?;
        return InputShape::parse_nhwc(text.trim())
            .map_err(|error| anyhow!("{error} (recorded in {})", recorded.display()));
    }
    if let Some(parent) = graph_path.parent() {
        let sibling = parent.join(stages::INPUT_SHAPE_FILE);
        if sibling.exists() {
            let text = std::fs::read_to_string(&sibling)
                .with_context(|| format!("reading {}", sibling.display()))?;
            return InputShape::parse_nhwc(text.trim())
                .map_err(|error| anyhow!("{error} (recorded in {})", sibling.display()));
        }
    }
    bail!(
        "no input shape: the manifest carries no spatial extents, so pass --input-shape N,H,W,C \
         (or record one during `tessera import --input-shape`)"
    )
}

fn write_plan(path: &Path, plan: &tessera::layout::ExecutionPlan) -> Result<()> {
    let text = plan
        .to_toml_string()
        .map_err(|error| anyhow!("serializing the execution plan: {error}"))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn format_shape(shape: InputShape) -> String {
    let [n, h, w, c] = shape.to_nhwc();
    format!("N{n} H{h} W{w} C{c}")
}
