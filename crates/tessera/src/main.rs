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
use tessera::lower::{InputShape, LowerReport, lower_graph};
use tessera::mlir::{self, MlirArtifact, MlirOptions, SourceGraph};
use tessera::planner::{self, Policy, ProvenanceInput};
use tessera::rtl::{self, RtlConfig, SimulationOutcome, SystemVerilogBackend};
use tessera::stages::{self, ArtifactKind, ChainCheck, StageDir};
use tessera::target::Ti60Target;
use tessera::weights::{GraphWeightResolver, NamedBlobResolver};

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
    #[arg(long, value_name = "FILE", conflicts_with = "primitive")]
    graph: Option<PathBuf>,
    /// Lowered primitive graph (`PlanInput` JSON) to solve directly.
    #[arg(long, value_name = "FILE", conflicts_with = "graph")]
    primitive: Option<PathBuf>,
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
    let (input, report, source_label, source_file) = match (&args.graph, &args.primitive) {
        (Some(graph), None) => {
            let artifact = load_graph_file(graph)?;
            let staged = materialize(&dir, graph, stages::GRAPH_FILE)?;
            copy_weight_blob(
                &dir,
                graph.parent().unwrap_or_else(|| Path::new(".")),
                artifact.provenance().weight_file.as_str(),
            )?;
            let shape = resolve_input_shape(&dir, args.input_shape.as_deref(), &staged)?;
            write_input_shape(&dir, shape)?;
            let (plan, report) =
                lower_graph(artifact.graph(), shape).map_err(|error| anyhow!("{error}"))?;
            (
                plan,
                report,
                graph.display().to_string(),
                stages::GRAPH_FILE,
            )
        }
        (None, Some(primitive)) => {
            let text = std::fs::read_to_string(primitive)
                .with_context(|| format!("reading {}", primitive.display()))?;
            let plan = stages::parse_primitive(&text).map_err(|error| anyhow!("{error}"))?;
            let canonical =
                serde_json::to_string_pretty(&plan).context("serializing the primitive graph")?;
            dir.write_text(stages::PRIMITIVE_FILE, &canonical)
                .with_context(|| {
                    format!("writing {}", dir.path(stages::PRIMITIVE_FILE).display())
                })?;
            (
                plan,
                LowerReport::default(),
                primitive.display().to_string(),
                stages::PRIMITIVE_FILE,
            )
        }
        (None, None) => bail!("select an input with --graph <graph.tgraph> or --primitive <file>"),
        (Some(_), Some(_)) => unreachable!("clap rejects --graph with --primitive"),
    };
    let source_hash = dir.sha256(source_file)?;

    let primitive_path = dir.path(stages::PRIMITIVE_FILE);
    if source_file != stages::PRIMITIVE_FILE {
        let primitive_json = serde_json::to_string_pretty(&input)
            .context("serializing the lowered primitive graph")?;
        dir.write_text(stages::PRIMITIVE_FILE, &primitive_json)
            .with_context(|| format!("writing {}", primitive_path.display()))?;
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

    for note in &report.notes {
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
            println!("layout: INFEASIBLE, no candidate satisfied the target and policy");
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
            bail!(
                "no feasible layout under {} and {}; the rejected-candidate summary is in {}",
                dir.path(stages::TARGET_FILE).display(),
                dir.path(stages::POLICY_FILE).display(),
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
                "  search: {} explored, {} expanded, truncated {}",
                artifact.search.explored, artifact.search.expanded_nodes, artifact.search.truncated
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

    let bundle = match (graph, weights) {
        (Some(graph_path), _) => {
            let artifact = load_graph_file(graph_path)?;
            let base = graph_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            let resolver =
                GraphWeightResolver::load(&artifact, base).map_err(|error| anyhow!("{error}"))?;
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
