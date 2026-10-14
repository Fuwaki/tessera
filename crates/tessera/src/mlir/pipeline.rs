//! The pinned optimization pipeline and its Melior-backed execution.
//!
//! The pass order is not arbitrary. `nngen/tools/mlir_optimizer/README.md:58-73`
//! measured two orderings on the real `enc.0.dw -> enc.0.pw` subgraph:
//!
//! * tensor-level `linalg-fuse-elementwise-ops` before bufferization moves the
//!   depthwise requantization into the pointwise channel reduction, where it can
//!   be repeated once per output channel (157,286,400 requantization multiplies
//!   instead of 2,457,600 at 640x480);
//! * fusing after loop lowering keeps the requantization at per-pixel
//!   granularity instead.
//!
//! Tessera therefore pins `linalg-generalize-named-ops`,
//! `one-shot-bufferize{bufferize-function-boundaries}`,
//! `convert-linalg-to-affine-loops`,
//! `affine-loop-fusion{mode=producer}`, `canonicalize`, `cse`, and this module
//! has no way to insert tensor-level fusion before bufferization.
//!
//! Without the `melior` feature the module still renders and validates the
//! pipeline, and [`run_pipeline`] reports [`MlirError::MeliorUnavailable`]
//! instead of pretending to optimize.

use super::artifact::{MlirError, PipelineStage};
use std::fmt;

/// Melior version this layer is written against. The crate must be pinned to
/// exactly this release, because both the pass names and the Melior API used
/// here are release specific.
pub const PINNED_MELIOR_VERSION: &str = "0.28.2";

/// The known-good pass order for the `depthwise -> requant -> pointwise` block.
pub fn default_pipeline() -> Vec<PipelineStage> {
    vec![
        PipelineStage::func("linalg-generalize-named-ops"),
        PipelineStage::with_options(
            "one-shot-bufferize",
            &[("bufferize-function-boundaries", "")],
        ),
        PipelineStage::module("convert-linalg-to-affine-loops"),
        PipelineStage::func_with("affine-loop-fusion", &[("mode", "producer")]),
        PipelineStage::module("canonicalize"),
        PipelineStage::module("cse"),
    ]
}

/// Render a pipeline as `builtin.module(...)`, grouping consecutive
/// `func.func`-nested passes.
pub fn render_pipeline(stages: &[PipelineStage]) -> Result<String, MlirError> {
    if stages.is_empty() {
        return Err(MlirError::Pipeline {
            detail: "pipeline must contain at least one pass".to_string(),
        });
    }
    let mut groups: Vec<String> = Vec::new();
    let mut func_group: Vec<String> = Vec::new();
    for stage in stages {
        validate_pass_name(&stage.pass)?;
        if stage.nested_in_func {
            func_group.push(stage.render());
        } else {
            if !func_group.is_empty() {
                groups.push(format!("func.func({})", func_group.join(",")));
                func_group.clear();
            }
            groups.push(stage.render());
        }
    }
    if !func_group.is_empty() {
        groups.push(format!("func.func({})", func_group.join(",")));
    }
    Ok(format!("builtin.module({})", groups.join(",")))
}

fn validate_pass_name(name: &str) -> Result<(), MlirError> {
    if name.is_empty() {
        return Err(MlirError::Pipeline {
            detail: "pass name must not be empty".to_string(),
        });
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(MlirError::Pipeline {
            detail: format!("pass name {name} contains characters outside [A-Za-z0-9_-]"),
        });
    }
    Ok(())
}

/// Result of running the pinned pipeline in process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizationOutcome {
    /// Module text after the pipeline ran.
    pub module: String,
    /// Pipeline that produced it, in execution order.
    pub pipeline: Vec<PipelineStage>,
}

impl fmt::Display for OptimizationOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "optimized module ({} bytes) via {} passes",
            self.module.len(),
            self.pipeline.len()
        )
    }
}

/// Whether this build can execute MLIR passes in process.
pub const fn melior_available() -> bool {
    cfg!(feature = "melior")
}

#[cfg(feature = "melior")]
mod imp {
    use super::*;
    use melior::Context;
    use melior::dialect::DialectRegistry;
    use melior::ir::Module;
    use melior::ir::operation::{OperationLike, OperationPrintingFlags};
    use melior::pass::PassManager;
    use melior::utility::{parse_pass_pipeline, register_all_dialects, register_all_passes};
    use std::sync::{Mutex, Once};

    static REGISTER: Once = Once::new();
    /// MLIR keeps process-global pass registration state and is documented as
    /// not thread safe, so every in-process compilation serializes here.
    static MLIR_LOCK: Mutex<()> = Mutex::new(());

    fn initialize() {
        REGISTER.call_once(register_all_passes);
    }

    /// Parse, verify, run the pipeline and print the module back.
    ///
    /// The input text is verified before the pipeline runs, so a malformed
    /// emission is reported as a parse/verification failure of this layer rather
    /// than as a confusing downstream pass failure.
    pub fn run_pipeline(
        source: &str,
        stages: &[PipelineStage],
    ) -> Result<OptimizationOutcome, MlirError> {
        let pipeline = render_pipeline(stages)?;
        let _guard = MLIR_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        initialize();

        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let mut module = Module::parse(&context, source).ok_or_else(|| MlirError::Melior {
            detail: "Melior could not parse the emitted module".to_string(),
        })?;
        if !module.as_operation().verify() {
            return Err(MlirError::Melior {
                detail: "emitted module does not verify (see MLIR diagnostics above)".to_string(),
            });
        }

        let manager = PassManager::new(&context);
        manager.enable_verifier(true);
        parse_pass_pipeline(manager.as_operation_pass_manager(), &pipeline).map_err(|error| {
            MlirError::Pipeline {
                detail: format!("pipeline `{pipeline}` was rejected: {error}"),
            }
        })?;
        manager
            .run(&mut module)
            .map_err(|error| MlirError::Melior {
                detail: format!("pipeline `{pipeline}` failed: {error}"),
            })?;

        // Locations are printed: the emitter's `loc("tessera:<role>#<index>")`
        // tags are the stable mapping from optimized loops back to the operators
        // that produced them, and they must survive into the persisted text for
        // the analysis (and a reader) to see.
        let module = module
            .as_operation()
            .to_string_with_flags(OperationPrintingFlags::new().enable_debug_info(true, false))
            .map_err(|error| MlirError::Melior {
                detail: format!("printing the optimized module failed: {error}"),
            })?;

        Ok(OptimizationOutcome {
            module,
            pipeline: stages.to_vec(),
        })
    }
}

#[cfg(not(feature = "melior"))]
mod imp {
    use super::*;

    /// A build without the `melior` feature validates the pipeline but refuses
    /// to claim it optimized anything.
    pub fn run_pipeline(
        _source: &str,
        stages: &[PipelineStage],
    ) -> Result<OptimizationOutcome, MlirError> {
        let pipeline = render_pipeline(stages)?;
        Err(MlirError::MeliorUnavailable {
            detail: format!(
                "this build has no `melior` feature: rebuild with `--features melior` against melior {PINNED_MELIOR_VERSION} to run `{pipeline}`"
            ),
        })
    }
}

pub use imp::run_pipeline;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pipeline_renders_the_verified_order() {
        let text = render_pipeline(&default_pipeline()).unwrap();
        assert_eq!(
            text,
            "builtin.module(func.func(linalg-generalize-named-ops),one-shot-bufferize{bufferize-function-boundaries},convert-linalg-to-affine-loops,func.func(affine-loop-fusion{mode=producer}),canonicalize,cse)"
        );
    }

    #[test]
    fn tensor_level_fusion_cannot_precede_bufferization() {
        let text = render_pipeline(&default_pipeline()).unwrap();
        assert!(!text.contains("linalg-fuse-elementwise-ops"));
        let bufferize = text.find("one-shot-bufferize").unwrap();
        let to_loops = text.find("convert-linalg-to-affine-loops").unwrap();
        let fusion = text.find("affine-loop-fusion").unwrap();
        assert!(bufferize < to_loops && to_loops < fusion);
    }

    #[test]
    fn empty_pipeline_and_bad_pass_names_are_rejected() {
        assert!(matches!(
            render_pipeline(&[]),
            Err(MlirError::Pipeline { .. })
        ));
        assert!(matches!(
            render_pipeline(&[PipelineStage::module("bad pass")]),
            Err(MlirError::Pipeline { .. })
        ));
        assert!(matches!(
            render_pipeline(&[PipelineStage::module("")]),
            Err(MlirError::Pipeline { .. })
        ));
    }

    #[test]
    fn gated_pipeline_reports_availability_truthfully() {
        if melior_available() {
            let outcome = run_pipeline("module {}", &default_pipeline());
            assert!(outcome.is_ok(), "{outcome:?}");
        } else {
            match run_pipeline("module {}", &default_pipeline()) {
                Err(MlirError::MeliorUnavailable { detail }) => {
                    assert!(detail.contains(PINNED_MELIOR_VERSION), "{detail}");
                }
                other => panic!("expected MeliorUnavailable, got {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_module_is_rejected_before_optimization() {
        if !melior_available() {
            return;
        }
        let result = run_pipeline("module { func.func @broken( }", &default_pipeline());
        assert!(
            matches!(result, Err(MlirError::Melior { .. })),
            "{result:?}"
        );
    }
}
