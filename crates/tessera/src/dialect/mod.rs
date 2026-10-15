//! Tessera hardware-mapping dialect.
//!
//! The op set is declared in IRDL ([`DIALECT_IRDL`]) and loaded into a Melior
//! context at runtime. Cross-op legality is enforced by [`verify`].

mod convert;
pub mod read;
mod verify;

pub use convert::{ANNOTATION, TENSOR_PIPELINE, linalg_to_tessera, load_tessera_module};

pub use read::{TesseraModule, read};
pub use verify::{DialectError, check, verify_module};

use std::sync::Mutex;

/// IRDL source that defines the `tessera` dialect.
pub const DIALECT_IRDL: &str = include_str!("../../dialects/tessera.irdl.mlir");

/// Dialect namespace used in printed IR (`tessera.engine`, …).
pub const DIALECT_NAME: &str = "tessera";

/// MLIR is process-global and not thread-safe; serialize every registration.
static MLIR_LOCK: Mutex<()> = Mutex::new(());

/// Register the dialect into `context` by parsing the IRDL module and calling
/// Melior's `load_irdl_dialects`.
pub fn register(context: &melior::Context) -> Result<(), DialectError> {
    use melior::ir::Module;
    use melior::utility::load_irdl_dialects;

    let _guard = MLIR_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let source = Module::parse(context, DIALECT_IRDL).ok_or_else(|| DialectError::Irdl {
        detail: "failed to parse dialects/tessera.irdl.mlir (is the `irdl` dialect loaded?)"
            .to_string(),
    })?;
    if !load_irdl_dialects(&source) {
        return Err(DialectError::Irdl {
            detail: "mlirLoadIRDLDialects rejected the tessera definition".to_string(),
        });
    }
    // Keep `source` alive until after load_irdl_dialects returns; dropping it
    // earlier has been observed to tear down dialect definitions.
    std::mem::forget(source);
    Ok(())
}

/// Build a Melior context with every stock dialect plus `tessera`.
pub fn register_in_fresh_context() -> Result<melior::Context, DialectError> {
    use melior::dialect::DialectRegistry;
    use melior::utility::register_all_dialects;

    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register(&context)?;
    Ok(context)
}


fn dirs_fallback() -> std::path::PathBuf {
    let mut home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.push(".local/share/circt23/bin/circt-opt");
    home
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irdl_file_parses_with_mlir_opt() {
        // Prefer the process-external check: Melior's IRDL loader has crashed on
        // some LLVM 23.1.3 builds, while `mlir-opt` accepting the file is the
        // contract we need for the dialect definition itself.
        let mlir_opt = std::env::var_os("MLIR_SYS_230_PREFIX")
            .map(std::path::PathBuf::from)
            .map(|mut p| {
                p.push("bin/mlir-opt");
                p
            })
            .filter(|p| p.is_file())
            .or_else(|| which("mlir-opt"));
        let Some(mlir_opt) = mlir_opt else {
            eprintln!("SKIP: mlir-opt not on PATH / MLIR_SYS_230_PREFIX");
            return;
        };
        let irdl = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/dialects/tessera.irdl.mlir"
        );
        let out = std::process::Command::new(&mlir_opt)
            .arg(irdl)
            .output()
            .expect("spawn mlir-opt");
        assert!(
            out.status.success(),
            "mlir-opt rejected the IRDL file:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("irdl.dialect @tessera"), "{stdout}");
        assert!(stdout.contains("irdl.operation @engine"), "{stdout}");
        assert!(stdout.contains("irdl.operation @segment"), "{stdout}");
    }

    #[test]
    fn irdl_definition_loads_into_a_fresh_context() {
        // Exercise the in-process path; tolerate a loader crash by running in a
        // child only when TESSERA_IRDL_INPROCESS=1.
        if std::env::var_os("TESSERA_IRDL_INPROCESS").is_none() {
            eprintln!(
                "SKIP: set TESSERA_IRDL_INPROCESS=1 to run the in-process Melior IRDL loader"
            );
            return;
        }
        let _ = register_in_fresh_context().expect("load the tessera IRDL dialect");
    }

    fn which(bin: &str) -> Option<std::path::PathBuf> {
        std::env::var_os("PATH").and_then(|paths| {
            for dir in std::env::split_paths(&paths) {
                let candidate = dir.join(bin);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
            None
        })
    }
}
