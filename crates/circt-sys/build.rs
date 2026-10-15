//! Link the static CIRCT libraries built against LLVM/MLIR 23.1.3.
//!
//! All `libCIRCT*.a` archives are merged into one archive so the linker can
//! resolve the circular references between CIRCT libraries in a single scan.
//! MLIR/LLVM symbols come from `mlir-sys`, which links after this crate.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CIRCT_PREFIX");
    let prefix = std::env::var_os("CIRCT_PREFIX")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share/circt23")))
        .expect("set CIRCT_PREFIX to the CIRCT install prefix");
    let lib = prefix.join("lib");
    let mut archives: Vec<PathBuf> = std::fs::read_dir(&lib)
        .unwrap_or_else(|e| panic!("reading {}: {e} (see docs/toolchain/circt.md)", lib.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.starts_with("libCIRCT") && n.ends_with(".a")
        })
        .collect();
    archives.sort();
    assert!(!archives.is_empty(), "no libCIRCT*.a under {}", lib.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let merged = out.join("libcirct_all.a");
    let mut script = format!("create {}\n", merged.display());
    for a in &archives {
        script.push_str(&format!("addlib {}\n", a.display()));
        println!("cargo:rerun-if-changed={}", a.display());
    }
    script.push_str("save\nend\n");
    let _ = std::fs::remove_file(&merged);
    let mut child = Command::new("ar")
        .arg("-M")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("spawn ar");
    use std::io::Write;
    child.stdin.take().unwrap().write_all(script.as_bytes()).unwrap();
    assert!(child.wait().unwrap().success(), "ar -M failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=circt_all");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
