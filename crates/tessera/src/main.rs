//! Tessera: a hardware-aware compiler for quantized neural-network inference.
//!
//! The initial binary is intentionally small. Compiler layers will be added
//! behind stable IR boundaries rather than coupling graph optimization to RTL.

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "tessera",
    version,
    about = "Compile quantized graphs into hardware execution plans and RTL"
)]
struct Args {
    /// Print the compiler pipeline that will be implemented by this build.
    #[arg(long)]
    describe: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.describe {
        println!("graph -> MLIR optimization -> hardware plan -> RTL");
        return Ok(());
    }

    println!("tessera: no input selected; use --describe");
    Ok(())
}
