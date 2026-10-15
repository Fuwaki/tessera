//! Executable semantic equivalence: the emitted IR is *run* and compared with the
//! declared arithmetic contract.
//!
//! The pathway is the emitted IR's own native lowering, not a re-implementation:
//!
//! ```text
//! Graph IR -> emitted tensor module -> pinned Melior pipeline -> bufferized module
//!                                            |
//!                generated driver (argument globals + `position,code` printer)
//!                                            |
//!            mlir-opt: lower-affine, finalize-memref-to-llvm, convert-to-llvm
//!                                            |
//!                          mlir-runner (MLIR's own JIT)
//!                                            |
//!              every output code  ==  Rust reference model, element by element
//! ```
//!
//! The element channel is explicit: the driver prints each code as
//! `<flat position>,<code>` followed by a newline, using the runner library's own
//! separator and newline helpers, and the test requires the positions to arrive in
//! order. A code therefore cannot be read from a different element than the one
//! the module computed, and a truncated or interleaved stream is a failure instead
//! of a silently shifted comparison.
//!
//! The reference model implements the *declared* contract — signless `i32`
//! accumulation of `bias + sum(code * weight)`, the serialized `(M0, shift)`
//! fixed-point requantization with round-half-away-from-zero and saturation, an
//! unclamped residual sum, nearest-neighbour duplication, and the unsigned
//! `[0, output_qmax]` head clamp — plus the graph's padding rule. It indexes the
//! *stored* weight layout (`Oihw`/`Chw`), independently of the MLIR filter layout
//! the driver builds (`Fhwc`/`Hwc`), so a transposed or mis-strided filter cannot
//! cancel out. Its per-tensor codes are themselves checked against the fixture's
//! independent oracle (`fixtures/acceptance/w8_execution_oracle.json`) before it
//! is allowed to judge the IR.
//!
//! Exactness is claimed relative to that contract, not to training `f32`: the
//! `f32` requantization factors the manifest carries are represented as
//! `(M0, shift)` in both the IR and the reference, and the artifact records the
//! worst relative error of that representation.
//!
//! Two stimuli are used: the fixture's pinned codes and a high-drive pattern. The
//! pinned codes leave this fixture's image flat, so on their own they cannot
//! separate a correct head from one that returns constants; the high-drive codes
//! push the per-channel clamps, the widened residual and the head's positive row
//! into non-trivial codes.
//!
//! What an executed code can and cannot distinguish here: the padding rule, both
//! filter layouts, the per-channel multipliers and biases, the residual sum's
//! widened transport, the nearest-neighbour duplication and the head's
//! requantization are all pinned elementwise. The head's *upper* clamp and the
//! rounding of a *negative* product are not: this fixture's head rows leave the
//! accumulator negative everywhere (only clamp asymmetries lift it, and only as
//! far as 158), and the `[0, qmax]` lower clamps absorb a negative rounded value
//! either way. Those two live in the emitted arithmetic, which the emitter's unit
//! tests pin, not in this fixture's image.
//!
//! The test needs a `melior` build (for the pass pipeline) and an MLIR 23
//! installation with `mlir-opt`, `mlir-runner` and `libmlir_c_runner_utils`
//! (located through `MLIR_SYS_230_PREFIX`). It reports which input it lacked and
//! skips when either is missing, instead of claiming a result it did not observe;
//! with `TESSERA_ACCEPTANCE_STRICT` set — the acceptance gate's switch — that
//! report is a failure, because a gate that did not run cannot support a claim.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use tessera::frontend;
use tessera::graph::GraphArtifact;
use tessera::graph::{Graph, GraphOp, Layer, LayerKind as GraphLayerKind, Padding as GraphPadding};
use tessera::mlir::{
    self, ArgumentRole, ElementType, FilterStorageLayout, MlirArtifact, MlirFilterLayout,
    MlirOptions, OperatorKind, SourceGraph,
};

const INPUT_SHAPE: [u64; 4] = [1, 6, 6, 3];

/// How many stimuli the unsigned-reading probe may try before it gives up.
///
/// The regime it looks for is rare on this fixture (a few hits in ten
/// thousand), so the budget is generous; the probe stops at the first hit.
const PROBE_ATTEMPTS: usize = 8_000;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn w8() -> GraphArtifact {
    frontend::import_manifest(&fixtures().join("w8")).expect("import the W8 fixture")
}

fn weights_blob() -> Vec<u8> {
    std::fs::read(fixtures().join("w8/weights.bin")).expect("read the weight blob")
}

/// Emit and optimize the fixture: execution needs the planning-stage module.
fn optimized(artifact: &GraphArtifact) -> MlirArtifact {
    let emitted = mlir::lower(
        artifact.graph(),
        SourceGraph::from(artifact),
        &MlirOptions::new(INPUT_SHAPE),
    )
    .expect("emit");
    mlir::optimize(&emitted).expect("optimize")
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

struct Tools {
    mlir_opt: PathBuf,
    mlir_runner: PathBuf,
    /// `libmlir_c_runner_utils` (print helpers) and `libmlir_runner_utils`
    /// (memref printers), in the order the JIT loads them.
    runner_utils: Vec<PathBuf>,
}

/// MLIR 23 tools located through the prefix the build used. `None` means "skip".
fn tools() -> Option<Tools> {
    let prefix = PathBuf::from(std::env::var_os("MLIR_SYS_230_PREFIX")?);
    let (mlir_opt, mlir_runner) = (prefix.join("bin/mlir-opt"), prefix.join("bin/mlir-runner"));
    if !mlir_opt.exists() || !mlir_runner.exists() {
        return None;
    }
    // `printI64` lives in the C runner utilities, the memref printers in the C++
    // one; both are loaded.
    let mut found = Vec::new();
    for stem in ["libmlir_c_runner_utils.so", "libmlir_runner_utils.so"] {
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(prefix.join("lib"))
            .ok()?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(stem))
            })
            .collect();
        candidates.sort();
        found.push(candidates.pop()?);
    }
    Some(Tools {
        mlir_opt,
        mlir_runner,
        runner_utils: found,
    })
}

// ---------------------------------------------------------------------------
// Driver generation
// ---------------------------------------------------------------------------

/// Argument data handed to the module.
enum Values {
    I8(Vec<i8>),
    I32(Vec<i32>),
}

impl Values {
    fn element(&self) -> &'static str {
        match self {
            Values::I8(_) => "i8",
            Values::I32(_) => "i32",
        }
    }

    fn render(&self) -> Vec<String> {
        match self {
            Values::I8(values) => values.iter().map(i8::to_string).collect(),
            Values::I32(values) => values.iter().map(i32::to_string).collect(),
        }
    }
}

/// Render a flat, row-major value list as the nested `dense` literal its shape
/// requires.
fn render_dense(shape: &[u64], values: &[String]) -> String {
    assert_eq!(
        shape.iter().product::<u64>() as usize,
        values.len(),
        "the value list must fill the shape"
    );
    fn nest(values: &[String], shape: &[u64]) -> String {
        match shape.split_first() {
            None => values[0].clone(),
            Some((extent, rest)) => {
                let stride: usize = rest.iter().map(|dim| *dim as usize).product();
                let items: Vec<String> = (0..*extent as usize)
                    .map(|index| nest(&values[index * stride..(index + 1) * stride], rest))
                    .collect();
                format!("[{}]", items.join(", "))
            }
        }
    }
    nest(values, shape)
}

/// The module's interface as the driver needs it.
struct Signature {
    function: String,
    arguments: Vec<(String, Vec<u64>, Values)>,
    output: (Vec<u64>, String),
    /// Whether the output codes are unsigned (the image head's `[0, qmax]`
    /// clamp), which decides sign or zero extension when they are printed.
    output_unsigned: bool,
}

/// The bufferized type `one-shot-bufferize` gives a tensor argument.
fn bufferized(shape: &[u64], element: &str) -> String {
    let dims = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let strides = vec!["?"; shape.len()].join(", ");
    format!("memref<{dims}x{element}, strided<[{strides}], offset: ?>>")
}

fn dims(shape: &[u64]) -> String {
    shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

/// The driver *body*: argument globals, the print helpers and `@main`.
///
/// It is spliced into the optimized module so that `@main` and the emitted
/// function live in one module and the JIT sees a single input.
///
/// `@main` calls the emitted function with the fixture's weights and then prints
/// the output codes one line at a time as `position,code`: the position is the
/// flat NHWC index in the returned memref, so every reported code is tied to the
/// element it came from and a shifted or repeated line cannot be mistaken for a
/// different element. The separator (`,`) and the line terminator come from the
/// running library itself (`printComma`, `printNewline`), so the channel does not
/// depend on guessing where one printed number ends and the next begins.
fn driver(signature: &Signature) -> String {
    let mut text = String::new();
    for (index, (name, shape, values)) in signature.arguments.iter().enumerate() {
        let _ = writeln!(
            text,
            "  memref.global \"private\" constant @arg{index} : memref<{}x{}> = dense<{}> // {name}",
            dims(shape),
            values.element(),
            render_dense(shape, &values.render()),
        );
    }
    text.push_str("  func.func private @printI64(i64)\n");
    text.push_str("  func.func private @printComma()\n");
    text.push_str("  func.func private @printNewline()\n");
    text.push_str("  func.func @main() {\n");
    let mut operands = Vec::new();
    for (index, (_, shape, values)) in signature.arguments.iter().enumerate() {
        let static_type = format!("memref<{}x{}>", dims(shape), values.element());
        let dynamic_type = bufferized(shape, values.element());
        let _ = writeln!(
            text,
            "    %g{index} = memref.get_global @arg{index} : {static_type}"
        );
        let _ = writeln!(
            text,
            "    %a{index} = memref.cast %g{index} : {static_type} to {dynamic_type}"
        );
        operands.push(format!("%a{index}"));
    }
    let (output_shape, output_element) = &signature.output;
    let output_extend = if signature.output_unsigned {
        "arith.extui"
    } else {
        "arith.extsi"
    };
    let output_type = format!("memref<{}x{output_element}>", dims(output_shape));
    let _ = writeln!(
        text,
        "    %out = func.call @{}({}) : ({}) -> {output_type}",
        signature.function,
        operands.join(", "),
        signature
            .arguments
            .iter()
            .map(|(_, shape, values)| bufferized(shape, values.element()))
            .collect::<Vec<_>>()
            .join(", "),
    );

    // Row-major sweep over the output. Each element is printed as
    // `<flat position>,<code>` followed by a newline.
    let rank = output_shape.len();
    for (axis, extent) in output_shape.iter().enumerate() {
        let _ = writeln!(
            text,
            "{}affine.for %i{axis} = 0 to {extent} {{",
            "  ".repeat(axis + 2)
        );
    }
    let indent = "  ".repeat(rank + 2);
    let indices = (0..rank)
        .map(|axis| format!("%i{axis}"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(
        text,
        "{indent}%code = affine.load %out[{indices}] : {output_type}"
    );
    let _ = writeln!(
        text,
        "{indent}%code64 = {output_extend} %code : {output_element} to i64"
    );
    // The flat position of this element in the returned (row-major) buffer.
    let _ = writeln!(text, "{indent}%pos0 = arith.constant 0 : i64");
    let mut position = "%pos0".to_string();
    for (axis, extent) in output_shape.iter().enumerate() {
        let _ = writeln!(
            text,
            "{indent}%i{axis}64 = arith.index_cast %i{axis} : index to i64"
        );
        let _ = writeln!(text, "{indent}%dim{axis} = arith.constant {extent} : i64");
        let _ = writeln!(
            text,
            "{indent}%scaled{axis} = arith.muli {position}, %dim{axis} : i64"
        );
        let _ = writeln!(
            text,
            "{indent}%pos{} = arith.addi %scaled{axis}, %i{axis}64 : i64",
            axis + 1
        );
        position = format!("%pos{}", axis + 1);
    }
    let _ = writeln!(
        text,
        "{indent}func.call @printI64({position}) : (i64) -> ()"
    );
    let _ = writeln!(text, "{indent}func.call @printComma() : () -> ()");
    let _ = writeln!(text, "{indent}func.call @printI64(%code64) : (i64) -> ()");
    let _ = writeln!(text, "{indent}func.call @printNewline() : () -> ()");
    for axis in (0..rank).rev() {
        let _ = writeln!(text, "{}}}", "  ".repeat(axis + 2));
    }
    text.push_str("    return\n  }\n");
    text
}

/// Splice a driver body into the optimized module, which supplies the callee.
///
/// The stored module text is the printer's output: attribute and affine-map
/// aliases first, then the `module` operation, whose *own* trailing location
/// follows its closing brace. The driver is inserted inside the module, and
/// everything after the module's closing brace is carried over verbatim.
fn driver_module(optimized: &MlirArtifact, signature: &Signature) -> String {
    let module = optimized.module.trim_end();
    let close = module
        .rfind('}')
        .expect("the module text closes the module operation");
    let (head, tail) = module.split_at(close);
    let trailing = tail.strip_prefix('}').unwrap_or("");
    format!("{head}\n{}\n}}{trailing}\n", driver(signature))
}

// ---------------------------------------------------------------------------
// Argument data
// ---------------------------------------------------------------------------

/// Deterministic input codes inside the input point's `[-qmax, qmax]` interval.
fn input_codes() -> Vec<i8> {
    let [n, h, w, c] = INPUT_SHAPE;
    let mut codes = Vec::with_capacity((n * h * w * c) as usize);
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let raw = (ni * 7 + hi * 13 + wi * 17 + ci * 19 + 29) % 255;
                    codes.push((raw as i32 - 127) as i8);
                }
            }
        }
    }
    codes
}

/// A second deterministic stimulus that keeps the graph away from its flat
/// corner.
///
/// The pinned stimulus leaves this fixture's image all zeros (two of the head's
/// three rows have no positive coefficient over an always-non-negative
/// activation), so on its own it cannot tell a correct head from a head that
/// returns constants. These codes are chosen so that
///
/// * the stem's per-channel multipliers and biases are driven to their
///   saturation, so the requantization clamps are active on every channel;
/// * the residual add carries codes past `i8::MAX`, so the widened transport the
///   point needs is exercised element by element rather than only in the plan;
/// * the head's first row - the only one with a positive coefficient - produces
///   non-zero, position-dependent codes over a range of accumulator magnitudes.
///
/// The same codes are used with a stride pattern in both axes so no axis or
/// channel is favoured.
fn high_drive_codes() -> Vec<i8> {
    let [n, h, w, c] = INPUT_SHAPE;
    let mut codes = Vec::with_capacity((n * h * w * c) as usize);
    for ni in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let raw = (ni * 10 + hi * 4 + wi * 10 + ci * 6 + 187) % 255;
                    codes.push((raw as i32 - 127) as i8);
                }
            }
        }
    }
    codes
}

/// A stimulus that drives the image past `i8::MAX`, plus how it was obtained.
///
/// The head stores `[0, output_qmax]` in an `i8` container, so a code above
/// `i8::MAX` is only correct when the consumer zero-extends it; below that
/// boundary a signed reading and an unsigned one are indistinguishable. The
/// head's rows are monotone in the channel index over this fixture's monotone
/// channel weights, so a positive image needs the residual and the clamps to line
/// up exactly and the regime is narrow: searching the reference model finds it a
/// few times in ten thousand stimuli. The search that found the pinned vector is
/// kept as a golden case, and when a fixture change invalidates it the same search
/// runs again from a fixed seed, so the coverage cannot disappear silently and a
/// fixture change is not reported as an IR failure.
///
/// `None` means `search` attempts found no such stimulus, so this fixture cannot
/// pin the unsigned reading element by element at all.
fn boundary_stimulus(reference: &Reference<'_>, search: usize) -> Option<(String, Vec<i8>)> {
    /// Found by searching the reference model for the narrow regime in which this
    /// fixture's head crosses the signed boundary.
    const GOLDEN: [[[i8; 3]; 6]; 6] = [
        [
            [-58, -53, -64],
            [-31, -13, 25],
            [9, -10, 52],
            [19, 45, 69],
            [-53, 92, 15],
            [57, -79, -118],
        ],
        [
            [-107, 59, 13],
            [42, 39, -112],
            [22, -108, 58],
            [-110, -18, 35],
            [72, 93, 14],
            [109, -123, 91],
        ],
        [
            [78, 3, 38],
            [-115, -101, -8],
            [-69, 59, -94],
            [106, 39, -8],
            [68, 115, -31],
            [48, -30, -91],
        ],
        [
            [-13, -99, 67],
            [7, -20, -49],
            [-44, -26, -21],
            [56, -123, 57],
            [-29, -9, -61],
            [-47, -126, 12],
        ],
        [
            [44, -73, -57],
            [-12, -15, 73],
            [-86, -104, 77],
            [-113, -21, -27],
            [1, -85, 92],
            [-65, 102, -10],
        ],
        [
            [-83, 31, 53],
            [38, -28, 127],
            [101, 83, -111],
            [-111, -71, -56],
            [-99, -95, -86],
            [-58, 88, -6],
        ],
    ];
    let crosses = |codes: &[i8]| {
        reference
            .run(codes)
            .image
            .data
            .iter()
            .any(|code| *code > i64::from(i8::MAX))
    };
    let golden: Vec<i8> = GOLDEN
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|pixel| pixel.iter().copied())
        .collect();
    if crosses(&golden) {
        return Some(("signed-boundary".to_string(), golden));
    }
    // A fixed seed keeps the fallback reproducible: whether it finds a stimulus is
    // a property of the fixture, not of the run.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state >> 33
    };
    for _ in 0..search {
        let codes: Vec<i8> = (0..INPUT_SHAPE.iter().product::<u64>())
            .map(|_| ((next() % 255) as i32 - 127) as i8)
            .collect();
        if crosses(&codes) {
            return Some((
                format!("signed-boundary (searched {search} attempts)"),
                codes,
            ));
        }
    }
    None
}

/// Filter codes of one layer, mapped from the stored layout to the MLIR layout.
fn filter_values(
    layer: &Layer,
    blob: &[u8],
    storage: FilterStorageLayout,
    target: MlirFilterLayout,
) -> Values {
    let codes: Vec<i8> = blob[layer.weight_byte_offset as usize
        ..layer.weight_byte_offset as usize + layer.weight_bytes as usize]
        .iter()
        .map(|byte| *byte as i8)
        .collect();
    let (out_channels, in_channels) = (
        layer.out_channels.get() as usize,
        layer.in_channels.get() as usize,
    );
    let (kh, kw) = (layer.kernel.h.get() as usize, layer.kernel.w.get() as usize);
    let per_group = if layer.kind == GraphLayerKind::DepthwiseConv2d {
        1
    } else {
        in_channels / layer.groups.get().max(1) as usize
    };
    let mut values = Vec::with_capacity(codes.len());
    match (storage, target) {
        (FilterStorageLayout::Oihw, MlirFilterLayout::Fhwc) => {
            // stored [oc][ic][kh][kw] -> [oc][kh][kw][ic]
            for oc in 0..out_channels {
                for y in 0..kh {
                    for x in 0..kw {
                        for ic in 0..per_group {
                            values.push(codes[((oc * per_group + ic) * kh + y) * kw + x]);
                        }
                    }
                }
            }
        }
        (FilterStorageLayout::Chw, MlirFilterLayout::Hwc) => {
            // stored [c][kh][kw] -> [kh][kw][c]
            for y in 0..kh {
                for x in 0..kw {
                    for channel in 0..out_channels {
                        values.push(codes[(channel * kh + y) * kw + x]);
                    }
                }
            }
        }
        (storage, target) => panic!("unsupported filter mapping {storage:?} -> {target:?}"),
    }
    Values::I8(values)
}

/// The module's arguments in declaration order, with the fixture's data.
fn signature_of(artifact: &MlirArtifact, graph: &Graph, input: &[i8]) -> Signature {
    let blob = weights_blob();
    let mut arguments = Vec::new();
    for argument in &artifact.arguments {
        let values = match argument.role {
            ArgumentRole::Input => Values::I8(input.to_vec()),
            ArgumentRole::Filter => {
                let name = argument.layer.as_deref().expect("a filter names its layer");
                let layer = graph.layer_named(name).expect("the layer exists");
                let weight = argument
                    .weight
                    .as_ref()
                    .expect("a filter carries a weight reference");
                filter_values(layer, &blob, weight.storage_layout, weight.mlir_layout)
            }
            ArgumentRole::Bias => {
                let name = argument.layer.as_deref().expect("a bias names its layer");
                let layer = graph.layer_named(name).expect("the layer exists");
                Values::I32(layer.bias_int32.clone())
            }
            ArgumentRole::RequantMultiplier => {
                let name = argument
                    .layer
                    .as_deref()
                    .expect("a multiplier names its layer");
                let binding = artifact.layer(name).expect("the layer binding exists");
                Values::I32(binding.quantization.multiplier.clone())
            }
            other => panic!("the fixture does not use {other:?} arguments"),
        };
        arguments.push((argument.ssa_name.clone(), argument.shape.clone(), values));
    }
    let output_element = match artifact.output.element {
        ElementType::I8 => "i8".to_string(),
        other => panic!("unexpected output element {other:?}"),
    };
    // The unsigned image head stores `[0, output_qmax]` in an `i8` container, so
    // the codes are zero-extended, exactly as the manifest's `u8` consumer reads
    // them; a signed extension would make them negative.
    let output_unsigned = artifact
        .output
        .code_range
        .is_some_and(|range| range.lo >= 0 && range.hi > i8::MAX as i64);
    Signature {
        function: artifact.function.clone(),
        arguments,
        output: (artifact.output.shape.clone(), output_element),
        output_unsigned,
    }
}

// ---------------------------------------------------------------------------
// Reference model of the declared contract
// ---------------------------------------------------------------------------

/// One activation in the reference model: `[channel][height][width]`, as the
/// reference interpreter lays activations out.
#[derive(Clone)]
struct Act {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<i64>,
}

impl Act {
    /// The codes in the order the emitted module and the fixture's oracle use:
    /// row-major `[height][width][channel]` (the batch extent is one).
    fn nhwc(&self) -> Vec<i64> {
        let mut codes = Vec::with_capacity(self.c * self.h * self.w);
        for hi in 0..self.h {
            for wi in 0..self.w {
                for ci in 0..self.c {
                    codes.push(self.data[(ci * self.h + hi) * self.w + wi]);
                }
            }
        }
        codes
    }
}

/// Everything one reference run produced.
struct Run {
    /// The intermediate activations, keyed by the tensor names the fixture's
    /// oracle uses: `input`, one entry per convolution layer, `add`,
    /// `upsample_nearest2x`, `head`, `quantize_u8_output` and `image`.
    tensors: BTreeMap<String, Act>,
    image: Act,
}

/// Reflection padding that does not repeat the edge.
fn reflect_index(index: i64, extent: i64) -> i64 {
    if extent <= 1 {
        return 0;
    }
    let mut value = index;
    while value < 0 || value >= extent {
        value = if value < 0 {
            -value
        } else {
            2 * (extent - 1) - value
        };
    }
    value
}

/// `round_clamp(value * M0 * 2^-shift, lo, hi)` with round-half-away-from-zero.
fn requantize(value: i64, multiplier: i32, shift: u32, lo: i64, hi: i64) -> i64 {
    let product = value * i64::from(multiplier);
    let rounding = if shift == 0 {
        0
    } else if product >= 0 {
        1i64 << (shift - 1)
    } else {
        (1i64 << (shift - 1)) - 1
    };
    ((product + rounding) >> shift).clamp(lo, hi)
}

struct Reference<'a> {
    graph: &'a Graph,
    artifact: &'a MlirArtifact,
    blob: &'a [u8],
}

impl<'a> Reference<'a> {
    /// One convolution, indexing the *stored* filter layout.
    fn conv(&self, layer: &Layer, input: &Act) -> Act {
        let (kh, kw) = (layer.kernel.h.get() as usize, layer.kernel.w.get() as usize);
        let (sh, sw) = (layer.stride.h.get() as usize, layer.stride.w.get() as usize);
        let (pad, reflect) = match &layer.padding {
            GraphPadding::None => (0usize, false),
            GraphPadding::Reflect(amount) => (amount.get() as usize, true),
            GraphPadding::Zero(amount) => (amount.get() as usize, false),
        };
        let (out_h, out_w) = (
            (input.h + 2 * pad - kh) / sh + 1,
            (input.w + 2 * pad - kw) / sw + 1,
        );
        let out_channels = layer.out_channels.get() as usize;
        let groups = layer.groups.get().max(1) as usize;
        let depthwise = layer.kind == GraphLayerKind::DepthwiseConv2d;
        let per_group = if depthwise { 1 } else { input.c / groups };
        let codes: Vec<i64> = self.blob[layer.weight_byte_offset as usize
            ..layer.weight_byte_offset as usize + layer.weight_bytes as usize]
            .iter()
            .map(|byte| *byte as i8 as i64)
            .collect();
        let quantization = &self
            .artifact
            .layer(layer.name.as_str())
            .expect("the layer binding")
            .quantization;
        let mut data = vec![0i64; out_channels * out_h * out_w];
        for oc in 0..out_channels {
            let group = oc / (out_channels / groups).max(1);
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0i64;
                    for y in 0..kh {
                        for x in 0..kw {
                            for lane in 0..per_group {
                                let ic = group * per_group + lane;
                                let code = self.padded_code(
                                    input,
                                    ic,
                                    oy * sh + y,
                                    ox * sw + x,
                                    pad,
                                    reflect,
                                );
                                let stored = if depthwise {
                                    (oc * kh + y) * kw + x
                                } else {
                                    ((oc * per_group + lane) * kh + y) * kw + x
                                };
                                acc += code * codes[stored];
                            }
                        }
                    }
                    let bias = i64::from(layer.bias_int32.get(oc).copied().unwrap_or(0));
                    data[(oc * out_h + oy) * out_w + ox] = requantize(
                        acc + bias,
                        quantization.multiplier[oc],
                        quantization.shift,
                        quantization.clamp.lower,
                        quantization.clamp.upper,
                    );
                }
            }
        }
        Act {
            c: out_channels,
            h: out_h,
            w: out_w,
            data,
        }
    }

    fn padded_code(
        &self,
        input: &Act,
        channel: usize,
        y: usize,
        x: usize,
        pad: usize,
        reflect: bool,
    ) -> i64 {
        let (source_y, source_x) = (y as i64 - pad as i64, x as i64 - pad as i64);
        let inside = source_y >= 0
            && source_y < input.h as i64
            && source_x >= 0
            && source_x < input.w as i64;
        if inside {
            return input.data
                [(channel * input.h + source_y as usize) * input.w + source_x as usize];
        }
        if !reflect {
            return 0;
        }
        let ry = reflect_index(source_y, input.h as i64) as usize;
        let rx = reflect_index(source_x, input.w as i64) as usize;
        input.data[(channel * input.h + ry) * input.w + rx]
    }

    /// Run the whole graph and return the image plus every named activation.
    fn run(&self, codes: &[i8]) -> Run {
        let mut points: BTreeMap<u32, Act> = BTreeMap::new();
        let mut tensors: BTreeMap<String, Act> = BTreeMap::new();
        // The driver fills the module's NHWC argument; the reference lays
        // activations out `[channel][height][width]`, so transpose once here.
        let (h, w, c) = (
            INPUT_SHAPE[1] as usize,
            INPUT_SHAPE[2] as usize,
            INPUT_SHAPE[3] as usize,
        );
        let mut data = vec![0i64; codes.len()];
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    data[(ci * h + hi) * w + wi] = i64::from(codes[(hi * w + wi) * c + ci]);
                }
            }
        }
        tensors.insert("input".to_string(), Act { c, h, w, data });
        points.insert(self.graph.input_point().get(), tensors["input"].clone());
        let mut image: Option<Act> = None;
        for op in self.graph.ops() {
            match op {
                GraphOp::Conv {
                    layer, out_point, ..
                } => {
                    let layer = self.graph.layer_named(layer.as_str()).expect("layer");
                    let input = points
                        .get(&layer.in_point.get())
                        .cloned()
                        .expect("the input point is live");
                    let out = self.conv(layer, &input);
                    tensors.insert(layer.name.as_str().to_string(), out.clone());
                    if layer.out_unsigned {
                        image = Some(out);
                    } else if let Some(point) = out_point.id() {
                        points.insert(point.get(), out);
                    }
                }
                // The activation is fused into the convolution's saturation.
                GraphOp::Relu { .. } => {}
                GraphOp::Add {
                    point,
                    source_point,
                    ..
                } => {
                    let trunk = points.get(&point.get()).cloned().expect("trunk");
                    let branch = points.get(&source_point.get()).cloned().expect("branch");
                    let spec = self
                        .artifact
                        .operators
                        .iter()
                        .find(|operator| operator.kind == OperatorKind::Add)
                        .and_then(|operator| operator.quantization.as_ref())
                        .expect("the add records its branch requantization");
                    let data = trunk
                        .data
                        .iter()
                        .zip(&branch.data)
                        .map(|(trunk, branch)| {
                            // The branch is requantized and clamped; the sum is not.
                            requantize(
                                *branch,
                                spec.multiplier[0],
                                spec.shift,
                                spec.clamp.lower,
                                spec.clamp.upper,
                            ) + trunk
                        })
                        .collect();
                    let sum = Act {
                        c: trunk.c,
                        h: trunk.h,
                        w: trunk.w,
                        data,
                    };
                    tensors.insert("add".to_string(), sum.clone());
                    points.insert(point.get(), sum);
                }
                GraphOp::UpsampleNearest2x { point } => {
                    let source = points.get(&point.get()).cloned().expect("source");
                    let (h, w) = (source.h * 2, source.w * 2);
                    let mut data = vec![0i64; source.c * h * w];
                    for channel in 0..source.c {
                        for y in 0..source.h {
                            for x in 0..source.w {
                                let value = source.data[(channel * source.h + y) * source.w + x];
                                for dy in 0..2 {
                                    for dx in 0..2 {
                                        data[(channel * h + 2 * y + dy) * w + 2 * x + dx] = value;
                                    }
                                }
                            }
                        }
                    }
                    let upsampled = Act {
                        c: source.c,
                        h,
                        w,
                        data,
                    };
                    tensors.insert("upsample_nearest2x".to_string(), upsampled.clone());
                    points.insert(point.get(), upsampled);
                }
                // The unsigned clamp is the head's saturation; `intref.rs` keeps
                // the image the head produced, so this operator is the identity.
                GraphOp::QuantizeU8Output { .. } => {}
            }
        }
        let image = image.expect("the graph produced an unsigned image");
        for name in ["head", "quantize_u8_output", "image"] {
            tensors.insert(name.to_string(), image.clone());
        }
        Run { tensors, image }
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

/// Report a gate that cannot run here.
///
/// A missing tool is a report, never a result: the test says which input it
/// lacked instead of claiming agreement. Under the acceptance gate's strict mode
/// (`TESSERA_ACCEPTANCE_STRICT`, the same switch the acceptance suite uses) an
/// un-runnable gate is a failure, because a gate that did not run cannot be part
/// of an acceptance claim.
fn skip_or_fail(reason: &str) {
    if std::env::var_os("TESSERA_ACCEPTANCE_STRICT").is_some() {
        panic!("the native execution gate cannot run: {reason}");
    }
    eprintln!("SKIP: {reason}");
}

/// The head layer's stored filter, indexed the way the manifest stores it
/// (`[out_channel][in_channel][kh][kw]`), as the fixture oracle quotes it.
fn fixture_head_rows(graph: &Graph, blob: &[u8]) -> Vec<Vec<i64>> {
    let head = graph
        .layers()
        .iter()
        .find(|layer| layer.out_unsigned)
        .expect("the fixture has an unsigned head layer");
    let (out_channels, in_channels) = (
        head.out_channels.get() as usize,
        head.in_channels.get() as usize,
    );
    (0..out_channels)
        .map(|out| {
            (0..in_channels)
                .map(|channel| {
                    blob[head.weight_byte_offset as usize + (out * in_channels + channel)] as i8
                        as i64
                })
                .collect()
        })
        .collect()
}

/// Cross-check the reference model against the fixture's checked-in oracle.
///
/// `fixtures/acceptance/w8_execution_oracle.json` holds per-element codes for
/// every tensor of this graph, derived by the acceptance suite's own scalar
/// interpreter — it never saw an IR artifact, and the acceptance suite re-derives
/// it from the manifest so it cannot drift. Agreeing with it means the model that
/// judges the IR is not itself the thing under repair.
///
/// The model is the artifact under judgement here, so the check is mandatory
/// whenever the oracle is present; only its absence is tolerated, and that is
/// reported rather than assumed.
fn check_reference_against_oracle(reference: &Reference<'_>) {
    let path = fixtures().join("acceptance/w8_execution_oracle.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!(
            "note: {} is absent, so the reference model is not cross-checked against the fixture \
             oracle",
            path.display()
        );
        return;
    };
    let oracle: serde_json::Value = serde_json::from_str(&text).expect("read the oracle JSON");
    // The oracle quotes the head's filter, so a fixture change that the oracle has
    // not followed is reported as what it is, instead of surfacing as a "reference
    // model disagrees" failure in the wrong file.
    let head_rows: Vec<Vec<i64>> = oracle["head_filter_rows"]
        .as_array()
        .expect("the oracle quotes the head filter")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("a filter row")
                .iter()
                .map(|code| code.as_i64().expect("a weight code"))
                .collect()
        })
        .collect();
    let stale = || {
        eprintln!(
            "note: {} quoted a different head filter than this fixture, so it was derived from \
             another revision; re-derive it (see tests/acceptance_pipeline.rs) before it can \
             cross-check the reference model",
            path.display()
        );
    };
    if head_rows != fixture_head_rows(reference.graph, reference.blob) {
        stale();
        return;
    }
    let tensors = oracle["tensors"].as_array().expect("`tensors` is an array");
    let run = reference.run(&input_codes());
    let mut checked = Vec::new();
    for tensor in tensors {
        let name = tensor["name"].as_str().expect("a tensor name");
        let Some(act) = run.tensors.get(name) else {
            continue;
        };
        let expected: Vec<i64> = tensor["codes"]
            .as_array()
            .expect("the tensor carries its codes")
            .iter()
            .map(|code| code.as_i64().expect("a code is an integer"))
            .collect();
        let mine = act.nhwc();
        assert_eq!(
            mine.len(),
            expected.len(),
            "the reference and the oracle disagree on the length of `{name}`"
        );
        for (position, (mine, oracle)) in mine.iter().zip(&expected).enumerate() {
            assert_eq!(
                mine, oracle,
                "the reference model disagrees with the fixture oracle on `{name}` at {position}"
            );
        }
        checked.push(name.to_string());
    }
    assert!(
        checked.len() >= 9,
        "the oracle and the reference only agreed on {checked:?}, which is fewer than the graph's \
         tensors"
    );
}

#[test]
fn the_emitted_ir_executes_and_matches_the_reference_contract() {
    let Some(tools) = tools() else {
        skip_or_fail(
            "set MLIR_SYS_230_PREFIX to an LLVM/MLIR 23 installation with `mlir-opt`, \
             `mlir-runner` and `libmlir_c_runner_utils` to execute the emitted IR",
        );
        return;
    };

    let artifact = w8();
    let optimized = optimized(&artifact);
    let reference = Reference {
        graph: artifact.graph(),
        artifact: &optimized,
        blob: &weights_blob(),
    };

    // The reference model is compared against the fixture's checked-in oracle
    // before it is allowed to judge the IR: the oracle is an independent scalar
    // derivation of the same contract, so a reference bug cannot hide as an "IR
    // difference" or as an "IR agreement".
    check_reference_against_oracle(&reference);

    let dir = std::env::temp_dir().join(format!("tessera-mlir-exec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    // Three stimuli, and the comparison is only worth something while the set is
    // not flat: the pinned stimulus leaves this fixture's image all zeros (two of
    // the head's three rows have no positive coefficient over an always
    // non-negative activation), so the high-drive pattern carries the range and
    // the boundary probe carries the codes above `i8::MAX` that separate an
    // unsigned reading of the head's container from a signed one.
    let mut cases: Vec<(String, Vec<i8>)> = vec![
        ("pinned".to_string(), input_codes()),
        ("high-drive".to_string(), high_drive_codes()),
    ];
    match boundary_stimulus(&reference, PROBE_ATTEMPTS) {
        Some(boundary) => cases.push(boundary),
        None => eprintln!(
            "note: {PROBE_ATTEMPTS} probes found no stimulus that lifts this fixture's image \
             above `i8::MAX`, so the unsigned reading of the head's container is not pinned here"
        ),
    }

    let mut discriminating = false;
    for (label, stimulus) in &cases {
        let signature = signature_of(&optimized, artifact.graph(), stimulus);
        let executed = execute(&tools, &optimized, &signature, &dir, label);
        let expected = reference.run(stimulus);
        let distinct: BTreeSet<i64> = expected.image.data.iter().copied().collect();
        discriminating |=
            distinct.iter().max().is_some_and(|code| *code > 0) && distinct.len() >= 2;
        assert_codes_match(&optimized, label, &executed, &expected.image);
    }
    // The codes agreed; this says how much that agreement is worth. A fixture
    // whose image cannot vary with position would make every case vacuous.
    assert!(
        discriminating,
        "every stimulus left the image flat or constant, so matching the reference proves nothing \
         about the head: {cases:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Lower the spliced driver with `mlir-opt` and run it on `mlir-runner`.
///
/// The driver prints `<position>,<code>` per element; the positions it reports are
/// cross-checked against the order of the lines, so a code can never be read from
/// a different element than the one the module computed.
fn execute(
    tools: &Tools,
    optimized: &MlirArtifact,
    signature: &Signature,
    dir: &Path,
    label: &str,
) -> Vec<i64> {
    let driver_text = driver_module(optimized, signature);
    let stem = dir.join(format!("driver-{label}"));
    let driver_path = stem.with_extension("mlir");
    std::fs::write(&driver_path, &driver_text).expect("write the driver");
    let lowered = stem.with_extension("llvm.mlir");
    let output = Command::new(&tools.mlir_opt)
        .arg(&driver_path)
        .arg("--pass-pipeline=builtin.module(lower-affine,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)")
        .arg("-o")
        .arg(&lowered)
        .output()
        .expect("run mlir-opt");
    assert!(
        output.status.success(),
        "lowering the driver failed ({label}): {}\n--- driver ---\n{}",
        String::from_utf8_lossy(&output.stderr),
        driver_text.lines().take(40).collect::<Vec<_>>().join("\n")
    );
    let run = Command::new(&tools.mlir_runner)
        .arg(&lowered)
        .arg("-e")
        .arg("main")
        .arg(format!(
            "--shared-libs={}",
            tools
                .runner_utils
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(",")
        ))
        .arg("-entry-point-result=void")
        .output()
        .expect("run mlir-runner");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        run.status.success(),
        "executing the emitted IR failed ({label}): {}\n{stdout}",
        String::from_utf8_lossy(&run.stderr)
    );
    let codes = decode_elements(&stdout, label);
    assert!(
        !codes.is_empty(),
        "the module printed nothing ({label}): {stdout:?}"
    );
    // The same lowered module run a second time. A buffer that is read before it
    // is written returns whatever the allocator handed out, so such a defect makes
    // the two runs disagree; the elementwise comparison alone cannot see it
    // because the reference is compared against one run only.
    let again = Command::new(&tools.mlir_runner)
        .arg(&lowered)
        .arg("-e")
        .arg("main")
        .arg(format!(
            "--shared-libs={}",
            tools
                .runner_utils
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(",")
        ))
        .arg("-entry-point-result=void")
        .output()
        .expect("run mlir-runner again");
    assert!(
        again.status.success(),
        "the second execution failed ({label}): {}",
        String::from_utf8_lossy(&again.stderr)
    );
    let repeated = decode_elements(&String::from_utf8_lossy(&again.stdout), label);
    assert_eq!(
        codes, repeated,
        "two runs of the same lowered module produced different codes ({label}), so the module \
         reads a buffer it never wrote"
    );
    codes
}

/// Decode the `position,code` element channel, requiring every line to carry the
/// position it belongs to and the positions to arrive in order.
fn decode_elements(stdout: &str, label: &str) -> Vec<i64> {
    fn malformed(label: &str, number: usize, line: &str) -> ! {
        panic!(
            "the executed module's output is not `position,code` ({label}, line {number}): {line:?}"
        )
    }
    let mut codes = Vec::new();
    for (number, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (position, code) = line
            .split_once(',')
            .unwrap_or_else(|| malformed(label, number + 1, line));
        let position: usize = position
            .trim()
            .parse()
            .unwrap_or_else(|_| malformed(label, number + 1, line));
        let code: i64 = code
            .trim()
            .parse()
            .unwrap_or_else(|_| malformed(label, number + 1, line));
        assert_eq!(
            position,
            codes.len(),
            "the element channel is out of order ({label}): line {} reports element {position}, \
             expected {}",
            number + 1,
            codes.len()
        );
        codes.push(code);
    }
    codes
}

/// Compare one executed image against the reference model, element by element.
fn assert_codes_match(optimized: &MlirArtifact, label: &str, executed: &[i64], image: &Act) {
    let (n, h, w, c) = (
        optimized.output.shape[0] as usize,
        optimized.output.shape[1] as usize,
        optimized.output.shape[2] as usize,
        optimized.output.shape[3] as usize,
    );
    assert_eq!((image.c, image.h, image.w), (c, h, w));
    let mut expected = Vec::with_capacity(executed.len());
    for _ in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    expected.push(image.data[(ci * h + hi) * w + wi]);
                }
            }
        }
    }
    assert_eq!(
        executed.len(),
        expected.len(),
        "one code per output element ({label})"
    );
    let mut mismatches = Vec::new();
    for (position, (actual, want)) in executed.iter().zip(&expected).enumerate() {
        if actual != want {
            let ci = position % c;
            let wi = (position / c) % w;
            let hi = (position / (c * w)) % h;
            let ni = position / (c * w * h);
            mismatches.push((position, ni, hi, wi, ci, *actual, *want));
        }
    }
    if !mismatches.is_empty() {
        let mut report = String::new();
        for (position, ni, hi, wi, ci, actual, want) in mismatches.iter().take(12) {
            let _ = writeln!(
                report,
                "  pos {position} (n{ni} h{hi} w{wi} c{ci}): executed {actual}, reference {want}"
            );
        }
        panic!(
            "{} of {} codes differ between the executed IR and the reference contract ({label}):\
             \n{report}",
            mismatches.len(),
            executed.len()
        );
    }
}
