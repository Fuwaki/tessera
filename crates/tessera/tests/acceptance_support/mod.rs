//! Shared scaffolding for the acceptance suite.
//!
//! This module is deliberately independent of the compiler's internals: it reads
//! `manifest.json`, `weights.bin` and `input.json` from `fixtures/acceptance/`
//! with its own small schema, evaluates the graph with its own scalar reference,
//! and compares the result against:
//!
//! * the frozen vectors in `expected.json` (so a fixture can never drift
//!   silently, and the vectors are re-derived on every run), and
//! * the `tessera.sim_trace.v1` dump the RTL testbench writes, when the full
//!   graph pipeline can emit and simulate (so the RTL is judged against numbers
//!   that were never produced by the generator that produced the RTL).
//!
//! The scalar reference implements the integer datapath the hardware model is
//! defined by, not the floating-point reference `intref.rs` uses:
//!
//! ```text
//! acc        = bias_int32[oc] + sum over taps of (activation code * weight code)
//! product    = acc * M0[oc]
//! offset     = product >= 0 ? 2^(shift-1)     : 2^(shift-1) - 1
//! code       = clamp((product + offset) >> shift, lo, hi)     // >> is arithmetic
//! shift      = floor(log2((2^17-1) / max(requant))), clamped to [0, 31]
//! M0[c]      = round_half_away(requant[c] * 2^shift)
//! ```
//!
//! `M0`/`shift` are re-derived here from the manifest's `f32` requantization
//! vectors rather than copied from any artifact, and `acceptance_pipeline.rs`
//! cross-checks the re-derivation against what lowering actually produced.

#![allow(dead_code)]

use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Format tag of the frozen vector files.
pub const VECTORS_FORMAT: &str = "tessera.acceptance_vectors.v1";
/// Format tag of the acceptance input files.
pub const INPUT_FORMAT: &str = "tessera.acceptance_input.v1";
/// Format tag of the full-graph simulation trace.
pub const TRACE_FORMAT: &str = "tessera.sim_trace.v1";
/// Largest multiplier the 17-bit hardware multiplier accepts.
pub const MAX_MULTIPLIER: i64 = (1 << 17) - 1;
/// Largest shift the serialized fixed-point form accepts.
pub const MAX_SHIFT: u32 = 31;

// ---------------------------------------------------------------------------
// Paths and process helpers
// ---------------------------------------------------------------------------

/// Repository `fixtures/` directory, resolved from the crate manifest dir.
pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// One checked-in acceptance fixture directory.
pub fn fixture_dir(name: &str) -> PathBuf {
    fixtures().join("acceptance").join(name)
}

/// A fresh temporary stage directory for a CLI-driven acceptance run.
pub fn stage_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tessera-acceptance-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the stage directory");
    dir
}

/// The `tessera` binary under test.
pub fn tessera_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tessera"))
}

/// Runs the CLI and returns its output; the caller decides what a non-zero exit
/// means.
pub fn run_tessera(args: &[String]) -> Output {
    Command::new(tessera_bin())
        .args(args)
        .output()
        .expect("run the tessera binary")
}

/// `stdout` of a CLI invocation that must succeed.
pub fn run_ok(args: &[String]) -> String {
    let output = run_tessera(args);
    assert!(
        output.status.success(),
        "tessera {} failed\nstdout:\n{}\nstderr:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// `stderr` (plus `stdout`) of a CLI invocation that must fail.
pub fn run_err(args: &[String]) -> String {
    let output = run_tessera(args);
    assert!(
        !output.status.success(),
        "tessera {} was expected to fail\nstdout:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stdout)
    );
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    )
}

fn s(value: impl AsRef<str>) -> String {
    value.as_ref().to_string()
}

/// Builds an owned CLI argument list.
pub fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|value| s(value)).collect()
}

// ---------------------------------------------------------------------------
// Manifest schema (the subset the reference needs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub act_qmax: f32,
    pub output_qmax: f32,
    #[serde(default)]
    pub points: Vec<String>,
    pub point_scales: Vec<f32>,
    pub graph: Vec<ManifestOp>,
    pub layers: Vec<ManifestLayer>,
    pub weight_file: String,
    pub weight_bytes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestOp {
    pub op: String,
    #[serde(default)]
    pub layer: Option<String>,
    #[serde(default)]
    pub point: Option<usize>,
    #[serde(default)]
    pub source_point: Option<usize>,
    #[serde(default)]
    pub scale: Option<f32>,
}

fn one() -> u32 {
    1
}

fn eight() -> u8 {
    8
}

#[derive(Debug, Clone, Deserialize)]
pub struct ManifestLayer {
    pub name: String,
    pub op: String,
    pub in_channels: u32,
    pub out_channels: u32,
    #[serde(default = "one")]
    pub groups: u32,
    /// Declared activation width; omitted layers inherit the manifest's `bits`.
    /// The suite only decodes `i8-x1` blobs and asserts that much.
    #[serde(default = "eight")]
    pub act_bits: u8,
    /// Declared weight width; see [`ManifestLayer::act_bits`].
    #[serde(default = "eight")]
    pub weight_bits: u8,
    pub out_relu: bool,
    pub out_unsigned: bool,
    pub kernel: [u32; 2],
    pub stride: [u32; 2],
    #[serde(default)]
    pub reflect_pad: u32,
    pub in_point: usize,
    pub out_point: usize,
    pub in_scale: f32,
    pub out_scale: f32,
    #[serde(default)]
    pub weight_scale: Vec<f32>,
    #[serde(default)]
    pub bias: Vec<f32>,
    pub bias_int32: Vec<i32>,
    pub requant: Vec<f32>,
    pub weight_offset: usize,
    pub weight_count: usize,
}

impl Manifest {
    /// Whether a layer is a depthwise convolution (one weight channel per output).
    pub fn is_depthwise(layer: &ManifestLayer) -> bool {
        layer.op.starts_with("depthwise")
    }

    /// Mutable layer by name, so a test can perturb one field of its own copy.
    pub fn layer_mut(&mut self, name: &str) -> &mut ManifestLayer {
        self.layers
            .iter_mut()
            .find(|layer| layer.name == name)
            .unwrap_or_else(|| panic!("manifest has no layer {name:?}"))
    }

    /// Layer by name.
    pub fn layer(&self, name: &str) -> &ManifestLayer {
        self.layers
            .iter()
            .find(|layer| layer.name == name)
            .unwrap_or_else(|| panic!("manifest has no layer {name:?}"))
    }

    /// Name of a graph point.
    pub fn point_name(&self, index: usize) -> String {
        self.points
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("point{index}"))
    }
}

/// A fixture directory loaded from disk.
pub struct Fixture {
    pub name: String,
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub manifest_text: String,
    pub weights: Vec<u8>,
    pub input: AcceptanceInput,
    pub vectors: ExpectedVectors,
}

impl Fixture {
    /// Loads `fixtures/acceptance/<name>/{manifest.json,weights.bin,input.json,expected.json}`.
    pub fn load(name: &str) -> Fixture {
        let dir = fixture_dir(name);
        let manifest_text = std::fs::read_to_string(dir.join("manifest.json"))
            .unwrap_or_else(|error| panic!("read {}/manifest.json: {error}", dir.display()));
        let manifest: Manifest = serde_json::from_str(&manifest_text)
            .unwrap_or_else(|error| panic!("parse {}/manifest.json: {error}", dir.display()));
        let weights = std::fs::read(dir.join(&manifest.weight_file)).unwrap_or_else(|error| {
            panic!("read {}/{}: {error}", dir.display(), manifest.weight_file)
        });
        assert_eq!(
            weights.len(),
            manifest.weight_bytes,
            "{name}: weights.bin length must match the declared weight_bytes"
        );
        let input: AcceptanceInput =
            serde_json::from_str(&read(&dir.join("input.json"))).expect("parse input.json");
        let vectors: ExpectedVectors =
            serde_json::from_str(&read(&dir.join("expected.json"))).expect("parse expected.json");
        assert_eq!(vectors.format, VECTORS_FORMAT, "{name}: vectors format");
        assert_eq!(vectors.fixture, name, "{name}: vectors name the fixture");
        assert_eq!(input.format, INPUT_FORMAT, "{name}: input format");
        assert_eq!(input.fixture, name, "{name}: input names the fixture");
        Fixture {
            name: name.to_string(),
            dir,
            manifest,
            manifest_text,
            weights,
            input,
            vectors,
        }
    }

    /// The input codes of one frame, in `(c, h, w)` order.
    pub fn frame_input_codes(&self, frame: usize) -> Vec<i64> {
        self.input
            .frames
            .iter()
            .find(|entry| entry.index as usize == frame)
            .unwrap_or_else(|| panic!("input.json has no frame {frame}"))
            .codes
            .clone()
    }

    /// Input planes of every frame, ready for `run_frame`.
    pub fn input_planes(&self) -> Vec<Plane> {
        let shape = self.input.shape;
        self.input
            .frames
            .iter()
            .map(|frame| Plane::new(shape.c, shape.h, shape.w, frame.codes.clone()))
            .collect()
    }

    /// The frozen expectation of one frame.
    pub fn frame_vectors(&self, frame: usize) -> &ExpectedFrame {
        self.vectors
            .frames
            .iter()
            .find(|entry| entry.index as usize == frame)
            .unwrap_or_else(|| panic!("expected.json has no frame {frame}"))
    }

    /// A per-layer expectation record.
    pub fn expected_layer(&self, name: &str) -> &ExpectedLayer {
        self.vectors
            .layers
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("expected.json has no layer {name:?}"))
    }

    /// The prefix the emitted schedule step must carry for a canonical operator
    /// name. The mapping lives in `expected.json` so an operator naming scheme
    /// can change without silently weakening the comparison.
    pub fn step_match(&self, name: &str) -> String {
        self.vectors
            .layers
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.step_match.clone())
            .unwrap_or_else(|| name.to_string())
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

// ---------------------------------------------------------------------------
// input.json / expected.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AcceptanceInput {
    pub format: String,
    pub fixture: String,
    pub shape: Shape,
    pub frames: Vec<AcceptanceInputFrame>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcceptanceInputFrame {
    pub index: u32,
    pub codes: Vec<i64>,
    #[serde(default)]
    pub image_u8: Option<Vec<Vec<Vec<u8>>>>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Shape {
    pub c: u32,
    pub h: u32,
    pub w: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedVectors {
    pub format: String,
    pub fixture: String,
    pub shape: Shape,
    #[serde(default)]
    pub derivation: Vec<String>,
    pub layers: Vec<ExpectedLayer>,
    pub frames: Vec<ExpectedFrame>,
    pub input_frames: Vec<ExpectedInputFrame>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedLayer {
    pub name: String,
    /// Prefix the emitted schedule step's operator name must carry.
    pub step_match: String,
    pub elem: String,
    pub shape: Shape,
    #[serde(default)]
    pub bias_int32: Option<Vec<i32>>,
    #[serde(default)]
    pub requant: Option<ExpectedRequant>,
    #[serde(default)]
    pub residual: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedRequant {
    pub m0: Vec<i64>,
    pub shift: u32,
    pub clamp: [i64; 2],
    pub rounding: String,
    #[serde(default)]
    pub per_channel_multipliers: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedFrame {
    pub index: u32,
    pub ops: Vec<ExpectedOp>,
    pub output_u8: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedOp {
    pub name: String,
    pub elem: String,
    pub codes: Vec<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedInputFrame {
    pub index: u32,
    pub input_codes: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Fixed-point serialization (independent of `mlir/quant.rs`)
// ---------------------------------------------------------------------------

/// `round()` half away from zero, exactly as the emitted datapath defines it.
pub fn round_half_away(value: f64) -> f64 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        -((-value + 0.5).floor())
    }
}

/// Rounding offsets added before the arithmetic shift:
/// `(2^(shift-1), 2^(shift-1) - 1)`; a degenerate `shift == 0` needs none.
pub fn rounding_offsets(shift: u32) -> (i64, i64) {
    if shift == 0 {
        (0, 0)
    } else {
        let positive = 1i64 << (shift - 1);
        (positive, positive - 1)
    }
}

/// Derives `(M0, shift)` from a per-channel requantization vector, so that
/// `requant[c] ~= M0[c] * 2^-shift` with every `M0[c]` inside the 17-bit budget.
pub fn serialize_requant(requant: &[f32]) -> (Vec<i64>, u32) {
    assert!(
        !requant.is_empty(),
        "a requantization vector cannot be empty"
    );
    let max_abs = requant
        .iter()
        .map(|raw| (*raw as f64).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs.is_finite() && max_abs > 0.0,
        "requantization factors must be finite and positive, got {requant:?}"
    );
    let ratio = MAX_MULTIPLIER as f64 / max_abs;
    let shift = (ratio.log2().floor() as i64).clamp(0, MAX_SHIFT as i64) as u32;
    let scale = (1i64 << shift) as f64;
    let m0 = requant
        .iter()
        .map(|raw| round_half_away(*raw as f64 * scale) as i64)
        .collect::<Vec<_>>();
    assert!(
        m0.iter()
            .all(|value| *value >= 0 && *value <= MAX_MULTIPLIER),
        "derived multipliers {m0:?} leave [0, {MAX_MULTIPLIER}] at shift {shift}"
    );
    (m0, shift)
}

/// The integer requantization step: signed product, sign-dependent rounding
/// offset, arithmetic shift right, then clamp.
pub fn requantize(acc: i64, m0: i64, shift: u32, lo: i64, hi: i64) -> i64 {
    let product = acc * m0;
    let (positive, negative) = rounding_offsets(shift);
    let rounded = if product >= 0 { positive } else { negative };
    let shifted = if shift == 0 {
        product
    } else {
        (product + rounded) >> shift
    };
    shifted.clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// The scalar reference
// ---------------------------------------------------------------------------

/// One activation tensor as signed integer codes, `(c, h, w)` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub codes: Vec<i64>,
}

impl Plane {
    /// Builds a plane, checking the code count.
    pub fn new(c: u32, h: u32, w: u32, codes: Vec<i64>) -> Plane {
        assert_eq!(
            codes.len() as u64,
            u64::from(c) * u64::from(h) * u64::from(w),
            "plane {c}x{h}x{w} needs {} codes, got {}",
            u64::from(c) * u64::from(h) * u64::from(w),
            codes.len()
        );
        Plane { c, h, w, codes }
    }

    /// Code at `(channel, y, x)`.
    pub fn at(&self, channel: u32, y: u32, x: u32) -> i64 {
        self.codes[((channel * self.h + y) * self.w + x) as usize]
    }

    /// Shape as `(c, h, w)`.
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.c, self.h, self.w)
    }

    /// Number of codes that leave the signed 8-bit domain.
    pub fn outside_i8(&self) -> usize {
        self.codes
            .iter()
            .filter(|code| **code < -128 || **code > 127)
            .count()
    }
}

/// The fixed-point parameters one operator applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequantParams {
    pub m0: Vec<i64>,
    pub shift: u32,
    pub lo: i64,
    pub hi: i64,
}

/// One evaluated operator.
#[derive(Debug, Clone)]
pub struct OpReference {
    /// Canonical operator name this suite uses (`stem`, `add`, `upsample`, ...).
    pub name: String,
    pub plane: Plane,
    /// `i8`, `i16` or `u8`, derived from the codes the same way the trace does.
    pub elem: &'static str,
    /// `None` for operators that do not requantize (upsample, output clamp).
    pub requant: Option<RequantParams>,
}

/// One evaluated frame.
#[derive(Debug, Clone)]
pub struct FrameReference {
    pub index: usize,
    pub input: Plane,
    pub ops: Vec<OpReference>,
    pub image: Plane,
}

/// Reflection index, as PyTorch and the reference interpreter define it: the
/// edge is not repeated.
fn reflect_index(index: i64, n: u32) -> u32 {
    let n = i64::from(n);
    if n <= 1 {
        return 0;
    }
    let mut i = index;
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * (n - 1) - i };
    }
    i as u32
}

fn clamp_bounds(layer: &ManifestLayer, act_qmax: i64, output_qmax: i64) -> (i64, i64) {
    if layer.out_unsigned {
        (0, output_qmax)
    } else if layer.out_relu {
        (0, act_qmax)
    } else {
        (-act_qmax, act_qmax)
    }
}

/// Reflect-pads a plane by `pad`, edges not repeated.
fn reflect_pad(plane: &Plane, pad: u32) -> Plane {
    if pad == 0 {
        return plane.clone();
    }
    let h = plane.h + 2 * pad;
    let w = plane.w + 2 * pad;
    let mut codes = vec![0i64; (plane.c * h * w) as usize];
    for channel in 0..plane.c {
        for y in 0..h {
            let sy = reflect_index(i64::from(y) - i64::from(pad), plane.h);
            for x in 0..w {
                let sx = reflect_index(i64::from(x) - i64::from(pad), plane.w);
                codes[((channel * h + y) * w + x) as usize] = plane.at(channel, sy, sx);
            }
        }
    }
    Plane::new(plane.c, h, w, codes)
}

/// Zero-pads a plane. Used only to prove a fixture is sensitive to the
/// difference between reflection and zero padding.
fn zero_pad(plane: &Plane, pad: u32) -> Plane {
    if pad == 0 {
        return plane.clone();
    }
    let h = plane.h + 2 * pad;
    let w = plane.w + 2 * pad;
    let mut codes = vec![0i64; (plane.c * h * w) as usize];
    for channel in 0..plane.c {
        for y in 0..plane.h {
            for x in 0..plane.w {
                codes[((channel * h + (y + pad)) * w + (x + pad)) as usize] =
                    plane.at(channel, y, x);
            }
        }
    }
    Plane::new(plane.c, h, w, codes)
}

/// One convolution: pad, accumulate through the int32 bias, requantize, clamp.
pub struct ConvOutcome {
    pub plane: Plane,
    pub params: RequantParams,
}

fn convolve(
    layer: &ManifestLayer,
    weights: &[i8],
    input: &Plane,
    act_qmax: i64,
    output_qmax: i64,
    pad: fn(&Plane, u32) -> Plane,
) -> ConvOutcome {
    let (m0, shift) = serialize_requant(&layer.requant);
    let (lo, hi) = clamp_bounds(layer, act_qmax, output_qmax);
    let (kh, kw) = (layer.kernel[0], layer.kernel[1]);
    let (sh, sw) = (layer.stride[0], layer.stride[1]);
    let padded = pad(input, layer.reflect_pad);
    let out_h = (padded.h - kh) / sh + 1;
    let out_w = (padded.w - kw) / sw + 1;
    let depthwise = Manifest::is_depthwise(layer);
    let in_per_group = layer.in_channels / layer.groups.max(1);

    let mut codes = vec![0i64; (layer.out_channels * out_h * out_w) as usize];
    for oc in 0..layer.out_channels {
        let base = if depthwise {
            oc * kh * kw
        } else {
            oc * in_per_group * kh * kw
        };
        for oy in 0..out_h {
            for ox in 0..out_w {
                let mut acc = i64::from(layer.bias_int32[oc as usize]);
                if depthwise {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let a = padded.at(oc, oy * sh + ky, ox * sw + kx);
                            let b = i64::from(weights[(base + ky * kw + kx) as usize]);
                            acc += a * b;
                        }
                    }
                } else {
                    for ic in 0..in_per_group {
                        let weight_base = base + ic * kh * kw;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let a = padded.at(ic, oy * sh + ky, ox * sw + kx);
                                let b = i64::from(weights[(weight_base + ky * kw + kx) as usize]);
                                acc += a * b;
                            }
                        }
                    }
                }
                let index = ((oc * out_h + oy) * out_w + ox) as usize;
                codes[index] = requantize(acc, m0[oc as usize], shift, lo, hi);
            }
        }
    }
    ConvOutcome {
        plane: Plane::new(layer.out_channels, out_h, out_w, codes),
        params: RequantParams { m0, shift, lo, hi },
    }
}

/// Nearest-neighbour 2x upsampling: every code becomes a 2x2 block.
fn upsample2x(plane: &Plane) -> Plane {
    let h = plane.h * 2;
    let w = plane.w * 2;
    let mut codes = vec![0i64; (plane.c * h * w) as usize];
    for channel in 0..plane.c {
        for y in 0..h {
            for x in 0..w {
                codes[((channel * h + y) * w + x) as usize] = plane.at(channel, y / 2, x / 2);
            }
        }
    }
    Plane::new(plane.c, h, w, codes)
}

/// The integer element type a step's stored codes require.
fn elem_of(codes: &[i64], unsigned: bool) -> &'static str {
    if unsigned {
        "u8"
    } else if codes.iter().any(|code| *code < -128 || *code > 127) {
        "i16"
    } else {
        "i8"
    }
}

/// Weight codes of one layer, decoded from a raw blob.
pub fn layer_weights_of(layer: &ManifestLayer, weights: &[u8]) -> Vec<i8> {
    assert_eq!(layer.weight_bits, 8, "only i8-x1 blobs are supported");
    let start = layer.weight_offset;
    let end = start + layer.weight_count;
    assert!(
        end <= weights.len(),
        "layer {} declares codes {start}..{end} of a {}-byte blob",
        layer.name,
        weights.len()
    );
    weights[start..end].iter().map(|byte| *byte as i8).collect()
}

/// How the residual sum is transported out of an `add`.
///
/// The reference contract (and the RTL backend) carries the unclamped sum at its
/// natural width, so a residual of two int8 codes is a 9-bit value. An artifact
/// that narrows the sum back to `i8` *wraps* it instead, which changes every
/// consumer downstream of the addition. The suite asserts the carrying contract
/// and uses the wrapping variant only to attribute a divergence to that
/// narrowing, never as an expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualCarry {
    /// `sum = clamp_sym(requant(branch)) + trunk`, unclamped and un-narrowed.
    Carry,
    /// The same sum narrowed to `i8` (two's complement wrap), as an artifact
    /// whose result tensor is still declared `i8` would transport it.
    WrapToI8,
}

/// Evaluates one frame of a manifest graph with an explicit input plane.
///
/// The graph is walked in manifest order; `relu` markers are no-ops because the
/// fused activation is already the convolution's clamp.
pub fn run_frame(manifest: &Manifest, weights: &[u8], input: &Plane) -> (Vec<OpReference>, Plane) {
    run_frame_with(manifest, weights, input, ResidualCarry::Carry)
}

/// [`run_frame`] with an explicit residual transport rule.
pub fn run_frame_with(
    manifest: &Manifest,
    weights: &[u8],
    input: &Plane,
    carry: ResidualCarry,
) -> (Vec<OpReference>, Plane) {
    let act_qmax = manifest.act_qmax as i64;
    let output_qmax = manifest.output_qmax as i64;
    let mut points: Vec<Option<Plane>> = vec![None; manifest.point_scales.len()];
    let mut image: Option<Plane> = None;
    let mut ops: Vec<OpReference> = Vec::new();
    points[0] = Some(input.clone());

    for op in &manifest.graph {
        match op.op.as_str() {
            "conv" => {
                let name = op
                    .layer
                    .as_deref()
                    .unwrap_or_else(|| panic!("conv op without a layer: {op:?}"));
                let layer = manifest.layer(name);
                let source = points[layer.in_point]
                    .clone()
                    .unwrap_or_else(|| panic!("point {} is not live for {name}", layer.in_point));
                let codes = layer_weights_of(layer, weights);
                let outcome = convolve(layer, &codes, &source, act_qmax, output_qmax, reflect_pad);
                if layer.out_unsigned {
                    image = Some(outcome.plane.clone());
                }
                if layer.out_point < points.len() {
                    points[layer.out_point] = Some(outcome.plane.clone());
                }
                let elem = elem_of(&outcome.plane.codes, layer.out_unsigned);
                ops.push(OpReference {
                    name: name.to_string(),
                    plane: outcome.plane,
                    elem,
                    requant: Some(outcome.params),
                });
            }
            "relu" => {}
            "add" => {
                let point = op.point.expect("add without a point");
                let source = op.source_point.expect("add without a source_point");
                let trunk = points[point]
                    .clone()
                    .unwrap_or_else(|| panic!("point {point} is not live for add"));
                let branch = points[source]
                    .clone()
                    .unwrap_or_else(|| panic!("point {source} is not live for add"));
                assert_eq!(
                    (trunk.c, trunk.h, trunk.w),
                    (branch.c, branch.h, branch.w),
                    "add operands must share a shape"
                );
                let factor = manifest.point_scales[source] / manifest.point_scales[point];
                let (m0, shift) = serialize_requant(&[factor]);
                let mut codes = Vec::with_capacity(trunk.codes.len());
                for (a, b) in branch.codes.iter().zip(&trunk.codes) {
                    let requantized = requantize(*a, m0[0], shift, -act_qmax, act_qmax);
                    // The sum is deliberately NOT clamped: the residual carries
                    // the 9-bit value into the following operators.
                    let sum = requantized + *b;
                    codes.push(match carry {
                        ResidualCarry::Carry => sum,
                        ResidualCarry::WrapToI8 => i64::from(sum as i8),
                    });
                }
                let plane = Plane::new(trunk.c, trunk.h, trunk.w, codes);
                let elem = elem_of(&plane.codes, false);
                points[point] = Some(plane.clone());
                ops.push(OpReference {
                    name: "add".to_string(),
                    plane,
                    elem,
                    requant: Some(RequantParams {
                        m0,
                        shift,
                        lo: -act_qmax,
                        hi: act_qmax,
                    }),
                });
            }
            "upsample_nearest2x" => {
                let point = op.point.expect("upsample without a point");
                let source = points[point]
                    .clone()
                    .unwrap_or_else(|| panic!("point {point} is not live for upsample"));
                let plane = upsample2x(&source);
                let elem = elem_of(&plane.codes, false);
                points[point] = Some(plane.clone());
                ops.push(OpReference {
                    name: "upsample_nearest2x".to_string(),
                    plane,
                    elem,
                    requant: None,
                });
            }
            "quantize_u8_output" => {
                let plane = image
                    .clone()
                    .expect("quantize_u8_output before an unsigned layer produced the image");
                ops.push(OpReference {
                    name: "quantize_u8_output".to_string(),
                    plane,
                    elem: "u8",
                    requant: None,
                });
            }
            other => panic!("the acceptance reference does not implement graph op {other:?}"),
        }
    }
    let image = image.expect("the graph must produce an image");
    (ops, image)
}

/// Evaluates every frame of a fixture.
pub fn run_fixture(fixture: &Fixture) -> Vec<FrameReference> {
    fixture
        .input_planes()
        .into_iter()
        .enumerate()
        .map(|(index, input)| {
            let (ops, image) = run_frame(&fixture.manifest, &fixture.weights, &input);
            FrameReference {
                index,
                input,
                ops,
                image,
            }
        })
        .collect()
}

/// Builds a plane from an NHWC code stream, as the emitted module and the
/// execution driver order their tensors.
pub fn plane_from_nhwc(codes: &[i64], c: u32, h: u32, w: u32) -> Plane {
    let mut data = vec![0i64; codes.len()];
    for hi in 0..h as usize {
        for wi in 0..w as usize {
            for ci in 0..c as usize {
                data[(ci * h as usize + hi) * w as usize + wi] =
                    codes[(hi * w as usize + wi) * c as usize + ci];
            }
        }
    }
    Plane::new(c, h, w, data)
}

/// Renders a plane as an NHWC code stream.
pub fn plane_to_nhwc(plane: &Plane) -> Vec<i64> {
    let mut out = Vec::with_capacity(plane.codes.len());
    for hi in 0..plane.h as usize {
        for wi in 0..plane.w as usize {
            for ci in 0..plane.c as usize {
                out.push(plane.at(ci as u32, hi as u32, wi as u32));
            }
        }
    }
    out
}

/// Quantizes a `u8` image in [0, 255] into point-0 codes:
/// `code = clamp(round_half_away(pixel / 255 / scale), -qmax, qmax)`.
pub fn quantize_image(image: &[Vec<Vec<u8>>], scale: f32, act_qmax: i64) -> Plane {
    let c = image.len() as u32;
    let h = image[0].len() as u32;
    let w = image[0][0].len() as u32;
    let mut codes = Vec::with_capacity((c * h * w) as usize);
    for channel in image {
        assert_eq!(channel.len() as u32, h, "ragged image rows");
        for row in channel {
            assert_eq!(row.len() as u32, w, "ragged image rows");
            for pixel in row {
                let real = f64::from(*pixel) / 255.0;
                let code = round_half_away(real / f64::from(scale)) as i64;
                codes.push(code.clamp(-act_qmax, act_qmax));
            }
        }
    }
    Plane::new(c, h, w, codes)
}

/// One convolution of one layer with reflection padding, using the layer's own
/// clamp. Exposed so a single layer can be evaluated on its own — for example to
/// decide between two candidate weight layouts — without an image-producing head.
pub fn convolve_layer(
    manifest: &Manifest,
    weights: &[u8],
    layer: &ManifestLayer,
    input: &Plane,
) -> Plane {
    let codes = layer_weights_of(layer, weights);
    convolve(
        layer,
        &codes,
        input,
        manifest.act_qmax as i64,
        manifest.output_qmax as i64,
        reflect_pad,
    )
    .plane
}

/// Reflection sensitivity probe: the same convolution with zero padding, so a
/// fixture that claims to test the border can prove the border matters.
pub fn convolve_with_zero_pad(
    manifest: &Manifest,
    weights: &[u8],
    layer: &ManifestLayer,
    input: &Plane,
) -> Plane {
    let codes = layer_weights_of(layer, weights);
    convolve(
        layer,
        &codes,
        input,
        manifest.act_qmax as i64,
        manifest.output_qmax as i64,
        zero_pad,
    )
    .plane
}

// ---------------------------------------------------------------------------
// Trace parsing and comparison
// ---------------------------------------------------------------------------

/// One block of a `tessera.sim_trace.v1` dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBlock {
    /// `input`, `output`, or the operator name of a `step` block.
    pub label: String,
    pub elem: String,
    pub shape: (u32, u32, u32),
    pub codes: Vec<i64>,
}

/// One simulated frame.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceFrame {
    pub index: u32,
    pub input: Option<TraceBlock>,
    pub steps: Vec<TraceBlock>,
    pub output: Option<TraceBlock>,
    pub cycles: Option<u64>,
    pub stalls: Option<(u64, u64)>,
}

/// A parsed simulation trace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trace {
    pub frames: Vec<TraceFrame>,
}

impl Trace {
    /// The frame with the given index.
    pub fn frame(&self, index: u32) -> &TraceFrame {
        self.frames
            .iter()
            .find(|frame| frame.index == index)
            .unwrap_or_else(|| panic!("trace has no frame {index}"))
    }
}

fn frame_mut<'a>(frames: &'a mut Vec<TraceFrame>, index: u32) -> &'a mut TraceFrame {
    if let Some(position) = frames.iter().position(|frame| frame.index == index) {
        return &mut frames[position];
    }
    frames.push(TraceFrame {
        index,
        ..TraceFrame::default()
    });
    frames.last_mut().expect("just pushed")
}

fn parse_shape(parts: &[&str]) -> Result<(u32, u32, u32), String> {
    if parts.len() < 3 {
        return Err(format!("expected `<c> <h> <w>`, got {parts:?}"));
    }
    let extent = |index: usize| -> Result<u32, String> {
        parts[index]
            .parse()
            .map_err(|_| format!("bad extent {:?}", parts[index]))
    };
    Ok((extent(0)?, extent(1)?, extent(2)?))
}

/// Parses a `tessera.sim_trace.v1` dump, failing with the offending line.
pub fn parse_trace(text: &str) -> Result<Trace, String> {
    let mut frames: Vec<TraceFrame> = Vec::new();
    let mut header_seen = false;
    let mut pending: Option<(u32, TraceBlock)> = None;
    for (number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let complain = |detail: String| format!("line {}: {detail}", number + 1);
        if let Some(format) = line.strip_prefix("format ") {
            let format = format.trim();
            if format != TRACE_FORMAT {
                return Err(complain(format!("unexpected trace format {format:?}")));
            }
            header_seen = true;
            continue;
        }
        if let Some(rest) = line.strip_prefix("frame ") {
            if !header_seen {
                return Err(complain("a block precedes the format header".to_string()));
            }
            if let Some((index, block)) = pending.take() {
                attach(frame_mut(&mut frames, index), block);
            }
            let parts: Vec<&str> = rest.split_whitespace().collect();
            let index: u32 = parts
                .first()
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| complain(format!("bad frame index in {line:?}")))?;
            match parts.get(1).copied() {
                Some("input") => {
                    let shape = parse_shape(&parts[2..]).map_err(complain)?;
                    pending = Some((
                        index,
                        TraceBlock {
                            label: "input".to_string(),
                            elem: "i8".to_string(),
                            shape,
                            codes: Vec::new(),
                        },
                    ));
                }
                Some("step") => {
                    // `frame <n> step <index> <op> <elem> <c> <h> <w>`.
                    if parts
                        .get(2)
                        .and_then(|value| value.parse::<u32>().ok())
                        .is_none()
                    {
                        return Err(complain(format!("step without a step index in {line:?}")));
                    }
                    let label = parts
                        .get(3)
                        .ok_or_else(|| complain("step without an operator name".to_string()))?
                        .to_string();
                    let elem = parts
                        .get(4)
                        .ok_or_else(|| complain("step without an element type".to_string()))?
                        .to_string();
                    let shape = parse_shape(&parts[5..]).map_err(complain)?;
                    pending = Some((
                        index,
                        TraceBlock {
                            label,
                            elem,
                            shape,
                            codes: Vec::new(),
                        },
                    ));
                }
                Some("output") => {
                    let shape = parse_shape(&parts[2..]).map_err(complain)?;
                    pending = Some((
                        index,
                        TraceBlock {
                            label: "output".to_string(),
                            elem: "u8".to_string(),
                            shape,
                            codes: Vec::new(),
                        },
                    ));
                }
                Some("cycles") => {
                    let value = parts
                        .get(2)
                        .and_then(|value| value.parse().ok())
                        .ok_or_else(|| complain("bad cycle count".to_string()))?;
                    frame_mut(&mut frames, index).cycles = Some(value);
                }
                Some("stalls") => {
                    let a = parts.get(2).and_then(|value| value.parse::<u64>().ok());
                    let b = parts.get(3).and_then(|value| value.parse::<u64>().ok());
                    if let (Some(a), Some(b)) = (a, b) {
                        frame_mut(&mut frames, index).stalls = Some((a, b));
                    }
                }
                other => {
                    return Err(complain(format!(
                        "unknown trace block {:?}",
                        other.unwrap_or("<missing>")
                    )));
                }
            }
            continue;
        }
        let value: i64 = line
            .parse()
            .map_err(|_| complain(format!("expected an integer, got {line:?}")))?;
        match pending.as_mut() {
            Some((_, block)) => block.codes.push(value),
            None => return Err(complain("a value line outside any block".to_string())),
        }
    }
    if let Some((index, block)) = pending.take() {
        attach(frame_mut(&mut frames, index), block);
    }
    Ok(Trace { frames })
}

fn attach(frame: &mut TraceFrame, block: TraceBlock) {
    match block.label.as_str() {
        "input" => frame.input = Some(block),
        "output" => frame.output = Some(block),
        _ => frame.steps.push(block),
    }
}

fn first_mismatches(got: &[i64], want: &[i64], shape: (u32, u32, u32)) -> Vec<String> {
    let (_, h, w) = shape;
    let mut reported = Vec::new();
    for (index, (a, b)) in got.iter().zip(want).enumerate() {
        if a == b {
            continue;
        }
        let plane = (h as usize) * (w as usize);
        let channel = index / plane;
        let rest = index % plane;
        reported.push(format!(
            "[c{channel},y{},x{}] trace={a} reference={b}",
            rest / w as usize,
            rest % w as usize
        ));
        if reported.len() == 8 {
            break;
        }
    }
    reported
}

/// Compares one trace block against an independently computed plane.
pub fn compare_block(
    label: &str,
    block: &TraceBlock,
    expected: &OpReference,
) -> Result<(), String> {
    let plane = &expected.plane;
    if block.shape != plane.shape() {
        return Err(format!(
            "{label}: trace shape {:?} does not match the reference's {:?}",
            block.shape,
            plane.shape()
        ));
    }
    if block.codes.len() != plane.codes.len() {
        return Err(format!(
            "{label}: {} codes in the trace, {} in the reference",
            block.codes.len(),
            plane.codes.len()
        ));
    }
    if expected.elem == "u8" {
        if block.elem != "u8" {
            return Err(format!(
                "{label}: an unsigned clamp must be labelled u8, the trace says {:?}",
                block.elem
            ));
        }
    } else if plane.outside_i8() > 0 {
        if block.elem != "i16" {
            return Err(format!(
                "{label}: {} codes leave the i8 domain but the trace labels the block {:?}",
                plane.outside_i8(),
                block.elem
            ));
        }
    } else if block.elem != "i8" && block.elem != "i16" {
        return Err(format!(
            "{label}: expected an in-range signed block, the trace labels it {:?}",
            block.elem
        ));
    }
    let mismatches = first_mismatches(&block.codes, &plane.codes, plane.shape());
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!("{label}: codes differ ({})", mismatches.join(", ")))
    }
}

/// Compares a frozen `expected.json` operator record against the reference.
pub fn compare_expected_op(
    label: &str,
    expected: &ExpectedOp,
    reference: &OpReference,
    shape: (u32, u32, u32),
) -> Result<(), String> {
    if expected.name != reference.name {
        return Err(format!(
            "{label}: expected.json names the operator {:?}, the reference produced {:?}",
            expected.name, reference.name
        ));
    }
    if expected.elem != reference.elem {
        return Err(format!(
            "{label}/{}: expected.json says {}, the reference derives {}",
            reference.name, expected.elem, reference.elem
        ));
    }
    if expected.codes.len() != reference.plane.codes.len() {
        return Err(format!(
            "{label}/{}: {} frozen codes, {} recomputed",
            reference.name,
            expected.codes.len(),
            reference.plane.codes.len()
        ));
    }
    if reference.plane.shape() != shape {
        return Err(format!(
            "{label}/{}: reference shape {:?}, fixture shape {shape:?}",
            reference.name,
            reference.plane.shape()
        ));
    }
    let mismatches = first_mismatches(&expected.codes, &reference.plane.codes, shape);
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{label}/{}: frozen vectors differ from the reference ({})",
            reference.name,
            mismatches.join(", ")
        ))
    }
}

/// Compares a frozen requantization record against the reference's derivation.
pub fn compare_expected_requant(
    label: &str,
    expected: &ExpectedRequant,
    reference: Option<&RequantParams>,
) -> Result<(), String> {
    let Some(reference) = reference else {
        return Err(format!(
            "{label}: the reference did not requantize this step"
        ));
    };
    if expected.m0 != reference.m0 {
        return Err(format!(
            "{label}: frozen M0 {:?} does not match the derived {:?}",
            expected.m0, reference.m0
        ));
    }
    if expected.shift != reference.shift {
        return Err(format!(
            "{label}: frozen shift {} does not match the derived {}",
            expected.shift, reference.shift
        ));
    }
    if expected.clamp != [reference.lo, reference.hi] {
        return Err(format!(
            "{label}: frozen clamp {:?} does not match the derived [{}, {}]",
            expected.clamp, reference.lo, reference.hi
        ));
    }
    if expected.rounding != "half_away_from_zero" {
        return Err(format!(
            "{label}: fixture declares rounding {:?}, only half_away_from_zero is verified",
            expected.rounding
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stimulus, simulation and stage helpers
// ---------------------------------------------------------------------------

/// Renders `stim_input.hex`: all frames concatenated, frame-major, one
/// two-hex-digit byte per line, `(c, h, w)` order inside a frame.
pub fn stim_input_hex(frames: &[Plane]) -> String {
    let mut text = String::new();
    for frame in frames {
        for code in &frame.codes {
            text.push_str(&format!("{:02x}\n", (*code as i8) as u8));
        }
    }
    text
}

/// Reads the emitted bundle's top module out of `<stage>/rtl/rtl_bundle.json`.
pub fn bundle_top_module(stage: &Path) -> Option<String> {
    let text = std::fs::read_to_string(stage.join("rtl").join("rtl_bundle.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value
        .get("top_module")
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

/// The result of trying to run a compiled simulation binary.
pub enum SimRun {
    /// The simulation binary ran; its raw output is returned.
    Ran {
        status: std::process::ExitStatus,
        stdout: String,
        stderr: String,
    },
    /// No simulation binary was available, with the reason.
    Unavailable(String),
}

/// Runs the emitted simulation binary with the suite's own stimulus and trace
/// paths. The bundle must have been built by an earlier CLI invocation.
pub fn run_simulation(stage: &Path, stimulus: &Path, trace: &Path) -> SimRun {
    let Some(top) = bundle_top_module(stage) else {
        return SimRun::Unavailable(format!(
            "{} has no rtl/rtl_bundle.json with a top_module",
            stage.display()
        ));
    };
    let work_dir = stage.join("sim");
    let binary = work_dir.join("obj_dir").join(format!("{top}_sim"));
    if !binary.exists() {
        return SimRun::Unavailable(format!("{} does not exist", binary.display()));
    }
    match Command::new(&binary)
        .current_dir(&work_dir)
        .arg(format!("+STIM={}", stimulus.display()))
        .arg(format!("+TRACE={}", trace.display()))
        .arg("+NOCHECK")
        .output()
    {
        Ok(output) => SimRun::Ran {
            status: output.status,
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        },
        Err(error) => SimRun::Unavailable(format!("running {}: {error}", binary.display())),
    }
}

/// Whether the suite runs as the acceptance gate rather than as a development
/// check. In strict mode an unavailable gate is a failure, not a note.
pub fn strict_mode() -> bool {
    std::env::var("TESSERA_ACCEPTANCE_STRICT")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
}

/// Whether the suite should rewrite its frozen documents instead of verifying
/// them. Set `TESSERA_ACCEPTANCE_REGEN=1` after a deliberate fixture change.
pub fn regen_mode() -> bool {
    std::env::var("TESSERA_ACCEPTANCE_REGEN")
        .map(|value| !value.is_empty() && value != "0")
        .unwrap_or(false)
}

/// Writes `rebuilt` over the checked-in document in regeneration mode, and
/// otherwise fails when the two disagree, naming the command that refreshes it.
pub fn regenerate_or_compare(
    path: &Path,
    checked_in: &serde_json::Value,
    rebuilt: &serde_json::Value,
    what: &str,
    command: &str,
) {
    if regen_mode() {
        let text = format!(
            "{}\n",
            serde_json::to_string_pretty(rebuilt).expect("serialize the document")
        );
        std::fs::write(path, text)
            .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
        eprintln!("regenerated {what} at {}", path.display());
        return;
    }
    if checked_in != rebuilt {
        panic!(
            "{what} is stale: {} no longer matches what the reference derives from the fixture.\n\
             Regenerate it deliberately with:\n  {command}",
            path.display()
        );
    }
}

/// Reports a gate that could not run.
///
/// Development runs print the reason and continue (`SKIP`); the acceptance gate
/// (`TESSERA_ACCEPTANCE_STRICT=1`, which `scripts/acceptance_full_graph.sh
/// --strict` sets) fails instead, so an uncovered gate can never be mistaken for
/// a passing one.
pub fn uncovered(label: &str, reason: &str) {
    if strict_mode() {
        panic!(
            "{label}: required by the acceptance gate but unavailable, so the gate cannot pass:\n{}",
            reason.trim()
        );
    }
    eprintln!(
        "SKIP: {label}; this run does NOT satisfy that acceptance gate.\nreason: {}",
        reason.trim()
    );
}

/// Whether the error text is the documented fail-closed refusal of a plan the
/// RTL backend does not implement yet, as opposed to a new failure.
pub fn is_documented_backend_refusal(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    lowered.contains("single pointwise step")
        || lowered.contains("unsupported layout")
        || lowered.contains("v1 emits")
        || lowered.contains("outside the emitted envelope")
        || lowered.contains("not implemented")
        || lowered.contains("is not supported")
        || lowered.contains("is not emitted by")
}
