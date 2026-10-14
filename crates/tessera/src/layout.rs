//! Layout vocabulary, the persisted `LayoutArtifact`, and the backend-ready
//! `ExecutionPlan` projection.
//!
//! A [`LayoutArtifact`] is the reproducible output of layout search: the chosen
//! execution mode, engine, tile geometry, fusion decisions, buffer placement
//! (with the arena each buffer lives in and its byte offset), schedule steps
//! (with resolved padding, exact code intervals, and the MAC/cycle derivation
//! behind each step's estimate), resource/cycle/DDR estimates, slack, the
//! objective that ranked candidates, how much of the search space was actually
//! covered ([`PlanCertainty`]), and a summary of rejected alternatives —
//! including a per-mode one. It is self-describing (`format`,
//! `schema_version`, `provenance`) and self-contained: the resource limits it
//! was solved against are embedded so that [`LayoutArtifact::verify`] and
//! [`LayoutArtifact::execution_plan`] need no external inputs.
//!
//! [`LayoutArtifact::validate`] is deliberately strict, because the artifact is
//! the contract between the planner and the RTL backend: it rejects a plan whose
//! steps claim more lanes than the engine instantiates, whose MAC count cannot
//! be performed in the recorded lane iterations, whose cycle breakdown does not
//! follow from the schedule, whose buffer arenas overlap, or whose certainty
//! label claims more than the recorded search proved.
//!
//! Persistence uses the project TOML subset ([`crate::target::toml_lite`]) and
//! is deterministic: collections are normalized (sorted) before serialization,
//! so two equal artifacts always render byte-identically.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::target::toml_lite::{self, TomlError};

/// Layout artifact format tag.
pub const LAYOUT_FORMAT: &str = "tessera.layout";
/// Layout artifact schema version implemented by this build.
pub const LAYOUT_SCHEMA_VERSION: u32 = 1;
/// Execution-plan format tag.
pub const PLAN_FORMAT: &str = "tessera.execution_plan";
/// Execution-plan schema version implemented by this build.
pub const PLAN_SCHEMA_VERSION: u32 = 1;
/// Target format tag (re-exported so callers need not import `target`).
pub use crate::target::TARGET_FORMAT;
/// Target schema version supported by this build.
pub use crate::target::TARGET_SCHEMA_VERSION;

// ---------------------------------------------------------------------------
// Shared vocabulary
// ---------------------------------------------------------------------------

/// Stable tensor identity within one graph artifact.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct TensorId(pub u32);

impl TensorId {
    /// Construct a tensor id.
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    /// Raw index.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}

/// Stored element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElemType {
    /// Signed 8-bit.
    I8,
    /// Signed 4-bit packed codes (two per byte, low nibble first).
    I4,
    /// Signed 9-bit value: the unclamped sum of two signed 8-bit operands,
    /// range `-256..=255`.
    ///
    /// A residual `add` must not be clamped to int8 — the sum of two int8
    /// codes is representable only in nine bits — so the value is carried at
    /// 9-bit semantics in a 16-bit storage word. `bits()` reports the *value*
    /// width (which selects the 9-bit single-product MAC path on the target),
    /// while [`ElemType::storage_bits`] reports the transport width.
    I9,
    /// Signed 16-bit transport container.
    ///
    /// A container, not a width: a widened activation whose exact interval fits
    /// in nine bits still travels in an `i16` word, and the *interval* — not
    /// the container — is what sizes the datapath. `bits()` reports 16 so an
    /// unbounded `i16` cannot masquerade as an 8-bit operand.
    I16,
    /// Unsigned 8-bit.
    U8,
    /// Signed 32-bit accumulator or bias.
    I32,
    /// IEEE binary32 (analysis only; never a hardware operand in v1).
    F32,
}

impl ElemType {
    /// Value width in bits: the width of the arithmetic meaning of an element.
    pub const fn bits(self) -> u32 {
        match self {
            ElemType::I8 | ElemType::U8 => 8,
            ElemType::I4 => 4,
            ElemType::I9 => 9,
            ElemType::I16 => 16,
            ElemType::I32 | ElemType::F32 => 32,
        }
    }

    /// Storage width in bits: how many bits one element occupies in memory.
    ///
    /// Differs from [`ElemType::bits`] only where the value width is not a
    /// whole number of bytes: a 9-bit residual sum is transported in a 16-bit
    /// word, and a 4-bit code is packed two per byte.
    pub const fn storage_bits(self) -> u32 {
        match self {
            ElemType::I4 => 4,
            ElemType::I9 | ElemType::I16 => 16,
            other => other.bits(),
        }
    }

    /// Inclusive value range of one element.
    pub const fn value_range(self) -> (i32, i32) {
        match self {
            ElemType::I8 => (-128, 127),
            ElemType::I4 => (-8, 7),
            ElemType::I9 => (-256, 255),
            ElemType::I16 => (i16::MIN as i32, i16::MAX as i32),
            ElemType::U8 => (0, 255),
            ElemType::I32 => (i32::MIN, i32::MAX),
            ElemType::F32 => (i32::MIN, i32::MAX),
        }
    }

    /// Whether `self` can hold every value of `other` without clamping.
    pub const fn can_hold(self, other: ElemType) -> bool {
        let (lo_self, hi_self) = self.value_range();
        let (lo_other, hi_other) = other.value_range();
        lo_self <= lo_other && hi_self >= hi_other
    }
}

/// Inclusive interval of activation codes a tensor actually carries.
///
/// Exact, not a container width. A point written by a convolution holds
/// `[-qmax, qmax]` (or `[0, output_qmax]` for the unsigned head), while a point
/// written by a residual `add` holds the sum of its operands' intervals — on a
/// symmetric int8 model that is `[-254, 254]`, which needs nine signed bits.
/// The planner sizes its MAC path and the backend sizes its datapath from this
/// interval, never from the element container.
///
/// This mirrors `crate::mlir::artifact::CodeRange` field for field; the layout
/// vocabulary cannot depend on a stage module, so the contract is restated here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodeRange {
    /// Smallest code the tensor can hold.
    pub lo: i64,
    /// Largest code the tensor can hold.
    pub hi: i64,
}

impl CodeRange {
    /// Construct an interval.
    pub const fn new(lo: i64, hi: i64) -> Self {
        Self { lo, hi }
    }

    /// Narrowest signed container width, in bits, that holds the interval.
    pub fn signed_bits(&self) -> u32 {
        let bound = self.lo.unsigned_abs().max(self.hi.unsigned_abs());
        if bound == 0 {
            return 1;
        }
        (u64::BITS - bound.leading_zeros()) + 1
    }

    /// Interval of the sum of two intervals.
    pub const fn sum(self, other: Self) -> Self {
        Self {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Whether every code fits in a signed byte.
    pub const fn fits_i8(self) -> bool {
        self.lo >= i8::MIN as i64 && self.hi <= i8::MAX as i64
    }

    /// Whether the interval is non-empty.
    pub const fn is_valid(self) -> bool {
        self.lo <= self.hi
    }

    /// Whether `self` contains every code of `other`.
    pub const fn contains(self, other: Self) -> bool {
        self.lo <= other.lo && self.hi >= other.hi
    }
}

/// NCHW tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorShape {
    /// Batch (v1 models are batch-1).
    pub n: u32,
    /// Channels.
    pub c: u32,
    /// Height.
    pub h: u32,
    /// Width.
    pub w: u32,
}

impl TensorShape {
    /// Construct a shape.
    pub const fn new(n: u32, c: u32, h: u32, w: u32) -> Self {
        Self { n, c, h, w }
    }

    /// Total element count.
    pub fn elements(self) -> u64 {
        self.n as u64 * self.c as u64 * self.h as u64 * self.w as u64
    }

    /// Bytes needed to store this tensor at the given element type.
    pub fn bytes(self, elem: ElemType) -> u64 {
        // I4 packs two codes per byte; every other type occupies
        // `storage_bits / 8` bytes per element (I9 is transported in 16 bits).
        match elem {
            ElemType::I4 => self.elements().div_ceil(2),
            other => self.elements() * (other.storage_bits() as u64 / 8),
        }
    }

    /// Spatial extent `[height, width]`.
    pub const fn spatial(self) -> [u32; 2] {
        [self.h, self.w]
    }
}

/// A tensor reference carried through the artifact: identity, name, shape and
/// element type. Never tensor contents.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRef {
    /// Stable id.
    pub id: TensorId,
    /// Stable symbol name.
    pub name: String,
    /// Shape.
    pub shape: TensorShape,
    /// Stored element type.
    pub elem: ElemType,
}

impl TensorRef {
    /// Storage bytes of this tensor.
    pub fn bytes(&self) -> u64 {
        self.shape.bytes(self.elem)
    }
}

/// Operator kind in the lowered primitive graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    /// Dense 3x3 (or NxN) convolution.
    Conv2d,
    /// Depthwise convolution (`groups == in_channels`).
    DepthwiseConv2d,
    /// 1x1 convolution (`groups == 1`, kernel 1x1).
    PointwiseConv2d,
    /// Elementwise addition (residual/skip).
    Add,
    /// Standalone requantization.
    Requant,
    /// Standalone activation clamp.
    Activation,
    /// 2x nearest-neighbor upsampling.
    NearestUpsample2x,
    /// Final unsigned 8-bit output quantization.
    QuantizeOutput,
}

impl OpKind {
    /// Whether the operator performs a MAC reduction.
    pub const fn is_convolution(self) -> bool {
        matches!(
            self,
            OpKind::Conv2d | OpKind::DepthwiseConv2d | OpKind::PointwiseConv2d
        )
    }

    /// Whether the operator is a pure elementwise transform.
    pub const fn is_elementwise(self) -> bool {
        matches!(
            self,
            OpKind::Add | OpKind::Requant | OpKind::Activation | OpKind::QuantizeOutput
        )
    }
}

/// Physically distinct execution organisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// One specialized engine per operator.
    Dedicated,
    /// Layer-pipelined streaming engines with ready/valid handshakes.
    Streaming,
    /// One shared engine reused across operators with tile scheduling.
    SharedTile,
    /// Streaming elementwise paths plus a shared convolution engine.
    Hybrid,
}

impl ExecutionMode {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            ExecutionMode::Dedicated => "dedicated",
            ExecutionMode::Streaming => "streaming",
            ExecutionMode::SharedTile => "shared_tile",
            ExecutionMode::Hybrid => "hybrid",
        }
    }
}

/// Schedule granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Granularity {
    /// One pass over the whole frame.
    Frame,
    /// Row stripes.
    Row,
    /// Rectangular tiles.
    Tile,
}

/// Where a buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPlacement {
    /// Flip-flop registers.
    Register,
    /// Handshake FIFO / SRL storage.
    Fifo,
    /// Convolution line buffer.
    LineBuffer,
    /// Block RAM (RAM10).
    Bram,
    /// External DDR.
    Ddr,
}

/// One live range to be placed in a shared arena.
///
/// `start` and `end` are inclusive step indices: the value must be resident
/// from the step that produces it to the step that last consumes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaRange {
    /// First step at which the value must be resident.
    pub start: u64,
    /// Last step at which the value must be resident.
    pub end: u64,
    /// Words the value occupies.
    pub words: u64,
}

/// Deterministic lifetime allocation of a shared arena.
///
/// Returns the word offset of every range, in the input order, and the high
/// water mark the arena must hold. Ranges are placed in `(start, end, input
/// order)` order, each at the **lowest offset that does not conflict with an
/// already placed range whose live range overlaps its own**, so two ranges that
/// overlap in time never share a word.
///
/// The extent is the placement's real high water mark, and it is *not* the peak
/// simultaneous word count: with mixed sizes a range can be pushed above a hole
/// it does not fit into, and the general weighted interval storage placement
/// problem has no simple greedy that attains the peak. The plan therefore
/// charges the extent this function returns, and the backend must call this
/// same function — an allocator that assumes the peak is attainable corrupts
/// memory. A concrete case where summing the sizes of the ranges still live at
/// a range's start is wrong, and where this function is not: ranges
/// `(0,0,10)`, `(0,3,5)`, `(1,2,10)`. Summing gives the third range offset `5`,
/// overlapping the second at `[10,15)`; lowest-offset first fit gives `0`, `10`,
/// `15`.
pub fn allocate_arena(ranges: &[ArenaRange]) -> (Vec<u64>, u64) {
    let mut offsets = vec![0u64; ranges.len()];
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|index| (ranges[*index].start, ranges[*index].end, *index));

    // (offset, words, index) of every range already placed, in placement order.
    let mut placed: Vec<(u64, u64, usize)> = Vec::with_capacity(ranges.len());
    let mut high = 0u64;
    for index in order {
        let range = ranges[index];
        let mut candidates: Vec<u64> = Vec::with_capacity(placed.len() + 1);
        candidates.push(0);
        for (offset, words, _) in &placed {
            candidates.push(offset.saturating_add(*words));
        }
        candidates.sort_unstable();
        candidates.dedup();

        let fits = |candidate: u64| -> bool {
            placed.iter().all(|(offset, words, other)| {
                let placed_range = &ranges[*other];
                let conflicts = range.start <= placed_range.end && placed_range.start <= range.end;
                !conflicts
                    || candidate.saturating_add(range.words) <= *offset
                    || candidate >= offset.saturating_add(*words)
            })
        };
        // The candidate equal to the highest placed end always fits, so a
        // placement always exists; the fallback keeps the function total.
        let offset = candidates
            .iter()
            .copied()
            .find(|candidate| fits(*candidate))
            .unwrap_or(high);
        offsets[index] = offset;
        high = high.max(offset.saturating_add(range.words));
        placed.push((offset, range.words, index));
    }
    (offsets, high)
}

/// The address space a buffer is allocated in.
///
/// `placement` says *what kind* of storage a buffer uses; `region` says *which
/// arena* it occupies, which is what the backend needs to turn a buffer into a
/// concrete address range. Weights are read-only ROM, activations live in the
/// on-chip activation RAM, and graph I/O lives in the DDR frame-buffer arena.
///
/// The three read-only memories of the shared engine are *separate address
/// spaces*, not one ROM: their words are different widths, so a single byte
/// offset cannot describe a position in all of them at once. The engine emits
/// one `w_rom` (8-bit codes), one `b_rom` (accumulator-width bias words) and
/// one `m0_rom` (17-bit multipliers), each with its own physical RAM and its
/// own per-step base index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferRegion {
    /// Read-only weight ROM (weights resident on chip), one 8-bit code per
    /// entry whatever the manifest packing was.
    WeightRom,
    /// Read-only bias ROM: `bias_int32` words, one per output channel, in the
    /// engine's accumulator width.
    BiasRom,
    /// Read-only requantization ROM: the per-output-channel `M0` multipliers,
    /// 17 bits each.
    MultiplierRom,
    /// On-chip activation RAM (BRAM / line buffers / SRL).
    ActivationRam,
    /// DDR frame-buffer arena (graph inputs and outputs).
    FrameBuffer,
    /// Register/FIFO transport with no addressable storage.
    Stream,
}

impl BufferRegion {
    /// Whether buffers in this region occupy addressable storage.
    pub const fn is_addressable(self) -> bool {
        !matches!(self, BufferRegion::Stream)
    }
}

/// Concrete spatial padding applied by one step, resolved to input pixels.
///
/// The planner resolves the operator's declared padding (`zero`, `same`,
/// `reflect`) into explicit border sizes, so the backend never re-derives them
/// from a shape identity. `reflect` records the border fill rule: `true` means
/// the border is mirrored, `false` means it is zero-filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PaddingSpec {
    /// Padding above the input plane.
    pub top: u32,
    /// Padding below the input plane.
    pub bottom: u32,
    /// Padding left of the input plane.
    pub left: u32,
    /// Padding right of the input plane.
    pub right: u32,
    /// Reflect the border instead of zero-filling it.
    pub reflect: bool,
}

impl PaddingSpec {
    /// No padding.
    pub const NONE: Self = Self {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
        reflect: false,
    };

    /// Whether no border pixel is added.
    pub const fn is_zero(self) -> bool {
        self.top == 0 && self.bottom == 0 && self.left == 0 && self.right == 0
    }

    /// Largest border on any side.
    pub const fn max_extent(self) -> u32 {
        let h = if self.top > self.bottom {
            self.top
        } else {
            self.bottom
        };
        let w = if self.left > self.right {
            self.left
        } else {
            self.right
        };
        if h > w { h } else { w }
    }

    /// Total padded extent of a `[h, w]` input plane.
    pub const fn padded_extent(self, plane: [u32; 2]) -> [u32; 2] {
        [
            plane[0] + self.top + self.bottom,
            plane[1] + self.left + self.right,
        ]
    }
}

/// Requantization rounding policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rounding {
    /// Round half away from zero (`rnd_half_away`), the manifest contract.
    ///
    /// The offset is `2^(shift-1)` for a non-negative product and
    /// `2^(shift-1) - 1` for a negative one, so a negative tie moves away from
    /// zero rather than toward positive infinity: the two rules differ exactly
    /// at negative ties, which is why this variant is not called `half_up`.
    HalfAwayFromZero,
    /// Round half to even.
    HalfEven,
    /// Truncate toward negative infinity.
    Truncate,
}

/// Requant contract: normalized multiplier, right shift, rounding and the
/// quantizer's own output clamp applied after the multiply-shift.
///
/// `m0` is the scalar hardware operand (uniform across output channels). When
/// the source model declares per-output-channel multipliers they are carried
/// verbatim in `per_channel_multipliers` (same 17-bit encoding, one entry per
/// output channel, execution order); an empty vector means "uniform, use
/// `m0`". The planner never samples or rescales either form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requant {
    /// Normalized multiplier `M0` in `[0, 2^17 - 1]`.
    pub m0: i64,
    /// Post-multiply arithmetic right shift `S`, in `0..=31`.
    pub shift: i32,
    /// Rounding mode.
    pub rounding: Rounding,
    /// Inclusive lower clamp of the quantizer output.
    pub saturate_lo: i32,
    /// Inclusive upper clamp of the quantizer output.
    pub saturate_hi: i32,
    /// Per-output-channel multipliers, verbatim from the source artifact.
    /// Empty means uniform (all channels use `m0`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_channel_multipliers: Vec<i64>,
    /// Maximum relative error of the fixed-point requant approximation, in
    /// parts per million, when the source artifact declares one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiplier_error_ppm: Option<u32>,
}

/// Highest representable normalized requant multiplier (`2^17 - 1`).
pub const REQUANT_MAX_MULTIPLIER: i64 = (1 << 17) - 1;

impl Requant {
    /// A uniform requant contract.
    pub const fn uniform(m0: i64, shift: i32, rounding: Rounding, lo: i32, hi: i32) -> Self {
        Self {
            m0,
            shift,
            rounding,
            saturate_lo: lo,
            saturate_hi: hi,
            per_channel_multipliers: Vec::new(),
            multiplier_error_ppm: None,
        }
    }

    /// Validate the hardware-operand representation.
    pub fn validate(&self, out_channels: u32) -> Result<(), String> {
        if self.m0 < 0 || self.m0 > REQUANT_MAX_MULTIPLIER {
            return Err(format!(
                "requant m0={} is outside the 17-bit multiplier field [0, {}]",
                self.m0, REQUANT_MAX_MULTIPLIER
            ));
        }
        if !(0..=31).contains(&self.shift) {
            return Err(format!("requant shift={} must be in 0..=31", self.shift));
        }
        if self.saturate_lo > self.saturate_hi {
            return Err(format!(
                "requant saturate range [{}, {}] is empty",
                self.saturate_lo, self.saturate_hi
            ));
        }
        if !self.per_channel_multipliers.is_empty() {
            if self.per_channel_multipliers.len() != out_channels as usize {
                return Err(format!(
                    "per-channel multipliers: expected {} entries (out_channels), found {}",
                    out_channels,
                    self.per_channel_multipliers.len()
                ));
            }
            if let Some(bad) = self
                .per_channel_multipliers
                .iter()
                .find(|m| **m < 0 || **m > REQUANT_MAX_MULTIPLIER)
            {
                return Err(format!(
                    "per-channel multiplier {bad} is outside the 17-bit field [0, {}]",
                    REQUANT_MAX_MULTIPLIER
                ));
            }
        }
        Ok(())
    }
}

/// Activation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    /// Identity.
    None,
    /// ReLU: clamp below at zero.
    Relu,
    /// ReLU6: clamp to `[0, 6]` in the quantized domain.
    Relu6,
    /// Unsigned clamp to `[0, 255]`; only legal on the final output op.
    UnsignedClamp,
}

/// Activation attached to an operator output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Activation {
    /// Activation kind.
    pub kind: ActivationKind,
    /// Inclusive lower clamp.
    pub clamp_lo: i32,
    /// Inclusive upper clamp.
    pub clamp_hi: i32,
}

impl Activation {
    /// Identity activation over the given quantized range.
    pub const fn identity(clamp_lo: i32, clamp_hi: i32) -> Self {
        Self {
            kind: ActivationKind::None,
            clamp_lo,
            clamp_hi,
        }
    }

    /// The effective clamp after intersecting with a requant saturate range.
    pub fn effective_clamp(self, requant: Option<&Requant>) -> (i32, i32) {
        match requant {
            Some(r) => (
                self.clamp_lo.max(r.saturate_lo),
                self.clamp_hi.min(r.saturate_hi),
            ),
            None => (self.clamp_lo, self.clamp_hi),
        }
    }
}

/// Rectangular tile geometry in output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileGeometry {
    /// Tile width in output pixels.
    pub width: u32,
    /// Tile height in output pixels.
    pub height: u32,
}

impl TileGeometry {
    /// Construct a tile geometry.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Tile area in pixels.
    pub const fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// Convolution engine folding configuration.
///
/// `simd × pe × tap_parallel` is the number of multiply-accumulate lanes the
/// engine *physically instantiates*, and the datapath widths describe the
/// registers and memories around them. The RTL backend must instantiate
/// exactly these lanes and words: a plan that claims lanes the emitter
/// serializes would make every cycle estimate below a fiction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    /// Engine name (`shared_conv0`, `stream_dw0`, …).
    pub name: String,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Activation word width in the engine's activation memory, in bits.
    ///
    /// This is the storage width, not the manifest code width: the v1 shared
    /// engine keeps 9-bit signed residual sums, so its activation RAM stores
    /// signed 16-bit words.
    #[serde(default = "default_activation_bits")]
    pub activation_bits: u32,
    /// Accumulator width in bits.
    #[serde(default = "default_accumulator_bits")]
    pub accumulator_bits: u32,
    /// Stored weight word width in bits.
    #[serde(default = "default_weight_bits")]
    pub weight_bits: u32,
    /// Activation words the plan budgets for the engine's arena.
    ///
    /// The engine's activation storage is one pooled RAM, and a step-major
    /// sequential schedule only has to keep the tensors that are still live
    /// when it runs, so this is the high water mark of the lifetime allocation
    /// [`allocate_arena`] computes from the schedule's live ranges — not the sum
    /// of every activation buffer's depth. It is at least the peak simultaneous
    /// word count and it can be larger, because a range that does not fit a hole
    /// is placed above it. A backend that materializes the arena must fit inside
    /// this figure and must use the same allocator; reporting a smaller arena
    /// than the schedule needs would understate the structure. Zero for an
    /// organisation with no RAM arena (the legacy streaming envelope transports
    /// activations through registers and FIFOs).
    #[serde(default = "default_arena_words")]
    pub arena_words: u64,
}

fn default_arena_words() -> u64 {
    0
}

fn default_activation_bits() -> u32 {
    8
}

fn default_accumulator_bits() -> u32 {
    32
}

fn default_weight_bits() -> u32 {
    8
}

/// Default arena for a buffer deserialized without an explicit region:
/// on-chip activation RAM, the only region that always exists.
fn default_buffer_region() -> BufferRegion {
    BufferRegion::ActivationRam
}

impl EngineConfig {
    /// Total multiplier lanes.
    pub const fn lanes(&self) -> u64 {
        self.simd as u64 * self.pe as u64 * self.tap_parallel as u64
    }

    /// Whether the declared datapath widths are non-zero.
    pub const fn widths_are_sound(&self) -> bool {
        self.activation_bits > 0 && self.accumulator_bits > 0 && self.weight_bits > 0
    }
}

/// Objective metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveTerm {
    /// Minimize frame cycles.
    MinimizeCycles,
    /// Minimize total DDR bytes per frame.
    MinimizeDdrBytes,
    /// Minimize RAM10 blocks.
    MinimizeRam10,
    /// Minimize DSP48 blocks.
    MinimizeDsp,
    /// Minimize XLR / logic units.
    MinimizeLogic,
}

impl ObjectiveTerm {
    /// Extract this metric from a resource vector.
    pub fn value(self, estimate: &ResourceEstimate) -> u64 {
        match self {
            ObjectiveTerm::MinimizeCycles => estimate.frame_cycles,
            ObjectiveTerm::MinimizeDdrBytes => {
                estimate.ddr_read_bytes_per_frame + estimate.ddr_write_bytes_per_frame
            }
            ObjectiveTerm::MinimizeRam10 => estimate.ram10,
            ObjectiveTerm::MinimizeDsp => estimate.dsp48,
            ObjectiveTerm::MinimizeLogic => estimate.xlr,
        }
    }
}

/// Lexicographic objective: primary term, then ordered tie-breakers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveSpec {
    /// Primary metric.
    pub primary: ObjectiveTerm,
    /// Tie-breakers, most significant first.
    #[serde(default)]
    pub secondary: Vec<ObjectiveTerm>,
}

impl ObjectiveSpec {
    /// Rank vector for an estimate.
    pub fn rank(&self, estimate: &ResourceEstimate) -> Vec<u64> {
        std::iter::once(self.primary)
            .chain(self.secondary.iter().copied())
            .map(|term| term.value(estimate))
            .collect()
    }

    /// Whether `a` is strictly better than `b`.
    pub fn is_better(&self, a: &ResourceEstimate, b: &ResourceEstimate) -> bool {
        self.rank(a) < self.rank(b)
    }
}

/// Legal producer/consumer fusions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionKind {
    /// Depthwise → requant → pointwise in one tile, keeping the intermediate
    /// activation on chip.
    DepthwiseRequantPointwise,
    /// Short residual add fused into the consuming tile.
    ShortResidual,
}

impl FusionKind {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            FusionKind::DepthwiseRequantPointwise => "depthwise_requant_pointwise",
            FusionKind::ShortResidual => "short_residual",
        }
    }
}

/// Conditions that make a producer/consumer boundary illegal to fuse across.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionBoundary {
    /// The producer value has more than one consumer.
    FanoutValue,
    /// The pair spans a DDR round trip.
    CrossDdrBoundary,
}

impl FusionBoundary {
    /// Stable string name.
    pub const fn as_str(self) -> &'static str {
        match self {
            FusionBoundary::FanoutValue => "fanout_value",
            FusionBoundary::CrossDdrBoundary => "cross_ddr_boundary",
        }
    }
}

// ---------------------------------------------------------------------------
// Estimates, limits, slack
// ---------------------------------------------------------------------------

/// Resource and cost estimate for a plan, produced by the target model.
///
/// Also used as the embedded limit vector ([`PlanLimits`]) with identical
/// fields, so slack is a straight subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEstimate {
    /// DSP48 blocks.
    pub dsp48: u64,
    /// RAM10 blocks.
    pub ram10: u64,
    /// XLR / logic units.
    pub xlr: u64,
    /// Frame cycles.
    pub frame_cycles: u64,
    /// DDR bytes read per frame.
    pub ddr_read_bytes_per_frame: u64,
    /// DDR bytes written per frame.
    pub ddr_write_bytes_per_frame: u64,
}

impl ResourceEstimate {
    /// All-zero estimate.
    pub const ZERO: Self = Self {
        dsp48: 0,
        ram10: 0,
        xlr: 0,
        frame_cycles: 0,
        ddr_read_bytes_per_frame: 0,
        ddr_write_bytes_per_frame: 0,
    };

    /// Whether every resource fits within `limits`.
    pub fn fits_within(&self, limits: &ResourceEstimate) -> bool {
        self.dsp48 <= limits.dsp48
            && self.ram10 <= limits.ram10
            && self.xlr <= limits.xlr
            && self.frame_cycles <= limits.frame_cycles
            && self.ddr_read_bytes_per_frame <= limits.ddr_read_bytes_per_frame
            && self.ddr_write_bytes_per_frame <= limits.ddr_write_bytes_per_frame
    }

    /// Legible description of the first violated limit, if any.
    pub fn first_violation(&self, limits: &ResourceEstimate) -> Option<(&'static str, u64, u64)> {
        let checks = [
            ("dsp48", self.dsp48, limits.dsp48),
            ("ram10", self.ram10, limits.ram10),
            ("xlr", self.xlr, limits.xlr),
            ("frame_cycles", self.frame_cycles, limits.frame_cycles),
            (
                "ddr_read_bytes_per_frame",
                self.ddr_read_bytes_per_frame,
                limits.ddr_read_bytes_per_frame,
            ),
            (
                "ddr_write_bytes_per_frame",
                self.ddr_write_bytes_per_frame,
                limits.ddr_write_bytes_per_frame,
            ),
        ];
        checks.into_iter().find(|(_, used, limit)| used > limit)
    }
}

/// The resource limits a layout was solved against, embedded in the artifact.
pub type PlanLimits = ResourceEstimate;

/// Derate a cycle count by a declared MAC-array utilisation percentage.
///
/// The one place the planner and the backend must agree on: `iterations` MAC
/// sweeps at `utilisation_percent` percent efficiency cost
/// `ceil(iterations × 100 / utilisation)` cycles. Exposed so the RTL backend
/// derives its schedule length with the *same* function instead of restating
/// the formula.
pub fn derate_cycles(iterations: u64, utilisation_percent: u32) -> u64 {
    let util = utilisation_percent.max(1) as u64;
    iterations.saturating_mul(100).div_ceil(util)
}

/// Per-step MAC/cycle derivation, recorded so the estimate is auditable and
/// the backend can prove its generated schedule matches it.
///
/// The engine instantiates `lanes` multiply-accumulate lanes for the whole
/// shared engine. A step whose fold uses fewer lanes still runs the same
/// structure: each *lane iteration* is one sweep of the lane array, doing up to
/// `lanes` products. `macs` is the true product count, so
/// `macs <= lane_iterations × lanes` must hold — a plan that claimed more
/// parallelism than its MAC count allows is rejected at the artifact boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepCost {
    /// MAC lanes the engine instantiates for this step (`simd × pe × tap`).
    pub lanes: u64,
    /// Lane-array sweeps the step performs (one per cycle of the MAC sweep).
    pub lane_iterations: u64,
    /// Multiply-accumulate products actually performed (`<= lane_iterations × lanes`).
    pub macs: u64,
    /// Fixed per-step / per-tile setup cycles.
    pub setup_cycles: u64,
    /// Declared MAC-array utilisation applied to `lane_iterations`.
    pub utilisation_percent: u32,
}

impl Default for StepCost {
    fn default() -> Self {
        Self {
            lanes: 1,
            lane_iterations: 0,
            macs: 0,
            setup_cycles: 0,
            utilisation_percent: 100,
        }
    }
}

impl StepCost {
    /// Cycles spent in the MAC sweep.
    pub fn mac_cycles(&self) -> u64 {
        derate_cycles(self.lane_iterations, self.utilisation_percent)
    }

    /// Total cycles attributed to the step (MAC sweep plus setup).
    pub fn total_cycles(&self) -> u64 {
        self.mac_cycles().saturating_add(self.setup_cycles)
    }

    /// Whether the recorded parallelism is consistent with the recorded MAC
    /// count: at most `lanes` products per lane iteration.
    pub fn parallelism_is_sound(&self) -> bool {
        self.lanes > 0 && self.macs <= self.lane_iterations.saturating_mul(self.lanes)
    }
}

/// Frame-level cycle accounting, so an estimate can be decomposed and
/// cross-checked instead of being a single opaque number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CycleBreakdown {
    /// Declared MAC-array utilisation for this plan.
    pub utilisation_percent: u32,
    /// Sum of every step's derated MAC-sweep cycles.
    pub mac_cycles: u64,
    /// Sum of every step's fixed setup cycles.
    pub setup_cycles: u64,
    /// DMA cycles charged for non-overlapped DDR traffic.
    pub dma_cycles: u64,
    /// `mac_cycles + setup_cycles + dma_cycles`; equals `ResourceEstimate::frame_cycles`.
    pub total_cycles: u64,
}

impl Default for CycleBreakdown {
    fn default() -> Self {
        Self {
            utilisation_percent: 100,
            mac_cycles: 0,
            setup_cycles: 0,
            dma_cycles: 0,
            total_cycles: 0,
        }
    }
}

impl CycleBreakdown {
    /// Assemble a breakdown from its parts and compute the total.
    pub fn new(
        utilisation_percent: u32,
        mac_cycles: u64,
        setup_cycles: u64,
        dma_cycles: u64,
    ) -> Self {
        Self {
            utilisation_percent,
            mac_cycles,
            setup_cycles,
            dma_cycles,
            total_cycles: mac_cycles
                .saturating_add(setup_cycles)
                .saturating_add(dma_cycles),
        }
    }

    /// Whether `total_cycles` is the sum of its parts.
    pub const fn is_consistent(&self) -> bool {
        self.total_cycles == self.mac_cycles + self.setup_cycles + self.dma_cycles
    }
}

/// Signed headroom: `limit - used`, positive when a budget is respected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceSlack {
    /// DSP48 headroom.
    pub dsp48: i64,
    /// RAM10 headroom.
    pub ram10: i64,
    /// XLR headroom.
    pub xlr: i64,
    /// Frame-cycle headroom.
    pub frame_cycles: i64,
    /// DDR read headroom.
    pub ddr_read_bytes_per_frame: i64,
    /// DDR write headroom.
    pub ddr_write_bytes_per_frame: i64,
}

impl ResourceSlack {
    /// Slack of `used` against `limits`.
    pub fn between(used: &ResourceEstimate, limits: &PlanLimits) -> Self {
        fn d(limit: u64, used: u64) -> i64 {
            let v = limit as i128 - used as i128;
            v.clamp(i64::MIN as i128, i64::MAX as i128) as i64
        }
        Self {
            dsp48: d(limits.dsp48, used.dsp48),
            ram10: d(limits.ram10, used.ram10),
            xlr: d(limits.xlr, used.xlr),
            frame_cycles: d(limits.frame_cycles, used.frame_cycles),
            ddr_read_bytes_per_frame: d(
                limits.ddr_read_bytes_per_frame,
                used.ddr_read_bytes_per_frame,
            ),
            ddr_write_bytes_per_frame: d(
                limits.ddr_write_bytes_per_frame,
                used.ddr_write_bytes_per_frame,
            ),
        }
    }

    /// Whether every budget is respected.
    pub const fn all_non_negative(&self) -> bool {
        self.dsp48 >= 0
            && self.ram10 >= 0
            && self.xlr >= 0
            && self.frame_cycles >= 0
            && self.ddr_read_bytes_per_frame >= 0
            && self.ddr_write_bytes_per_frame >= 0
    }
}

// ---------------------------------------------------------------------------
// Provenance and artifact contents
// ---------------------------------------------------------------------------

/// Reproducibility metadata for a persisted artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// Content hash of the input graph artifact.
    pub graph_sha256: String,
    /// Content hash of the target description.
    pub target_sha256: String,
    /// Content hash of the policy description.
    pub policy_sha256: String,
    /// Compiler version string.
    pub compiler_version: String,
    /// Target name.
    pub target_name: String,
    /// Target revision.
    pub target_revision: String,
    /// Policy schema version.
    pub policy_schema_version: u32,
    /// Search seed (v1 search is deterministic; the seed is recorded and
    /// recomputed identically).
    pub seed: u64,
    /// Additional solver options that affect the result.
    #[serde(default)]
    pub options: BTreeMap<String, String>,
}

impl Provenance {
    /// Compute a stable combined key for cache invalidation.
    pub fn cache_key(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            self.graph_sha256,
            self.target_sha256,
            self.policy_sha256,
            self.compiler_version,
            self.target_name,
            self.target_revision,
            self.seed
        )
    }
}

/// A concrete plan was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    /// A legal plan meeting every hard constraint was found.
    Feasible,
    /// No legal plan exists under these target/policy constraints.
    Infeasible,
}

/// How much the search actually proved.
///
/// `PlanStatus` says *what* was found; `PlanCertainty` says how much of the
/// search space was covered, so a truncated search is never presented as a
/// proof. A truncated *feasible* plan is a best effort, not an optimum; a
/// truncated *infeasible* result is inconclusive, not proof that nothing fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanCertainty {
    /// The whole enumerated space was explored and a feasible plan exists;
    /// it is optimal for the recorded objective over that space.
    Optimal,
    /// A feasible plan was found, but the search hit its expansion cap: a
    /// better plan may exist.
    BestEffort,
    /// The whole enumerated space was explored and nothing is feasible.
    ProvenInfeasible,
    /// The search was interrupted before the space was covered; feasibility
    /// and optimality are both unknown.
    Inconclusive,
}

impl Default for PlanCertainty {
    fn default() -> Self {
        // The safe default: a deserialized artifact that does not say it
        // proved anything must not be read as an optimality claim.
        PlanCertainty::Inconclusive
    }
}

impl PlanCertainty {
    /// Derive the certainty from the outcome and whether the search finished.
    pub const fn from_search(status: PlanStatus, truncated: bool) -> Self {
        match (status, truncated) {
            (PlanStatus::Feasible, false) => PlanCertainty::Optimal,
            (PlanStatus::Feasible, true) => PlanCertainty::BestEffort,
            (PlanStatus::Infeasible, false) => PlanCertainty::ProvenInfeasible,
            (PlanStatus::Infeasible, true) => PlanCertainty::Inconclusive,
        }
    }

    /// Whether the search covered the whole enumerated space.
    pub const fn is_conclusive(self) -> bool {
        matches!(
            self,
            PlanCertainty::Optimal | PlanCertainty::ProvenInfeasible
        )
    }

    /// Whether this result may be reported as an optimum.
    pub const fn is_proven_optimal(self) -> bool {
        matches!(self, PlanCertainty::Optimal)
    }

    /// Stable string name used in diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            PlanCertainty::Optimal => "optimal",
            PlanCertainty::BestEffort => "best_effort",
            PlanCertainty::ProvenInfeasible => "proven_infeasible",
            PlanCertainty::Inconclusive => "inconclusive",
        }
    }
}

/// One scheduled operator execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScheduleStep {
    /// Monotone step index.
    pub step: u32,
    /// Stable operator/layer name.
    pub op: String,
    /// Operator kind.
    pub op_kind: OpKind,
    /// Engine name.
    pub engine: String,
    /// Schedule granularity.
    pub mode: Granularity,
    /// Tile geometry, when the granularity is tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Input channels.
    pub in_channels: u32,
    /// Output channels.
    pub out_channels: u32,
    /// Input spatial extent `[h, w]`.
    pub in_spatial: [u32; 2],
    /// Output spatial extent `[h, w]`.
    pub out_spatial: [u32; 2],
    /// Kernel extent `[kh, kw]`.
    pub kernel: [u32; 2],
    /// Convolution groups.
    pub groups: u32,
    /// Stride `[sh, sw]`.
    pub stride: [u32; 2],
    /// Concrete padding this step applies, resolved to input pixels.
    ///
    /// Present for spatial operators. Without it the backend cannot tell a
    /// mirrored border from a zero-filled one at equal geometry, so the padding
    /// rule is data in the plan, never re-derived by the emitter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub padding: Option<PaddingSpec>,
    /// Nearest-neighbour upsampling factor, for upsample steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upsample_scale: Option<u32>,
    /// Exact code interval of the activation this step reads.
    ///
    /// A residual sum spans `[-254, 254]` and must be multiplied at that width;
    /// the container type alone does not say so.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_code_range: Option<CodeRange>,
    /// Exact code interval of the activation this step writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_code_range: Option<CodeRange>,
    /// Requant contract, when the operator requantizes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requant: Option<Requant>,
    /// Activation applied to the output.
    pub activation: Activation,
    /// Weight tensor reference (never contents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<TensorRef>,
    /// Bias tensor reference (never contents).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<TensorRef>,
    /// Input tensors.
    pub inputs: Vec<TensorId>,
    /// Output tensors.
    pub outputs: Vec<TensorId>,
    /// Input buffer name (resolves into the artifact's `buffers`).
    pub input_buffer: String,
    /// Output buffer name (resolves into the artifact's `buffers`).
    pub output_buffer: String,
    /// Estimated cycles for this step.
    pub estimated_cycles: u64,
    /// MAC and cycle derivation behind `estimated_cycles`.
    #[serde(default)]
    pub cost: StepCost,
    /// Operators fused into this step.
    #[serde(default)]
    pub fused_with: Vec<String>,
}

/// A buffer binding produced by the planner and mapped by the target model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferBinding {
    /// Buffer name.
    pub name: String,
    /// Placement.
    pub placement: MemoryPlacement,
    /// Address arena the buffer occupies.
    #[serde(default = "default_buffer_region")]
    pub region: BufferRegion,
    /// Byte offset of the buffer inside its arena.
    ///
    /// Deterministic and validated non-overlapping within a region, so the
    /// backend can turn a buffer name into a concrete address without
    /// re-inventing an allocation order.
    #[serde(default)]
    pub base_offset_bytes: u64,
    /// Tensor held, when the buffer backs a graph tensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<TensorId>,
    /// Element width in bits.
    pub width_bits: u64,
    /// Number of words.
    pub depth: u64,
    /// Physical RAM10 blocks this buffer's own `width_bits × depth` geometry
    /// needs if it were its own memory (the same mapping the resource estimate
    /// uses, so the number is not a second guess).
    ///
    /// The engine's memories are *pooled*: one weight ROM, one bias ROM, one
    /// multiplier ROM and one activation arena each hold every step's slice of
    /// that region and are mapped once. This field is therefore the standalone
    /// figure for the slice, and the region total in
    /// [`ResourceEstimate::ram10`](crate::layout::ResourceEstimate) is the
    /// pooled — usually smaller — physical figure that the plan is judged on.
    pub ram10_blocks: u64,
    /// Storage bytes the buffer's words occupy, which is the span the region's
    /// offsets advance by. One 17-bit word is three bytes, so this is the byte
    /// footprint rather than `width × depth / 8`.
    pub bytes: u64,
    /// Read ports required.
    pub read_ports: u64,
    /// Write ports required.
    pub write_ports: u64,
    /// Ping-pong depth (1 = single-buffered).
    pub ping_pong_depth: u32,
    /// Producers of the value held.
    #[serde(default)]
    pub producers: Vec<String>,
    /// Consumers of the value held.
    #[serde(default)]
    pub consumers: Vec<String>,
}

/// Per-operator fold assignment in the selected plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpAssignment {
    /// Operator name.
    pub op: String,
    /// Operator kind.
    pub op_kind: OpKind,
    /// Input-channel SIMD lanes.
    pub simd: u32,
    /// Output-channel PE lanes.
    pub pe: u32,
    /// Kernel tap parallelism.
    pub tap_parallel: u32,
    /// Input buffer name.
    pub input_buffer: String,
    /// Output buffer name.
    pub output_buffer: String,
    /// Operators fused into this one.
    #[serde(default)]
    pub fused_with: Vec<String>,
}

/// The selected execution organisation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    /// Execution mode.
    pub execution_mode: ExecutionMode,
    /// Granularity.
    pub granularity: Granularity,
    /// Tile geometry when tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Engine folding configuration.
    pub engine: EngineConfig,
    /// Weight residency.
    pub weight_placement: MemoryPlacement,
    /// Activation/intermediate residency.
    pub intermediate_placement: MemoryPlacement,
    /// Whether legal fusions were applied.
    pub fusion_enabled: bool,
    /// Per-operator assignments, ordered by operator name.
    pub ops: Vec<OpAssignment>,
}

/// A fusion that the selected plan applied.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionDecision {
    /// Producer operator name.
    pub producer: String,
    /// Consumer operator name.
    pub consumer: String,
    /// Fusion kind.
    pub kind: FusionKind,
    /// DDR bytes per frame avoided by keeping the intermediate on chip.
    pub saved_ddr_bytes: u64,
}

/// Why a candidate was rejected.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    /// DSP48 budget exceeded.
    DspBudget { used: u64, budget: u64 },
    /// RAM10 budget exceeded.
    RamBudget { used: u64, budget: u64 },
    /// XLR budget exceeded.
    XlrBudget { used: u64, budget: u64 },
    /// Frame cycle budget exceeded.
    FrameCycles { used: u64, budget: u64 },
    /// DDR read budget exceeded.
    DdrReadBudget { used: u64, budget: u64 },
    /// DDR write budget exceeded.
    DdrWriteBudget { used: u64, budget: u64 },
    /// A fold factor does not divide the operator dimension.
    IllegalFold { op: String, detail: String },
    /// Operator kind is not supported by the v1 planner.
    UnsupportedOperator { op: String, kind: OpKind },
    /// Execution mode not permitted by policy.
    UnsupportedMode { mode: ExecutionMode },
    /// The mode is permitted but no v1 backend implements the architecture it
    /// describes. Recorded instead of emitting a plan the emitter cannot honor.
    UnsupportedArchitecture { mode: ExecutionMode, detail: String },
    /// Memory placement not permitted by policy.
    UnsupportedPlacement { placement: MemoryPlacement },
    /// The input graph is malformed.
    InvalidGraph { detail: String },
    /// A policy rule forbids the candidate.
    PolicyViolation { detail: String },
    /// Search exhausted without a feasible candidate.
    NoCandidate { detail: String },
}

impl RejectReason {
    /// Stable category used for counting and reporting.
    pub const fn category(&self) -> &'static str {
        match self {
            RejectReason::DspBudget { .. } => "dsp_budget",
            RejectReason::RamBudget { .. } => "ram_budget",
            RejectReason::XlrBudget { .. } => "xlr_budget",
            RejectReason::FrameCycles { .. } => "frame_cycles",
            RejectReason::DdrReadBudget { .. } => "ddr_read_budget",
            RejectReason::DdrWriteBudget { .. } => "ddr_write_budget",
            RejectReason::IllegalFold { .. } => "illegal_fold",
            RejectReason::UnsupportedOperator { .. } => "unsupported_operator",
            RejectReason::UnsupportedMode { .. } => "unsupported_mode",
            RejectReason::UnsupportedArchitecture { .. } => "unsupported_architecture",
            RejectReason::UnsupportedPlacement { .. } => "unsupported_placement",
            RejectReason::InvalidGraph { .. } => "invalid_graph",
            RejectReason::PolicyViolation { .. } => "policy_violation",
            RejectReason::NoCandidate { .. } => "no_candidate",
        }
    }
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::DspBudget { used, budget } => {
                write!(f, "DSP budget exceeded: {used} > {budget}")
            }
            RejectReason::RamBudget { used, budget } => {
                write!(f, "RAM10 budget exceeded: {used} > {budget}")
            }
            RejectReason::XlrBudget { used, budget } => {
                write!(f, "XLR budget exceeded: {used} > {budget}")
            }
            RejectReason::FrameCycles { used, budget } => {
                write!(f, "frame cycles exceeded: {used} > {budget}")
            }
            RejectReason::DdrReadBudget { used, budget } => {
                write!(f, "DDR read budget exceeded: {used} > {budget}")
            }
            RejectReason::DdrWriteBudget { used, budget } => {
                write!(f, "DDR write budget exceeded: {used} > {budget}")
            }
            RejectReason::IllegalFold { op, detail } => write!(f, "illegal fold on {op}: {detail}"),
            RejectReason::UnsupportedOperator { op, kind } => {
                write!(f, "unsupported operator {op} ({kind:?})")
            }
            RejectReason::UnsupportedMode { mode } => {
                write!(f, "execution mode {mode:?} not permitted")
            }
            RejectReason::UnsupportedArchitecture { mode, detail } => {
                write!(f, "execution mode {mode:?} is not implemented: {detail}")
            }
            RejectReason::UnsupportedPlacement { placement } => {
                write!(f, "placement {placement:?} not permitted")
            }
            RejectReason::InvalidGraph { detail } => write!(f, "invalid graph: {detail}"),
            RejectReason::PolicyViolation { detail } => write!(f, "policy violation: {detail}"),
            RejectReason::NoCandidate { detail } => write!(f, "no feasible candidate: {detail}"),
        }
    }
}

/// One rejected candidate, recorded for inspection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedCandidate {
    /// Candidate execution mode.
    pub execution_mode: ExecutionMode,
    /// Candidate granularity.
    pub granularity: Granularity,
    /// Candidate tile geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Candidate weight placement.
    pub weight_placement: MemoryPlacement,
    /// Candidate intermediate placement.
    pub intermediate_placement: MemoryPlacement,
    /// Whether fusions were enabled.
    pub fusion_enabled: bool,
    /// Why it was rejected.
    pub reason: RejectReason,
}

/// Outcome of the search restricted to one execution mode.
///
/// A single global rejection summary hides *why* a mode was unusable: with
/// per-mode accounting, "streaming needs 464 DSP48 and only 152 are usable"
/// becomes a readable fact instead of a truncated single-candidate result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeSummary {
    /// Candidate configurations enumerated for this mode.
    pub candidates: u64,
    /// Candidates that produced a feasible plan.
    pub feasible: u64,
    /// Rejections per [`RejectReason::category`].
    #[serde(default)]
    pub counts: BTreeMap<String, u64>,
    /// First rejection recorded for this mode, the sharpest diagnostic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_reason: Option<RejectReason>,
}

/// Bounded summary of the rejected search space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectionSummary {
    /// Number of rejected candidate configurations.
    pub rejected_candidates: u64,
    /// Count per [`RejectReason::category`].
    #[serde(default)]
    pub counts: BTreeMap<String, u64>,
    /// First few rejected candidates, in deterministic order.
    #[serde(default)]
    pub samples: Vec<RejectedCandidate>,
    /// Per-execution-mode outcome, keyed by [`ExecutionMode::as_str`].
    #[serde(default)]
    pub by_mode: BTreeMap<String, ModeSummary>,
}

/// Search statistics for reproducibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchStats {
    /// Candidate configurations explored.
    pub explored: u64,
    /// Node expansions performed by the branch-and-bound inner search.
    pub expanded_nodes: u64,
    /// Candidate cap from policy.
    pub max_candidates: u64,
    /// Node-expansion share each outer candidate may consume.
    #[serde(default)]
    pub per_candidate_budget: u64,
    /// Outer candidates whose inner search was cut by the budget.
    #[serde(default)]
    pub truncated_candidates: u64,
    /// Whether the cap stopped any search early.
    pub truncated: bool,
}

/// The persisted layout artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutArtifact {
    /// Must equal [`LAYOUT_FORMAT`].
    pub format: String,
    /// Must equal [`LAYOUT_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Feasibility status.
    pub status: PlanStatus,
    /// How much of the search space was covered.
    #[serde(default)]
    pub certainty: PlanCertainty,
    /// Reproducibility metadata.
    pub provenance: Provenance,
    /// Resource limits the search was performed against.
    pub limits: PlanLimits,
    /// Objective used to rank candidates.
    pub objective: ObjectiveSpec,
    /// Selected organisation, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    /// Resource estimate, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceEstimate>,
    /// Frame cycle decomposition behind `resources.frame_cycles`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycles: Option<CycleBreakdown>,
    /// Slack, present only when feasible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slack: Option<ResourceSlack>,
    /// Tensor table (activations, weights, graph I/O).
    #[serde(default)]
    pub tensors: Vec<TensorRef>,
    /// Schedule steps, ordered by `step`.
    #[serde(default)]
    pub schedule: Vec<ScheduleStep>,
    /// Buffers, ordered deterministically.
    #[serde(default)]
    pub buffers: Vec<BufferBinding>,
    /// Applied fusions, ordered deterministically.
    #[serde(default)]
    pub fusions: Vec<FusionDecision>,
    /// Rejected-candidate summary.
    pub rejections: RejectionSummary,
    /// Search statistics.
    pub search: SearchStats,
}

/// Errors produced while loading, validating or projecting a layout artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// TOML syntax or schema binding failure.
    Config(TomlError),
    /// Wrong `format` tag.
    WrongFormat { expected: String, found: String },
    /// Unsupported `schema_version`.
    UnsupportedSchema { found: u32, supported: u32 },
    /// Structural inconsistency.
    Invalid { detail: String },
    /// A resource exceeds the embedded limit.
    OverBudget {
        resource: &'static str,
        used: u64,
        budget: u64,
    },
    /// The artifact records an infeasible plan and cannot be projected.
    Infeasible { detail: String },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::Config(e) => write!(f, "{e}"),
            LayoutError::WrongFormat { expected, found } => {
                write!(
                    f,
                    "wrong format tag: expected {expected:?}, found {found:?}"
                )
            }
            LayoutError::UnsupportedSchema { found, supported } => write!(
                f,
                "unsupported schema_version {found} (this build supports {supported})"
            ),
            LayoutError::Invalid { detail } => write!(f, "invalid layout artifact: {detail}"),
            LayoutError::OverBudget {
                resource,
                used,
                budget,
            } => write!(f, "over budget: {resource} used {used} > limit {budget}"),
            LayoutError::Infeasible { detail } => {
                write!(f, "layout artifact is infeasible: {detail}")
            }
        }
    }
}

impl std::error::Error for LayoutError {}

impl From<TomlError> for LayoutError {
    fn from(e: TomlError) -> Self {
        LayoutError::Config(e)
    }
}

impl LayoutArtifact {
    /// Parse an artifact from TOML text, normalize it, and validate it.
    pub fn from_toml_str(src: &str) -> Result<Self, LayoutError> {
        let mut artifact: LayoutArtifact = toml_lite::from_toml_str(src)?;
        artifact.normalize();
        artifact.validate()?;
        Ok(artifact)
    }

    /// Read and validate an artifact from disk.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, LayoutError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot read {}: {e}", path.as_ref().display()),
        })?;
        Self::from_toml_str(&text)
    }

    /// Deterministic TOML rendering (normalized, then serialized).
    pub fn to_toml_string(&self) -> Result<String, LayoutError> {
        let mut normalized = self.clone();
        normalized.normalize();
        Ok(toml_lite::to_toml_string_of(&normalized)?)
    }

    /// Write the artifact to disk as TOML.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), LayoutError> {
        let text = self.to_toml_string()?;
        std::fs::write(path.as_ref(), text).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot write {}: {e}", path.as_ref().display()),
        })
    }

    /// JSON rendering, for tooling that prefers JSON.
    pub fn to_json_string(&self) -> Result<String, LayoutError> {
        let mut normalized = self.clone();
        normalized.normalize();
        serde_json::to_string_pretty(&normalized).map_err(|e| LayoutError::Invalid {
            detail: format!("cannot encode JSON: {e}"),
        })
    }

    /// Parse an artifact from JSON text, normalize it, and validate it.
    pub fn from_json_str(src: &str) -> Result<Self, LayoutError> {
        let mut artifact: LayoutArtifact =
            serde_json::from_str(src).map_err(|e| LayoutError::Invalid {
                detail: format!("cannot decode JSON: {e}"),
            })?;
        artifact.normalize();
        artifact.validate()?;
        Ok(artifact)
    }

    /// Put every collection in deterministic order.
    pub fn normalize(&mut self) {
        self.tensors.sort_by_key(|t| t.id);
        self.schedule.sort_by_key(|s| s.step);
        self.buffers.sort_by(|a, b| {
            (a.region, a.placement, &a.name).cmp(&(b.region, b.placement, &b.name))
        });
        self.fusions.sort();
        if let Some(selection) = &mut self.selection {
            selection.ops.sort_by(|a, b| a.op.cmp(&b.op));
            for step in &mut self.schedule {
                step.fused_with.sort();
            }
        }
        self.rejections.samples.sort();
    }

    /// Recompute the frame cycle decomposition implied by the schedule.
    fn recomputed_cycles(&self) -> CycleBreakdown {
        let mut mac_cycles = 0u64;
        let mut setup_cycles = 0u64;
        for step in &self.schedule {
            mac_cycles = mac_cycles.saturating_add(step.cost.mac_cycles());
            setup_cycles = setup_cycles.saturating_add(step.cost.setup_cycles);
        }
        CycleBreakdown::new(
            self.cycles.map(|c| c.utilisation_percent).unwrap_or(100),
            mac_cycles,
            setup_cycles,
            self.cycles.map(|c| c.dma_cycles).unwrap_or(0),
        )
    }

    /// Validate the artifact's structure and, when feasible, its budgets.
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.format != LAYOUT_FORMAT {
            return Err(LayoutError::WrongFormat {
                expected: LAYOUT_FORMAT.into(),
                found: self.format.clone(),
            });
        }
        if self.schema_version != LAYOUT_SCHEMA_VERSION {
            return Err(LayoutError::UnsupportedSchema {
                found: self.schema_version,
                supported: LAYOUT_SCHEMA_VERSION,
            });
        }

        // The recorded certainty must follow from the recorded search outcome:
        // a truncated search can never be labeled optimal or proven-infeasible,
        // and a complete one must not hide behind a weaker claim.
        let expected_certainty = PlanCertainty::from_search(self.status, self.search.truncated);
        if self.certainty != expected_certainty {
            return Err(LayoutError::Invalid {
                detail: format!(
                    "certainty {} does not match the recorded search (status {:?}, truncated {}); \
                     expected {}",
                    self.certainty.as_str(),
                    self.status,
                    self.search.truncated,
                    expected_certainty.as_str()
                ),
            });
        }

        // Schedule ordering, buffer resolution and per-step cycle derivation.
        let mut last: Option<u32> = None;
        for step in &self.schedule {
            if let Some(prev) = last {
                if step.step <= prev {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "schedule steps are not strictly increasing: {} after {}",
                            step.step, prev
                        ),
                    });
                }
            }
            last = Some(step.step);
            if let Some(requant) = &step.requant {
                requant
                    .validate(step.out_channels)
                    .map_err(|detail| LayoutError::Invalid {
                        detail: format!("step {} ({}): {detail}", step.step, step.op),
                    })?;
            }
            for buffer in [&step.input_buffer, &step.output_buffer] {
                if !buffer.is_empty() && !self.buffers.iter().any(|b| &b.name == buffer) {
                    return Err(LayoutError::Invalid {
                        detail: format!("step {} references unknown buffer {buffer:?}", step.step),
                    });
                }
            }
            // The recorded fold is the physical lane count; the recorded cost
            // must not claim more parallelism than the MAC count licenses.
            let lanes = u64::from(step.simd)
                .saturating_mul(u64::from(step.pe))
                .saturating_mul(u64::from(step.tap_parallel));
            if step.cost.lanes != lanes {
                return Err(LayoutError::Invalid {
                    detail: format!(
                        "step {} ({}): cost.lanes {} does not match simd*pe*tap {}",
                        step.step, step.op, step.cost.lanes, lanes
                    ),
                });
            }
            if !step.cost.parallelism_is_sound() {
                return Err(LayoutError::Invalid {
                    detail: format!(
                        "step {} ({}): {} MACs cannot be performed in {} lane iterations of {} \
                         lanes without claiming parallelism the plan does not instantiate",
                        step.step,
                        step.op,
                        step.cost.macs,
                        step.cost.lane_iterations,
                        step.cost.lanes
                    ),
                });
            }
            if step.estimated_cycles != step.cost.total_cycles() {
                return Err(LayoutError::Invalid {
                    detail: format!(
                        "step {} ({}): estimated_cycles {} does not match the recorded cost \
                         derivation {}",
                        step.step,
                        step.op,
                        step.estimated_cycles,
                        step.cost.total_cycles()
                    ),
                });
            }
            if let Some(cycles) = self.cycles {
                if step.cost.utilisation_percent != cycles.utilisation_percent {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "step {} ({}): utilisation {} differs from the frame utilisation {}",
                            step.step,
                            step.op,
                            step.cost.utilisation_percent,
                            cycles.utilisation_percent
                        ),
                    });
                }
            }
            if step.padding.is_some()
                && !step.op_kind.is_convolution()
                && step.op_kind != OpKind::NearestUpsample2x
            {
                return Err(LayoutError::Invalid {
                    detail: format!(
                        "step {} ({}): padding is only meaningful on spatial operators",
                        step.step, step.op
                    ),
                });
            }
        }

        // Buffer arenas: addresses inside one region must not overlap, and a
        // stream buffer has no addressable storage.
        let mut by_region: BTreeMap<BufferRegion, Vec<&BufferBinding>> = BTreeMap::new();
        for buffer in &self.buffers {
            if buffer.bytes == 0 {
                return Err(LayoutError::Invalid {
                    detail: format!("buffer {:?} has zero bytes", buffer.name),
                });
            }
            if buffer.region.is_addressable() {
                by_region.entry(buffer.region).or_default().push(buffer);
            } else if buffer.base_offset_bytes != 0 {
                return Err(LayoutError::Invalid {
                    detail: format!(
                        "stream buffer {:?} must not carry an addressable offset",
                        buffer.name
                    ),
                });
            }
        }
        for (region, buffers) in by_region {
            let mut ranges: Vec<(u64, u64, &str)> = buffers
                .iter()
                .map(|b| {
                    (
                        b.base_offset_bytes,
                        b.base_offset_bytes + b.bytes,
                        b.name.as_str(),
                    )
                })
                .collect();
            ranges.sort_unstable();
            // The activation arena is allocated by tensor lifetime
            // (`allocate_arena`), so two activations may share words — but only
            // when their live ranges are disjoint. The lifetimes are derived
            // from the schedule here, not trusted from the planner: a tensor is
            // live from the step that produces it (from the start for a graph
            // input, which no step produces) to the last step that reads it
            // (to the end for a tensor no step reads, i.e. a graph output).
            if region == BufferRegion::ActivationRam {
                let mut first: BTreeMap<TensorId, u32> = BTreeMap::new();
                let mut last: BTreeMap<TensorId, u32> = BTreeMap::new();
                let mut final_step = 0u32;
                for step in &self.schedule {
                    final_step = final_step.max(step.step);
                    for output in &step.outputs {
                        first.entry(*output).or_insert(step.step);
                    }
                    for input in &step.inputs {
                        last.insert(*input, step.step);
                    }
                }
                let lifetime = |buffer: &BufferBinding| -> (u32, u32) {
                    match buffer.tensor {
                        Some(id) => (
                            first.get(&id).copied().unwrap_or(0),
                            last.get(&id).copied().unwrap_or(final_step),
                        ),
                        // A buffer that backs no tensor cannot be proven dead,
                        // so it is live for the whole schedule.
                        None => (0, final_step),
                    }
                };
                let indexed: Vec<(u64, u64, usize)> = buffers
                    .iter()
                    .enumerate()
                    .map(|(index, b)| (b.base_offset_bytes, b.base_offset_bytes + b.bytes, index))
                    .collect();
                let mut indexed = indexed;
                indexed.sort_unstable();
                for a in 0..indexed.len() {
                    for b in (a + 1)..indexed.len() {
                        let (start_a, end_a, index_a) = indexed[a];
                        let (start_b, end_b, index_b) = indexed[b];
                        if start_b >= end_a {
                            continue;
                        }
                        let name_a = buffers[index_a].name.as_str();
                        let name_b = buffers[index_b].name.as_str();
                        let (live_from_a, live_to_a) = lifetime(&buffers[index_a]);
                        let (live_from_b, live_to_b) = lifetime(&buffers[index_b]);
                        if live_from_a <= live_to_b && live_from_b <= live_to_a {
                            return Err(LayoutError::Invalid {
                                detail: format!(
                                    "ActivationRam buffers {name_a:?} [{}..{}] and {name_b:?} \
                                     [{}..{}] share words but are live at the same time \
                                     (steps {}..{} and {}..{})",
                                    start_a,
                                    end_a,
                                    start_b,
                                    end_b,
                                    live_from_a,
                                    live_to_a,
                                    live_from_b,
                                    live_to_b
                                ),
                            });
                        }
                    }
                }
                continue;
            }
            for pair in ranges.windows(2) {
                let (start_a, end_a, name_a) = pair[0];
                let (start_b, _, name_b) = pair[1];
                if end_a > start_b {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "{region:?} buffers {name_a:?} [{}..{}] and {name_b:?} (starts {}) \
                             overlap",
                            start_a, end_a, start_b
                        ),
                    });
                }
            }
        }

        // Tensor-table consistency for weight/bias references.
        for step in &self.schedule {
            for reference in [&step.weights, &step.bias].into_iter().flatten() {
                if !self.tensors.iter().any(|t| t.id == reference.id) {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "step {} references tensor {} absent from the tensor table",
                            step.step, reference.id
                        ),
                    });
                }
            }
        }

        match self.status {
            PlanStatus::Feasible => {
                let selection = self
                    .selection
                    .as_ref()
                    .ok_or_else(|| LayoutError::Invalid {
                        detail: "feasible artifact has no selection".into(),
                    })?;
                let resources = self
                    .resources
                    .as_ref()
                    .ok_or_else(|| LayoutError::Invalid {
                        detail: "feasible artifact has no resource estimate".into(),
                    })?;
                let cycles = self.cycles.ok_or_else(|| LayoutError::Invalid {
                    detail: "feasible artifact has no cycle breakdown".into(),
                })?;
                let slack = self.slack.as_ref().ok_or_else(|| LayoutError::Invalid {
                    detail: "feasible artifact has no slack report".into(),
                })?;
                if self.schedule.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact has an empty schedule".into(),
                    });
                }
                if selection.ops.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact has no operator assignments".into(),
                    });
                }
                if !selection.engine.widths_are_sound() {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact declares a zero-width engine datapath".into(),
                    });
                }
                if selection.engine.lanes() == 0 {
                    return Err(LayoutError::Invalid {
                        detail: "feasible artifact instantiates zero engine lanes".into(),
                    });
                }
                for step in &self.schedule {
                    let lanes = u64::from(step.simd)
                        .saturating_mul(u64::from(step.pe))
                        .saturating_mul(u64::from(step.tap_parallel));
                    if lanes > selection.engine.lanes() {
                        return Err(LayoutError::Invalid {
                            detail: format!(
                                "step {} ({}) uses {lanes} lanes but the engine only instantiates \
                                 {}",
                                step.step,
                                step.op,
                                selection.engine.lanes()
                            ),
                        });
                    }
                }
                if !cycles.is_consistent() {
                    return Err(LayoutError::Invalid {
                        detail: "cycle breakdown total is not the sum of its parts".into(),
                    });
                }
                let recomputed = self.recomputed_cycles();
                if recomputed.mac_cycles != cycles.mac_cycles
                    || recomputed.setup_cycles != cycles.setup_cycles
                    || recomputed.total_cycles != cycles.total_cycles
                {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "cycle breakdown does not match the schedule: recorded {} MAC + {} \
                             setup + {} dma, derived {} MAC + {} setup",
                            cycles.mac_cycles,
                            cycles.setup_cycles,
                            cycles.dma_cycles,
                            recomputed.mac_cycles,
                            recomputed.setup_cycles
                        ),
                    });
                }
                if cycles.total_cycles != resources.frame_cycles {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "frame cycle estimate {} does not match the cycle breakdown total {}",
                            resources.frame_cycles, cycles.total_cycles
                        ),
                    });
                }
                if let Some((resource, used, budget)) = resources.first_violation(&self.limits) {
                    return Err(LayoutError::OverBudget {
                        resource,
                        used,
                        budget,
                    });
                }
                let recomputed = ResourceSlack::between(resources, &self.limits);
                if &recomputed != slack || !slack.all_non_negative() {
                    return Err(LayoutError::Invalid {
                        detail: "slack report is inconsistent with the resource estimate".into(),
                    });
                }
            }
            PlanStatus::Infeasible => {
                if self.selection.is_some()
                    || self.resources.is_some()
                    || self.slack.is_some()
                    || self.cycles.is_some()
                {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must not carry a selection".into(),
                    });
                }
                if !self.schedule.is_empty() {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must not carry a schedule".into(),
                    });
                }
                if self.rejections.rejected_candidates == 0 {
                    return Err(LayoutError::Invalid {
                        detail: "infeasible artifact must explain why no candidate was legal"
                            .into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Re-check every hard constraint against the embedded limits.
    pub fn verify(&self) -> Result<(), LayoutError> {
        self.validate()?;
        if self.status != PlanStatus::Feasible {
            return Err(LayoutError::Infeasible {
                detail: "artifact records an infeasible plan".into(),
            });
        }
        Ok(())
    }

    /// Whether the search covered the whole enumerated space and found the
    /// optimum for the recorded objective.
    pub fn is_proven_optimal(&self) -> bool {
        self.certainty.is_proven_optimal()
    }

    /// Whether the outcome is a proof (optimal, or that nothing is feasible).
    pub fn is_conclusive(&self) -> bool {
        self.certainty.is_conclusive()
    }

    /// Project a validated artifact onto the backend-ready [`ExecutionPlan`].
    pub fn execution_plan(&self) -> Result<ExecutionPlan, LayoutError> {
        self.verify()?;
        let selection = self
            .selection
            .as_ref()
            .ok_or_else(|| LayoutError::Invalid {
                detail: "feasible artifact has no selection".into(),
            })?;
        let resources = self.resources.ok_or_else(|| LayoutError::Invalid {
            detail: "feasible artifact has no resource estimate".into(),
        })?;
        let cycles = self.cycles.ok_or_else(|| LayoutError::Invalid {
            detail: "feasible artifact has no cycle breakdown".into(),
        })?;

        // Deterministic weight/bias tensor table: unique references by id.
        let mut weights: Vec<TensorRef> = Vec::new();
        for step in &self.schedule {
            for reference in [&step.weights, &step.bias].into_iter().flatten() {
                if !weights.iter().any(|w| w.id == reference.id) {
                    weights.push(reference.clone());
                }
            }
        }
        weights.sort_by_key(|w| w.id);

        Ok(ExecutionPlan {
            format: PLAN_FORMAT.into(),
            schema_version: PLAN_SCHEMA_VERSION,
            certainty: self.certainty,
            provenance: self.provenance.clone(),
            execution_mode: selection.execution_mode,
            engine: selection.engine.clone(),
            tile: selection.tile,
            steps: self.schedule.clone(),
            buffers: self.buffers.clone(),
            tensors: self.tensors.clone(),
            weights,
            resources,
            cycles,
        })
    }
}

/// Backend-ready projection of a validated layout artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    /// Must equal [`PLAN_FORMAT`].
    pub format: String,
    /// Must equal [`PLAN_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// How much of the search space the layout covered.
    #[serde(default)]
    pub certainty: PlanCertainty,
    /// Provenance inherited from the layout.
    pub provenance: Provenance,
    /// Execution mode.
    pub execution_mode: ExecutionMode,
    /// Engine configuration.
    pub engine: EngineConfig,
    /// Tile geometry, when tiled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<TileGeometry>,
    /// Schedule steps, ordered by `step`.
    pub steps: Vec<ScheduleStep>,
    /// Buffers.
    pub buffers: Vec<BufferBinding>,
    /// Tensor table.
    pub tensors: Vec<TensorRef>,
    /// Weight and bias tensors, ordered by id.
    pub weights: Vec<TensorRef>,
    /// Resource estimate.
    pub resources: ResourceEstimate,
    /// Frame cycle decomposition behind `resources.frame_cycles`.
    pub cycles: CycleBreakdown,
}

impl ExecutionPlan {
    /// Deterministic serialization and re-parse, primarily for RTL metadata.
    pub fn to_toml_string(&self) -> Result<String, LayoutError> {
        Ok(toml_lite::to_toml_string_of(self)?)
    }

    /// Parse an execution plan from TOML text.
    pub fn from_toml_str(src: &str) -> Result<Self, LayoutError> {
        let plan: ExecutionPlan = toml_lite::from_toml_str(src)?;
        if plan.format != PLAN_FORMAT {
            return Err(LayoutError::WrongFormat {
                expected: PLAN_FORMAT.into(),
                found: plan.format.clone(),
            });
        }
        if plan.schema_version != PLAN_SCHEMA_VERSION {
            return Err(LayoutError::UnsupportedSchema {
                found: plan.schema_version,
                supported: PLAN_SCHEMA_VERSION,
            });
        }
        // Stage consumers validate before use: a plan whose cycle accounting
        // does not add up, or whose steps reference absent buffers, is not
        // consumable even though it parsed.
        if plan.steps.is_empty() {
            return Err(LayoutError::Invalid {
                detail: "execution plan has no steps".into(),
            });
        }
        if !plan.cycles.is_consistent() {
            return Err(LayoutError::Invalid {
                detail: "execution plan cycle breakdown is not the sum of its parts".into(),
            });
        }
        if plan.cycles.total_cycles != plan.resources.frame_cycles {
            return Err(LayoutError::Invalid {
                detail: format!(
                    "execution plan frame cycles {} do not match the cycle breakdown total {}",
                    plan.resources.frame_cycles, plan.cycles.total_cycles
                ),
            });
        }
        for step in &plan.steps {
            for name in [&step.input_buffer, &step.output_buffer] {
                if !name.is_empty() && !plan.buffers.iter().any(|b| &b.name == name) {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "execution plan step {} references unknown buffer {name:?}",
                            step.step
                        ),
                    });
                }
            }
        }
        let mut last: Option<u32> = None;
        for step in &plan.steps {
            if let Some(prev) = last {
                if step.step <= prev {
                    return Err(LayoutError::Invalid {
                        detail: format!(
                            "execution plan steps are not strictly increasing: {} after {}",
                            step.step, prev
                        ),
                    });
                }
            }
            last = Some(step.step);
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> PlanLimits {
        PlanLimits {
            dsp48: 152,
            ram10: 88,
            xlr: 40_568,
            frame_cycles: 10_000_000,
            ddr_read_bytes_per_frame: 90_000_000,
            ddr_write_bytes_per_frame: 70_000_000,
        }
    }

    fn feasible_artifact() -> LayoutArtifact {
        let tensors = vec![
            TensorRef {
                id: TensorId(0),
                name: "image".into(),
                shape: TensorShape::new(1, 3, 480, 640),
                elem: ElemType::I8,
            },
            TensorRef {
                id: TensorId(1),
                name: "enc.0.w".into(),
                shape: TensorShape::new(32, 3, 3, 3),
                elem: ElemType::I8,
            },
            TensorRef {
                id: TensorId(2),
                name: "enc.0.b".into(),
                shape: TensorShape::new(32, 1, 1, 1),
                elem: ElemType::I32,
            },
        ];
        let resources = ResourceEstimate {
            dsp48: 96,
            ram10: 80,
            xlr: 12_000,
            frame_cycles: 9_866_272,
            ddr_read_bytes_per_frame: 921_600,
            ddr_write_bytes_per_frame: 921_600,
        };
        LayoutArtifact {
            format: LAYOUT_FORMAT.into(),
            schema_version: LAYOUT_SCHEMA_VERSION,
            status: PlanStatus::Feasible,
            certainty: PlanCertainty::Optimal,
            provenance: Provenance {
                graph_sha256: "g".into(),
                target_sha256: "t".into(),
                policy_sha256: "p".into(),
                compiler_version: "0.1.0".into(),
                target_name: "ti60f225".into(),
                target_revision: "r1".into(),
                policy_schema_version: 1,
                seed: 1,
                options: BTreeMap::new(),
            },
            limits: limits(),
            objective: ObjectiveSpec {
                primary: ObjectiveTerm::MinimizeCycles,
                secondary: vec![
                    ObjectiveTerm::MinimizeDdrBytes,
                    ObjectiveTerm::MinimizeRam10,
                ],
            },
            selection: Some(Selection {
                execution_mode: ExecutionMode::SharedTile,
                granularity: Granularity::Tile,
                tile: Some(TileGeometry::new(80, 60)),
                engine: EngineConfig {
                    name: "shared_conv0".into(),
                    simd: 8,
                    pe: 8,
                    tap_parallel: 1,
                    activation_bits: 8,
                    accumulator_bits: 32,
                    weight_bits: 8,
                    arena_words: 0,
                },
                weight_placement: MemoryPlacement::Bram,
                intermediate_placement: MemoryPlacement::Ddr,
                fusion_enabled: true,
                ops: vec![OpAssignment {
                    op: "enc.0".into(),
                    op_kind: OpKind::Conv2d,
                    simd: 8,
                    pe: 8,
                    tap_parallel: 1,
                    input_buffer: "image_buf".into(),
                    output_buffer: "enc0_out".into(),
                    fused_with: vec![],
                }],
            }),
            resources: Some(resources),
            cycles: Some(CycleBreakdown::new(100, 9_866_272, 0, 0)),
            slack: Some(ResourceSlack::between(&resources, &limits())),
            tensors,
            schedule: vec![ScheduleStep {
                step: 0,
                op: "enc.0".into(),
                op_kind: OpKind::Conv2d,
                engine: "shared_conv0".into(),
                mode: Granularity::Tile,
                tile: Some(TileGeometry::new(80, 60)),
                simd: 8,
                pe: 8,
                tap_parallel: 1,
                in_channels: 3,
                out_channels: 32,
                in_spatial: [480, 640],
                out_spatial: [480, 640],
                kernel: [3, 3],
                groups: 1,
                stride: [1, 1],
                padding: Some(PaddingSpec {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                    reflect: false,
                }),
                upsample_scale: None,
                in_code_range: Some(CodeRange::new(-128, 127)),
                out_code_range: Some(CodeRange::new(0, 127)),
                requant: Some(Requant {
                    m0: 98_304,
                    shift: 16,
                    rounding: Rounding::HalfAwayFromZero,
                    saturate_lo: -128,
                    saturate_hi: 127,
                    per_channel_multipliers: Vec::new(),
                    multiplier_error_ppm: None,
                }),
                activation: Activation {
                    kind: ActivationKind::Relu,
                    clamp_lo: 0,
                    clamp_hi: 127,
                },
                weights: Some(TensorRef {
                    id: TensorId(1),
                    name: "enc.0.w".into(),
                    shape: TensorShape::new(32, 3, 3, 3),
                    elem: ElemType::I8,
                }),
                bias: Some(TensorRef {
                    id: TensorId(2),
                    name: "enc.0.b".into(),
                    shape: TensorShape::new(32, 1, 1, 1),
                    elem: ElemType::I32,
                }),
                inputs: vec![TensorId(0)],
                outputs: vec![],
                input_buffer: "image_buf".into(),
                output_buffer: "enc0_out".into(),
                estimated_cycles: 9_866_272,
                cost: StepCost {
                    lanes: 64,
                    lane_iterations: 9_866_272,
                    macs: 9_866_272,
                    setup_cycles: 0,
                    utilisation_percent: 100,
                },
                fused_with: vec![],
            }],
            buffers: vec![
                BufferBinding {
                    name: "image_buf".into(),
                    placement: MemoryPlacement::Ddr,
                    region: BufferRegion::FrameBuffer,
                    base_offset_bytes: 0,
                    tensor: Some(TensorId(0)),
                    width_bits: 8,
                    depth: 921_600,
                    ram10_blocks: 0,
                    bytes: 921_600,
                    read_ports: 1,
                    write_ports: 1,
                    ping_pong_depth: 1,
                    producers: vec![],
                    consumers: vec!["enc.0".into()],
                },
                BufferBinding {
                    name: "enc0_out".into(),
                    placement: MemoryPlacement::Ddr,
                    region: BufferRegion::FrameBuffer,
                    base_offset_bytes: 921_600,
                    tensor: None,
                    width_bits: 8,
                    depth: 921_600,
                    ram10_blocks: 0,
                    bytes: 921_600,
                    read_ports: 1,
                    write_ports: 1,
                    ping_pong_depth: 1,
                    producers: vec!["enc.0".into()],
                    consumers: vec![],
                },
            ],
            fusions: vec![],
            rejections: RejectionSummary {
                rejected_candidates: 12,
                counts: BTreeMap::from([("frame_cycles".to_string(), 12)]),
                samples: vec![],
                by_mode: BTreeMap::new(),
            },
            search: SearchStats {
                explored: 13,
                expanded_nodes: 240,
                max_candidates: 100_000,
                per_candidate_budget: 100_000,
                truncated_candidates: 0,
                truncated: false,
            },
        }
    }

    #[test]
    fn artifact_round_trips_through_toml() {
        let mut a = feasible_artifact();
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn artifact_round_trips_through_json() {
        let mut a = feasible_artifact();
        a.normalize();
        let text = a.to_json_string().unwrap();
        let b = LayoutArtifact::from_json_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn serialization_is_deterministic_across_input_order() {
        let a = feasible_artifact();
        let mut b = feasible_artifact();
        b.buffers.reverse();
        b.schedule.reverse();
        b.tensors.reverse();
        assert_eq!(
            a.to_toml_string().unwrap(),
            b.to_toml_string().unwrap(),
            "normalization must remove input ordering differences"
        );
    }

    #[test]
    fn over_budget_estimate_is_rejected() {
        let mut a = feasible_artifact();
        let resources = a.resources.as_mut().unwrap();
        resources.dsp48 = 999;
        let err = a.validate().unwrap_err();
        assert!(matches!(
            err,
            LayoutError::OverBudget {
                resource: "dsp48",
                ..
            }
        ));

        // Slack must also match the (fixed) resources.
        let mut b = feasible_artifact();
        b.resources.as_mut().unwrap().frame_cycles = 11_000_000;
        assert!(b.validate().is_err());
    }

    #[test]
    fn infeasible_artifact_must_explain_itself() {
        let mut a = feasible_artifact();
        a.status = PlanStatus::Infeasible;
        a.certainty = PlanCertainty::ProvenInfeasible;
        a.selection = None;
        a.resources = None;
        a.cycles = None;
        a.slack = None;
        a.schedule.clear();
        a.rejections.rejected_candidates = 0;
        assert!(a.validate().is_err());

        a.rejections.rejected_candidates = 4;
        a.rejections.counts.insert("dsp_budget".to_string(), 4);
        a.rejections.samples.push(RejectedCandidate {
            execution_mode: ExecutionMode::Streaming,
            granularity: Granularity::Row,
            tile: None,
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Fifo,
            fusion_enabled: false,
            reason: RejectReason::DspBudget {
                used: 304,
                budget: 152,
            },
        });
        a.validate().unwrap();
        assert!(matches!(
            a.execution_plan(),
            Err(LayoutError::Infeasible { .. })
        ));
    }

    #[test]
    fn per_channel_multipliers_round_trip_and_validate() {
        let mut a = feasible_artifact();
        a.schedule[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = (0..32).map(|i| 65_536 + i).collect();
        a.schedule[0].requant.as_mut().unwrap().multiplier_error_ppm = Some(37);
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            b.schedule[0]
                .requant
                .as_ref()
                .unwrap()
                .per_channel_multipliers
                .len(),
            32
        );

        // Wrong length is rejected at the artifact boundary.
        let mut c = feasible_artifact();
        c.schedule[0]
            .requant
            .as_mut()
            .unwrap()
            .per_channel_multipliers = vec![1, 2, 3];
        assert!(c.validate().is_err());
    }

    #[test]
    fn rejection_reason_samples_round_trip_through_toml() {
        let mut a = feasible_artifact();
        a.status = PlanStatus::Infeasible;
        a.certainty = PlanCertainty::ProvenInfeasible;
        a.selection = None;
        a.resources = None;
        a.cycles = None;
        a.slack = None;
        a.schedule.clear();
        a.rejections.rejected_candidates = 1;
        a.rejections.counts.insert("ram_budget".into(), 1);
        a.rejections.samples.push(RejectedCandidate {
            execution_mode: ExecutionMode::Hybrid,
            granularity: Granularity::Tile,
            tile: Some(TileGeometry::new(80, 60)),
            weight_placement: MemoryPlacement::Bram,
            intermediate_placement: MemoryPlacement::Bram,
            fusion_enabled: true,
            reason: RejectReason::RamBudget {
                used: 120,
                budget: 88,
            },
        });
        a.normalize();
        a.validate().unwrap();
        let text = a.to_toml_string().unwrap();
        let b = LayoutArtifact::from_toml_str(&text).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn execution_plan_projects_weights_and_steps() {
        let a = feasible_artifact();
        let plan = a.execution_plan().unwrap();
        assert_eq!(plan.format, PLAN_FORMAT);
        assert_eq!(plan.execution_mode, ExecutionMode::SharedTile);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.weights.len(), 2);
        assert_eq!(plan.weights[0].id, TensorId(1));
        assert_eq!(plan.weights[1].id, TensorId(2));

        let text = plan.to_toml_string().unwrap();
        let reparsed = ExecutionPlan::from_toml_str(&text).unwrap();
        assert_eq!(plan, reparsed);
    }

    #[test]
    fn wrong_format_and_schema_are_rejected_on_load() {
        let a = feasible_artifact();
        let text = a
            .to_toml_string()
            .unwrap()
            .replace("format = \"tessera.layout\"", "format = \"tessera.nope\"");
        assert!(matches!(
            LayoutArtifact::from_toml_str(&text),
            Err(LayoutError::WrongFormat { .. })
        ));

        let mut b = feasible_artifact();
        b.schema_version = 9;
        assert!(matches!(
            b.validate(),
            Err(LayoutError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn schedule_referencing_unknown_buffer_is_rejected() {
        let mut a = feasible_artifact();
        a.schedule[0].output_buffer = "missing".into();
        assert!(matches!(a.validate(), Err(LayoutError::Invalid { .. })));
    }

    #[test]
    fn slack_arithmetic_handles_deficit() {
        let used = ResourceEstimate {
            dsp48: 160,
            ..ResourceEstimate::ZERO
        };
        let slack = ResourceSlack::between(&used, &limits());
        assert_eq!(slack.dsp48, -8);
        assert!(!slack.all_non_negative());
    }

    #[test]
    fn objective_ordering_is_lexicographic() {
        let objective = ObjectiveSpec {
            primary: ObjectiveTerm::MinimizeCycles,
            secondary: vec![ObjectiveTerm::MinimizeDdrBytes],
        };
        let a = ResourceEstimate {
            frame_cycles: 100,
            ddr_read_bytes_per_frame: 50,
            ddr_write_bytes_per_frame: 0,
            ..ResourceEstimate::ZERO
        };
        let b = ResourceEstimate {
            frame_cycles: 100,
            ddr_read_bytes_per_frame: 10,
            ddr_write_bytes_per_frame: 0,
            ..ResourceEstimate::ZERO
        };
        assert!(objective.is_better(&b, &a));
        assert!(!objective.is_better(&a, &b));
    }

    // -----------------------------------------------------------------------
    // Regression tests for the v1 plan/backend contract
    // -----------------------------------------------------------------------

    /// A truncated search must never be labeled optimal, and the label must
    /// follow from the recorded outcome rather than from the author's hope.
    #[test]
    fn truncated_search_cannot_claim_optimality() {
        let mut a = feasible_artifact();
        a.search.truncated = true;
        // Still claiming the optimum: rejected.
        assert!(matches!(a.validate(), Err(LayoutError::Invalid { .. })));
        a.certainty = PlanCertainty::BestEffort;
        a.validate().unwrap();
        assert!(!a.is_proven_optimal());
        assert!(!a.is_conclusive());

        // An interrupted infeasible search is inconclusive, not a proof.
        let mut b = feasible_artifact();
        b.status = PlanStatus::Infeasible;
        b.certainty = PlanCertainty::Inconclusive;
        b.selection = None;
        b.resources = None;
        b.cycles = None;
        b.slack = None;
        b.schedule.clear();
        b.rejections.rejected_candidates = 3;
        b.rejections.counts.insert("ram_budget".into(), 3);
        b.search.truncated = true;
        b.validate().unwrap();
        assert!(!b.is_conclusive());

        // The same artifact labeled as a proof is rejected.
        b.certainty = PlanCertainty::ProvenInfeasible;
        assert!(b.validate().is_err());
    }

    /// The bug this guards: a plan that claims fewer lane iterations than its
    /// MAC count requires is claiming parallelism the emitter does not build.
    #[test]
    fn a_step_cannot_claim_more_parallelism_than_its_macs_allow() {
        let mut a = feasible_artifact();
        a.schedule[0].simd = 1;
        a.schedule[0].pe = 1;
        a.schedule[0].tap_parallel = 1;
        a.schedule[0].cost.lane_iterations = 1;
        a.schedule[0].cost.macs = 64;
        a.schedule[0].cost.lanes = 1;
        a.schedule[0].estimated_cycles = 1;
        a.cycles = Some(CycleBreakdown::new(100, 1, 0, 0));
        a.resources.as_mut().unwrap().frame_cycles = 1;
        let err = a.validate().unwrap_err();
        assert!(
            err.to_string().contains("lane iterations"),
            "unexpected error: {err}"
        );

        // One lane doing 64 MACs needs 64 iterations, and that is accepted. The
        // recorded fold is one lane — `cost.lanes` is the physical lane count,
        // so it can only be one when `simd × pe × tap` is one.
        let mut b = feasible_artifact();
        b.schedule[0].simd = 1;
        b.schedule[0].pe = 1;
        b.schedule[0].tap_parallel = 1;
        b.schedule[0].cost.lane_iterations = 64;
        b.schedule[0].cost.macs = 64;
        b.schedule[0].cost.lanes = 1;
        b.schedule[0].estimated_cycles = 64;
        b.cycles = Some(CycleBreakdown::new(100, 64, 0, 0));
        b.resources.as_mut().unwrap().frame_cycles = 64;
        let resources = *b.resources.as_ref().unwrap();
        b.slack = Some(ResourceSlack::between(&resources, &b.limits));
        b.validate().unwrap();
    }

    /// A step whose recorded lane count disagrees with its recorded fold is
    /// rejected: the two describe the same hardware and cannot drift.
    #[test]
    fn a_step_may_not_misreport_its_own_fold() {
        let mut a = feasible_artifact();
        a.schedule[0].cost.lanes = 1;
        a.schedule[0].cost.lane_iterations = a.schedule[0].cost.macs;
        let err = a.validate().unwrap_err();
        assert!(
            err.to_string().contains("does not match simd*pe*tap"),
            "unexpected error: {err}"
        );
    }

    /// A step may not use more lanes than the engine instantiates.
    #[test]
    fn a_step_cannot_use_lanes_the_engine_does_not_instantiate() {
        let mut a = feasible_artifact();
        a.selection.as_mut().unwrap().engine.simd = 1;
        a.selection.as_mut().unwrap().engine.pe = 1;
        let err = a.validate().unwrap_err();
        assert!(
            err.to_string().contains("the engine only instantiates"),
            "unexpected error: {err}"
        );
    }

    /// Cycle accounting must be the sum of its parts and match the schedule.
    #[test]
    fn cycle_breakdown_must_match_the_schedule() {
        let mut a = feasible_artifact();
        a.cycles = Some(CycleBreakdown::new(100, 9_866_272, 0, 7));
        assert!(
            a.validate().is_err(),
            "an unexplained dma charge must be rejected"
        );

        let mut b = feasible_artifact();
        b.cycles = Some(CycleBreakdown::new(100, 1, 0, 0));
        assert!(
            b.validate().is_err(),
            "a breakdown that contradicts the steps must be rejected"
        );
    }

    /// The bug this guards: summing the sizes of the ranges still live when a
    /// range starts ignores where those ranges actually sit, so it can place a
    /// range straight through a surviving allocation. Concrete ranges
    /// `(0,0,10)`, `(0,3,5)`, `(1,2,10)`: the naive sum puts the third at `5`,
    /// inside the second at `[10,15)`.
    #[test]
    fn arena_allocation_never_lets_two_live_ranges_share_a_word() {
        let ranges = [
            ArenaRange {
                start: 0,
                end: 0,
                words: 10,
            },
            ArenaRange {
                start: 0,
                end: 3,
                words: 5,
            },
            ArenaRange {
                start: 1,
                end: 2,
                words: 10,
            },
        ];
        let (offsets, high) = allocate_arena(&ranges);
        // The first is dead by step 1, so the third reuses its words: only the
        // second and third are live together, and they are disjoint.
        assert_eq!(offsets, vec![0, 10, 0]);
        assert_eq!(high, 15);

        // The property, checked exhaustively over every pair: two ranges that
        // are live at the same time never share a word.
        for a in 0..ranges.len() {
            for b in (a + 1)..ranges.len() {
                let time_overlap =
                    ranges[a].start <= ranges[b].end && ranges[b].start <= ranges[a].end;
                if !time_overlap {
                    continue;
                }
                let (lo_a, hi_a) = (offsets[a], offsets[a] + ranges[a].words);
                let (lo_b, hi_b) = (offsets[b], offsets[b] + ranges[b].words);
                assert!(
                    hi_a <= lo_b || hi_b <= lo_a,
                    "ranges {a} [{lo_a}..{hi_a}) and {b} [{lo_b}..{hi_b}) are live at the same \
                     time but share words"
                );
            }
        }
        for (index, range) in ranges.iter().enumerate() {
            assert!(offsets[index] + range.words <= high);
        }

        // The extent is the placement's own high water mark and may exceed the
        // peak simultaneous word count; it must never be smaller.
        let peak = (0..=3)
            .map(|step| {
                ranges
                    .iter()
                    .filter(|r| r.start <= step && step <= r.end)
                    .map(|r| r.words)
                    .sum::<u64>()
            })
            .max()
            .unwrap();
        assert!(high >= peak, "an allocation is never smaller than the peak");

        // Placement is deterministic and independent of the input order.
        let reversed: Vec<ArenaRange> = ranges.iter().rev().copied().collect();
        let (reversed_offsets, reversed_high) = allocate_arena(&reversed);
        assert_eq!(reversed_high, high);
        for (index, range) in ranges.iter().enumerate() {
            let mirror = ranges.len() - 1 - index;
            assert_eq!(offsets[index], reversed_offsets[mirror]);
            assert!(range.words > 0);
        }
    }

    /// A set of ranges that fits in one word each and never overlap in time
    /// still has to be laid out without aliasing anything live.
    #[test]
    fn arena_allocation_shares_words_only_across_disjoint_lifetimes() {
        let ranges = [
            ArenaRange {
                start: 0,
                end: 1,
                words: 4,
            },
            ArenaRange {
                start: 2,
                end: 3,
                words: 4,
            },
            ArenaRange {
                start: 1,
                end: 2,
                words: 8,
            },
        ];
        let (offsets, high) = allocate_arena(&ranges);
        // Placement order is (0,1), (1,2), (2,3): the third is live with the
        // second only, so it takes the words the first left behind at 0; the
        // second sits above the first at 4, and the extent is the second's top.
        assert_eq!(offsets, vec![0, 0, 4]);
        assert_eq!(high, 12);
        for index in 0..ranges.len() {
            assert!(offsets[index] + ranges[index].words <= high);
        }
    }

    /// Addresses inside one arena must not overlap, and stream buffers must not
    /// pretend to be addressable.
    #[test]
    fn buffer_arenas_do_not_overlap() {
        let mut a = feasible_artifact();
        a.buffers[1].base_offset_bytes = 0;
        let err = a.validate().unwrap_err();
        assert!(
            err.to_string().contains("overlap"),
            "unexpected error: {err}"
        );

        let mut b = feasible_artifact();
        b.buffers[0].region = BufferRegion::Stream;
        b.buffers[0].base_offset_bytes = 16;
        assert!(
            b.validate().is_err(),
            "a stream buffer must not carry an offset"
        );

        // A real stream buffer without an offset is legal.
        let mut c = feasible_artifact();
        c.buffers[0].region = BufferRegion::Stream;
        c.buffers[0].base_offset_bytes = 0;
        c.buffers[0].placement = MemoryPlacement::Fifo;
        c.validate().unwrap();
    }

    /// The activation arena is allocated by tensor lifetime, so shared words are
    /// legal only between activations whose live ranges are disjoint. A mutated
    /// plan that aliases two *simultaneously live* activations is rejected: the
    /// validator derives the lifetimes from the schedule instead of trusting the
    /// planner's offsets.
    #[test]
    fn arena_aliasing_is_only_allowed_across_disjoint_lifetimes() {
        // One step: tensor 1 is produced and read at step 0, tensor 2 is read
        // by a second step at 1 and produced there, so 1 and 2 never overlap.
        let two_steps = |artifact: &mut LayoutArtifact| {
            let mut second = artifact.schedule[0].clone();
            second.step = 1;
            second.inputs = vec![TensorId(2)];
            second.outputs = vec![];
            // The second step carries no cost, so the recorded cycle breakdown
            // still matches the schedule it was derived from.
            second.cost.lane_iterations = 0;
            second.cost.macs = 0;
            second.cost.setup_cycles = 0;
            second.estimated_cycles = 0;
            artifact.schedule[0].inputs = vec![TensorId(0), TensorId(1)];
            artifact.schedule[0].outputs = vec![TensorId(1)];
            artifact.schedule.push(second);
        };

        // Live at the same time (tensor 2 is produced at step 0 and read at 1,
        // tensor 1 is produced at 0 too) and sharing words: rejected.
        let mut overlapping = feasible_artifact();
        two_steps(&mut overlapping);
        overlapping.schedule[0].inputs = vec![TensorId(0)];
        overlapping.schedule[0].outputs = vec![TensorId(1), TensorId(2)];
        overlapping.buffers[0].region = BufferRegion::ActivationRam;
        overlapping.buffers[0].tensor = Some(TensorId(2));
        overlapping.buffers[0].base_offset_bytes = 0;
        overlapping.buffers[1].region = BufferRegion::ActivationRam;
        overlapping.buffers[1].tensor = Some(TensorId(1));
        overlapping.buffers[1].base_offset_bytes = 0;
        let err = overlapping.validate().unwrap_err();
        assert!(
            err.to_string().contains("live at the same time"),
            "unexpected error: {err}"
        );

        // Disjoint lifetimes sharing the same words: allowed.
        let mut shared = feasible_artifact();
        two_steps(&mut shared);
        // Tensor 1 dies at step 0 (step 0 reads it back), tensor 2 is produced
        // at step 1: their live ranges are disjoint, so they may share words.
        shared.schedule[1].inputs = vec![];
        shared.schedule[1].outputs = vec![TensorId(2)];
        shared.buffers[0].region = BufferRegion::ActivationRam;
        shared.buffers[0].tensor = Some(TensorId(1));
        shared.buffers[0].base_offset_bytes = 0;
        shared.buffers[1].region = BufferRegion::ActivationRam;
        shared.buffers[1].tensor = Some(TensorId(2));
        shared.buffers[1].base_offset_bytes = 0;
        shared.validate().unwrap();
    }

    /// The plan projection must carry the cycle breakdown and certainty, and
    /// re-validate on load.
    #[test]
    fn execution_plan_carries_certainty_and_cycles() {
        let a = feasible_artifact();
        let plan = a.execution_plan().unwrap();
        assert_eq!(plan.certainty, PlanCertainty::Optimal);
        assert_eq!(plan.cycles.total_cycles, plan.resources.frame_cycles);
        let text = plan.to_toml_string().unwrap();
        let reparsed = ExecutionPlan::from_toml_str(&text).unwrap();
        assert_eq!(plan, reparsed);

        let mut broken = plan.clone();
        broken.cycles.dma_cycles = 5;
        assert!(
            ExecutionPlan::from_toml_str(&broken.to_toml_string().unwrap()).is_err(),
            "a plan whose cycles do not add up must not load"
        );
    }
}
