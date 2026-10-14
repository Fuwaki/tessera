//! Ti60 target model, resource mapping rules, and constraint-file parsing.
//!
//! This module is the single owner of:
//!
//! * strongly typed hardware units and budgets ([`DspBlocks`], [`Ram10Blocks`],
//!   [`XlrUnits`], [`FrameCycles`], [`Hertz`], [`ByteCount`], [`Bits`], …);
//! * the [`Ti60Target`] device description and its physical mapping rules
//!   (`map_mac`, `map_buffer`, requant cost, DDR/cycle budgets);
//! * a small, dependency-free TOML-subset parser ([`toml_lite`]) used to read
//!   target and policy constraint files.
//!
//! The Ti60 model is grounded on the numbers measured for the current
//! fpga_src build (`docs/st_accel_architecture.md`, `st_stream/plan.py`):
//! 160 DSP48, 256 RAM10 blocks, 60800 XLR, a 100 MHz core clock, a 2 int8 MAC
//! per DSP48 dual mapping (`11x10 + 8x8`), and cadenced DDR3 bandwidth.
//!
//! The target file is *data*, never code: unknown fields, unknown enum values
//! and infeasible reservations are hard errors, never silently ignored.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use self::toml_lite::TomlError;

// ---------------------------------------------------------------------------
// Typed units
// ---------------------------------------------------------------------------

macro_rules! unit_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl $name {
            /// Multiplicative/additive identity.
            pub const ZERO: Self = Self(0);

            /// Construct a value.
            pub const fn new(v: u64) -> Self {
                Self(v)
            }

            /// Raw underlying count.
            pub const fn get(self) -> u64 {
                self.0
            }

            /// Saturating addition.
            pub const fn saturating_add(self, other: Self) -> Self {
                Self(self.0.saturating_add(other.0))
            }

            /// Saturating subtraction (never wraps).
            pub const fn saturating_sub(self, other: Self) -> Self {
                Self(self.0.saturating_sub(other.0))
            }

            /// Checked subtraction, `None` when `other > self`.
            pub const fn checked_sub(self, other: Self) -> Option<Self> {
                match self.0.checked_sub(other.0) {
                    Some(v) => Some(Self(v)),
                    None => None,
                }
            }

            /// Checked addition.
            pub const fn checked_add(self, other: Self) -> Option<Self> {
                match self.0.checked_add(other.0) {
                    Some(v) => Some(Self(v)),
                    None => None,
                }
            }
        }

        impl std::ops::Add for $name {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl std::ops::AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0;
            }
        }

        impl std::iter::Sum for $name {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                Self(iter.map(|v| v.0).sum())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

unit_newtype!(
    /// DSP48 block count.
    DspBlocks
);
unit_newtype!(
    /// Physical RAM10 (10 Kbit) block count.
    Ram10Blocks
);
unit_newtype!(
    /// Titanium eXtra Logic Resource units.
    XlrUnits
);
unit_newtype!(
    /// Clock or frame cycle count.
    FrameCycles
);
unit_newtype!(
    /// Byte count.
    ByteCount
);
unit_newtype!(
    /// Bit width or bit capacity.
    Bits
);
unit_newtype!(
    /// Arithmetic MAC lane count.
    MacLanes
);
unit_newtype!(
    /// Memory port count.
    Ports
);

/// Core clock frequency in hertz.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Hertz(u64);

impl Hertz {
    /// Construct a frequency.
    pub const fn new(hz: u64) -> Self {
        Self(hz)
    }

    /// Raw hertz value.
    pub const fn hz(self) -> u64 {
        self.0
    }

    /// Cycles available in one frame at this clock for the given frame rate.
    pub fn cycles_for_frame(self, frame_hz: Hertz) -> Option<FrameCycles> {
        if frame_hz.0 == 0 || self.0 == 0 {
            return None;
        }
        Some(FrameCycles(self.0 / frame_hz.0))
    }
}

impl fmt::Display for Hertz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Sustained byte throughput in bytes per second.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct BytesPerSecond(u64);

impl BytesPerSecond {
    /// Construct a throughput.
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    /// Raw bytes per second.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Bytes that may be moved in `cycles` at `clock`.
    ///
    /// Uses `u128` internally so wide products cannot silently wrap.
    pub fn bytes_for(self, clock: Hertz, cycles: FrameCycles) -> Option<ByteCount> {
        if clock.0 == 0 {
            return None;
        }
        let v = (self.0 as u128) * (cycles.0 as u128) / (clock.0 as u128);
        u64::try_from(v).ok().map(ByteCount)
    }
}

impl fmt::Display for BytesPerSecond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced while parsing or validating a target file and while mapping
/// logical buffers/MAC lanes onto Ti60 physical resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetError {
    /// The file carries a different `format` tag than expected.
    WrongFormat { expected: String, found: String },
    /// The file carries an unsupported `schema_version`.
    UnsupportedSchema { found: u32, supported: u32 },
    /// A required field is missing or empty.
    MissingField { field: String },
    /// A value is outside the legal range.
    InvalidValue { field: String, detail: String },
    /// `reserved` exceeds `budget`, leaving no usable resource.
    ReservationOverBudget {
        resource: String,
        budget: u64,
        reserved: u64,
    },
    /// Two primitives share a name.
    DuplicatePrimitive { name: String },
    /// No RAM primitive is declared (or the requested one is absent).
    UnknownRamPrimitive { name: String },
    /// Operand widths are not mappable on the declared stream primitives.
    UnsupportedOperandWidth {
        activation_bits: u32,
        weight_bits: u32,
    },
    /// A MAC mode is not advertised by any stream primitive.
    UnsupportedMacMode { mode: String },
    /// A buffer geometry cannot be built from the declared RAM primitive.
    UnsupportedBufferGeometry { primitive: String, detail: String },
    /// A buffer requests more ports than the primitive (and its replicas) offer.
    UnsupportedPorts {
        primitive: String,
        requested_reads: u64,
        requested_writes: u64,
        primitive_reads: u64,
        primitive_writes: u64,
    },
    /// TOML syntax or schema binding failure.
    Config(TomlError),
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetError::WrongFormat { expected, found } => {
                write!(
                    f,
                    "wrong format tag: expected {expected:?}, found {found:?}"
                )
            }
            TargetError::UnsupportedSchema { found, supported } => write!(
                f,
                "unsupported schema_version {found} (this build supports {supported})"
            ),
            TargetError::MissingField { field } => write!(f, "missing required field {field:?}"),
            TargetError::InvalidValue { field, detail } => {
                write!(f, "invalid value for {field:?}: {detail}")
            }
            TargetError::ReservationOverBudget {
                resource,
                budget,
                reserved,
            } => write!(
                f,
                "reserved {resource}={reserved} exceeds budget {resource}={budget}"
            ),
            TargetError::DuplicatePrimitive { name } => {
                write!(f, "duplicate primitive {name:?}")
            }
            TargetError::UnknownRamPrimitive { name } => {
                write!(f, "unknown RAM primitive {name:?}")
            }
            TargetError::UnsupportedOperandWidth {
                activation_bits,
                weight_bits,
            } => write!(
                f,
                "unsupported operand widths: activation {activation_bits}-bit, weight {weight_bits}-bit"
            ),
            TargetError::UnsupportedMacMode { mode } => {
                write!(
                    f,
                    "MAC mode {mode:?} is not advertised by any stream primitive"
                )
            }
            TargetError::UnsupportedBufferGeometry { primitive, detail } => write!(
                f,
                "buffer cannot be mapped onto RAM primitive {primitive:?}: {detail}"
            ),
            TargetError::UnsupportedPorts {
                primitive,
                requested_reads,
                requested_writes,
                primitive_reads,
                primitive_writes,
            } => write!(
                f,
                "ports exceed primitive {primitive:?}: requested {requested_reads}R/{requested_writes}W, primitive {primitive_reads}R/{primitive_writes}W"
            ),
            TargetError::Config(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TargetError {}

impl From<TomlError> for TargetError {
    fn from(e: TomlError) -> Self {
        TargetError::Config(e)
    }
}

// ---------------------------------------------------------------------------
// Target description (serde schema = the TOML file schema)
// ---------------------------------------------------------------------------

/// Target file format tag.
pub const TARGET_FORMAT: &str = "tessera.target";
/// Target file schema version implemented by this build.
pub const TARGET_SCHEMA_VERSION: u32 = 1;

/// Physical MAC packing of one stream primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MacMode {
    /// Two int8 products share one DSP48 slice.
    Int8MacDual,
    /// One 9-bit signed product per DSP48 slice.
    Int9MacSingle,
}

impl MacMode {
    /// Stable string name used in diagnostics and reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            MacMode::Int8MacDual => "int8_mac_dual",
            MacMode::Int9MacSingle => "int9_mac_single",
        }
    }
}

/// `[clock]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClockSpec {
    /// Core clock in hertz.
    pub core_hz: u64,
}

/// `[budget]` section: the **total** device capacity made available to this
/// compiler, before any non-accelerator consumption.
///
/// Semantics are unambiguous and load-bearing:
///
/// ```text
/// usable(resource) = budget(resource) − reserved(resource)
/// ```
///
/// `budget` is the whole device (e.g. 160 DSP48 for the Ti60F225), not the
/// free remainder; `[reserved]` ([`ReservedSpec`]) is what the rest of the
/// design consumes outside the ML accelerator. The DDR fields are a sustained
/// *rate* (bytes per second), not a per-frame quantity: they are converted to
/// a per-frame byte budget against a frame-cycle count by
/// [`Ti60Target::ddr_budget`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    /// Total available DSP48 blocks.
    pub dsp48: u64,
    /// Total available RAM10 blocks.
    pub ram10: u64,
    /// Total available XLR units.
    pub xlr: u64,
    /// Sustained DDR read throughput in bytes per second.
    pub ddr_read_bytes_per_s: u64,
    /// Sustained DDR write throughput in bytes per second.
    pub ddr_write_bytes_per_s: u64,
}

/// `[reserved]` section: resources consumed outside the ML accelerator, on
/// the same device as [`BudgetSpec`].
///
/// These are *not* a second budget: the planner's usable budget is
/// `budget − reserved` for each resource (see [`Ti60Target::usable_budget`]).
/// A reservation larger than the corresponding budget is rejected by
/// [`Ti60Target::validate`], never silently clamped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReservedSpec {
    /// DSP48 blocks already used.
    #[serde(default)]
    pub dsp48: u64,
    /// RAM10 blocks already used.
    #[serde(default)]
    pub ram10: u64,
    /// XLR units already used.
    #[serde(default)]
    pub xlr: u64,
}

impl Default for ReservedSpec {
    fn default() -> Self {
        Self {
            dsp48: 0,
            ram10: 0,
            xlr: 0,
        }
    }
}

/// `[mac]` section: logical-to-physical MAC packing rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MacSpec {
    /// int8xint8 MAC lanes packed into one DSP48.
    pub int8xint8_per_dsp48: u64,
    /// int9xint8 MAC lanes packed into one DSP48.
    pub int9xint8_per_dsp48: u64,
    /// DSP48 blocks charged per requant multiplier *operand chunk*.
    pub requant_lane_dsp48: u64,
    /// Width of one signed multiply operand the DSP48's multiplier takes.
    ///
    /// The Efinix `EFX_DSP48` multiplier is 18×18; a wider product is built by
    /// decomposing the wider operand into 18-bit chunks and cascading the
    /// partial products (`IS_DECOMP_PP_MULT`, `CASCIN_ASR18`). The measured
    /// requant unit maps a 32-bit accumulator against an 18-bit `M0` to
    /// exactly two `EFX_DSP48`
    /// (`fpga_src/engine/out_leaf_rq/requant_unit.map.v`), which is what this
    /// width reproduces: `ceil(accumulator_bits / 18)` blocks per lane.
    #[serde(default = "default_dsp_multiplier_operand_bits")]
    pub dsp_multiplier_operand_bits: u64,
}

/// Default width of one DSP48 multiply operand.
fn default_dsp_multiplier_operand_bits() -> u64 {
    18
}

/// `[logic]` section: coarse XLR (logic) coefficients.
///
/// XLR is only modelled when these are non-zero; a target that declares no
/// logic coefficients reports `xlr = 0` and the XLR budget is vacuous. The
/// default reference value (`xlr_per_engine = 2500`) comes from the measured
/// `st_accel` XLR usage in `fpga_src/docs/st_accel_architecture.md`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogicSpec {
    /// XLR units charged per physical MAC lane.
    #[serde(default)]
    pub xlr_per_mac_lane: u64,
    /// XLR units charged per engine/control instance.
    #[serde(default)]
    pub xlr_per_engine: u64,
    /// XLR units charged per physical RAM10 block.
    #[serde(default)]
    pub xlr_per_ram_block: u64,
}

impl Default for LogicSpec {
    fn default() -> Self {
        Self {
            xlr_per_mac_lane: 0,
            xlr_per_engine: 0,
            xlr_per_ram_block: 0,
        }
    }
}

/// One legal physical organization of a RAM primitive.
///
/// A block RAM is *not* a rectangle of `max_width_bits × max_depth`. The Efinix
/// `EFX_RAM10` is a 10 Kbit simple dual-port block whose read and write ports
/// may only be configured to a fixed set of width/depth pairs. The vendor model
/// (`sim_models/verilog/efx_ram10.v`) enumerates them exactly:
///
/// ```text
/// width → depth
///   1   → 8192     5  → 2048    16 → 512
///   2   → 4096     8  → 1024    20 → 512
///   4   → 2048    10  → 1024
/// ```
///
/// There is no 40-bit organization: the widest `RAM10` port is 20 bits, and
/// `40 × 256` (the shape a naive `max_width_bits × max_depth` rectangle would
/// invent) is not a block at all. Treating the two maxima as independent would
/// let a buffer claim a geometry no block can realize, so the target declares
/// the organizations it physically supports and buffer mapping picks the one
/// that needs the fewest blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RamMode {
    /// Data width of one physical block in this organization.
    pub width_bits: u64,
    /// Addressable depth of one physical block in this organization.
    pub depth: u64,
}

impl RamMode {
    /// Storage cells addressable in this organization (`width × depth`).
    pub const fn cells(self) -> u64 {
        self.width_bits * self.depth
    }
}

/// One `[[ram.primitive]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RamPrimitive {
    /// Primitive name (e.g. `ram10`).
    pub name: String,
    /// Total capacity in bits.
    pub capacity_bits: u64,
    /// Maximum data width in bits of one physical block.
    pub max_width_bits: u64,
    /// Maximum depth of one physical block.
    pub max_depth: u64,
    /// Independent read ports per physical block.
    pub read_ports: u64,
    /// Independent write ports per physical block.
    pub write_ports: u64,
    /// Legal physical organizations (`[[ram.primitive.mode]]`). Empty means
    /// "undeclared": mapping then falls back to a single conservative
    /// rectangle derived from the capacity, never to independent maxima.
    #[serde(default)]
    pub mode: Vec<RamMode>,
}

/// `[ram]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RamSection {
    /// Declared RAM primitives.
    #[serde(default)]
    pub primitive: Vec<RamPrimitive>,
}

impl RamPrimitive {
    /// The organizations buffer mapping may choose from.
    ///
    /// Declared modes are used verbatim. A primitive that declares none falls
    /// back to one conservative rectangle that *does* fit the declared bit
    /// budget: `max_width_bits × min(max_depth, floor(capacity_bits /
    /// max_width_bits))`. For the 10 Kbit/20-bit `EFX_RAM10` that is `20 × 512`,
    /// which is a real organization; the fallback never invents a
    /// `20 × 1024` block that would be twice the physical capacity.
    pub fn effective_modes(&self) -> Vec<RamMode> {
        if !self.mode.is_empty() {
            return self.mode.clone();
        }
        let width = self.max_width_bits.max(1);
        let depth = (self.capacity_bits / width)
            .max(1)
            .min(self.max_depth.max(1));
        vec![RamMode {
            width_bits: width,
            depth,
        }]
    }
}

/// One `[[stream.primitive]]` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamPrimitive {
    /// Primitive name (e.g. `efx_dsp48`).
    pub name: String,
    /// MAC modes the primitive can realise.
    pub valid_modes: Vec<MacMode>,
}

/// `[stream]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSection {
    /// Declared stream primitives.
    #[serde(default)]
    pub primitive: Vec<StreamPrimitive>,
}

/// A declarative Ti60-family target description.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ti60Target {
    /// Must equal [`TARGET_FORMAT`].
    pub format: String,
    /// Must equal [`TARGET_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Board/target name.
    pub name: String,
    /// Target revision string, part of the provenance key.
    pub revision: String,
    /// Clock.
    pub clock: ClockSpec,
    /// Hard resource budgets.
    pub budget: BudgetSpec,
    /// Non-ML resource reservation.
    #[serde(default)]
    pub reserved: ReservedSpec,
    /// MAC packing rules.
    pub mac: MacSpec,
    /// Coarse logic/XLR coefficients (optional).
    #[serde(default)]
    pub logic: LogicSpec,
    /// RAM primitives.
    pub ram: RamSection,
    /// Stream primitives.
    pub stream: StreamSection,
}

/// Planner-facing usable budget after subtracting non-ML reservations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct UsableBudget {
    /// Usable DSP48 blocks.
    pub dsp48: DspBlocks,
    /// Usable RAM10 blocks.
    pub ram10: Ram10Blocks,
    /// Usable XLR units.
    pub xlr: XlrUnits,
}

/// Per-frame DDR byte budget derived from the frame cycle budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DdrBudget {
    /// Bytes that may be read per frame.
    pub read_bytes: ByteCount,
    /// Bytes that may be written per frame.
    pub write_bytes: ByteCount,
}

/// Result of mapping MAC lanes onto DSP48 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MacMapping {
    /// Physical DSP48 blocks consumed.
    pub dsp48: DspBlocks,
    /// Logical MAC lanes requested.
    pub lanes: MacLanes,
    /// MAC lanes packed into each DSP48.
    pub lanes_per_dsp48: u64,
    /// Nine-bit multipliers each lane needs for its activation width.
    pub multipliers_per_lane: u64,
    /// Selected physical packing mode.
    pub mode: MacMode,
}

/// A logical buffer to be mapped onto physical RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferRequest {
    /// Element data width in bits.
    pub width_bits: u64,
    /// Number of words.
    pub depth: u64,
    /// Simultaneous read ports required.
    pub read_ports: u64,
    /// Simultaneous write ports required.
    pub write_ports: u64,
}

/// Result of mapping a logical buffer onto physical RAM10 blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamMapping {
    /// Primitive used.
    pub primitive: String,
    /// Physical organization selected for one block.
    pub mode: RamMode,
    /// Physical blocks consumed by one replica.
    pub blocks_per_replica: Ram10Blocks,
    /// Physical blocks consumed (already includes replication).
    pub blocks: Ram10Blocks,
    /// Duplication factor driven by port requirements.
    pub replication: u64,
    /// Data width after per-block width rounding, in the selected organization.
    pub width_padded_bits: u64,
    /// Depth after per-block depth rounding, in the selected organization.
    pub depth_padded: u64,
    /// Capacity actually occupied by the requested payload (bits).
    pub capacity_bits_used: u64,
}

impl Ti60Target {
    /// Parse a target file from TOML text and validate it.
    pub fn from_toml_str(src: &str) -> Result<Self, TargetError> {
        let target: Ti60Target = toml_lite::from_toml_str(src)?;
        target.validate()?;
        Ok(target)
    }

    /// Read and parse a target file from disk.
    pub fn from_toml_file(path: impl AsRef<std::path::Path>) -> Result<Self, TargetError> {
        let text =
            std::fs::read_to_string(path.as_ref()).map_err(|e| TargetError::InvalidValue {
                field: "path".into(),
                detail: format!("cannot read {}: {e}", path.as_ref().display()),
            })?;
        Self::from_toml_str(&text)
    }

    /// Deterministic TOML rendering of this target.
    pub fn to_toml_string(&self) -> Result<String, TargetError> {
        Ok(toml_lite::to_toml_string_of(self)?)
    }

    /// Core clock as a typed frequency.
    pub const fn clock(&self) -> Hertz {
        Hertz(self.clock.core_hz)
    }

    /// Validate the declared target for internal consistency.
    pub fn validate(&self) -> Result<(), TargetError> {
        if self.format != TARGET_FORMAT {
            return Err(TargetError::WrongFormat {
                expected: TARGET_FORMAT.into(),
                found: self.format.clone(),
            });
        }
        if self.schema_version != TARGET_SCHEMA_VERSION {
            return Err(TargetError::UnsupportedSchema {
                found: self.schema_version,
                supported: TARGET_SCHEMA_VERSION,
            });
        }
        if self.name.trim().is_empty() {
            return Err(TargetError::MissingField {
                field: "name".into(),
            });
        }
        if self.clock.core_hz == 0 {
            return Err(TargetError::InvalidValue {
                field: "clock.core_hz".into(),
                detail: "must be non-zero".into(),
            });
        }
        if self.budget.dsp48 == 0 {
            return Err(TargetError::InvalidValue {
                field: "budget.dsp48".into(),
                detail: "must be non-zero".into(),
            });
        }
        if self.budget.ram10 == 0 {
            return Err(TargetError::InvalidValue {
                field: "budget.ram10".into(),
                detail: "must be non-zero".into(),
            });
        }
        if self.budget.ddr_read_bytes_per_s == 0 || self.budget.ddr_write_bytes_per_s == 0 {
            return Err(TargetError::InvalidValue {
                field: "budget.ddr_*_bytes_per_s".into(),
                detail: "declared sustained bandwidth must be non-zero".into(),
            });
        }
        if self.mac.int8xint8_per_dsp48 == 0
            || self.mac.int9xint8_per_dsp48 == 0
            || self.mac.requant_lane_dsp48 == 0
            || self.mac.dsp_multiplier_operand_bits == 0
        {
            return Err(TargetError::InvalidValue {
                field: "mac.*".into(),
                detail: "MAC packing factors must be non-zero".into(),
            });
        }
        self.usable_budget()?;

        if self.ram.primitive.is_empty() {
            return Err(TargetError::MissingField {
                field: "ram.primitive".into(),
            });
        }
        let mut names = BTreeMap::new();
        for p in &self.ram.primitive {
            if p.name.trim().is_empty() {
                return Err(TargetError::MissingField {
                    field: "ram.primitive.name".into(),
                });
            }
            if p.capacity_bits == 0 || p.max_width_bits == 0 || p.max_depth == 0 {
                return Err(TargetError::InvalidValue {
                    field: format!("ram.primitive[{}]", p.name),
                    detail: "capacity_bits, max_width_bits and max_depth must be non-zero".into(),
                });
            }
            if p.read_ports == 0 {
                return Err(TargetError::InvalidValue {
                    field: format!("ram.primitive[{}].read_ports", p.name),
                    detail: "must be non-zero".into(),
                });
            }
            let mut seen_modes = BTreeMap::new();
            for mode in &p.mode {
                if mode.width_bits == 0 || mode.depth == 0 {
                    return Err(TargetError::InvalidValue {
                        field: format!("ram.primitive[{}].mode", p.name),
                        detail: "width_bits and depth must be non-zero".into(),
                    });
                }
                if mode.width_bits > p.max_width_bits {
                    return Err(TargetError::InvalidValue {
                        field: format!("ram.primitive[{}].mode", p.name),
                        detail: format!(
                            "organization {}x{} is wider than max_width_bits {}",
                            mode.width_bits, mode.depth, p.max_width_bits
                        ),
                    });
                }
                if mode.depth > p.max_depth {
                    return Err(TargetError::InvalidValue {
                        field: format!("ram.primitive[{}].mode", p.name),
                        detail: format!(
                            "organization {}x{} is deeper than max_depth {}",
                            mode.width_bits, mode.depth, p.max_depth
                        ),
                    });
                }
                if mode.cells() > p.capacity_bits {
                    return Err(TargetError::InvalidValue {
                        field: format!("ram.primitive[{}].mode", p.name),
                        detail: format!(
                            "organization {}x{} holds {} bits, above the declared capacity of {}",
                            mode.width_bits,
                            mode.depth,
                            mode.cells(),
                            p.capacity_bits
                        ),
                    });
                }
                if seen_modes
                    .insert((mode.width_bits, mode.depth), ())
                    .is_some()
                {
                    return Err(TargetError::InvalidValue {
                        field: format!("ram.primitive[{}].mode", p.name),
                        detail: format!(
                            "organization {}x{} is declared twice",
                            mode.width_bits, mode.depth
                        ),
                    });
                }
            }
            if names.insert(p.name.clone(), ()).is_some() {
                return Err(TargetError::DuplicatePrimitive {
                    name: p.name.clone(),
                });
            }
        }
        if self.stream.primitive.is_empty() {
            return Err(TargetError::MissingField {
                field: "stream.primitive".into(),
            });
        }
        for p in &self.stream.primitive {
            if p.name.trim().is_empty() {
                return Err(TargetError::MissingField {
                    field: "stream.primitive.name".into(),
                });
            }
            if p.valid_modes.is_empty() {
                return Err(TargetError::InvalidValue {
                    field: format!("stream.primitive[{}].valid_modes", p.name),
                    detail: "must declare at least one MAC mode".into(),
                });
            }
        }
        Ok(())
    }

    /// Budget remaining for the planner after non-ML reservations.
    pub fn usable_budget(&self) -> Result<UsableBudget, TargetError> {
        let sub = |resource: &str, budget: u64, reserved: u64| -> Result<u64, TargetError> {
            budget
                .checked_sub(reserved)
                .ok_or_else(|| TargetError::ReservationOverBudget {
                    resource: resource.into(),
                    budget,
                    reserved,
                })
        };
        Ok(UsableBudget {
            dsp48: DspBlocks(sub("dsp48", self.budget.dsp48, self.reserved.dsp48)?),
            ram10: Ram10Blocks(sub("ram10", self.budget.ram10, self.reserved.ram10)?),
            xlr: XlrUnits(sub("xlr", self.budget.xlr, self.reserved.xlr)?),
        })
    }

    /// DDR bytes that may be moved in both directions within one frame.
    pub fn ddr_budget(&self, frame_cycles: FrameCycles) -> DdrBudget {
        let clock = self.clock();
        let read = BytesPerSecond(self.budget.ddr_read_bytes_per_s)
            .bytes_for(clock, frame_cycles)
            .unwrap_or(ByteCount(0));
        let write = BytesPerSecond(self.budget.ddr_write_bytes_per_s)
            .bytes_for(clock, frame_cycles)
            .unwrap_or(ByteCount(0));
        DdrBudget {
            read_bytes: read,
            write_bytes: write,
        }
    }

    /// Frame cycle budget for a frame rate at this clock.
    pub fn frame_cycles_for(&self, frame_hz: Hertz) -> Result<FrameCycles, TargetError> {
        self.clock()
            .cycles_for_frame(frame_hz)
            .ok_or_else(|| TargetError::InvalidValue {
                field: "frame_hz".into(),
                detail: "frame rate and clock must be non-zero".into(),
            })
    }

    /// Look up a RAM primitive by name.
    pub fn ram_primitive(&self, name: &str) -> Option<&RamPrimitive> {
        self.ram.primitive.iter().find(|p| p.name == name)
    }

    /// The RAM primitive used when a caller does not name one: `ram10` if
    /// declared, else the first declared primitive.
    pub fn default_ram_primitive(&self) -> Result<&RamPrimitive, TargetError> {
        self.ram_primitive("ram10")
            .or_else(|| self.ram.primitive.first())
            .ok_or(TargetError::MissingField {
                field: "ram.primitive".into(),
            })
    }

    /// Map MAC lanes onto physical DSP48 blocks.
    ///
    /// `activation_bits` is the width of the *codes* the lane multiplies, which
    /// is the exact interval the lowering states, not the container the value
    /// travels in. Operand widths select the physical path:
    ///
    /// * up to 8 bits against int8 weights uses the dual packing (two products
    ///   share one DSP48);
    /// * 9..=18 bits uses the single-product path, and a width beyond nine bits
    ///   needs `ceil(bits / 9)` nine-bit multipliers per lane — the device has
    ///   no wider multiplier, so a residual chain that widens the interval pays
    ///   for partial products instead of being silently truncated;
    /// * wider than 18 bits has no realization and is an error.
    pub fn map_mac(
        &self,
        activation_bits: u32,
        weight_bits: u32,
        lanes: MacLanes,
    ) -> Result<MacMapping, TargetError> {
        let (mode, lanes_per_dsp48, multipliers_per_lane) = match (activation_bits, weight_bits) {
            (a, w) if (1..=8).contains(&a) && w <= 8 => {
                (MacMode::Int8MacDual, self.mac.int8xint8_per_dsp48, 1)
            }
            (a, w) if (9..=18).contains(&a) && w <= 8 => (
                MacMode::Int9MacSingle,
                self.mac.int9xint8_per_dsp48,
                u64::from(a).div_ceil(9),
            ),
            _ => {
                return Err(TargetError::UnsupportedOperandWidth {
                    activation_bits,
                    weight_bits,
                });
            }
        };
        self.require_mac_mode(mode)?;
        let lanes_per_dsp48 = lanes_per_dsp48.max(1);
        let products = lanes.get().saturating_mul(multipliers_per_lane);
        Ok(MacMapping {
            dsp48: DspBlocks(products.div_ceil(lanes_per_dsp48)),
            lanes,
            lanes_per_dsp48,
            multipliers_per_lane,
            mode,
        })
    }

    /// DSP48 blocks charged for `lanes` requant multipliers whose wider operand
    /// is `operand_bits` wide.
    ///
    /// The requant multiplier computes `|acc| × M0`, so the wider operand is
    /// the accumulator. A DSP48 multiplies two 18-bit operands in one block;
    /// anything wider is decomposed into 18-bit chunks whose partial products
    /// cascade *inside* the same column, which costs one block per chunk. The
    /// measured Ti60 requant unit (32-bit accumulator × 18-bit `M0`) maps to
    /// two `EFX_DSP48`, so a plan is never told a 32-bit multiply is one block.
    pub fn map_requant(&self, lanes: MacLanes, operand_bits: u32) -> DspBlocks {
        let chunk = self.mac.dsp_multiplier_operand_bits.max(1);
        let chunks = u64::from(operand_bits).div_ceil(chunk).max(1);
        DspBlocks(lanes.get() * self.mac.requant_lane_dsp48.max(1) * chunks)
    }

    /// Coarse XLR estimate from physical lanes, engine instances and RAM10
    /// blocks. Zero unless the target declares logic coefficients.
    pub fn estimate_xlr(&self, lanes: u64, engines: u64, ram_blocks: u64) -> XlrUnits {
        XlrUnits(
            self.logic
                .xlr_per_engine
                .saturating_mul(engines)
                .saturating_add(self.logic.xlr_per_mac_lane.saturating_mul(lanes))
                .saturating_add(self.logic.xlr_per_ram_block.saturating_mul(ram_blocks)),
        )
    }

    /// Whether a MAC mode is advertised by some declared stream primitive.
    pub fn require_mac_mode(&self, mode: MacMode) -> Result<(), TargetError> {
        if self
            .stream
            .primitive
            .iter()
            .any(|p| p.valid_modes.contains(&mode))
        {
            Ok(())
        } else {
            Err(TargetError::UnsupportedMacMode {
                mode: mode.as_str().into(),
            })
        }
    }

    /// Map a logical buffer onto physical RAM blocks.
    ///
    /// One physical block is not a `max_width × max_depth` rectangle: it is a
    /// fixed bit budget organized in a small set of legal width/depth shapes
    /// ([`RamPrimitive::mode`]). The mapping therefore tiles the requested
    /// `width × depth` with the cheapest legal organization
    /// (`ceil(width/w) × ceil(depth/d)` blocks, minimized over the declared
    /// modes), then replicates the result to satisfy the requested port count.
    /// Bytes alone are never used as the capacity test: a 1-bit-wide,
    /// 20 000-deep buffer needs more blocks than its bit count suggests because
    /// no single block is that deep.
    ///
    /// "Cheapest" is a total order, so the result never depends on declaration
    /// order: fewest blocks first, then the fewest wasted storage cells
    /// (`blocks × cells − payload`), then the narrowest block, then the
    /// shallowest.
    ///
    /// When a primitive declares no modes, the fallback is the single
    /// conservative rectangle `max_width_bits × min(max_depth,
    /// floor(capacity_bits / max_width_bits))` — again a shape that fits one
    /// block's bit budget, never a rectangle of both maxima at once.
    pub fn map_buffer(
        &self,
        primitive: &str,
        req: BufferRequest,
    ) -> Result<RamMapping, TargetError> {
        let prim =
            self.ram_primitive(primitive)
                .ok_or_else(|| TargetError::UnknownRamPrimitive {
                    name: primitive.into(),
                })?;
        if req.width_bits == 0 || req.depth == 0 {
            return Err(TargetError::UnsupportedBufferGeometry {
                primitive: prim.name.clone(),
                detail: "width_bits and depth must be non-zero".into(),
            });
        }
        if req.read_ports == 0 {
            return Err(TargetError::UnsupportedBufferGeometry {
                primitive: prim.name.clone(),
                detail: "at least one read port is required".into(),
            });
        }
        if prim.write_ports == 0 && req.write_ports > 0 {
            return Err(TargetError::UnsupportedPorts {
                primitive: prim.name.clone(),
                requested_reads: req.read_ports,
                requested_writes: req.write_ports,
                primitive_reads: prim.read_ports,
                primitive_writes: prim.write_ports,
            });
        }

        let modes = prim.effective_modes();
        let payload_bits = req.width_bits.saturating_mul(req.depth);
        // Cheapest legal organization, as a total order that never depends on
        // declaration order: fewest blocks, then the fewest wasted storage
        // cells, then a port at least as wide as the requested word (so a word
        // is not needlessly split across blocks), then the narrowest port, then
        // the shallowest.
        let mut best: Option<((u64, u64, u64, u64, u64), RamMode)> = None;
        for mode in modes {
            let width_slices = req.width_bits.div_ceil(mode.width_bits);
            let depth_slices = req.depth.div_ceil(mode.depth);
            let blocks = width_slices.saturating_mul(depth_slices).max(1);
            let wasted = blocks
                .saturating_mul(mode.cells())
                .saturating_sub(payload_bits);
            let splits_words = u64::from(mode.width_bits < req.width_bits);
            let key = (blocks, wasted, splits_words, mode.width_bits, mode.depth);
            if best.map(|(existing, _)| key < existing).unwrap_or(true) {
                best = Some((key, mode));
            }
        }
        let (_, mode) = best.ok_or_else(|| TargetError::UnsupportedBufferGeometry {
            primitive: prim.name.clone(),
            detail: "primitive declares no usable physical organization".into(),
        })?;
        let blocks_per_replica =
            req.width_bits.div_ceil(mode.width_bits) * req.depth.div_ceil(mode.depth);
        let width_padded_bits = req.width_bits.div_ceil(mode.width_bits) * mode.width_bits;
        let depth_padded = req.depth.div_ceil(mode.depth) * mode.depth;

        let read_replicas = req.read_ports.div_ceil(prim.read_ports);
        let write_replicas = if prim.write_ports == 0 {
            1
        } else {
            req.write_ports.div_ceil(prim.write_ports)
        };
        let replication = read_replicas.max(write_replicas).max(1);

        Ok(RamMapping {
            primitive: prim.name.clone(),
            mode,
            blocks_per_replica: Ram10Blocks(blocks_per_replica),
            blocks: Ram10Blocks(blocks_per_replica * replication),
            replication,
            width_padded_bits,
            depth_padded,
            capacity_bits_used: payload_bits,
        })
    }

    /// Reference Ti60F225 description measured from the current fpga_src build.
    ///
    /// Sources: `fpga_src/docs/st_accel_architecture.md` and
    /// `nngen/examples/st_stream/plan.py` (`dsp_total=160, dsp_non_st=8`,
    /// `bram_total=256, bram_non_st=168`, `xlr_total=60800, xlr_non_st=20232`,
    /// 100 MHz core clock, ~1.6 GB/s DDR3).
    pub fn ti60f225_reference() -> Self {
        Self {
            format: TARGET_FORMAT.into(),
            schema_version: TARGET_SCHEMA_VERSION,
            name: "ti60f225".into(),
            revision: "fpga_src-2026-10-10".into(),
            clock: ClockSpec {
                core_hz: 100_000_000,
            },
            budget: BudgetSpec {
                dsp48: 160,
                ram10: 256,
                xlr: 60_800,
                ddr_read_bytes_per_s: 900_000_000,
                ddr_write_bytes_per_s: 700_000_000,
            },
            reserved: ReservedSpec {
                dsp48: 8,
                ram10: 168,
                xlr: 20_232,
            },
            mac: MacSpec {
                int8xint8_per_dsp48: 2,
                int9xint8_per_dsp48: 1,
                requant_lane_dsp48: 1,
                dsp_multiplier_operand_bits: 18,
            },
            logic: LogicSpec {
                xlr_per_mac_lane: 0,
                xlr_per_engine: 2_500,
                xlr_per_ram_block: 0,
            },
            ram: RamSection {
                primitive: vec![RamPrimitive {
                    name: "ram10".into(),
                    capacity_bits: 10_240,
                    max_width_bits: 20,
                    max_depth: 8192,
                    read_ports: 1,
                    write_ports: 1,
                    // The organizations of the Efinix `EFX_RAM10`, verbatim
                    // from `sim_models/verilog/efx_ram10.v`. The block is a
                    // 10 Kbit simple dual port; these are *all* the width/depth
                    // pairs its ports can take, so `40 × 256` — the rectangle a
                    // naive `max_width × max_depth` would invent — does not
                    // exist, and neither does a 40-bit port.
                    mode: vec![
                        RamMode {
                            width_bits: 1,
                            depth: 8192,
                        },
                        RamMode {
                            width_bits: 2,
                            depth: 4096,
                        },
                        RamMode {
                            width_bits: 4,
                            depth: 2048,
                        },
                        RamMode {
                            width_bits: 5,
                            depth: 2048,
                        },
                        RamMode {
                            width_bits: 8,
                            depth: 1024,
                        },
                        RamMode {
                            width_bits: 10,
                            depth: 1024,
                        },
                        RamMode {
                            width_bits: 16,
                            depth: 512,
                        },
                        RamMode {
                            width_bits: 20,
                            depth: 512,
                        },
                    ],
                }],
            },
            stream: StreamSection {
                primitive: vec![StreamPrimitive {
                    name: "efx_dsp48".into(),
                    valid_modes: vec![MacMode::Int8MacDual, MacMode::Int9MacSingle],
                }],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Minimal TOML subset parser / emitter
// ---------------------------------------------------------------------------

/// A small, dependency-free TOML subset parser and emitter.
///
/// Supported syntax (v1):
///
/// * comments (`#` to end of line) and blank lines;
/// * bare keys, quoted keys, and dotted key paths (`a.b.c = 1`);
/// * table headers `[a.b]` and arrays of tables `[[a.b]]`;
/// * basic strings with the standard escapes, literal strings `'...'`;
/// * integers (decimal with `_`, or `0x`/`0o`/`0b`), floats, booleans;
/// * arrays, single-line or multi-line, with trailing commas.
///
/// Explicitly unsupported (reported as errors, never guessed): inline tables
/// `{...}`, multi-line strings `"""..."""`, dates/times, and heterogeneous
/// table/array mixing that TOML itself rejects.
///
/// Values are produced as [`serde_json::Value`] so that the typed schemas stay
/// the single source of truth via `serde`; the emitter is the exact inverse for
/// the value shapes our schemas produce.
pub mod toml_lite {
    use std::fmt;

    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use serde_json::{Map, Number, Value};

    /// Syntax or schema error with an optional 1-based source line.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TomlError {
        /// Human-readable message.
        pub message: String,
        /// 1-based line number, when the error is positional.
        pub line: Option<usize>,
    }

    impl TomlError {
        /// Construct an error at a line.
        pub fn at(line: usize, message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                line: Some(line),
            }
        }

        /// Construct a positionless error (used for schema binding failures).
        pub fn plain(message: impl Into<String>) -> Self {
            Self {
                message: message.into(),
                line: None,
            }
        }
    }

    impl fmt::Display for TomlError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.line {
                Some(line) => write!(f, "line {line}: {}", self.message),
                None => write!(f, "{}", self.message),
            }
        }
    }

    impl std::error::Error for TomlError {}

    /// Parse TOML text into a JSON value tree.
    pub fn parse_toml(src: &str) -> Result<Value, TomlError> {
        Parser::new(src).parse_document()
    }

    /// Deserialize a typed value from TOML text.
    pub fn from_toml_str<T: DeserializeOwned>(src: &str) -> Result<T, TomlError> {
        let value = parse_toml(src)?;
        serde_json::from_value(value).map_err(|e| TomlError::plain(e.to_string()))
    }

    /// Serialize a typed value as deterministic TOML text.
    pub fn to_toml_string_of<T: Serialize>(value: &T) -> Result<String, TomlError> {
        let json = serde_json::to_value(value)
            .map_err(|e| TomlError::plain(format!("cannot encode value: {e}")))?;
        let Value::Object(root) = json else {
            return Err(TomlError::plain(
                "top-level TOML value must be a table".to_string(),
            ));
        };
        let mut out = String::new();
        emit_table(&root, &[], &mut out);
        Ok(out)
    }

    // ---- parser -----------------------------------------------------------

    struct Parser<'a> {
        bytes: &'a [u8],
        pos: usize,
        line: usize,
    }

    impl<'a> Parser<'a> {
        fn new(src: &'a str) -> Self {
            Self {
                bytes: src.as_bytes(),
                pos: 0,
                line: 1,
            }
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn peek_at(&self, k: usize) -> Option<u8> {
            self.bytes.get(self.pos + k).copied()
        }

        fn bump(&mut self) -> Option<u8> {
            let c = self.bytes.get(self.pos).copied();
            if let Some(b) = c {
                self.pos += 1;
                if b == b'\n' {
                    self.line += 1;
                }
            }
            c
        }

        fn err<T>(&self, message: impl Into<String>) -> Result<T, TomlError> {
            Err(TomlError::at(self.line, message))
        }

        fn skip_inline_ws(&mut self) {
            while matches!(self.peek(), Some(b' ') | Some(b'\t') | Some(b'\r')) {
                self.bump();
            }
        }

        fn skip_ws_comments_newlines(&mut self) {
            loop {
                match self.peek() {
                    Some(b' ') | Some(b'\t') | Some(b'\r') | Some(b'\n') => {
                        self.bump();
                    }
                    Some(b'#') => {
                        while !matches!(self.peek(), None | Some(b'\n')) {
                            self.bump();
                        }
                    }
                    _ => break,
                }
            }
        }

        fn skip_to_eol(&mut self) {
            while !matches!(self.peek(), None | Some(b'\n')) {
                if self.peek() == Some(b'#') {
                    break;
                }
                self.bump();
            }
            if self.peek() == Some(b'#') {
                while !matches!(self.peek(), None | Some(b'\n')) {
                    self.bump();
                }
            }
        }

        fn parse_document(mut self) -> Result<Value, TomlError> {
            let mut root = Map::new();
            // Path of the table currently accepting key/value pairs.
            let mut current: Vec<String> = Vec::new();
            loop {
                self.skip_ws_comments_newlines();
                match self.peek() {
                    None => break,
                    Some(b'[') => {
                        let array = self.peek_at(1) == Some(b'[');
                        self.bump();
                        if array {
                            self.bump();
                        }
                        self.skip_inline_ws();
                        let path = self.parse_key_path()?;
                        self.skip_inline_ws();
                        if self.peek() != Some(b']') {
                            return self.err("expected ']' to close table header");
                        }
                        self.bump();
                        if array {
                            if self.peek() != Some(b']') {
                                return self.err("expected ']]' to close array-of-tables header");
                            }
                            self.bump();
                        }
                        self.skip_inline_ws();
                        if let Some(c) = self.peek() {
                            if c != b'\n' && c != b'#' {
                                return self
                                    .err("unexpected trailing characters after table header");
                            }
                        }
                        if array {
                            open_array_of_tables(&mut root, &path)?;
                            current = path;
                        } else {
                            open_table(&mut root, &path)?;
                            current = path;
                        }
                    }
                    Some(_) => {
                        let path = self.parse_key_path()?;
                        self.skip_inline_ws();
                        if self.peek() != Some(b'=') {
                            return self.err("expected '=' after key");
                        }
                        self.bump();
                        self.skip_inline_ws();
                        let value = self.parse_value()?;
                        self.skip_to_eol();
                        insert_value(&mut root, &current, &path, value)?;
                    }
                }
            }
            Ok(Value::Object(root))
        }

        fn parse_key_path(&mut self) -> Result<Vec<String>, TomlError> {
            let mut path = Vec::new();
            loop {
                self.skip_inline_ws();
                let key = match self.peek() {
                    Some(b'"') => self.parse_basic_string()?,
                    Some(b'\'') => self.parse_literal_string()?,
                    Some(_) => {
                        let start = self.pos;
                        while let Some(c) = self.peek() {
                            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                                self.bump();
                            } else {
                                break;
                            }
                        }
                        if start == self.pos {
                            return self.err("expected a key");
                        }
                        String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned()
                    }
                    None => return self.err("unexpected end of input while reading a key"),
                };
                path.push(key);
                self.skip_inline_ws();
                if self.peek() == Some(b'.') {
                    self.bump();
                    continue;
                }
                break;
            }
            Ok(path)
        }

        fn parse_value(&mut self) -> Result<Value, TomlError> {
            match self.peek() {
                Some(b'"') => Ok(Value::String(self.parse_basic_string()?)),
                Some(b'\'') => Ok(Value::String(self.parse_literal_string()?)),
                Some(b'[') => self.parse_array(),
                Some(b'{') => self.err("inline tables are not supported"),
                Some(b't') | Some(b'f') => self.parse_bool(),
                Some(_) => self.parse_number(),
                None => self.err("unexpected end of input while reading a value"),
            }
        }

        fn parse_basic_string(&mut self) -> Result<String, TomlError> {
            if self.peek() == Some(b'"') && self.peek_at(1) == Some(b'"') {
                return self.err("multi-line strings are not supported");
            }
            self.bump(); // opening quote
            let mut out = String::new();
            loop {
                match self.bump() {
                    None => return self.err("unterminated string"),
                    Some(b'"') => break,
                    Some(b'\\') => {
                        let esc = self
                            .bump()
                            .ok_or_else(|| TomlError::at(self.line, "unterminated escape"))?;
                        match esc {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'n' => out.push('\n'),
                            b't' => out.push('\t'),
                            b'r' => out.push('\r'),
                            b'b' => out.push('\u{8}'),
                            b'f' => out.push('\u{c}'),
                            b'u' => out.push(self.parse_unicode(4)?),
                            b'U' => out.push(self.parse_unicode(8)?),
                            other => {
                                return self.err(format!("unsupported escape \\{}", other as char));
                            }
                        }
                    }
                    Some(b'\n') => return self.err("unterminated string"),
                    Some(c) => out.push(c as char),
                }
            }
            Ok(out)
        }

        fn parse_unicode(&mut self, digits: usize) -> Result<char, TomlError> {
            let mut code: u32 = 0;
            for _ in 0..digits {
                let c = self
                    .bump()
                    .ok_or_else(|| TomlError::at(self.line, "truncated unicode escape"))?;
                let d = (c as char).to_digit(16).ok_or_else(|| {
                    TomlError::at(self.line, "invalid hex digit in unicode escape")
                })?;
                code = code * 16 + d;
            }
            char::from_u32(code)
                .ok_or_else(|| TomlError::at(self.line, "invalid unicode scalar value"))
        }

        fn parse_literal_string(&mut self) -> Result<String, TomlError> {
            if self.peek() == Some(b'\'') && self.peek_at(1) == Some(b'\'') {
                return self.err("multi-line strings are not supported");
            }
            self.bump(); // opening quote
            let mut out = String::new();
            loop {
                match self.bump() {
                    None | Some(b'\n') => return self.err("unterminated literal string"),
                    Some(b'\'') => break,
                    Some(c) => out.push(c as char),
                }
            }
            Ok(out)
        }

        fn parse_bool(&mut self) -> Result<Value, TomlError> {
            let start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_alphabetic() {
                    self.bump();
                } else {
                    break;
                }
            }
            match &self.bytes[start..self.pos] {
                b"true" => Ok(Value::Bool(true)),
                b"false" => Ok(Value::Bool(false)),
                other => self.err(format!(
                    "expected a value, found {:?}",
                    String::from_utf8_lossy(other)
                )),
            }
        }

        fn parse_number(&mut self) -> Result<Value, TomlError> {
            let start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'+' || c == b'-' {
                    self.bump();
                } else {
                    break;
                }
            }
            let token = String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned();
            if token.is_empty() {
                return self.err("expected a value");
            }
            parse_number_token(&token).map_err(|m| TomlError::at(self.line, m))
        }

        fn parse_array(&mut self) -> Result<Value, TomlError> {
            self.bump(); // '['
            let mut items = Vec::new();
            loop {
                self.skip_ws_comments_newlines();
                match self.peek() {
                    Some(b']') => {
                        self.bump();
                        break;
                    }
                    None => return self.err("unterminated array"),
                    _ => {
                        items.push(self.parse_value()?);
                        self.skip_ws_comments_newlines();
                        match self.peek() {
                            Some(b',') => {
                                self.bump();
                            }
                            Some(b']') => {
                                self.bump();
                                break;
                            }
                            _ => return self.err("expected ',' or ']' in array"),
                        }
                    }
                }
            }
            Ok(Value::Array(items))
        }
    }

    fn parse_number_token(token: &str) -> Result<Value, String> {
        let negative = token.starts_with('-');
        let body = token.trim_start_matches(['+', '-']);
        let cleaned: String = body.chars().filter(|c| *c != '_').collect();

        if let Some(hex) = cleaned
            .strip_prefix("0x")
            .or_else(|| cleaned.strip_prefix("0X"))
        {
            let v =
                u64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex integer: {e}"))?;
            return integer_value(negative, v as i128);
        }
        if let Some(oct) = cleaned
            .strip_prefix("0o")
            .or_else(|| cleaned.strip_prefix("0O"))
        {
            let v =
                u64::from_str_radix(oct, 8).map_err(|e| format!("invalid octal integer: {e}"))?;
            return integer_value(negative, v as i128);
        }
        if let Some(bin) = cleaned
            .strip_prefix("0b")
            .or_else(|| cleaned.strip_prefix("0B"))
        {
            let v =
                u64::from_str_radix(bin, 2).map_err(|e| format!("invalid binary integer: {e}"))?;
            return integer_value(negative, v as i128);
        }

        let is_float = cleaned.contains('.')
            || cleaned.contains('e')
            || cleaned.contains('E')
            || cleaned.eq_ignore_ascii_case("inf")
            || cleaned.eq_ignore_ascii_case("nan");
        if is_float {
            let v: f64 = cleaned
                .parse()
                .map_err(|e| format!("invalid float {token:?}: {e}"))?;
            let v = if negative { -v } else { v };
            let n = Number::from_f64(v).ok_or_else(|| format!("float {token:?} is not finite"))?;
            return Ok(Value::Number(n));
        }

        let v: i128 = cleaned
            .parse()
            .map_err(|e| format!("invalid integer {token:?}: {e}"))?;
        integer_value(negative, v)
    }

    fn integer_value(negative: bool, magnitude: i128) -> Result<Value, String> {
        let signed = if negative { -magnitude } else { magnitude };
        if let Ok(v) = i64::try_from(signed) {
            return Ok(Value::Number(Number::from(v)));
        }
        if let Ok(v) = u64::try_from(signed) {
            return Ok(Value::Number(Number::from(v)));
        }
        Err(format!("integer {signed} does not fit in 64 bits"))
    }

    /// Descend one key, creating an empty table when absent. When the key holds
    /// an array of tables (as written by `[[path]]`), the destination is its
    /// most recently pushed element — TOML's "last element is current" rule.
    fn descend<'a>(
        parent: &'a mut Map<String, Value>,
        key: &str,
    ) -> Result<&'a mut Map<String, Value>, TomlError> {
        if !parent.contains_key(key) {
            parent.insert(key.to_string(), Value::Object(Map::new()));
        }
        match parent.get_mut(key) {
            Some(Value::Object(m)) => Ok(m),
            Some(Value::Array(items)) => match items.last_mut() {
                Some(Value::Object(m)) => Ok(m),
                _ => Err(TomlError::plain(format!(
                    "key {key:?} is an array without a current table element"
                ))),
            },
            _ => Err(TomlError::plain(format!(
                "key {key:?} is already defined as a scalar value"
            ))),
        }
    }

    fn open_table(root: &mut Map<String, Value>, path: &[String]) -> Result<(), TomlError> {
        let mut cur = root;
        for key in path {
            cur = descend(cur, key)?;
        }
        Ok(())
    }

    fn open_array_of_tables(
        root: &mut Map<String, Value>,
        path: &[String],
    ) -> Result<(), TomlError> {
        let (last, head) = path
            .split_last()
            .ok_or_else(|| TomlError::plain("empty table header"))?;
        let mut cur = root;
        for key in head {
            cur = descend(cur, key)?;
        }
        if !cur.contains_key(last) {
            cur.insert(last.to_string(), Value::Array(Vec::new()));
        }
        match cur.get_mut(last) {
            Some(Value::Array(items)) => {
                items.push(Value::Object(Map::new()));
                Ok(())
            }
            _ => Err(TomlError::plain(format!(
                "key {last:?} is already defined as a non-array value"
            ))),
        }
    }

    fn insert_value(
        root: &mut Map<String, Value>,
        current: &[String],
        path: &[String],
        value: Value,
    ) -> Result<(), TomlError> {
        let mut cur = root;
        for key in current {
            cur = descend(cur, key)?;
        }
        let (last, head) = path
            .split_last()
            .ok_or_else(|| TomlError::plain("empty key"))?;
        for key in head {
            cur = descend(cur, key)?;
        }
        if cur.contains_key(last) {
            return Err(TomlError::plain(format!(
                "duplicate key {last:?} in the same table"
            )));
        }
        cur.insert(last.clone(), value);
        Ok(())
    }

    // ---- emitter ----------------------------------------------------------

    fn is_bare_key(key: &str) -> bool {
        !key.is_empty()
            && key
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    }

    fn format_key(key: &str) -> String {
        if is_bare_key(key) {
            key.to_string()
        } else {
            format!("\"{}\"", key.replace('\\', "\\\\").replace('"', "\\\""))
        }
    }

    fn emit_scalar(value: &Value) -> String {
        match value {
            Value::Null => String::new(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => format!("\"{}\"", escape_string(s)),
            Value::Array(items) => emit_inline_array(items),
            Value::Object(_) => String::new(),
        }
    }

    fn escape_string(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                '\r' => out.push_str("\\r"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    fn emit_inline_array(items: &[Value]) -> String {
        let parts: Vec<String> = items.iter().map(emit_scalar).collect();
        format!("[{}]", parts.join(", "))
    }

    fn is_array_of_tables(value: &Value) -> bool {
        match value {
            Value::Array(items) => {
                !items.is_empty() && items.iter().all(|v| matches!(v, Value::Object(_)))
            }
            _ => false,
        }
    }

    fn emit_table(table: &Map<String, Value>, path: &[String], out: &mut String) {
        let mut scalars: Vec<(&String, &Value)> = Vec::new();
        let mut subtables: Vec<(&String, &Value)> = Vec::new();
        let mut arrays: Vec<(&String, &Value)> = Vec::new();
        for (k, v) in table {
            if matches!(v, Value::Object(_)) {
                subtables.push((k, v));
            } else if is_array_of_tables(v) {
                arrays.push((k, v));
            } else {
                scalars.push((k, v));
            }
        }
        for (k, v) in &scalars {
            out.push_str(&format!("{} = {}\n", format_key(k), emit_scalar(v)));
        }
        for (k, v) in &subtables {
            let mut child = path.to_vec();
            child.push((*k).clone());
            out.push_str(&format!(
                "\n[{}]\n",
                child
                    .iter()
                    .map(|s| format_key(s))
                    .collect::<Vec<_>>()
                    .join(".")
            ));
            if let Value::Object(m) = v {
                emit_table(m, &child, out);
            }
        }
        for (k, v) in &arrays {
            let mut child = path.to_vec();
            child.push((*k).clone());
            let header: String = child
                .iter()
                .map(|s| format_key(s))
                .collect::<Vec<_>>()
                .join(".");
            if let Value::Array(items) = v {
                for item in items {
                    out.push_str(&format!("\n[[{}]]\n", header));
                    if let Value::Object(m) = item {
                        emit_table(m, &child, out);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> Ti60Target {
        Ti60Target::ti60f225_reference()
    }

    #[test]
    fn reference_target_is_consistent() {
        let t = reference();
        t.validate().expect("reference target must validate");
        let usable = t.usable_budget().unwrap();
        assert_eq!(usable.dsp48, DspBlocks(152));
        assert_eq!(usable.ram10, Ram10Blocks(88));
        assert_eq!(usable.xlr, XlrUnits(40_568));
    }

    #[test]
    fn map_mac_dual_single_and_wide() {
        let t = reference();
        let dual = t.map_mac(8, 8, MacLanes(64)).unwrap();
        assert_eq!(dual.mode, MacMode::Int8MacDual);
        assert_eq!(dual.lanes_per_dsp48, 2);
        assert_eq!(dual.multipliers_per_lane, 1);
        assert_eq!(dual.dsp48, DspBlocks(32));

        let single = t.map_mac(9, 8, MacLanes(64)).unwrap();
        assert_eq!(single.mode, MacMode::Int9MacSingle);
        assert_eq!(single.multipliers_per_lane, 1);
        assert_eq!(single.dsp48, DspBlocks(64));

        let i4 = t.map_mac(8, 4, MacLanes(7)).unwrap();
        assert_eq!(i4.mode, MacMode::Int8MacDual);
        assert_eq!(i4.dsp48, DspBlocks(4));

        // A ten-bit interval has no ten-bit multiplier: it needs two nine-bit
        // partial products, and the DSP cost doubles with it.
        let wide = t.map_mac(10, 8, MacLanes(64)).unwrap();
        assert_eq!(wide.mode, MacMode::Int9MacSingle);
        assert_eq!(wide.multipliers_per_lane, 2);
        assert_eq!(wide.dsp48, DspBlocks(128));
        assert_eq!(
            t.map_mac(18, 8, MacLanes(1)).unwrap().multipliers_per_lane,
            2
        );

        assert!(matches!(
            t.map_mac(19, 8, MacLanes(4)),
            Err(TargetError::UnsupportedOperandWidth { .. })
        ));
        assert!(matches!(
            t.map_mac(0, 8, MacLanes(4)),
            Err(TargetError::UnsupportedOperandWidth { .. })
        ));
    }

    /// The requantization multiplier is `|acc| × M0`, and the DSP48's
    /// multiplier only takes an 18-bit operand: a wider accumulator is
    /// decomposed into cascaded partial products, one DSP48 each. The measured
    /// Ti60 requant unit (`engine/out_leaf_rq/requant_unit.map.v`) maps a
    /// 32-bit accumulator against an 18-bit `M0` to exactly two `EFX_DSP48`,
    /// so a 32-bit plan may never be told its requant unit costs one block.
    #[test]
    fn map_requant_decomposes_a_wide_accumulator_per_18_bit_chunk() {
        let t = reference();
        assert_eq!(t.map_requant(MacLanes(1), 8), DspBlocks(1));
        assert_eq!(t.map_requant(MacLanes(1), 18), DspBlocks(1));
        assert_eq!(t.map_requant(MacLanes(1), 19), DspBlocks(2));
        assert_eq!(
            t.map_requant(MacLanes(1), 32),
            DspBlocks(2),
            "the measured 32x18 requant unit is two cascaded EFX_DSP48"
        );
        assert_eq!(t.map_requant(MacLanes(4), 32), DspBlocks(8));

        // A target that declares no multiplier operand width defaults to the
        // EFX_DSP48's 18 bits rather than to one block per lane.
        let mut defaulted = reference();
        defaulted.mac.dsp_multiplier_operand_bits = 0;
        assert!(defaulted.validate().is_err());
    }

    #[test]
    fn map_buffer_tiles_with_legal_organizations() {
        let t = reference();
        let declared = t.ram.primitive[0].mode.clone();
        // 64 bits x 100 words: the cheapest real organization is 16x512, four
        // width slices and one depth slice. 20x512 also needs four blocks but
        // wastes more of each block, and no 40-bit organization exists.
        let m = t
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 64,
                    depth: 100,
                    read_ports: 1,
                    write_ports: 1,
                },
            )
            .unwrap();
        assert_eq!(m.mode.width_bits, 16);
        assert_eq!(m.mode.depth, 512);
        assert!(declared.contains(&m.mode));
        assert_eq!(m.width_padded_bits, 64);
        assert_eq!(m.depth_padded, 512);
        assert_eq!(m.blocks_per_replica, Ram10Blocks(4));
        assert_eq!(m.replication, 1);
        assert_eq!(m.blocks, Ram10Blocks(4));

        // 8 bits x 512 words fits one block, and the narrowest fitting
        // organization is the 8-bit-wide one — an 8-bit code never has to be
        // padded to a wider port.
        let m2 = t
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 8,
                    depth: 512,
                    read_ports: 2,
                    write_ports: 1,
                },
            )
            .unwrap();
        assert_eq!(m2.mode.width_bits, 8);
        assert_eq!(m2.mode.depth, 1024);
        assert_eq!(m2.blocks_per_replica, Ram10Blocks(1));
        assert_eq!(m2.replication, 2);
        assert_eq!(m2.blocks, Ram10Blocks(2));

        // 50 144 weight codes — the ink_dw weight blob at one code per entry —
        // need 49 blocks of the 8-bit-wide organization, not one per layer.
        let weights = t
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 8,
                    depth: 50_144,
                    read_ports: 1,
                    write_ports: 0,
                },
            )
            .unwrap();
        assert_eq!(weights.mode.width_bits, 8);
        assert_eq!(weights.mode.depth, 1024);
        assert_eq!(weights.blocks, Ram10Blocks(49));

        // 8 bits x 2048 words needs two blocks in the same organization, with
        // no wasted depth at all.
        let m3 = t
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 8,
                    depth: 2048,
                    read_ports: 1,
                    write_ports: 1,
                },
            )
            .unwrap();
        assert_eq!(m3.mode.width_bits, 8);
        assert_eq!(m3.depth_padded, 2048);
        assert_eq!(m3.blocks, Ram10Blocks(2));

        // Zero request is rejected.
        assert!(
            t.map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 0,
                    depth: 10,
                    read_ports: 1,
                    write_ports: 1
                }
            )
            .is_err()
        );
    }

    /// The bug this guards: `max_width × max_depth` is 40 960 bits on a
    /// 10 240-bit block. A 40-bit-wide, 1024-deep buffer must therefore cost
    /// four blocks, not one, and no reported padding may claim a 40x1024 block.
    #[test]
    fn map_buffer_never_claims_a_geometry_wider_than_one_block_holds() {
        let t = reference();
        let prim = t.default_ram_primitive().unwrap();
        let m = t
            .map_buffer(
                "ram10",
                BufferRequest {
                    width_bits: 40,
                    depth: 1024,
                    read_ports: 1,
                    write_ports: 1,
                },
            )
            .unwrap();
        assert_eq!(m.blocks, Ram10Blocks(4));
        assert!(
            m.mode.cells() <= prim.capacity_bits,
            "the selected organization {}x{} holds {} bits, above the block capacity {}",
            m.mode.width_bits,
            m.mode.depth,
            m.mode.cells(),
            prim.capacity_bits
        );
        assert!(m.capacity_bits_used <= m.blocks_per_replica.get() * prim.capacity_bits);
        assert_eq!(m.width_padded_bits, 40);
        assert_eq!(m.depth_padded, 1024);
    }

    #[test]
    fn declared_ram_modes_must_fit_one_block() {
        // A 40x1024 organization is four blocks' worth of bits: rejected.
        let mut t = reference();
        t.ram.primitive[0].mode.push(RamMode {
            width_bits: 40,
            depth: 1024,
        });
        assert!(matches!(
            t.validate(),
            Err(TargetError::InvalidValue { .. })
        ));

        // So is an organization wider than the primitive's declared maximum.
        let mut t = reference();
        t.ram.primitive[0].mode.push(RamMode {
            width_bits: 80,
            depth: 128,
        });
        assert!(t.validate().is_err());

        // A zero-width organization is not an organization.
        let mut t = reference();
        t.ram.primitive[0].mode.push(RamMode {
            width_bits: 0,
            depth: 10,
        });
        assert!(t.validate().is_err());
    }

    /// Without declared modes the fallback is a *conservative single
    /// rectangle*, never the product of the two independent maxima.
    #[test]
    fn undeclared_modes_fall_back_to_a_capacity_consistent_rectangle() {
        let mut t = reference();
        let declared = t.ram.primitive[0].mode.clone();
        t.ram.primitive[0].mode.clear();
        let modes = t.ram.primitive[0].effective_modes();
        assert_eq!(modes.len(), 1);
        assert_eq!(modes[0].width_bits, 20);
        assert_eq!(modes[0].depth, 512);
        assert!(
            declared.contains(&modes[0]),
            "the fallback must be a real EFX_RAM10 organization, not {}x{}",
            modes[0].width_bits,
            modes[0].depth
        );
        t.validate().unwrap();
    }

    #[test]
    fn ddr_and_frame_budget_arithmetic() {
        let t = reference();
        let frame = t.frame_cycles_for(Hertz::new(10)).unwrap();
        assert_eq!(frame, FrameCycles(10_000_000));
        let ddr = t.ddr_budget(frame);
        // 900 MB/s / 100 MHz * 10M cycles = 90 MB per frame.
        assert_eq!(ddr.read_bytes, ByteCount(90_000_000));
        assert_eq!(ddr.write_bytes, ByteCount(70_000_000));
    }

    #[test]
    fn reservation_over_budget_is_rejected() {
        let mut t = reference();
        t.reserved.ram10 = 999;
        assert!(matches!(
            t.validate(),
            Err(TargetError::ReservationOverBudget { .. })
        ));
    }

    /// The checked-in fixture and the built-in reference must be the same
    /// target: everything else (the CLI, the acceptance suite, the emit policy)
    /// loads the file, while the tests and the RTL examples compare against the
    /// reference, so a drift between them would silently change what a plan is
    /// judged on.
    #[test]
    fn the_checked_in_target_fixture_is_the_reference_target() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/targets/ti60f225.toml");
        let t = Ti60Target::from_toml_file(&path)
            .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
        assert_eq!(t, reference());
    }

    #[test]
    fn parses_documented_target_file() {
        let src = r#"
# Ti60 target
format = "tessera.target"
schema_version = 1
name = "ti60f225"
revision = "fpga_src-2026-10-10"

[clock]
core_hz = 100_000_000

[budget]
dsp48 = 160
ram10 = 256
xlr = 60_800
ddr_read_bytes_per_s = 900_000_000
ddr_write_bytes_per_s = 700_000_000

[reserved]
dsp48 = 8
ram10 = 168
xlr = 20_232

[mac]
int8xint8_per_dsp48 = 2
int9xint8_per_dsp48 = 1
requant_lane_dsp48 = 1
dsp_multiplier_operand_bits = 18

[logic]
xlr_per_engine = 2_500

[[ram.primitive]]
name = "ram10"
capacity_bits = 10_240
max_width_bits = 20
max_depth = 8192
read_ports = 1
write_ports = 1

[[ram.primitive.mode]]
width_bits = 1
depth = 8192

[[ram.primitive.mode]]
width_bits = 2
depth = 4096

[[ram.primitive.mode]]
width_bits = 4
depth = 2048

[[ram.primitive.mode]]
width_bits = 5
depth = 2048

[[ram.primitive.mode]]
width_bits = 8
depth = 1024

[[ram.primitive.mode]]
width_bits = 10
depth = 1024

[[ram.primitive.mode]]
width_bits = 16
depth = 512

[[ram.primitive.mode]]
width_bits = 20
depth = 512

[[stream.primitive]]
name = "efx_dsp48"
valid_modes = ["int8_mac_dual", "int9_mac_single"]
"#;
        let t = Ti60Target::from_toml_str(src).unwrap();
        assert_eq!(t, reference());
        // Round-trip through the emitter.
        let text = t.to_toml_string().unwrap();
        let t2 = Ti60Target::from_toml_str(&text).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn unknown_field_is_an_error() {
        let src = r#"
format = "tessera.target"
schema_version = 1
name = "x"
revision = "r"
surprise = 3
[clock]
core_hz = 1
[budget]
dsp48 = 1
ram10 = 1
xlr = 1
ddr_read_bytes_per_s = 1
ddr_write_bytes_per_s = 1
[mac]
int8xint8_per_dsp48 = 1
int9xint8_per_dsp48 = 1
requant_lane_dsp48 = 1
[[ram.primitive]]
name = "ram10"
capacity_bits = 10240
max_width_bits = 40
max_depth = 1024
read_ports = 1
write_ports = 1
[[stream.primitive]]
name = "efx_dsp48"
valid_modes = ["int8_mac_dual"]
"#;
        let err = Ti60Target::from_toml_str(src).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("surprise"), "unexpected message: {msg}");
    }

    #[test]
    fn unknown_mac_mode_is_an_error() {
        let src = r#"
format = "tessera.target"
schema_version = 1
name = "x"
revision = "r"
[clock]
core_hz = 1
[budget]
dsp48 = 1
ram10 = 1
xlr = 1
ddr_read_bytes_per_s = 1
ddr_write_bytes_per_s = 1
[mac]
int8xint8_per_dsp48 = 1
int9xint8_per_dsp48 = 1
requant_lane_dsp48 = 1
[[ram.primitive]]
name = "ram10"
capacity_bits = 10240
max_width_bits = 40
max_depth = 1024
read_ports = 1
write_ports = 1
[[stream.primitive]]
name = "efx_dsp48"
valid_modes = ["int8_mac_duall"]
"#;
        let err = Ti60Target::from_toml_str(src).unwrap_err();
        assert!(err.to_string().contains("int8_mac_duall"), "got: {err}");
    }

    #[test]
    fn parser_reports_line_numbers() {
        let err = toml_lite::parse_toml("a = 1\nb = \n").unwrap_err();
        assert_eq!(err.line, Some(2));
    }

    #[test]
    fn unsupported_schema_is_rejected() {
        let mut t = reference();
        t.schema_version = 7;
        assert!(matches!(
            t.validate(),
            Err(TargetError::UnsupportedSchema { .. })
        ));
    }
}
