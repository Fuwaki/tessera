//! Typed, target-independent graph IR for Tessera.
//!
//! The graph frontend imports a quantized model description into this IR. Everything
//! here describes *what the model computes*: tensors ("points"), edges, operators and
//! quantization contracts. It never encodes how the computation is realized in
//! hardware (no tiling, buffers, engines, schedules or placement); that belongs to the
//! planner and the RTL backend.
//!
//! Design rules (see `QUALITY.md` §2, §3):
//!
//! * domain distinctions that affect correctness are types, not conventions:
//!   [`PointId`], [`LayerId`], [`Channels`], [`Extent`], [`BitWidth`], [`Padding`],
//!   [`Rounding`], [`Packing`], [`LayerOp`];
//! * quantization stays explicit and is never re-inferred from lowered arithmetic:
//!   [`QuantContract`] plus per-layer [`Layer::requant`] and `bias_int32`;
//! * the output image is a distinct tensor from the internal points, represented by
//!   [`PointRef::ImageSentinel`] rather than by an out-of-range integer;
//! * newtype invariants (non-zero channels, group counts, kernel/stride/padding
//!   extents, supported bit widths, finite positive scales and qmaxes, the symmetric
//!   zero point, positive requantization coefficients) are enforced by the validating
//!   constructors *and* by the `serde` path, so a tampered artifact fails as a typed
//!   error instead of reaching arithmetic with an out-of-range value;
//! * invalid states are rejected by [`Graph::validate`], which is run by the frontend
//!   on import and by [`GraphArtifact::validate`] on load.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Format name of the persisted graph artifact.
pub const GRAPH_FORMAT: &str = "tessera.graph";
/// Schema version of the persisted graph artifact.
pub const GRAPH_SCHEMA_VERSION: u32 = 1;
/// Manifest format imported by the current frontend.
pub const MANIFEST_FORMAT: &str = "fpga_st_train.int8.v1";
/// Smallest bit width represented by [`BitWidth`].
pub const MIN_BIT_WIDTH: u8 = 2;
/// Largest bit width represented by [`BitWidth`].
pub const MAX_BIT_WIDTH: u8 = 16;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A structural, semantic or quantization violation in a graph.
///
/// Every variant names the offending layer/operator/point so a caller can report the
/// stage, identity and violated constraint rather than a bare "invalid graph".
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    Empty {
        what: &'static str,
    },
    PointScaleCount {
        points: usize,
        scales: usize,
    },
    PointScaleNonPositive {
        point: usize,
        value: f32,
    },
    InvalidBitWidth {
        what: &'static str,
        bits: u8,
    },
    QmaxMismatch {
        what: &'static str,
        declared: f32,
        expected: f32,
    },
    UnsupportedPacking {
        packing: String,
        weight_bits: u8,
    },
    ZeroDimension {
        layer: String,
        what: &'static str,
        value: usize,
    },
    BadGroups {
        layer: String,
        groups: u32,
        in_channels: u32,
        out_channels: u32,
    },
    KernelShape {
        layer: String,
        detail: String,
    },
    WeightCountMismatch {
        layer: String,
        declared: usize,
        expected: usize,
    },
    VectorLengthMismatch {
        layer: String,
        what: &'static str,
        declared: usize,
        expected: usize,
    },
    PointOutOfRange {
        layer: String,
        what: &'static str,
        point: u32,
        point_count: usize,
    },
    HeadSentinelNotUnsigned {
        layer: String,
    },
    OpReluMismatch {
        layer: String,
        op: String,
        out_relu: bool,
    },
    LayerScaleMismatch {
        layer: String,
        what: &'static str,
        declared: f32,
        expected: f32,
    },
    NonZeroZeroPoint {
        layer: String,
        value: i32,
    },
    NonFinite {
        what: String,
        value: f32,
    },
    DuplicateLayer {
        name: String,
    },
    UnknownLayer {
        op_index: usize,
        layer: String,
    },
    ConvPointMismatch {
        op_index: usize,
        layer: String,
        op_point: u32,
        layer_in_point: u32,
    },
    OpOutPointMismatch {
        op_index: usize,
        layer: String,
        declared: String,
        expected: String,
    },
    OpScaleMismatch {
        op_index: usize,
        kind: &'static str,
        declared: f32,
        expected: f32,
    },
    OpFieldMissing {
        op_index: usize,
        kind: &'static str,
        field: &'static str,
    },
    ReluNotAfterConv {
        op_index: usize,
        layer: String,
    },
    AddPointIdentity {
        op_index: usize,
        point: u32,
    },
    UnwrittenPoint {
        op_index: usize,
        kind: &'static str,
        point: u32,
    },
    InputPointRewritten {
        op_index: usize,
        layer: String,
    },
    DeadPoint {
        point: u32,
    },
    MissingOutputOp,
    QuantizeNotLast {
        op_index: usize,
    },
    ImageNotProduced,
    WeightOffsetOutOfRange {
        layer: String,
        offset: usize,
        count: usize,
        total: usize,
    },
    WeightLayoutMismatch {
        layer: String,
        what: &'static str,
        expected: usize,
        declared: usize,
    },
    WeightByteTotal {
        declared: usize,
        expected: usize,
    },
    UnsupportedOperator {
        op: String,
    },
    UnsupportedGraphOp {
        op: String,
    },
    UnsupportedLut {
        count: usize,
    },
    MissingBits,
    MissingScale {
        layer: String,
        what: &'static str,
    },
    ManifestFormat {
        path: String,
        declared: String,
    },
    WeightFileLength {
        file: String,
        declared: usize,
        actual: usize,
    },
    UnsafeWeightPath {
        file: String,
    },
    ShapeChannelMismatch {
        point: u32,
        expected: u32,
        found: u32,
    },
    ShapeMissing {
        point: u32,
    },
    ShapeMismatch {
        point: u32,
        source: u32,
        detail: String,
    },
    ShapeUnderflow {
        point: u32,
        detail: String,
    },
    NumericOverflow {
        what: &'static str,
        value: usize,
    },
    ArtifactFormat {
        declared: String,
    },
    ArtifactSchemaVersion {
        declared: u32,
    },
    MetricsStale,
    Io {
        path: String,
        detail: String,
    },
    Json {
        path: String,
        detail: String,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Empty { what } => write!(f, "graph has no {what}"),
            GraphError::PointScaleCount { points, scales } => write!(
                f,
                "graph declares {points} points but {scales} point scales"
            ),
            GraphError::PointScaleNonPositive { point, value } => write!(
                f,
                "point {point}: scale {value} must be finite and positive"
            ),
            GraphError::InvalidBitWidth { what, bits } => write!(
                f,
                "{what} bit width {bits} is outside {MIN_BIT_WIDTH}..={MAX_BIT_WIDTH}"
            ),
            GraphError::QmaxMismatch {
                what,
                declared,
                expected,
            } => write!(
                f,
                "{what} is {declared} but the declared bit widths imply {expected}"
            ),
            GraphError::UnsupportedPacking {
                packing,
                weight_bits,
            } => write!(
                f,
                "unsupported weight packing {packing:?} for {weight_bits}-bit weights"
            ),
            GraphError::ZeroDimension { layer, what, value } => {
                if layer.is_empty() {
                    write!(f, "{what} must be non-zero, found {value}")
                } else {
                    write!(f, "layer {layer}: {what} must be non-zero, found {value}")
                }
            }
            GraphError::BadGroups {
                layer,
                groups,
                in_channels,
                out_channels,
            } => {
                if layer.is_empty() {
                    write!(
                        f,
                        "groups {groups} must be >= 1 and divide in_channels {in_channels} \
                         and out_channels {out_channels}"
                    )
                } else {
                    write!(
                        f,
                        "layer {layer}: groups {groups} must be >= 1 and divide \
                         in_channels {in_channels} and out_channels {out_channels}"
                    )
                }
            }
            GraphError::KernelShape { layer, detail } => {
                write!(f, "layer {layer}: {detail}")
            }
            GraphError::WeightCountMismatch {
                layer,
                declared,
                expected,
            } => write!(
                f,
                "layer {layer}: weight_count {declared} does not match the implied {expected}"
            ),
            GraphError::VectorLengthMismatch {
                layer,
                what,
                declared,
                expected,
            } => write!(
                f,
                "layer {layer}: {what} has {declared} entries, expected {expected}"
            ),
            GraphError::PointOutOfRange {
                layer,
                what,
                point,
                point_count,
            } => write!(
                f,
                "layer {layer}: {what} {point} is outside 0..{point_count}"
            ),
            GraphError::HeadSentinelNotUnsigned { layer } => write!(
                f,
                "layer {layer}: writes the image sentinel but is not marked out_unsigned"
            ),
            GraphError::OpReluMismatch {
                layer,
                op,
                out_relu,
            } => write!(
                f,
                "layer {layer}: op {op:?} contradicts out_relu {out_relu}"
            ),
            GraphError::LayerScaleMismatch {
                layer,
                what,
                declared,
                expected,
            } => write!(
                f,
                "layer {layer}: {what} {declared} does not match the point scale {expected}"
            ),
            GraphError::NonZeroZeroPoint { layer, value } => write!(
                f,
                "layer {layer}: act_zero_point {value} is unsupported, the scheme is \
                 symmetric with zero point 0"
            ),
            GraphError::NonFinite { what, value } => {
                write!(f, "{what} is not a finite number: {value}")
            }
            GraphError::DuplicateLayer { name } => write!(f, "duplicate layer name {name:?}"),
            GraphError::UnknownLayer { op_index, layer } => {
                write!(f, "graph op {op_index} references unknown layer {layer:?}")
            }
            GraphError::ConvPointMismatch {
                op_index,
                layer,
                op_point,
                layer_in_point,
            } => write!(
                f,
                "graph op {op_index}: conv {layer} reads point {op_point} but the layer \
                 declares in_point {layer_in_point}"
            ),
            GraphError::OpOutPointMismatch {
                op_index,
                layer,
                declared,
                expected,
            } => write!(
                f,
                "graph op {op_index}: conv {layer} writes {declared} but the layer declares \
                 {expected}"
            ),
            GraphError::OpScaleMismatch {
                op_index,
                kind,
                declared,
                expected,
            } => write!(
                f,
                "graph op {op_index} ({kind}): scale {declared} does not match the point \
                 scale {expected}"
            ),
            GraphError::OpFieldMissing {
                op_index,
                kind,
                field,
            } => write!(
                f,
                "graph op {op_index} ({kind}): required field {field:?} is missing"
            ),
            GraphError::ReluNotAfterConv { op_index, layer } => write!(
                f,
                "graph op {op_index}: relu for layer {layer} is not a post-convolution \
                 activation"
            ),
            GraphError::AddPointIdentity { op_index, point } => write!(
                f,
                "graph op {op_index}: add uses point {point} as both operands"
            ),
            GraphError::UnwrittenPoint {
                op_index,
                kind,
                point,
            } => write!(
                f,
                "graph op {op_index} ({kind}): point {point} has no producer yet"
            ),
            GraphError::InputPointRewritten { op_index, layer } => write!(
                f,
                "graph op {op_index}: layer {layer} overwrites the graph input point 0"
            ),
            GraphError::DeadPoint { point } => write!(
                f,
                "point {point} is declared but never produced by any graph op"
            ),
            GraphError::MissingOutputOp => write!(
                f,
                "graph has no quantize_u8_output op and therefore no image output"
            ),
            GraphError::QuantizeNotLast { op_index } => write!(
                f,
                "graph op {op_index}: quantize_u8_output must be the final op"
            ),
            GraphError::ImageNotProduced => write!(
                f,
                "graph quantizes an image but no out_unsigned layer produced one"
            ),
            GraphError::WeightOffsetOutOfRange {
                layer,
                offset,
                count,
                total,
            } => write!(
                f,
                "layer {layer}: weight codes {offset}..{} exceed the {total}-code stream",
                offset + count
            ),
            GraphError::WeightLayoutMismatch {
                layer,
                what,
                expected,
                declared,
            } => write!(
                f,
                "layer {layer}: {what} is {declared}, expected {expected} in the packed stream"
            ),
            GraphError::WeightByteTotal { declared, expected } => write!(
                f,
                "weight stream declares {declared} bytes but the layer layout needs {expected}"
            ),
            GraphError::UnsupportedOperator { op } => write!(
                f,
                "unsupported layer operator {op:?}; supported: conv2d, conv2d+relu, \
                 depthwise_conv2d, depthwise_conv2d+relu"
            ),
            GraphError::UnsupportedGraphOp { op } => write!(
                f,
                "unsupported graph op {op:?}; supported: conv, relu, add, \
                 upsample_nearest2x, quantize_u8_output"
            ),
            GraphError::UnsupportedLut { count } => write!(
                f,
                "manifest declares {count} lut layers; the int8 convolution frontend does \
                 not implement lut_conv2d"
            ),
            GraphError::MissingBits => write!(
                f,
                "manifest declares neither bits nor a qmax that implies symmetric widths"
            ),
            GraphError::MissingScale { layer, what } => write!(
                f,
                "layer {layer}: {what} is required for the image sentinel but was not provided"
            ),
            GraphError::ManifestFormat { path, declared } => write!(
                f,
                "{path} declares format {declared:?} but this frontend imports {MANIFEST_FORMAT:?}"
            ),
            GraphError::WeightFileLength {
                file,
                declared,
                actual,
            } => write!(
                f,
                "weight file {file} is {actual} bytes but the manifest declares {declared}"
            ),
            GraphError::UnsafeWeightPath { file } => write!(
                f,
                "weight file {file:?} must be a plain relative name inside the artifact \
                 directory: no root, no prefix and no \"..\" component"
            ),
            GraphError::ShapeChannelMismatch {
                point,
                expected,
                found,
            } => write!(
                f,
                "point {point}: consumer expects {expected} channels, producer gives {found}"
            ),
            GraphError::ShapeMissing { point } => write!(
                f,
                "point {point}: shape is unknown; the op order does not produce it first"
            ),
            GraphError::ShapeMismatch {
                point,
                source,
                detail,
            } => write!(
                f,
                "point {point} and source point {source} cannot be combined: {detail}"
            ),
            GraphError::ShapeUnderflow { point, detail } => write!(
                f,
                "point {point}: spatial extent too small for the convolution: {detail}"
            ),
            GraphError::NumericOverflow { what, value } => {
                write!(f, "{what} overflows the representable range at {value}")
            }
            GraphError::ArtifactFormat { declared } => {
                write!(f, "artifact format {declared:?} is not {GRAPH_FORMAT:?}")
            }
            GraphError::ArtifactSchemaVersion { declared } => write!(
                f,
                "artifact schema version {declared} is not {GRAPH_SCHEMA_VERSION}"
            ),
            GraphError::MetricsStale => write!(
                f,
                "artifact metrics do not match the metrics recomputed from its graph"
            ),
            GraphError::Io { path, detail } => write!(f, "cannot access {path}: {detail}"),
            GraphError::Json { path, detail } => write!(f, "invalid JSON in {path}: {detail}"),
        }
    }
}

impl std::error::Error for GraphError {}

type GResult<T> = Result<T, GraphError>;

// ---------------------------------------------------------------------------
// Identifiers and scalars
// ---------------------------------------------------------------------------

/// Index of a quantized activation tensor ("point") inside [`Graph::points`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PointId(u32);

impl PointId {
    pub fn new(value: u32) -> Self {
        PointId(value)
    }
    pub fn get(self) -> u32 {
        self.0
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable layer identity in graph order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerId(u32);

impl LayerId {
    pub fn new(value: u32) -> Self {
        LayerId(value)
    }
    pub fn get(self) -> u32 {
        self.0
    }
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Model-given activation name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerName(String);

impl LayerName {
    pub fn new(name: impl Into<String>) -> Self {
        LayerName(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LayerName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Model-given point name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PointName(String);

impl PointName {
    pub fn new(name: impl Into<String>) -> Self {
        PointName(name.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PointName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Quantized element count along one axis; always non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Channels(u32);

impl<'de> Deserialize<'de> for Channels {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Channels::try_new(value).map_err(serde::de::Error::custom)
    }
}

impl Channels {
    /// Crate-internal constructor for values already proven non-zero at the boundary.
    pub(crate) const fn from_validated(value: u32) -> Self {
        Channels(value)
    }
    pub fn try_new(value: u32) -> GResult<Self> {
        if value == 0 {
            return Err(GraphError::ZeroDimension {
                layer: String::new(),
                what: "channels",
                value: 0,
            });
        }
        Ok(Channels(value))
    }
    /// Raw constructor for deserialization; [`Graph::validate`] checks the invariant.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<Channels> for u32 {
    fn from(value: Channels) -> u32 {
        value.0
    }
}

/// A spatial, kernel or padding extent; always non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Extent(u32);

impl<'de> Deserialize<'de> for Extent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Extent::try_new(value).map_err(serde::de::Error::custom)
    }
}

impl Extent {
    pub fn try_new(value: u32) -> GResult<Self> {
        if value == 0 {
            return Err(GraphError::ZeroDimension {
                layer: String::new(),
                what: "extent",
                value: 0,
            });
        }
        Ok(Extent(value))
    }
    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<Extent> for u32 {
    fn from(value: Extent) -> u32 {
        value.0
    }
}

/// Convolution group count; always >= 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct GroupCount(u32);

impl<'de> Deserialize<'de> for GroupCount {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        GroupCount::try_new(value).map_err(serde::de::Error::custom)
    }
}

impl GroupCount {
    /// Crate-internal constructor for values already proven non-zero at the boundary.
    pub(crate) const fn from_validated(value: u32) -> Self {
        GroupCount(value)
    }
    pub fn try_new(value: u32) -> GResult<Self> {
        if value == 0 {
            return Err(GraphError::BadGroups {
                layer: String::new(),
                groups: value,
                in_channels: 0,
                out_channels: 0,
            });
        }
        Ok(GroupCount(value))
    }
    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<GroupCount> for u32 {
    fn from(value: GroupCount) -> u32 {
        value.0
    }
}

/// A supported integer bit width for activations or weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct BitWidth(u8);

impl<'de> Deserialize<'de> for BitWidth {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        BitWidth::try_new("declared", value).map_err(serde::de::Error::custom)
    }
}

impl BitWidth {
    pub fn try_new(what: &'static str, bits: u8) -> GResult<Self> {
        if !(MIN_BIT_WIDTH..=MAX_BIT_WIDTH).contains(&bits) {
            return Err(GraphError::InvalidBitWidth { what, bits });
        }
        Ok(BitWidth(bits))
    }
    pub fn get(self) -> u8 {
        self.0
    }
    /// Largest magnitude of a symmetric two's-complement code: `2^(bits-1) - 1`.
    pub fn signed_qmax(self) -> u32 {
        (1u32 << (self.0 - 1)) - 1
    }
    /// Largest code of an unsigned tensor: `2^bits - 1`.
    pub fn unsigned_qmax(self) -> u32 {
        (1u32 << self.0) - 1
    }
    /// Inverse of [`BitWidth::signed_qmax`] for a declared floating-point qmax.
    pub fn from_signed_qmax(qmax: f32) -> Option<Self> {
        (MIN_BIT_WIDTH..=MAX_BIT_WIDTH)
            .find(|bits| BitWidth(*bits).signed_qmax() as f32 == qmax)
            .map(BitWidth)
    }
}

/// Bit widths of an activation/weight scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bits {
    pub acts: BitWidth,
    pub weights: BitWidth,
}

/// A finite, positive scale factor (real units per code).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Scale(f32);

impl<'de> Deserialize<'de> for Scale {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Scale::try_new("declared scale", value).map_err(serde::de::Error::custom)
    }
}

impl Scale {
    pub fn try_new(what: &'static str, value: f32) -> GResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(GraphError::NonFinite {
                what: what.to_owned(),
                value,
            });
        }
        Ok(Scale(value))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// A finite, positive saturation limit expressed as an integer code magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub struct QMax(f32);

impl<'de> Deserialize<'de> for QMax {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        QMax::try_new("declared qmax", value).map_err(serde::de::Error::custom)
    }
}

impl QMax {
    pub fn try_new(what: &'static str, value: f32) -> GResult<Self> {
        if !value.is_finite() || value <= 0.0 {
            return Err(GraphError::NonFinite {
                what: what.to_owned(),
                value,
            });
        }
        Ok(QMax(value))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Activation zero point. The supported scheme is symmetric, so this is always 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ZeroPoint(i32);

impl<'de> Deserialize<'de> for ZeroPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = i32::deserialize(deserializer)?;
        if value != 0 {
            return Err(serde::de::Error::custom(GraphError::NonZeroZeroPoint {
                layer: "graph".to_owned(),
                value,
            }));
        }
        Ok(ZeroPoint(0))
    }
}

impl ZeroPoint {
    pub fn symmetric() -> Self {
        ZeroPoint(0)
    }
    pub fn get(self) -> i32 {
        self.0
    }
}

/// Element type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    Int8,
    Int4,
    UInt8,
    Int32,
}

/// Rounding applied when a wide accumulator is requantized to an activation code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rounding {
    /// Round half away from zero; ties go to the larger magnitude. Matches the
    /// reference integer interpreter and the NNgen `rnd_half_away` oracle.
    HalfAwayFromZero,
}

impl Default for Rounding {
    fn default() -> Self {
        Rounding::HalfAwayFromZero
    }
}

/// How signed weight codes are packed into `weights.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Packing {
    /// One 8-bit code per byte, two's complement.
    I8x1,
    /// Two 4-bit codes per byte, low nibble first, each two's complement.
    I4x2,
}

impl Packing {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "i8-x1" => Some(Packing::I8x1),
            "i4-x2" => Some(Packing::I4x2),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Packing::I8x1 => "i8-x1",
            Packing::I4x2 => "i4-x2",
        }
    }
    /// Number of codes stored in one byte.
    pub fn codes_per_byte(self) -> usize {
        match self {
            Packing::I8x1 => 1,
            Packing::I4x2 => 2,
        }
    }
    /// Weight bit width implied by the packing.
    pub fn weight_bits(self) -> u8 {
        match self {
            Packing::I8x1 => 8,
            Packing::I4x2 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Convolution kind implied by the manifest operator string and geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerKind {
    Conv2d,
    DepthwiseConv2d,
    PointwiseConv2d,
}

/// The manifest layer operator, including its fused post-convolution activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerOp {
    Conv2d,
    Conv2dRelu,
    DepthwiseConv2d,
    DepthwiseConv2dRelu,
}

impl LayerOp {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "conv2d" => Some(LayerOp::Conv2d),
            "conv2d+relu" => Some(LayerOp::Conv2dRelu),
            "depthwise_conv2d" => Some(LayerOp::DepthwiseConv2d),
            "depthwise_conv2d+relu" => Some(LayerOp::DepthwiseConv2dRelu),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            LayerOp::Conv2d => "conv2d",
            LayerOp::Conv2dRelu => "conv2d+relu",
            LayerOp::DepthwiseConv2d => "depthwise_conv2d",
            LayerOp::DepthwiseConv2dRelu => "depthwise_conv2d+relu",
        }
    }
    pub fn has_relu(self) -> bool {
        matches!(self, LayerOp::Conv2dRelu | LayerOp::DepthwiseConv2dRelu)
    }
    pub fn is_depthwise(self) -> bool {
        matches!(
            self,
            LayerOp::DepthwiseConv2d | LayerOp::DepthwiseConv2dRelu
        )
    }
}

/// Kind of a step in the execution-ordered graph op list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphOpKind {
    Conv,
    Relu,
    Add,
    UpsampleNearest2x,
    QuantizeU8Output,
}

/// One step of the execution-ordered graph.
///
/// Convolutions appear here as well, so the list is a complete dataflow description.
/// `upsample_nearest2x` and `add` update their point in place, exactly as the
/// reference integer interpreter does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphOp {
    /// Apply layer `layer`, reading `in_point` and writing `out_point`.
    Conv {
        layer: LayerName,
        in_point: PointId,
        out_point: PointRef,
    },
    /// Fused activation of `layer` (no state change; exists for inspection fidelity).
    Relu { layer: LayerName },
    /// `point += requant(source_point, point_scale)`.
    Add {
        point: PointId,
        source_point: PointId,
        scale: Scale,
    },
    /// Double `point` in both spatial axes, nearest neighbour.
    UpsampleNearest2x { point: PointId },
    /// Quantize the produced image to unsigned 8-bit codes.
    QuantizeU8Output { scale: Scale },
}

impl GraphOp {
    pub fn kind(&self) -> GraphOpKind {
        match self {
            GraphOp::Conv { .. } => GraphOpKind::Conv,
            GraphOp::Relu { .. } => GraphOpKind::Relu,
            GraphOp::Add { .. } => GraphOpKind::Add,
            GraphOp::UpsampleNearest2x { .. } => GraphOpKind::UpsampleNearest2x,
            GraphOp::QuantizeU8Output { .. } => GraphOpKind::QuantizeU8Output,
        }
    }
    /// Human-readable op tag, matching the manifest vocabulary.
    pub fn tag(&self) -> &'static str {
        match self {
            GraphOp::Conv { .. } => "conv",
            GraphOp::Relu { .. } => "relu",
            GraphOp::Add { .. } => "add",
            GraphOp::UpsampleNearest2x { .. } => "upsample_nearest2x",
            GraphOp::QuantizeU8Output { .. } => "quantize_u8_output",
        }
    }
    pub fn layer(&self) -> Option<&LayerName> {
        match self {
            GraphOp::Conv { layer, .. } | GraphOp::Relu { layer } => Some(layer),
            _ => None,
        }
    }
    /// The primary point this op reads or updates.
    pub fn point(&self) -> Option<PointId> {
        match self {
            GraphOp::Conv { in_point, .. } => Some(*in_point),
            GraphOp::Add { point, .. } | GraphOp::UpsampleNearest2x { point } => Some(*point),
            _ => None,
        }
    }
    pub fn source_point(&self) -> Option<PointId> {
        match self {
            GraphOp::Add { source_point, .. } => Some(*source_point),
            _ => None,
        }
    }
    pub fn scale(&self) -> Option<Scale> {
        match self {
            GraphOp::Add { scale, .. } | GraphOp::QuantizeU8Output { scale } => Some(*scale),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry and padding
// ---------------------------------------------------------------------------

/// Convolution kernel geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Kernel {
    pub h: Extent,
    pub w: Extent,
}

impl<'de> Deserialize<'de> for Kernel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            h: u32,
            w: u32,
        }
        let raw = Raw::deserialize(deserializer)?;
        Kernel::try_new(raw.h, raw.w).map_err(serde::de::Error::custom)
    }
}

impl Kernel {
    pub fn try_new(h: u32, w: u32) -> GResult<Self> {
        Ok(Kernel {
            h: Extent::try_new(h).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "kernel height",
                value: h as usize,
            })?,
            w: Extent::try_new(w).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "kernel width",
                value: w as usize,
            })?,
        })
    }
    pub fn h(&self) -> Extent {
        self.h
    }
    pub fn w(&self) -> Extent {
        self.w
    }
}

/// Convolution stride geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Stride {
    pub h: Extent,
    pub w: Extent,
}

impl<'de> Deserialize<'de> for Stride {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            h: u32,
            w: u32,
        }
        let raw = Raw::deserialize(deserializer)?;
        Stride::try_new(raw.h, raw.w).map_err(serde::de::Error::custom)
    }
}

impl Stride {
    pub fn try_new(h: u32, w: u32) -> GResult<Self> {
        Ok(Stride {
            h: Extent::try_new(h).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "stride height",
                value: h as usize,
            })?,
            w: Extent::try_new(w).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "stride width",
                value: w as usize,
            })?,
        })
    }
    pub fn h(&self) -> Extent {
        self.h
    }
    pub fn w(&self) -> Extent {
        self.w
    }
}

/// Padding applied before a valid convolution.
///
/// The manifest uses reflection padding; it is preserved explicitly rather than being
/// normalized away, because the border semantics of zero padding and reflection
/// padding differ and the RTL backend must be able to reject the wrong mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Padding {
    None,
    /// Reflect border padding of the given amount on all four sides.
    Reflect(Extent),
    /// Zero padding of the given amount on all four sides.
    Zero(Extent),
}

impl<'de> Deserialize<'de> for Padding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        /// Wire shape of [`Padding`]: the payload is the raw amount.
        #[derive(Deserialize)]
        enum Raw {
            None,
            Reflect(u32),
            Zero(u32),
        }
        match Raw::deserialize(deserializer)? {
            Raw::None => Ok(Padding::None),
            Raw::Reflect(amount) => Padding::reflect(amount).map_err(serde::de::Error::custom),
            Raw::Zero(amount) => Padding::zero(amount).map_err(serde::de::Error::custom),
        }
    }
}

impl Padding {
    pub fn none() -> Self {
        Padding::None
    }
    pub fn reflect(amount: u32) -> GResult<Self> {
        Ok(Padding::Reflect(Extent::try_new(amount).map_err(|_| {
            GraphError::ZeroDimension {
                layer: String::new(),
                what: "reflect padding",
                value: 0,
            }
        })?))
    }
    pub fn zero(amount: u32) -> GResult<Self> {
        Ok(Padding::Zero(Extent::try_new(amount).map_err(|_| {
            GraphError::ZeroDimension {
                layer: String::new(),
                what: "zero padding",
                value: 0,
            }
        })?))
    }
    pub fn amount(self) -> Option<Extent> {
        match self {
            Padding::None => None,
            Padding::Reflect(amount) | Padding::Zero(amount) => Some(amount),
        }
    }
    pub fn is_reflect(self) -> bool {
        matches!(self, Padding::Reflect(_))
    }
}

// ---------------------------------------------------------------------------
// Requantization
// ---------------------------------------------------------------------------

/// Per-output-channel requantization coefficients: `out_code = round(acc * r)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct RequantCoefficients(Vec<f32>);

impl<'de> Deserialize<'de> for RequantCoefficients {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let values = Vec::<f32>::deserialize(deserializer)?;
        RequantCoefficients::try_new(values).map_err(serde::de::Error::custom)
    }
}

impl RequantCoefficients {
    pub fn try_new(values: Vec<f32>) -> GResult<Self> {
        for value in &values {
            if !value.is_finite() || *value <= 0.0 {
                return Err(GraphError::NonFinite {
                    what: "requant coefficient".to_owned(),
                    value: *value,
                });
            }
        }
        Ok(RequantCoefficients(values))
    }
    pub fn values(&self) -> &[f32] {
        &self.0
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Largest relative approximation error of [`Self::fixed_point`] for `shift`.
    pub fn relative_error(&self, shift: u32) -> f32 {
        let denom = 1i64 << shift;
        self.0
            .iter()
            .map(|r| {
                let m = round_half_away_f64(*r as f64 * denom as f64);
                let approx = m / denom as f64;
                ((approx - *r as f64).abs() / (*r as f64).abs().max(1e-30)) as f32
            })
            .fold(0.0f32, f32::max)
    }
    /// Largest multiplier magnitude produced by [`Self::fixed_point`] for `shift`.
    pub fn max_multiplier(&self, shift: u32) -> i64 {
        let denom = 1i64 << shift;
        self.0
            .iter()
            .map(|r| round_half_away_f64(*r as f64 * denom as f64) as i64)
            .map(i64::abs)
            .max()
            .unwrap_or(0)
    }
    /// Normalize the coefficients into integer multipliers plus a shared right shift.
    ///
    /// The largest shift `s <= max_shift` is chosen such that every
    /// `round(r_c * 2^s)` fits in `max_multiplier`; [`Rounding::HalfAwayFromZero`]
    /// matches the reference integer interpreter and the NNgen oracle.
    pub fn fixed_point(&self, max_multiplier: i64, max_shift: u32) -> FixedPointRequant {
        let max_r = self.0.iter().cloned().fold(0.0f32, f32::max);
        let mut shift = 0u32;
        if max_r > 0.0 {
            while shift < max_shift {
                let candidate = shift + 1;
                let m = round_half_away_f64(max_r as f64 * (1i64 << candidate) as f64);
                if m <= max_multiplier as f64 {
                    shift = candidate;
                } else {
                    break;
                }
            }
        }
        let denom = 1i64 << shift;
        let multipliers = self
            .0
            .iter()
            .map(|r| round_half_away_f64(*r as f64 * denom as f64) as i64)
            .collect();
        FixedPointRequant {
            multipliers,
            shift,
            rounding: Rounding::HalfAwayFromZero,
        }
    }
}

/// Integer requantization: `out_code = clamp(sign(acc) * ((|acc| * m_c) >> shift))`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedPointRequant {
    pub multipliers: Vec<i64>,
    pub shift: u32,
    pub rounding: Rounding,
}

/// `round(x)` with ties away from zero, on the real line.
fn round_half_away_f64(value: f64) -> f64 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        -((-value + 0.5).floor())
    }
}

// ---------------------------------------------------------------------------
// Quantization contract
// ---------------------------------------------------------------------------

/// The quantization contract shared by every tensor in the graph.
///
/// Quantization is semantic data: it stays explicit through lowering instead of being
/// re-derived from lowered arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantContract {
    pub bits: Bits,
    /// Largest magnitude of an internal activation code (symmetric, zero point 0).
    pub act_qmax: QMax,
    /// Largest activation code of the unsigned output image.
    pub output_qmax: QMax,
    /// Largest magnitude of a weight code (weights are signed).
    pub weight_qmax: QMax,
    pub act_zero_point: ZeroPoint,
    /// Accumulator element type before requantization.
    pub accumulator: Dtype,
    pub requant_rounding: Rounding,
    pub packing: Packing,
}

impl QuantContract {
    pub fn bits(&self) -> Bits {
        self.bits
    }
    pub fn act_qmax(&self) -> f32 {
        self.act_qmax.get()
    }
    pub fn output_qmax(&self) -> f32 {
        self.output_qmax.get()
    }
    pub fn weight_qmax(&self) -> f32 {
        self.weight_qmax.get()
    }
    pub fn packing(&self) -> Packing {
        self.packing
    }
    pub fn codes_per_byte(&self) -> usize {
        self.packing.codes_per_byte()
    }
    fn check(&self) -> GResult<()> {
        // Every qmax below is derived from the widths, so re-check the supported range
        // here: an out-of-range width would underflow `signed_qmax`.
        BitWidth::try_new("acts", self.bits.acts.get())?;
        BitWidth::try_new("weights", self.bits.weights.get())?;
        let expected_act = self.bits.acts.signed_qmax() as f32;
        if qmax_eq(self.act_qmax.get(), expected_act) == false {
            return Err(GraphError::QmaxMismatch {
                what: "act_qmax",
                declared: self.act_qmax.get(),
                expected: expected_act,
            });
        }
        let expected_out = self.bits.acts.unsigned_qmax() as f32;
        if qmax_eq(self.output_qmax.get(), expected_out) == false {
            return Err(GraphError::QmaxMismatch {
                what: "output_qmax",
                declared: self.output_qmax.get(),
                expected: expected_out,
            });
        }
        let expected_weight = self.bits.weights.signed_qmax() as f32;
        if qmax_eq(self.weight_qmax.get(), expected_weight) == false {
            return Err(GraphError::QmaxMismatch {
                what: "weight_qmax",
                declared: self.weight_qmax.get(),
                expected: expected_weight,
            });
        }
        if self.act_zero_point.get() != 0 {
            return Err(GraphError::NonZeroZeroPoint {
                layer: "graph".to_owned(),
                value: self.act_zero_point.get(),
            });
        }
        if self.packing.weight_bits() != self.bits.weights.get() {
            return Err(GraphError::UnsupportedPacking {
                packing: self.packing.as_str().to_owned(),
                weight_bits: self.bits.weights.get(),
            });
        }
        Ok(())
    }
}

fn qmax_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-3
}

/// Relative float equality used for scale cross-checks.
pub(crate) fn scale_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1e-12)
}

// ---------------------------------------------------------------------------
// Tensors and layers
// ---------------------------------------------------------------------------

/// A quantized activation tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub id: PointId,
    pub name: PointName,
    /// Real units per code for this tensor.
    pub scale: Scale,
}

/// A producer slot: either an internal point or the output image.
///
/// The image is not an element of [`Graph::points`]; encoding it as an enum keeps the
/// sentinel from being confused with a real index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PointRef {
    Point(PointId),
    /// The unsigned image produced by an `out_unsigned` layer.
    ImageSentinel,
}

impl PointRef {
    pub fn id(self) -> Option<PointId> {
        match self {
            PointRef::Point(id) => Some(id),
            PointRef::ImageSentinel => None,
        }
    }
    pub fn is_image(self) -> bool {
        matches!(self, PointRef::ImageSentinel)
    }
}

/// One convolution of the model, in graph order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: LayerName,
    /// Manifest operator string, including the fused activation.
    pub op: LayerOp,
    /// Structural classification derived from `op` plus geometry.
    pub kind: LayerKind,
    pub in_channels: Channels,
    pub out_channels: Channels,
    pub groups: GroupCount,
    pub act_bits: BitWidth,
    pub weight_bits: BitWidth,
    pub act_zero_point: ZeroPoint,
    /// A ReLU sits between the accumulator and requantization, so output codes clamp
    /// to `[0, act_qmax]` instead of `[-act_qmax, act_qmax]`.
    pub out_relu: bool,
    /// Output codes are unsigned `[0, 2^act_bits - 1]`. Only the head sets this.
    pub out_unsigned: bool,
    pub kernel: Kernel,
    pub stride: Stride,
    pub padding: Padding,
    /// Quantized input point.
    pub in_point: PointId,
    /// Quantized output point, or the image sentinel for the head.
    pub out_point: PointRef,
    pub in_scale: Scale,
    /// Scale of this layer's output. For the image sentinel this is the image scale.
    pub out_scale: Scale,
    /// Per-output-channel weight dequantization scale (empty when the manifest omits it).
    pub weight_scale: Vec<f32>,
    /// fp32 bias, for reference only.
    pub bias: Vec<f32>,
    /// `round(bias / (in_scale * weight_scale))`, added to the int32 accumulator.
    pub bias_int32: Vec<i32>,
    pub requant: RequantCoefficients,
    /// Offset of this layer's codes inside the unpacked code stream.
    pub weight_offset: usize,
    /// Number of weight codes in this layer.
    pub weight_count: usize,
    /// Offset of this layer's codes inside `weights.bin`.
    pub weight_byte_offset: usize,
    /// Bytes this layer occupies in `weights.bin`.
    pub weight_bytes: usize,
}

impl Layer {
    /// Multipy-accumulate count for one output pixel of this layer.
    pub fn macs_per_pixel(&self) -> u64 {
        let groups = self.groups.get().max(1) as u64;
        let in_per_group = self.in_channels.get() as u64 / groups;
        self.out_channels.get() as u64
            * in_per_group
            * self.kernel.h.get() as u64
            * self.kernel.w.get() as u64
    }
    /// Weight codes implied by the geometry and grouping.
    pub fn expected_weight_count(&self) -> usize {
        let in_per_group = (self.in_channels.get() / self.groups.get().max(1)) as usize;
        self.out_channels.get() as usize
            * in_per_group
            * self.kernel.h.get() as usize
            * self.kernel.w.get() as usize
    }
}

/// Description of the packed weight blob consumed alongside the graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightStream {
    /// File name relative to the manifest directory.
    pub file: String,
    /// Size of the file in bytes.
    pub byte_len: usize,
    /// Total number of weight codes across all layers.
    pub code_count: usize,
    pub packing: Packing,
}

// ---------------------------------------------------------------------------
// Shape inference
// ---------------------------------------------------------------------------

/// Channels and spatial extent of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape {
    pub channels: Channels,
    pub height: Extent,
    pub width: Extent,
}

impl TensorShape {
    pub fn try_new(channels: u32, height: u32, width: u32) -> GResult<Self> {
        Ok(TensorShape {
            channels: Channels::try_new(channels).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "shape channels",
                value: channels as usize,
            })?,
            height: Extent::try_new(height).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "shape height",
                value: height as usize,
            })?,
            width: Extent::try_new(width).map_err(|_| GraphError::ZeroDimension {
                layer: String::new(),
                what: "shape width",
                value: width as usize,
            })?,
        })
    }
    pub fn elements(&self) -> u64 {
        self.channels.get() as u64 * self.height.get() as u64 * self.width.get() as u64
    }
}

/// Result of propagating a concrete input tensor through the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeReport {
    /// Shape of every internal point, indexed by [`PointId`].
    pub point_shapes: Vec<TensorShape>,
    /// Shape of the output image.
    pub image: TensorShape,
    /// Total multiply-accumulate operations over the inferred shape.
    pub macs: u64,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Static, target-independent graph metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub point_count: usize,
    pub graph_op_count: usize,
    pub layer_count: usize,
    pub conv_layer_count: usize,
    pub depthwise_layer_count: usize,
    pub pointwise_layer_count: usize,
    pub relu_layer_count: usize,
    pub add_count: usize,
    pub upsample_count: usize,
    pub weight_codes: usize,
    pub weight_bytes: usize,
    pub macs_per_output_pixel: u64,
    pub per_layer: Vec<LayerMacs>,
    pub point_fanout: Vec<PointFanout>,
}

/// Per-layer contribution to [`GraphMetrics`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerMacs {
    pub name: LayerName,
    pub macs_per_pixel: u64,
    pub weight_codes: usize,
}

/// Consumer count of one point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointFanout {
    pub point: PointId,
    pub consumers: u32,
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// The typed, target-independent graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// Source format this graph was imported from, e.g. `fpga_st_train.int8.v1`.
    pub source_format: String,
    /// Source-declared architecture family.
    pub arch: String,
    pub quant: QuantContract,
    pub points: Vec<Point>,
    /// Execution order.
    pub ops: Vec<GraphOp>,
    pub layers: Vec<Layer>,
    pub weight_stream: WeightStream,
    /// Free-form source notes, preserved for traceability.
    pub notes: Vec<String>,
}

impl Graph {
    /// Execution-ordered ops.
    pub fn ops(&self) -> &[GraphOp] {
        &self.ops
    }
    /// Convolutions in graph order.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
    pub fn layer(&self, name: &LayerName) -> Option<&Layer> {
        self.layers.iter().find(|layer| &layer.name == name)
    }
    pub fn layer_named(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|layer| layer.name.as_str() == name)
    }
    pub fn points(&self) -> &[Point] {
        &self.points
    }
    /// Number of internal points. The image sentinel has id equal to this value.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
    /// Point scales in id order. Each point owns its scale, so this borrows an iterator
    /// rather than a second copy of the array.
    pub fn point_scales(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.points.iter().map(|point| point.scale.get())
    }
    pub fn input_point(&self) -> PointId {
        PointId::new(0)
    }
    /// The id of the output-image sentinel (`point_count`).
    pub fn head_sentinel(&self) -> PointId {
        PointId::new(self.points.len() as u32)
    }
    pub fn quant(&self) -> &QuantContract {
        &self.quant
    }
    pub fn bits(&self) -> Bits {
        self.quant.bits
    }
    pub fn act_qmax(&self) -> f32 {
        self.quant.act_qmax()
    }
    pub fn output_qmax(&self) -> f32 {
        self.quant.output_qmax()
    }
    pub fn weight_qmax(&self) -> f32 {
        self.quant.weight_qmax()
    }
    pub fn packing(&self) -> Packing {
        self.quant.packing
    }
    pub fn weight_stream(&self) -> &WeightStream {
        &self.weight_stream
    }
    pub fn arch(&self) -> &str {
        &self.arch
    }
    pub fn notes(&self) -> &[String] {
        &self.notes
    }
    /// Layers that produce the unsigned image.
    pub fn head_layers(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter().filter(|layer| layer.out_unsigned)
    }
    pub fn point_scale(&self, point: PointId) -> Option<f32> {
        self.points.get(point.index()).map(|p| p.scale.get())
    }

    /// Check every structural, group, point-reference and quant-metadata invariant.
    pub fn validate(&self) -> GResult<()> {
        if self.points.is_empty() {
            return Err(GraphError::Empty { what: "points" });
        }
        if self.layers.is_empty() {
            return Err(GraphError::Empty { what: "layers" });
        }
        if self.ops.is_empty() {
            return Err(GraphError::Empty { what: "graph ops" });
        }
        let point_count = self.points.len();
        if point_count > u32::MAX as usize {
            return Err(GraphError::NumericOverflow {
                what: "point count",
                value: point_count,
            });
        }
        // Re-check the scalar invariants that the constructors enforce, so a graph
        // assembled or deserialized outside them still fails as a typed error instead of
        // computing with an out-of-range value. The contract validates its own widths
        // before deriving qmaxes from them.
        check_weight_file(&self.weight_stream.file)?;
        self.quant.check()?;

        for (index, point) in self.points.iter().enumerate() {
            if point.id.get() as usize != index {
                return Err(GraphError::PointOutOfRange {
                    layer: "graph".to_owned(),
                    what: "point id",
                    point: point.id.get(),
                    point_count,
                });
            }
            if !point.scale.get().is_finite() || point.scale.get() <= 0.0 {
                return Err(GraphError::PointScaleNonPositive {
                    point: index,
                    value: point.scale.get(),
                });
            }
        }

        // Layer-local checks.
        let mut seen: Vec<&str> = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.id.get() as usize != index {
                return Err(GraphError::DuplicateLayer {
                    name: layer.name.to_string(),
                });
            }
            if seen.iter().any(|name| *name == layer.name.as_str()) {
                return Err(GraphError::DuplicateLayer {
                    name: layer.name.to_string(),
                });
            }
            seen.push(layer.name.as_str());
            self.validate_layer(layer, point_count)?;
        }

        self.validate_weight_layout()?;
        self.validate_ops(point_count)?;
        Ok(())
    }

    fn validate_layer(&self, layer: &Layer, point_count: usize) -> GResult<()> {
        let name = layer.name.to_string();
        let in_channels = layer.in_channels.get();
        let out_channels = layer.out_channels.get();
        if in_channels == 0 {
            return Err(GraphError::ZeroDimension {
                layer: name,
                what: "in_channels",
                value: 0,
            });
        }
        if out_channels == 0 {
            return Err(GraphError::ZeroDimension {
                layer: name,
                what: "out_channels",
                value: 0,
            });
        }
        let groups = layer.groups.get();
        if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(GraphError::BadGroups {
                layer: name,
                groups,
                in_channels,
                out_channels,
            });
        }
        // The geometry extents are non-zero by construction; re-check them here because
        // a zero stride would divide by zero in shape inference and lowering.
        for (what, value) in [
            ("kernel height", layer.kernel.h.get()),
            ("kernel width", layer.kernel.w.get()),
            ("stride height", layer.stride.h.get()),
            ("stride width", layer.stride.w.get()),
        ] {
            if value == 0 {
                return Err(GraphError::ZeroDimension {
                    layer: name.clone(),
                    what,
                    value: value as usize,
                });
            }
        }
        if let Some(amount) = layer.padding.amount() {
            if amount.get() == 0 {
                return Err(GraphError::ZeroDimension {
                    layer: name.clone(),
                    what: "padding",
                    value: 0,
                });
            }
        }
        if layer.act_bits != self.quant.bits.acts {
            return Err(GraphError::InvalidBitWidth {
                what: "layer activation",
                bits: layer.act_bits.get(),
            });
        }
        if layer.weight_bits != self.quant.bits.weights {
            return Err(GraphError::InvalidBitWidth {
                what: "layer weight",
                bits: layer.weight_bits.get(),
            });
        }
        if layer.act_zero_point.get() != 0 {
            return Err(GraphError::NonZeroZeroPoint {
                layer: name,
                value: layer.act_zero_point.get(),
            });
        }
        if layer.op.has_relu() != layer.out_relu {
            return Err(GraphError::OpReluMismatch {
                layer: name,
                op: layer.op.as_str().to_owned(),
                out_relu: layer.out_relu,
            });
        }
        if layer.kind == LayerKind::DepthwiseConv2d {
            if groups != in_channels || out_channels != in_channels {
                return Err(GraphError::KernelShape {
                    layer: name,
                    detail: format!(
                        "depthwise convolution requires groups == in_channels == out_channels, \
                         found groups {groups}, in_channels {in_channels}, out_channels {out_channels}"
                    ),
                });
            }
        }
        let expected = layer.expected_weight_count();
        if layer.weight_count != expected {
            return Err(GraphError::WeightCountMismatch {
                layer: name,
                declared: layer.weight_count,
                expected,
            });
        }
        if layer.in_point.index() >= point_count {
            return Err(GraphError::PointOutOfRange {
                layer: name,
                what: "in_point",
                point: layer.in_point.get(),
                point_count,
            });
        }
        match layer.out_point {
            PointRef::Point(id) => {
                if id.index() >= point_count {
                    return Err(GraphError::PointOutOfRange {
                        layer: name,
                        what: "out_point",
                        point: id.get(),
                        point_count,
                    });
                }
            }
            PointRef::ImageSentinel => {
                if !layer.out_unsigned {
                    return Err(GraphError::HeadSentinelNotUnsigned { layer: name });
                }
            }
        }
        let channels = out_channels as usize;
        if layer.bias_int32.len() != channels {
            return Err(GraphError::VectorLengthMismatch {
                layer: name,
                what: "bias_int32",
                declared: layer.bias_int32.len(),
                expected: channels,
            });
        }
        if layer.requant.len() != channels {
            return Err(GraphError::VectorLengthMismatch {
                layer: name,
                what: "requant",
                declared: layer.requant.len(),
                expected: channels,
            });
        }
        if !layer.weight_scale.is_empty() && layer.weight_scale.len() != channels {
            return Err(GraphError::VectorLengthMismatch {
                layer: name,
                what: "weight_scale",
                declared: layer.weight_scale.len(),
                expected: channels,
            });
        }
        if !layer.bias.is_empty() && layer.bias.len() != channels {
            return Err(GraphError::VectorLengthMismatch {
                layer: name,
                what: "bias",
                declared: layer.bias.len(),
                expected: channels,
            });
        }
        for value in &layer.weight_scale {
            if !value.is_finite() || *value <= 0.0 {
                return Err(GraphError::NonFinite {
                    what: format!("{name} weight_scale"),
                    value: *value,
                });
            }
        }
        for value in &layer.bias {
            if !value.is_finite() {
                return Err(GraphError::NonFinite {
                    what: format!("{name} bias"),
                    value: *value,
                });
            }
        }
        for value in layer.requant.values() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(GraphError::NonFinite {
                    what: format!("{name} requant coefficient"),
                    value: *value,
                });
            }
        }
        let in_scale =
            self.point_scale(layer.in_point)
                .ok_or_else(|| GraphError::PointOutOfRange {
                    layer: name.clone(),
                    what: "in_point",
                    point: layer.in_point.get(),
                    point_count,
                })?;
        if !scale_eq(layer.in_scale.get(), in_scale) {
            return Err(GraphError::LayerScaleMismatch {
                layer: name,
                what: "in_scale",
                declared: layer.in_scale.get(),
                expected: in_scale,
            });
        }
        if let PointRef::Point(id) = layer.out_point {
            let out_scale = self
                .point_scale(id)
                .ok_or_else(|| GraphError::PointOutOfRange {
                    layer: name.clone(),
                    what: "out_point",
                    point: id.get(),
                    point_count,
                })?;
            if !scale_eq(layer.out_scale.get(), out_scale) {
                return Err(GraphError::LayerScaleMismatch {
                    layer: name,
                    what: "out_scale",
                    declared: layer.out_scale.get(),
                    expected: out_scale,
                });
            }
        }
        Ok(())
    }

    fn validate_weight_layout(&self) -> GResult<()> {
        let per_byte = self.quant.codes_per_byte();
        let mut expected_offset = 0usize;
        let mut expected_byte_offset = 0usize;
        for layer in &self.layers {
            let name = layer.name.to_string();
            if layer.weight_offset != expected_offset {
                return Err(GraphError::WeightLayoutMismatch {
                    layer: name,
                    what: "weight_offset",
                    expected: expected_offset,
                    declared: layer.weight_offset,
                });
            }
            if layer.weight_byte_offset != expected_byte_offset {
                return Err(GraphError::WeightLayoutMismatch {
                    layer: name,
                    what: "weight_byte_offset",
                    expected: expected_byte_offset,
                    declared: layer.weight_byte_offset,
                });
            }
            if layer.weight_count % per_byte != 0 {
                return Err(GraphError::WeightLayoutMismatch {
                    layer: name,
                    what: "weight_count packing",
                    expected: layer.weight_count.div_ceil(per_byte) * per_byte,
                    declared: layer.weight_count,
                });
            }
            let bytes = layer.weight_count / per_byte;
            if layer.weight_bytes != bytes {
                return Err(GraphError::WeightLayoutMismatch {
                    layer: name,
                    what: "weight_bytes",
                    expected: bytes,
                    declared: layer.weight_bytes,
                });
            }
            expected_offset = expected_offset.checked_add(layer.weight_count).ok_or(
                GraphError::NumericOverflow {
                    what: "weight code count",
                    value: expected_offset,
                },
            )?;
            expected_byte_offset = expected_byte_offset.checked_add(layer.weight_bytes).ok_or(
                GraphError::NumericOverflow {
                    what: "weight byte count",
                    value: expected_byte_offset,
                },
            )?;
        }
        if self.weight_stream.code_count != expected_offset {
            return Err(GraphError::WeightLayoutMismatch {
                layer: "graph".to_owned(),
                what: "weight_stream.code_count",
                expected: expected_offset,
                declared: self.weight_stream.code_count,
            });
        }
        if self.weight_stream.byte_len != expected_byte_offset {
            return Err(GraphError::WeightByteTotal {
                declared: self.weight_stream.byte_len,
                expected: expected_byte_offset,
            });
        }
        if self.weight_stream.packing != self.quant.packing {
            return Err(GraphError::UnsupportedPacking {
                packing: self.weight_stream.packing.as_str().to_owned(),
                weight_bits: self.quant.bits.weights.get(),
            });
        }
        Ok(())
    }

    fn validate_ops(&self, point_count: usize) -> GResult<()> {
        let mut available = vec![false; point_count];
        if point_count > 0 {
            available[0] = true;
        }
        let mut image_produced = false;
        let mut quantize_at: Option<usize> = None;

        for (index, op) in self.ops.iter().enumerate() {
            match op {
                GraphOp::Conv {
                    layer,
                    in_point,
                    out_point,
                } => {
                    let model = self.layer(layer).ok_or_else(|| GraphError::UnknownLayer {
                        op_index: index,
                        layer: layer.to_string(),
                    })?;
                    if model.in_point != *in_point {
                        return Err(GraphError::ConvPointMismatch {
                            op_index: index,
                            layer: layer.to_string(),
                            op_point: in_point.get(),
                            layer_in_point: model.in_point.get(),
                        });
                    }
                    if *out_point != model.out_point {
                        return Err(GraphError::OpOutPointMismatch {
                            op_index: index,
                            layer: layer.to_string(),
                            declared: describe_point_ref(*out_point),
                            expected: describe_point_ref(model.out_point),
                        });
                    }
                    match out_point {
                        PointRef::Point(id) => {
                            if id.index() >= point_count {
                                return Err(GraphError::PointOutOfRange {
                                    layer: layer.to_string(),
                                    what: "out_point",
                                    point: id.get(),
                                    point_count,
                                });
                            }
                            if id.get() == 0 {
                                return Err(GraphError::InputPointRewritten {
                                    op_index: index,
                                    layer: layer.to_string(),
                                });
                            }
                            available[id.index()] = true;
                        }
                        PointRef::ImageSentinel => image_produced = true,
                    }
                    // An unsigned head produces the image whether it writes a real
                    // point or the sentinel.
                    if model.out_unsigned {
                        image_produced = true;
                    }
                    if !available[in_point.index()] {
                        return Err(GraphError::UnwrittenPoint {
                            op_index: index,
                            kind: "conv",
                            point: in_point.get(),
                        });
                    }
                }
                GraphOp::Relu { layer } => {
                    let model = self.layer(layer).ok_or_else(|| GraphError::UnknownLayer {
                        op_index: index,
                        layer: layer.to_string(),
                    })?;
                    if !model.out_relu {
                        return Err(GraphError::ReluNotAfterConv {
                            op_index: index,
                            layer: layer.to_string(),
                        });
                    }
                    let produced = match model.out_point {
                        PointRef::Point(id) => available.get(id.index()).copied().unwrap_or(false),
                        PointRef::ImageSentinel => image_produced,
                    } || (model.out_unsigned && image_produced);
                    if !produced {
                        return Err(GraphError::UnwrittenPoint {
                            op_index: index,
                            kind: "relu",
                            point: model.in_point.get(),
                        });
                    }
                }
                GraphOp::Add {
                    point,
                    source_point,
                    scale,
                } => {
                    if point == source_point {
                        return Err(GraphError::AddPointIdentity {
                            op_index: index,
                            point: point.get(),
                        });
                    }
                    for (what, id) in [("point", point), ("source_point", source_point)] {
                        if id.index() >= point_count {
                            return Err(GraphError::PointOutOfRange {
                                layer: "graph".to_owned(),
                                what,
                                point: id.get(),
                                point_count,
                            });
                        }
                        if !available[id.index()] {
                            return Err(GraphError::UnwrittenPoint {
                                op_index: index,
                                kind: "add",
                                point: id.get(),
                            });
                        }
                    }
                    let expected =
                        self.point_scale(*point)
                            .ok_or_else(|| GraphError::PointOutOfRange {
                                layer: "graph".to_owned(),
                                what: "point",
                                point: point.get(),
                                point_count,
                            })?;
                    if !scale_eq(scale.get(), expected) {
                        return Err(GraphError::OpScaleMismatch {
                            op_index: index,
                            kind: "add",
                            declared: scale.get(),
                            expected,
                        });
                    }
                }
                GraphOp::UpsampleNearest2x { point } => {
                    if point.index() >= point_count {
                        return Err(GraphError::PointOutOfRange {
                            layer: "graph".to_owned(),
                            what: "point",
                            point: point.get(),
                            point_count,
                        });
                    }
                    if !available[point.index()] {
                        return Err(GraphError::UnwrittenPoint {
                            op_index: index,
                            kind: "upsample_nearest2x",
                            point: point.get(),
                        });
                    }
                }
                GraphOp::QuantizeU8Output { .. } => {
                    if quantize_at.is_some() {
                        return Err(GraphError::QuantizeNotLast { op_index: index });
                    }
                    if !image_produced {
                        return Err(GraphError::ImageNotProduced);
                    }
                    quantize_at = Some(index);
                }
            }
        }

        let quantize_at = quantize_at.ok_or(GraphError::MissingOutputOp)?;
        if quantize_at + 1 != self.ops.len() {
            return Err(GraphError::QuantizeNotLast {
                op_index: quantize_at + 1,
            });
        }
        if let Some(missing) = (0..point_count).find(|index| !available[*index]) {
            return Err(GraphError::DeadPoint {
                point: missing as u32,
            });
        }
        Ok(())
    }

    /// Static metrics; independent of hardware and of a concrete input resolution.
    pub fn metrics(&self) -> GraphMetrics {
        let mut ops = GraphOpCounts::default();
        for op in &self.ops {
            match op.kind() {
                // Convolutions are counted per layer and relu markers mirror the fused
                // activation of a layer, so neither is a separate op metric.
                GraphOpKind::Conv | GraphOpKind::Relu => {}
                GraphOpKind::Add => ops.add += 1,
                GraphOpKind::UpsampleNearest2x => ops.upsample += 1,
                GraphOpKind::QuantizeU8Output => {}
            }
        }
        let mut per_layer = Vec::with_capacity(self.layers.len());
        let mut macs_per_pixel = 0u64;
        let mut weight_codes = 0usize;
        for layer in &self.layers {
            let macs = layer.macs_per_pixel();
            macs_per_pixel = macs_per_pixel.saturating_add(macs);
            weight_codes = weight_codes.saturating_add(layer.weight_count);
            per_layer.push(LayerMacs {
                name: layer.name.clone(),
                macs_per_pixel: macs,
                weight_codes: layer.weight_count,
            });
        }
        let mut fanout = vec![0u32; self.points.len()];
        for op in &self.ops {
            match op {
                GraphOp::Conv { in_point, .. } => fanout[in_point.index()] += 1,
                GraphOp::Add {
                    point,
                    source_point,
                    ..
                } => {
                    fanout[point.index()] += 1;
                    fanout[source_point.index()] += 1;
                }
                GraphOp::UpsampleNearest2x { point } => fanout[point.index()] += 1,
                GraphOp::Relu { .. } | GraphOp::QuantizeU8Output { .. } => {}
            }
        }
        let point_fanout = fanout
            .into_iter()
            .enumerate()
            .map(|(index, consumers)| PointFanout {
                point: PointId::new(index as u32),
                consumers,
            })
            .collect();
        GraphMetrics {
            point_count: self.points.len(),
            graph_op_count: self.ops.len(),
            layer_count: self.layers.len(),
            conv_layer_count: self
                .layers
                .iter()
                .filter(|l| l.kind == LayerKind::Conv2d)
                .count(),
            depthwise_layer_count: self
                .layers
                .iter()
                .filter(|l| l.kind == LayerKind::DepthwiseConv2d)
                .count(),
            pointwise_layer_count: self
                .layers
                .iter()
                .filter(|l| l.kind == LayerKind::PointwiseConv2d)
                .count(),
            relu_layer_count: self.layers.iter().filter(|l| l.out_relu).count(),
            add_count: ops.add,
            upsample_count: ops.upsample,
            weight_codes,
            weight_bytes: self.weight_stream.byte_len,
            macs_per_output_pixel: macs_per_pixel,
            per_layer,
            point_fanout,
        }
    }

    /// Propagate a concrete input tensor through the graph.
    ///
    /// Returns the shape of every internal point, the output image shape and the total
    /// multiply-accumulate count. Convolution geometry follows valid convolution after
    /// explicit padding: `out = (in + 2*pad - k)/stride + 1`.
    ///
    /// `add` and `upsample_nearest2x` update their point in place, so `point_shapes[i]`
    /// is the *final* shape of point `i` after those updates; MACs are attributed to the
    /// shape each convolution actually consumed.
    pub fn infer_shapes(&self, input: &TensorShape) -> GResult<ShapeReport> {
        let point_count = self.points.len();
        let mut shapes: Vec<Option<TensorShape>> = vec![None; point_count];
        let mut image: Option<TensorShape> = None;
        let mut macs = 0u64;
        shapes[0] = Some(*input);

        // The shape of the input point is authoritative; the first consumer must agree.
        for op in &self.ops {
            if let GraphOp::Conv {
                layer,
                in_point,
                out_point,
            } = op
            {
                if in_point.index() == 0 {
                    let model = self.layer(layer).ok_or_else(|| GraphError::UnknownLayer {
                        op_index: 0,
                        layer: layer.to_string(),
                    })?;
                    if model.in_channels.get() != input.channels.get() {
                        return Err(GraphError::ShapeChannelMismatch {
                            point: 0,
                            expected: model.in_channels.get(),
                            found: input.channels.get(),
                        });
                    }
                    let _ = out_point;
                    break;
                }
            }
        }

        for (index, op) in self.ops.iter().enumerate() {
            match op {
                GraphOp::Conv {
                    layer,
                    in_point,
                    out_point,
                } => {
                    let model = self.layer(layer).ok_or_else(|| GraphError::UnknownLayer {
                        op_index: index,
                        layer: layer.to_string(),
                    })?;
                    let in_shape = shapes[in_point.index()].ok_or(GraphError::ShapeMissing {
                        point: in_point.get(),
                    })?;
                    if in_shape.channels != model.in_channels {
                        return Err(GraphError::ShapeChannelMismatch {
                            point: in_point.get(),
                            expected: in_shape.channels.get(),
                            found: model.in_channels.get(),
                        });
                    }
                    let pad = model.padding.amount().map(Extent::get).unwrap_or(0);
                    let out_h = conv_extent(
                        in_shape.height.get(),
                        pad,
                        model.kernel.h.get(),
                        model.stride.h.get(),
                    )
                    .ok_or_else(|| GraphError::ShapeUnderflow {
                        point: in_point.get(),
                        detail: format!(
                            "height {} with pad {pad}, kernel {} and stride {}",
                            in_shape.height.get(),
                            model.kernel.h.get(),
                            model.stride.h.get()
                        ),
                    })?;
                    let out_w = conv_extent(
                        in_shape.width.get(),
                        pad,
                        model.kernel.w.get(),
                        model.stride.w.get(),
                    )
                    .ok_or_else(|| GraphError::ShapeUnderflow {
                        point: in_point.get(),
                        detail: format!(
                            "width {} with pad {pad}, kernel {} and stride {}",
                            in_shape.width.get(),
                            model.kernel.w.get(),
                            model.stride.w.get()
                        ),
                    })?;
                    let out_shape = TensorShape {
                        channels: model.out_channels,
                        height: Extent(out_h),
                        width: Extent(out_w),
                    };
                    let layer_macs = model
                        .macs_per_pixel()
                        .checked_mul(out_h as u64)
                        .and_then(|m| m.checked_mul(out_w as u64))
                        .ok_or(GraphError::NumericOverflow {
                            what: "layer MAC count",
                            value: out_h as usize,
                        })?;
                    macs = macs
                        .checked_add(layer_macs)
                        .ok_or(GraphError::NumericOverflow {
                            what: "total MAC count",
                            value: macs as usize,
                        })?;
                    if let PointRef::Point(id) = out_point {
                        shapes[id.index()] = Some(out_shape);
                    }
                    if model.out_unsigned {
                        image = Some(out_shape);
                    }
                }
                GraphOp::Relu { .. } => {}
                GraphOp::Add {
                    point,
                    source_point,
                    ..
                } => {
                    let lhs = shapes[point.index()]
                        .ok_or(GraphError::ShapeMissing { point: point.get() })?;
                    let rhs = shapes[source_point.index()].ok_or(GraphError::ShapeMissing {
                        point: source_point.get(),
                    })?;
                    if lhs != rhs {
                        return Err(GraphError::ShapeMismatch {
                            point: point.get(),
                            source: source_point.get(),
                            detail: format!(
                                "{}x{}x{} vs {}x{}x{}",
                                lhs.channels.get(),
                                lhs.height.get(),
                                lhs.width.get(),
                                rhs.channels.get(),
                                rhs.height.get(),
                                rhs.width.get()
                            ),
                        });
                    }
                    shapes[point.index()] = Some(lhs);
                }
                GraphOp::UpsampleNearest2x { point } => {
                    let shape = shapes[point.index()]
                        .ok_or(GraphError::ShapeMissing { point: point.get() })?;
                    let height =
                        shape
                            .height
                            .get()
                            .checked_mul(2)
                            .ok_or(GraphError::NumericOverflow {
                                what: "upsample height",
                                value: shape.height.get() as usize,
                            })?;
                    let width =
                        shape
                            .width
                            .get()
                            .checked_mul(2)
                            .ok_or(GraphError::NumericOverflow {
                                what: "upsample width",
                                value: shape.width.get() as usize,
                            })?;
                    shapes[point.index()] = Some(TensorShape {
                        channels: shape.channels,
                        height: Extent(height),
                        width: Extent(width),
                    });
                }
                GraphOp::QuantizeU8Output { .. } => {}
            }
        }

        let mut point_shapes = Vec::with_capacity(point_count);
        for (index, shape) in shapes.into_iter().enumerate() {
            point_shapes.push(shape.ok_or(GraphError::ShapeMissing {
                point: index as u32,
            })?);
        }
        let image = image.ok_or(GraphError::ImageNotProduced)?;
        Ok(ShapeReport {
            point_shapes,
            image,
            macs,
        })
    }
}

#[derive(Default)]
struct GraphOpCounts {
    add: usize,
    upsample: usize,
}

/// A weight-blob reference must stay inside the directory it is resolved against:
/// a plain relative name with no root, prefix or `..` component.
pub(crate) fn check_weight_file(file: &str) -> GResult<()> {
    let mut has_name = false;
    for component in std::path::Path::new(file).components() {
        match component {
            std::path::Component::Normal(_) => has_name = true,
            std::path::Component::CurDir => {}
            _ => {
                return Err(GraphError::UnsafeWeightPath {
                    file: file.to_owned(),
                });
            }
        }
    }
    if !has_name {
        return Err(GraphError::UnsafeWeightPath {
            file: file.to_owned(),
        });
    }
    Ok(())
}

fn describe_point_ref(point: PointRef) -> String {
    match point {
        PointRef::Point(id) => format!("point {id}"),
        PointRef::ImageSentinel => "the image sentinel".to_owned(),
    }
}

fn conv_extent(input: u32, pad: u32, kernel: u32, stride: u32) -> Option<u32> {
    let padded = input.checked_add(pad.checked_mul(2)?)?;
    if padded < kernel {
        return None;
    }
    Some((padded - kernel) / stride + 1)
}

// ---------------------------------------------------------------------------
// Persisted artifact
// ---------------------------------------------------------------------------

/// Provenance of a graph artifact: what it was imported from and how to verify it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Source artifact format, e.g. `fpga_st_train.int8.v1`.
    pub source_format: String,
    /// Importer identity, e.g. `tessera-frontend/0.1.0`.
    pub importer_version: String,
    /// SHA-256 of the source manifest bytes.
    pub manifest_sha256: String,
    /// SHA-256 of the packed weight blob.
    pub weights_sha256: String,
    /// Weight file name as declared by the manifest.
    pub weight_file: String,
    /// Weight blob size in bytes.
    pub weight_bytes: usize,
}

/// The persisted, versioned graph artifact (`graph.tgraph`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphArtifact {
    pub format: String,
    pub schema_version: u32,
    pub provenance: Provenance,
    pub graph: Graph,
    pub metrics: GraphMetrics,
}

impl GraphArtifact {
    /// Wrap a validated graph together with its provenance and recomputed metrics.
    pub fn new(graph: Graph, provenance: Provenance) -> GResult<Self> {
        graph.validate()?;
        let metrics = graph.metrics();
        Ok(GraphArtifact {
            format: GRAPH_FORMAT.to_owned(),
            schema_version: GRAPH_SCHEMA_VERSION,
            provenance,
            graph,
            metrics,
        })
    }

    /// Check format, schema version, graph invariants and metric freshness.
    pub fn validate(&self) -> GResult<()> {
        if self.format != GRAPH_FORMAT {
            return Err(GraphError::ArtifactFormat {
                declared: self.format.clone(),
            });
        }
        if self.schema_version != GRAPH_SCHEMA_VERSION {
            return Err(GraphError::ArtifactSchemaVersion {
                declared: self.schema_version,
            });
        }
        // Consumers stage and read the weight blob relative to the artifact directory, so
        // a tampered provenance must not be able to point outside it.
        check_weight_file(&self.provenance.weight_file)?;
        self.graph.validate()?;
        if self.metrics != self.graph.metrics() {
            return Err(GraphError::MetricsStale);
        }
        Ok(())
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
    pub fn metrics(&self) -> &GraphMetrics {
        &self.metrics
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ch(value: u32) -> Channels {
        Channels(value)
    }

    fn ext(value: u32) -> Extent {
        Extent(value)
    }

    fn sc(value: f32) -> Scale {
        Scale::try_new("test scale", value).unwrap()
    }

    fn contract(acts: u8, weights: u8, packing: Packing) -> QuantContract {
        QuantContract {
            bits: Bits {
                acts: BitWidth(acts),
                weights: BitWidth(weights),
            },
            act_qmax: QMax((1u32 << (acts - 1)) as f32 - 1.0),
            output_qmax: QMax((1u32 << acts) as f32 - 1.0),
            weight_qmax: QMax((1u32 << (weights - 1)) as f32 - 1.0),
            act_zero_point: ZeroPoint::symmetric(),
            accumulator: Dtype::Int32,
            requant_rounding: Rounding::HalfAwayFromZero,
            packing,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_layer(
        id: u32,
        name: &str,
        op: LayerOp,
        kind: LayerKind,
        in_channels: u32,
        out_channels: u32,
        groups: u32,
        kernel: (u32, u32),
        stride: (u32, u32),
        padding: Padding,
        in_point: u32,
        out_point: PointRef,
    ) -> Layer {
        let kernel = Kernel::try_new(kernel.0, kernel.1).unwrap();
        let stride = Stride::try_new(stride.0, stride.1).unwrap();
        let in_per_group = in_channels / groups;
        let weight_count = (out_channels * in_per_group * kernel.h.get() * kernel.w.get()) as usize;
        Layer {
            id: LayerId(id),
            name: LayerName::new(name),
            op,
            kind,
            in_channels: ch(in_channels),
            out_channels: ch(out_channels),
            groups: GroupCount(groups),
            act_bits: BitWidth(8),
            weight_bits: BitWidth(8),
            act_zero_point: ZeroPoint::symmetric(),
            out_relu: op.has_relu(),
            out_unsigned: out_point.is_image(),
            kernel,
            stride,
            padding,
            in_point: PointId(in_point),
            out_point,
            in_scale: sc(1.0),
            out_scale: sc(1.0),
            weight_scale: vec![1.0; out_channels as usize],
            bias: vec![0.0; out_channels as usize],
            bias_int32: vec![0; out_channels as usize],
            requant: RequantCoefficients::try_new(vec![0.5; out_channels as usize]).unwrap(),
            weight_offset: 0,
            weight_count,
            weight_byte_offset: 0,
            weight_bytes: weight_count,
        }
    }

    /// A small but structurally complete graph:
    /// image(0) -> conv+relu -> point(1) -> upsample in place -> head -> image.
    fn example_graph() -> Graph {
        let points = vec![
            Point {
                id: PointId(0),
                name: PointName::new("image"),
                scale: sc(0.01),
            },
            Point {
                id: PointId(1),
                name: PointName::new("enc.0.in"),
                scale: sc(0.02),
            },
        ];
        let mut first = conv_layer(
            0,
            "enc.0",
            LayerOp::Conv2dRelu,
            LayerKind::Conv2d,
            3,
            8,
            1,
            (3, 3),
            (1, 1),
            Padding::reflect(1).unwrap(),
            0,
            PointRef::Point(PointId(1)),
        );
        first.in_scale = sc(0.01);
        first.out_scale = sc(0.02);
        let mut head = conv_layer(
            1,
            "head",
            LayerOp::Conv2d,
            LayerKind::PointwiseConv2d,
            8,
            3,
            1,
            (1, 1),
            (1, 1),
            Padding::none(),
            1,
            PointRef::ImageSentinel,
        );
        head.in_scale = sc(0.02);
        head.out_scale = sc(1.0 / 255.0);
        let first_codes = first.weight_count;
        head.weight_offset = first_codes;
        head.weight_byte_offset = first_codes;
        let code_count = first_codes + head.weight_count;
        let ops = vec![
            GraphOp::Conv {
                layer: LayerName::new("enc.0"),
                in_point: PointId(0),
                out_point: PointRef::Point(PointId(1)),
            },
            GraphOp::Relu {
                layer: LayerName::new("enc.0"),
            },
            GraphOp::UpsampleNearest2x { point: PointId(1) },
            GraphOp::Conv {
                layer: LayerName::new("head"),
                in_point: PointId(1),
                out_point: PointRef::ImageSentinel,
            },
            GraphOp::QuantizeU8Output {
                scale: sc(1.0 / 255.0),
            },
        ];
        Graph {
            source_format: MANIFEST_FORMAT.to_owned(),
            arch: "test".to_owned(),
            quant: contract(8, 8, Packing::I8x1),
            points,
            ops,
            layers: vec![first, head],
            weight_stream: WeightStream {
                file: "weights.bin".to_owned(),
                byte_len: code_count,
                code_count,
                packing: Packing::I8x1,
            },
            notes: Vec::new(),
        }
    }

    fn provenance(graph: &Graph) -> Provenance {
        Provenance {
            source_format: MANIFEST_FORMAT.to_owned(),
            importer_version: "tessera-frontend/0.1.0".to_owned(),
            manifest_sha256: "0".repeat(64),
            weights_sha256: "1".repeat(64),
            weight_file: graph.weight_stream.file.clone(),
            weight_bytes: graph.weight_stream.byte_len,
        }
    }

    #[test]
    fn example_graph_is_valid() {
        example_graph().validate().unwrap();
    }

    #[test]
    fn layer_shapes_and_group_rules_hold() {
        let graph = example_graph();
        assert_eq!(graph.layers()[0].expected_weight_count(), 3 * 8 * 9);
        assert_eq!(graph.layers()[0].weight_count, 3 * 8 * 9);
        assert_eq!(graph.layers()[1].macs_per_pixel(), 3 * 8);
        assert_eq!(graph.point_count(), 2);
        assert_eq!(graph.head_sentinel(), PointId::new(2));
        assert_eq!(graph.input_point(), PointId::new(0));
        assert_eq!(graph.head_layers().count(), 1);
    }

    #[test]
    fn metrics_count_ops_layers_and_fanout() {
        let metrics = example_graph().metrics();
        assert_eq!(metrics.point_count, 2);
        assert_eq!(metrics.graph_op_count, 5);
        assert_eq!(metrics.layer_count, 2);
        assert_eq!(metrics.conv_layer_count, 1);
        assert_eq!(metrics.pointwise_layer_count, 1);
        assert_eq!(metrics.relu_layer_count, 1);
        assert_eq!(metrics.upsample_count, 1);
        assert_eq!(metrics.add_count, 0);
        assert_eq!(metrics.macs_per_output_pixel, 3 * 8 * 9 + 3 * 8);
        assert_eq!(metrics.weight_codes, 3 * 8 * 9 + 3 * 8);
        assert_eq!(metrics.weight_bytes, metrics.weight_codes);
        // Point 0 feeds the first convolution only.
        assert_eq!(metrics.point_fanout[0].consumers, 1);
        // Point 1 feeds the upsample and the head convolution.
        assert_eq!(metrics.point_fanout[1].consumers, 2);
        assert_eq!(metrics.per_layer[0].name.as_str(), "enc.0");
        assert_eq!(metrics.per_layer[1].macs_per_pixel, 24);
    }

    #[test]
    fn shape_inference_propagates_padding_stride_and_upsample() {
        let report = example_graph()
            .infer_shapes(&TensorShape::try_new(3, 6, 10).unwrap())
            .unwrap();
        // 3x3 kernel, reflect pad 1 and stride 1 keep the spatial extent; the upsample
        // then doubles point 1 in place, so the reported shape is the final one.
        assert_eq!(report.point_shapes[1].channels.get(), 8);
        assert_eq!(report.point_shapes[1].height.get(), 12);
        assert_eq!(report.point_shapes[1].width.get(), 20);
        assert_eq!(report.image.channels.get(), 3);
        assert_eq!(report.image.height.get(), 12);
        assert_eq!(report.image.width.get(), 20);
        assert_eq!(report.macs, (3 * 8 * 9) * 6 * 10 + (3 * 8) * 12 * 20);
    }

    #[test]
    fn shape_inference_rejects_a_wrong_input_channel_count() {
        let graph = example_graph();
        assert!(matches!(
            graph.infer_shapes(&TensorShape::try_new(4, 6, 10).unwrap()),
            Err(GraphError::ShapeChannelMismatch {
                point: 0,
                expected: 3,
                found: 4
            })
        ));
    }

    #[test]
    fn missing_producer_is_rejected() {
        let mut graph = example_graph();
        // Drop the head convolution: the image is never produced.
        graph.ops.remove(3);
        assert_eq!(graph.validate(), Err(GraphError::ImageNotProduced));
    }

    #[test]
    fn dead_point_is_rejected() {
        let mut graph = example_graph();
        graph.points.push(Point {
            id: PointId(2),
            name: PointName::new("dead"),
            scale: sc(0.04),
        });
        assert_eq!(graph.validate(), Err(GraphError::DeadPoint { point: 2 }));
    }

    #[test]
    fn an_unsigned_head_may_write_a_real_point_instead_of_the_sentinel() {
        let mut graph = example_graph();
        // The streaming manifests publish the image on a real point rather than on the
        // sentinel; both must count as producing the image.
        graph.layers[1].out_point = PointRef::Point(PointId(1));
        graph.layers[1].out_scale = sc(0.02);
        graph.ops[3] = GraphOp::Conv {
            layer: LayerName::new("head"),
            in_point: PointId(1),
            out_point: PointRef::Point(PointId(1)),
        };
        graph.validate().unwrap();
        let report = graph
            .infer_shapes(&TensorShape::try_new(3, 6, 10).unwrap())
            .unwrap();
        assert_eq!(report.image.channels.get(), 3);
        assert_eq!(report.image.height.get(), 12);
        assert_eq!(report.image.width.get(), 20);
        assert_eq!(graph.head_layers().count(), 1);
    }

    #[test]
    fn op_out_point_must_match_the_layer() {
        let mut graph = example_graph();
        graph.ops[3] = GraphOp::Conv {
            layer: LayerName::new("head"),
            in_point: PointId(1),
            out_point: PointRef::Point(PointId(1)),
        };
        assert!(matches!(
            graph.validate(),
            Err(GraphError::OpOutPointMismatch { op_index: 3, .. })
        ));
    }

    #[test]
    fn head_sentinel_requires_unsigned_output() {
        let mut graph = example_graph();
        graph.layers[1].out_unsigned = false;
        assert!(matches!(
            graph.validate(),
            Err(GraphError::HeadSentinelNotUnsigned { .. })
        ));
    }

    #[test]
    fn point_reference_beyond_range_is_rejected() {
        let mut graph = example_graph();
        graph.layers[0].in_point = PointId(9);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::PointOutOfRange {
                what: "in_point",
                point: 9,
                ..
            })
        ));
    }

    #[test]
    fn group_rule_violation_is_rejected() {
        let mut graph = example_graph();
        graph.layers[0].groups = GroupCount(4);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::BadGroups { .. })
        ));
    }

    #[test]
    fn depthwise_rules_are_checked() {
        let mut graph = example_graph();
        graph.layers[0].kind = LayerKind::DepthwiseConv2d;
        graph.layers[0].groups = GroupCount(4);
        graph.layers[0].in_channels = ch(4);
        graph.layers[0].out_channels = ch(4);
        graph.layers[0].weight_count = 4 * 9 * 2;
        assert!(matches!(
            graph.validate(),
            Err(GraphError::WeightCountMismatch { .. })
        ));

        let mut graph = example_graph();
        graph.layers[0].kind = LayerKind::DepthwiseConv2d;
        graph.layers[0].groups = GroupCount(4);
        graph.layers[0].in_channels = ch(4);
        graph.layers[0].out_channels = ch(8);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::KernelShape { .. })
        ));
    }

    #[test]
    fn packing_must_match_weight_bits() {
        let mut graph = example_graph();
        graph.quant.packing = Packing::I4x2;
        graph.weight_stream.packing = Packing::I4x2;
        assert!(matches!(
            graph.validate(),
            Err(GraphError::UnsupportedPacking { .. })
        ));
    }

    #[test]
    fn quantize_must_be_last_and_unique() {
        let mut graph = example_graph();
        graph.ops.push(GraphOp::QuantizeU8Output { scale: sc(0.5) });
        assert_eq!(
            graph.validate(),
            Err(GraphError::QuantizeNotLast { op_index: 5 })
        );
    }

    #[test]
    fn weight_layout_gap_is_rejected() {
        let mut graph = example_graph();
        graph.layers[1].weight_offset += 1;
        assert!(matches!(
            graph.validate(),
            Err(GraphError::WeightLayoutMismatch {
                what: "weight_offset",
                ..
            })
        ));
    }

    #[test]
    fn weight_stream_must_cover_every_layer() {
        let mut graph = example_graph();
        graph.weight_stream.byte_len -= 1;
        assert!(matches!(
            graph.validate(),
            Err(GraphError::WeightByteTotal { .. })
        ));
    }

    #[test]
    fn add_cannot_use_one_point_twice() {
        let mut graph = example_graph();
        graph.ops[1] = GraphOp::Add {
            point: PointId(1),
            source_point: PointId(1),
            scale: sc(0.02),
        };
        assert_eq!(
            graph.validate(),
            Err(GraphError::AddPointIdentity {
                op_index: 1,
                point: 1
            })
        );
    }

    #[test]
    fn add_scale_must_match_the_point_scale() {
        let mut graph = example_graph();
        graph.ops[1] = GraphOp::Add {
            point: PointId(1),
            source_point: PointId(0),
            scale: sc(0.5),
        };
        assert!(matches!(
            graph.validate(),
            Err(GraphError::OpScaleMismatch { kind: "add", .. })
        ));
    }

    #[test]
    fn missing_graph_op_field_is_reported_by_the_frontend() {
        // The frontend owns manifest field presence; this checks the error surface used
        // for it stays descriptive.
        let error = GraphError::OpFieldMissing {
            op_index: 7,
            kind: "conv",
            field: "layer",
        };
        assert_eq!(
            error.to_string(),
            "graph op 7 (conv): required field \"layer\" is missing"
        );
    }

    #[test]
    fn artifact_round_trips_and_detects_stale_metrics() {
        let graph = example_graph();
        let artifact = GraphArtifact::new(graph, provenance(&example_graph())).unwrap();
        artifact.validate().unwrap();
        let json = serde_json::to_string_pretty(&artifact).unwrap();
        let parsed: GraphArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, artifact);
        parsed.validate().unwrap();
        // Serialization is deterministic.
        assert_eq!(serde_json::to_string_pretty(&parsed).unwrap(), json);

        let mut stale = parsed;
        stale.metrics.layer_count += 1;
        assert_eq!(stale.validate(), Err(GraphError::MetricsStale));
    }

    #[test]
    fn artifact_rejects_foreign_schema() {
        let graph = example_graph();
        let mut artifact = GraphArtifact::new(graph, provenance(&example_graph())).unwrap();
        artifact.schema_version = 99;
        assert_eq!(
            artifact.validate(),
            Err(GraphError::ArtifactSchemaVersion { declared: 99 })
        );
        let mut artifact =
            GraphArtifact::new(example_graph(), provenance(&example_graph())).unwrap();
        artifact.format = "tessera.other".to_owned();
        assert!(matches!(
            artifact.validate(),
            Err(GraphError::ArtifactFormat { .. })
        ));
    }

    #[test]
    fn fixed_point_requant_bounds_multiplier_magnitude() {
        let coefficients =
            RequantCoefficients::try_new(vec![0.001_474_280_5, 0.5, 2.733_861_2]).unwrap();
        let fixed = coefficients.fixed_point((1 << 17) - 1, 31);
        assert_eq!(fixed.shift, 15);
        assert!(fixed.multipliers.iter().all(|m| m.abs() <= (1 << 17) - 1));
        assert_eq!(fixed.rounding, Rounding::HalfAwayFromZero);
        assert!(coefficients.relative_error(fixed.shift) < 0.01);
        assert_eq!(
            coefficients.max_multiplier(fixed.shift),
            fixed.multipliers.iter().map(|m| m.abs()).max().unwrap()
        );
    }

    #[test]
    fn requant_coefficients_must_be_positive() {
        assert!(RequantCoefficients::try_new(vec![0.5, 0.0]).is_err());
        assert!(RequantCoefficients::try_new(vec![0.5, f32::NAN]).is_err());
    }

    #[test]
    fn bit_width_qmax_helpers_agree() {
        assert_eq!(BitWidth(8).signed_qmax(), 127);
        assert_eq!(BitWidth(8).unsigned_qmax(), 255);
        assert_eq!(BitWidth(4).signed_qmax(), 7);
        assert_eq!(BitWidth::from_signed_qmax(127.0), Some(BitWidth(8)));
        assert_eq!(BitWidth::from_signed_qmax(7.0), Some(BitWidth(4)));
        assert_eq!(BitWidth::from_signed_qmax(100.0), None);
        assert!(BitWidth::try_new("acts", 1).is_err());
        assert!(BitWidth::try_new("acts", 17).is_err());
    }

    #[test]
    fn packing_properties() {
        assert_eq!(Packing::parse("i8-x1"), Some(Packing::I8x1));
        assert_eq!(Packing::parse("i4-x2"), Some(Packing::I4x2));
        assert_eq!(Packing::parse("i2-x4"), None);
        assert_eq!(Packing::I4x2.codes_per_byte(), 2);
        assert_eq!(Packing::I4x2.weight_bits(), 4);
        assert_eq!(Packing::I8x1.as_str(), "i8-x1");
    }

    #[test]
    fn padding_mode_is_preserved() {
        assert_eq!(Padding::none().amount(), None);
        assert!(Padding::reflect(2).unwrap().is_reflect());
        assert_eq!(Padding::reflect(2).unwrap().amount(), Some(ext(2)));
        assert!(!Padding::zero(1).unwrap().is_reflect());
        assert!(Padding::reflect(0).is_err());
    }

    /// The example artifact as a JSON tree, for surgical tampering.
    fn artifact_value() -> serde_json::Value {
        let artifact = GraphArtifact::new(example_graph(), provenance(&example_graph())).unwrap();
        serde_json::to_value(&artifact).unwrap()
    }

    fn reparse(value: &serde_json::Value) -> Result<GraphArtifact, serde_json::Error> {
        serde_json::from_value(value.clone())
    }

    #[test]
    fn artifact_deserialization_rejects_an_out_of_range_bit_width() {
        // A zero width would underflow `signed_qmax`; an oversized one is not supported.
        for bits in [0, 1, 17, 255] {
            let mut value = artifact_value();
            value["graph"]["quant"]["bits"]["acts"] = json!(bits);
            let error = reparse(&value).unwrap_err();
            assert!(
                error.to_string().contains(&format!("bit width {bits}")),
                "unexpected error for {bits} bits: {error}"
            );
        }
        let mut value = artifact_value();
        value["graph"]["quant"]["bits"]["weights"] = json!(32);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("bit width 32")
        );
    }

    #[test]
    fn artifact_deserialization_rejects_zero_geometry_and_padding() {
        for (block, field, needle) in [
            ("stride", "h", "stride height must be non-zero"),
            ("stride", "w", "stride width must be non-zero"),
            ("kernel", "h", "kernel height must be non-zero"),
            ("kernel", "w", "kernel width must be non-zero"),
        ] {
            let mut value = artifact_value();
            value["graph"]["layers"][0][block][field] = json!(0);
            let error = reparse(&value).unwrap_err();
            assert!(
                error.to_string().contains(needle),
                "tampering {block}.{field}: {error}"
            );
        }

        let mut value = artifact_value();
        assert_eq!(value["graph"]["layers"][0]["padding"]["Reflect"], json!(1));
        value["graph"]["layers"][0]["padding"] = json!({ "Reflect": 0 });
        let error = reparse(&value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("reflect padding must be non-zero"),
            "unexpected error: {error}"
        );

        let mut value = artifact_value();
        value["graph"]["layers"][0]["padding"] = json!({ "Zero": 0 });
        let error = reparse(&value).unwrap_err();
        assert!(
            error.to_string().contains("zero padding must be non-zero"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn artifact_deserialization_rejects_out_of_range_scalars_and_zero_point() {
        let mut value = artifact_value();
        value["graph"]["points"][0]["scale"] = json!(0.0);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("declared scale is not a finite number"),
            "zero point scale"
        );

        let mut value = artifact_value();
        value["graph"]["layers"][0]["in_channels"] = json!(0);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("channels must be non-zero")
        );

        let mut value = artifact_value();
        value["graph"]["layers"][0]["groups"] = json!(0);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("groups 0 must be >= 1")
        );

        let mut value = artifact_value();
        value["graph"]["quant"]["act_zero_point"] = json!(3);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("act_zero_point 3")
        );

        let mut value = artifact_value();
        value["graph"]["quant"]["act_qmax"] = json!(-1.0);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("declared qmax is not a finite number")
        );

        let mut value = artifact_value();
        value["graph"]["layers"][0]["requant"][0] = json!(0.0);
        assert!(
            reparse(&value)
                .unwrap_err()
                .to_string()
                .contains("requant coefficient")
        );
    }

    #[test]
    fn artifact_deserialization_rejects_an_escaping_weight_file() {
        for declared in [
            "../weights.bin",
            "/etc/passwd",
            "blobs/../../weights.bin",
            "",
        ] {
            // Provenance is checked on load, before a consumer stages the blob.
            let mut value = artifact_value();
            value["provenance"]["weight_file"] = json!(declared);
            let artifact: GraphArtifact = serde_json::from_value(value).unwrap();
            assert_eq!(
                artifact.validate(),
                Err(GraphError::UnsafeWeightPath {
                    file: declared.to_owned()
                })
            );

            // The graph's own reference is checked by `Graph::validate`.
            let mut value = artifact_value();
            value["graph"]["weight_stream"]["file"] = json!(declared);
            let artifact: GraphArtifact = serde_json::from_value(value).unwrap();
            assert_eq!(
                artifact.validate(),
                Err(GraphError::UnsafeWeightPath {
                    file: declared.to_owned()
                })
            );
        }
    }

    #[test]
    fn validate_rejects_zero_geometry_extents() {
        let mut graph = example_graph();
        graph.layers[0].stride.h = Extent(0);
        assert_eq!(
            graph.validate(),
            Err(GraphError::ZeroDimension {
                layer: "enc.0".to_owned(),
                what: "stride height",
                value: 0,
            })
        );

        let mut graph = example_graph();
        graph.layers[0].kernel.w = Extent(0);
        assert_eq!(
            graph.validate(),
            Err(GraphError::ZeroDimension {
                layer: "enc.0".to_owned(),
                what: "kernel width",
                value: 0,
            })
        );

        let mut graph = example_graph();
        graph.layers[0].padding = Padding::Reflect(Extent(0));
        assert_eq!(
            graph.validate(),
            Err(GraphError::ZeroDimension {
                layer: "enc.0".to_owned(),
                what: "padding",
                value: 0,
            })
        );
    }

    #[test]
    fn validate_rejects_an_out_of_range_bit_width_before_qmax_arithmetic() {
        // `QuantContract::check` would compute `signed_qmax` from these widths.
        let mut graph = example_graph();
        graph.quant.bits.acts = BitWidth(0);
        assert_eq!(
            graph.validate(),
            Err(GraphError::InvalidBitWidth {
                what: "acts",
                bits: 0
            })
        );

        let mut graph = example_graph();
        graph.quant.bits.weights = BitWidth(17);
        assert_eq!(
            graph.validate(),
            Err(GraphError::InvalidBitWidth {
                what: "weights",
                bits: 17
            })
        );

        // A layer cannot smuggle in a width the contract does not declare.
        let mut graph = example_graph();
        graph.layers[0].act_bits = BitWidth(0);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::InvalidBitWidth { .. })
        ));
    }

    #[test]
    fn validate_rejects_a_non_positive_requant_coefficient() {
        let mut graph = example_graph();
        graph.layers[0].requant = RequantCoefficients(vec![0.5; 8]);
        graph.layers[0].requant.0[3] = 0.0;
        assert_eq!(
            graph.validate(),
            Err(GraphError::NonFinite {
                what: "enc.0 requant coefficient".to_owned(),
                value: 0.0,
            })
        );
    }

    #[test]
    fn relu_layer_count_counts_layers_not_relu_ops() {
        let graph = example_graph();
        assert_eq!(graph.metrics().relu_layer_count, 1);

        // The relu markers are redundant: a streaming manifest may omit them entirely.
        let mut graph = example_graph();
        graph.ops.retain(|op| !matches!(op, GraphOp::Relu { .. }));
        graph.validate().unwrap();
        let metrics = graph.metrics();
        assert_eq!(metrics.relu_layer_count, 1);
        assert_eq!(metrics.graph_op_count, 4);

        // A second fused activation counts even without any marker of its own.
        graph.layers[1].op = LayerOp::Conv2dRelu;
        graph.layers[1].out_relu = true;
        graph.validate().unwrap();
        assert_eq!(graph.metrics().relu_layer_count, 2);
    }

    #[test]
    fn round_trip_preserves_layer_metrics_without_relu_ops() {
        let mut graph = example_graph();
        graph.ops.retain(|op| !matches!(op, GraphOp::Relu { .. }));
        let artifact = GraphArtifact::new(graph, provenance(&example_graph())).unwrap();
        assert_eq!(artifact.metrics().relu_layer_count, 1);
        let json = serde_json::to_string_pretty(&artifact).unwrap();
        let parsed: GraphArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, artifact);
        parsed.validate().unwrap();
    }
}
