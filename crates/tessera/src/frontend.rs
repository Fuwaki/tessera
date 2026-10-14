//! Manifest frontend: `fpga_st_train.int8.v1` → [`Graph`] → persisted [`GraphArtifact`].
//!
//! The frontend is the only place that knows the manifest's JSON vocabulary. It maps
//! every field into the typed IR in [`crate::graph`], validates the result, hashes the
//! inputs for provenance and never silently falls back on an unknown value: an
//! unsupported operator, packing or inconsistent length is an error.
//!
//! Deliberately *not* implemented here: any hardware decision. Buffers, folding,
//! placement and schedules belong to the planner.

use crate::graph::{
    BitWidth, Bits, Channels, Dtype, Extent, Graph, GraphArtifact, GraphError, GraphOp, GroupCount,
    Kernel, Layer, LayerId, LayerKind, LayerName, LayerOp, MANIFEST_FORMAT, Packing, Padding,
    Point, PointId, PointName, PointRef, Provenance, QMax, QuantContract, RequantCoefficients,
    Rounding, Scale, Stride, WeightStream, ZeroPoint, scale_eq,
};
use serde::Deserialize;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Importer identity recorded in artifact provenance.
pub const IMPORTER_VERSION: &str = concat!("tessera-frontend/", env!("CARGO_PKG_VERSION"));
/// Manifest file name read by [`import_manifest`].
pub const MANIFEST_FILE: &str = "manifest.json";

/// Result alias for frontend entry points.
///
/// The frontend reuses [`GraphError`] so that import, validation and load failures all
/// carry one diagnostic vocabulary (stage, identity, violated constraint).
pub type FrontendResult<T> = Result<T, GraphError>;

// ---------------------------------------------------------------------------
// Raw manifest
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawBits {
    acts: u8,
    weights: u8,
}

#[derive(Debug, Deserialize)]
struct RawOp {
    op: String,
    #[serde(default)]
    layer: Option<String>,
    #[serde(default)]
    point: Option<usize>,
    #[serde(default)]
    source_point: Option<usize>,
    #[serde(default)]
    scale: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawLayer {
    name: String,
    op: String,
    in_channels: usize,
    out_channels: usize,
    #[serde(default = "default_groups")]
    groups: usize,
    #[serde(default)]
    act_bits: Option<u8>,
    #[serde(default)]
    weight_bits: Option<u8>,
    #[serde(default)]
    act_zero_point: Option<i32>,
    out_relu: bool,
    out_unsigned: bool,
    kernel: [usize; 2],
    stride: [usize; 2],
    #[serde(default)]
    reflect_pad: usize,
    in_point: usize,
    out_point: usize,
    #[serde(default)]
    in_scale: Option<f32>,
    #[serde(default)]
    out_scale: Option<f32>,
    #[serde(default)]
    weight_scale: Option<Vec<f32>>,
    #[serde(default)]
    bias: Option<Vec<f32>>,
    bias_int32: Vec<i32>,
    requant: Vec<f32>,
    weight_offset: usize,
    weight_count: usize,
    #[serde(default)]
    weight_byte_offset: Option<usize>,
    #[serde(default)]
    weight_bytes: Option<usize>,
}

fn default_groups() -> usize {
    1
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    format: String,
    arch: String,
    #[serde(default)]
    bits: Option<RawBits>,
    act_qmax: f32,
    output_qmax: f32,
    weight_qmax: f32,
    #[serde(default)]
    notes: Vec<String>,
    #[serde(default)]
    points: Vec<String>,
    point_scales: Vec<f32>,
    graph: Vec<RawOp>,
    layers: Vec<RawLayer>,
    weight_file: String,
    weight_packing: String,
    weight_bytes: usize,
    #[serde(default)]
    lut_layers: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Import a manifest together with the weight blob it names.
///
/// `path` may be the `manifest.json` file itself or the directory that holds it. The
/// weight file is resolved relative to that directory.
pub fn import_manifest(path: &Path) -> FrontendResult<GraphArtifact> {
    let metadata = std::fs::metadata(path).map_err(|error| GraphError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let (manifest_path, weights_dir) = if metadata.is_dir() {
        (path.join(MANIFEST_FILE), path.to_path_buf())
    } else {
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        (path.to_path_buf(), dir)
    };
    let origin = manifest_path.display().to_string();
    let text = read_text(&manifest_path)?;
    let raw: RawManifest = parse_manifest(&text, &origin)?;
    let weights = read_bytes(&weights_dir.join(&raw.weight_file))?;
    artifact_from_raw(&raw, &text, &weights, &origin)
}

/// Import a manifest that is already in memory.
///
/// `origin` is used only for diagnostics. `weights` must be the `weights.bin` bytes;
/// its length is checked against the declared `weight_bytes`.
pub fn import_manifest_str(
    manifest: &str,
    weights: &[u8],
    origin: &str,
) -> FrontendResult<GraphArtifact> {
    let raw: RawManifest = parse_manifest(manifest, origin)?;
    artifact_from_raw(&raw, manifest, weights, origin)
}

fn parse_manifest(text: &str, origin: &str) -> FrontendResult<RawManifest> {
    serde_json::from_str(text).map_err(|error| GraphError::Json {
        path: origin.to_owned(),
        detail: error.to_string(),
    })
}

fn artifact_from_raw(
    raw: &RawManifest,
    text: &str,
    weights: &[u8],
    origin: &str,
) -> FrontendResult<GraphArtifact> {
    if weights.len() != raw.weight_bytes {
        return Err(GraphError::WeightFileLength {
            file: raw.weight_file.clone(),
            declared: raw.weight_bytes,
            actual: weights.len(),
        });
    }
    let graph = build_graph(raw, origin)?;
    let provenance = Provenance {
        source_format: raw.format.clone(),
        importer_version: IMPORTER_VERSION.to_owned(),
        manifest_sha256: sha256_hex(text.as_bytes()),
        weights_sha256: sha256_hex(weights),
        weight_file: raw.weight_file.clone(),
        weight_bytes: raw.weight_bytes,
    };
    let artifact = GraphArtifact::new(graph, provenance)?;
    // Prove the blob is decodable with the declared packing before it is trusted.
    unpack_weight_blob(weights, &artifact)?;
    Ok(artifact)
}

/// Serialize an artifact deterministically (`serde_json` pretty printing plus a
/// trailing newline). The artifact is validated first so a corrupt result is never
/// written.
pub fn save_artifact(artifact: &GraphArtifact, path: &Path) -> FrontendResult<()> {
    artifact.validate()?;
    let mut text = serde_json::to_string_pretty(artifact).map_err(|error| GraphError::Json {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    text.push('\n');
    std::fs::write(path, text).map_err(|error| GraphError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

/// Load and validate a persisted artifact.
pub fn load_artifact(path: &Path) -> FrontendResult<GraphArtifact> {
    let origin = path.display().to_string();
    let text = read_text(path)?;
    let artifact: GraphArtifact =
        serde_json::from_str(&text).map_err(|error| GraphError::Json {
            path: origin,
            detail: error.to_string(),
        })?;
    artifact.validate()?;
    Ok(artifact)
}

/// Decode the packed weight blob into the layer-ordered signed code stream.
///
/// This mirrors the reference unpacker: `i8-x1` stores one code per byte and `i4-x2`
/// stores two two's-complement nibbles per byte, low nibble first. It is used to prove
/// that the blob the manifest points at is decodable with the declared packing.
pub fn unpack_weight_blob(weights: &[u8], artifact: &GraphArtifact) -> FrontendResult<Vec<i8>> {
    let packing = artifact.graph.packing();
    let per_byte = packing.codes_per_byte();
    let mut codes = vec![0i8; artifact.graph.weight_stream.code_count];
    for layer in artifact.graph.layers() {
        let start = layer.weight_byte_offset;
        let end = start + layer.weight_bytes;
        if end > weights.len() {
            return Err(GraphError::WeightFileLength {
                file: artifact.graph.weight_stream.file.clone(),
                declared: end,
                actual: weights.len(),
            });
        }
        let mut decoded = 0usize;
        for byte in &weights[start..end] {
            for slot in 0..per_byte {
                let shift = 8 / per_byte * slot;
                let mask = (1u16 << packing.weight_bits()) - 1;
                let value = (u16::from(*byte) >> shift) & mask;
                let sign = 1u16 << (packing.weight_bits() - 1);
                let signed = if value & sign != 0 {
                    i32::from(value) - (1i32 << packing.weight_bits())
                } else {
                    i32::from(value)
                };
                let index = layer.weight_offset + decoded;
                if index >= codes.len() {
                    return Err(GraphError::WeightLayoutMismatch {
                        layer: layer.name.to_string(),
                        what: "decoded code index",
                        expected: codes.len(),
                        declared: index,
                    });
                }
                codes[index] = signed as i8;
                decoded += 1;
            }
        }
        if decoded != layer.weight_count {
            return Err(GraphError::WeightLayoutMismatch {
                layer: layer.name.to_string(),
                what: "decoded weight_count",
                expected: layer.weight_count,
                declared: decoded,
            });
        }
    }
    Ok(codes)
}

// ---------------------------------------------------------------------------
// Mapping
// ---------------------------------------------------------------------------

fn build_graph(raw: &RawManifest, origin: &str) -> FrontendResult<Graph> {
    if raw.format != MANIFEST_FORMAT {
        return Err(GraphError::ManifestFormat {
            path: origin.to_owned(),
            declared: raw.format.clone(),
        });
    }
    if !raw.lut_layers.is_empty() {
        return Err(GraphError::UnsupportedLut {
            count: raw.lut_layers.len(),
        });
    }

    let bits = resolve_bits(raw)?;
    let packing =
        Packing::parse(&raw.weight_packing).ok_or_else(|| GraphError::UnsupportedPacking {
            packing: raw.weight_packing.clone(),
            weight_bits: bits.weights.get(),
        })?;
    if packing.weight_bits() != bits.weights.get() {
        return Err(GraphError::UnsupportedPacking {
            packing: raw.weight_packing.clone(),
            weight_bits: bits.weights.get(),
        });
    }
    let codes_per_byte = packing.codes_per_byte();

    let point_count = raw.point_scales.len();
    if point_count == 0 {
        return Err(GraphError::Empty { what: "points" });
    }
    if point_count > u32::MAX as usize {
        return Err(GraphError::NumericOverflow {
            what: "point count",
            value: point_count,
        });
    }
    if !raw.points.is_empty() && raw.points.len() != point_count {
        return Err(GraphError::PointScaleCount {
            points: raw.points.len(),
            scales: point_count,
        });
    }
    let mut points = Vec::with_capacity(point_count);
    for (index, value) in raw.point_scales.iter().enumerate() {
        let scale = Scale::try_new("point scale", *value).map_err(|_| {
            GraphError::PointScaleNonPositive {
                point: index,
                value: *value,
            }
        })?;
        let name = raw
            .points
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("point{index}"));
        points.push(Point {
            id: PointId::new(index as u32),
            name: PointName::new(name),
            scale,
        });
    }

    let quant = QuantContract {
        bits,
        act_qmax: QMax::try_new("act_qmax", raw.act_qmax)?,
        output_qmax: QMax::try_new("output_qmax", raw.output_qmax)?,
        weight_qmax: QMax::try_new("weight_qmax", raw.weight_qmax)?,
        act_zero_point: ZeroPoint::symmetric(),
        accumulator: Dtype::Int32,
        requant_rounding: Rounding::HalfAwayFromZero,
        packing,
    };

    let mut layers = Vec::with_capacity(raw.layers.len());
    for (index, layer) in raw.layers.iter().enumerate() {
        layers.push(map_layer(
            layer,
            index,
            bits,
            point_count,
            codes_per_byte,
            &raw.point_scales,
        )?);
    }

    let ops = map_ops(&raw.graph, &layers, &raw.point_scales)?;

    let code_count = layers.iter().try_fold(0usize, |total, layer| {
        total
            .checked_add(layer.weight_count)
            .ok_or(GraphError::NumericOverflow {
                what: "weight code count",
                value: total,
            })
    })?;

    let graph = Graph {
        source_format: raw.format.clone(),
        arch: raw.arch.clone(),
        quant,
        points,
        ops,
        layers,
        weight_stream: WeightStream {
            file: raw.weight_file.clone(),
            byte_len: raw.weight_bytes,
            code_count,
            packing,
        },
        notes: raw.notes.clone(),
    };
    graph.validate()?;
    Ok(graph)
}

fn resolve_bits(raw: &RawManifest) -> FrontendResult<Bits> {
    match &raw.bits {
        Some(bits) => Ok(Bits {
            acts: BitWidth::try_new("acts", bits.acts)?,
            weights: BitWidth::try_new("weights", bits.weights)?,
        }),
        None => {
            let acts = BitWidth::from_signed_qmax(raw.act_qmax).ok_or(GraphError::MissingBits)?;
            let weights =
                BitWidth::from_signed_qmax(raw.weight_qmax).ok_or(GraphError::MissingBits)?;
            Ok(Bits { acts, weights })
        }
    }
}

fn extent(layer: &str, what: &'static str, value: usize) -> FrontendResult<Extent> {
    let narrowed = u32::try_from(value).map_err(|_| GraphError::NumericOverflow { what, value })?;
    Extent::try_new(narrowed).map_err(|_| GraphError::ZeroDimension {
        layer: layer.to_owned(),
        what,
        value,
    })
}

fn map_layer(
    raw: &RawLayer,
    index: usize,
    bits: Bits,
    point_count: usize,
    codes_per_byte: usize,
    point_scales: &[f32],
) -> FrontendResult<Layer> {
    let name = raw.name.clone();
    let op = LayerOp::parse(&raw.op)
        .ok_or_else(|| GraphError::UnsupportedOperator { op: raw.op.clone() })?;
    let in_channels = u32::try_from(raw.in_channels).map_err(|_| GraphError::NumericOverflow {
        what: "in_channels",
        value: raw.in_channels,
    })?;
    let out_channels =
        u32::try_from(raw.out_channels).map_err(|_| GraphError::NumericOverflow {
            what: "out_channels",
            value: raw.out_channels,
        })?;
    if in_channels == 0 {
        return Err(GraphError::ZeroDimension {
            layer: name,
            what: "in_channels",
            value: raw.in_channels,
        });
    }
    if out_channels == 0 {
        return Err(GraphError::ZeroDimension {
            layer: name,
            what: "out_channels",
            value: raw.out_channels,
        });
    }
    let groups = u32::try_from(raw.groups).map_err(|_| GraphError::NumericOverflow {
        what: "groups",
        value: raw.groups,
    })?;
    if groups == 0 {
        return Err(GraphError::BadGroups {
            layer: name,
            groups,
            in_channels,
            out_channels,
        });
    }
    if let Some(declared) = raw.act_bits {
        if declared != bits.acts.get() {
            return Err(GraphError::InvalidBitWidth {
                what: "layer activation",
                bits: declared,
            });
        }
    }
    if let Some(declared) = raw.weight_bits {
        if declared != bits.weights.get() {
            return Err(GraphError::InvalidBitWidth {
                what: "layer weight",
                bits: declared,
            });
        }
    }
    let zero_point = raw.act_zero_point.unwrap_or(0);
    if zero_point != 0 {
        return Err(GraphError::NonZeroZeroPoint {
            layer: name,
            value: zero_point,
        });
    }

    let kernel = Kernel {
        h: extent(&name, "kernel height", raw.kernel[0])?,
        w: extent(&name, "kernel width", raw.kernel[1])?,
    };
    let stride = Stride {
        h: extent(&name, "stride height", raw.stride[0])?,
        w: extent(&name, "stride width", raw.stride[1])?,
    };
    let padding = if raw.reflect_pad == 0 {
        Padding::none()
    } else {
        let amount = u32::try_from(raw.reflect_pad).map_err(|_| GraphError::NumericOverflow {
            what: "reflect_pad",
            value: raw.reflect_pad,
        })?;
        Padding::reflect(amount).map_err(|_| GraphError::ZeroDimension {
            layer: name.clone(),
            what: "reflect_pad",
            value: raw.reflect_pad,
        })?
    };

    let in_point =
        PointId::new(
            u32::try_from(raw.in_point).map_err(|_| GraphError::NumericOverflow {
                what: "in_point",
                value: raw.in_point,
            })?,
        );
    if raw.in_point >= point_count {
        return Err(GraphError::PointOutOfRange {
            layer: name,
            what: "in_point",
            point: in_point.get(),
            point_count,
        });
    }
    let out_point = if raw.out_point == point_count {
        PointRef::ImageSentinel
    } else if raw.out_point < point_count {
        PointRef::Point(PointId::new(raw.out_point as u32))
    } else {
        return Err(GraphError::PointOutOfRange {
            layer: name,
            what: "out_point",
            point: raw.out_point as u32,
            point_count,
        });
    };
    if out_point.is_image() && !raw.out_unsigned {
        return Err(GraphError::HeadSentinelNotUnsigned { layer: name });
    }

    let in_scale = Scale::try_new("in_scale", point_scales[raw.in_point])?;
    if let Some(declared) = raw.in_scale {
        if !scale_eq(declared, in_scale.get()) {
            return Err(GraphError::LayerScaleMismatch {
                layer: name,
                what: "in_scale",
                declared,
                expected: in_scale.get(),
            });
        }
    }
    let out_scale = match out_point {
        PointRef::Point(id) => {
            let expected = Scale::try_new("out_scale", point_scales[id.index()])?;
            if let Some(declared) = raw.out_scale {
                if !scale_eq(declared, expected.get()) {
                    return Err(GraphError::LayerScaleMismatch {
                        layer: name,
                        what: "out_scale",
                        declared,
                        expected: expected.get(),
                    });
                }
            }
            expected
        }
        PointRef::ImageSentinel => {
            let declared = raw.out_scale.ok_or_else(|| GraphError::MissingScale {
                layer: name.clone(),
                what: "out_scale (image scale)",
            })?;
            Scale::try_new("out_scale", declared)?
        }
    };

    let weight_count = raw.weight_count;
    if weight_count % codes_per_byte != 0 {
        return Err(GraphError::WeightLayoutMismatch {
            layer: name,
            what: "weight_count packing",
            expected: weight_count.div_ceil(codes_per_byte) * codes_per_byte,
            declared: weight_count,
        });
    }
    if raw.weight_offset % codes_per_byte != 0 {
        return Err(GraphError::WeightLayoutMismatch {
            layer: name,
            what: "weight_offset packing",
            expected: raw.weight_offset / codes_per_byte * codes_per_byte,
            declared: raw.weight_offset,
        });
    }
    let derived_byte_offset = raw.weight_offset / codes_per_byte;
    let derived_bytes = weight_count / codes_per_byte;
    if let Some(declared) = raw.weight_byte_offset {
        if declared != derived_byte_offset {
            return Err(GraphError::WeightLayoutMismatch {
                layer: name,
                what: "weight_byte_offset",
                expected: derived_byte_offset,
                declared,
            });
        }
    }
    if let Some(declared) = raw.weight_bytes {
        if declared != derived_bytes {
            return Err(GraphError::WeightLayoutMismatch {
                layer: name,
                what: "weight_bytes",
                expected: derived_bytes,
                declared,
            });
        }
    }

    Ok(Layer {
        id: LayerId::new(index as u32),
        name: LayerName::new(raw.name.clone()),
        op,
        kind: classify(op, groups, kernel),
        in_channels: Channels::from_validated(in_channels),
        out_channels: Channels::from_validated(out_channels),
        groups: GroupCount::from_validated(groups),
        act_bits: bits.acts,
        weight_bits: bits.weights,
        act_zero_point: ZeroPoint::symmetric(),
        out_relu: raw.out_relu,
        out_unsigned: raw.out_unsigned,
        kernel,
        stride,
        padding,
        in_point,
        out_point,
        in_scale,
        out_scale,
        weight_scale: raw.weight_scale.clone().unwrap_or_default(),
        bias: raw.bias.clone().unwrap_or_default(),
        bias_int32: raw.bias_int32.clone(),
        requant: RequantCoefficients::try_new(raw.requant.clone())?,
        weight_offset: raw.weight_offset,
        weight_count,
        weight_byte_offset: derived_byte_offset,
        weight_bytes: derived_bytes,
    })
}

fn classify(op: LayerOp, groups: u32, kernel: Kernel) -> LayerKind {
    if op.is_depthwise() {
        LayerKind::DepthwiseConv2d
    } else if kernel.h.get() == 1 && kernel.w.get() == 1 && groups == 1 {
        LayerKind::PointwiseConv2d
    } else {
        LayerKind::Conv2d
    }
}

fn map_ops(
    raw_ops: &[RawOp],
    layers: &[Layer],
    point_scales: &[f32],
) -> FrontendResult<Vec<GraphOp>> {
    let mut ops = Vec::with_capacity(raw_ops.len());
    for (index, raw) in raw_ops.iter().enumerate() {
        let op = match raw.op.as_str() {
            "conv" => {
                let name = raw
                    .layer
                    .clone()
                    .ok_or_else(|| GraphError::OpFieldMissing {
                        op_index: index,
                        kind: "conv",
                        field: "layer",
                    })?;
                let model = layers
                    .iter()
                    .find(|layer| layer.name.as_str() == name)
                    .ok_or_else(|| GraphError::UnknownLayer {
                        op_index: index,
                        layer: name.clone(),
                    })?;
                let point = raw.point.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "conv",
                    field: "point",
                })?;
                if point != model.in_point.index() {
                    return Err(GraphError::ConvPointMismatch {
                        op_index: index,
                        layer: name,
                        op_point: point as u32,
                        layer_in_point: model.in_point.get(),
                    });
                }
                if let Some(declared) = raw.scale {
                    let expected = Scale::try_new("point scale", point_scales[point])?;
                    if !scale_eq(declared, expected.get()) {
                        return Err(GraphError::OpScaleMismatch {
                            op_index: index,
                            kind: "conv",
                            declared,
                            expected: expected.get(),
                        });
                    }
                }
                GraphOp::Conv {
                    layer: model.name.clone(),
                    in_point: model.in_point,
                    out_point: model.out_point,
                }
            }
            "relu" => {
                let name = raw
                    .layer
                    .clone()
                    .ok_or_else(|| GraphError::OpFieldMissing {
                        op_index: index,
                        kind: "relu",
                        field: "layer",
                    })?;
                if !layers.iter().any(|layer| layer.name.as_str() == name) {
                    return Err(GraphError::UnknownLayer {
                        op_index: index,
                        layer: name,
                    });
                }
                GraphOp::Relu {
                    layer: LayerName::new(name),
                }
            }
            "add" => {
                let point = raw.point.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "add",
                    field: "point",
                })?;
                let source_point = raw.source_point.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "add",
                    field: "source_point",
                })?;
                let scale = raw.scale.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "add",
                    field: "scale",
                })?;
                GraphOp::Add {
                    point: PointId::new(point as u32),
                    source_point: PointId::new(source_point as u32),
                    scale: Scale::try_new("add scale", scale)?,
                }
            }
            "upsample_nearest2x" => {
                let point = raw.point.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "upsample_nearest2x",
                    field: "point",
                })?;
                GraphOp::UpsampleNearest2x {
                    point: PointId::new(point as u32),
                }
            }
            "quantize_u8_output" => {
                let scale = raw.scale.ok_or_else(|| GraphError::OpFieldMissing {
                    op_index: index,
                    kind: "quantize_u8_output",
                    field: "scale",
                })?;
                GraphOp::QuantizeU8Output {
                    scale: Scale::try_new("output scale", scale)?,
                }
            }
            other => {
                return Err(GraphError::UnsupportedGraphOp {
                    op: other.to_owned(),
                });
            }
        };
        ops.push(op);
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------
// IO and hashing
// ---------------------------------------------------------------------------

fn read_text(path: &Path) -> FrontendResult<String> {
    std::fs::read_to_string(path).map_err(|error| GraphError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn read_bytes(path: &Path) -> FrontendResult<Vec<u8>> {
    std::fs::read(path).map_err(|error| GraphError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

/// Minimal self-contained SHA-256 so provenance hashes need no extra dependency.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut message = Vec::with_capacity(data.len() + 72);
    message.extend_from_slice(data);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut schedule = [0u32; 64];
    for chunk in message.chunks_exact(64) {
        for (index, slot) in schedule.iter_mut().enumerate().take(16) {
            *slot = u32::from_be_bytes([
                chunk[4 * index],
                chunk[4 * index + 1],
                chunk[4 * index + 2],
                chunk[4 * index + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut out = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        out[4 * index..4 * index + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GRAPH_FORMAT, GRAPH_SCHEMA_VERSION, TensorShape};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// W8A8 fixture: a `stem` convolution, a depthwise 3x3 with fused relu, a pointwise
    /// 1x1, a residual add, a nearest upsample and an unsigned head. `groups` is omitted
    /// for the pointwise layers to cover the manifest default, and the byte offsets are
    /// declared for `stem` only to cover both the declared and the derived paths.
    const FIXTURE_W8: &str = r#"{
      "format": "fpga_st_train.int8.v1",
      "arch": "fixture",
      "bits": { "acts": 8, "weights": 8 },
      "act_qmax": 127.0,
      "output_qmax": 255.0,
      "weight_qmax": 127.0,
      "notes": ["synthetic fixture"],
      "points": ["image", "stem.in", "dw.in", "mid.in"],
      "point_scales": [0.007874016, 0.014, 0.02, 0.03],
      "weight_file": "weights.bin",
      "weight_packing": "i8-x1",
      "weight_bytes": 172,
      "graph": [
        { "op": "conv", "layer": "stem", "point": 0, "scale": 0.007874016 },
        { "op": "relu", "layer": "stem" },
        { "op": "conv", "layer": "dw", "point": 1, "scale": 0.014 },
        { "op": "relu", "layer": "dw" },
        { "op": "conv", "layer": "mid", "point": 2, "scale": 0.02 },
        { "op": "relu", "layer": "mid" },
        { "op": "add", "point": 3, "source_point": 1, "scale": 0.03 },
        { "op": "upsample_nearest2x", "point": 3 },
        { "op": "conv", "layer": "head", "point": 3, "scale": 0.03 },
        { "op": "quantize_u8_output", "scale": 0.003921569 }
      ],
      "layers": [
        {
          "name": "stem", "op": "conv2d+relu",
          "in_channels": 3, "out_channels": 4,
          "act_bits": 8, "weight_bits": 8, "act_zero_point": 0,
          "out_relu": true, "out_unsigned": false,
          "kernel": [3, 3], "stride": [1, 1], "reflect_pad": 1,
          "in_point": 0, "out_point": 1,
          "in_scale": 0.007874016, "out_scale": 0.014,
          "weight_scale": [0.01, 0.02, 0.03, 0.04],
          "bias": [0.1, -0.1, 0.2, -0.2],
          "bias_int32": [10, -10, 20, -20],
          "requant": [0.0056, 0.011, 0.017, 0.022],
          "weight_offset": 0, "weight_count": 108,
          "weight_byte_offset": 0, "weight_bytes": 108
        },
        {
          "name": "dw", "op": "depthwise_conv2d+relu",
          "in_channels": 4, "out_channels": 4, "groups": 4,
          "act_bits": 8, "weight_bits": 8, "act_zero_point": 0,
          "out_relu": true, "out_unsigned": false,
          "kernel": [3, 3], "stride": [1, 1], "reflect_pad": 1,
          "in_point": 1, "out_point": 2,
          "in_scale": 0.014, "out_scale": 0.02,
          "weight_scale": [0.01, 0.02, 0.03, 0.04],
          "bias": [0.0, 0.0, 0.0, 0.0],
          "bias_int32": [0, 0, 0, 0],
          "requant": [0.01, 0.01, 0.01, 0.01],
          "weight_offset": 108, "weight_count": 36
        },
        {
          "name": "mid", "op": "conv2d+relu",
          "in_channels": 4, "out_channels": 4,
          "out_relu": true, "out_unsigned": false,
          "kernel": [1, 1], "stride": [1, 1], "reflect_pad": 0,
          "in_point": 2, "out_point": 3,
          "in_scale": 0.02, "out_scale": 0.03,
          "weight_scale": [0.02, 0.02, 0.02, 0.02],
          "bias": [0.0, 0.0, 0.0, 0.0],
          "bias_int32": [0, 0, 0, 0],
          "requant": [0.01, 0.01, 0.01, 0.01],
          "weight_offset": 144, "weight_count": 16
        },
        {
          "name": "head", "op": "conv2d",
          "in_channels": 4, "out_channels": 3,
          "out_relu": false, "out_unsigned": true,
          "kernel": [1, 1], "stride": [1, 1], "reflect_pad": 0,
          "in_point": 3, "out_point": 4,
          "in_scale": 0.03, "out_scale": 0.003921569,
          "weight_scale": [0.02, 0.02, 0.02],
          "bias": [0.0, 0.0, 0.0],
          "bias_int32": [0, 0, 0],
          "requant": [0.1, 0.1, 0.1],
          "weight_offset": 160, "weight_count": 12
        }
      ]
    }"#;

    /// W4A8 fixture: the same geometry packed two codes per byte.
    const FIXTURE_W4: &str = r#"{
      "format": "fpga_st_train.int8.v1",
      "arch": "fixture",
      "act_qmax": 127.0,
      "output_qmax": 255.0,
      "weight_qmax": 7.0,
      "points": ["image"],
      "point_scales": [0.01],
      "weight_file": "weights.bin",
      "weight_packing": "i4-x2",
      "weight_bytes": 3,
      "graph": [
        { "op": "conv", "layer": "head", "point": 0, "scale": 0.01 },
        { "op": "quantize_u8_output", "scale": 0.003921569 }
      ],
      "layers": [
        {
          "name": "head", "op": "conv2d",
          "in_channels": 3, "out_channels": 2,
          "out_relu": false, "out_unsigned": true,
          "kernel": [1, 1], "stride": [1, 1], "reflect_pad": 0,
          "in_point": 0, "out_point": 1,
          "in_scale": 0.01, "out_scale": 0.003921569,
          "weight_scale": [0.5, 0.5],
          "bias": [0.0, 0.0],
          "bias_int32": [0, 0],
          "requant": [0.25, 0.25],
          "weight_offset": 0, "weight_count": 6
        }
      ]
    }"#;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tessera-frontend-{tag}-{}-{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn import(manifest: &str, weight_bytes: usize) -> GraphArtifact {
        import_manifest_str(manifest, &vec![0u8; weight_bytes], "fixture.json").expect("import")
    }

    #[test]
    fn imports_the_w8_fixture() {
        let artifact = import(FIXTURE_W8, 172);
        artifact.validate().unwrap();
        let graph = artifact.graph();
        assert_eq!(graph.arch(), "fixture");
        assert_eq!(graph.point_count(), 4);
        assert_eq!(graph.bits().acts.get(), 8);
        assert_eq!(graph.bits().weights.get(), 8);
        assert_eq!(graph.packing(), Packing::I8x1);
        assert_eq!(graph.act_qmax(), 127.0);
        assert_eq!(graph.output_qmax(), 255.0);
        assert_eq!(graph.weight_qmax(), 127.0);
        assert_eq!(graph.ops().len(), 10);
        assert_eq!(graph.layers().len(), 4);
        assert!(
            graph
                .layers()
                .iter()
                .all(|layer| layer.act_zero_point.get() == 0)
        );
        assert_eq!(graph.weight_stream().code_count, 172);
        assert_eq!(graph.weight_stream().byte_len, 172);
        assert_eq!(graph.notes().len(), 1);
        assert_eq!(graph.notes()[0], "synthetic fixture");

        let stem = graph.layer_named("stem").unwrap();
        assert_eq!(stem.kind, LayerKind::Conv2d);
        assert_eq!(stem.op, LayerOp::Conv2dRelu);
        assert_eq!(stem.in_channels.get(), 3);
        assert_eq!(stem.out_channels.get(), 4);
        assert_eq!(stem.groups.get(), 1);
        assert_eq!(stem.kernel.h.get(), 3);
        assert_eq!(stem.stride.w.get(), 1);
        assert!(stem.padding.is_reflect());
        assert_eq!(stem.padding.amount().map(Extent::get), Some(1));
        assert_eq!(stem.out_point, PointRef::Point(PointId::new(1)));
        assert_eq!(stem.bias_int32.len(), 4);
        assert_eq!(stem.bias.len(), 4);
        assert_eq!(stem.weight_scale.len(), 4);
        assert_eq!(stem.requant.len(), 4);
        assert_eq!(stem.weight_offset, 0);
        assert_eq!(stem.weight_byte_offset, 0);
        assert_eq!(stem.weight_bytes, 108);

        let depthwise = graph.layer_named("dw").unwrap();
        assert_eq!(depthwise.kind, LayerKind::DepthwiseConv2d);
        assert_eq!(depthwise.groups.get(), 4);
        assert_eq!(depthwise.weight_offset, 108);
        // Byte offsets are derived when the manifest omits them.
        assert_eq!(depthwise.weight_byte_offset, 108);
        assert_eq!(depthwise.weight_bytes, 36);
        assert_eq!(depthwise.expected_weight_count(), 4 * 9);

        let pointwise = graph.layer_named("mid").unwrap();
        // `groups` was omitted, so it defaults to 1 and the 1x1 conv is pointwise.
        assert_eq!(pointwise.groups.get(), 1);
        assert_eq!(pointwise.kind, LayerKind::PointwiseConv2d);
        assert_eq!(pointwise.weight_offset, 144);
        assert_eq!(pointwise.weight_bytes, 16);

        let head = graph.layer_named("head").unwrap();
        assert_eq!(head.out_point, PointRef::ImageSentinel);
        assert!(head.out_unsigned);
        assert_eq!(head.out_point.id(), None);
        assert!(head.out_scale.get() > 0.0);
        assert_eq!(graph.head_sentinel(), PointId::new(4));

        // Ops keep manifest order and map to typed variants.
        assert_eq!(graph.ops()[0].tag(), "conv");
        assert_eq!(graph.ops()[1].tag(), "relu");
        assert_eq!(graph.ops()[3].tag(), "relu");
        assert_eq!(graph.ops()[6].tag(), "add");
        assert_eq!(graph.ops()[6].point(), Some(PointId::new(3)));
        assert_eq!(graph.ops()[6].source_point(), Some(PointId::new(1)));
        assert_eq!(graph.ops()[6].scale().map(Scale::get), Some(0.03));
        assert_eq!(graph.ops()[7].tag(), "upsample_nearest2x");
        assert_eq!(graph.ops()[9].tag(), "quantize_u8_output");
        assert_eq!(graph.ops()[9].layer(), None);
        assert_eq!(
            graph.ops()[9].kind(),
            crate::graph::GraphOpKind::QuantizeU8Output
        );

        let metrics = artifact.metrics();
        assert_eq!(metrics.layer_count, 4);
        assert_eq!(metrics.conv_layer_count, 1);
        assert_eq!(metrics.depthwise_layer_count, 1);
        assert_eq!(metrics.pointwise_layer_count, 2);
        assert_eq!(metrics.relu_layer_count, 3);
        assert_eq!(metrics.add_count, 1);
        assert_eq!(metrics.upsample_count, 1);
        assert_eq!(metrics.graph_op_count, 10);
        assert_eq!(metrics.macs_per_output_pixel, 108 + 36 + 16 + 12);
        // Point 1 feeds the depthwise layer and the residual add.
        assert_eq!(metrics.point_fanout[1].consumers, 2);
        assert_eq!(metrics.per_layer[3].name.as_str(), "head");

        let report = graph
            .infer_shapes(&TensorShape::try_new(3, 8, 12).unwrap())
            .unwrap();
        assert_eq!(report.point_shapes[1].channels.get(), 4);
        assert_eq!(report.point_shapes[1].height.get(), 8);
        assert_eq!(report.point_shapes[1].width.get(), 12);
        // Point 3 is upsampled in place.
        assert_eq!(report.point_shapes[3].channels.get(), 4);
        assert_eq!(report.point_shapes[3].height.get(), 16);
        assert_eq!(report.point_shapes[3].width.get(), 24);
        assert_eq!(report.image.channels.get(), 3);
        assert_eq!(report.image.height.get(), 16);
        assert_eq!(report.image.width.get(), 24);
        assert_eq!(
            report.macs,
            108 * 8 * 12 + 36 * 8 * 12 + 16 * 8 * 12 + 12 * 16 * 24
        );
    }

    #[test]
    fn imports_the_w4_fixture_and_unpacks_nibbles() {
        let artifact = import(FIXTURE_W4, 3);
        artifact.validate().unwrap();
        let graph = artifact.graph();
        assert_eq!(graph.bits().weights.get(), 4);
        assert_eq!(graph.packing(), Packing::I4x2);
        assert_eq!(graph.weight_qmax(), 7.0);
        // bits were inferred from the qmax because the manifest omits them.
        assert_eq!(graph.bits().acts.get(), 8);
        assert_eq!(graph.layer_named("head").unwrap().weight_bytes, 3);
        assert!(matches!(
            unpack_weight_blob(&[0xf8, 0x12], &artifact),
            Err(GraphError::WeightFileLength { .. })
        ));
        // Low nibble first, two's complement: 0xf8 -> (-8, -1), 0x12 -> (2, 1).
        let full = unpack_weight_blob(&[0xf8, 0x12, 0x00], &artifact).unwrap();
        assert_eq!(full, vec![-8, -1, 2, 1, 0, 0]);
    }

    #[test]
    fn imports_a_manifest_without_bits_or_points() {
        // The streaming manifests omit `bits` and `points`; widths are recovered from the
        // symmetric qmax values and point names are positional.
        let manifest = FIXTURE_W8
            .replace("      \"bits\": { \"acts\": 8, \"weights\": 8 },\n", "")
            .replace(
                "      \"points\": [\"image\", \"stem.in\", \"dw.in\", \"mid.in\"],\n",
                "",
            );
        let artifact = import_manifest_str(&manifest, &vec![0u8; 172], "synthetic.json").unwrap();
        let graph = artifact.graph();
        assert_eq!(graph.bits().acts.get(), 8);
        assert_eq!(graph.bits().weights.get(), 8);
        assert_eq!(graph.point_count(), 4);
        assert_eq!(graph.points()[0].name.as_str(), "point0");
        assert_eq!(graph.points()[3].name.as_str(), "point3");
    }

    #[test]
    fn rejects_a_foreign_manifest_format() {
        let manifest = FIXTURE_W8.replace(MANIFEST_FORMAT, "fpga_st_train.lut.v1");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::ManifestFormat { .. })
        ));
    }

    #[test]
    fn rejects_lut_layers() {
        let manifest = FIXTURE_W8.replace(
            "\"graph\": [",
            "\"lut_layers\": [{ \"name\": \"enc.0\" }],\n      \"graph\": [",
        );
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnsupportedLut { count: 1 })
        ));
    }

    #[test]
    fn rejects_unsupported_packing() {
        let manifest = FIXTURE_W8.replace("i8-x1", "i2-x4");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnsupportedPacking { .. })
        ));
    }

    #[test]
    fn rejects_packing_that_contradicts_the_bit_width() {
        // Declares 4-bit packing while the widths stay 8-bit.
        let manifest = FIXTURE_W8.replace("i8-x1", "i4-x2");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnsupportedPacking { .. })
        ));
    }

    #[test]
    fn rejects_a_weight_file_of_the_wrong_length() {
        assert!(matches!(
            import_manifest_str(FIXTURE_W8, &vec![0u8; 171], "fixture.json"),
            Err(GraphError::WeightFileLength {
                declared: 172,
                actual: 171,
                ..
            })
        ));
    }

    #[test]
    fn rejects_a_point_index_out_of_range() {
        let manifest = FIXTURE_W8.replace(
            "\"in_point\": 0, \"out_point\": 1",
            "\"in_point\": 9, \"out_point\": 1",
        );
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::PointOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_a_declared_scale_that_contradicts_the_point_scale() {
        let manifest = FIXTURE_W8.replace("\"scale\": 0.014 }", "\"scale\": 0.5 }");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::OpScaleMismatch { .. })
        ));
    }

    #[test]
    fn rejects_a_layer_name_that_no_layer_defines() {
        let manifest = FIXTURE_W8.replace("\"layer\": \"mid\"", "\"layer\": \"missing\"");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnknownLayer { .. })
        ));
    }

    #[test]
    fn rejects_an_unsupported_layer_operator() {
        let manifest = FIXTURE_W8.replace("depthwise_conv2d+relu", "depthwise_conv2d+sigmoid");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnsupportedOperator { .. })
        ));
    }

    #[test]
    fn rejects_an_unsupported_graph_op() {
        let manifest = FIXTURE_W8.replace("upsample_nearest2x", "upsample_bilinear");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::UnsupportedGraphOp { .. })
        ));
    }

    #[test]
    fn rejects_a_zero_channel_layer() {
        let manifest = FIXTURE_W8.replace(
            "\"in_channels\": 4, \"out_channels\": 4, \"groups\": 4,",
            "\"in_channels\": 0, \"out_channels\": 4, \"groups\": 4,",
        );
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::ZeroDimension { .. })
        ));
    }

    #[test]
    fn rejects_a_weight_count_that_does_not_match_the_geometry() {
        let manifest = FIXTURE_W8
            .replace("\"weight_count\": 108", "\"weight_count\": 107")
            .replace("\"weight_bytes\": 108", "\"weight_bytes\": 107");
        let error = import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json").unwrap_err();
        assert!(
            matches!(
                error,
                GraphError::WeightCountMismatch {
                    declared: 107,
                    expected: 108,
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_a_declared_byte_offset_that_disagrees_with_the_codes() {
        let manifest = FIXTURE_W8.replace("\"weight_byte_offset\": 0", "\"weight_byte_offset\": 3");
        let error = import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json").unwrap_err();
        assert!(
            matches!(
                error,
                GraphError::WeightLayoutMismatch {
                    what: "weight_byte_offset",
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_a_weight_count_that_breaks_byte_packing() {
        let manifest = FIXTURE_W4.replace("\"weight_count\": 6", "\"weight_count\": 5");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 3], "fixture.json"),
            Err(GraphError::WeightLayoutMismatch {
                what: "weight_count packing",
                ..
            })
        ));
    }

    #[test]
    fn rejects_a_nonzero_activation_zero_point() {
        let manifest = FIXTURE_W8.replace("\"act_zero_point\": 0", "\"act_zero_point\": 3");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::NonZeroZeroPoint { .. })
        ));
    }

    #[test]
    fn rejects_a_non_positive_weight_scale() {
        let manifest = FIXTURE_W8.replace(
            "\"weight_scale\": [0.01, 0.02, 0.03, 0.04]",
            "\"weight_scale\": [0.01, 0.0, 0.03, 0.04]",
        );
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::NonFinite { .. })
        ));
    }

    #[test]
    fn rejects_a_negative_requant_coefficient() {
        let manifest = FIXTURE_W8.replace(
            "\"requant\": [0.1, 0.1, 0.1]",
            "\"requant\": [0.1, -0.1, 0.1]",
        );
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::NonFinite { .. })
        ));
    }

    #[test]
    fn rejects_a_bias_vector_length_mismatch() {
        let manifest =
            FIXTURE_W8.replace("\"bias_int32\": [0, 0, 0, 0]", "\"bias_int32\": [0, 0, 0]");
        assert!(matches!(
            import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json"),
            Err(GraphError::VectorLengthMismatch {
                what: "bias_int32",
                ..
            })
        ));
    }

    #[test]
    fn rejects_a_point_and_scale_count_mismatch() {
        let manifest = FIXTURE_W8.replace(
            "\"points\": [\"image\", \"stem.in\", \"dw.in\", \"mid.in\"]",
            "\"points\": [\"image\", \"dw.in\"]",
        );
        let error = import_manifest_str(&manifest, &vec![0u8; 172], "fixture.json").unwrap_err();
        assert!(
            matches!(
                error,
                GraphError::PointScaleCount {
                    points: 2,
                    scales: 4
                }
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn artifact_save_load_and_import_from_disk_round_trip() {
        let dir = TempDir::new("import");
        std::fs::write(dir.path.join("manifest.json"), FIXTURE_W8).unwrap();
        std::fs::write(dir.path.join("weights.bin"), vec![0u8; 172]).unwrap();

        let artifact = import_manifest(&dir.path).expect("import from disk");
        artifact.validate().unwrap();
        assert_eq!(artifact.format, GRAPH_FORMAT);
        assert_eq!(artifact.schema_version, GRAPH_SCHEMA_VERSION);
        assert_eq!(artifact.provenance.source_format, MANIFEST_FORMAT);
        assert_eq!(artifact.provenance.weight_file, "weights.bin");
        assert_eq!(artifact.provenance.weight_bytes, 172);
        assert_eq!(
            artifact.provenance.manifest_sha256,
            sha256_hex(FIXTURE_W8.as_bytes())
        );
        assert_eq!(artifact.provenance.weights_sha256, sha256_hex(&[0u8; 172]));

        let path = dir.path.join("graph.tgraph");
        save_artifact(&artifact, &path).unwrap();
        let loaded = load_artifact(&path).unwrap();
        assert_eq!(loaded, artifact);
        // Deterministic on-disk representation.
        let first = std::fs::read_to_string(&path).unwrap();
        save_artifact(&loaded, &path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), first);
        assert!(first.ends_with('\n'));
    }

    #[test]
    fn load_rejects_a_tampered_artifact() {
        let dir = TempDir::new("tamper");
        let artifact = import(FIXTURE_W8, 172);
        let path = dir.path.join("graph.tgraph");
        save_artifact(&artifact, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let tampered = text.replace("\"weight_codes\": 172", "\"weight_codes\": 63");
        std::fs::write(&path, tampered).unwrap();
        assert_eq!(load_artifact(&path), Err(GraphError::MetricsStale));
    }

    #[test]
    fn import_reports_a_missing_weight_file() {
        let dir = TempDir::new("missing-weights");
        std::fs::write(dir.path.join("manifest.json"), FIXTURE_W8).unwrap();
        assert!(matches!(
            import_manifest(&dir.path),
            Err(GraphError::Io { .. })
        ));
    }

    #[test]
    fn import_reports_a_missing_manifest() {
        let dir = TempDir::new("missing-manifest");
        assert!(matches!(
            import_manifest(&dir.path),
            Err(GraphError::Io { .. })
        ));
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_handles_multiblock_input() {
        let repeated = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&repeated),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn unpack_rejects_a_blob_shorter_than_the_layout_needs() {
        let artifact = import(FIXTURE_W8, 172);
        assert!(matches!(
            unpack_weight_blob(&[0u8; 160], &artifact),
            Err(GraphError::WeightFileLength { .. })
        ));
    }
}
