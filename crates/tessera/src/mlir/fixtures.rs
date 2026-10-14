//! Shared test fixture: a small but structurally faithful graph.
//!
//! Shapes follow the manifest contract (`fpga_st_train.int8.v1`): the manifest
//! carries no spatial extents, so the fixture pairs a 6x6x3 input with
//! [`MlirOptions`].
//!
//! ```text
//! p0 (6x6x3)
//!   c0 conv 3->4, 3x3, stride 1, reflect 1, relu  -> p1 (6x6x4)
//!   c1 conv 3->4, 1x1, stride 1, no padding       -> p2 (6x6x4)
//!   add: p1 += requant(p2)                        -> p1 (6x6x4)
//!   head conv 4->3, 1x1, unsigned output          -> image (6x6x3)
//! ```

use crate::graph::{
    BitWidth, Bits, Channels, Dtype, Graph, GraphOp, GroupCount, Kernel, Layer, LayerId, LayerKind,
    LayerName, LayerOp, Packing, Padding, Point, PointId, PointName, PointRef, QMax, QuantContract,
    RequantCoefficients, Rounding, Scale, Stride, WeightStream, ZeroPoint,
};
use crate::mlir::MlirOptions;

/// Input activation shape of the fixture graph, in `NHWC` order.
pub const INPUT_SHAPE: [u64; 4] = [1, 6, 6, 3];

/// Options matching [`sample_graph`].
pub fn sample_options() -> MlirOptions {
    MlirOptions::new(INPUT_SHAPE)
}

fn scale(value: f32) -> Scale {
    Scale::try_new("fixture scale", value).expect("positive scale")
}

fn qmax(value: f32) -> QMax {
    QMax::try_new("fixture qmax", value).expect("positive qmax")
}

fn width(bits: u8) -> BitWidth {
    BitWidth::try_new("fixture width", bits).expect("valid width")
}

fn in_scale_for(point: u32) -> f32 {
    // Distinct, positive scales so the add factor is not accidentally 1.
    match point {
        0 => 0.007_874_016,
        1 => 0.009_141_972,
        _ => 0.010_968_110_5,
    }
}

#[allow(clippy::too_many_arguments)]
fn layer(
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
    out_relu: bool,
    out_unsigned: bool,
    max_requant: f32,
    weight_offset: usize,
) -> Layer {
    let codes_per_channel =
        (in_channels / groups.max(1)) as usize * kernel.0 as usize * kernel.1 as usize;
    let requant: Vec<f32> = (0..out_channels)
        .map(|channel| max_requant * (0.9 + 0.1 * (channel % 2) as f32))
        .collect();
    let out_scale = match out_point {
        PointRef::ImageSentinel => 1.0 / 255.0,
        PointRef::Point(point) => in_scale_for(point.get()),
    };
    Layer {
        id: LayerId::new(id),
        name: LayerName::new(name),
        op,
        kind,
        in_channels: Channels::try_new(in_channels).expect("channels"),
        out_channels: Channels::try_new(out_channels).expect("channels"),
        groups: GroupCount::try_new(groups).expect("groups"),
        act_bits: width(8),
        weight_bits: width(8),
        act_zero_point: ZeroPoint::symmetric(),
        out_relu,
        out_unsigned,
        kernel: Kernel::try_new(kernel.0, kernel.1).expect("kernel"),
        stride: Stride::try_new(stride.0, stride.1).expect("stride"),
        padding,
        in_point: PointId::new(in_point),
        out_point,
        in_scale: scale(in_scale_for(in_point)),
        out_scale: scale(out_scale),
        weight_scale: vec![0.0015; out_channels as usize],
        bias: vec![0.01; out_channels as usize],
        bias_int32: (0..out_channels as usize)
            .map(|channel| channel as i32 * 3 - 5)
            .collect(),
        requant: RequantCoefficients::try_new(requant).expect("requant"),
        weight_offset,
        weight_count: out_channels as usize * codes_per_channel,
        weight_byte_offset: weight_offset,
        weight_bytes: out_channels as usize * codes_per_channel,
    }
}

/// A validated-by-construction graph covering every emitted v1 operator.
pub fn sample_graph() -> Graph {
    let c0 = layer(
        0,
        "enc.0",
        LayerOp::Conv2dRelu,
        LayerKind::Conv2d,
        3,
        4,
        1,
        (3, 3),
        (1, 1),
        Padding::reflect(1).expect("padding"),
        0,
        PointRef::Point(PointId::new(1)),
        true,
        false,
        0.001_6,
        0,
    );
    let c1 = layer(
        1,
        "res.0.c1",
        LayerOp::Conv2d,
        LayerKind::Conv2d,
        3,
        4,
        1,
        (1, 1),
        (1, 1),
        Padding::none(),
        0,
        PointRef::Point(PointId::new(2)),
        false,
        false,
        0.002_1,
        c0.weight_count,
    );
    let head = layer(
        2,
        "head",
        LayerOp::Conv2d,
        LayerKind::PointwiseConv2d,
        4,
        3,
        1,
        (1, 1),
        (1, 1),
        Padding::none(),
        1,
        PointRef::ImageSentinel,
        false,
        true,
        0.001_9,
        c0.weight_count + c1.weight_count,
    );
    let code_count: usize = c0.weight_count + c1.weight_count + head.weight_count;

    let points = vec![
        Point {
            id: PointId::new(0),
            name: PointName::new("image"),
            scale: scale(in_scale_for(0)),
        },
        Point {
            id: PointId::new(1),
            name: PointName::new("enc.0.out"),
            scale: scale(in_scale_for(1)),
        },
        Point {
            id: PointId::new(2),
            name: PointName::new("res.0.c1.out"),
            scale: scale(in_scale_for(2)),
        },
    ];

    let ops = vec![
        GraphOp::Conv {
            layer: LayerName::new("enc.0"),
            in_point: PointId::new(0),
            out_point: PointRef::Point(PointId::new(1)),
        },
        GraphOp::Relu {
            layer: LayerName::new("enc.0"),
        },
        GraphOp::Conv {
            layer: LayerName::new("res.0.c1"),
            in_point: PointId::new(0),
            out_point: PointRef::Point(PointId::new(2)),
        },
        GraphOp::Add {
            point: PointId::new(1),
            source_point: PointId::new(2),
            // `GraphOp::Add` carries the *destination* point's scale.
            scale: scale(in_scale_for(1)),
        },
        GraphOp::Conv {
            layer: LayerName::new("head"),
            in_point: PointId::new(1),
            out_point: PointRef::ImageSentinel,
        },
        GraphOp::QuantizeU8Output {
            scale: scale(1.0 / 255.0),
        },
    ];

    Graph {
        source_format: "fpga_st_train.int8.v1".to_string(),
        arch: "tiny".to_string(),
        quant: QuantContract {
            bits: Bits {
                acts: width(8),
                weights: width(8),
            },
            act_qmax: qmax(127.0),
            output_qmax: qmax(255.0),
            weight_qmax: qmax(127.0),
            act_zero_point: ZeroPoint::symmetric(),
            accumulator: Dtype::Int32,
            requant_rounding: Rounding::HalfAwayFromZero,
            packing: Packing::I8x1,
        },
        points,
        ops,
        layers: vec![c0, c1, head],
        weight_stream: WeightStream {
            file: "weights.bin".to_string(),
            byte_len: code_count,
            code_count,
            packing: Packing::I8x1,
        },
        notes: vec!["fixture".to_string()],
    }
}
