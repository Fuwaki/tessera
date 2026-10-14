//! Acceptance vectors: the fixtures' frozen expectations, re-derived.
//!
//! Every number this file checks is recomputed here from `manifest.json`,
//! `weights.bin` and `input.json` alone, by the independent scalar reference in
//! `acceptance_support`. Nothing is taken from the compiler, from lowering, or
//! from the RTL: if the reference and the frozen vectors disagree, one of them
//! is wrong and the fixture cannot be trusted.

mod acceptance_support;

use acceptance_support::{
    ExpectedOp, Fixture, Plane, compare_expected_op, compare_expected_requant,
    convolve_with_zero_pad, fixture_dir, quantize_image, requantize, run_fixture,
    serialize_requant,
};

/// The scaling factor that makes the rounding fixture's mid layer expose exact
/// half-way accumulators.
fn distinct(values: &[i64]) -> usize {
    let mut seen = values.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

#[test]
fn rounding_fixture_is_hand_derivable_and_matches_the_reference() {
    let fixture = Fixture::load("rounding");
    let frames = run_fixture(&fixture);
    assert_eq!(frames.len(), 2, "the fixture drives two frames");

    // The per-channel requantization is exactly [0.25, 0.125, 0.25, 0.25], so
    // shift = 18 and M0 = [65536, 32768, 65536, 65536]. Those numbers make the
    // accumulator grid land exactly on .5 for the input codes the fixture uses,
    // which is what makes the table below checkable with a calculator.
    let mid_requant = &fixture.manifest.layer("rnd").requant;
    assert_eq!(mid_requant.clone(), vec![0.25f32, 0.125, 0.25, 0.25]);
    assert_eq!(
        serialize_requant(mid_requant),
        (vec![65536, 32768, 65536, 65536], 18)
    );
    let head_requant = &fixture.manifest.layer("head").requant;
    assert_eq!(
        serialize_requant(head_requant),
        (vec![65536, 65536, 65536], 16)
    );

    // Frame 0 input positions are [-2,4,0,0], [2,4,0,0], [0,4,0,0], [-2,2,0,0].
    // `rnd` is a 1x1 identity convolution with bias_int32 [0,0,2000,-2000], so
    //   channel 0: -2 * 0.25 = -0.5 -> -1   (round half away from zero, negative)
    //   channel 1:  4 * 0.125 = 0.5 -> 1    (round half away from zero, positive)
    //   channel 2: 2000 * 0.25 = 500 -> 127 (saturate at the relu-free signed clamp)
    //   channel 3: -2000 * 0.25 = -500 -> -127
    let mid = &frames[0].ops[0];
    assert_eq!(mid.name, "rnd");
    assert_eq!((mid.plane.c, mid.plane.h, mid.plane.w), (4, 2, 2));
    assert_eq!(mid.plane.at(0, 0, 0), -1);
    assert_eq!(mid.plane.at(0, 0, 1), 1);
    assert_eq!(mid.plane.at(0, 1, 0), 0);
    assert_eq!(mid.plane.at(0, 1, 1), -1);
    assert_eq!(mid.plane.at(1, 0, 0), 1);
    assert_eq!(mid.plane.at(1, 1, 1), 0);
    assert_eq!(mid.plane.at(2, 0, 0), 127);
    assert_eq!(mid.plane.at(3, 0, 0), -127);

    // Frame 1 confirms the same channels from the other side: channel 0 at
    // position (0,0) is +2 -> 0.5 -> 1, channel 1 at (1,0) is -4 -> -0.5 -> -1.
    let mid_frame1 = &frames[1].ops[0];
    assert_eq!(mid_frame1.plane.at(0, 0, 0), 1);
    assert_eq!(mid_frame1.plane.at(1, 0, 0), -1);
    assert_eq!(mid_frame1.plane.at(0, 0, 1), 0);

    // The head selects channels 1..3 with a requantization of exactly 1.0, so
    // the unsigned clamp [0, 255] is a passthrough:
    //   channel 0 = rnd channel 1 -> [1, 1, 1, 0]
    //   channel 1 = rnd channel 2 -> [127, 127, 127, 127]
    //   channel 2 = rnd channel 3 -> [-127, ...] -> [0, 0, 0, 0]
    let head = &frames[0].ops[1];
    assert_eq!(head.name, "head");
    assert_eq!(head.elem, "u8");
    assert_eq!(head.plane.at(0, 0, 0), 1);
    assert_eq!(head.plane.at(0, 0, 1), 1);
    assert_eq!(head.plane.at(0, 1, 0), 1);
    assert_eq!(head.plane.at(0, 1, 1), 0);
    assert_eq!(head.plane.at(1, 0, 0), 127);
    assert_eq!(head.plane.at(2, 1, 1), 0);
    assert_eq!(frames[0].image.codes, head.plane.codes);
}

#[test]
fn rounding_fixture_matches_its_frozen_vectors() {
    let fixture = Fixture::load("rounding");
    let frames = run_fixture(&fixture);

    for frame in &frames {
        let frozen = fixture.frame_vectors(frame.index);
        assert_eq!(
            frozen.ops.len(),
            frame.ops.len(),
            "frame {}: frozen operator count",
            frame.index
        );
        for (expected, reference) in frozen.ops.iter().zip(&frame.ops) {
            compare_expected_op(
                &format!("rounding/frame{}", frame.index),
                expected,
                reference,
                reference.plane.shape(),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        }
        assert_eq!(
            frozen.output_u8, frame.image.codes,
            "frame {}: the frozen image differs from the reference",
            frame.index
        );
    }

    for layer in &fixture.vectors.layers {
        let reference = frames[0]
            .ops
            .iter()
            .find(|op| op.name == layer.name)
            .unwrap_or_else(|| panic!("the reference did not evaluate layer {}", layer.name));
        if let Some(requant) = &layer.requant {
            compare_expected_requant(&layer.name, requant, reference.requant.as_ref())
                .unwrap_or_else(|error| panic!("{error}"));
        }
        assert_eq!(
            (layer.shape.c, layer.shape.h, layer.shape.w),
            reference.plane.shape(),
            "layer {}: frozen shape",
            layer.name
        );
    }
}

#[test]
fn graph_fixture_matches_its_frozen_vectors() {
    let fixture = Fixture::load("graph");
    let frames = run_fixture(&fixture);
    assert_eq!(frames.len(), 2);

    assert_eq!(
        frames[0]
            .ops
            .iter()
            .map(|op| op.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "stem",
            "dw",
            "mid",
            "add",
            "upsample_nearest2x",
            "head",
            "quantize_u8_output"
        ],
        "the fixture must exercise every acceptance operator in order"
    );

    for frame in &frames {
        let frozen = fixture.frame_vectors(frame.index);
        assert_eq!(frozen.ops.len(), frame.ops.len());
        for (expected, reference) in frozen.ops.iter().zip(&frame.ops) {
            compare_expected_op(
                &format!("graph/frame{}", frame.index),
                expected,
                reference,
                reference.plane.shape(),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        }
        assert_eq!(frozen.output_u8, frame.image.codes, "frame {}", frame.index);
    }
}

#[test]
fn graph_fixture_covers_every_documented_acceptance_case() {
    let fixture = Fixture::load("graph");
    let frames = run_fixture(&fixture);

    // 1. Reflection padding is real: with zero padding the stem output differs.
    let stem = fixture.manifest.layer("stem");
    let zero_padded =
        convolve_with_zero_pad(&fixture.manifest, &fixture.weights, stem, &frames[0].input);
    let reflected = &frames[0].ops[0].plane;
    assert_ne!(
        zero_padded.codes, reflected.codes,
        "the fixture must distinguish reflection padding from zero padding"
    );
    assert!(
        zero_padded
            .codes
            .iter()
            .zip(&reflected.codes)
            .any(|(a, b)| a != b),
        "at least one border output must depend on the padding mode"
    );

    // 2. The residual carries 9-bit values: the add output leaves [-128, 127]
    //    and the next two operators carry that value forward unchanged.
    let mut residual_outside_i8 = 0usize;
    for frame in &frames {
        let add = &frame.ops[3];
        assert_eq!(add.name, "add");
        assert_eq!(add.elem, "i16");
        assert!(
            add.plane.outside_i8() > 0,
            "frame {}: the residual must exceed the signed 8-bit domain",
            frame.index
        );
        residual_outside_i8 += add.plane.outside_i8();
        assert!(
            add.plane.codes.iter().any(|code| *code > 127),
            "frame {}: a positive 9-bit carry must be present",
            frame.index
        );
        assert!(
            add.plane
                .codes
                .iter()
                .all(|code| *code >= -254 && *code <= 254),
            "frame {}: the residual domain is [-254, 254]",
            frame.index
        );

        // 3. Nearest 2x upsampling duplicates every code into a 2x2 block and
        //    keeps the 9-bit values, and the unsigned head reproduces them.
        let up = &frame.ops[4];
        assert_eq!((up.plane.c, up.plane.h, up.plane.w), (4, 10, 10));
        for c in 0..4 {
            for y in 0..10 {
                for x in 0..10 {
                    assert_eq!(
                        up.plane.at(c, y, x),
                        add.plane.at(c, y / 2, x / 2),
                        "upsample ({c},{y},{x}) is a 2x2 duplication"
                    );
                }
            }
        }
        let head = &frame.ops[5];
        let quantize = &frame.ops[6];
        assert_eq!(head.elem, "u8");
        assert_eq!(quantize.elem, "u8");
        assert_eq!(
            head.plane.codes, quantize.plane.codes,
            "the unsigned output clamp feeds the image unchanged"
        );
        assert_eq!(frame.image.codes, head.plane.codes);
        assert!(head.plane.codes.iter().all(|code| (0..=255).contains(code)));
    }
    assert!(residual_outside_i8 > 0);

    // 4. Both ends of the unsigned clamp are exercised across the two frames.
    let head_codes: Vec<i64> = frames
        .iter()
        .flat_map(|frame| frame.ops[5].plane.codes.iter().copied())
        .collect();
    assert!(
        head_codes.iter().any(|code| *code == 0),
        "the unsigned head must clamp negatives to 0"
    );
    assert!(
        head_codes.iter().any(|code| *code == 255),
        "the unsigned head must saturate at output_qmax"
    );

    // 5. The relu clamp is exercised on both ends too.
    let stem_codes: Vec<i64> = frames
        .iter()
        .flat_map(|frame| frame.ops[0].plane.codes.iter().copied())
        .collect();
    assert!(
        stem_codes.iter().any(|code| *code == 0),
        "relu must produce zeros"
    );
    assert!(
        stem_codes.iter().any(|code| *code == 127),
        "relu must saturate at act_qmax"
    );

    // 6. Every layer's per-channel multipliers are heterogeneous, so a datapath
    //    that ignored the per-channel index would fail.
    for layer in ["stem", "dw", "mid", "head"] {
        let requant = &fixture.manifest.layer(layer).requant;
        assert!(
            distinct(
                &requant
                    .iter()
                    .map(|value| (*value as f64 * 1e12) as i64)
                    .collect::<Vec<_>>()
            ) >= 3,
            "{layer}: requantization must differ per channel, got {requant:?}"
        );
    }

    // 7. The input codes are the documented quantization of the source image.
    let scale = fixture.manifest.point_scales[0];
    for frame in &fixture.input.frames {
        let image = frame
            .image_u8
            .as_ref()
            .unwrap_or_else(|| panic!("frame {} has no source image", frame.index));
        let plane = quantize_image(image, scale, fixture.manifest.act_qmax as i64);
        assert_eq!(
            plane.codes, frame.codes,
            "frame {}: input codes must be `clamp(round_half_away(pixel/255/scale))`",
            frame.index
        );
    }

    // 8. The weight blob is contiguous and fully accounted for.
    let mut expected_offset = 0usize;
    for layer in &fixture.manifest.layers {
        assert_eq!(
            layer.weight_offset, expected_offset,
            "layer {}: weight offsets must be contiguous",
            layer.name
        );
        expected_offset += layer.weight_count;
    }
    assert_eq!(expected_offset, fixture.manifest.weight_bytes);
}

#[test]
fn per_tensor_fingerprints_are_stable_and_the_image_is_not_degenerate() {
    // A fixture whose output is a constant (or only reaches its clamp ends) can
    // hide a broken path: an output-only comparison would pass while nothing was
    // exercised. This test prints a per-tensor table (run with `--nocapture`) and
    // refuses to accept a degenerate image.
    for name in ["rounding", "graph"] {
        let fixture = Fixture::load(name);
        let frames = run_fixture(&fixture);
        let mut image_codes = Vec::new();
        for frame in &frames {
            println!("-- {name} frame {} --", frame.index);
            for op in &frame.ops {
                let codes = &op.plane.codes;
                println!(
                    "  {:<22} {:<3} {:?} min={:>4} max={:>4} sum={:>7} outside_i8={:>3}",
                    op.name,
                    op.elem,
                    op.plane.shape(),
                    codes.iter().min().expect("codes"),
                    codes.iter().max().expect("codes"),
                    codes.iter().sum::<i64>(),
                    op.plane.outside_i8()
                );
            }
            image_codes.extend(frame.image.codes.iter().copied());
        }
        let image_distinct = distinct(&image_codes);
        println!(
            "  image: {} codes, {image_distinct} distinct",
            image_codes.len()
        );
        if name == "graph" {
            // The whole-pipeline fixture stands or falls on its image.
            assert!(
                image_distinct >= 4,
                "{name}: the image must vary (got {image_distinct} distinct codes), otherwise an \
                 output-only comparison proves nothing"
            );
            assert!(
                image_codes.iter().any(|code| *code > 0 && *code < 127),
                "{name}: the image must contain an interior value, not only clamp ends"
            );
        } else {
            // The rounding probe is deliberately tiny; its discriminating tensor is
            // the signed intermediate, which must exercise both signs and the
            // half-way codes rather than a clamp alone.
            let mid: Vec<i64> = frames
                .iter()
                .flat_map(|frame| frame.ops[0].plane.codes.iter().copied())
                .collect();
            assert!(
                distinct(&mid) >= 4,
                "{name}: the signed intermediate must vary (got {} distinct)",
                distinct(&mid)
            );
            assert!(
                mid.iter().any(|code| *code < 0) && mid.iter().any(|code| *code > 0),
                "{name}: the signed intermediate must exercise both signs"
            );
        }
    }
}

#[test]
fn the_comparison_detects_a_perturbed_expected_code() {
    // A control: the comparison must not pass vacuously.
    let fixture = Fixture::load("rounding");
    let frames = run_fixture(&fixture);
    let good = &fixture.frame_vectors(0).ops[0];
    let perturbed = ExpectedOp {
        name: good.name.clone(),
        elem: good.elem.clone(),
        codes: good
            .codes
            .iter()
            .enumerate()
            .map(|(index, code)| if index == 5 { code + 1 } else { *code })
            .collect(),
    };
    let shape = (
        fixture.vectors.shape.c,
        fixture.vectors.shape.h,
        fixture.vectors.shape.w,
    );
    let error = compare_expected_op("control", &perturbed, &frames[0].ops[0], shape)
        .expect_err("a perturbed vector must be reported");
    assert!(
        error.contains("frozen vectors differ"),
        "the report must say what disagreed: {error}"
    );
    assert!(
        error.contains("[c1,y0,x1]"),
        "the report must name the coordinate: {error}"
    );
}

#[test]
fn requantization_matches_hand_computed_rounding_and_clamping() {
    // requant = 0.25 -> shift 18, M0 = 65536. A product of exactly half an LSB
    // must round away from zero in both directions.
    let (m0, shift) = serialize_requant(&[0.25]);
    assert_eq!((m0[0], shift), (65536, 18));
    assert_eq!(requantize(2, m0[0], shift, -127, 127), 1, "+0.5 rounds up");
    assert_eq!(
        requantize(-2, m0[0], shift, -127, 127),
        -1,
        "-0.5 rounds down"
    );
    assert_eq!(
        requantize(1, m0[0], shift, -127, 127),
        0,
        "+0.25 rounds to 0"
    );
    assert_eq!(
        requantize(-1, m0[0], shift, -127, 127),
        0,
        "-0.25 rounds to 0"
    );
    assert_eq!(
        requantize(3, m0[0], shift, -127, 127),
        1,
        "+0.75 rounds to 1"
    );
    assert_eq!(
        requantize(-3, m0[0], shift, -127, 127),
        -1,
        "-0.75 rounds to -1"
    );
    assert_eq!(
        requantize(2000, m0[0], shift, -127, 127),
        127,
        "saturates high"
    );
    assert_eq!(
        requantize(-2000, m0[0], shift, -127, 127),
        -127,
        "saturates low"
    );
    assert_eq!(
        requantize(2000, m0[0], shift, 0, 255),
        255,
        "unsigned clamp saturates high"
    );
    assert_eq!(
        requantize(-2000, m0[0], shift, 0, 255),
        0,
        "unsigned clamp saturates low"
    );

    // A shift of zero must be a plain truncation, not a rounding.
    let (unit, zero_shift) = serialize_requant(&[100000.0]);
    assert_eq!(zero_shift, 0);
    assert_eq!(unit[0], 100000);
    assert_eq!(
        requantize(7, unit[0], zero_shift, -1_000_000, 1_000_000),
        700_000
    );
    assert_eq!(
        requantize(-7, unit[0], zero_shift, -1_000_000, 1_000_000),
        -700_000
    );
    assert_eq!(requantize(-7, unit[0], zero_shift, -1, 1), -1, "clamped");
}

#[test]
fn the_fixture_directories_are_self_contained() {
    for name in ["rounding", "graph"] {
        let dir = fixture_dir(name);
        for file in [
            "manifest.json",
            "weights.bin",
            "input.json",
            "expected.json",
        ] {
            assert!(
                dir.join(file).exists(),
                "fixtures/acceptance/{name}/{file} is missing"
            );
        }
        let fixture = Fixture::load(name);
        assert_eq!(fixture.manifest.format, "fpga_st_train.int8.v1");
        assert_eq!(fixture.manifest.weight_file, "weights.bin");
        assert_eq!(fixture.manifest.weight_bytes, fixture.weights.len());
        assert!(
            !fixture.manifest.weight_file.contains('/')
                && !fixture.manifest.weight_file.contains('\\'),
            "{name}: the weight file must resolve inside the fixture directory"
        );
        // The input frames are the graph input codes, and each frame is a full
        // tensor: the shape and the code count must agree exactly.
        for frame in &fixture.input.frames {
            let expected =
                (fixture.input.shape.c * fixture.input.shape.h * fixture.input.shape.w) as usize;
            assert_eq!(
                frame.codes.len(),
                expected,
                "{name}: frame {} code count",
                frame.index
            );
        }
        assert_eq!(
            fixture.input.frames.len(),
            2,
            "{name}: two frames are driven"
        );
        assert_eq!(
            Plane::new(
                fixture.input.shape.c,
                fixture.input.shape.h,
                fixture.input.shape.w,
                fixture.input.frames[0].codes.clone()
            )
            .outside_i8(),
            0,
            "{name}: the graph input must fit the signed 8-bit domain"
        );
    }
}
