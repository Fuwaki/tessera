//! Legal spatial-shape candidates for one operator.
//!
//! A candidate is a real fold the backend can instantiate for the operator: the
//! SIMD factor must divide the operator's input channels, the PE factor must
//! divide its output channels, and this backend's MAC array has no tap
//! parallelism (`tap_parallel == 1`). The lane product must lie inside the
//! caller's per-engine lane bounds, and the shape must be one of the shapes the
//! caller enumerated.

use crate::solver::{Candidate, MappingBounds, SpatialShape};

/// Enumerate the legal fold candidates for one operator.
///
/// `in_channels`/`out_channels` are the operator's own channel counts,
/// `macs` its multiply-accumulate product count for the whole frame. The result
/// is sorted and free of duplicates, so the encoding is deterministic.
pub fn candidates_for(
    in_channels: u32,
    out_channels: u32,
    macs: u64,
    bounds: &MappingBounds,
) -> Vec<Candidate> {
    let mut shapes: Vec<SpatialShape> = bounds.shapes.clone();
    if shapes.is_empty() {
        // No enumeration given: derive every divisor product this backend can
        // instantiate, so an empty shape list means "anything legal".
        for simd in divisors(in_channels) {
            for pe in divisors(out_channels) {
                shapes.push(SpatialShape {
                    simd,
                    pe,
                    tap_parallel: 1,
                });
            }
        }
    }
    shapes.sort_by_key(|s| (s.simd, s.pe, s.tap_parallel));
    shapes.dedup();

    let mut out = Vec::new();
    for shape in shapes {
        if shape.simd == 0 || shape.pe == 0 || shape.tap_parallel != 1 {
            continue;
        }
        if in_channels % shape.simd != 0 || out_channels % shape.pe != 0 {
            continue;
        }
        let lanes = shape.lanes();
        if lanes < u64::from(bounds.min_lanes_per_engine)
            || lanes > u64::from(bounds.max_lanes_per_engine)
        {
            continue;
        }
        out.push(Candidate {
            shape,
            lanes,
            temporal_iterations: macs.div_ceil(lanes.max(1)),
        });
    }
    out.sort_by_key(|c| (c.shape.simd, c.shape.pe, c.shape.tap_parallel));
    out.dedup();
    out
}

/// Positive divisors of `n`, ascending. Zero yields an empty list.
fn divisors(n: u32) -> Vec<u32> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut d = 1u32;
    while d <= n {
        if n % d == 0 {
            out.push(d);
        }
        d += 1;
    }
    out
}
