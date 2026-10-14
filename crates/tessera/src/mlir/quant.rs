//! Serialization of the manifest quantization contract into integer MLIR form.
//!
//! The manifest (`fpga_st_train.int8.v1`) stores a per-output-channel `f32`
//! requantization factor. Hardware does not multiply by `f32`; both the existing
//! `manifest_to_nngen.py` frontend and the verified MLIR experiment serialize the
//! factor as a fixed-point multiplier and shift:
//!
//! ```text
//! requant[c] ~= M0[c] * 2^-shift     with 0 <= M0[c] <= 2^17 - 1
//! ```
//!
//! `M0` is rounded half away from zero and the shift is the largest value that
//! keeps every channel inside the 17-bit multiplier budget
//! (`nngen/examples/st_manifest/manifest_to_nngen.py:37-49`). This module owns
//! that conversion so the emitter never invents quantization parameters
//! implicitly: a layer either carries an already-serialized
//! [`RequantSpec`] or the frontend hands over the manifest `f32` vector and this
//! module serializes it, reporting the worst per-channel relative error.
//!
//! The emitted integer datapath is the one verified in
//! `nngen/tools/mlir_optimizer`: widen the accumulator to `i64`, multiply by
//! `M0`, add `2^(shift-1)` for non-negative products and `2^(shift-1) - 1`
//! otherwise, arithmetic-shift right by `shift`, truncate, then clamp. That is
//! round-half-away-from-zero, matching `intref.rs::round_clamp` modulo the
//! documented `M0`/`shift` quantization error.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Largest multiplier the 17-bit hardware multiplier accepts.
pub const MAX_MULTIPLIER: i64 = (1 << 17) - 1;

/// Largest shift the serialized form accepts (`manifest_to_nngen.py:37`).
pub const MAX_SHIFT: u32 = 31;

/// Rounding used by every requantization step in the emitted datapath.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundingMode {
    /// `round()` semantics of the reference: half away from zero.
    HalfAwayFromZero,
}

impl RoundingMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            RoundingMode::HalfAwayFromZero => "half_away_from_zero",
        }
    }
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Saturation regime applied after the requantization shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampKind {
    /// `[-act_qmax, act_qmax]`: a plain symmetric quantization point.
    Symmetric,
    /// `[0, act_qmax]`: `conv2d+relu`; the clamp *is* the activation
    /// (`intref.rs:341-349`).
    Relu,
    /// `[0, output_qmax]` (255): the unsigned image head. Codes are stored in a
    /// signless `i8` tensor with the identical bit pattern, i.e. the consumer
    /// reinterprets the bytes as `u8`.
    UnsignedOutput,
}

impl ClampKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            ClampKind::Symmetric => "symmetric",
            ClampKind::Relu => "relu",
            ClampKind::UnsignedOutput => "unsigned_output",
        }
    }
}

/// Explicit saturation bounds carried next to the emitted MLIR.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClampSpec {
    pub kind: ClampKind,
    pub lower: i64,
    pub upper: i64,
    /// `true` when the codes are unsigned `u8` values stored in a signless
    /// `i8` tensor.
    pub unsigned: bool,
}

impl ClampSpec {
    /// Symmetric point: `[-qmax, qmax]`.
    pub fn symmetric(act_qmax: f32) -> Self {
        let bound = act_qmax as i64;
        ClampSpec {
            kind: ClampKind::Symmetric,
            lower: -bound,
            upper: bound,
            unsigned: false,
        }
    }

    /// `conv2d+relu` point: `[0, qmax]`.
    pub fn relu(act_qmax: f32) -> Self {
        ClampSpec {
            kind: ClampKind::Relu,
            lower: 0,
            upper: act_qmax as i64,
            unsigned: false,
        }
    }

    /// Unsigned image head: `[0, output_qmax]`.
    pub fn unsigned_output(output_qmax: f32) -> Self {
        ClampSpec {
            kind: ClampKind::UnsignedOutput,
            lower: 0,
            upper: output_qmax as i64,
            unsigned: true,
        }
    }

    /// Lower bound for the branch operand of an `add`, which is requantized onto
    /// the destination grid before being added (`intref.rs:263`).
    pub fn symmetric_branch(act_qmax: f32) -> Self {
        let bound = act_qmax as i64;
        ClampSpec {
            kind: ClampKind::Symmetric,
            lower: -bound,
            upper: bound,
            unsigned: false,
        }
    }
}

/// Fixed-point requantization parameters for one operator.
///
/// `multiplier` has one entry per output channel in the operator's output
/// channel order; `shift` is uniform per operator so the emitted epilogue can
/// keep a single constant shift, exactly like the verified fixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequantSpec {
    pub multiplier: Vec<i32>,
    pub shift: u32,
    pub rounding: RoundingMode,
    pub clamp: ClampSpec,
}

impl RequantSpec {
    /// Rounding offset added to a non-negative product: `2^(shift - 1)`.
    ///
    /// The reference adds `2^(shift-1) - 1` for negative products
    /// (see [`RequantSpec::round_negative`]), so that the arithmetic right shift
    /// after the addition rounds half away from zero. A degenerate `shift == 0`
    /// performs no scaling and therefore needs no rounding offset.
    pub fn round_positive(&self) -> i64 {
        rounding_offsets(self.shift).0
    }

    /// Rounding offset added to a negative product.
    pub fn round_negative(&self) -> i64 {
        rounding_offsets(self.shift).1
    }

    /// Number of per-channel multipliers, i.e. output channels.
    pub fn channels(&self) -> usize {
        self.multiplier.len()
    }
}

/// Failure of the `f32` -> `(M0, shift)` serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// A requantization factor was zero, negative, or not finite. Symmetric
    /// quantization only produces strictly positive factors.
    NonPositiveFactor { channel: usize, raw: String },
    /// The channel count did not match the operator's output channels.
    ChannelCount { expected: u32, found: usize },
    /// The requested multiplier or shift cannot be represented.
    Overflow { detail: String },
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::NonPositiveFactor { channel, raw } => write!(
                f,
                "requantization factor for output channel {channel} is not a finite positive number ({raw})"
            ),
            QuantError::ChannelCount { expected, found } => write!(
                f,
                "requantization factor count {found} does not match {expected} output channels"
            ),
            QuantError::Overflow { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Serialize a per-channel `f32` requantization vector exactly like
/// `manifest_to_nngen.py:pick_shift` / `requant_to_m0_shift`.
///
/// `channel_count` is the operator's output channel count; `clamp` selects the
/// saturation regime. The returned spec also reports the worst per-channel
/// relative error of the fixed-point approximation so callers can record it in
/// the artifact instead of assuming exactness.
pub fn serialize_requant(
    requant: &[f32],
    channel_count: u32,
    clamp: ClampSpec,
) -> Result<(RequantSpec, f64), QuantError> {
    if requant.len() != channel_count as usize {
        return Err(QuantError::ChannelCount {
            expected: channel_count,
            found: requant.len(),
        });
    }
    if requant.is_empty() {
        return Err(QuantError::Overflow {
            detail: "requantization vector is empty".to_string(),
        });
    }

    let mut max_abs = 0.0f64;
    for (channel, raw) in requant.iter().enumerate() {
        let value = *raw as f64;
        if !value.is_finite() || value <= 0.0 {
            return Err(QuantError::NonPositiveFactor {
                channel,
                raw: format!("{raw}"),
            });
        }
        max_abs = max_abs.max(value.abs());
    }

    // pick_shift: floor(log2(max_m0 / max_requant)) clamped to [0, 31].
    let shift = if max_abs <= 0.0 {
        0
    } else {
        let ratio = MAX_MULTIPLIER as f64 / max_abs;
        let candidate = ratio.log2().floor() as i64;
        candidate.clamp(0, MAX_SHIFT as i64) as u32
    };

    let scale = (1i64 << shift) as f64;
    let mut multiplier = Vec::with_capacity(requant.len());
    let mut worst_relative_error = 0.0f64;
    for (channel, raw) in requant.iter().enumerate() {
        let scaled = *raw as f64 * scale;
        let rounded = round_half_away(scaled);
        if !rounded.is_finite() || rounded < 0.0 || rounded > MAX_MULTIPLIER as f64 {
            return Err(QuantError::Overflow {
                detail: format!(
                    "requantization multiplier for output channel {channel} is {rounded}, outside [0, {MAX_MULTIPLIER}] at shift {shift}"
                ),
            });
        }
        let m0 = rounded as i64;
        let approx = m0 as f64 / scale;
        let relative = (approx - *raw as f64).abs() / (*raw as f64).max(f64::MIN_POSITIVE);
        worst_relative_error = worst_relative_error.max(relative);
        multiplier.push(m0 as i32);
    }

    Ok((
        RequantSpec {
            multiplier,
            shift,
            rounding: RoundingMode::HalfAwayFromZero,
            clamp,
        },
        worst_relative_error,
    ))
}

/// Serialize a single scalar factor, as used by the elementwise `add` branch
/// (`intref.rs:261-263` multiplies by `point_scales[src] / point_scales[dst]`).
pub fn serialize_scalar_requant(
    factor: f32,
    clamp: ClampSpec,
) -> Result<(RequantSpec, f64), QuantError> {
    serialize_requant(&[factor], 1, clamp)
}

/// Rounding offsets added before the arithmetic right shift:
/// `(2^(shift - 1), 2^(shift - 1) - 1)`.
///
/// Adding the larger offset to a non-negative product and the smaller one to a
/// negative product makes `shrsi` round half away from zero, matching
/// `intref.rs::round_clamp`. A degenerate `shift == 0` needs no offset.
pub const fn rounding_offsets(shift: u32) -> (i64, i64) {
    if shift == 0 {
        (0, 0)
    } else {
        let positive = 1i64 << (shift - 1);
        (positive, positive - 1)
    }
}

/// `round()` half away from zero, matching `intref.rs::round_clamp`.
pub fn round_half_away(value: f64) -> f64 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        -((-value + 0.5).floor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_shift_respects_multiplier_budget() {
        let requant = [0.0012943891f32, 0.0014688311, 0.0013799476, 0.0012818772];
        let (spec, error) = serialize_requant(&requant, 4, ClampSpec::relu(127.0)).unwrap();
        assert_eq!(spec.rounding, RoundingMode::HalfAwayFromZero);
        assert!(spec.multiplier.iter().all(|m| *m >= 0));
        assert!(
            spec.multiplier
                .iter()
                .all(|m| (*m as i64) <= MAX_MULTIPLIER)
        );
        assert!(spec.shift <= MAX_SHIFT);
        assert!(error < 1e-4, "relative error {error} too large");
    }

    #[test]
    fn shift_matches_reference_recipe_for_known_layer() {
        // `enc.0` of the committed W8A8 ink manifest has max requant
        // 0.0015629381, so pick_shift selects 26: 2^26 * 0.0015629381 =
        // 104 908 < 2^17 - 1 and 2^27 * 0.0015629381 exceeds the budget.
        let (spec, _) = serialize_requant(&[0.0015629381], 1, ClampSpec::relu(127.0)).unwrap();
        assert_eq!(spec.shift, 26);
        assert_eq!(spec.multiplier, vec![104_887]);
        assert_eq!(spec.round_positive(), 1 << 25);
        assert_eq!(spec.round_negative(), (1 << 25) - 1);
    }

    #[test]
    fn rejects_non_positive_and_mismatched_lengths() {
        assert!(matches!(
            serialize_requant(&[0.0], 1, ClampSpec::relu(127.0)),
            Err(QuantError::NonPositiveFactor { .. })
        ));
        assert!(matches!(
            serialize_requant(&[-1.0], 1, ClampSpec::relu(127.0)),
            Err(QuantError::NonPositiveFactor { .. })
        ));
        assert!(matches!(
            serialize_requant(&[0.5, 0.5], 1, ClampSpec::relu(127.0)),
            Err(QuantError::ChannelCount { .. })
        ));
        assert!(matches!(
            serialize_requant(&[], 0, ClampSpec::relu(127.0)),
            Err(QuantError::Overflow { .. })
        ));
    }

    #[test]
    fn round_half_away_matches_reference() {
        assert_eq!(round_half_away(2.5), 3.0);
        assert_eq!(round_half_away(-2.5), -3.0);
        assert_eq!(round_half_away(2.4), 2.0);
        assert_eq!(round_half_away(-0.5), -1.0);
    }

    /// The emitted integer datapath must implement the reference
    /// round-half-away-from-zero, not a truncation.
    ///
    /// `factor` is chosen exactly representable (`M0 * 2^-shift`), so the
    /// reference `f32` product is exact and both paths must agree bit for bit.
    #[test]
    fn integer_requantization_matches_the_reference_rounding() {
        let factor = 0.5f32;
        let (spec, error) = serialize_requant(&[factor], 1, ClampSpec::symmetric(127.0)).unwrap();
        let m0 = spec.multiplier[0] as i64;
        let shift = spec.shift;
        assert_eq!(error, 0.0, "0.5 is exactly representable");
        assert_eq!(m0 as f64 / (1u64 << shift) as f64, factor as f64);
        let (round_positive, round_negative) = rounding_offsets(shift);

        let reference = |acc: i64| -> i64 {
            let value = (acc as f32) * factor;
            let rounded = if value >= 0.0 {
                (value as f64 + 0.5).floor()
            } else {
                -((-value as f64 + 0.5).floor())
            };
            rounded.clamp(-127.0, 127.0) as i64
        };
        let emitted = |acc: i64| -> i64 {
            let product = acc * m0;
            let rounding = if product >= 0 {
                round_positive
            } else {
                round_negative
            };
            let scaled = (product + rounding) >> shift;
            scaled.clamp(-127, 127)
        };

        for acc in (-4096..4096).step_by(7) {
            assert_eq!(emitted(acc), reference(acc), "acc={acc}");
        }
        // Exactly representable ties: the whole point of the extra rounding term.
        for half in [-255i64, -127, -3, -1, 1, 3, 127, 255] {
            let acc = half * 2;
            assert_eq!(emitted(acc), reference(acc), "tie acc={acc}");
        }
    }

    #[test]
    fn clamp_regimes_match_reference_bounds() {
        let symmetric = ClampSpec::symmetric(127.0);
        assert_eq!((symmetric.lower, symmetric.upper), (-127, 127));
        let relu = ClampSpec::relu(127.0);
        assert_eq!((relu.lower, relu.upper), (0, 127));
        let head = ClampSpec::unsigned_output(255.0);
        assert_eq!((head.lower, head.upper), (0, 255));
        assert!(head.unsigned);
    }
}
