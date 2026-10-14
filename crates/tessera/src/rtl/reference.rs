//! Scalar reference for the full-graph backend.
//!
//! Two evaluations live here, and they are deliberately different things:
//!
//! * [`run_integer`] evaluates the exact datapath the emitted RTL implements:
//!   `i64` accumulation, then `(acc*M0 + round) >>> shift` with the
//!   sign-dependent rounding offset, then the step's saturation clamp. This is
//!   what the generated testbench compares against, bit for bit.
//! * [`run_f32`] evaluates the *manifest* semantics
//!   (`round_clamp(acc as f32 * requant)`) that `intref.rs` defines, using the
//!   graph layer's `f32` requantization coefficients. It exists only to report
//!   how far the fixed-point datapath is from the training-time model;
//!   [`divergence`] quantifies it per step. No stage claims the two agree bit
//!   for bit: the difference is the documented `M0`/`shift` approximation.
//!
//! Both are written directly from the resolved model and share no arithmetic
//! with the emitter (which produces SystemVerilog text), so a match between a
//! simulation and this reference is independent evidence, not an echo.

use super::RtlError;
use super::full::{Border, CodeDomain, FullGraphModel, ResolvedStep, Shape3};

/// One step's codes, in `(channel, height, width)` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepRun {
    /// Schedule index.
    pub index: usize,
    /// Operator name.
    pub name: String,
    /// Output shape.
    pub shape: Shape3,
    /// Code domain of the stored values.
    pub domain: CodeDomain,
    /// Stored codes.
    pub codes: Vec<i64>,
}

/// One frame's evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRun {
    /// Frame index.
    pub index: usize,
    /// Graph input codes as stored in the arena.
    pub input: Vec<i64>,
    /// Per-step outputs, in schedule order.
    pub steps: Vec<StepRun>,
    /// Unstalled cycle count of the generated FSM for one frame.
    pub cycles: u64,
}

/// A complete reference run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceRun {
    /// Frames, in order.
    pub frames: Vec<FrameRun>,
    /// Steps whose `f32` reference could not be computed (no coefficients in
    /// the graph artifact); [`run_f32`] reuses the integer result for them.
    pub f32_unavailable: Vec<String>,
}

/// Per-step distance between the integer datapath and the manifest `f32` one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepDivergence {
    /// Schedule index.
    pub index: usize,
    /// Operator name.
    pub name: String,
    /// Codes that differ.
    pub differing: usize,
    /// Codes compared.
    pub total: usize,
    /// Largest absolute difference.
    pub max_abs_diff: i64,
}

/// Evaluate the datapath the RTL implements.
pub fn run_integer(model: &FullGraphModel, frames: &[Vec<i64>]) -> Result<ReferenceRun, RtlError> {
    run(model, frames, false)
}

/// Evaluate the manifest `f32` semantics (`intref.rs`), for divergence evidence.
pub fn run_f32(model: &FullGraphModel, frames: &[Vec<i64>]) -> Result<ReferenceRun, RtlError> {
    run(model, frames, true)
}

fn run(model: &FullGraphModel, frames: &[Vec<i64>], float: bool) -> Result<ReferenceRun, RtlError> {
    let mut out = Vec::with_capacity(frames.len());
    let mut unavailable: Vec<String> = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        let mut slots: Vec<Vec<i64>> = model
            .slots
            .iter()
            .map(|slot| vec![0i64; slot.shape.elements() as usize])
            .collect();
        if frame.len() != slots[model.input_slot].len() {
            return Err(RtlError::ShapeMismatch {
                layer: model.slots[model.input_slot].name.clone(),
                what: "stimulus frame".to_string(),
                expected: slots[model.input_slot].len() as u64,
                actual: frame.len() as u64,
            });
        }
        slots[model.input_slot] = frame.clone();
        let mut steps = Vec::with_capacity(model.steps.len());
        for step in &model.steps {
            let codes = eval_step(step, &mut slots, float, &mut unavailable)?;
            steps.push(StepRun {
                index: step.index,
                name: step.name.clone(),
                shape: model.slots[step.output].shape,
                domain: model.slots[step.output].domain,
                codes,
            });
        }
        out.push(FrameRun {
            index,
            input: frame.clone(),
            steps,
            cycles: model.cycles_unstalled,
        });
    }
    unavailable.sort();
    unavailable.dedup();
    Ok(ReferenceRun {
        frames: out,
        f32_unavailable: unavailable,
    })
}

fn eval_step(
    step: &ResolvedStep,
    slots: &mut Vec<Vec<i64>>,
    float: bool,
    unavailable: &mut Vec<String>,
) -> Result<Vec<i64>, RtlError> {
    let out_shape = step.out_shape;
    let oh = out_shape.h as usize;
    let ow = out_shape.w as usize;
    let oc = out_shape.c as usize;
    let mut out = vec![0i64; out_shape.elements() as usize];

    match step.op_kind {
        crate::layout::OpKind::Conv2d | crate::layout::OpKind::PointwiseConv2d => {
            let input = slots[step.inputs[0]].clone();
            let kh = step.kernel[0] as usize;
            let kw = step.kernel[1] as usize;
            let icg = step.in_per_group as usize;
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias_of(step, channel);
                        for ic in 0..icg {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    if let Some(value) =
                                        sample(&input, step, ic as u32, oy, ox, ky, kx)
                                    {
                                        acc += value
                                            * i64::from(
                                                step.weights
                                                    [((channel * icg + ic) * kh + ky) * kw + kx],
                                            );
                                    }
                                }
                            }
                        }
                        out[(channel * oh + oy) * ow + ox] =
                            requant(step, channel, acc, float, unavailable)?;
                    }
                }
            }
        }
        crate::layout::OpKind::DepthwiseConv2d => {
            let input = slots[step.inputs[0]].clone();
            let kh = step.kernel[0] as usize;
            let kw = step.kernel[1] as usize;
            // Depthwise weights are `(out_channel, 1, kh, kw)` and each group owns
            // exactly one input channel, so the group index is the channel index.
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias_of(step, channel);
                        for ky in 0..kh {
                            for kx in 0..kw {
                                if let Some(value) =
                                    sample(&input, step, channel as u32, oy, ox, ky, kx)
                                {
                                    acc += value
                                        * i64::from(step.weights[(channel * kh + ky) * kw + kx]);
                                }
                            }
                        }
                        out[(channel * oh + oy) * ow + ox] =
                            requant(step, channel, acc, float, unavailable)?;
                    }
                }
            }
        }
        crate::layout::OpKind::Add => {
            let trunk = slots[step.inputs[0]].clone();
            let branch = slots[step.inputs[1]].clone();
            for index in 0..out.len() {
                let scaled = requant(step, 0, branch[index], float, unavailable)?;
                // The branch is requantized and clamped; the sum is NOT clamped
                // (`intref.rs:257-268`), so the code can leave the i8 domain.
                out[index] = trunk[index] + scaled;
            }
        }
        crate::layout::OpKind::NearestUpsample2x => {
            let input = slots[step.inputs[0]].clone();
            let ih = step.in_shape.h as usize;
            let iw = step.in_shape.w as usize;
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        out[(channel * oh + oy) * ow + ox] =
                            input[(channel * ih + oy / 2) * iw + ox / 2];
                    }
                }
            }
        }
        _ => {
            let input = slots[step.inputs[0]].clone();
            let has_requant = !step.requant.is_empty();
            for index in 0..out.len() {
                out[index] = if has_requant {
                    requant(step, 0, input[index], float, unavailable)?
                } else {
                    input[index].clamp(step.clamp.0, step.clamp.1)
                };
            }
        }
    }
    slots[step.output] = out.clone();
    Ok(out)
}

fn bias_of(step: &ResolvedStep, channel: usize) -> i64 {
    i64::from(step.bias.get(channel).copied().unwrap_or(0))
}

/// One convolution input sample, applying the border rule.
fn sample(
    input: &[i64],
    step: &ResolvedStep,
    ic: u32,
    oy: usize,
    ox: usize,
    ky: usize,
    kx: usize,
) -> Option<i64> {
    let ih = i64::from(step.in_shape.h);
    let iw = i64::from(step.in_shape.w);
    let pad = i64::from(step.border.amount());
    let y = oy as i64 * i64::from(step.stride[0]) + ky as i64 - pad;
    let x = ox as i64 * i64::from(step.stride[1]) + kx as i64 - pad;
    let (y, x) = match step.border {
        Border::Reflect { .. } => (reflect(y, ih), reflect(x, iw)),
        Border::Zero { .. } => {
            if y < 0 || y >= ih || x < 0 || x >= iw {
                return None;
            }
            (y, x)
        }
    };
    Some(input[((i64::from(ic) * ih + y) * iw + x) as usize])
}

/// Reflection that does not repeat the edge, matching `intref.rs::reflect_index`.
fn reflect(mut index: i64, extent: i64) -> i64 {
    if extent <= 1 {
        return 0;
    }
    while index < 0 || index >= extent {
        index = if index < 0 {
            -index
        } else {
            2 * (extent - 1) - index
        };
    }
    index
}

/// Round half away from zero on the real line, as `intref.rs::round_clamp` does.
fn round_half_away(value: f32) -> f32 {
    value.round()
}

/// Requantize one accumulator, then saturate.
///
/// `float` selects the manifest `f32` path; the integer path is the emitted
/// datapath, `(acc*M0 + round) >>> shift` with `round = 2^(sh-1)` for
/// non-negative products and `2^(sh-1) - 1` otherwise.
fn requant(
    step: &ResolvedStep,
    channel: usize,
    acc: i64,
    float: bool,
    unavailable: &mut Vec<String>,
) -> Result<i64, RtlError> {
    let (lo, hi) = step.clamp;
    let value = if float && step.requant_f32.len() > channel {
        let coefficient = step.requant_f32[channel];
        round_half_away(acc as f32 * coefficient) as i64
    } else {
        if float {
            unavailable.push(format!("{} (channel {channel})", step.name));
        }
        let entry = step
            .requant
            .get(channel)
            .or_else(|| step.requant.first())
            .ok_or_else(|| RtlError::UnsupportedQuantization {
                layer: step.name.clone(),
                detail: format!("no requantization record for channel {channel}"),
            })?;
        let multiplier = i64::from(entry.multiplier);
        let shift = u32::from(entry.shift);
        let product = acc * multiplier;
        let rounding = if shift == 0 {
            0
        } else if product >= 0 {
            1i64 << (shift - 1)
        } else {
            (1i64 << (shift - 1)) - 1
        };
        (product + rounding) >> shift
    };
    Ok(value.clamp(lo, hi))
}

/// Per-step distance between the integer datapath and the manifest `f32` model.
///
/// Only the first frame is compared: the divergence is a property of the
/// datapath, not of the stimulus.
pub fn divergence(integer: &ReferenceRun, float: &ReferenceRun) -> Vec<StepDivergence> {
    let mut out = Vec::new();
    let Some(frame) = integer.frames.first() else {
        return out;
    };
    let Some(other) = float.frames.first() else {
        return out;
    };
    for (step, reference) in frame.steps.iter().zip(other.steps.iter()) {
        if step.codes.len() != reference.codes.len() {
            continue;
        }
        let mut differing = 0usize;
        let mut max_abs_diff = 0i64;
        for (a, b) in step.codes.iter().zip(reference.codes.iter()) {
            if a != b {
                differing += 1;
                max_abs_diff = max_abs_diff.max((a - b).abs());
            }
        }
        if differing > 0 {
            out.push(StepDivergence {
                index: step.index,
                name: step.name.clone(),
                differing,
                total: step.codes.len(),
                max_abs_diff,
            });
        }
    }
    out
}
