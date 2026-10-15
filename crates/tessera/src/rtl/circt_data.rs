//! Resolve solved tessera decisions into concrete weights/bias/M0 and evaluate
//! the integer datapath (same formula as [`super::reference`]).
//!
//! This is the functional contract the CIRCT hybrid must match bit-for-bit.

use std::collections::BTreeMap;

use crate::dialect::read::{OpKind as DopKind, OperatorOp, Source, TesseraModule};
use crate::graph::{Graph, LayerKind};
use crate::layout::{ElemType, TensorId, TensorRef, TensorShape};
use crate::lower::weight_tensor_name;
use crate::rtl::{RtlError, WeightResolver};

/// One operator with ROM contents and geometry ready for emission / evaluation.
#[derive(Debug, Clone)]
pub struct ResolvedOp {
    pub index: u32,
    pub name: String,
    pub kind: DopKind,
    pub ic: u32,
    pub oc: u32,
    pub ih: u32,
    pub iw: u32,
    pub oh: u32,
    pub ow: u32,
    pub kernel: [u32; 2],
    pub stride: [u32; 2],
    pub pad: u32,
    pub reflect: bool,
    pub groups: u32,
    /// Activation producer op indices; empty means graph input for first slot.
    pub act_sources: Vec<Option<u32>>,
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
    pub m0: Vec<i64>,
    pub shift: u32,
    pub clamp: [i64; 2],
}

impl ResolvedOp {
    pub fn in_words(&self) -> u64 {
        u64::from(self.ic) * u64::from(self.ih) * u64::from(self.iw)
    }
    pub fn out_words(&self) -> u64 {
        u64::from(self.oc) * u64::from(self.oh) * u64::from(self.ow)
    }
}

/// Resolve every operator against the graph artifact and weight blob.
pub fn resolve_ops(
    model: &TesseraModule,
    graph: &Graph,
    weights: &dyn WeightResolver,
) -> Result<Vec<ResolvedOp>, RtlError> {
    let mut out = Vec::with_capacity(model.operators.len());
    for op in &model.operators {
        out.push(resolve_one(op, model, graph, weights)?);
    }
    Ok(out)
}

fn resolve_one(
    op: &OperatorOp,
    model: &TesseraModule,
    graph: &Graph,
    weights: &dyn WeightResolver,
) -> Result<ResolvedOp, RtlError> {
    let out_shape = &op.output.shape;
    if out_shape.len() != 4 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!("op {} output rank {} (want NHWC)", op.name, out_shape.len()),
        });
    }
    let oh = out_shape[1] as u32;
    let ow = out_shape[2] as u32;
    let oc = out_shape[3] as u32;

    let act: Vec<_> = op.activations().collect();
    let (ic, ih, iw) = if let Some((_, ty)) = act.first() {
        if ty.shape.len() != 4 {
            return Err(RtlError::UnsupportedLayout {
                detail: format!("op {} activation rank", op.name),
            });
        }
        (ty.shape[3] as u32, ty.shape[1] as u32, ty.shape[2] as u32)
    } else {
        (oc, oh, ow)
    };

    let mut act_sources = Vec::new();
    for (src, _) in &act {
        act_sources.push(match src {
            Source::Arg(0) => None,
            Source::Op(i) => Some(*i),
            Source::Arg(n) => {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!("op {} reads weight arg {n} as activation", op.name),
                });
            }
        });
    }

    let (weights_i8, bias, m0, shift) = if op.kind.is_conv() {
        load_conv_params(op, graph, weights, ic, oc)?
    } else if matches!(op.kind, DopKind::Add) {
        let m0 = if op.multiplier.is_empty() {
            vec![1]
        } else {
            op.multiplier.clone()
        };
        (Vec::new(), Vec::new(), m0, op.shift as u32)
    } else {
        (Vec::new(), Vec::new(), Vec::new(), op.shift as u32)
    };

    let _ = model;
    Ok(ResolvedOp {
        index: op.index,
        name: op.name.clone(),
        kind: op.kind,
        ic,
        oc,
        ih,
        iw,
        oh,
        ow,
        kernel: [op.kernel[0] as u32, op.kernel[1] as u32],
        stride: [op.stride[0] as u32, op.stride[1] as u32],
        pad: op.pad as u32,
        reflect: op.reflect,
        groups: op.groups.max(1) as u32,
        act_sources,
        weights: weights_i8,
        bias,
        m0,
        shift,
        clamp: op.clamp,
    })
}

fn load_conv_params(
    op: &OperatorOp,
    graph: &Graph,
    weights: &dyn WeightResolver,
    ic: u32,
    oc: u32,
) -> Result<(Vec<i8>, Vec<i32>, Vec<i64>, u32), RtlError> {
    let layer = graph.layer_named(&op.name).ok_or_else(|| RtlError::UnsupportedLayout {
        detail: format!("no graph layer named {} for conv weights", op.name),
    })?;
    let (wn, wc, wh, ww) = match layer.kind {
        LayerKind::DepthwiseConv2d => (oc, 1u32, op.kernel[0] as u32, op.kernel[1] as u32),
        _ => (oc, ic / op.groups.max(1) as u32, op.kernel[0] as u32, op.kernel[1] as u32),
    };
    let tref = TensorRef {
        id: TensorId::new(0),
        name: weight_tensor_name(&op.name),
        shape: TensorShape::new(wn, wc, wh, ww),
        elem: ElemType::I8,
    };
    let raw = weights.tensor_bytes(&tref)?;
    let expected = (wn * wc * wh * ww) as usize;
    if raw.len() != expected {
        return Err(RtlError::ShapeMismatch {
            layer: op.name.clone(),
            what: format!("weight bytes for {}", tref.name),
            expected: expected as u64,
            actual: raw.len() as u64,
        });
    }
    let weights_i8: Vec<i8> = raw.iter().map(|b| *b as i8).collect();
    let mut bias = layer.bias_int32.clone();
    if bias.len() != oc as usize {
        bias.resize(oc as usize, 0);
    }
    let mut m0 = op.multiplier.clone();
    if m0.is_empty() {
        m0.push(1);
    }
    if m0.len() == 1 && oc > 1 {
        m0 = vec![m0[0]; oc as usize];
    }
    if m0.len() != oc as usize {
        return Err(RtlError::UnsupportedQuantization {
            layer: op.name.clone(),
            detail: format!("multiplier len {} != oc {oc}", m0.len()),
        });
    }
    Ok((weights_i8, bias, m0, op.shift as u32))
}

/// Evaluate one frame of integer codes (CHW-flat per tensor in op order).
///
/// Layout of each plane: `code[c * H * W + y * W + x]`. Returns the graph
/// output plane (last operator result).
pub fn evaluate_integer(
    ops: &[ResolvedOp],
    input: &[i64],
    results: &[u32],
) -> Result<Vec<i64>, RtlError> {
    let mut slots: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    // Graph input is keyed by sentinel.
    const INPUT: u32 = u32::MAX;
    slots.insert(INPUT, input.to_vec());

    for op in ops {
        let out = eval_op(op, &slots)?;
        slots.insert(op.index, out);
    }
    let out_idx = results.first().copied().ok_or_else(|| RtlError::UnsupportedLayout {
        detail: "tessera module has no result".into(),
    })?;
    slots
        .remove(&out_idx)
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!("missing result op #{out_idx}"),
        })
}

pub(crate) fn eval_op(op: &ResolvedOp, slots: &BTreeMap<u32, Vec<i64>>) -> Result<Vec<i64>, RtlError> {
    let fetch = |src: Option<u32>| -> Result<&[i64], RtlError> {
        let key = src.unwrap_or(u32::MAX);
        slots.get(&key).map(|v| v.as_slice()).ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!("op {} missing activation {:?}", op.name, src),
        })
    };

    let oc = op.oc as usize;
    let oh = op.oh as usize;
    let ow = op.ow as usize;
    let mut out = vec![0i64; oc * oh * ow];

    match op.kind {
        DopKind::Conv2d | DopKind::PointwiseConv2d => {
            let input = fetch(op.act_sources.first().copied().flatten())?;
            let icg = (op.ic / op.groups.max(1)) as usize;
            let kh = op.kernel[0] as usize;
            let kw = op.kernel[1] as usize;
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = i64::from(op.bias.get(channel).copied().unwrap_or(0));
                        for ic in 0..icg {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    if let Some(v) = sample(input, op, ic as u32, oy, ox, ky, kx) {
                                        let w = op.weights[((channel * icg + ic) * kh + ky) * kw + kx];
                                        acc += v * i64::from(w);
                                    }
                                }
                            }
                        }
                        out[(channel * oh + oy) * ow + ox] = requant(op, channel, acc)?;
                    }
                }
            }
        }
        DopKind::DepthwiseConv2d => {
            let input = fetch(op.act_sources.first().copied().flatten())?;
            let kh = op.kernel[0] as usize;
            let kw = op.kernel[1] as usize;
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = i64::from(op.bias.get(channel).copied().unwrap_or(0));
                        for ky in 0..kh {
                            for kx in 0..kw {
                                if let Some(v) = sample(input, op, channel as u32, oy, ox, ky, kx) {
                                    let w = op.weights[(channel * kh + ky) * kw + kx];
                                    acc += v * i64::from(w);
                                }
                            }
                        }
                        out[(channel * oh + oy) * ow + ox] = requant(op, channel, acc)?;
                    }
                }
            }
        }
        DopKind::Add => {
            let trunk = fetch(op.act_sources.first().copied().flatten())?;
            let branch = fetch(op.act_sources.get(1).copied().flatten())?;
            for i in 0..out.len() {
                let scaled = requant(op, 0, branch[i])?;
                out[i] = trunk[i] + scaled;
            }
        }
        DopKind::Upsample => {
            let input = fetch(op.act_sources.first().copied().flatten())?;
            let ih = op.ih as usize;
            let iw = op.iw as usize;
            for channel in 0..oc {
                for oy in 0..oh {
                    for ox in 0..ow {
                        out[(channel * oh + oy) * ow + ox] =
                            input[(channel * ih + oy / 2) * iw + ox / 2];
                    }
                }
            }
        }
        DopKind::Quantize | DopKind::Relu | DopKind::Requant => {
            let input = fetch(op.act_sources.first().copied().flatten())?;
            for i in 0..out.len() {
                out[i] = if !op.m0.is_empty() && op.m0[0] != 0 && op.kind != DopKind::Quantize {
                    requant(op, 0, input[i])?
                } else {
                    input[i].clamp(op.clamp[0], op.clamp[1])
                };
            }
        }
    }
    Ok(out)
}

fn requant(op: &ResolvedOp, channel: usize, acc: i64) -> Result<i64, RtlError> {
    let m0 = *op
        .m0
        .get(channel)
        .or_else(|| op.m0.first())
        .ok_or_else(|| RtlError::UnsupportedQuantization {
            layer: op.name.clone(),
            detail: format!("no M0 for channel {channel}"),
        })?;
    let shift = op.shift;
    let product = acc * m0;
    let rounding = if shift == 0 {
        0
    } else if product >= 0 {
        1i64 << (shift - 1)
    } else {
        (1i64 << (shift - 1)) - 1
    };
    let value = (product + rounding) >> shift;
    Ok(value.clamp(op.clamp[0], op.clamp[1]))
}

fn sample(
    input: &[i64],
    op: &ResolvedOp,
    ic: u32,
    oy: usize,
    ox: usize,
    ky: usize,
    kx: usize,
) -> Option<i64> {
    let ih = i64::from(op.ih);
    let iw = i64::from(op.iw);
    let pad = i64::from(op.pad);
    let y = oy as i64 * i64::from(op.stride[0]) + ky as i64 - pad;
    let x = ox as i64 * i64::from(op.stride[1]) + kx as i64 - pad;
    let (y, x) = if op.reflect {
        (reflect(y, ih), reflect(x, iw))
    } else {
        if y < 0 || y >= ih || x < 0 || x >= iw {
            return None;
        }
        (y, x)
    };
    Some(input[((i64::from(ic) * ih + y) * iw + x) as usize])
}

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

/// Build a deterministic stimulus frame for the graph input (CHW-flat).
pub fn stimulus_frame(ic: u32, ih: u32, iw: u32, seed: u64) -> Vec<i64> {
    let mut g = crate::rtl::Xorshift64::new(seed);
    let n = (ic * ih * iw) as usize;
    (0..n)
        .map(|_| i64::from(g.next_i8()).clamp(-127, 127))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{linalg_to_tessera, register_in_fresh_context, verify_module};
    use crate::frontend;
    use crate::lower::InputShape;
    use crate::mlir::{self, MlirOptions};
    use crate::planner::{Policy, ProvenanceInput};
    use crate::rtl::circt_ctx::load_circt_dialects;
    use crate::solver::MappingBounds;
    use crate::solver::dialect::solve_tessera;
    use crate::target::Ti60Target;
    use crate::weights::GraphWeightResolver;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
    }

    #[test]
    fn w8_integer_reference_runs_and_is_deterministic() {
        let shape = InputShape::new(3, 6, 6).expect("shape");
        let artifact = frontend::import_manifest(&fixtures().join("w8")).expect("import");
        let linalg = mlir::emit_text(artifact.graph(), &MlirOptions::new(shape.to_nhwc())).expect("emit");
        let context = register_in_fresh_context().expect("dialect");
        load_circt_dialects(&context);
        let module = linalg_to_tessera(&context, &linalg).expect("convert");
        let target = Ti60Target::from_toml_file(fixtures().join("targets/ti60f225.toml")).expect("t");
        let policy = Policy::from_toml_file(fixtures().join("policies/explore.toml")).expect("p");
        let provenance = ProvenanceInput {
            graph_sha256: artifact.provenance().manifest_sha256.clone(),
            target_sha256: "t".into(),
            policy_sha256: "p".into(),
            compiler_version: env!("CARGO_PKG_VERSION").into(),
        };
        let bounds = MappingBounds::hybrid_for(&target, &policy).expect("bounds");
        solve_tessera(&context, &module, &target, &policy, &provenance, &bounds).expect("solve");
        let model = verify_module(&module).expect("verify");
        let resolver = GraphWeightResolver::load(&artifact, fixtures().join("w8")).expect("weights");
        let ops = resolve_ops(&model, artifact.graph(), &resolver).expect("resolve");
        assert_eq!(ops.len(), 7, "w8 op count");
        let input = stimulus_frame(3, 6, 6, 0x5eed_0001);
        let out = evaluate_integer(&ops, &input, &model.results).expect("eval");
        assert_eq!(out.len(), 12 * 12 * 3);
        let out2 = evaluate_integer(&ops, &input, &model.results).expect("eval2");
        assert_eq!(out, out2);
        let nz: usize = out.iter().filter(|c| **c != 0).count();
        eprintln!("w8 ref: {} ops, nz={}/{}, out[0..8]={:?}", ops.len(), nz, out.len(), &out[..8]);
        assert!(ops.iter().any(|o| !o.weights.is_empty()), "conv weights resolved");
    }
}
