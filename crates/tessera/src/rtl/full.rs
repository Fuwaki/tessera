//! Full-graph SystemVerilog backend: one RAM-backed, statically scheduled engine.
//!
//! # What this backend emits
//!
//! [`lower_full_graph`] consumes a validated planner [`ExecutionPlan`] plus a
//! [`FullGraphModel`] (the plan bound to the graph it was lowered from) and
//! emits a **single synthesizable SystemVerilog module that computes the whole
//! graph**: a static schedule FSM over the plan's `ScheduleStep`s, exactly *one*
//! shared MAC multiplier, *one* shared requantization multiplier and *one*
//! shared activation arena, with ROM-resident weights, bias and requantization
//! constants. There is no CPU, no instruction fetch and no per-layer dispatch:
//! the schedule is the generator, and each step's kernel is a generated element
//! stream with typed counters feeding muxed address/control logic.
//!
//! ```text
//! clk, rst, start                 frame start (pulse), one frame at a time
//! in_valid / in_ready             graph input codes, one code (i8) per beat
//! out_valid / out_ready           final image codes, one code (u8) per beat
//! done                            held high once the frame's output is accepted
//! dbg_en / dbg_addr / dbg_data    shared read port, serviced while idle
//! dbg_hold_en / dbg_cont / dbg_held   verification single-step (no schedule effect)
//! ```
//!
//! # Emitted structure (the thing the planner's cost must describe)
//!
//! * **One activation arena** `act_mem`: `NW` signed 16-bit words with *one*
//!   synchronous read port and *one* synchronous write port (a simple dual-port
//!   RAM cell), addressed by the schedule. Words are 16 bits because the
//!   residual `add` is *not* clamped (`intref.rs:257-268`):
//!   `dst = clamp_sym(round_half_away(branch * M0 >> shift)) + trunk`, so codes
//!   leave the i8 domain (9-bit sums) and the following upsample/head consume
//!   them unchanged.
//! * **One shared MAC multiplier** `prod = mem_rdata * w_data` and **one shared
//!   accumulator**, used by every convolution step of the schedule. Steps differ
//!   only in the muxed address/control they present to the shared engine, so the
//!   emitted file contains one multiplier, not one per convolution.
//! * **One shared requantization multiplier** behind the single store pipeline
//!   (`rq`), shared by every step's epilogue.
//! * **Three packed constant ROMs**: `w_rom` (one i8 code per entry, i4
//!   unpacked at generation time), `b_rom` (`bias_int32` per output channel) and
//!   `m0_rom` (the `17`-bit requantization multiplier per output channel). Each
//!   is one physical memory for the whole model with per-layer offsets recorded
//!   as generation-time constants: no per-layer BRAM and no per-layer rounding.
//! * **Arena reuse by tensor lifetime**: a tensor occupies arena words from the
//!   step that produces it to the last step that reads it, and a dead tensor's
//!   words are reused. The planner budgets the same lifetime peak
//!   (`exact_activation_blocks`), so a sum-of-all-buffers arena would be exactly
//!   the "budget ignored" mismatch this module refuses.
//!
//! # Timed datapath (read/MAC/epilogue/store)
//!
//! Every read is synchronous (one cycle of latency). The address stage of cycle
//! `n` presents the element `n`; the MAC stage of cycle `n+1` consumes it, so a
//! product is accumulated every cycle while the next operand is being fetched:
//!
//! ```text
//! cycle 1        issue element 0 address            (pipeline fill)
//! cycle n+1      MAC element n-1, issue element n   (accumulate, epilogue+store
//!                                                    on the group's last element)
//! cycle N+1      MAC element N-1 (last), epilogue/store of the last group
//! cycle N+2      store drain, step boundary
//! ```
//!
//! * dense/depthwise convolution: `read_beats = MACs`, one epilogue and one
//!   store beat per output code (overlapped), `total = MACs + 2`;
//! * one-read elementwise step (requant, activation, upsample, quantize):
//!   `read_beats = out_elements`, `total = out_elements + 2`;
//! * residual `add`: two operands on one read port, so `read_beats = 2 *
//!   out_elements` and `total = 2 * out_elements + 2`;
//! * frame: `load_beats (NI) + Σ step beats + drain_beats (2 * NO)`.
//!
//! The store pipeline is one deep: the epilogue is computed from registers
//! (`acc`/`mem_rdata`, the captured multiplier and bounds) and writes the arena from a
//! register, so the requantization multiplier sits between registers. The
//! recorded [`FullGraphModel::frame`] beats are the unstalled schedule; the
//! generated testbench *measures* them and compares. DMA/read stalls are counted
//! separately by the testbench (input gaps and output backpressure), never
//! folded into the schedule number.
//!
//! # Numerical contract
//!
//! Requantization is the fixed-point datapath of [`crate::mlir::quant`]:
//! `p = acc * M0`, `r = p >= 0 ? 2^(sh-1) : 2^(sh-1)-1`, `q = (p + r) >>> sh`,
//! `out = clamp(q, lo, hi)`. That is round-half-away-from-zero on the fixed
//! point. It is **not** bit-identical to the manifest's `f32` coefficients
//! (`intref.rs::round_clamp(acc as f32 * requant)`); the difference is the
//! documented `M0`/`shift` approximation error carried in
//! `Requant::multiplier_error_ppm`. [`crate::rtl::reference`] computes both the
//! integer datapath (bit-exact against the HDL) and the `f32` reference and
//! reports their divergence; no stage claims the manifest `f32` path is
//! reproduced bit for bit.
//!
//! The accumulator width is derived from the *whole* reduction of every step:
//! `|operand| * 128 * taps * channels + |bias|` (channels = input channels per
//! group, so a dense step's channel loop is covered), and the plan's declared
//! `engine.accumulator_bits` must cover it or the backend refuses by name.
//!
//! Reflection padding matches `intref.rs::reflect_index` (edge not repeated) for
//! a single step; a padding amount that is not a single step (amount >= extent)
//! is rejected rather than guessed.
//!
//! # Explicitly unsupported (rejected with [`RtlError`], never guessed)
//!
//! Grouped convolutions other than dense (`groups == 1`) and depthwise
//! (`groups == in_channels == out_channels`); any fold other than the shared
//! engine's single lane (`simd = pe = tap_parallel = 1`); non-frame scheduling
//! granularity (row/tile plans, and a plan that carries a tile geometry, are
//! refused instead of silently flattened); non-convolution folds (they are not
//! instantiated and are recorded in the bundle notes instead); unsigned or
//! asymmetric zero-points; activation element types other than `i8`/`u8`;
//! requantization rounding other than half-away-from-zero; saturation bounds
//! outside `i16`; per-channel multiplier counts that do not match the step's
//! output channels; weight tensors whose byte count does not match the planned
//! geometry; an activation tensor the plan does not budget in its activation-RAM
//! region, or an arena that exceeds that region; and a plan whose steps do not
//! correspond one-to-one, in order, with the graph's lowered operators.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use super::{
    ChannelRequant, PlanContext, PlannedOp, Provenance, RTL_BUNDLE_FORMAT,
    RTL_BUNDLE_SCHEMA_VERSION, RtlBundle, RtlConfig, RtlError, RtlFile, RtlFileKind, RtlMetadata,
    TileGeometry as PlanTile, WeightResolver, Xorshift64, bits_for, fnv1a64, slug, sv_signed,
    unpack_i4_x2,
};
use crate::graph::{Graph, Layer, Padding as GraphPadding};
use crate::layout::{
    BufferRegion, ElemType, ExecutionMode, ExecutionPlan, Granularity, OpKind, Requant,
    ScheduleStep, TensorId,
};
use crate::lower::{InputShape, lower_graph};
use crate::planner::{OpDetail, Padding as PlanPadding, PrimitiveOp};
use crate::rtl::reference::{self, ReferenceRun};

/// Stable identifier of the kernel family implemented by this backend.
pub const FULL_GRAPH_KERNEL_ID: &str = "tessera.full_graph.i8i8.v1";
/// Width of one activation arena word, in bits.
pub const ACTIVATION_WORD_BITS: u32 = 16;
/// Multipliers the emitted shared MAC array physically instantiates.
///
/// The engine is *single lane*: one shared multiplier, one shared accumulator
/// and one shared requantizer, used by every step of the schedule. A plan whose
/// steps fold more lanes (or whose engine aggregate does) would claim
/// parallelism the emitted structure does not have, so such a plan is rejected
/// by name instead of being silently serialized.
pub const SHARED_ENGINE_LANES: u64 = 1;
/// Number of frames the generated testbench drives.
pub const TESTBENCH_FRAMES: usize = 2;

/// Extra product-path bits above the accumulator width.
///
/// `rq` multiplies an `ACCW`-bit accumulator by a `17`-bit multiplier and adds a
/// rounding offset of up to `2^30`, so the product path carries `ACCW + 17 + 2`
/// bits and the generated `p + r` can never overflow inside the function.
const PRODUCT_PATH_SLACK: u32 = 19;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Spatial shape of one activation tensor, channel-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape3 {
    /// Channels.
    pub c: u32,
    /// Height.
    pub h: u32,
    /// Width.
    pub w: u32,
}

impl Shape3 {
    /// Element count.
    pub fn elements(self) -> u64 {
        u64::from(self.c) * u64::from(self.h) * u64::from(self.w)
    }

    /// `c*h*w`.
    pub fn label(self) -> String {
        format!("{}x{}x{}", self.c, self.h, self.w)
    }
}

/// Code domain of an activation slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeDomain {
    /// Signed 8-bit codes.
    I8,
    /// Signed 9-bit codes: the unclamped residual sum of two i8 codes.
    I9,
    /// Signed 16-bit transport container.
    I16,
    /// Unsigned 8-bit codes.
    U8,
}

impl CodeDomain {
    /// Stable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            CodeDomain::I8 => "i8",
            CodeDomain::I9 => "i9",
            CodeDomain::I16 => "i16",
            CodeDomain::U8 => "u8",
        }
    }
}

/// Border handling of a convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Border {
    /// Reflection padding that does not repeat the edge (`intref.rs::reflect_index`).
    Reflect {
        /// Padding amount on every side.
        amount: u32,
    },
    /// Constant zero padding.
    Zero {
        /// Padding amount on every side.
        amount: u32,
    },
}

impl Border {
    /// Padding amount.
    pub const fn amount(self) -> u32 {
        match self {
            Border::Reflect { amount } | Border::Zero { amount } => amount,
        }
    }

    /// Whether the border reflects.
    pub const fn is_reflect(self) -> bool {
        matches!(self, Border::Reflect { .. })
    }
}

/// One activation tensor of the emitted arena.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationSlot {
    /// Tensor symbol name from the plan.
    pub name: String,
    /// Tensor identity in the plan.
    pub tensor: u32,
    /// Shape.
    pub shape: Shape3,
    /// Code domain.
    pub domain: CodeDomain,
    /// Signed word offset inside the arena.
    pub offset: u64,
    /// Largest magnitude the slot may hold, derived from the producing step.
    pub max_abs: i64,
    /// First schedule step at which the value must be resident (0 for the graph
    /// input, which the load stage writes before step 0).
    pub first_step: u64,
    /// Last schedule step at which the value must still be resident: the last
    /// step that reads it, or the last step when nothing reads it (the graph
    /// output stays resident through the drain stage). Inclusive, so a value
    /// produced and consumed by the same step is live at that one step.
    pub last_step: u64,
}

/// Unstalled beats of one step, split by the stage that spends them.
///
/// These are the *generated* structure's beats, derived from the element stream
/// the emitter walks - not the plan's estimate, which is recorded separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct StepBeats {
    /// Beats issuing an operand read address on the one shared read port.
    pub read_beats: u64,
    /// Beats accumulating one product into the shared accumulator.
    pub mac_beats: u64,
    /// Beats performing the requantize/saturate epilogue (one per output group;
    /// overlapped with `mac_beats` for convolution steps).
    pub epilogue_beats: u64,
    /// Beats writing one result word into the arena (overlapped with the beats
    /// that produce them; the store stage is one deep).
    pub store_beats: u64,
    /// Pipeline fill (`1`) plus store drain (`1`).
    pub pipeline_beats: u64,
    /// Total unstalled beats of the step.
    pub total_beats: u64,
}

/// Unstalled beat budget of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameBeats {
    /// Input load beats (`NI`).
    pub load_beats: u64,
    /// Sum of every step's beats.
    pub step_beats: u64,
    /// Output drain beats (`2 * NO`: one read issue and one handshake beat per
    /// output code).
    pub drain_beats: u64,
    /// Total unstalled beats of one frame.
    pub total_beats: u64,
}

/// One resolved schedule step of the emitted engine.
///
/// `PartialEq` only: the manifest `f32` coefficients carried for the divergence
/// report are not `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedStep {
    /// Schedule index.
    pub index: usize,
    /// Operator name.
    pub name: String,
    /// Planned operator kind.
    pub op: PlannedOp,
    /// Layout operator kind.
    pub op_kind: OpKind,
    /// SIMD lanes over the reduction dimension (must be 1: the shared engine is
    /// single-lane).
    pub simd: u32,
    /// Output-channel lanes (must be 1).
    pub pe: u32,
    /// Tap parallelism declared by the plan (must be 1).
    pub tap_parallel: u32,
    /// Input shape.
    pub in_shape: Shape3,
    /// Output shape.
    pub out_shape: Shape3,
    /// Kernel extent `[kh, kw]`.
    pub kernel: [u32; 2],
    /// Stride `[sh, sw]`.
    pub stride: [u32; 2],
    /// Groups.
    pub groups: u32,
    /// Input channels per group.
    pub in_per_group: u32,
    /// Border handling.
    pub border: Border,
    /// Upsample scale (elementwise upsample steps only).
    pub upsample_scale: u32,
    /// Weights in `(out_channel, in_per_group, kh, kw)` order.
    pub weights: Vec<i8>,
    /// `bias_int32` per output channel, empty when the layer has none.
    pub bias: Vec<i32>,
    /// Per-output-channel fixed-point requantization.
    pub requant: Vec<ChannelRequant>,
    /// Manifest `f32` requantization coefficients (evidence, never hardware).
    pub requant_f32: Vec<f32>,
    /// Effective saturation bounds after intersecting requant and activation.
    pub clamp: (i64, i64),
    /// Arena slot indices of the operands, in operand order.
    pub inputs: Vec<usize>,
    /// Arena slot index of the result.
    pub output: usize,
    /// `kernel[0] * kernel[1]`.
    pub taps: u64,
    /// Multiply-accumulate count of the step.
    pub macs: u64,
    /// Plan-declared cycle estimate for this step.
    pub plan_cycles: u64,
    /// Generated unstalled beats, split by pipeline stage.
    pub beats: StepBeats,
}

impl Default for ResolvedStep {
    fn default() -> Self {
        ResolvedStep {
            index: 0,
            name: String::new(),
            op: PlannedOp::Conv2d,
            op_kind: OpKind::Conv2d,
            simd: 1,
            pe: 1,
            tap_parallel: 1,
            in_shape: Shape3 { c: 1, h: 1, w: 1 },
            out_shape: Shape3 { c: 1, h: 1, w: 1 },
            kernel: [1, 1],
            stride: [1, 1],
            groups: 1,
            in_per_group: 1,
            border: Border::Zero { amount: 0 },
            upsample_scale: 1,
            weights: Vec::new(),
            bias: Vec::new(),
            requant: Vec::new(),
            requant_f32: Vec::new(),
            clamp: (-128, 127),
            inputs: Vec::new(),
            output: 0,
            taps: 1,
            macs: 0,
            plan_cycles: 0,
            beats: StepBeats::default(),
        }
    }
}

impl ResolvedStep {
    /// Whether the step is a convolution.
    pub fn is_conv(&self) -> bool {
        self.op_kind.is_convolution()
    }

    /// Whether the step is depthwise.
    pub fn is_depthwise(&self) -> bool {
        self.op_kind == OpKind::DepthwiseConv2d
    }

    /// Multipliers the step sweeps in the shared array (`1` for convolution,
    /// `0` for steps that only move or requantize codes).
    pub fn lanes(&self) -> u64 {
        if self.is_conv() {
            SHARED_ENGINE_LANES
        } else {
            0
        }
    }

    /// Generated unstalled beats.
    pub fn cycles(&self) -> u64 {
        self.beats.total_beats
    }

    /// Elements (operand slots) the step's element stream walks.
    fn elements(&self) -> u64 {
        match self.op_kind {
            OpKind::Conv2d | OpKind::PointwiseConv2d => self.macs,
            OpKind::DepthwiseConv2d => self.macs,
            _ => self.out_shape.elements(),
        }
    }

    /// Whether the step's element stream reads two operands per element.
    fn reads_two_operands(&self) -> bool {
        self.op_kind == OpKind::Add
    }
}

/// A graph bound to the plan it was lowered into, with weight contents resolved.
///
/// Construction fails closed: a plan that does not correspond one-to-one, in
/// order, with the graph's lowered operators, or that asks for a construct the
/// emitted structure cannot implement, is an [`RtlError`].
#[derive(Debug, Clone, PartialEq)]
pub struct FullGraphModel {
    /// Graph name.
    pub name: String,
    /// Arena slot index of the graph input tensor.
    pub input_slot: usize,
    /// Arena slot index of the graph output tensor.
    pub output_slot: usize,
    /// Every activation tensor the schedule touches, in first-use order.
    pub slots: Vec<ActivationSlot>,
    /// Steps in schedule order.
    pub steps: Vec<ResolvedStep>,
    /// Total arena words after lifetime reuse.
    pub arena_words: u64,
    /// Words the same slots would need without reuse (sum of every slot), so the
    /// reuse the planner budgets is visible next to it.
    pub arena_words_without_reuse: u64,
    /// Activation word width in bits.
    pub word_width: u32,
    /// Signed accumulator width.
    pub acc_width: u32,
    /// Signed product path width of the requantizer.
    pub product_width: u32,
    /// Largest magnitude the accumulator can hold by construction
    /// (`|operand| * 128 * taps * channels + |bias|` over every step).
    pub acc_bound: i64,
    /// Multipliers in the shared MAC array ([`SHARED_ENGINE_LANES`]).
    pub mac_lanes: u64,
    /// Unstalled beats of one frame.
    pub cycles_unstalled: u64,
    /// Frame beats split into load / steps / drain.
    pub frame: FrameBeats,
    /// Plan-declared frame cycle estimate.
    pub plan_cycles: u64,
    /// Total multiply-accumulate count of one frame.
    pub total_macs: u64,
    /// Honest notes about the emitted structure and the plan.
    pub notes: Vec<String>,
    /// Digest (`fnv1a64:<hex>`) of every byte the emitted RTL embeds: weights,
    /// `bias_int32` values and per-channel multipliers, in schedule order.
    pub contents_digest: String,
}

impl FullGraphModel {
    /// Bind a graph to the plan it was lowered into and resolve all contents.
    ///
    /// `weights` resolves `weights:{layer}` tensors; `bias_int32` and the border
    /// semantics come from the graph artifact, and both are cross-checked
    /// against the plan so an inconsistent plan is an error, not a wrong result.
    pub fn resolve(
        graph: &Graph,
        plan: &ExecutionPlan,
        weights: &dyn WeightResolver,
    ) -> Result<Self, RtlError> {
        validate_static_arena_plan(plan)?;
        let input_shape = plan_input_shape(plan)?;
        let (lowered, report) = lower_graph(
            graph,
            InputShape::new(input_shape.c, input_shape.h, input_shape.w).map_err(|error| {
                RtlError::UnsupportedLayout {
                    detail: format!(
                        "full-graph backend: the plan's input shape is unusable: {error}"
                    ),
                }
            })?,
        )
        .map_err(|error| RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: lowering the graph for the plan failed: {error}; the plan and \
                 the graph must come from the same lowering"
            ),
        })?;

        if lowered.ops.len() != plan.steps.len() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: the plan has {} steps but the graph lowers to {} \
                     operators; the plan and the graph do not describe the same graph",
                    plan.steps.len(),
                    lowered.ops.len()
                ),
            });
        }

        let mut slots: Vec<ActivationSlot> = Vec::new();
        let mut slot_of: BTreeMap<TensorId, usize> = BTreeMap::new();
        let mut steps: Vec<ResolvedStep> = Vec::with_capacity(plan.steps.len());
        let mut notes: Vec<String> = Vec::new();

        for (index, (lowered_op, step)) in lowered.ops.iter().zip(plan.steps.iter()).enumerate() {
            if lowered_op.name != step.op {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!(
                        "full-graph backend: plan step {index} is {:?} but the graph lowers \
                         operator {index} to {:?}",
                        step.op, lowered_op.name
                    ),
                });
            }
            let inputs = step
                .inputs
                .iter()
                .map(|id| ensure_slot(&mut slots, &mut slot_of, plan, *id))
                .collect::<Result<Vec<usize>, RtlError>>()?;
            let output_id = *step
                .outputs
                .first()
                .ok_or_else(|| RtlError::UnsupportedLayout {
                    detail: format!("full-graph backend: step {} has no output tensor", step.op),
                })?;
            let output = ensure_slot(&mut slots, &mut slot_of, plan, output_id)?;

            let resolved = resolve_step(
                graph, plan, step, lowered_op, index, &inputs, output, weights, &mut notes,
            )?;
            steps.push(resolved);
        }

        let input_tensor = *plan
            .steps
            .first()
            .and_then(|s| s.inputs.first())
            .ok_or_else(|| RtlError::UnsupportedLayout {
                detail: "full-graph backend: the plan has no steps".to_string(),
            })?;
        let input_slot =
            *slot_of
                .get(&input_tensor)
                .ok_or_else(|| RtlError::UnsupportedLayout {
                    detail: format!(
                        "full-graph backend: input tensor {input_tensor} has no arena slot"
                    ),
                })?;
        let output_slot =
            steps
                .last()
                .map(|s| s.output)
                .ok_or_else(|| RtlError::UnsupportedLayout {
                    detail: "full-graph backend: the plan has no steps".to_string(),
                })?;

        // Slot magnitudes, in schedule order: the plan's declared code interval
        // when it states one (the authority: a chained residual is planned at its
        // real width), otherwise derived from the producing step. The derivation
        // is order-dependent - an `add` carries the unclamped sum of its
        // operands, and an upsample copies codes - so it must run in step order
        // rather than read every slot at once.
        slots[input_slot].max_abs = plan
            .steps
            .first()
            .and_then(|step| step.in_code_range)
            .map(|range| range.lo.abs().max(range.hi.abs()))
            .unwrap_or(128);
        let word_limit = 1i64 << (ACTIVATION_WORD_BITS - 1);
        let mut cost_notes: Vec<String> = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            let derived = match step.op_kind {
                OpKind::Add => {
                    let trunk = slots[step.inputs[0]].max_abs;
                    let branch = step.clamp.0.abs().max(step.clamp.1.abs());
                    trunk.saturating_add(branch)
                }
                OpKind::NearestUpsample2x => slots[step.inputs[0]].max_abs,
                _ => step.clamp.0.abs().max(step.clamp.1.abs()),
            };
            let declared = plan.steps[index]
                .out_code_range
                .map(|range| range.lo.abs().max(range.hi.abs()));
            let magnitude = declared.unwrap_or(derived);
            slots[step.output].max_abs = magnitude;
            if magnitude >= word_limit {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!(
                        "step {}: the produced codes reach magnitude {}, beyond the emitted \
                         {ACTIVATION_WORD_BITS}-bit activation word",
                        step.name, magnitude
                    ),
                });
            }
            // The plan records the fold it solved for; the emitter records the
            // structure it generated. A disagreement is reported, never hidden.
            let cost = plan.steps[index].cost;
            if step.is_conv() && (cost.macs != step.macs || cost.lanes != step.lanes()) {
                cost_notes.push(format!(
                    "step {index} ({}): plan cost lanes={} macs={} lane_iterations={} vs \
                     generated lanes={} macs={}",
                    step.name,
                    cost.lanes,
                    cost.macs,
                    cost.lane_iterations,
                    step.lanes(),
                    step.macs
                ));
            }
            if declared.is_some() && declared != Some(derived) {
                cost_notes.push(format!(
                    "step {index} ({}): the plan's declared output interval magnitude {magnitude} \
                     differs from the emitter's derivation {derived}; the plan's interval was used",
                    step.name
                ));
            }
        }

        // Accumulator bound: the whole reduction of the widest step, bias
        // included. `channels` is the input channels one output code sums over,
        // so a dense step's channel loop is covered: the emitted accumulator
        // holds every product of every channel, not only of one.
        let mut acc_bound: i64 = 0;
        for step in &steps {
            if !step.is_conv() {
                continue;
            }
            let operand = slots[step.inputs[0]].max_abs;
            let channels = if step.is_depthwise() {
                1
            } else {
                i64::from(step.in_per_group)
            };
            let bias = step
                .bias
                .iter()
                .map(|v| i64::from(*v).abs())
                .max()
                .unwrap_or(0);
            let bound = operand
                .saturating_mul(128)
                .saturating_mul(i64::try_from(step.taps).unwrap_or(i64::MAX))
                .saturating_mul(channels)
                .saturating_add(bias);
            acc_bound = acc_bound.max(bound);
        }
        let acc_bound = acc_bound.max(128);
        let acc_width = (bits_for(u32::try_from(acc_bound).unwrap_or(u32::MAX)) + 1).max(17);
        let product_width = acc_width + PRODUCT_PATH_SLACK;
        let max_shift = steps
            .iter()
            .flat_map(|s| s.requant.iter().map(|c| u32::from(c.shift)))
            .max()
            .unwrap_or(0);
        if max_shift >= product_width {
            return Err(RtlError::UnsupportedQuantization {
                layer: "<full graph>".to_string(),
                detail: format!(
                    "a requantization shift of {max_shift} does not fit the emitted \
                     {product_width}-bit product path"
                ),
            });
        }
        if plan.engine.activation_bits != ACTIVATION_WORD_BITS {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: the emitted arena word is {ACTIVATION_WORD_BITS} bits \
                     (the residual carry needs the wider word) but the plan's engine declares \
                     activation_bits = {}",
                    plan.engine.activation_bits
                ),
            });
        }
        if plan.engine.accumulator_bits < acc_width {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: the emitted accumulator needs {acc_width} bits for the \
                     worst-case |acc| {acc_bound} = |operand| * 128 * taps * channels + |bias|, \
                     but the plan's engine declares accumulator_bits = {}; the plan's datapath \
                     and the emitted one must agree",
                    plan.engine.accumulator_bits
                ),
            });
        }

        // Arena placement: by tensor lifetime, then checked against the plan's
        // own activation-RAM budget.
        let (first_step, last_step) = live_ranges(slots.len(), &steps, input_slot, output_slot);
        for (slot, (first, last)) in slots
            .iter_mut()
            .zip(first_step.iter().zip(last_step.iter()))
        {
            slot.first_step = *first;
            slot.last_step = *last;
        }
        let allocation = allocate_arena(&mut slots);
        if plan.engine.arena_words > 0 && allocation.words != plan.engine.arena_words {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: the emitted arena allocates {} words but the plan budgets                      {}; both sides must call layout::allocate_arena on the same live ranges, so a                      disagreement means the plan and the schedule it was solved for are not the                      same",
                    allocation.words, plan.engine.arena_words
                ),
            });
        }
        validate_arena_against_plan(plan, &slots, allocation.words)?;

        let mut step_beats_total: u64 = 0;
        for step in &mut steps {
            step.beats = step_beats(step);
            step_beats_total = step_beats_total.saturating_add(step.beats.total_beats);
        }
        let frame = FrameBeats {
            load_beats: input_shape.elements(),
            step_beats: step_beats_total,
            drain_beats: 2 * slots[output_slot].shape.elements(),
            total_beats: 0,
        };
        let frame = FrameBeats {
            total_beats: frame
                .load_beats
                .saturating_add(frame.step_beats)
                .saturating_add(frame.drain_beats),
            ..frame
        };

        let mac_lanes = SHARED_ENGINE_LANES;
        let total_macs = steps.iter().map(|s| s.macs).sum();
        let plan_cycles = plan.steps.iter().map(|s| s.estimated_cycles).sum();

        if !report.notes.is_empty() {
            notes.push(format!("graph lowering notes: {}", report.notes.join("; ")));
        }
        notes.push(
            "activation arena words are 16-bit because the residual `add` is unclamped \
             (intref.rs:257-268), so codes leave the i8 domain after a residual sum"
                .to_string(),
        );
        notes.push(format!(
            "one shared single-lane engine: exactly one 16x8 MAC multiplier and one {acc_width}x17 \
             requantization multiplier serve every step of the schedule (DSP count does not grow \
             with the layer count); steps differ only in the muxed address/control they present"
        ));
        notes.push(format!(
            "arena allocation reuses words by tensor lifetime: {} words peak against {} words if \
             every slot were kept live; the plan's activation-RAM region covers it",
            allocation.words, allocation.without_reuse
        ));
        let elementwise: Vec<String> = steps
            .iter()
            .filter(|s| !s.is_conv() && (s.simd > 1 || s.pe > 1))
            .map(|s| format!("{}(simd={},pe={})", s.name, s.simd, s.pe))
            .collect();
        if !elementwise.is_empty() {
            notes.push(format!(
                "elementwise/upsample/quantize steps run one element per cycle; the plan's \
                 simd/pe fold is not instantiated for them: [{}]",
                elementwise.join(", ")
            ));
        }
        if !cost_notes.is_empty() {
            notes.push(format!(
                "plan-vs-generated structure differences: {}",
                cost_notes.join("; ")
            ));
        }
        notes.push(
            "requantization is the fixed-point `(acc*M0 + round) >> shift` datapath; the \
             manifest's f32 coefficients are reported separately and are NOT reproduced bit \
             for bit (bounded by Requant::multiplier_error_ppm)"
                .to_string(),
        );
        notes.push(
            "timed structure: every read is registered (one cycle); the MAC stage consumes the \
             previous address stage, so a product lands every cycle and a store drains one cycle \
             after its element; the epilogue multiplier sits between registers. The recorded \
             frame beats are the unstalled schedule and are measured by the generated testbench; \
             input gaps and output backpressure are counted separately and are never folded in"
                .to_string(),
        );

        let mut digest_input: Vec<u8> = Vec::new();
        for step in &steps {
            for weight in &step.weights {
                digest_input.push(*weight as u8);
            }
            for bias in &step.bias {
                digest_input.extend_from_slice(&bias.to_le_bytes());
            }
            for channel in &step.requant {
                digest_input.extend_from_slice(&channel.multiplier.to_le_bytes());
                digest_input.push(channel.shift);
                digest_input.extend_from_slice(&channel.lo.to_le_bytes());
                digest_input.extend_from_slice(&channel.hi.to_le_bytes());
            }
        }
        let contents_digest = format!("fnv1a64:{:016x}", fnv1a64(&digest_input));

        Ok(FullGraphModel {
            name: plan
                .provenance
                .options
                .get("graph_name")
                .cloned()
                .unwrap_or_else(|| "graph".to_string()),
            input_slot,
            output_slot,
            slots,
            steps,
            arena_words: allocation.words,
            arena_words_without_reuse: allocation.without_reuse,
            word_width: ACTIVATION_WORD_BITS,
            acc_width,
            product_width,
            acc_bound,
            mac_lanes,
            cycles_unstalled: frame.total_beats,
            frame,
            plan_cycles,
            total_macs,
            notes,
            contents_digest,
        })
    }

    /// The arena slot of a tensor id.
    pub fn slot(&self, id: usize) -> &ActivationSlot {
        &self.slots[id]
    }

    /// Generated beats for one step.
    pub fn step_cycles(&self, index: usize) -> u64 {
        self.steps[index].beats.total_beats
    }
}

/// Live range of every slot, in `(first_step, last_step)` inclusive order.
///
/// A tensor is resident from the step that writes it to the last step that reads
/// it; the graph input is written by the load stage before step 0, and the graph
/// output is read by the drain stage after the last step. This is the lifetime
/// the arena allocation and the emitted `FullGraphSlotSummary::live` both use, so
/// an independent checker can rediscover the emitted placement from the artifact.
fn live_ranges(
    slots: usize,
    steps: &[ResolvedStep],
    input_slot: usize,
    output_slot: usize,
) -> (Vec<u64>, Vec<u64>) {
    let mut first = vec![u64::MAX; slots];
    // `u64::MAX` marks "nothing consumes it": step 0 is a legitimate last
    // consumer, so zero cannot be the sentinel.
    let mut last = vec![u64::MAX; slots];
    for step in steps {
        let index = step.index as u64;
        if index < first[step.output] {
            first[step.output] = index;
        }
        for input in &step.inputs {
            if last[*input] == u64::MAX || index > last[*input] {
                last[*input] = index;
            }
        }
    }
    let final_step = (steps.len() as u64).saturating_sub(1);
    for index in 0..slots {
        if first[index] == u64::MAX {
            first[index] = 0;
        }
        if last[index] == u64::MAX {
            // Nothing consumes this tensor: the drain stage still reads the
            // graph output, so it stays resident to the last step.
            last[index] = final_step;
        }
    }
    first[input_slot] = 0;
    last[output_slot] = last[output_slot].max(final_step);
    (first, last)
}

/// Result of the lifetime-aware arena allocation.
struct ArenaPlan {
    /// Peak arena words.
    words: u64,
    /// Words the same slots would need if none were reused.
    without_reuse: u64,
}

/// Place every slot by tensor lifetime with the shared allocator.
///
/// A step-major sequential engine only has to keep the tensors that are still
/// live when it runs: a slot is live from the step that writes it to the last
/// step that reads it, so two slots may share words exactly when their live
/// ranges are disjoint. [`crate::layout::allocate_arena`] is the one allocator
/// the planner leaf also charges from, so the plan's `engine.arena_words` and
/// the emitted arena are the same number by construction - and its returned
/// high-water mark is what both sides must budget (a lifetime arena is *not*
/// the peak simultaneous word count: a weighted interval placement can be
/// pushed above a hole it does not fit, so the extent is the honest figure).
fn allocate_arena(slots: &mut [ActivationSlot]) -> ArenaPlan {
    let ranges: Vec<crate::layout::ArenaRange> = slots
        .iter()
        .map(|slot| crate::layout::ArenaRange {
            start: slot.first_step,
            end: slot.last_step,
            words: slot.shape.elements().max(1),
        })
        .collect();
    let (offsets, words) = crate::layout::allocate_arena(&ranges);
    let mut without_reuse = 0u64;
    for (slot, offset) in slots.iter_mut().zip(offsets.iter()) {
        slot.offset = *offset;
        without_reuse = without_reuse.saturating_add(slot.shape.elements().max(1));
    }
    ArenaPlan {
        words: words.max(1),
        without_reuse: without_reuse.max(1),
    }
}

/// Prove the emitted arena fits the plan's activation-RAM budget.
///
/// Two mismatches are refused by name rather than emitted silently: a tensor the
/// plan never budgeted (the engine would hold data the plan did not pay for) and
/// an arena larger than the region the plan allocated.
fn validate_arena_against_plan(
    plan: &ExecutionPlan,
    slots: &[ActivationSlot],
    arena_words: u64,
) -> Result<(), RtlError> {
    let word_bits = u64::from(ACTIVATION_WORD_BITS);
    let mut region_bytes = 0u64;
    for buffer in &plan.buffers {
        if buffer.region == BufferRegion::ActivationRam {
            region_bytes = region_bytes.max(buffer.base_offset_bytes.saturating_add(buffer.bytes));
        }
    }
    for slot in slots {
        let budgeted = plan.buffers.iter().any(|buffer| {
            buffer.region == BufferRegion::ActivationRam
                && buffer.tensor == Some(TensorId(slot.tensor))
                && buffer.depth.saturating_mul(buffer.width_bits)
                    >= slot.shape.elements().saturating_mul(word_bits)
        });
        if !budgeted {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: the plan budgets no activation-RAM buffer of at least {} \
                     {}-bit words for tensor {} (slot {}); every tensor the emitted engine stores \
                     must appear in the plan's activation-RAM region, so a plan that does not \
                     budget one is not the plan of this structure",
                    slot.shape.elements(),
                    word_bits,
                    slot.name,
                    slot.tensor
                ),
            });
        }
    }
    let needed_bytes = arena_words.saturating_mul(word_bits).div_ceil(8);
    if region_bytes == 0 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the plan declares no activation-RAM region, but the emitted \
                 engine needs {needed_bytes} bytes of on-chip activation storage; the plan and \
                 the emitted structure do not describe the same memory"
            ),
        });
    }
    if needed_bytes > region_bytes {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the emitted activation arena needs {needed_bytes} bytes \
                 ({arena_words} words), but the plan's activation-RAM region covers only \
                 {region_bytes} bytes; the plan under-budgets the arena the schedule produces"
            ),
        });
    }
    Ok(())
}

/// Generated unstalled beats of one step, derived from its element stream.
fn step_beats(step: &ResolvedStep) -> StepBeats {
    let elements = step.elements();
    let groups = if step.is_conv() {
        step.out_shape.elements()
    } else if step.op_kind == OpKind::Add {
        step.out_shape.elements()
    } else {
        step.out_shape.elements()
    };
    let read_beats = if step.reads_two_operands() {
        elements.saturating_mul(2)
    } else {
        elements
    };
    let mac_beats = if step.is_conv() { elements } else { 0 };
    StepBeats {
        read_beats,
        mac_beats,
        epilogue_beats: if step.is_conv() { groups } else { 0 },
        store_beats: groups,
        pipeline_beats: 2,
        total_beats: read_beats.saturating_add(2),
    }
}

/// Input shape of the plan, read from the graph input tensor.
fn plan_input_shape(plan: &ExecutionPlan) -> Result<Shape3, RtlError> {
    let find = |name: &str| plan.tensors.iter().find(|t| t.name == name);
    let tensor = find("p0.v0").or_else(|| {
        plan.steps
            .first()
            .and_then(|s| s.inputs.first())
            .and_then(|id| plan.tensors.iter().find(|t| t.id == *id))
    });
    let tensor = tensor.ok_or_else(|| RtlError::UnsupportedLayout {
        detail: "full-graph backend: the plan carries no graph input tensor (expected `p0.v0`); \
                 the plan must be the projection of a lowered graph"
            .to_string(),
    })?;
    Ok(Shape3 {
        c: tensor.shape.c,
        h: tensor.shape.h,
        w: tensor.shape.w,
    })
}

/// Register one activation tensor of the plan and return its arena slot index.
fn ensure_slot(
    slots: &mut Vec<ActivationSlot>,
    slot_of: &mut BTreeMap<TensorId, usize>,
    plan: &ExecutionPlan,
    id: TensorId,
) -> Result<usize, RtlError> {
    if let Some(index) = slot_of.get(&id) {
        return Ok(*index);
    }
    let tensor =
        plan.tensors
            .iter()
            .find(|t| t.id == id)
            .ok_or_else(|| RtlError::UnsupportedLayout {
                detail: format!(
                    "full-graph backend: tensor {id} is not in the plan's tensor table"
                ),
            })?;
    let domain = match tensor.elem {
        ElemType::I8 => CodeDomain::I8,
        ElemType::I9 => CodeDomain::I9,
        ElemType::I16 => CodeDomain::I16,
        ElemType::U8 => CodeDomain::U8,
        other => {
            return Err(RtlError::UnsupportedElementType {
                layer: tensor.name.clone(),
                tensor: tensor.name.clone(),
                element_type: other.into(),
            });
        }
    };
    let index = slots.len();
    slots.push(ActivationSlot {
        name: tensor.name.clone(),
        tensor: id.0,
        shape: Shape3 {
            c: tensor.shape.c,
            h: tensor.shape.h,
            w: tensor.shape.w,
        },
        domain,
        offset: 0,
        max_abs: 0,
        first_step: 0,
        last_step: 0,
    });
    slot_of.insert(id, index);
    Ok(index)
}

/// Resolve one step against the graph and the plan.
#[allow(clippy::too_many_arguments)]
fn resolve_step(
    graph: &Graph,
    plan: &ExecutionPlan,
    step: &ScheduleStep,
    lowered_op: &PrimitiveOp,
    index: usize,
    inputs: &[usize],
    output: usize,
    weights: &dyn WeightResolver,
    notes: &mut Vec<String>,
) -> Result<ResolvedStep, RtlError> {
    if step.tap_parallel != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {} ({}): tap_parallel {} is not emittable: the emitted shared engine has \
                 exactly {SHARED_ENGINE_LANES} multiplier, so the planner's cycle model would \
                 claim parallelism the hardware does not have; re-solve with tap_parallel = 1",
                step.step, step.op, step.tap_parallel
            ),
        });
    }
    if step.simd != 1 || step.pe != 1 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {} ({}): the emitted shared engine is single-lane (simd = pe = 1, \
                 {SHARED_ENGINE_LANES} multiplier), but the plan asks for simd {} x pe {}; \
                 re-solve with the shared engine's fold",
                step.step, step.op, step.simd, step.pe
            ),
        });
    }
    if step.mode != Granularity::Frame {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {} ({}): the emitted engine makes one whole-frame pass per step, but this \
                 step is scheduled at `{:?}` granularity; re-solve with granularities = \
                 [\"frame\"]",
                step.step, step.op, step.mode
            ),
        });
    }
    if step.tile.is_some() {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {} ({}): the step carries a tile geometry ({:?}) that the whole-frame \
                 engine does not implement; a tiled plan is refused instead of being flattened \
                 into one pass",
                step.step,
                step.op,
                step.tile.map(|t| (t.width, t.height))
            ),
        });
    }
    let in_tensor = plan
        .tensors
        .iter()
        .find(|t| t.id == step.inputs[0])
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: input tensor is absent from the tensor table",
                step.op
            ),
        })?;
    let out_tensor = plan
        .tensors
        .iter()
        .find(|t| t.id == step.outputs[0])
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: output tensor is absent from the tensor table",
                step.op
            ),
        })?;
    let in_shape = Shape3 {
        c: in_tensor.shape.c,
        h: in_tensor.shape.h,
        w: in_tensor.shape.w,
    };
    let out_shape = Shape3 {
        c: out_tensor.shape.c,
        h: out_tensor.shape.h,
        w: out_tensor.shape.w,
    };

    let mut resolved = ResolvedStep {
        index,
        name: step.op.clone(),
        op: planned_op(step.op_kind),
        op_kind: step.op_kind,
        in_shape,
        out_shape,
        kernel: step.kernel,
        stride: step.stride,
        groups: step.groups,
        inputs: inputs.to_vec(),
        output,
        taps: u64::from(step.kernel[0]) * u64::from(step.kernel[1]),
        plan_cycles: step.estimated_cycles,
        ..ResolvedStep::default()
    };

    if step.op_kind.is_convolution() {
        let layer = graph
            .layer_named(&step.op)
            .ok_or_else(|| RtlError::UnsupportedLayout {
                detail: format!(
                    "step {} ({}): the graph has no layer with this name; the plan must be the \
                     projection of this graph",
                    step.step, step.op
                ),
            })?;
        resolve_conv(&mut resolved, layer, step, lowered_op, plan, weights, notes)?;
    } else {
        resolve_elementwise(&mut resolved, step)?;
    }
    Ok(resolved)
}

fn resolve_conv(
    resolved: &mut ResolvedStep,
    layer: &Layer,
    step: &ScheduleStep,
    lowered_op: &PrimitiveOp,
    plan: &ExecutionPlan,
    weights: &dyn WeightResolver,
    notes: &mut Vec<String>,
) -> Result<(), RtlError> {
    let depthwise = step.op_kind == OpKind::DepthwiseConv2d;
    let in_channels = layer.in_channels.get();
    let out_channels = layer.out_channels.get();
    let groups = layer.groups.get();
    if (step.in_channels, step.out_channels, step.groups) != (in_channels, out_channels, groups) {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: plan channels {}->{} groups {} do not match the graph layer {}->{} \
                 groups {}",
                step.op,
                step.in_channels,
                step.out_channels,
                step.groups,
                in_channels,
                out_channels,
                groups
            ),
        });
    }
    if !depthwise && groups != 1 {
        return Err(RtlError::UnsupportedOperator {
            layer: step.op.clone(),
            op: PlannedOp::Conv2d,
            supported: "dense convolution (groups == 1) or depthwise (groups == in_channels == \
                        out_channels); grouped convolution is not implemented",
        });
    }
    if depthwise && (groups != in_channels || in_channels != out_channels) {
        return Err(RtlError::UnsupportedOperator {
            layer: step.op.clone(),
            op: PlannedOp::Depthwise,
            supported: "depthwise convolution with groups == in_channels == out_channels",
        });
    }
    let kernel = [layer.kernel.h().get(), layer.kernel.w().get()];
    if kernel != step.kernel {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: plan kernel {:?} does not match the graph layer's {:?}",
                step.op, step.kernel, kernel
            ),
        });
    }
    if (layer.stride.h().get(), layer.stride.w().get()) != (step.stride[0], step.stride[1]) {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: plan stride {:?} does not match the graph layer's {:?}",
                step.op,
                step.stride,
                [layer.stride.h().get(), layer.stride.w().get()]
            ),
        });
    }
    let in_per_group = if depthwise {
        1
    } else {
        in_channels / groups.max(1)
    };
    if in_per_group == 0 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: the layer's {} input channels do not divide its {} groups",
                step.op, in_channels, groups
            ),
        });
    }

    let border = match layer.padding {
        GraphPadding::None => Border::Zero { amount: 0 },
        GraphPadding::Zero(amount) => Border::Zero {
            amount: amount.get(),
        },
        GraphPadding::Reflect(amount) => Border::Reflect {
            amount: amount.get(),
        },
    };
    let plan_padding = match &lowered_op.detail {
        OpDetail::Conv(detail) => detail.padding,
        _ => {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "step {}: the graph lowers this layer to a non-convolution operator",
                    step.op
                ),
            });
        }
    };
    let plan_reflect = matches!(plan_padding, PlanPadding::Reflect);
    if plan_reflect != border.is_reflect() && border.amount() > 0 {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: the plan's padding mode and the graph layer's disagree ({plan_padding:?} \
                 vs {:?}); refusing to guess which border semantics to implement",
                step.op, layer.padding
            ),
        });
    }
    if let Some(spec) = step.padding {
        let plan_amount = spec.top.max(spec.bottom).max(spec.left).max(spec.right);
        if plan_amount != border.amount() || spec.reflect != border.is_reflect() {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "step {}: the plan's resolved padding (amount {}, reflect {}) does not match \
                     the graph layer's ({:?}); the plan must be the projection of this graph",
                    step.op, plan_amount, spec.reflect, layer.padding
                ),
            });
        }
    }
    if border.is_reflect() {
        let amount = border.amount();
        if amount >= resolved.in_shape.h || amount >= resolved.in_shape.w {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "step {}: reflection padding of {amount} needs a single reflection step \
                     (amount < input extent); the emitted indexer implements exactly \
                     intref.rs::reflect_index for one step",
                    step.op
                ),
            });
        }
    } else if border.amount() > 0 {
        let amount = border.amount();
        if resolved.in_shape.h + 2 * amount < resolved.kernel[0]
            || resolved.in_shape.w + 2 * amount < resolved.kernel[1]
        {
            return Err(RtlError::UnsupportedLayout {
                detail: format!(
                    "step {}: the padded input is smaller than the kernel (padding {amount}, \
                     kernel {:?}, input {})",
                    step.op,
                    resolved.kernel,
                    resolved.in_shape.label()
                ),
            });
        }
    }

    let weight_ref = step
        .weights
        .as_ref()
        .ok_or_else(|| RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: a convolution step without weights is malformed",
                step.op
            ),
        })?;
    if !plan.weights.iter().any(|w| w.id == weight_ref.id) {
        return Err(RtlError::WeightsNotResident {
            layer: step.op.clone(),
            tensor: weight_ref.name.clone(),
            detail: "it is absent from ExecutionPlan::weights".to_string(),
        });
    }
    let expected_weights = u64::from(out_channels) * u64::from(in_per_group) * resolved.taps;
    let raw = weights.tensor_bytes(weight_ref)?;
    let flat: Vec<i8> = match weight_ref.elem {
        ElemType::I8 => raw.iter().map(|b| *b as i8).collect(),
        ElemType::I4 => unpack_i4_x2(&step.op, &weight_ref.name, &raw, expected_weights)?,
        other => {
            return Err(RtlError::UnsupportedElementType {
                layer: step.op.clone(),
                tensor: weight_ref.name.clone(),
                element_type: other.into(),
            });
        }
    };
    if flat.len() as u64 != expected_weights {
        return Err(RtlError::ShapeMismatch {
            layer: step.op.clone(),
            what: format!("weight tensor {}", weight_ref.name),
            expected: expected_weights,
            actual: flat.len() as u64,
        });
    }

    let bias: Vec<i32> = layer.bias_int32.clone();
    if bias.len() != out_channels as usize {
        return Err(RtlError::ShapeMismatch {
            layer: step.op.clone(),
            what: "bias_int32 length (out_channels)".to_string(),
            expected: u64::from(out_channels),
            actual: bias.len() as u64,
        });
    }
    let has_bias = bias.iter().any(|v| *v != 0);
    if has_bias != step.bias.is_some() {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: the plan {} a bias tensor but the graph layer's bias_int32 is {}; the \
                 plan must be the projection of this graph",
                step.op,
                if step.bias.is_some() {
                    "carries"
                } else {
                    "has no"
                },
                if has_bias { "non-zero" } else { "all zero" }
            ),
        });
    }

    let requant = step
        .requant
        .as_ref()
        .ok_or_else(|| RtlError::UnsupportedQuantization {
            layer: step.op.clone(),
            detail: "a convolution step without requantization is not implementable".to_string(),
        })?;
    let clamp = step.activation.effective_clamp(Some(requant));
    let clamp = (i64::from(clamp.0), i64::from(clamp.1));
    let channels = requant_channels(&step.op, requant, out_channels)?;
    // The clamp is uniform across the step's channels: `Requant::saturate_*` and
    // `Activation` are scalar, so the per-channel records only vary in `m0`.
    let channels: Vec<ChannelRequant> = channels
        .into_iter()
        .map(|channel| ChannelRequant {
            lo: i16::try_from(clamp.0).unwrap_or(i16::MIN),
            hi: i16::try_from(clamp.1).unwrap_or(i16::MAX),
            ..channel
        })
        .collect();
    if clamp.0 < i64::from(i16::MIN) || clamp.1 > i64::from(i16::MAX) {
        return Err(RtlError::UnsupportedQuantization {
            layer: step.op.clone(),
            detail: format!(
                "saturation bounds [{}, {}] do not fit the emitted 16-bit clamp ROM",
                clamp.0, clamp.1
            ),
        });
    }

    let f32_requant = layer.requant.values().to_vec();
    if f32_requant.len() != out_channels as usize {
        notes.push(format!(
            "step {}: the graph layer's f32 requantization vector has {} entries for {} output \
             channels; the divergence report skips this step",
            step.op,
            f32_requant.len(),
            out_channels
        ));
    }

    resolved.in_per_group = in_per_group;
    resolved.border = border;
    resolved.weights = flat;
    resolved.bias = bias;
    resolved.requant = channels;
    resolved.requant_f32 = f32_requant;
    resolved.clamp = clamp;
    // One element per (output element, input channel, tap): the single-lane
    // engine walks the whole reduction of every output code.
    resolved.macs = resolved
        .taps
        .saturating_mul(u64::from(if depthwise { 1 } else { in_per_group }))
        .saturating_mul(resolved.out_shape.elements());
    Ok(())
}

fn resolve_elementwise(resolved: &mut ResolvedStep, step: &ScheduleStep) -> Result<(), RtlError> {
    if resolved.in_shape != resolved.out_shape && step.op_kind != OpKind::NearestUpsample2x {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "step {}: elementwise shapes {} -> {} must match unless the step upsamples",
                step.op,
                resolved.in_shape.label(),
                resolved.out_shape.label()
            ),
        });
    }
    match step.op_kind {
        OpKind::NearestUpsample2x => {
            if resolved.out_shape.h != resolved.in_shape.h * 2
                || resolved.out_shape.w != resolved.in_shape.w * 2
                || resolved.out_shape.c != resolved.in_shape.c
            {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!(
                        "step {}: nearest upsample must double the spatial extent exactly \
                         ({} -> {})",
                        step.op,
                        resolved.in_shape.label(),
                        resolved.out_shape.label()
                    ),
                });
            }
            if let Some(scale) = step.upsample_scale {
                if scale != 2 {
                    return Err(RtlError::UnsupportedLayout {
                        detail: format!(
                            "step {}: the plan declares upsample scale {scale}; the emitted copy \
                             kernel implements exactly 2x",
                            step.op
                        ),
                    });
                }
            }
            resolved.upsample_scale = 2;
            resolved.macs = 0;
        }
        OpKind::Add => {
            if resolved.inputs.len() != 2 {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!("step {}: add must have two operand slots", step.op),
                });
            }
            if resolved.in_shape != resolved.out_shape {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!(
                        "step {}: the residual operands must share the output shape ({} -> {})",
                        step.op,
                        resolved.in_shape.label(),
                        resolved.out_shape.label()
                    ),
                });
            }
            let requant =
                step.requant
                    .as_ref()
                    .ok_or_else(|| RtlError::UnsupportedQuantization {
                        layer: step.op.clone(),
                        detail: "an add step without a branch requantization is malformed"
                            .to_string(),
                    })?;
            let clamp = step.activation.effective_clamp(Some(requant));
            let clamp = (i64::from(clamp.0), i64::from(clamp.1));
            let channels = requant_channels(&step.op, requant, 1)?;
            resolved.requant = channels
                .into_iter()
                .map(|channel| ChannelRequant {
                    lo: i16::try_from(clamp.0).unwrap_or(i16::MIN),
                    hi: i16::try_from(clamp.1).unwrap_or(i16::MAX),
                    ..channel
                })
                .collect();
            resolved.clamp = clamp;
            resolved.macs = 0;
        }
        OpKind::Requant | OpKind::Activation | OpKind::QuantizeOutput => {
            if resolved.inputs.len() != 1 {
                return Err(RtlError::UnsupportedLayout {
                    detail: format!("step {}: this step must have exactly one operand", step.op),
                });
            }
            let clamp = match &step.requant {
                Some(requant) => {
                    let clamp = step.activation.effective_clamp(Some(requant));
                    let clamp = (i64::from(clamp.0), i64::from(clamp.1));
                    resolved.requant = requant_channels(&step.op, requant, 1)?
                        .into_iter()
                        .map(|channel| ChannelRequant {
                            lo: i16::try_from(clamp.0).unwrap_or(i16::MIN),
                            hi: i16::try_from(clamp.1).unwrap_or(i16::MAX),
                            ..channel
                        })
                        .collect();
                    clamp
                }
                None => {
                    let clamp = step.activation.effective_clamp(None);
                    (i64::from(clamp.0), i64::from(clamp.1))
                }
            };
            if clamp.0 < i64::from(i16::MIN) || clamp.1 > i64::from(i16::MAX) {
                return Err(RtlError::UnsupportedQuantization {
                    layer: step.op.clone(),
                    detail: format!(
                        "saturation bounds [{}, {}] do not fit the emitted 16-bit clamp ROM",
                        clamp.0, clamp.1
                    ),
                });
            }
            resolved.clamp = clamp;
            resolved.macs = 0;
        }
        other => {
            return Err(RtlError::UnsupportedOperator {
                layer: step.op.clone(),
                op: planned_op(other),
                supported: "dense/depthwise convolution, add, requant, activation, nearest \
                            upsample and unsigned output quantization",
            });
        }
    }
    Ok(())
}

/// Requantization parameters of a step's per-channel multipliers.
fn requant_channels(
    layer: &str,
    requant: &Requant,
    channels: u32,
) -> Result<Vec<ChannelRequant>, RtlError> {
    if requant.rounding != crate::layout::Rounding::HalfAwayFromZero {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!(
                "rounding {:?} is not reproducible by the generated datapath; supported: half \
                 away from zero",
                requant.rounding
            ),
        });
    }
    if !(0..=31).contains(&requant.shift) {
        return Err(RtlError::UnsupportedQuantization {
            layer: layer.to_string(),
            detail: format!("shift {} is outside 0..=31", requant.shift),
        });
    }
    let multipliers: Vec<i64> = if requant.per_channel_multipliers.is_empty() {
        vec![requant.m0; channels as usize]
    } else {
        if requant.per_channel_multipliers.len() != channels as usize {
            return Err(RtlError::ShapeMismatch {
                layer: layer.to_string(),
                what: "per-channel multipliers".to_string(),
                expected: u64::from(channels),
                actual: requant.per_channel_multipliers.len() as u64,
            });
        }
        requant.per_channel_multipliers.clone()
    };
    let mut out = Vec::with_capacity(multipliers.len());
    for (index, multiplier) in multipliers.into_iter().enumerate() {
        if !(0..(1i64 << 17)).contains(&multiplier) {
            return Err(RtlError::UnsupportedQuantization {
                layer: layer.to_string(),
                detail: format!(
                    "channel {index}: multiplier {multiplier} does not fit the 17-bit multiplier \
                     ROM ([0, 2^17))"
                ),
            });
        }
        out.push(ChannelRequant {
            multiplier: multiplier as u32,
            shift: requant.shift as u8,
            lo: -128,
            hi: 127,
        });
    }
    Ok(out)
}

/// Map a layout operator kind to the backend's planned-operator vocabulary.
pub fn planned_op(kind: OpKind) -> PlannedOp {
    match kind {
        OpKind::PointwiseConv2d => PlannedOp::PointwiseDot,
        OpKind::Conv2d => PlannedOp::Conv2d,
        OpKind::DepthwiseConv2d => PlannedOp::Depthwise,
        OpKind::Add => PlannedOp::Add,
        OpKind::Requant => PlannedOp::Requant,
        OpKind::Activation => PlannedOp::Activation,
        OpKind::NearestUpsample2x => PlannedOp::UpsampleNearest,
        OpKind::QuantizeOutput => PlannedOp::QuantizeOutput,
    }
}

// ---------------------------------------------------------------------------
// Bundle
// ---------------------------------------------------------------------------

/// Emit the full-graph RTL bundle for a resolved model.
pub fn lower_full_graph(
    plan: &ExecutionPlan,
    model: &FullGraphModel,
    config: &RtlConfig,
) -> Result<RtlBundle, RtlError> {
    let frames = build_frames(model, config);
    lower_full_graph_with_frames(plan, model, config, &frames)
}

/// Emit the full-graph RTL bundle for an explicit stimulus.
///
/// The stimulus determines the embedded reference, so delivering it explicitly
/// (rather than re-deriving it from [`RtlConfig`]) keeps the emitted bundle and
/// the comparison data provably consistent - the boundary fixtures use this to
/// drive an all-maximum code stream.
pub fn lower_full_graph_with_frames(
    plan: &ExecutionPlan,
    model: &FullGraphModel,
    config: &RtlConfig,
    frames: &[Vec<i64>],
) -> Result<RtlBundle, RtlError> {
    config.validate()?;
    if plan.steps.len() != model.steps.len() {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the model has {} steps but the plan has {}",
                model.steps.len(),
                plan.steps.len()
            ),
        });
    }
    if frames.is_empty() {
        return Err(RtlError::UnsupportedLayout {
            detail: "full-graph backend: an emitted testbench needs at least one stimulus frame"
                .to_string(),
        });
    }
    validate_static_arena_plan(plan)?;
    let mut provenance = plan_provenance(plan)?;
    provenance.weights_sha256 = model.contents_digest.clone();
    provenance.validate()?;

    let dut_name = format!("{}_full_graph_v1", slug(&config.module_prefix));
    let tb_name = format!("{dut_name}_tb");
    let top_module = if config.emit_testbench {
        tb_name.clone()
    } else {
        dut_name.clone()
    };

    let reference = reference::run_integer(model, frames)?;
    let manifest = reference::run_f32(model, frames).ok();
    let divergence = manifest
        .as_ref()
        .map(|m| reference::divergence(&reference, m))
        .unwrap_or_default();
    let checkpoints = build_checkpoints(model, &reference);

    let mut files: Vec<RtlFile> = Vec::new();
    files.push(RtlFile {
        name: format!("{dut_name}.sv"),
        kind: RtlFileKind::SystemVerilog,
        contents: emit_dut(model, &dut_name, &provenance, &config.note()),
    });
    if config.emit_memory_images {
        files.push(RtlFile {
            name: format!("{dut_name}.weights.hex"),
            kind: RtlFileKind::MemoryImage,
            contents: emit_weight_image(model),
        });
        files.push(RtlFile {
            name: "stim_input.hex".to_string(),
            kind: RtlFileKind::MemoryImage,
            contents: emit_stimulus_image(frames),
        });
        files.push(RtlFile {
            name: "expected.hex".to_string(),
            kind: RtlFileKind::MemoryImage,
            contents: emit_expected_image(&reference),
        });
        files.push(RtlFile {
            name: "reference_steps.txt".to_string(),
            kind: RtlFileKind::MemoryImage,
            contents: emit_reference_text(model, &reference, &divergence),
        });
    }
    if config.emit_testbench {
        files.push(RtlFile {
            name: format!("{tb_name}.sv"),
            kind: RtlFileKind::Testbench,
            contents: emit_testbench(
                model,
                &dut_name,
                &tb_name,
                &provenance,
                &reference,
                &checkpoints,
            ),
        });
        files.push(RtlFile {
            name: "run_verilator.sh".to_string(),
            kind: RtlFileKind::Script,
            contents: emit_run_script(&dut_name, &tb_name),
        });
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));

    let digests: Vec<super::FileDigest> = files
        .iter()
        .map(|f| super::FileDigest {
            name: f.name.clone(),
            kind: f.kind,
            bytes: f.contents.len(),
            fnv1a64: format!("{:016x}", fnv1a64(f.contents.as_bytes())),
        })
        .collect();
    let mut digest_input = String::new();
    for digest in &digests {
        let _ = writeln!(
            digest_input,
            "{} {} {}",
            digest.name, digest.bytes, digest.fnv1a64
        );
    }
    let bundle_digest = format!("{:016x}", fnv1a64(digest_input.as_bytes()));

    let last = model.steps.last();
    let plan_context = PlanContext {
        plan_format: plan.format.clone(),
        plan_schema_version: plan.schema_version,
        execution_mode: plan.execution_mode.as_str().to_string(),
        granularity: "frame".to_string(),
        engine: plan.engine.name.clone(),
        step_index: 0,
        step_count: model.steps.len() as u32,
        estimated_cycles: model.plan_cycles,
        spatial_in: [
            model.slots[model.input_slot].shape.h,
            model.slots[model.input_slot].shape.w,
        ],
        spatial_out: [
            model.slots[model.output_slot].shape.h,
            model.slots[model.output_slot].shape.w,
        ],
    };
    let mut semantic_notes = model.notes.clone();
    semantic_notes.push(format!(
        "provenance: weights are resolved through the supplied WeightResolver for `weights:{{layer}}` \
         tensors and the embedded bytes digest to {}; `bias_int32` comes from the graph artifact's \
         layer and the plan's bias-tensor presence is cross-checked against it; the plan's declared \
         weights digest was {:?}",
        model.contents_digest,
        plan.provenance.options.get("weights_sha256").cloned()
    ));
    semantic_notes.push(format!(
        "physical mapping: one shared activation arena of {} x {}-bit words with lifetime reuse \
         ({} words if every tensor stayed live), {} MAC multiplier, one requantization multiplier, \
         and {} unstalled beats per frame; graph-internal activations never leave the chip (no \
         internal DMA and no DDR port), so the plan's DDR figures for internal buffers are not \
         realized by this structure, only the graph input and output cross the module boundary",
        model.arena_words,
        model.word_width,
        model.arena_words_without_reuse,
        model.mac_lanes,
        model.cycles_unstalled
    ));
    semantic_notes.push(format!(
        "generated unstalled beats {} = load {} + steps {} + drain {}; plan estimate {} \
         (plan/estimate = {:.3}); the plan number is an estimate and is never presented as measured",
        model.cycles_unstalled,
        model.frame.load_beats,
        model.frame.step_beats,
        model.frame.drain_beats,
        model.plan_cycles,
        if model.cycles_unstalled == 0 {
            0.0
        } else {
            model.plan_cycles as f64 / model.cycles_unstalled as f64
        }
    ));
    semantic_notes.push(format!(
        "plan arena budget: engine.arena_words = {} vs emitted {} words (same lifetime allocation, \
         so the plan charges exactly the emitted high-water mark)",
        plan.engine.arena_words, model.arena_words
    ));
    if !divergence.is_empty() {
        semantic_notes.push(format!(
            "manifest f32 vs emitted fixed point: {} step(s) differ, worst per-step code distance \
             {} (documented M0/shift approximation, not a bit-exact claim)",
            divergence.len(),
            divergence.iter().map(|d| d.max_abs_diff).max().unwrap_or(0)
        ));
    }

    let metadata = RtlMetadata {
        format: RTL_BUNDLE_FORMAT.to_string(),
        schema_version: RTL_BUNDLE_SCHEMA_VERSION,
        kernel_id: FULL_GRAPH_KERNEL_ID.to_string(),
        provenance: provenance.clone(),
        plan: plan_context,
        top_module,
        kernel_module: dut_name.clone(),
        layer: last.map(|s| s.name.clone()).unwrap_or_default(),
        tile: PlanTile {
            simd: SHARED_ENGINE_LANES as u32,
            noc: SHARED_ENGINE_LANES as u32,
            k: 1,
            fold: 1,
        },
        in_channels: u64::from(model.slots[model.input_slot].shape.c),
        out_channels: u64::from(model.slots[model.output_slot].shape.c),
        accumulator_width: model.acc_width,
        max_abs_accumulator: u64::try_from(model.acc_bound).unwrap_or(u64::MAX),
        kernel_cycles_per_window: model.cycles_unstalled,
        requantization: last.map(|s| s.requant.clone()).unwrap_or_default(),
        requant_multiplier_error_ppm: plan
            .steps
            .iter()
            .filter_map(|s| s.requant.as_ref().and_then(|r| r.multiplier_error_ppm))
            .max(),
        config: config.clone(),
        files: digests,
        supported_operators: vec![
            format!(
                "full graph: dense conv2d (any kernel/stride, groups=1), depthwise conv2d, add \
                 with branch requantization, nearest 2x upsample, activation/requant, unsigned \
                 output quantization [{FULL_GRAPH_KERNEL_ID}]"
            ),
            format!(
                "requantization: fixed-point M0 in [0, 2^17) per output channel, uniform shift, \
                 round half away from zero, then saturation"
            ),
            format!(
                "borders: reflection (intref.rs::reflect_index, single step) and zero padding; \
                 activations: signed i8 with a 16-bit internal word for 9-bit residual sums; one \
                 shared {SHARED_ENGINE_LANES}-lane MAC array over a whole-frame arena"
            ),
        ],
        supported_interface: vec![
            "clk, rst (synchronous, active-high reset), start (frame start pulse), done (held \
             until the next start)"
                .to_string(),
            "in_data [7:0] signed int8 graph input codes, one code per beat, in_valid/in_ready"
                .to_string(),
            format!(
                "out_data [7:0] unsigned image codes, one code per beat, out_valid/out_ready; the \
                 frame's {} output codes are streamed in channel-major order",
                model.slots[model.output_slot].shape.elements()
            ),
            format!(
                "dbg_en/dbg_addr[{}-1:0]/dbg_data read the {}x{}-bit activation arena through the \
                 engine's single read port, serviced only while the FSM is idle or held \
                 (verification port, no second read port)",
                bits_for(model.arena_words as u32),
                model.arena_words,
                ACTIVATION_WORD_BITS
            ),
            "dbg_hold_en/dbg_cont/dbg_held: verification single-step. With dbg_hold_en low the \
             FSM runs its schedule unchanged; with it high the FSM holds after the input load and \
             after every step (dbg_held high) until dbg_cont is pulsed, so the testbench can read \
             each step's codes through the shared debug port before the arena reuses them"
                .to_string(),
            "testbench plusargs: +STIM=<path> +EXPECTED=<path> +TRACE=<path> +NOCHECK".to_string(),
        ],
        unsupported: unsupported_summary(),
        reuse_candidates: vec![
            "nngen/examples/st_stream/rtl/sts_lib.v: sts_rq (bit-identical fx requantization)"
                .to_string(),
            "nngen/examples/st_stream/rtl/sts_mvau (streaming SIMD/PE folding MAC array; the \
             emitted engine is the same shared-array idea with one lane)"
                .to_string(),
            "nngen/examples/st_stream/rtl/sts_rom (weight ROM with $readmemh initialization)"
                .to_string(),
        ],
        reproduce: vec![
            format!(
                "emit: tessera::rtl::full::lower_full_graph -> {dut_name}.sv + {tb_name}.sv + \
                 rtl_bundle.json"
            ),
            format!(
                "simulate: verilator --binary --timing -Wno-fatal --top-module {tb_name} -o \
                 {tb_name}_sim {dut_name}.sv {tb_name}.sv && ./obj_dir/{tb_name}_sim"
            ),
            "verify: the testbench prints TESSA_RTL_PASS with mismatches=0".to_string(),
        ],
        bundle_digest,
        full_graph: Some(full_graph_summary(model, &semantic_notes)),
    };
    let metadata_json =
        serde_json::to_string_pretty(&metadata).map_err(|source| RtlError::Serialize {
            context: "serializing full-graph bundle metadata",
            source,
        })?;
    files.push(RtlFile {
        name: "rtl_bundle.json".to_string(),
        kind: RtlFileKind::Metadata,
        contents: metadata_json,
    });

    Ok(RtlBundle { metadata, files })
}

/// The emitted engine is one static-schedule RAM arena shared by every step.
///
/// That is exactly the `shared_tile` organization, so a plan labelled
/// `streaming` or `dedicated` - whose schedule, per-step engines and resource
/// promises describe one engine per operator - must not be presented as if it
/// were this structure. A tiled or row-granularity plan is refused for the same
/// reason: its tile geometry, halo re-reads and per-tile setup cycles describe a
/// schedule this engine does not implement, and flattening it into one
/// whole-frame pass would silently discard them.
fn validate_static_arena_plan(plan: &ExecutionPlan) -> Result<(), RtlError> {
    if plan.execution_mode != ExecutionMode::SharedTile {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the emitted engine is one static-schedule activation arena \
                 with one shared MAC array covering every step, i.e. the `shared_tile` \
                 organization, but the plan declares `{}`. Re-solve with execution_modes = \
                 [\"shared_tile\"]; a `{}` layout's per-operator engines, buffers and DDR \
                 promises are not what this backend emits",
                plan.execution_mode.as_str(),
                plan.execution_mode.as_str()
            ),
        });
    }
    if let Some(tile) = plan.tile {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the plan carries a tile geometry ({}x{}) but the emitted \
                 engine makes one whole-frame pass per step and implements no tile schedule; \
                 re-solve with granularities = [\"frame\"]",
                tile.width, tile.height
            ),
        });
    }
    if plan
        .steps
        .iter()
        .any(|step| step.mode != Granularity::Frame || step.tile.is_some())
    {
        return Err(RtlError::UnsupportedLayout {
            detail: "full-graph backend: at least one step is scheduled at row/tile granularity; \
                     the emitted engine implements whole-frame steps only, so the tile geometry \
                     and per-tile setup cycles the plan paid for would be discarded by emitting \
                     it. Re-solve with granularities = [\"frame\"]"
                .to_string(),
        });
    }
    if plan.engine.lanes() != SHARED_ENGINE_LANES {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: the plan's engine declares {} lanes (simd {} x pe {} x tap \
                 {}) but the emitted engine has exactly {SHARED_ENGINE_LANES} shared \
                 multiplier; the plan's datapath and the emitted one must agree",
                plan.engine.lanes(),
                plan.engine.simd,
                plan.engine.pe,
                plan.engine.tap_parallel
            ),
        });
    }
    Ok(())
}

/// Provenance of a plan, validated for attribution.
fn plan_provenance(plan: &ExecutionPlan) -> Result<Provenance, RtlError> {
    if plan.format != crate::layout::PLAN_FORMAT {
        return Err(RtlError::UnsupportedLayout {
            detail: format!(
                "full-graph backend: expected a {} artifact, found {:?}",
                crate::layout::PLAN_FORMAT,
                plan.format
            ),
        });
    }
    let source = &plan.provenance;
    Ok(Provenance {
        compiler_version: source.compiler_version.clone(),
        git_revision: source
            .options
            .get("git_revision")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        dirty: source
            .options
            .get("dirty")
            .map(|v| v == "true")
            .unwrap_or(false),
        graph_schema_version: source
            .options
            .get("graph_schema_version")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        plan_schema_version: plan.schema_version,
        target: source.target_name.clone(),
        target_revision: source.target_revision.clone(),
        input_manifest_sha256: source.graph_sha256.clone(),
        weights_sha256: source
            .options
            .get("weights_sha256")
            .cloned()
            .unwrap_or_else(|| "not-recorded".to_string()),
        planner_config_sha256: source.policy_sha256.clone(),
    })
}

/// Machine-readable summary of the generated structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullGraphSummary {
    /// Input shape.
    pub input: Shape3,
    /// Output shape.
    pub output: Shape3,
    /// Arena words after lifetime reuse.
    pub arena_words: u64,
    /// Arena words if every slot stayed live (the no-reuse upper bound).
    pub arena_words_without_reuse: u64,
    /// Activation word width in bits.
    pub word_width: u32,
    /// Signed accumulator width.
    pub accumulator_width: u32,
    /// Worst-case accumulator magnitude the width is derived from.
    pub accumulator_bound: i64,
    /// Signed product path width of the requantizer.
    pub product_width: u32,
    /// Multipliers in the shared MAC array.
    pub mac_lanes: u64,
    /// Total multiply-accumulate count per frame.
    pub total_macs: u64,
    /// Unstalled beats of one frame in the generated FSM.
    pub cycles_unstalled: u64,
    /// Frame beats split into load / steps / drain.
    pub frame: FrameBeats,
    /// Sum of the plan's per-step cycle estimates.
    pub plan_cycles: u64,
    /// Per-step structure, beats and plan comparison.
    pub steps: Vec<FullGraphStepSummary>,
    /// Arena slots with their offsets.
    pub slots: Vec<FullGraphSlotSummary>,
    /// Honest notes about the structure, the plan and the numerical contract.
    pub semantic_notes: Vec<String>,
}

/// One step of [`FullGraphSummary`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullGraphStepSummary {
    /// Schedule index.
    pub index: usize,
    /// Operator name.
    pub op: String,
    /// Operator kind.
    pub kind: String,
    /// Multipliers the step sweeps.
    pub mac_lanes: u64,
    /// Multiply-accumulate count.
    pub macs: u64,
    /// Beats issuing operand read addresses.
    pub read_beats: u64,
    /// Beats accumulating.
    pub mac_beats: u64,
    /// Epilogue beats (overlapped).
    pub epilogue_beats: u64,
    /// Arena store beats (overlapped).
    pub store_beats: u64,
    /// Pipeline fill and drain beats.
    pub pipeline_beats: u64,
    /// Generated unstalled beats of the step.
    pub generated_cycles: u64,
    /// Plan-declared estimate.
    pub plan_estimated_cycles: u64,
}

/// One arena slot of [`FullGraphSummary`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullGraphSlotSummary {
    /// Tensor name.
    pub name: String,
    /// Signed word offset.
    pub offset: u64,
    /// Elements.
    pub elements: u64,
    /// Storage bytes (`elements * word_width / 8`).
    pub bytes: u64,
    /// Code domain.
    pub domain: CodeDomain,
    /// Largest magnitude the slot may hold.
    pub max_abs: i64,
    /// Inclusive `[first_step, last_step]` the slot must be resident for.
    pub live: [u64; 2],
}

fn full_graph_summary(model: &FullGraphModel, notes: &[String]) -> FullGraphSummary {
    FullGraphSummary {
        input: model.slots[model.input_slot].shape,
        output: model.slots[model.output_slot].shape,
        arena_words: model.arena_words,
        arena_words_without_reuse: model.arena_words_without_reuse,
        word_width: model.word_width,
        accumulator_width: model.acc_width,
        accumulator_bound: model.acc_bound,
        product_width: model.product_width,
        mac_lanes: model.mac_lanes,
        total_macs: model.total_macs,
        cycles_unstalled: model.cycles_unstalled,
        frame: model.frame,
        plan_cycles: model.plan_cycles,
        steps: model
            .steps
            .iter()
            .map(|s| FullGraphStepSummary {
                index: s.index,
                op: s.name.clone(),
                kind: format!("{:?}", s.op_kind),
                mac_lanes: s.lanes(),
                macs: s.macs,
                read_beats: s.beats.read_beats,
                mac_beats: s.beats.mac_beats,
                epilogue_beats: s.beats.epilogue_beats,
                store_beats: s.beats.store_beats,
                pipeline_beats: s.beats.pipeline_beats,
                generated_cycles: s.beats.total_beats,
                plan_estimated_cycles: s.plan_cycles,
            })
            .collect(),
        slots: model
            .slots
            .iter()
            .map(|s| FullGraphSlotSummary {
                name: s.name.clone(),
                offset: s.offset,
                elements: s.shape.elements(),
                bytes: s.shape.elements() * u64::from(model.word_width) / 8,
                domain: s.domain,
                max_abs: s.max_abs,
                live: [s.first_step, s.last_step],
            })
            .collect(),
        semantic_notes: notes.to_vec(),
    }
}

fn unsupported_summary() -> Vec<String> {
    vec![
        "grouped convolution other than dense (groups=1) and depthwise (groups=in_channels)"
            .to_string(),
        format!(
            "any fold other than the shared engine's single lane (simd = pe = tap_parallel = 1, \
             {SHARED_ENGINE_LANES} multiplier)"
        ),
        "row/tile scheduling granularity: the emitted engine makes one whole-frame pass per step"
            .to_string(),
        "asymmetric quantization (non-zero activation or weight zero-points)".to_string(),
        "rounding other than half away from zero".to_string(),
        "saturation bounds outside signed 16 bits".to_string(),
        "activation element types other than i8/u8".to_string(),
        "reflection padding that needs more than a single reflection step".to_string(),
        "a tensor the plan does not budget in its activation-RAM region, or an arena larger than \
         that region"
            .to_string(),
        "a plan whose steps do not correspond one-to-one with the graph's lowered operators"
            .to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Stimulus
// ---------------------------------------------------------------------------

/// Deterministic stimulus frames: `TESTBENCH_FRAMES` frames of graph input codes.
pub fn build_frames(model: &FullGraphModel, config: &RtlConfig) -> Vec<Vec<i64>> {
    let count = model.slots[model.input_slot].shape.elements();
    let mut frames = Vec::with_capacity(TESTBENCH_FRAMES);
    for frame in 0..TESTBENCH_FRAMES {
        let seed = config.fixture_seed ^ 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(frame as u64 + 1);
        let mut rng = Xorshift64::new(seed);
        let mut codes = Vec::with_capacity(count as usize);
        for _ in 0..count {
            codes.push(i64::from(rng.next_i8()));
        }
        frames.push(codes);
    }
    frames
}

/// One verification checkpoint of the generated testbench.
struct Checkpoint {
    /// Trace keyword: `input` or `step`.
    keyword: &'static str,
    /// Schedule index for step checkpoints (`usize::MAX` for the load).
    index: usize,
    /// Operator name.
    name: String,
    /// Element label written into the trace.
    elem: String,
    /// Shape written into the trace.
    shape: Shape3,
    /// Arena word offset of the checkpoint's codes.
    offset: u64,
    /// Number of codes.
    count: u64,
    /// Base of this checkpoint's block inside `expected.hex`.
    exp_base: u64,
}

/// The checkpoints a held frame verifies, in the order the FSM reaches them.
///
/// The FSM holds after the input load and after every step, so the codes read
/// through the debug port are the step's own outputs - with lifetime reuse the
/// arena a step wrote is overwritten later in the frame, so the read has to
/// happen at the boundary rather than after the frame.
fn build_checkpoints(model: &FullGraphModel, reference: &ReferenceRun) -> Vec<Checkpoint> {
    let mut checkpoints = Vec::with_capacity(model.steps.len() + 1);
    let input = &model.slots[model.input_slot];
    checkpoints.push(Checkpoint {
        keyword: "input",
        index: usize::MAX,
        name: input.name.clone(),
        elem: input.domain.as_str().to_string(),
        shape: input.shape,
        offset: input.offset,
        count: input.shape.elements(),
        exp_base: 0,
    });
    let mut base = input.shape.elements();
    for step in &model.steps {
        let slot = &model.slots[step.output];
        checkpoints.push(Checkpoint {
            keyword: "step",
            index: step.index,
            name: step.name.clone(),
            elem: element_label(model, step),
            shape: slot.shape,
            offset: slot.offset,
            count: slot.shape.elements(),
            exp_base: base,
        });
        base += slot.shape.elements();
    }
    let _ = reference;
    checkpoints
}

/// Trace element label of a step's stored codes.
pub fn element_label(model: &FullGraphModel, step: &ResolvedStep) -> String {
    let slot = &model.slots[step.output];
    if slot.domain == CodeDomain::U8 {
        return "u8".to_string();
    }
    if slot.max_abs > 128 {
        "i16".to_string()
    } else {
        "i8".to_string()
    }
}

// ---------------------------------------------------------------------------
// Emission: the DUT
// ---------------------------------------------------------------------------

fn bundle_header(provenance: &Provenance, note: &str, what: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// Generated by tessera {FULL_GRAPH_KERNEL_ID}. Do not edit by hand."
    );
    let _ = writeln!(
        out,
        "// {what}; one RAM-backed static-schedule engine over every plan step."
    );
    let _ = writeln!(
        out,
        "// compiler_version: {}  git_revision: {}  dirty: {}",
        provenance.compiler_version, provenance.git_revision, provenance.dirty
    );
    let _ = writeln!(
        out,
        "// target: {} ({})  plan_schema_version: {}  graph_schema_version: {}",
        provenance.target,
        provenance.target_revision,
        provenance.plan_schema_version,
        provenance.graph_schema_version
    );
    let _ = writeln!(
        out,
        "// input_manifest_sha256: {}  weights_digest: {}  planner_config_sha256: {}",
        provenance.input_manifest_sha256,
        provenance.weights_sha256,
        provenance.planner_config_sha256
    );
    let _ = writeln!(out, "// config: {note}");
    out
}

fn emit_dut(model: &FullGraphModel, dut_name: &str, provenance: &Provenance, note: &str) -> String {
    let sw = model.word_width;
    let aw = arena_address_bits(model);
    let accw = model.acc_width;
    let pw = model.product_width;
    let lprod = sw + 8;
    let states = model.steps.len() + 4;
    let sb = bits_for(states as u32).max(2);

    let weight_total = model
        .steps
        .iter()
        .map(|s| s.weights.len() as u64)
        .sum::<u64>();
    let bias_total = model
        .steps
        .iter()
        .map(|s| s.out_shape.c as u64)
        .sum::<u64>();
    let m0_total = model
        .steps
        .iter()
        .map(|s| s.requant.len() as u64)
        .sum::<u64>();
    // The packed ROMs are one arena per width class, each far deeper than the
    // activation arena can be; they need their own address widths, or a layer's
    // offset would be truncated to the arena's.
    let rom_bits = |words: u64| bits_for(u32::try_from(words.max(1)).unwrap_or(u32::MAX)).max(1);
    let waw = rom_bits(weight_total);
    let baw = rom_bits(bias_total);
    let maw = rom_bits(m0_total);

    let mut s = bundle_header(provenance, note, "full-graph engine");
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s, "module {dut_name} #(");
    let _ = writeln!(s, "    parameter int    AW = {aw},");
    let _ = writeln!(s, "    parameter string WFILE = \"\"");
    let _ = writeln!(s, ") (");
    let _ = writeln!(s, "    input  logic                  clk,");
    let _ = writeln!(s, "    input  logic                  rst,");
    let _ = writeln!(s, "    input  logic                  start,");
    let _ = writeln!(s, "    output logic                  done,");
    let _ = writeln!(s, "    input  logic [7:0]            in_data,");
    let _ = writeln!(s, "    input  logic                  in_valid,");
    let _ = writeln!(s, "    output logic                  in_ready,");
    let _ = writeln!(s, "    output logic [7:0]            out_data,");
    let _ = writeln!(s, "    output logic                  out_valid,");
    let _ = writeln!(s, "    input  logic                  out_ready,");
    let _ = writeln!(s, "    input  logic                  dbg_en,");
    let _ = writeln!(s, "    input  logic [AW-1:0]         dbg_addr,");
    let _ = writeln!(s, "    output logic signed [15:0]    dbg_data,");
    let _ = writeln!(s, "    input  logic                  dbg_hold_en,");
    let _ = writeln!(s, "    input  logic                  dbg_cont,");
    let _ = writeln!(s, "    output logic                  dbg_held");
    let _ = writeln!(s, ");");
    let _ = writeln!(s, "    // ---- geometry ----");
    let _ = writeln!(s, "    localparam int SW    = {sw};");
    let _ = writeln!(s, "    localparam int ACCW  = {accw};");
    let _ = writeln!(s, "    localparam int PW    = {pw};");
    let _ = writeln!(s, "    localparam int LPROD = {lprod};");
    let _ = writeln!(s, "    localparam int NW    = {};", model.arena_words);
    let _ = writeln!(
        s,
        "    localparam int NI    = {};",
        model.slots[model.input_slot].shape.elements()
    );
    let _ = writeln!(
        s,
        "    localparam int NO    = {};",
        model.slots[model.output_slot].shape.elements()
    );
    let _ = writeln!(
        s,
        "    localparam int IN_BASE  = {};",
        model.slots[model.input_slot].offset
    );
    let _ = writeln!(
        s,
        "    localparam int OUT_BASE = {};",
        model.slots[model.output_slot].offset
    );
    let _ = writeln!(s, "    localparam int WTOTAL = {};", weight_total.max(1));
    let _ = writeln!(s, "    localparam int BTOTAL = {};", bias_total.max(1));
    let _ = writeln!(s, "    localparam int MTOTAL = {};", m0_total.max(1));
    let _ = writeln!(
        s,
        "    localparam int LOAD_BEATS  = {};",
        model.frame.load_beats
    );
    let _ = writeln!(
        s,
        "    localparam int DRAIN_BEATS = {};",
        model.frame.drain_beats
    );
    let _ = writeln!(
        s,
        "    localparam int FRAME_BEATS = {};",
        model.cycles_unstalled
    );
    let _ = writeln!(
        s,
        "    localparam logic signed [ACCW-1:0] ACC_ZERO = {{ACCW{{1'b0}}}};"
    );
    let _ = writeln!(
        s,
        "    // store-value selectors of the one shared store datapath"
    );
    let _ = writeln!(s, "    localparam logic [2:0] KIND_ACC   = 3'd0;");
    let _ = writeln!(s, "    localparam logic [2:0] KIND_RQ    = 3'd1;");
    let _ = writeln!(s, "    localparam logic [2:0] KIND_COPY  = 3'd2;");
    let _ = writeln!(s, "    localparam logic [2:0] KIND_CLAMP = 3'd3;");
    let _ = writeln!(s, "    localparam logic [2:0] KIND_ADD   = 3'd4;");
    let _ = writeln!(
        s,
        "    // the fixed-point rounding unit of rq, as a PW-bit signed value"
    );
    let _ = writeln!(
        s,
        "    localparam logic signed [PW-1:0] PW_ONE = {{{{PW-1{{1'b0}}}}, 1'b1}};"
    );
    let _ = writeln!(s);
    for step in &model.steps {
        emit_step_constants(&mut s, step, model);
    }
    let _ = writeln!(s, "    // ---- state encoding ----");
    let mut state_names = vec!["S_IDLE".to_string(), "S_LOAD".to_string()];
    let mut state_index = std::collections::BTreeMap::new();
    for (name, index) in [("S_IDLE", 0usize), ("S_LOAD", 1usize)] {
        state_index.insert(name.to_string(), index);
    }
    for step in &model.steps {
        state_names.push(format!("S_OP{}", step.index));
        state_index.insert(format!("S_OP{}", step.index), state_names.len() - 1);
    }
    state_names.push("S_HOLD".to_string());
    state_index.insert("S_HOLD".to_string(), state_names.len() - 1);
    state_names.push("S_OUT".to_string());
    state_index.insert("S_OUT".to_string(), state_names.len() - 1);
    for (index, name) in state_names.iter().enumerate() {
        let _ = writeln!(
            s,
            "    localparam logic [{}-1:0] {name} = {index};",
            sb.max(2).max(2)
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // ---- one synchronous activation arena: one read port, one write port ----"
    );
    let _ = writeln!(s, "    logic signed [SW-1:0] act_mem [0:NW-1];");
    let _ = writeln!(s, "    logic [AW-1:0]        mem_raddr;");
    let _ = writeln!(s, "    logic                 mem_ren;");
    let _ = writeln!(s, "    logic signed [SW-1:0] mem_rdata;");
    let _ = writeln!(s, "    logic [AW-1:0]        mem_waddr;");
    let _ = writeln!(s, "    logic                 mem_wen;");
    let _ = writeln!(s, "    logic signed [SW-1:0] mem_wdata;");
    let _ = writeln!(
        s,
        "    // ---- packed constant ROMs (one memory each, per-layer offsets) ----"
    );
    let _ = writeln!(s, "    logic signed [7:0]      w_rom  [0:WTOTAL-1];");
    let _ = writeln!(s, "    logic signed [ACCW-1:0] b_rom  [0:BTOTAL-1];");
    let _ = writeln!(s, "    logic [16:0]            m0_rom [0:MTOTAL-1];");
    let _ = writeln!(s, "    logic [{waw}-1:0]         w_addr;");
    let _ = writeln!(s, "    logic signed [7:0]      w_data;");
    let _ = writeln!(s, "    logic [{baw}-1:0]         b_addr;");
    let _ = writeln!(s, "    logic signed [ACCW-1:0] b_data;");
    let _ = writeln!(s, "    logic [{maw}-1:0]         m_addr;");
    let _ = writeln!(s, "    logic [16:0]            m_data;");
    let _ = writeln!(s, "    always_ff @(posedge clk) begin");
    let _ = writeln!(s, "        if (mem_wen) act_mem[mem_waddr] <= mem_wdata;");
    let _ = writeln!(s, "        if (mem_ren) mem_rdata <= act_mem[mem_raddr];");
    let _ = writeln!(s, "        w_data <= w_rom[w_addr];");
    let _ = writeln!(s, "        b_data <= b_rom[b_addr];");
    let _ = writeln!(s, "        m_data <= m0_rom[m_addr];");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    emit_rom_contents(&mut s, model);
    let _ = writeln!(s);
    emit_functions(&mut s);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // ---- counters, flags and the one-deep store pipeline ----"
    );
    let _ = writeln!(s, "    logic [{}-1:0] state;", sb);
    let _ = writeln!(s, "    logic [{}-1:0] hold_next;", sb);
    let _ = writeln!(s, "    logic signed [31:0] cy, cx, oc, ic, t;");
    let _ = writeln!(s, "    logic signed [31:0] li, oi;");
    let _ = writeln!(s, "    logic [1:0] eof;");
    let _ = writeln!(s, "    logic [1:0] ad_ph;");
    let _ = writeln!(s, "    logic       out_ph;");
    let _ = writeln!(s, "    logic       done_r;");
    let _ = writeln!(s, "    logic [AW-1:0] el_mem_addr, el_st_addr;");
    let _ = writeln!(s, "    logic [{waw}-1:0] el_w_addr;");
    let _ = writeln!(s, "    logic [{baw}-1:0] el_b_addr;");
    let _ = writeln!(s, "    logic [{maw}-1:0] el_m_addr;");
    let _ = writeln!(
        s,
        "    logic          el_valid, el_ren, el_first, el_last, el_zero, el_store, el_a_capture;"
    );
    let _ = writeln!(s, "    logic [2:0]    el_kind;");
    let _ = writeln!(s, "    logic [4:0]    el_sh;");
    let _ = writeln!(s, "    logic signed [15:0] el_lo, el_hi;");
    for step in &model.steps {
        if step.is_conv() {
            let k = step.index;
            let _ = writeln!(s, "    logic signed [31:0] sy{k}, sx{k};");
            let _ = writeln!(s, "    logic ok{k};");
        }
    }

    let _ = writeln!(s, "    logic [4:0]             q_sh;");
    let _ = writeln!(s, "    logic signed [15:0]     q_lo, q_hi;");
    let _ = writeln!(s, "    logic [AW-1:0]          q_sa;");
    let _ = writeln!(s, "    logic [2:0]             q_kind;");
    let _ = writeln!(
        s,
        "    logic                   q_valid, q_first, q_last, q_zero;"
    );
    let _ = writeln!(s, "    logic signed [ACCW-1:0] acc, a_hold;");
    let _ = writeln!(s, "    logic signed [LPROD-1:0] x_ext, w_ext, lane_prod;");
    let _ = writeln!(s, "    logic [2:0]             st_kind;");
    let _ = writeln!(s, "    logic                   p_valid;");
    let _ = writeln!(s, "    logic [AW-1:0]          p_addr;");
    let _ = writeln!(s, "    logic signed [SW-1:0]   p_value;");
    let _ = writeln!(s, "    logic signed [ACCW-1:0] rq_in;");
    let _ = writeln!(s, "    logic signed [SW-1:0]   rq_out, st_value;");
    let _ = writeln!(s, "    logic                   dbg_sel;");
    let _ = writeln!(s, "    logic                   load_we;");
    let _ = writeln!(
        s,
        "    // ---- one shared multiplier: both operands are extended to LPROD first ----"
    );
    let _ = writeln!(
        s,
        "    // `mem_rdata` is the registered read of the element whose flags sit in q_*:"
    );
    let _ = writeln!(
        s,
        "    // the address stage of cycle k presents act_mem's address and the weight ROM's"
    );
    let _ = writeln!(
        s,
        "    // address for element k, both land in `mem_rdata`/`w_data` for cycle k+1, and the"
    );
    let _ = writeln!(
        s,
        "    // q_* flags were registered from that same cycle - so an extra operand register"
    );
    let _ = writeln!(
        s,
        "    // would hold element k-1 and drift from the flags it is paired with."
    );
    let _ = writeln!(
        s,
        "    always_comb begin x_ext = mem_rdata; w_ext = w_data; end"
    );
    let _ = writeln!(s, "    assign lane_prod = x_ext * w_ext;");
    if accw > lprod {
        let _ = writeln!(
            s,
            "    wire signed [ACCW-1:0] prod_acc = {{{{ACCW-LPROD{{lane_prod[LPROD-1]}}}}, lane_prod}};"
        );
    } else if accw == lprod {
        let _ = writeln!(s, "    wire signed [ACCW-1:0] prod_acc = lane_prod;");
    } else {
        let _ = writeln!(
            s,
            "    wire signed [ACCW-1:0] prod_acc = lane_prod[ACCW-1:0];"
        );
    }
    let _ = writeln!(
        s,
        "    wire signed [ACCW-1:0] acc_base = q_first ? b_data : acc;"
    );
    let _ = writeln!(
        s,
        "    wire signed [ACCW-1:0] acc_next = acc_base + (q_zero ? ACC_ZERO : prod_acc);"
    );
    let _ = writeln!(
        s,
        "    // ---- one shared requantizer behind the store pipeline ----"
    );
    let _ = writeln!(s, "    assign st_kind = el_store ? KIND_ADD : q_kind;");
    let _ = writeln!(s, "    always_comb begin");
    let _ = writeln!(s, "        rq_in = sx(mem_rdata);");
    let _ = writeln!(s, "        if (st_kind == KIND_ACC) rq_in = acc_next;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(
        s,
        "    assign rq_out = rq(rq_in, m_data, q_sh, q_lo, q_hi);"
    );
    let _ = writeln!(s, "    always_comb begin");
    let _ = writeln!(s, "        unique case (st_kind)");
    let _ = writeln!(s, "            KIND_ACC:   st_value = rq_out;");
    let _ = writeln!(s, "            KIND_RQ:    st_value = rq_out;");
    let _ = writeln!(s, "            KIND_COPY:  st_value = mem_rdata;");
    let _ = writeln!(
        s,
        "            KIND_CLAMP: st_value = cl16(mem_rdata, q_lo, q_hi);"
    );
    let _ = writeln!(s, "            KIND_ADD:   st_value = a_hold + rq_out;");
    let _ = writeln!(s, "            default:    st_value = rq_out;");
    let _ = writeln!(s, "        endcase");
    let _ = writeln!(s, "    end");
    let _ = writeln!(
        s,
        "    wire store_capture = (q_valid && q_last) || el_store;"
    );
    let _ = writeln!(s, "    // ---- ports ----");
    let _ = writeln!(
        s,
        "    assign load_we = (state == S_LOAD) && in_valid && in_ready;"
    );
    let _ = writeln!(s, "    assign mem_wen   = load_we || p_valid;");
    let _ = writeln!(
        s,
        "    assign mem_waddr = load_we ? (IN_BASE + li) : p_addr;"
    );
    let _ = writeln!(
        s,
        "    assign mem_wdata = load_we ? {{{{8{{in_data[7]}}}}, in_data}} : p_value;"
    );
    let _ = writeln!(
        s,
        "    assign dbg_sel = dbg_en && ((state == S_IDLE) || (state == S_HOLD));"
    );
    let _ = writeln!(
        s,
        "    assign mem_raddr = dbg_sel ? dbg_addr : el_mem_addr;"
    );
    let _ = writeln!(s, "    assign mem_ren   = dbg_sel ? 1'b1 : el_ren;");
    let _ = writeln!(s, "    assign w_addr = el_w_addr;");
    let _ = writeln!(s, "    assign b_addr = el_b_addr;");
    let _ = writeln!(s, "    assign m_addr = el_m_addr;");
    let _ = writeln!(
        s,
        "    assign dbg_data = dbg_sel ? mem_rdata : {{SW{{1'b0}}}};"
    );
    let _ = writeln!(s, "    assign dbg_held = (state == S_HOLD);");
    let _ = writeln!(s, "    assign in_ready = (state == S_LOAD);");
    let _ = writeln!(s, "    assign out_valid = (state == S_OUT) && out_ph;");
    let _ = writeln!(s, "    assign out_data = mem_rdata[7:0];");
    let _ = writeln!(s, "    assign done = done_r;");
    let _ = writeln!(s);
    emit_control_mux(&mut s, model);
    let _ = writeln!(s);
    emit_fsm(&mut s, model, &state_index);
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s, "`default_nettype wire");
    s
}

/// Address width of the emitted arena.
fn arena_address_bits(model: &FullGraphModel) -> u32 {
    bits_for(u32::try_from(model.arena_words).unwrap_or(u32::MAX))
        .max(1)
        .max(1)
}

fn emit_step_constants(out: &mut String, step: &ResolvedStep, model: &FullGraphModel) {
    let k = step.index;
    let _ = writeln!(
        out,
        "    // ---- step {k}: {} ({:?}), {} words at {} ----",
        step.name,
        step.op_kind,
        model.slots[step.output].shape.elements(),
        model.slots[step.output].offset
    );
    let _ = writeln!(
        out,
        "    localparam int S{k}_IB = {};",
        model.slots[step.inputs[0]].offset
    );
    let _ = writeln!(out, "    localparam int S{k}_IH = {};", step.in_shape.h);
    let _ = writeln!(out, "    localparam int S{k}_IW = {};", step.in_shape.w);
    let _ = writeln!(
        out,
        "    localparam int S{k}_OB = {};",
        model.slots[step.output].offset
    );
    let _ = writeln!(out, "    localparam int S{k}_OH = {};", step.out_shape.h);
    let _ = writeln!(out, "    localparam int S{k}_OW = {};", step.out_shape.w);
    let _ = writeln!(out, "    localparam int S{k}_OC = {};", step.out_shape.c);
    let _ = writeln!(
        out,
        "    localparam int S{k}_BEATS = {};",
        step.beats.total_beats
    );
    if let Some(second) = step.inputs.get(1) {
        let _ = writeln!(
            out,
            "    localparam int S{k}_IB2 = {};",
            model.slots[*second].offset
        );
    }
    if step.is_conv() {
        let _ = writeln!(out, "    localparam int S{k}_TAPS = {};", step.taps);
        let _ = writeln!(out, "    localparam int S{k}_ICG = {};", step.in_per_group);
        let _ = writeln!(out, "    localparam int S{k}_KH = {};", step.kernel[0]);
        let _ = writeln!(out, "    localparam int S{k}_KW = {};", step.kernel[1]);
        let _ = writeln!(out, "    localparam int S{k}_SH = {};", step.stride[0]);
        let _ = writeln!(out, "    localparam int S{k}_SW = {};", step.stride[1]);
        let _ = writeln!(
            out,
            "    localparam int S{k}_PAD = {};",
            step.border.amount()
        );
        let _ = writeln!(
            out,
            "    localparam bit S{k}_REFLECT = {};",
            if step.border.is_reflect() {
                "1'b1"
            } else {
                "1'b0"
            }
        );
        let _ = writeln!(
            out,
            "    localparam int S{k}_WB = {};",
            weight_base(model, k)
        );
        let _ = writeln!(out, "    localparam int S{k}_BB = {};", bias_base(model, k));
    }
    let _ = writeln!(
        out,
        "    localparam logic signed [15:0] S{k}_LO = {};",
        sv_signed(16, step.clamp.0)
    );
    let _ = writeln!(
        out,
        "    localparam logic signed [15:0] S{k}_HI = {};",
        sv_signed(16, step.clamp.1)
    );
    if !step.requant.is_empty() {
        let _ = writeln!(out, "    localparam int S{k}_MB = {};", m0_base(model, k));
        let _ = writeln!(
            out,
            "    localparam logic [4:0] S{k}_SHIFT = 5'd{};",
            step.requant[0].shift
        );
    } else {
        let _ = writeln!(out, "    localparam int S{k}_MB = 0;");
        let _ = writeln!(out, "    localparam logic [4:0] S{k}_SHIFT = 5'd0;");
    }
    let _ = writeln!(out);
}

fn weight_base(model: &FullGraphModel, step: usize) -> u64 {
    model
        .steps
        .iter()
        .take(step)
        .map(|s| s.weights.len() as u64)
        .sum()
}

fn bias_base(model: &FullGraphModel, step: usize) -> u64 {
    model
        .steps
        .iter()
        .take(step)
        .map(|s| s.out_shape.c as u64)
        .sum()
}

fn m0_base(model: &FullGraphModel, step: usize) -> u64 {
    model
        .steps
        .iter()
        .take(step)
        .map(|s| s.requant.len() as u64)
        .sum()
}

fn emit_rom_contents(out: &mut String, model: &FullGraphModel) {
    let accw = model.acc_width;
    let _ = writeln!(out, "    // ---- packed constant contents ----");
    let _ = writeln!(out, "    initial begin");
    let _ = writeln!(out, "        if (WFILE == \"\") begin");
    let mut index = 0u64;
    for step in &model.steps {
        if step.weights.is_empty() {
            continue;
        }
        let _ = writeln!(
            out,
            "            // step {} ({}) weights, (out_channel, in_per_group, kh, kw)",
            step.index, step.name
        );
        for weight in &step.weights {
            let _ = writeln!(
                out,
                "            w_rom[{index}] = {};",
                sv_signed(8, i64::from(*weight))
            );
            index += 1;
        }
    }
    let _ = writeln!(out, "        end else begin");
    let _ = writeln!(out, "            $readmemh(WFILE, w_rom);");
    let _ = writeln!(out, "        end");
    // The bias ROM is indexed by `bias_base(k) + output_channel`, and every step
    // reserves `out_shape.c` entries (an elementwise or upsample step reserves
    // zeroes) so the index of step k and the fill order cannot drift apart.
    let mut index = 0u64;
    for step in &model.steps {
        for channel in 0..step.out_shape.c {
            let value = step.bias.get(channel as usize).copied().unwrap_or(0);
            let _ = writeln!(
                out,
                "        b_rom[{index}] = {};",
                sv_signed(accw, i64::from(value))
            );
            index += 1;
        }
    }
    let mut index = 0u64;
    for step in &model.steps {
        for channel in &step.requant {
            let _ = writeln!(out, "        m0_rom[{index}] = 17'd{};", channel.multiplier);
            index += 1;
        }
    }
    let _ = writeln!(out, "    end");
}

fn emit_functions(out: &mut String) {
    let _ = writeln!(out, "    // ---- shared arithmetic ----");
    let _ = writeln!(
        out,
        "    // Sign-extend a 16-bit arena word into the accumulator domain."
    );
    let _ = writeln!(
        out,
        "    function automatic logic signed [ACCW-1:0] sx(input logic signed [SW-1:0] v);"
    );
    let _ = writeln!(out, "        begin sx = v; end");
    let _ = writeln!(out, "    endfunction");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "    // Saturate a 16-bit arena word (identity widths, kept explicit so a"
    );
    let _ = writeln!(
        out,
        "    // future widening cannot silently change the semantics)."
    );
    let _ = writeln!(
        out,
        "    function automatic logic signed [SW-1:0] cl16(input logic signed [SW-1:0] v,"
    );
    let _ = writeln!(
        out,
        "                                             input logic signed [15:0] lo,"
    );
    let _ = writeln!(
        out,
        "                                             input logic signed [15:0] hi);"
    );
    let _ = writeln!(out, "        begin");
    let _ = writeln!(out, "            if (v > hi) cl16 = hi;");
    let _ = writeln!(out, "            else if (v < lo) cl16 = lo;");
    let _ = writeln!(out, "            else cl16 = v;");
    let _ = writeln!(out, "        end");
    let _ = writeln!(out, "    endfunction");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "    // The one requantization datapath of the engine, shared by every store in"
    );
    let _ = writeln!(
        out,
        "    // the schedule: p = a*M0 ; r = p>=0 ? 2^(sh-1) : 2^(sh-1)-1 ;"
    );
    let _ = writeln!(
        out,
        "    // q = (p+r)>>>sh ; clamp(q, lo, hi). Both operands are extended to PW so"
    );
    let _ = writeln!(
        out,
        "    // the product is not truncated to an operand width."
    );
    let _ = writeln!(
        out,
        "    function automatic logic signed [SW-1:0] rq(input logic signed [ACCW-1:0] a_in,"
    );
    let _ = writeln!(
        out,
        "                                                input logic [16:0] m0,"
    );
    let _ = writeln!(
        out,
        "                                                input logic [4:0] sh,"
    );
    let _ = writeln!(
        out,
        "                                                input logic signed [15:0] lo_in,"
    );
    let _ = writeln!(
        out,
        "                                                input logic signed [15:0] hi_in);"
    );
    let _ = writeln!(out, "        logic signed [PW-1:0] a, m, p, r, q, lo, hi;");
    let _ = writeln!(out, "        begin");
    let _ = writeln!(out, "            a  = a_in;");
    out.push_str("            m  = {{(PW-17){1'b0}}, m0};\n");
    let _ = writeln!(out, "            lo = lo_in;");
    let _ = writeln!(out, "            hi = hi_in;");
    let _ = writeln!(out, "            p  = a * m;");
    out.push_str(
        "            r  = (sh == 5'd0) ? {PW{1'b0}} : ((p >= 0) ? (PW_ONE << (sh - 5'd1)) : ((PW_ONE << (sh - 5'd1)) - PW_ONE));\n",
    );
    let _ = writeln!(out, "            q  = (p + r) >>> sh;");
    let _ = writeln!(out, "            if (q > hi) q = hi;");
    let _ = writeln!(out, "            else if (q < lo) q = lo;");
    let _ = writeln!(out, "            rq = q[SW-1:0];");
    let _ = writeln!(out, "        end");
    let _ = writeln!(out, "    endfunction");
}

/// The one combinational schedule mux: which step's addresses and controls the
/// shared read port, the constant ROMs and the MAC stage see.
fn emit_control_mux(out: &mut String, model: &FullGraphModel) {
    let _ = writeln!(
        out,
        "    // ---- schedule control: one read port and one store port, muxed per step ----"
    );
    let _ = writeln!(out, "    always_comb begin");
    let _ = writeln!(
        out,
        "        el_valid = 1'b0; el_ren = 1'b0; el_zero = 1'b0;"
    );
    let _ = writeln!(
        out,
        "        el_first = 1'b0; el_last = 1'b0; el_store = 1'b0; el_a_capture = 1'b0;"
    );
    let _ = writeln!(
        out,
        "        el_kind = KIND_ACC; el_sh = 5'd0; el_lo = 16'sd0; el_hi = 16'sd0;"
    );
    let _ = writeln!(
        out,
        "        el_mem_addr = '0; el_w_addr = '0; el_b_addr = '0; el_m_addr = '0; el_st_addr = '0;"
    );
    let _ = writeln!(out, "        unique case (state)");
    let _ = writeln!(out, "            S_LOAD: begin end");
    for step in &model.steps {
        emit_step_control(out, step);
    }
    let _ = writeln!(out, "            S_OUT: begin");
    let _ = writeln!(out, "                if (!out_ph) begin");
    let _ = writeln!(out, "                    el_ren = 1'b1;");
    let _ = writeln!(out, "                    el_mem_addr = OUT_BASE + oi;");
    let _ = writeln!(out, "                end");
    let _ = writeln!(out, "            end");
    let _ = writeln!(out, "            default: begin end");
    let _ = writeln!(out, "        endcase");
    let _ = writeln!(out, "    end");
}

/// Address and control one step presents to the shared engine.
fn emit_step_control(out: &mut String, step: &ResolvedStep) {
    let k = step.index;
    let _ = writeln!(out, "            S_OP{k}: begin");
    match step.op_kind {
        OpKind::Conv2d | OpKind::PointwiseConv2d | OpKind::DepthwiseConv2d => {
            let depthwise = step.is_depthwise();
            let _ = writeln!(out, "                if (eof == 2'd0) begin");
            let _ = writeln!(out, "                    el_valid = 1'b1;");
            if depthwise {
                let _ = writeln!(out, "                    el_first = (t == 0);");
                let _ = writeln!(out, "                    el_last  = (t == S{k}_TAPS-1);");
            } else {
                let _ = writeln!(out, "                    el_first = (ic == 0) && (t == 0);");
                let _ = writeln!(
                    out,
                    "                    el_last  = (ic == S{k}_ICG-1) && (t == S{k}_TAPS-1);"
                );
            }
            let _ = writeln!(out, "                    el_kind  = KIND_ACC;");
            let _ = writeln!(
                out,
                "                    el_lo = S{k}_LO; el_hi = S{k}_HI; el_sh = S{k}_SHIFT;"
            );
            let _ = writeln!(out, "                    el_m_addr = S{k}_MB + oc;");
            let _ = writeln!(out, "                    el_b_addr = S{k}_BB + oc;");
            if depthwise {
                let _ = writeln!(
                    out,
                    "                    el_w_addr = S{k}_WB + oc*S{k}_TAPS + t;"
                );
            } else {
                let _ = writeln!(
                    out,
                    "                    el_w_addr = S{k}_WB + ((oc*S{k}_ICG + ic)*S{k}_TAPS + t);"
                );
            }
            let _ = writeln!(
                out,
                "                    sy{k} = 0; sx{k} = 0; ok{k} = 1'b1;"
            );
            let _ = writeln!(out, "                    unique case (t)");
            for tap in 0..step.taps as usize {
                let _ = writeln!(out, "                        {tap}: begin");
                emit_tap_body(out, step, k, tap);
                let _ = writeln!(out, "                        end");
            }
            let _ = writeln!(
                out,
                "                        default: begin ok{k} = 1'b0; sy{k} = 0; sx{k} = 0; end"
            );
            let _ = writeln!(out, "                    endcase");
            let _ = writeln!(out, "                    el_zero = ~ok{k};");
            if depthwise {
                let _ = writeln!(
                    out,
                    "                    el_mem_addr = ok{k} ? (S{k}_IB + (oc*S{k}_IH + sy{k})*S{k}_IW + sx{k}) : '0;"
                );
            } else {
                let _ = writeln!(
                    out,
                    "                    el_mem_addr = ok{k} ? (S{k}_IB + (ic*S{k}_IH + sy{k})*S{k}_IW + sx{k}) : '0;"
                );
            }
            let _ = writeln!(out, "                    el_ren = 1'b1;");
            let _ = writeln!(
                out,
                "                    el_st_addr = S{k}_OB + (oc*S{k}_OH + cy)*S{k}_OW + cx;"
            );
            let _ = writeln!(out, "                end");
        }
        OpKind::Add => {
            let _ = writeln!(out, "                if (eof == 2'd0) begin");
            let _ = writeln!(
                out,
                "                    el_lo = S{k}_LO; el_hi = S{k}_HI; el_sh = S{k}_SHIFT;"
            );
            let _ = writeln!(out, "                    el_m_addr = S{k}_MB;");
            let _ = writeln!(
                out,
                "                    el_st_addr = S{k}_OB + (oc*S{k}_OH + cy)*S{k}_OW + cx;"
            );
            let _ = writeln!(out, "                    unique case (ad_ph)");
            let _ = writeln!(out, "                        2'd0: begin");
            let _ = writeln!(out, "                            el_ren = 1'b1;");
            let _ = writeln!(
                out,
                "                            el_mem_addr = S{k}_IB + (oc*S{k}_IH + cy)*S{k}_IW + cx;"
            );
            let _ = writeln!(out, "                        end");
            let _ = writeln!(out, "                        2'd1: begin");
            let _ = writeln!(out, "                            el_ren = 1'b1;");
            let _ = writeln!(
                out,
                "                            el_mem_addr = S{k}_IB2 + (oc*S{k}_IH + cy)*S{k}_IW + cx;"
            );
            let _ = writeln!(out, "                            el_a_capture = 1'b1;");
            let _ = writeln!(out, "                        end");
            let _ = writeln!(out, "                        default: begin");
            let _ = writeln!(out, "                            el_store = 1'b1;");
            let _ = writeln!(
                out,
                "                            el_ren = (cx + 1 < S{k}_OW) || (cy + 1 < S{k}_OH) || (oc + 1 < S{k}_OC);"
            );
            let _ = writeln!(
                out,
                "                            el_mem_addr = (cx + 1 < S{k}_OW)"
            );
            let _ = writeln!(
                out,
                "                                ? (S{k}_IB + ((oc*S{k}_OH + cy)*S{k}_OW + cx) + 1)"
            );
            let _ = writeln!(out, "                                : ((cy + 1 < S{k}_OH)");
            let _ = writeln!(
                out,
                "                                    ? (S{k}_IB + (oc*S{k}_OH + cy + 1)*S{k}_OW)"
            );
            let _ = writeln!(
                out,
                "                                    : (S{k}_IB + (oc + 1)*S{k}_OH*S{k}_OW));"
            );
            let _ = writeln!(out, "                        end");
            let _ = writeln!(out, "                    endcase");
            let _ = writeln!(out, "                end");
        }
        OpKind::NearestUpsample2x => {
            let _ = writeln!(out, "                if (eof == 2'd0) begin");
            let _ = writeln!(out, "                    el_valid = 1'b1;");
            let _ = writeln!(out, "                    el_first = 1'b1; el_last = 1'b1;");
            let _ = writeln!(out, "                    el_kind = KIND_COPY;");
            let _ = writeln!(
                out,
                "                    el_mem_addr = S{k}_IB + (oc*S{k}_IH + (cy >> 1))*S{k}_IW + (cx >> 1);"
            );
            let _ = writeln!(out, "                    el_ren = 1'b1;");
            let _ = writeln!(
                out,
                "                    el_st_addr = S{k}_OB + (oc*S{k}_OH + cy)*S{k}_OW + cx;"
            );
            let _ = writeln!(out, "                end");
        }
        OpKind::Requant | OpKind::Activation | OpKind::QuantizeOutput => {
            let _ = writeln!(out, "                if (eof == 2'd0) begin");
            let _ = writeln!(out, "                    el_valid = 1'b1;");
            let _ = writeln!(out, "                    el_first = 1'b1; el_last = 1'b1;");
            if step.requant.is_empty() {
                let _ = writeln!(out, "                    el_kind = KIND_CLAMP;");
            } else {
                let _ = writeln!(out, "                    el_kind = KIND_RQ;");
            }
            let _ = writeln!(
                out,
                "                    el_lo = S{k}_LO; el_hi = S{k}_HI; el_sh = S{k}_SHIFT;"
            );
            let _ = writeln!(out, "                    el_m_addr = S{k}_MB;");
            let _ = writeln!(
                out,
                "                    el_mem_addr = S{k}_IB + (oc*S{k}_IH + cy)*S{k}_IW + cx;"
            );
            let _ = writeln!(out, "                    el_ren = 1'b1;");
            let _ = writeln!(
                out,
                "                    el_st_addr = S{k}_OB + (oc*S{k}_OH + cy)*S{k}_OW + cx;"
            );
            let _ = writeln!(out, "                end");
        }
    }
    let _ = writeln!(out, "            end");
}

/// One tap of a convolution's index decoder: source pixel and validity.
fn emit_tap_body(out: &mut String, step: &ResolvedStep, k: usize, tap: usize) {
    let kw = step.kernel[1] as usize;
    let ky = (tap / kw) as i64;
    let kx = (tap % kw) as i64;
    let pad = i64::from(step.border.amount());
    let _ = writeln!(
        out,
        "                            sy{k} = cy*{} + ({});",
        step.stride[0],
        ky - pad
    );
    let _ = writeln!(
        out,
        "                            sx{k} = cx*{} + ({});",
        step.stride[1],
        kx - pad
    );
    let ih = step.in_shape.h as i64;
    let iw = step.in_shape.w as i64;
    if step.border.is_reflect() {
        let _ = writeln!(
            out,
            "                            if (sy{k} < 0) sy{k} = -sy{k};"
        );
        let _ = writeln!(
            out,
            "                            if (sy{k} >= {ih}) sy{k} = {} - sy{k};",
            2 * (ih - 1)
        );
        let _ = writeln!(
            out,
            "                            if (sx{k} < 0) sx{k} = -sx{k};"
        );
        let _ = writeln!(
            out,
            "                            if (sx{k} >= {iw}) sx{k} = {} - sx{k};",
            2 * (iw - 1)
        );
        let _ = writeln!(out, "                            ok{k} = 1'b1;");
    } else {
        let _ = writeln!(
            out,
            "                            ok{k} = (sy{k} >= 0) && (sy{k} < {ih}) && (sx{k} >= 0) && (sx{k} < {iw});"
        );
    }
}

fn emit_state_width(model: &FullGraphModel) -> u32 {
    bits_for((model.steps.len() + 4) as u32).max(2)
}

fn emit_fsm(
    out: &mut String,
    model: &FullGraphModel,
    _state_index: &std::collections::BTreeMap<String, usize>,
) {
    let sb = emit_state_width(model);
    let _ = writeln!(out, "    // ---- static schedule FSM ----");
    let _ = writeln!(out, "    always_ff @(posedge clk) begin");
    let _ = writeln!(out, "        if (rst) begin");
    let _ = writeln!(out, "            state <= S_IDLE;");
    let _ = writeln!(out, "            hold_next <= S_IDLE;");
    let _ = writeln!(
        out,
        "            cy <= 0; cx <= 0; oc <= 0; ic <= 0; t <= 0;"
    );
    let _ = writeln!(
        out,
        "            li <= 0; oi <= 0; eof <= 2'd0; ad_ph <= 2'd0; out_ph <= 1'b0;"
    );
    let _ = writeln!(out, "            done_r <= 1'b0; p_valid <= 1'b0;");
    let _ = writeln!(out, "            acc <= ACC_ZERO; a_hold <= ACC_ZERO;");
    let _ = writeln!(
        out,
        "            q_valid <= 1'b0; q_first <= 1'b0; q_last <= 1'b0; q_zero <= 1'b0;"
    );
    let _ = writeln!(out, "        end else begin");
    let _ = writeln!(
        out,
        "            // address stage -> MAC stage (one element of latency)"
    );
    let _ = writeln!(out, "            q_valid <= el_valid;");
    let _ = writeln!(out, "            q_first <= el_first;");
    let _ = writeln!(out, "            q_last  <= el_last;");
    let _ = writeln!(out, "            q_zero  <= el_zero;");
    let _ = writeln!(out, "            q_kind  <= el_kind;");
    let _ = writeln!(out, "            q_sh    <= el_sh;");
    let _ = writeln!(out, "            q_lo    <= el_lo;");
    let _ = writeln!(out, "            q_hi    <= el_hi;");
    let _ = writeln!(out, "            q_sa    <= el_st_addr;");
    let _ = writeln!(
        out,
        "            if (el_a_capture) a_hold <= sx(mem_rdata);"
    );
    let _ = writeln!(out, "            // the shared MAC stage");
    let _ = writeln!(out, "            if (q_valid) acc <= acc_next;");
    let _ = writeln!(out, "            // the one-deep store pipeline");
    let _ = writeln!(out, "            p_valid <= store_capture;");
    let _ = writeln!(out, "            if (store_capture) begin");
    let _ = writeln!(
        out,
        "                p_addr  <= el_store ? el_st_addr : q_sa;"
    );
    let _ = writeln!(out, "                p_value <= st_value;");
    let _ = writeln!(out, "            end");
    let _ = writeln!(out, "            unique case (state)");
    let _ = writeln!(out, "                S_IDLE: begin");
    let _ = writeln!(out, "                    if (start) begin");
    let _ = writeln!(out, "                        done_r <= 1'b0;");
    let _ = writeln!(
        out,
        "                        cy <= 0; cx <= 0; oc <= 0; ic <= 0; t <= 0; li <= 0; oi <= 0;"
    );
    let _ = writeln!(
        out,
        "                        eof <= 2'd0; ad_ph <= 2'd0; out_ph <= 1'b0;"
    );
    let _ = writeln!(
        out,
        "                        acc <= ACC_ZERO; a_hold <= ACC_ZERO;"
    );
    let _ = writeln!(out, "                        state <= S_LOAD;");
    let _ = writeln!(out, "                    end");
    let _ = writeln!(out, "                end");
    let _ = writeln!(out, "                S_LOAD: begin");
    let _ = writeln!(out, "                    if (in_valid && in_ready) begin");
    let _ = writeln!(out, "                        if (li + 1 == NI) begin");
    let _ = writeln!(out, "                            li <= 0;");
    emit_go(out, 28, "S_OP0", &["eof <= 2'd0;", "ad_ph <= 2'd0;"]);
    let _ = writeln!(out, "                        end else li <= li + 1;");
    let _ = writeln!(out, "                    end");
    let _ = writeln!(out, "                end");
    let _ = writeln!(out, "                S_HOLD: begin");
    let _ = writeln!(
        out,
        "                    if (dbg_cont || !dbg_hold_en) state <= hold_next;"
    );
    let _ = writeln!(out, "                end");
    for step in &model.steps {
        emit_step_fsm(out, model, step);
    }
    let _ = writeln!(out, "                S_OUT: begin");
    let _ = writeln!(out, "                    if (!out_ph) out_ph <= 1'b1;");
    let _ = writeln!(out, "                    else if (out_ready) begin");
    let _ = writeln!(out, "                        if (oi + 1 < NO) begin");
    let _ = writeln!(
        out,
        "                            oi <= oi + 1; out_ph <= 1'b0;"
    );
    let _ = writeln!(out, "                        end else begin");
    let _ = writeln!(
        out,
        "                            oi <= 0; out_ph <= 1'b0; done_r <= 1'b1;"
    );
    let _ = writeln!(out, "                            state <= S_IDLE;");
    let _ = writeln!(out, "                        end");
    let _ = writeln!(out, "                    end");
    let _ = writeln!(out, "                end");
    let _ = writeln!(out, "                default: state <= S_IDLE;");
    let _ = writeln!(out, "            endcase");
    let _ = writeln!(out, "        end");
    let _ = writeln!(out, "    end");
    let _ = writeln!(out, "    // state width: {sb} bits");
}

/// Step boundary: run the next step, or hold there when the verification
/// single-step is enabled.
fn emit_go(out: &mut String, indent: usize, next: &str, resets: &[&str]) {
    let pad = " ".repeat(indent);
    let _ = writeln!(
        out,
        "{pad}if (dbg_hold_en) begin state <= S_HOLD; hold_next <= {next}; end"
    );
    let _ = writeln!(out, "{pad}else state <= {next};");
    for reset in resets {
        let _ = writeln!(out, "{pad}{reset}");
    }
}

fn emit_step_fsm(out: &mut String, model: &FullGraphModel, step: &ResolvedStep) {
    let k = step.index;
    let next = if k + 1 < model.steps.len() {
        format!("S_OP{}", k + 1)
    } else {
        "S_OUT".to_string()
    };
    let mut resets: Vec<String> = vec!["eof <= 2'd0;".to_string(), "ad_ph <= 2'd0;".to_string()];
    if next == "S_OUT" {
        resets.push("out_ph <= 1'b0;".to_string());
    }
    let resets: Vec<&str> = resets.iter().map(|s| s.as_str()).collect();
    let _ = writeln!(out, "                S_OP{k}: begin");
    match step.op_kind {
        OpKind::Conv2d | OpKind::PointwiseConv2d | OpKind::DepthwiseConv2d => {
            let dims: Vec<(String, u64)> = if step.is_depthwise() {
                vec![
                    ("t".to_string(), step.taps),
                    ("oc".to_string(), u64::from(step.out_shape.c)),
                    ("cx".to_string(), u64::from(step.out_shape.w)),
                    ("cy".to_string(), u64::from(step.out_shape.h)),
                ]
            } else {
                vec![
                    ("t".to_string(), step.taps),
                    ("ic".to_string(), u64::from(step.in_per_group)),
                    ("oc".to_string(), u64::from(step.out_shape.c)),
                    ("cx".to_string(), u64::from(step.out_shape.w)),
                    ("cy".to_string(), u64::from(step.out_shape.h)),
                ]
            };
            let _ = writeln!(out, "                    if (eof == 2'd0) begin");
            emit_advance(out, 24, &dims, "eof <= 2'd1;", 0);
            let _ = writeln!(
                out,
                "                    end else if (eof == 2'd1) eof <= 2'd2;"
            );
            let _ = writeln!(out, "                    else begin");
            emit_go(out, 24, &next, &resets);
            let _ = writeln!(out, "                    end");
        }
        OpKind::Add => {
            let dims: Vec<(String, u64)> = vec![
                ("cx".to_string(), u64::from(step.out_shape.w)),
                ("cy".to_string(), u64::from(step.out_shape.h)),
                ("oc".to_string(), u64::from(step.out_shape.c)),
            ];
            let _ = writeln!(out, "                    if (eof == 2'd0) begin");
            let _ = writeln!(out, "                        unique case (ad_ph)");
            let _ = writeln!(out, "                            2'd0: ad_ph <= 2'd1;");
            let _ = writeln!(out, "                            2'd1: ad_ph <= 2'd2;");
            let _ = writeln!(out, "                            default: begin");
            let _ = writeln!(out, "                                ad_ph <= 2'd1;");
            emit_advance(out, 32, &dims, "eof <= 2'd1;", 0);
            let _ = writeln!(out, "                            end");
            let _ = writeln!(out, "                        endcase");
            let _ = writeln!(out, "                    end else begin");
            emit_go(out, 24, &next, &resets);
            let _ = writeln!(out, "                    end");
        }
        OpKind::NearestUpsample2x
        | OpKind::Requant
        | OpKind::Activation
        | OpKind::QuantizeOutput => {
            let dims: Vec<(String, u64)> = vec![
                ("cx".to_string(), u64::from(step.out_shape.w)),
                ("cy".to_string(), u64::from(step.out_shape.h)),
                ("oc".to_string(), u64::from(step.out_shape.c)),
            ];
            let _ = writeln!(out, "                    if (eof == 2'd0) begin");
            emit_advance(out, 24, &dims, "eof <= 2'd1;", 0);
            let _ = writeln!(
                out,
                "                    end else if (eof == 2'd1) eof <= 2'd2;"
            );
            let _ = writeln!(out, "                    else begin");
            emit_go(out, 24, &next, &resets);
            let _ = writeln!(out, "                    end");
        }
    }
    let _ = writeln!(out, "                end");
}

fn emit_advance(out: &mut String, indent: usize, dims: &[(String, u64)], tail: &str, at: usize) {
    let pad = " ".repeat(indent);
    let (name, size) = &dims[at];
    let _ = writeln!(out, "{pad}if ({name} + 1 < {size}) {name} <= {name} + 1;");
    let _ = writeln!(out, "{pad}else begin");
    let _ = writeln!(out, "{pad}    {name} <= 0;");
    if at + 1 == dims.len() {
        let _ = writeln!(out, "{pad}    {tail}");
    } else {
        emit_advance(out, indent + 4, dims, tail, at + 1);
    }
    let _ = writeln!(out, "{pad}end");
}

// ---------------------------------------------------------------------------
// Emission: images, reference text and the testbench
// ---------------------------------------------------------------------------

fn emit_weight_image(model: &FullGraphModel) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// tessera {FULL_GRAPH_KERNEL_ID} weight ROM image (i8, one per line)"
    );
    for step in &model.steps {
        if step.weights.is_empty() {
            continue;
        }
        let _ = writeln!(out, "// step {}: {}", step.index, step.name);
        for weight in &step.weights {
            let _ = writeln!(out, "{:02x}", (*weight as u8) as u32);
        }
    }
    out
}

fn emit_stimulus_image(frames: &[Vec<i64>]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// tessera {FULL_GRAPH_KERNEL_ID} stimulus: {} frames of graph input codes, NCHW, one i8 per line",
        frames.len()
    );
    for (index, frame) in frames.iter().enumerate() {
        let _ = writeln!(out, "// frame {index}");
        for code in frame {
            let _ = writeln!(out, "{:02x}", (*code as u8) as u32);
        }
    }
    out
}

fn emit_expected_image(run: &ReferenceRun) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "// tessera {FULL_GRAPH_KERNEL_ID} reference: 16-bit two's-complement, frame-major then step-major"
    );
    for frame in &run.frames {
        let _ = writeln!(out, "// frame {} input", frame.index);
        for code in &frame.input {
            let _ = writeln!(out, "{:04x}", (*code as u16) as u32);
        }
        for step in &frame.steps {
            let _ = writeln!(
                out,
                "// frame {} step {} {}",
                frame.index, step.index, step.name
            );
            for code in &step.codes {
                let _ = writeln!(out, "{:04x}", (*code as u16) as u32);
            }
        }
    }
    out
}

fn emit_reference_text(
    model: &FullGraphModel,
    run: &ReferenceRun,
    divergence: &[reference::StepDivergence],
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# tessera {FULL_GRAPH_KERNEL_ID} scalar reference (integer datapath, bit-exact against the HDL)"
    );
    let _ = writeln!(out, "# graph: {}", model.name);
    let _ = writeln!(
        out,
        "# arena: {} words x {} bits ({} words without lifetime reuse), {} multiplier(s), \
         acc_width {} (|acc| <= {}), product_width {}",
        model.arena_words,
        model.word_width,
        model.arena_words_without_reuse,
        model.mac_lanes,
        model.acc_width,
        model.acc_bound,
        model.product_width
    );
    let _ = writeln!(
        out,
        "# frame beats {} = load {} + steps {} + drain {} (plan estimate {})",
        model.cycles_unstalled,
        model.frame.load_beats,
        model.frame.step_beats,
        model.frame.drain_beats,
        model.plan_cycles
    );
    let _ = writeln!(out, "# total_macs {}", model.total_macs);
    let _ = writeln!(out, "# steps:");
    for step in &model.steps {
        let _ = writeln!(
            out,
            "#   {} {} lanes={} macs={} read_beats={} mac_beats={} epilogue_beats={} \
             store_beats={} pipeline_beats={} generated_cycles={} plan_estimated_cycles={}",
            step.index,
            step.name,
            step.lanes(),
            step.macs,
            step.beats.read_beats,
            step.beats.mac_beats,
            step.beats.epilogue_beats,
            step.beats.store_beats,
            step.beats.pipeline_beats,
            step.beats.total_beats,
            step.plan_cycles
        );
    }
    let _ = writeln!(out, "# slots:");
    for slot in &model.slots {
        let _ = writeln!(
            out,
            "#   {} offset={} shape={} domain={} max_abs={}",
            slot.name,
            slot.offset,
            slot.shape.label(),
            slot.domain.as_str(),
            slot.max_abs
        );
    }
    let _ = writeln!(
        out,
        "# manifest f32 vs emitted fixed point (NOT a bit-exact claim):"
    );
    if divergence.is_empty() {
        let _ = writeln!(out, "#   no step differs on this stimulus");
    }
    for entry in divergence {
        let _ = writeln!(
            out,
            "#   step {} {}: {}/{} codes differ, max |difference| {}",
            entry.index, entry.name, entry.differing, entry.total, entry.max_abs_diff
        );
    }
    for note in &model.notes {
        let _ = writeln!(out, "# note: {note}");
    }
    let _ = writeln!(out, "format tessera.reference_run.v1");
    for frame in &run.frames {
        let _ = writeln!(out, "frame {} input {}", frame.index, frame.input.len());
        writeln_codes(&mut out, &frame.input);
        for step in &frame.steps {
            let _ = writeln!(
                out,
                "frame {} step {} {} {} {}x{}x{}",
                frame.index,
                step.index,
                step.name,
                step.domain.as_str(),
                step.shape.c,
                step.shape.h,
                step.shape.w
            );
            writeln_codes(&mut out, &step.codes);
        }
        let _ = writeln!(out, "frame {} cycles {}", frame.index, frame.cycles);
    }
    out
}

fn writeln_codes(out: &mut String, codes: &[i64]) {
    for code in codes {
        let _ = writeln!(out, "{code}");
    }
}

/// Self-checking testbench.
///
/// Frame 0 measures the unstalled schedule: no holds, no input gaps, `out_ready`
/// held high, so the FSM's beats are the model's beats and the testbench
/// compares the counter against `FRAME_BEATS`.
///
/// Frame 1 verifies the arena: `dbg_hold_en` makes the FSM stop after the input
/// load and after every step, so each step's codes can be read through the
/// engine's single (shared) debug port before a later step reuses those words.
/// The same frame injects input gaps and output backpressure and checks that the
/// presented output beat is stable while `out_ready` is low; those stall cycles
/// are counted separately and are never folded into the schedule number.
fn emit_testbench(
    model: &FullGraphModel,
    dut_name: &str,
    tb_name: &str,
    provenance: &Provenance,
    reference: &ReferenceRun,
    checkpoints: &[Checkpoint],
) -> String {
    let aw = arena_address_bits(model);
    let frames = reference.frames.len().max(1);
    let ni = model.slots[model.input_slot].shape.elements();
    let no = model.slots[model.output_slot].shape.elements();
    let total: u64 = reference
        .frames
        .first()
        .map(|frame| {
            frame.input.len() as u64
                + frame
                    .steps
                    .iter()
                    .map(|s| s.codes.len() as u64)
                    .sum::<u64>()
        })
        .unwrap_or(1);
    // `expected.hex` is frame-major: the input block, then one block per step.
    // The output port streams the last step's codes, so their base is the input
    // block plus every earlier step's block.
    let output_exp_base = ni
        + model
            .steps
            .iter()
            .take(model.steps.len().saturating_sub(1))
            .map(|s| s.out_shape.elements())
            .sum::<u64>();
    let timeout = model
        .cycles_unstalled
        .saturating_mul(4)
        .saturating_add(
            model
                .arena_words
                .saturating_mul(8)
                .saturating_mul(frames as u64),
        )
        .saturating_add(50_000);

    let mut s = bundle_header(
        provenance,
        "self-checking testbench",
        "full-graph testbench",
    );
    let _ = writeln!(s, "`default_nettype none");
    let _ = writeln!(s, "`timescale 1ns/1ps");
    let _ = writeln!(s, "module {tb_name};");
    let _ = writeln!(s, "    localparam int AW = {aw};");
    let _ = writeln!(s, "    localparam int NF = {frames};");
    let _ = writeln!(s, "    localparam int NI = {ni};");
    let _ = writeln!(s, "    localparam int NO = {no};");
    let _ = writeln!(s, "    localparam int TOTAL = {total};");
    let _ = writeln!(s, "    localparam int NCP = {};", checkpoints.len());
    let _ = writeln!(
        s,
        "    localparam int FRAME_BEATS = {};",
        model.cycles_unstalled
    );
    let _ = writeln!(s, "    localparam int TIMEOUT = {timeout};");
    let _ = writeln!(s, "    logic clk = 1'b0;");
    let _ = writeln!(s, "    logic rst = 1'b1;");
    let _ = writeln!(s, "    logic start = 1'b0;");
    let _ = writeln!(s, "    logic in_valid = 1'b0;");
    let _ = writeln!(s, "    logic in_ready;");
    let _ = writeln!(s, "    logic [7:0] in_data = 8'h00;");
    let _ = writeln!(s, "    logic out_valid;");
    let _ = writeln!(s, "    logic out_ready = 1'b0;");
    let _ = writeln!(s, "    logic [7:0] out_data;");
    let _ = writeln!(s, "    logic done;");
    let _ = writeln!(s, "    logic dbg_en = 1'b0;");
    let _ = writeln!(s, "    logic [AW-1:0] dbg_addr = '0;");
    let _ = writeln!(s, "    logic signed [15:0] dbg_data;");
    let _ = writeln!(s, "    logic dbg_hold_en = 1'b0;");
    let _ = writeln!(s, "    logic dbg_cont = 1'b0;");
    let _ = writeln!(s, "    logic dbg_held;");
    let _ = writeln!(s, "    logic [7:0] stim [0:NF*NI-1];");
    let _ = writeln!(s, "    logic signed [15:0] exp_mem [0:NF*TOTAL-1];");
    let _ = writeln!(s, "    logic [7:0] out_stream [0:NO-1];");
    let _ = writeln!(s, "    string stim_path = \"stim_input.hex\";");
    let _ = writeln!(s, "    string exp_path  = \"expected.hex\";");
    let _ = writeln!(s, "    string trace_path = \"sim_outputs.txt\";");
    let _ = writeln!(s, "    logic no_check = 1'b0;");
    let _ = writeln!(s, "    int fd;");
    let _ = writeln!(s, "    int mism = 0;");
    let _ = writeln!(s, "    int checks = 0;");
    let _ = writeln!(s, "    int cyc = 0;");
    let _ = writeln!(s, "    int in_stall = 0;");
    let _ = writeln!(s, "    int out_stall = 0;");
    let _ = writeln!(s, "    int hold_cycles = 0;");
    let _ = writeln!(s, "    int frame = 0;");
    let _ = writeln!(s, "    int i = 0;");
    let _ = writeln!(s, "    int ii = 0;");
    let _ = writeln!(s, "    int bad_n [0:0];");
    let _ = writeln!(s, "    int c0 = 0;");
    let _ = writeln!(s, "    int h0 = 0;");
    let _ = writeln!(s, "    int frame_beats = 0;");
    let _ = writeln!(s, "    logic signed [15:0] value;");
    let _ = writeln!(s, "    logic [7:0] got;");
    let _ = writeln!(s);
    let _ = writeln!(s, "    {dut_name} #(.AW(AW)) dut (");
    let _ = writeln!(
        s,
        "        .clk(clk), .rst(rst), .start(start), .done(done),"
    );
    let _ = writeln!(
        s,
        "        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),"
    );
    let _ = writeln!(
        s,
        "        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),"
    );
    let _ = writeln!(
        s,
        "        .dbg_en(dbg_en), .dbg_addr(dbg_addr), .dbg_data(dbg_data),"
    );
    let _ = writeln!(
        s,
        "        .dbg_hold_en(dbg_hold_en), .dbg_cont(dbg_cont), .dbg_held(dbg_held));"
    );
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        forever #5 clk = ~clk;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(s, "    always @(posedge clk) begin");
    let _ = writeln!(s, "        cyc <= cyc + 1;");
    let _ = writeln!(s, "        if (cyc > TIMEOUT) begin");
    let _ = writeln!(
        s,
        "            $display(\"TESSA_RTL_TIMEOUT cycles=%0d\", cyc);"
    );
    let _ = writeln!(s, "            $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "            $finish;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // One input code per accepted handshake; held high across beats so"
    );
    let _ = writeln!(
        s,
        "    // an unstalled load costs exactly one cycle per code."
    );
    let _ = writeln!(s, "    task automatic send(input logic [7:0] data);");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            in_data = data; in_valid = 1'b1;");
    let _ = writeln!(
        s,
        "            while (!in_ready) begin in_stall = in_stall + 1; @(negedge clk); end"
    );
    let _ = writeln!(s, "            @(posedge clk);");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endtask");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // One output code. With `stall_check` the beat is held low-ready for"
    );
    let _ = writeln!(
        s,
        "    // two cycles and must not change; those cycles are counted separately."
    );
    let _ = writeln!(
        s,
        "    task automatic recv(input logic stall_check, output logic [7:0] o);"
    );
    let _ = writeln!(s, "        logic [7:0] held;");
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            out_ready = 1'b1;");
    let _ = writeln!(s, "            while (!out_valid) @(negedge clk);");
    let _ = writeln!(s, "            held = out_data;");
    let _ = writeln!(s, "            if (stall_check) begin");
    let _ = writeln!(s, "                out_ready = 1'b0;");
    let _ = writeln!(s, "                repeat (2) begin");
    let _ = writeln!(s, "                    @(negedge clk);");
    let _ = writeln!(s, "                    if (out_data !== held) begin");
    let _ = writeln!(
        s,
        "                        $display(\"TESSA_RTL_ERROR unstable_output\");"
    );
    let _ = writeln!(s, "                        mism = mism + 1;");
    let _ = writeln!(s, "                    end");
    let _ = writeln!(s, "                end");
    let _ = writeln!(s, "                out_stall = out_stall + 2;");
    let _ = writeln!(s, "                out_ready = 1'b1;");
    let _ = writeln!(s, "                @(posedge clk);");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "                out_ready = 1'b0;");
    let _ = writeln!(s, "            end else begin");
    let _ = writeln!(s, "                @(posedge clk);");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "            o = held;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endtask");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "    // Debug read through the engine's single read port: the address is"
    );
    let _ = writeln!(
        s,
        "    // presented for one cycle and the registered read is sampled after it."
    );
    let _ = writeln!(
        s,
        "    task automatic read_code(input int addr, output logic signed [15:0] v);"
    );
    let _ = writeln!(s, "        begin");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            dbg_en = 1'b1; dbg_addr = addr[AW-1:0];");
    let _ = writeln!(s, "            @(negedge clk);");
    let _ = writeln!(s, "            v = dbg_data;");
    let _ = writeln!(s, "            dbg_en = 1'b0;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "    endtask");
    let _ = writeln!(s);
    let _ = writeln!(s, "    initial begin");
    let _ = writeln!(s, "        no_check = $test$plusargs(\"NOCHECK\");");
    let _ = writeln!(s, "        void'($value$plusargs(\"STIM=%s\", stim_path));");
    let _ = writeln!(
        s,
        "        void'($value$plusargs(\"EXPECTED=%s\", exp_path));"
    );
    let _ = writeln!(
        s,
        "        void'($value$plusargs(\"TRACE=%s\", trace_path));"
    );
    let _ = writeln!(s, "        $readmemh(stim_path, stim);");
    let _ = writeln!(s, "        if (!no_check) $readmemh(exp_path, exp_mem);");
    let _ = writeln!(s, "        fd = $fopen(trace_path, \"w\");");
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"format tessera.sim_trace.v1\\n\");"
    );
    let _ = writeln!(s, "        repeat (4) @(negedge clk);");
    let _ = writeln!(s, "        rst = 1'b0;");
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "        // ---- frame 0: unstalled schedule measurement ----"
    );
    let _ = writeln!(s, "        frame = 0;");
    let _ = writeln!(s, "        dbg_hold_en = 1'b0;");
    let _ = writeln!(s, "        out_ready = 1'b1;");
    let _ = writeln!(
        s,
        "        @(negedge clk); start = 1'b1; c0 = cyc; in_data = stim[0]; in_valid = 1'b1;"
    );
    let _ = writeln!(s, "        @(negedge clk); start = 1'b0;");
    let _ = writeln!(s, "        @(posedge clk);");
    let _ = writeln!(s, "        for (i = 1; i < NI; i = i + 1) begin");
    let _ = writeln!(s, "            @(negedge clk); in_data = stim[i];");
    let _ = writeln!(s, "            @(posedge clk);");
    let _ = writeln!(s, "        end");
    let _ = writeln!(
        s,
        "        @(negedge clk); in_data = 8'h00; in_valid = 1'b0;"
    );
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"frame 0 input {c} {h} {w}\\n\", frame);",
        c = model.slots[model.input_slot].shape.c,
        h = model.slots[model.input_slot].shape.h,
        w = model.slots[model.input_slot].shape.w
    );
    let _ = writeln!(
        s,
        "        for (ii = 0; ii < NI; ii = ii + 1) $fwrite(fd, \"%0d\\n\", $signed(stim[ii]));"
    );
    let _ = writeln!(s, "        for (i = 0; i < NO; i = i + 1) begin");
    let _ = writeln!(s, "            recv(1'b0, got);");
    let _ = writeln!(s, "            checks = checks + 1;");
    let _ = writeln!(s, "            out_stream[i] = got;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "        while (!done) @(negedge clk);");
    let _ = writeln!(s, "        frame_beats = cyc - c0 - 1;");
    let _ = writeln!(s, "        if (frame_beats != FRAME_BEATS) begin");
    let _ = writeln!(
        s,
        "            $display(\"TESSA_RTL_ERROR beats frame=0 measured=%0d expected=%0d\", frame_beats, FRAME_BEATS);"
    );
    let _ = writeln!(s, "            mism = mism + 1;");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "        for (i = 0; i < NO; i = i + 1) begin",);
    let _ = writeln!(
        s,
        "            if (!no_check && (out_stream[i] !== exp_mem[i + {output_exp_base}][7:0])) begin"
    );
    let _ = writeln!(s, "                mism = mism + 1;");
    let _ = writeln!(
        s,
        "                if (mism < 12) $display(\"TESSA_RTL_MISMATCH frame=0 out[%0d] expected=%0d actual=%0d\", i, $signed(exp_mem[i + {output_exp_base}]), out_stream[i]);"
    );
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    emit_checkpoint_block(
        &mut s,
        8,
        &Checkpoint {
            keyword: "output",
            index: usize::MAX,
            name: model.slots[model.output_slot].name.clone(),
            elem: String::new(),
            shape: model.slots[model.output_slot].shape,
            offset: model.slots[model.output_slot].offset,
            count: no,
            exp_base: output_exp_base,
        },
        "frame * TOTAL",
        false,
    );
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"frame 0 cycles %0d expected %0d\\n\", frame_beats, FRAME_BEATS);"
    );
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"frame 0 stalls %0d %0d holds %0d\\n\", in_stall, out_stall, hold_cycles);"
    );
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "        // ---- frame 1: step-held arena checkpoints, input gaps, backpressure ----"
    );
    let _ = writeln!(s, "        frame = 1;");
    let _ = writeln!(s, "        in_stall = 0; out_stall = 0; hold_cycles = 0;");
    let _ = writeln!(s, "        dbg_hold_en = 1'b1;");
    let _ = writeln!(s, "        out_ready = 1'b0;");
    let _ = writeln!(s, "        @(negedge clk); start = 1'b1; c0 = cyc;");
    let _ = writeln!(s, "        @(negedge clk); start = 1'b0;");
    let _ = writeln!(s, "        for (i = 0; i < NI; i = i + 1) begin");
    let _ = writeln!(s, "            send(stim[NI + i]);");
    let _ = writeln!(s, "            if ((i % 7) == 3 && (i + 1 < NI)) begin");
    let _ = writeln!(s, "                @(negedge clk);");
    let _ = writeln!(s, "                in_data = 8'h00; in_valid = 1'b0;");
    let _ = writeln!(s, "                repeat (3) @(negedge clk);");
    let _ = writeln!(s, "                in_stall = in_stall + 3;");
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    let _ = writeln!(
        s,
        "        @(negedge clk); in_data = 8'h00; in_valid = 1'b0;"
    );
    for (index, checkpoint) in checkpoints.iter().enumerate() {
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "        // checkpoint {index}: {} ({} words at {})",
            checkpoint.name, checkpoint.count, checkpoint.offset
        );
        let _ = writeln!(s, "        h0 = cyc;");
        let _ = writeln!(s, "        while (!dbg_held) @(negedge clk);");
        let _ = writeln!(s, "        hold_cycles = hold_cycles + (cyc - h0);");
        let _ = writeln!(s, "        repeat (2) @(negedge clk);");
        emit_checkpoint_block(&mut s, 8, checkpoint, "frame * TOTAL", true);
        let _ = writeln!(
            s,
            "        dbg_cont = 1'b1; @(negedge clk); dbg_cont = 1'b0;"
        );
        let _ = writeln!(
            s,
            "        $display(\"TESSA_RTL_CHECKPOINT frame=%0d checkpoint={index} cycles=%0d\", frame, cyc);"
        );
        let _ = writeln!(s, "        $fflush(fd);");
    }
    let _ = writeln!(s);
    let _ = writeln!(s, "        out_ready = 1'b0;");
    let _ = writeln!(
        s,
        "        $display(\"TESSA_RTL_PHASE frame=%0d drain-checkpoints=%0d\", frame, NCP);"
    );
    let _ = writeln!(s, "        $fflush(fd);");
    let _ = writeln!(s, "        for (i = 0; i < NO; i = i + 1) begin");
    let _ = writeln!(s, "            recv(1'b1, got);");
    let _ = writeln!(s, "            checks = checks + 1;");
    let _ = writeln!(s, "            out_stream[i] = got;");
    let _ = writeln!(
        s,
        "            if (!no_check && (got !== exp_mem[TOTAL + i + {output_exp_base}][7:0])) begin"
    );
    let _ = writeln!(s, "                mism = mism + 1;");
    let _ = writeln!(
        s,
        "                if (mism < 12) $display(\"TESSA_RTL_MISMATCH frame=1 out[%0d] expected=%0d actual=%0d\", i, $signed(exp_mem[TOTAL + i + {output_exp_base}]), got);"
    );
    let _ = writeln!(s, "            end");
    let _ = writeln!(s, "        end");
    let _ = writeln!(s, "        while (!done) @(negedge clk);");
    let _ = writeln!(s, "        frame_beats = cyc - c0 - 1;");
    let _ = writeln!(s, "        if (frame_beats < FRAME_BEATS) begin");
    let _ = writeln!(
        s,
        "            $display(\"TESSA_RTL_ERROR beats frame=1 measured=%0d below schedule=%0d\", frame_beats, FRAME_BEATS);"
    );
    let _ = writeln!(s, "            mism = mism + 1;");
    let _ = writeln!(s, "        end");
    emit_checkpoint_block(
        &mut s,
        8,
        &Checkpoint {
            keyword: "output",
            index: usize::MAX,
            name: model.slots[model.output_slot].name.clone(),
            elem: String::new(),
            shape: model.slots[model.output_slot].shape,
            offset: model.slots[model.output_slot].offset,
            count: no,
            exp_base: output_exp_base,
        },
        "TOTAL",
        false,
    );
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"frame 1 cycles %0d expected %0d\\n\", frame_beats, FRAME_BEATS);"
    );
    let _ = writeln!(
        s,
        "        $fwrite(fd, \"frame 1 stalls %0d %0d holds %0d\\n\", in_stall, out_stall, hold_cycles);"
    );
    let _ = writeln!(s, "        dbg_hold_en = 1'b0;");
    let _ = writeln!(s, "        $fclose(fd);");
    let _ = writeln!(
        s,
        "        $display(\"TESSA_RTL_RESULT module={tb_name} frames=%0d checks=%0d mismatches=%0d cycles=%0d\", NF, checks, mism, cyc);"
    );
    let _ = writeln!(
        s,
        "        if (no_check) $display(\"TESSA_RTL_TRACE_ONLY\");"
    );
    let _ = writeln!(
        s,
        "        else if (mism == 0) $display(\"TESSA_RTL_PASS\");"
    );
    let _ = writeln!(s, "        else $display(\"TESSA_RTL_FAIL\");");
    let _ = writeln!(s, "        $finish;");
    let _ = writeln!(s, "    end");
    let _ = writeln!(s, "endmodule");
    let _ = writeln!(s, "`default_nettype wire");
    s
}

/// One checkpoint's trace block plus its comparison against the reference.
fn emit_checkpoint_block(
    out: &mut String,
    indent: usize,
    checkpoint: &Checkpoint,
    array_base: &str,
    _step: bool,
) {
    let pad = " ".repeat(indent);
    let shape = checkpoint.shape;
    if checkpoint.keyword == "input" {
        let _ = writeln!(
            out,
            "{pad}$fwrite(fd, \"frame %0d input {} {} {}\\n\", frame);",
            shape.c, shape.h, shape.w
        );
    } else if checkpoint.index == usize::MAX {
        let _ = writeln!(
            out,
            "{pad}$fwrite(fd, \"frame %0d output {} {} {}\\n\", frame);",
            shape.c, shape.h, shape.w
        );
    } else {
        let _ = writeln!(
            out,
            "{pad}$fwrite(fd, \"frame %0d step {} {} {} {} {} {}\\n\", frame);",
            checkpoint.index, checkpoint.name, checkpoint.elem, shape.c, shape.h, shape.w
        );
    }
    let _ = writeln!(out, "{pad}bad_n[0] = 0;");
    let _ = writeln!(
        out,
        "{pad}for (ii = 0; ii < {count}; ii = ii + 1) begin",
        count = checkpoint.count
    );
    let _ = writeln!(
        out,
        "{pad}    read_code({offset} + ii, value);",
        offset = checkpoint.offset
    );
    let _ = writeln!(out, "{pad}    $fwrite(fd, \"%0d\\n\", $signed(value));");
    let _ = writeln!(out, "{pad}    checks = checks + 1;");
    let _ = writeln!(
        out,
        "{pad}    if (!no_check && value !== exp_mem[{array_base} + {base} + ii]) begin",
        base = checkpoint.exp_base
    );
    let _ = writeln!(out, "{pad}        mism = mism + 1;");
    let _ = writeln!(out, "{pad}        bad_n[0] = bad_n[0] + 1;");
    let _ = writeln!(
        out,
        "{pad}        if (mism < 12) $display(\"TESSA_RTL_STEP_MISMATCH frame=%0d checkpoint={index} i=%0d expected=%0d actual=%0d\", frame, ii, $signed(exp_mem[{array_base} + {base} + ii]), $signed(value));",
        base = checkpoint.exp_base,
        index = checkpoint.index
    );
    let _ = writeln!(out, "{pad}    end");
    let _ = writeln!(out, "{pad}end");
    let _ = writeln!(
        out,
        "{pad}if (bad_n[0] > 0) $display(\"TESSA_RTL_CHECKPOINT_MISMATCH frame=%0d checkpoint={index} mismatches=%0d of {count}\", frame, bad_n[0]);",
        index = checkpoint.index,
        count = checkpoint.count
    );
}

fn emit_run_script(dut_name: &str, tb_name: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "#!/bin/sh");
    let _ = writeln!(s, "# Generated by tessera {FULL_GRAPH_KERNEL_ID}.");
    let _ = writeln!(s, "# Requires Verilator (https://verilator.org).");
    let _ = writeln!(
        s,
        "# Plusargs: +STIM=<path> +EXPECTED=<path> +TRACE=<path> +NOCHECK"
    );
    let _ = writeln!(s, "set -eu");
    let _ = writeln!(s, "cd \"$(dirname \"$0\")\"");
    let _ = writeln!(
        s,
        "verilator --binary --timing -Wno-fatal -Wno-DECLFILENAME \\"
    );
    let _ = writeln!(
        s,
        "    --top-module {tb_name} -o {tb_name}_sim --Mdir obj_dir \\"
    );
    let _ = writeln!(s, "    {dut_name}.sv {tb_name}.sv");
    let _ = writeln!(s, "./obj_dir/{tb_name}_sim \"$@\"");
    s
}

impl super::SystemVerilogBackend {
    /// Lower a multi-step plan together with the graph it was lowered from.
    ///
    /// Drop-in companion to
    /// [`SystemVerilogBackend::lower_execution_plan`](super::SystemVerilogBackend::lower_execution_plan):
    /// same shape of call, plus the graph artifact the plan was projected from
    /// (which is where `bias_int32` and the border semantics are read).
    pub fn lower_full_graph(
        &self,
        plan: &ExecutionPlan,
        graph: &Graph,
        weights: &dyn WeightResolver,
        config: &RtlConfig,
    ) -> Result<RtlBundle, RtlError> {
        lower_full_graph_from_graph(plan, graph, weights, config)
    }
}

/// Bind a plan and a graph, then emit the bundle in one call.
pub fn lower_full_graph_from_graph(
    plan: &ExecutionPlan,
    graph: &Graph,
    weights: &dyn WeightResolver,
    config: &RtlConfig,
) -> Result<RtlBundle, RtlError> {
    let model = FullGraphModel::resolve(graph, plan, weights)?;
    lower_full_graph(plan, &model, config)
}
