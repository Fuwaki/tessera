//! Lowering a checked mapping into a schema-2 `LayoutArtifact`.
//!
//! The schedule, buffers, costs, requant and selection fields are filled from
//! the verified assignment, and the [`HardwareMapping`] records the engines,
//! bindings, pipeline and residences the backend instantiates. Nothing here
//! re-solves anything: the schedule order is the graph's operator order, the
//! per-step cost is `temporal_iterations + 1` cycles at 100 % utilisation, and
//! the frame length is the checker's critical path. `selection.engine` is only
//! the widest-instance summary; the real structure lives in `mapping`.

use crate::layout::{
    BufferBinding, BufferKind, BufferRegion, BufferResidence, CycleBreakdown, EngineConfig,
    EngineInstance, EngineRole, HardwareMapping, LAYOUT_FORMAT, LAYOUT_SCHEMA_VERSION_V2,
    MappingSegment,
    MemoryPlacement, OpAssignment, OpEngineBinding, OpKind, PaddingSpec, PipelineLink, PlanCertainty,
    PlanLimits, PlanStatus, Provenance, RejectionSummary, ResourceEstimate, ResourceSlack,
    ScheduleStep, SearchStats, Selection, StepCost, TensorRef,
};
use crate::planner::{OpDetail, Padding, PlanInput, Policy};
use crate::solver::check::Checked;
use crate::solver::{Context, SolverError};
use crate::target::{BufferRequest, Ti60Target};

/// Everything the lowering needs.
pub(crate) struct LowerInput<'a> {
    /// The lowered primitive graph.
    pub input: &'a PlanInput,
    /// The target being programmed.
    pub target: &'a Ti60Target,
    /// The policy the solve was performed against.
    pub policy: &'a Policy,
    /// Reproducibility metadata.
    pub provenance: Provenance,
    /// Target limits.
    pub limits: PlanLimits,
    /// The encoding context.
    pub context: &'a Context,
    /// The verified assignment.
    pub checked: &'a Checked,
}

/// Build the schema-2 artifact for a checked mapping.
pub(crate) fn lower(lower: &LowerInput) -> Result<crate::layout::LayoutArtifact, SolverError> {
    let context = lower.context;
    let checked = lower.checked;
    let ops = &context.ops;
    let engine_names: Vec<String> = (0..checked.engines.len())
        .map(|index| format!("mac{index}"))
        .collect();
    let ordinal_of_slot = |slot: usize| -> Result<usize, SolverError> {
        checked
            .slots
            .iter()
            .position(|candidate| *candidate == slot)
            .ok_or_else(|| SolverError::Check {
                detail: format!("binding references uninstantiated engine slot {slot}"),
            })
    };
    let engine_name = |op: usize| -> Result<String, SolverError> {
        let binding = checked
            .bindings
            .iter()
            .find(|binding| binding.op == op)
            .ok_or_else(|| SolverError::Check {
                detail: format!("operator {op} has no binding"),
            })?;
        Ok(engine_names[ordinal_of_slot(binding.engine)?].clone())
    };

    // Tensor -> buffer name: a pipeline cut owns a stream buffer, a graph input
    // is staged from its frame buffer, everything else is the producer's buffer.
    let mut link_name: std::collections::BTreeMap<crate::layout::TensorId, String> =
        std::collections::BTreeMap::new();
    for link in &checked.links {
        let edge = &context.problem.edges[link.edge];
        let producer = &ops[edge.producer].name;
        let consumer = &ops[edge.consumer].name;
        link_name.insert(edge.tensor, format!("stream.{producer}->{consumer}"));
    }
    let producer_op = |tensor: crate::layout::TensorId| -> Option<String> {
        lower
            .input
            .ops
            .iter()
            .find(|op| op.outputs.contains(&tensor))
            .map(|op| op.name.clone())
    };
    let buffer_name = |tensor: crate::layout::TensorId| -> String {
        if let Some(name) = link_name.get(&tensor) {
            return name.clone();
        }
        if context.graph_inputs.contains(&tensor) {
            let name = lower
                .input
                .tensor(tensor)
                .map(|t| t.name.clone())
                .unwrap_or_else(|| tensor.to_string());
            return format!("stage.in.{name}");
        }
        match producer_op(tensor) {
            Some(producer) => format!("onchip.{producer}"),
            None => format!("stage.tensor.{tensor}"),
        }
    };

    // ---- buffers ----------------------------------------------------------
    let mut buffers: Vec<BufferBinding> = Vec::new();
    for residence in &checked.residences {
        let name = buffer_name(residence.tensor);
        if buffers.iter().any(|buffer| buffer.name == name) {
            continue;
        }
        let width_bits = lower
            .input
            .tensor(residence.tensor)
            .map(|t| u64::from(t.elem.storage_bits()))
            .unwrap_or(8);
        let depth = residence.words.max(1);
        let bytes_per_word = width_bits.div_ceil(8);
        let (placement, region, base_offset_bytes) = match residence.kind {
            BufferKind::Fifo => (MemoryPlacement::Fifo, BufferRegion::Stream, 0),
            _ => (
                MemoryPlacement::Bram,
                BufferRegion::ActivationRam,
                residence.offset_words.saturating_mul(bytes_per_word),
            ),
        };
        buffers.push(BufferBinding {
            name,
            placement,
            region,
            base_offset_bytes,
            tensor: Some(residence.tensor),
            width_bits,
            depth,
            ram10_blocks: 0,
            bytes: bytes_per_word.saturating_mul(depth),
            read_ports: 1,
            write_ports: 1,
            ping_pong_depth: 1,
            producers: Vec::new(),
            consumers: Vec::new(),
        });
    }

    // ---- schedule ---------------------------------------------------------
    let mut schedule = Vec::with_capacity(ops.len());
    let mut assignments = Vec::with_capacity(ops.len());
    for (index, op) in ops.iter().enumerate() {
        let binding = checked
            .bindings
            .iter()
            .find(|binding| binding.op == index)
            .ok_or_else(|| SolverError::Check {
                detail: format!("operator {} has no binding", op.name),
            })?;
        let shape = binding.candidate.shape;
        let lanes = binding.candidate.lanes;
        let temporal = binding.candidate.temporal_iterations;
        let input_buffer = buffer_name(op.inputs[0]);
        let output_buffer = buffer_name(op.outputs[0]);
        let engine = engine_names[ordinal_of_slot(binding.engine)?].clone();
        assignments.push(OpAssignment {
            op: op.name.clone(),
            op_kind: op.kind,
            simd: shape.simd,
            pe: shape.pe,
            tap_parallel: shape.tap_parallel,
            input_buffer: input_buffer.clone(),
            output_buffer: output_buffer.clone(),
            fused_with: Vec::new(),
        });
        schedule.push(ScheduleStep {
            step: index as u32,
            op: op.name.clone(),
            op_kind: op.kind,
            engine,
            mode: crate::layout::Granularity::Frame,
            tile: None,
            simd: shape.simd,
            pe: shape.pe,
            tap_parallel: shape.tap_parallel,
            in_channels: op.in_channels,
            out_channels: op.out_channels,
            in_spatial: op.in_spatial,
            out_spatial: op.out_spatial,
            kernel: op.kernel,
            groups: op.groups,
            stride: op.stride,
            padding: padding_spec_for(op, lower.input),
            upsample_scale: if op.kind == OpKind::NearestUpsample2x {
                Some(2)
            } else {
                None
            },
            in_code_range: op.in_code_range,
            out_code_range: op.out_code_range,
            requant: op.requant.clone(),
            activation: op.activation,
            weights: op.weights.clone(),
            bias: op.bias.clone(),
            inputs: op.inputs.clone(),
            outputs: op.outputs.clone(),
            input_buffer,
            output_buffer,
            estimated_cycles: temporal.saturating_add(1),
            cost: StepCost {
                lanes,
                lane_iterations: temporal,
                macs: op.macs,
                setup_cycles: 1,
                utilisation_percent: 100,
            },
            fused_with: Vec::new(),
        });
    }

    // ---- mapping ----------------------------------------------------------
    let engines: Vec<EngineInstance> = checked
        .engines
        .iter()
        .enumerate()
        .map(|(ordinal, engine)| EngineInstance {
            name: engine_names[ordinal].clone(),
            role: if engine.dedicated {
                EngineRole::Dedicated
            } else {
                EngineRole::Shared
            },
            simd: engine.shape.simd,
            pe: engine.shape.pe,
            tap_parallel: engine.shape.tap_parallel,
            activation_bits: context.activation_bits,
            accumulator_bits: context.accumulator_bits,
            weight_bits: context.weight_bits,
        })
        .collect();
    let mut bindings: Vec<OpEngineBinding> = Vec::with_capacity(checked.bindings.len());
    for binding in &checked.bindings {
        bindings.push(OpEngineBinding {
            op: ops[binding.op].name.clone(),
            engine: engine_names[ordinal_of_slot(binding.engine)?].clone(),
            candidate: binding.candidate.shape.name(),
            simd: binding.candidate.shape.simd,
            pe: binding.candidate.shape.pe,
            tap_parallel: binding.candidate.shape.tap_parallel,
            temporal_iterations: binding.candidate.temporal_iterations,
            macs: ops[binding.op].macs,
        });
    }
    let pipeline: Vec<PipelineLink> = checked
        .links
        .iter()
        .map(|link| {
            let edge = &context.problem.edges[link.edge];
            Ok(PipelineLink {
                producer_engine: engine_name(edge.producer)?,
                consumer_engine: engine_name(edge.consumer)?,
                tensor: edge.tensor,
                depth_elements: 1,
            })
        })
        .collect::<Result<_, SolverError>>()?;
    let residences: Vec<BufferResidence> = checked
        .residences
        .iter()
        .map(|residence| BufferResidence {
            tensor: residence.tensor,
            kind: residence.kind,
            words: residence.words,
            offset_words: residence.offset_words,
            producer_op: residence.producer_op.clone(),
            consumer_ops: residence.consumer_ops.clone(),
        })
        .collect();
    let segments: Vec<MappingSegment> = checked
        .segments
        .iter()
        .map(|seg| {
            let mode = if seg.shared { "shared" } else { "dedicated" };
            let eng_names: Vec<String> = seg
                .engines
                .iter()
                .map(|slot| {
                    let ordinal = ordinal_of_slot(*slot).expect("segment engine slot");
                    engine_names[ordinal].clone()
                })
                .collect();
            MappingSegment {
                index: seg.id,
                mode: mode.into(),
                ops: seg.ops.iter().map(|i| ops[*i].name.clone()).collect(),
                engines: eng_names,
                cycles: seg.cycles,
            }
        })
        .collect();
    let mapping = HardwareMapping {
        engines: engines.clone(),
        bindings,
        pipeline,
        buffers: residences,
        segments,
        makespan_cycles: checked.makespan_cycles,
    };

    // ---- selection summary and resources ----------------------------------
    let total_lanes: u64 = engines.iter().map(|engine| engine.lanes()).sum();
    let widest = engines
        .iter()
        .max_by_key(|engine| engine.lanes())
        .ok_or_else(|| SolverError::Check {
            detail: "the mapping instantiates no engine".into(),
        })?;
    let selection = Selection {
        execution_mode: if engines
            .iter()
            .all(|engine| engine.role == EngineRole::Shared)
        {
            crate::layout::ExecutionMode::SharedTile
        } else {
            crate::layout::ExecutionMode::Hybrid
        },
        granularity: crate::layout::Granularity::Frame,
        tile: None,
        engine: EngineConfig {
            name: widest.name.clone(),
            simd: widest.simd,
            pe: widest.pe,
            tap_parallel: widest.tap_parallel,
            activation_bits: context.activation_bits,
            accumulator_bits: context.accumulator_bits,
            weight_bits: context.weight_bits,
            arena_words: checked.arena_words,
        },
        weight_placement: MemoryPlacement::Bram,
        intermediate_placement: MemoryPlacement::Bram,
        fusion_enabled: false,
        ops: assignments,
    };

    let ram10 = estimate_ram10(
        lower.target,
        ops,
        checked.arena_words,
        context.activation_bits,
    );
    let resources = ResourceEstimate {
        dsp48: checked.dsp48,
        ram10,
        xlr: lower
            .target
            .estimate_xlr(total_lanes, engines.len() as u64, ram10)
            .get(),
        frame_cycles: checked.makespan_cycles,
        ddr_read_bytes_per_frame: 0,
        ddr_write_bytes_per_frame: 0,
    };
    let cycles = CycleBreakdown::new(100, checked.makespan_cycles, 0, 0);
    let slack = ResourceSlack::between(&resources, &lower.limits);
    let tensors: Vec<TensorRef> = lower
        .input
        .tensors
        .iter()
        .map(|t| TensorRef {
            id: t.id,
            name: t.name.clone(),
            shape: t.shape,
            elem: t.elem,
        })
        .collect();
    let rejections = RejectionSummary {
        rejected_candidates: 0,
        counts: std::collections::BTreeMap::new(),
        samples: Vec::new(),
        by_mode: std::collections::BTreeMap::new(),
    };
    let search = SearchStats {
        explored: 1,
        expanded_nodes: 0,
        max_candidates: lower.policy.search.max_candidates,
        per_candidate_budget: lower.policy.search.max_candidates,
        truncated_candidates: 0,
        truncated: false,
    };

    let mut artifact = crate::layout::LayoutArtifact {
        format: LAYOUT_FORMAT.into(),
        schema_version: LAYOUT_SCHEMA_VERSION_V2,
        status: PlanStatus::Feasible,
        certainty: PlanCertainty::from_search(PlanStatus::Feasible, false),
        provenance: lower.provenance.clone(),
        limits: lower.limits,
        objective: lower.policy.objective.clone(),
        selection: Some(selection),
        resources: Some(resources),
        cycles: Some(cycles),
        slack: Some(slack),
        tensors,
        schedule,
        buffers,
        fusions: Vec::new(),
        rejections,
        search,
        mapping: Some(mapping),
    };
    artifact.normalize();
    Ok(artifact)
}

/// RAM10 blocks for the mapped organization, from the real target mapping.
///
/// Weights live in one pooled read-only ROM and the activations in one arena;
/// both are mapped with [`Ti60Target::map_buffer`], so the figure is the
/// device's own organization rather than a second guess.
fn estimate_ram10(
    target: &Ti60Target,
    ops: &[crate::solver::PlanOp],
    arena_words: u64,
    activation_bits: u32,
) -> u64 {
    let primitive = target
        .default_ram_primitive()
        .map(|primitive| primitive.name.clone())
        .unwrap_or_else(|_| "ram10".to_string());
    let mut blocks = 0u64;
    for op in ops {
        if let Some(weights) = &op.weights {
            let depth = weights.shape.elements().max(1);
            if let Ok(mapping) = target.map_buffer(
                &primitive,
                BufferRequest {
                    width_bits: u64::from(weights.elem.storage_bits()).max(1),
                    depth,
                    read_ports: 1,
                    write_ports: 0,
                },
            ) {
                blocks = blocks.saturating_add(mapping.blocks.get());
            }
        }
    }
    if let Ok(mapping) = target.map_buffer(
        &primitive,
        BufferRequest {
            width_bits: u64::from(activation_bits).max(1),
            depth: arena_words.max(1),
            read_ports: 1,
            write_ports: 1,
        },
    ) {
        blocks = blocks.saturating_add(mapping.blocks.get());
    }
    blocks
}

/// Project a convolution's planner padding onto the layout [`PaddingSpec`].
fn padding_spec_for(op: &crate::solver::PlanOp, input: &PlanInput) -> Option<PaddingSpec> {
    if !op.kind.is_convolution() {
        return None;
    }
    let detail = input
        .ops
        .iter()
        .find(|candidate| candidate.name == op.name)
        .and_then(|prim| match &prim.detail {
            OpDetail::Conv(detail) => Some(detail),
            _ => None,
        });
    let Some(detail) = detail else {
        return Some(PaddingSpec::NONE);
    };
    Some(match &detail.padding {
        Padding::Zero {
            top,
            bottom,
            left,
            right,
        } => PaddingSpec {
            top: *top,
            bottom: *bottom,
            left: *left,
            right: *right,
            reflect: false,
        },
        Padding::Reflect | Padding::Same => {
            let reflect = matches!(detail.padding, Padding::Reflect);
            let pad_h = ((op.out_spatial[0].saturating_sub(1))
                .saturating_mul(op.stride[0])
                .saturating_add(op.kernel[0]))
            .saturating_sub(op.in_spatial[0]);
            let pad_w = ((op.out_spatial[1].saturating_sub(1))
                .saturating_mul(op.stride[1])
                .saturating_add(op.kernel[1]))
            .saturating_sub(op.in_spatial[1]);
            PaddingSpec {
                top: pad_h / 2,
                bottom: pad_h - pad_h / 2,
                left: pad_w / 2,
                right: pad_w - pad_w / 2,
                reflect,
            }
        }
    })
}
