//! The persisted primitive graph: Graph IR's planning vocabulary, lowered from
//! the *optimized* MLIR artifact.
//!
//! Why this is a distinct artifact and not just a `PlanInput` dump: the planner
//! consumes a typed primitive graph ([`PlanInput`]) and nothing else, but the
//! decisions that produced that graph — which optimizer work it descends from,
//! which operators the optimizer fused into one loop nest, which buffers the
//! bufferizer materialized, and the exact interval of codes every activation
//! carries — are part of the compiler's evidence. They are persisted here, next
//! to the graph they describe, so `verify` can re-check them and a reader can see
//! why a datapath is nine bits wide instead of one byte.
//!
//! ```text
//! optimized.json (tessera.mlir, planning stage, with analysis)
//!   -> lower              this module, via crate::lower::lower_mlir_artifact
//!   -> primitive.json     the PlanInput the planner and backend consume
//!   -> primitives.json    this artifact: intervals, fusion groups, buffers, provenance
//! ```
//!
//! Nothing here re-runs the graph lowering: the plan is built from the artifact's
//! operator bindings and verified against the structured analysis of the
//! optimized module. A graph-stage artifact is rejected.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use crate::layout::ElemType;
use crate::lower::{LowerError, LowerReport, lower_mlir_artifact};
use crate::mlir::{
    MlirArtifact, MlirError, MlirStage, ModuleAnalysis, OperatorKind, PipelineStage, content_hash,
};
use crate::planner::PlanInput;

/// Value of [`PrimitiveArtifact::format`].
pub const PRIMITIVE_FORMAT: &str = "tessera.primitive";

/// Value of [`PrimitiveArtifact::schema_version`].
///
/// Version 2 changes what invariance means: [`PrimitiveArtifact::intervals`]
/// holds only the activation tensors the lowering bounded (weight and bias
/// tensors are described by their plan entry, not by a fabricated 32-bit
/// interval), and [`PrimitiveArtifact::allocated_bytes`] reports an unknown
/// total as `None` rather than as zero. A version 1 artifact is rejected rather
/// than read under the new semantics.
pub const PRIMITIVE_SCHEMA_VERSION: u32 = 2;

/// Exact interval of codes one plan *activation* tensor carries, next to its
/// transport type.
///
/// Only activation tensors have one: a weight or bias tensor's codes are
/// external blob data with no interval, so it is described by its plan entry
/// (name, shape, element type) and never by a fabricated signed-32-bit range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInterval {
    /// Plan tensor name.
    pub tensor: String,
    /// Planner element type: the value semantic, not the container.
    pub elem: ElemType,
    /// Element type of the MLIR value it was lowered from.
    pub mlir_element: String,
    /// Smallest code the tensor can hold.
    pub lo: i64,
    /// Largest code the tensor can hold.
    pub hi: i64,
    /// Narrowest signed width that holds the interval: what sizes the MAC path.
    pub signed_bits: u32,
}

/// One operator inside a loop nest of the optimized module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupMember {
    /// Operator index in the artifact's execution order.
    pub operator: usize,
    /// Operator name as the planner sees it.
    pub name: String,
    /// Operator kind.
    pub kind: OperatorKind,
    /// Roles of this operator's operations inside the nest, sorted.
    pub roles: Vec<String>,
}

/// What one loop nest of the optimized module does.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionGroup {
    /// Position of the nest in the module.
    pub ordinal: usize,
    /// Loop nesting depth of the nest root.
    pub depth: u32,
    /// Whether more than one operator's work was fused into this nest.
    pub fused: bool,
    /// Operators the nest performs, sorted by operator index.
    pub members: Vec<GroupMember>,
    /// Operation names inside the nest, with occurrence counts.
    pub operations: BTreeMap<String, usize>,
    /// Bytes of the buffers allocated inside the nest.
    pub buffer_bytes: u64,
}

/// One buffer the optimized module allocates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferRecord {
    /// Tessera tag of the allocation, when it has one.
    pub tag: Option<String>,
    /// Static shape, empty when dynamically shaped.
    pub shape: Vec<u64>,
    /// Element type spelling.
    pub element: String,
    /// Size in bytes, when statically shaped.
    pub bytes: Option<u64>,
    /// How many affine loops enclose the allocation.
    pub loop_depth: u32,
}

/// Identity of the optimized artifact this primitive graph descends from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrimitiveProvenance {
    /// Format of the source artifact.
    pub mlir_format: String,
    /// Schema version of the source artifact.
    pub mlir_schema_version: u32,
    /// Stage the source artifact was at.
    pub mlir_stage: MlirStage,
    /// Hash of the source artifact's module text.
    pub mlir_module_hash: String,
    /// Graph-manifest hash the source artifact descends from.
    pub mlir_source_content_hash: String,
    /// Melior identity of the optimizer run, when the artifact records one.
    pub melior_version: Option<String>,
    /// Pass pipeline the module was produced by.
    pub pipeline: Vec<PipelineStage>,
    /// Tessera compiler version that produced this artifact.
    pub compiler_version: String,
}

/// A lowered primitive graph, its evidence and its provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveArtifact {
    pub format: String,
    pub schema_version: u32,
    /// Concrete input shape of the module, in `NHWC` order.
    pub input_shape: Vec<u64>,
    /// The planner's input.
    pub plan: PlanInput,
    /// Code intervals of the plan's activation tensors.
    pub intervals: Vec<TensorInterval>,
    /// Loop nests of the optimized module, as fusion groups.
    pub groups: Vec<FusionGroup>,
    /// Buffers of the optimized module.
    pub buffers: Vec<BufferRecord>,
    /// What the lowering decided, in order.
    pub notes: Vec<String>,
    pub provenance: PrimitiveProvenance,
}

impl PrimitiveArtifact {
    /// JSON rendering with a stable, pretty layout.
    pub fn to_json(&self) -> Result<String, LowerError> {
        serde_json::to_string_pretty(self).map_err(|error| LowerError::Artifact {
            detail: format!("serializing the primitive artifact: {error}"),
        })
    }

    /// Parse an artifact from its JSON rendering, checking format and schema.
    pub fn from_json(text: &str) -> Result<Self, LowerError> {
        let artifact: PrimitiveArtifact =
            serde_json::from_str(text).map_err(|error| LowerError::Artifact {
                detail: format!("parsing the primitive artifact: {error}"),
            })?;
        artifact.validate_header()?;
        Ok(artifact)
    }

    /// Write the JSON artifact, creating parent directories.
    pub fn save(&self, path: &Path) -> Result<(), LowerError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|error| LowerError::Artifact {
                    detail: format!("creating {}: {error}", parent.display()),
                })?;
            }
        }
        std::fs::write(path, self.to_json()? + "\n").map_err(|error| LowerError::Artifact {
            detail: format!("writing {}: {error}", path.display()),
        })
    }

    /// Read a JSON artifact from disk.
    pub fn load(path: &Path) -> Result<Self, LowerError> {
        let text = std::fs::read_to_string(path).map_err(|error| LowerError::Artifact {
            detail: format!("reading {}: {error}", path.display()),
        })?;
        Self::from_json(&text)
    }

    /// The planner input as JSON, for `solve --primitive`.
    pub fn plan_json(&self) -> Result<String, LowerError> {
        serde_json::to_string_pretty(&self.plan).map_err(|error| LowerError::Artifact {
            detail: format!("serializing the primitive graph: {error}"),
        })
    }

    /// Reject artifacts of a format or schema this build does not understand.
    pub fn validate_header(&self) -> Result<(), LowerError> {
        if self.format != PRIMITIVE_FORMAT {
            return Err(LowerError::Artifact {
                detail: format!(
                    "expected artifact format {PRIMITIVE_FORMAT}, found {}",
                    self.format
                ),
            });
        }
        if self.schema_version != PRIMITIVE_SCHEMA_VERSION {
            return Err(LowerError::Artifact {
                detail: format!(
                    "expected schema version {PRIMITIVE_SCHEMA_VERSION}, found {}",
                    self.schema_version
                ),
            });
        }
        self.plan
            .validate()
            .map_err(|detail| LowerError::Invalid { detail })
    }

    /// Re-check the artifact against the optimized module it claims to descend from.
    ///
    /// The recorded hashes are recomputed, not trusted, and then the optimized
    /// module is *re-lowered*: the plan, activation intervals, fusion groups,
    /// buffers and notes must be exactly what [`lower`] produces from it. A
    /// persisted `primitives.json` whose plan or evidence was rewritten while its
    /// provenance was left intact therefore fails, and so does an optimized
    /// module whose text, analysis or recorded hash was edited.
    ///
    /// Re-lowering goes through the structured analysis of the module, so in a
    /// build without the `melior` feature this cannot run and reports an error
    /// instead of accepting the artifact's own claims.
    pub fn check(&self, source: &MlirArtifact) -> Result<(), LowerError> {
        source
            .validate_header()
            .map_err(|error| LowerError::Artifact {
                detail: error.to_string(),
            })?;
        source.check(None).map_err(|error| LowerError::Artifact {
            detail: error.to_string(),
        })?;
        // Recompute the module text hash rather than compare recorded strings.
        let module_hash = content_hash(&source.module);
        if self.provenance.mlir_module_hash != module_hash {
            return Err(LowerError::Artifact {
                detail: format!(
                    "artifact records module hash {}, the optimized module's text hashes to \
                     {module_hash}",
                    self.provenance.mlir_module_hash
                ),
            });
        }
        if self.provenance.mlir_source_content_hash != source.provenance.source_content_hash {
            return Err(LowerError::Artifact {
                detail: format!(
                    "artifact records graph hash {}, the optimized artifact has {}",
                    self.provenance.mlir_source_content_hash, source.provenance.source_content_hash
                ),
            });
        }
        let relowered = lower(source)?;
        if let Some(difference) = self.difference(&relowered) {
            return Err(LowerError::Artifact {
                detail: format!(
                    "the artifact is not what re-lowering the optimized module produces: \
                     {difference}"
                ),
            });
        }
        Ok(())
    }

    /// The first way this artifact differs from a freshly lowered one.
    ///
    /// The plan, its activation intervals, the fusion groups, the buffers and the
    /// lowering notes are the evidence this artifact carries; all of them must
    /// match the deterministic lowering of the source module.
    fn difference(&self, other: &Self) -> Option<String> {
        if self.input_shape != other.input_shape {
            return Some(format!(
                "input shape {:?} is not {:?}",
                self.input_shape, other.input_shape
            ));
        }
        if self.plan != other.plan {
            return Some(describe_plan_difference(&self.plan, &other.plan));
        }
        if self.intervals != other.intervals {
            return Some(format!(
                "activation intervals are not those of the lowering ({} recorded, {} produced){}",
                self.intervals.len(),
                other.intervals.len(),
                first_interval_difference(&self.intervals, &other.intervals)
            ));
        }
        if self.groups != other.groups {
            return Some(format!(
                "fusion groups are not those of the lowering ({} recorded, {} produced)",
                self.groups.len(),
                other.groups.len()
            ));
        }
        if self.buffers != other.buffers {
            return Some(format!(
                "buffers are not those of the lowering ({} recorded, {} produced)",
                self.buffers.len(),
                other.buffers.len()
            ));
        }
        if self.notes != other.notes {
            return Some(format!(
                "lowering notes differ ({} recorded, {} produced)",
                self.notes.len(),
                other.notes.len()
            ));
        }
        None
    }

    /// Operators the optimizer fused into one loop nest.
    pub fn fused_groups(&self) -> Vec<&FusionGroup> {
        self.groups.iter().filter(|group| group.fused).collect()
    }

    /// The interval recorded for one plan tensor.
    pub fn interval(&self, tensor: &str) -> Option<&TensorInterval> {
        self.intervals.iter().find(|entry| entry.tensor == tensor)
    }

    /// Bytes the optimized module allocates.
    ///
    /// `None` when any allocation is dynamically shaped: its size is unknown, and
    /// reporting the remaining static bytes as the total would present a partial
    /// sum as a measured one.
    pub fn allocated_bytes(&self) -> Option<u64> {
        if self.buffers.iter().any(|buffer| buffer.bytes.is_none()) {
            None
        } else {
            Some(
                self.buffers
                    .iter()
                    .map(|buffer| buffer.bytes.unwrap_or(0))
                    .sum(),
            )
        }
    }

    /// How many allocations have an unknown (dynamically shaped) size.
    pub fn dynamic_buffers(&self) -> usize {
        self.buffers
            .iter()
            .filter(|buffer| buffer.bytes.is_none())
            .count()
    }
}

impl fmt::Display for PrimitiveArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = match self.allocated_bytes() {
            Some(bytes) => format!("{bytes} bytes"),
            None => format!(
                "unknown bytes over {} dynamic allocation(s)",
                self.dynamic_buffers()
            ),
        };
        write!(
            f,
            "{}: {} tensors, {} operators, {} loop groups ({} fused), {} buffers ({bytes}) from \
             module {}",
            self.format,
            self.plan.tensors.len(),
            self.plan.ops.len(),
            self.groups.len(),
            self.fused_groups().len(),
            self.buffers.len(),
            self.provenance.mlir_module_hash
        )
    }
}

/// Lower an optimized MLIR artifact into the persisted primitive graph.
///
/// The plan comes from [`lower_mlir_artifact`], which refuses an artifact whose
/// optimized module does not contain the operations, loop nests or padded buffers
/// its bindings promise. The groups, intervals and buffers persisted alongside it
/// are read from that same analysis, so the evidence and the plan cannot drift
/// apart.
pub fn lower(artifact: &MlirArtifact) -> Result<PrimitiveArtifact, LowerError> {
    let source = artifact.clone();
    let analysis = source
        .analyzed()
        .map_err(|error: MlirError| LowerError::Artifact {
            detail: error.to_string(),
        })?
        .clone();
    let (plan, report) = lower_mlir_artifact(&source)?;
    Ok(from_parts(&source, &analysis, plan, report))
}

/// Assemble the artifact from an already-lowered plan.
fn from_parts(
    source: &MlirArtifact,
    analysis: &ModuleAnalysis,
    plan: PlanInput,
    report: LowerReport,
) -> PrimitiveArtifact {
    // Intervals describe the *activation* tensors the lowering bounded. A weight
    // or bias tensor has no activation interval — its codes come from the
    // external blob — and publishing the "unknown" sentinel `[i32::MIN,
    // i32::MAX]` for it would tell a datapath sizer that a 4-bit weight tensor
    // needs 32 bits. Such tensors are described by their plan entry (name, shape
    // and element type) and carry no interval.
    let intervals = plan
        .tensors
        .iter()
        .filter(|tensor| {
            plan.inputs.contains(&tensor.id)
                || plan.ops.iter().any(|op| op.outputs.contains(&tensor.id))
        })
        .filter_map(|tensor| {
            let range = tensor.code_range?;
            let mlir_element = source
                .tensor(&tensor.name)
                .map(|binding| binding.element.mlir().to_string())
                .unwrap_or_else(|| activation_mlir_element(tensor.elem).to_string());
            Some(TensorInterval {
                tensor: tensor.name.clone(),
                elem: tensor.elem,
                mlir_element,
                lo: range.lo,
                hi: range.hi,
                signed_bits: range.signed_bits(),
            })
        })
        .collect();
    let groups = analysis
        .nests
        .iter()
        .map(|nest| FusionGroup {
            ordinal: nest.ordinal,
            depth: nest.depth,
            fused: nest.fused,
            members: nest
                .operator_indices
                .iter()
                .filter_map(|index| {
                    let operator = source.operators.get(*index)?;
                    Some(GroupMember {
                        operator: *index,
                        name: operator_name(operator),
                        kind: operator.kind,
                        roles: nest
                            .tags
                            .iter()
                            .filter(|tag| {
                                crate::mlir::OpTag::parse(tag)
                                    .is_some_and(|parsed| parsed.index == *index)
                            })
                            .filter_map(|tag| {
                                crate::mlir::OpTag::parse(tag).map(|parsed| parsed.role)
                            })
                            .collect(),
                    })
                })
                .collect(),
            operations: nest.operations.clone(),
            buffer_bytes: nest.buffer_bytes,
        })
        .collect();
    let buffers = analysis
        .buffers
        .iter()
        .map(|buffer| BufferRecord {
            tag: buffer.tag.clone(),
            shape: buffer.shape.clone(),
            element: buffer.element.clone(),
            bytes: buffer.bytes,
            loop_depth: buffer.loop_depth,
        })
        .collect();
    PrimitiveArtifact {
        format: PRIMITIVE_FORMAT.to_string(),
        schema_version: PRIMITIVE_SCHEMA_VERSION,
        input_shape: source.input_shape.clone(),
        plan,
        intervals,
        groups,
        buffers,
        notes: report.notes,
        provenance: PrimitiveProvenance {
            mlir_format: source.format.clone(),
            mlir_schema_version: source.schema_version,
            mlir_stage: source.stage,
            mlir_module_hash: source.provenance.module_hash.clone(),
            mlir_source_content_hash: source.provenance.source_content_hash.clone(),
            melior_version: source
                .provenance
                .melior
                .as_ref()
                .map(|melior| melior.melior_version.clone()),
            pipeline: source
                .provenance
                .melior
                .as_ref()
                .map(|melior| melior.pipeline.clone())
                .unwrap_or_default(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        },
    }
}

/// Operator name as the planner sees it.
fn operator_name(operator: &crate::mlir::OperatorBinding) -> String {
    match operator.kind {
        OperatorKind::Conv2d | OperatorKind::DepthwiseConv2d | OperatorKind::PointwiseConv2d => {
            operator
                .layer
                .clone()
                .unwrap_or_else(|| operator.tag.clone())
        }
        OperatorKind::Relu => operator
            .layer
            .clone()
            .unwrap_or_else(|| operator.tag.clone()),
        OperatorKind::Add => match operator.points.first() {
            Some(point) => format!("add.{point}"),
            None => format!("add.{}", operator.index),
        },
        OperatorKind::UpsampleNearest2x => match operator.points.first() {
            Some(point) => format!("upsample_nearest2x.{point}"),
            None => format!("upsample_nearest2x.{}", operator.index),
        },
        OperatorKind::QuantizeU8 => "quantize_u8_output".to_string(),
    }
}

/// MLIR element spelling for an activation the emitter transported in `elem`.
///
/// Only a fallback: an activation's exact element type is read from the module
/// binding it lowered from, and this is used when the plan carries a name the
/// module does not.
fn activation_mlir_element(elem: ElemType) -> &'static str {
    match elem {
        ElemType::I4 => "i4",
        ElemType::I8 | ElemType::U8 => "i8",
        ElemType::I9 | ElemType::I16 => "i16",
        ElemType::I32 | ElemType::F32 => "i32",
    }
}

/// Describe the first difference between two plans, field by field.
fn describe_plan_difference(mine: &PlanInput, theirs: &PlanInput) -> String {
    if mine.name != theirs.name {
        return format!("graph name {:?} is not {:?}", mine.name, theirs.name);
    }
    if mine.tensors.len() != theirs.tensors.len() {
        return format!(
            "plan has {} tensors, the lowering produces {}",
            mine.tensors.len(),
            theirs.tensors.len()
        );
    }
    for (index, (a, b)) in mine.tensors.iter().zip(&theirs.tensors).enumerate() {
        if a != b {
            return format!("tensor {index} ({}) {a:?} is not {b:?}", a.name);
        }
    }
    if mine.ops.len() != theirs.ops.len() {
        return format!(
            "plan has {} operators, the lowering produces {}",
            mine.ops.len(),
            theirs.ops.len()
        );
    }
    for (index, (a, b)) in mine.ops.iter().zip(&theirs.ops).enumerate() {
        if a != b {
            return format!("operator {index} ({}) {a:?} is not {b:?}", a.name);
        }
    }
    if mine.inputs != theirs.inputs {
        return format!("plan inputs {:?} are not {:?}", mine.inputs, theirs.inputs);
    }
    if mine.outputs != theirs.outputs {
        return format!(
            "plan outputs {:?} are not {:?}",
            mine.outputs, theirs.outputs
        );
    }
    "the plan differs".to_string()
}

/// The first interval that two interval lists disagree on, for a message.
fn first_interval_difference(mine: &[TensorInterval], theirs: &[TensorInterval]) -> String {
    for interval in mine {
        match theirs.iter().find(|other| other.tensor == interval.tensor) {
            Some(other) if other == interval => {}
            Some(other) => {
                return format!(
                    "; {}: [{}, {}] ({} signed bits) is not [{}, {}] ({} signed bits)",
                    interval.tensor,
                    interval.lo,
                    interval.hi,
                    interval.signed_bits,
                    other.lo,
                    other.hi,
                    other.signed_bits
                );
            }
            None => {
                return format!(
                    "; {} is not an activation interval of the lowering",
                    interval.tensor
                );
            }
        }
    }
    String::new()
}
