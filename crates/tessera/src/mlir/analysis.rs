//! Structured analysis of a module, read with Melior.
//!
//! The optimizer's output is not a report: it is the representation the
//! primitive lowering and the layout search are driven from. This module turns
//! the optimized text into structured, inspectable data —
//!
//! * every operation, with its dialect and occurrence count;
//! * every statically allocated buffer, with its shape, element type and size in
//!   bytes (the bufferization result);
//! * every affine loop nest, with the operations it contains and the Tessera
//!   operators those operations belong to (`*_loop_*` metadata);
//! * the fusion grouping: which operators' work one loop nest performs, derived
//!   from loop containment, not from arithmetic pattern matching.
//!
//! Nothing here guesses semantics from generated arithmetic. Operator identity
//! travels with the IR: [`crate::mlir::emit`] writes
//! `loc("tessera:<role>#<operator index>")` onto every structural operation, and
//! MLIR preserves that location through bufferization, loop lowering and loop
//! fusion. An operation whose tag the analyzer does not recognize is counted as
//! unattributed and reported, never interpreted.
//!
//! Without the `melior` feature there is no analyzer: [`analyze`] reports
//! [`MlirError::MeliorUnavailable`] rather than parsing text with ad-hoc rules.

use super::artifact::{MlirError, PipelineStage, StructuralMetrics};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Value of [`ModuleAnalysis::format`].
pub const ANALYSIS_FORMAT: &str = "tessera.mlir.analysis";

/// Value of [`ModuleAnalysis::schema_version`].
///
/// Version 2 adds [`AnalysisCounts::dynamic_buffers`]: a dynamically shaped
/// allocation is recorded with an empty shape and an unknown byte size, and the
/// count of them tells a reader that the byte total it sees is partial rather
/// than measured. Version 1 analyses (which had no such field and recorded a
/// truncated shape for dynamic allocations) are rejected instead of being
/// interpreted with the new semantics.
pub const ANALYSIS_SCHEMA_VERSION: u32 = 2;

/// Location namespace the emitter writes tags into.
pub const TAG_PREFIX: &str = "tessera:";

/// Separator between a tag's role and its operator index.
pub const TAG_INDEX_SEPARATOR: char = '#';

/// One Tessera operation tag, as written into an MLIR location.
///
/// The role says which part of an operator's implementation the operation is
/// (`conv`, `pad`, `epilogue`, `add`, `upsample`), and the index identifies the
/// operator in [`crate::mlir::MlirArtifact::operators`]. Both come from the
/// emitter; the analyzer only reads them back.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OpTag {
    /// Role within the operator's implementation.
    pub role: String,
    /// Operator index in the artifact's execution order.
    pub index: usize,
}

impl OpTag {
    /// Parse `role#index`.
    pub fn parse(text: &str) -> Option<Self> {
        let (role, index) = text.split_once(TAG_INDEX_SEPARATOR)?;
        if role.is_empty() {
            return None;
        }
        let index = index.parse::<usize>().ok()?;
        Some(OpTag {
            role: role.to_string(),
            index,
        })
    }

    /// Render as it appears in the module.
    pub fn render(&self) -> String {
        format!("{}{TAG_INDEX_SEPARATOR}{}", self.role, self.index)
    }
}

/// One statically allocated buffer of the optimized module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzedBuffer {
    /// Tessera tag of the allocation, when it has one.
    pub tag: Option<String>,
    /// Static shape; empty when the allocation is dynamically shaped.
    pub shape: Vec<u64>,
    /// Element type spelling, e.g. `i8`.
    pub element: String,
    /// Size in bytes; `None` for a dynamically shaped allocation.
    pub bytes: Option<u64>,
    /// Requested alignment, when the allocation declares one.
    pub alignment: Option<u64>,
    /// How many affine loops enclose the allocation.
    pub loop_depth: u32,
}

/// One affine loop nest: an outermost loop that is not just a wrapper for the
/// loops inside it.
///
/// A loop is a nest root when the set of operators its body performs differs
/// from the set its enclosing context performs. The batch loop of a whole
/// network, whose body only contains the per-row loops, is therefore not
/// reported as a nest of its own; the row nest that actually interleaves two
/// operators is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopNest {
    /// Position of the nest in the module.
    pub ordinal: usize,
    /// Tag of the first tagged operation in the nest, when there is one.
    pub tag: Option<String>,
    /// Distinct Tessera tags in the nest, sorted.
    pub tags: Vec<String>,
    /// Distinct operator indices the nest performs, sorted.
    pub operator_indices: Vec<usize>,
    /// Loop nesting depth of the nest root within its function.
    pub depth: u32,
    /// Whether the nest performs more than one operator's work: the
    /// `affine-loop-fusion` result.
    pub fused: bool,
    /// Operation names inside the nest, with occurrence counts.
    pub operations: BTreeMap<String, usize>,
    /// Bytes of the buffers allocated inside the nest.
    pub buffer_bytes: u64,
}

/// Counters describing the analyzed module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisCounts {
    /// Total operations, including nested ones.
    pub operations: usize,
    /// Operations per dialect, e.g. `arith`.
    pub dialects: BTreeMap<String, usize>,
    /// Total affine loops, including nested ones.
    pub loops: usize,
    /// Allocations, statically and dynamically shaped.
    pub buffers: usize,
    /// Allocations whose shape is not fully static.
    ///
    /// Their size is unknown, so [`AnalysisCounts::buffer_bytes`] excludes them:
    /// a reader must treat that total as partial whenever this is non-zero.
    pub dynamic_buffers: usize,
    /// Total bytes of the statically shaped allocations.
    pub buffer_bytes: u64,
    /// Operations that carry a Tessera tag.
    pub tagged_operations: usize,
    /// Tag strings that are not `role#index`, reported instead of being ignored.
    pub unrecognized_tags: Vec<String>,
}

/// Structured analysis of one module text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleAnalysis {
    pub format: String,
    pub schema_version: u32,
    /// Hash of the module text this analysis describes.
    pub module_hash: String,
    /// Toolchain that produced the analysis.
    pub melior_version: String,
    /// Pass pipeline whose output was analyzed.
    pub pipeline: Vec<PipelineStage>,
    pub counts: AnalysisCounts,
    /// Operation name → occurrences.
    pub operations: BTreeMap<String, usize>,
    /// Tessera tag → occurrences.
    pub tags: BTreeMap<String, usize>,
    /// Allocations, in module order.
    pub buffers: Vec<AnalyzedBuffer>,
    /// Loop nests, in module order.
    pub nests: Vec<LoopNest>,
}

impl ModuleAnalysis {
    /// Structural metrics as the artifact schema records them.
    pub fn structural(&self) -> StructuralMetrics {
        StructuralMetrics {
            linalg_ops: self
                .operations
                .iter()
                .filter(|(name, _)| name.starts_with("linalg."))
                .map(|(_, count)| *count)
                .sum(),
            affine_loops: self.counts.loops,
            memref_allocs: self.counts.buffers,
            allocated_bytes: self.counts.buffer_bytes,
        }
    }

    /// Occurrences of one Tessera tag.
    pub fn tag_count(&self, tag: &str) -> usize {
        self.tags.get(tag).copied().unwrap_or(0)
    }

    /// The nest that performs one operator's work.
    ///
    /// A single operator can be spread over more than one nest (a producer that
    /// could not be fused), so this returns the first nest in module order.
    pub fn nest_of_operator(&self, index: usize) -> Option<&LoopNest> {
        self.nests
            .iter()
            .find(|nest| nest.operator_indices.contains(&index))
    }

    /// Every nest that performs one operator's work.
    pub fn nests_of_operator(&self, index: usize) -> Vec<&LoopNest> {
        self.nests
            .iter()
            .filter(|nest| nest.operator_indices.contains(&index))
            .collect()
    }

    /// Buffers allocated with exactly this shape and element type.
    pub fn buffers_with(&self, shape: &[u64], element: &str) -> Vec<&AnalyzedBuffer> {
        self.buffers
            .iter()
            .filter(|buffer| buffer.shape == shape && buffer.element == element)
            .collect()
    }

    /// Nests that fused more than one operator's work.
    pub fn fused_nests(&self) -> Vec<&LoopNest> {
        self.nests.iter().filter(|nest| nest.fused).collect()
    }

    /// Total bytes of the statically shaped allocations.
    ///
    /// `None` when the module also allocates a dynamically shaped buffer: the
    /// byte total is then unknown, and reporting the known partial sum as if it
    /// were the whole total would be a measured-looking lie.
    pub fn allocated_bytes(&self) -> Option<u64> {
        if self.counts.dynamic_buffers == 0 {
            Some(self.counts.buffer_bytes)
        } else {
            None
        }
    }

    /// Every way in which this analysis differs from `other`, field by field.
    ///
    /// This is how a re-derived analysis is compared with the one an artifact
    /// recorded: an operation count, a Tessera tag, a loop nest, its fusion
    /// decision, its buffers — any difference is a mismatch that must be
    /// reported rather than trusted. An empty result means the two describe the
    /// same module.
    pub fn diff(&self, other: &Self) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.format != other.format {
            diffs.push(format!("format {:?} vs {:?}", self.format, other.format));
        }
        if self.schema_version != other.schema_version {
            diffs.push(format!(
                "schema version {} vs {}",
                self.schema_version, other.schema_version
            ));
        }
        if self.module_hash != other.module_hash {
            diffs.push(format!(
                "module hash {} vs {}",
                self.module_hash, other.module_hash
            ));
        }
        if self.melior_version != other.melior_version {
            diffs.push(format!(
                "melior version {:?} vs {:?}",
                self.melior_version, other.melior_version
            ));
        }
        if self.pipeline != other.pipeline {
            diffs.push(format!(
                "pipeline [{}] vs [{}]",
                render_pipeline_names(&self.pipeline),
                render_pipeline_names(&other.pipeline)
            ));
        }
        if self.counts != other.counts {
            diffs.push(format!("counts {:?} vs {:?}", self.counts, other.counts));
        }
        diffs.extend(count_diff("operation", &self.operations, &other.operations));
        diffs.extend(count_diff("tag", &self.tags, &other.tags));
        if self.buffers.len() != other.buffers.len() {
            diffs.push(format!(
                "{} allocations vs {}",
                self.buffers.len(),
                other.buffers.len()
            ));
        }
        for (index, (mine, theirs)) in self.buffers.iter().zip(&other.buffers).enumerate() {
            if mine != theirs {
                diffs.push(format!("buffer {index}: {mine:?} vs {theirs:?}"));
            }
        }
        if self.nests.len() != other.nests.len() {
            diffs.push(format!(
                "{} loop nests vs {}",
                self.nests.len(),
                other.nests.len()
            ));
        }
        for (index, (mine, theirs)) in self.nests.iter().zip(&other.nests).enumerate() {
            if mine != theirs {
                diffs.push(format!("loop nest {index}: {mine:?} vs {theirs:?}"));
            }
        }
        diffs
    }
}

/// Render a pipeline as `pass,pass,...` for a difference message.
fn render_pipeline_names(pipeline: &[PipelineStage]) -> String {
    pipeline
        .iter()
        .map(|stage| stage.render())
        .collect::<Vec<_>>()
        .join(",")
}

/// Differences between two occurrence maps, key by key.
fn count_diff(
    label: &str,
    mine: &BTreeMap<String, usize>,
    theirs: &BTreeMap<String, usize>,
) -> Vec<String> {
    let mut diffs = Vec::new();
    for (key, value) in mine {
        match theirs.get(key) {
            Some(other) if other == value => {}
            Some(other) => diffs.push(format!("{label} {key:?}: {value} vs {other}")),
            None => diffs.push(format!("{label} {key:?}: {value} vs absent")),
        }
    }
    for (key, value) in theirs {
        if !mine.contains_key(key) {
            diffs.push(format!("{label} {key:?}: absent vs {value}"));
        }
    }
    diffs
}

/// Whether this build can re-derive and verify a module's structured analysis.
///
/// Only a build with the `melior` feature can: without it there is no analyzer
/// that can see loop nesting or buffer types, so analysis verification must fail
/// loudly rather than pass on unverified metadata (`docs/developer-setup.md`
/// describes the toolchain).
pub const fn analysis_verification_available() -> bool {
    cfg!(feature = "melior")
}

/// Analyze a module with Melior.
///
/// With the `melior` feature this parses the text, verifies it, and walks it.
/// Without the feature it refuses, because a text scanner cannot see loop
/// nesting or buffer types and would have to guess.
#[cfg(feature = "melior")]
pub fn analyze(
    module: &str,
    melior_version: &str,
    pipeline: &[PipelineStage],
) -> Result<ModuleAnalysis, MlirError> {
    imp::analyze(module, melior_version, pipeline)
}

/// See the `melior` implementation; this build cannot analyze a module.
#[cfg(not(feature = "melior"))]
pub fn analyze(
    module: &str,
    melior_version: &str,
    pipeline: &[PipelineStage],
) -> Result<ModuleAnalysis, MlirError> {
    let _ = (module, melior_version, pipeline);
    Err(MlirError::MeliorUnavailable {
        detail: "structured module analysis needs the `melior` feature (see \
                 docs/developer-setup.md); no textual fallback is used because loop nesting \
                 and buffer types are not recoverable from text without guessing"
            .to_string(),
    })
}

#[cfg(feature = "melior")]
mod imp {
    use super::*;
    use melior::Context;
    use melior::dialect::DialectRegistry;
    use melior::ir::attribute::IntegerAttribute;
    use melior::ir::operation::{OperationLike, OperationRef};
    use melior::ir::r#type::{DimSize, IntegerType, MemRefType, ShapedTypeLike, Type};
    use melior::ir::{BlockLike, Module, RegionLike, ValueLike};
    use melior::utility::{register_all_dialects, register_all_passes};
    use std::collections::BTreeSet;
    use std::sync::{Mutex, Once};

    static REGISTER: Once = Once::new();
    /// MLIR keeps process-global registration state and is not thread safe, so
    /// analysis serializes against in-process compilation.
    static MLIR_LOCK: Mutex<()> = Mutex::new(());

    /// Parse, verify and walk the module.
    pub(super) fn analyze(
        module_text: &str,
        melior_version: &str,
        pipeline: &[PipelineStage],
    ) -> Result<ModuleAnalysis, MlirError> {
        let _guard = MLIR_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        REGISTER.call_once(register_all_passes);

        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        let context = Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let module = Module::parse(&context, module_text).ok_or_else(|| MlirError::Melior {
            detail: "Melior could not parse the module to analyze".to_string(),
        })?;
        if !module.as_operation().verify() {
            return Err(MlirError::Melior {
                detail: "the module to analyze does not verify".to_string(),
            });
        }

        let mut analyzer = Analyzer::default();
        analyzer.walk_op(module.as_operation(), &BTreeSet::new());

        Ok(ModuleAnalysis {
            format: ANALYSIS_FORMAT.to_string(),
            schema_version: ANALYSIS_SCHEMA_VERSION,
            module_hash: super::super::artifact::content_hash(module_text),
            melior_version: melior_version.to_string(),
            pipeline: pipeline.to_vec(),
            counts: AnalysisCounts {
                operations: analyzer.operations.values().sum(),
                dialects: analyzer.dialects,
                loops: analyzer.loops,
                buffers: analyzer.buffers.len(),
                dynamic_buffers: analyzer
                    .buffers
                    .iter()
                    .filter(|buffer| buffer.bytes.is_none())
                    .count(),
                buffer_bytes: analyzer
                    .buffers
                    .iter()
                    .map(|buffer| buffer.bytes.unwrap_or(0))
                    .sum(),
                tagged_operations: analyzer.tagged_operations,
                unrecognized_tags: analyzer.unrecognized_tags,
            },
            operations: analyzer.operations,
            tags: analyzer.tags,
            buffers: analyzer.buffers,
            nests: analyzer.nests,
        })
    }

    #[derive(Default)]
    struct Analyzer {
        operations: BTreeMap<String, usize>,
        dialects: BTreeMap<String, usize>,
        tags: BTreeMap<String, usize>,
        unrecognized_tags: Vec<String>,
        buffers: Vec<AnalyzedBuffer>,
        loops: usize,
        tagged_operations: usize,
        loop_stack: Vec<LoopFrame>,
        nests: Vec<LoopNest>,
    }

    /// One loop body while it is being walked.
    struct LoopFrame {
        depth: u32,
        /// Tag of the loop operation itself, inherited from the structured op it
        /// was lowered from, when it has one.
        tag: Option<String>,
        operations: BTreeMap<String, usize>,
        buffer_bytes: u64,
    }

    impl Analyzer {
        /// Walk one operation and its nested regions.
        ///
        /// `enclosing_tags` is the set of operators performed by the context this
        /// operation lives in (the enclosing loop body, or the function body when
        /// there is none). A loop whose body performs a different set is a nest
        /// root; a loop that only wraps loops performing the same work is not.
        fn walk_op(&mut self, op: OperationRef<'_, '_>, enclosing_tags: &BTreeSet<String>) {
            let name = op.name().as_string_ref().as_str().unwrap_or("").to_string();
            *self.operations.entry(name.clone()).or_insert(0) += 1;
            if let Some((dialect, _)) = name.split_once('.') {
                *self.dialects.entry(dialect.to_string()).or_insert(0) += 1;
            }
            let tag = self.read_tag(&op);
            if let Some(tag) = &tag {
                self.tagged_operations += 1;
                *self.tags.entry(tag.clone()).or_insert(0) += 1;
            }
            for frame in &mut self.loop_stack {
                *frame.operations.entry(name.clone()).or_insert(0) += 1;
            }

            let mut bytes = 0u64;
            if name == "memref.alloc" || name == "memref.alloca" {
                let buffer = buffer_of(&op, tag.as_deref(), self.loop_stack.len() as u32);
                bytes = buffer.bytes.unwrap_or(0);
                self.buffers.push(buffer);
            }
            if let Some(last) = self.loop_stack.last_mut() {
                last.buffer_bytes = last.buffer_bytes.saturating_add(bytes);
            }

            let is_loop = name == "affine.for" || name == "scf.for";
            // The operators a loop body performs are known before descending into
            // it, which is what makes the nest decision possible on the way in.
            let own_tags = if is_loop {
                self.region_tags(&op)
            } else {
                BTreeSet::new()
            };
            if is_loop {
                self.loops += 1;
                self.loop_stack.push(LoopFrame {
                    depth: self.loop_stack.len() as u32,
                    tag,
                    operations: BTreeMap::new(),
                    buffer_bytes: 0,
                });
            }

            let inner_context = if is_loop {
                own_tags.clone()
            } else {
                enclosing_tags.clone()
            };
            for region in op.regions() {
                let mut block = region.first_block();
                while let Some(current) = block {
                    let mut inner = current.first_operation();
                    while let Some(operation) = inner {
                        self.walk_op(operation, &inner_context);
                        inner = operation.next_in_block();
                    }
                    block = current.next_in_region();
                }
            }

            if is_loop {
                if let Some(frame) = self.loop_stack.pop() {
                    self.finish_nest(frame, own_tags, enclosing_tags);
                }
            }
        }

        /// Promote a closed loop body to a nest when it is not a wrapper.
        fn finish_nest(
            &mut self,
            frame: LoopFrame,
            own_tags: BTreeSet<String>,
            enclosing_tags: &BTreeSet<String>,
        ) {
            if own_tags.is_empty() || own_tags == *enclosing_tags {
                return;
            }
            let indices = operator_indices(&own_tags);
            let ordinal = self.nests.len();
            self.nests.push(LoopNest {
                ordinal,
                tag: frame.tag,
                tags: own_tags.iter().cloned().collect(),
                fused: indices.len() > 1,
                operator_indices: indices,
                depth: frame.depth,
                operations: frame.operations,
                buffer_bytes: frame.buffer_bytes,
            });
        }

        /// The tags of every operation inside an operation's regions.
        fn region_tags(&mut self, op: &OperationRef<'_, '_>) -> BTreeSet<String> {
            let mut tags = BTreeSet::new();
            for region in op.regions() {
                let mut block = region.first_block();
                while let Some(current) = block {
                    let mut inner = current.first_operation();
                    while let Some(operation) = inner {
                        if let Some(tag) = self.read_tag(&operation) {
                            tags.insert(tag);
                        }
                        tags.extend(self.region_tags(&operation));
                        inner = operation.next_in_block();
                    }
                    block = current.next_in_region();
                }
            }
            tags
        }

        /// The Tessera tag an operation carries, read from its location.
        ///
        /// Only the `tessera:` namespace is inspected; a tag that is not
        /// `role#index` is reported in `unrecognized_tags` instead of being
        /// interpreted.
        fn read_tag(&mut self, op: &OperationRef<'_, '_>) -> Option<String> {
            // Taking the location by value keeps the borrow local to the op.
            let rendered = format!("{}", op.location());
            let mut found = None;
            let mut search = rendered.as_str();
            while let Some(start) = search.find(TAG_PREFIX) {
                let rest = &search[start + TAG_PREFIX.len()..];
                let end = rest.find('"').unwrap_or(rest.len());
                let tag = &rest[..end];
                if OpTag::parse(tag).is_none() && !self.unrecognized_tags.iter().any(|t| t == tag) {
                    self.unrecognized_tags.push(tag.to_string());
                }
                if found.is_none() {
                    found = Some(tag.to_string());
                }
                search = &rest[end..];
            }
            found
        }
    }

    /// Describe an allocation from its result type.
    fn buffer_of(op: &OperationRef<'_, '_>, tag: Option<&str>, loop_depth: u32) -> AnalyzedBuffer {
        let alignment = op
            .attribute("alignment")
            .ok()
            .and_then(|attribute| IntegerAttribute::try_from(attribute).ok())
            .map(|attribute| attribute.value().unsigned_abs());
        let memref = op
            .result(0)
            .ok()
            .and_then(|result| MemRefType::try_from(result.r#type()).ok());
        let (shape, element, bytes) = match memref {
            Some(memref) => {
                let mut extents = Vec::with_capacity(memref.rank());
                let mut static_shape = true;
                let mut elements: Option<u64> = Some(1);
                for index in 0..memref.rank() {
                    match memref.dim_size(index) {
                        Ok(DimSize::Static(size)) => {
                            extents.push(size);
                            elements = elements.and_then(|total| total.checked_mul(size));
                        }
                        // A dynamic extent makes the whole shape unknown. It is
                        // *not* recorded as the statically known extents only:
                        // `memref<4x?xi8>` is not a rank-1 `[4]` allocation, and
                        // a truncated shape would be indistinguishable from one.
                        _ => {
                            static_shape = false;
                            elements = None;
                        }
                    }
                }
                let element = memref.element().to_string();
                let bytes = match (elements, element_bytes(&memref.element())) {
                    (Some(elements), Some(bits)) => {
                        Some(elements.saturating_mul(u64::from(bits.div_ceil(8))))
                    }
                    _ => None,
                };
                if static_shape {
                    (extents, element, bytes)
                } else {
                    (Vec::new(), element, None)
                }
            }
            None => (Vec::new(), String::new(), None),
        };
        AnalyzedBuffer {
            tag: tag.map(str::to_string),
            shape,
            element,
            bytes,
            alignment,
            loop_depth,
        }
    }

    /// Operator indices a tag set belongs to, sorted and deduplicated.
    fn operator_indices(tags: &BTreeSet<String>) -> Vec<usize> {
        let mut indices: Vec<usize> = tags
            .iter()
            .filter_map(|tag| OpTag::parse(tag))
            .map(|tag| tag.index)
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Element width in bits for the integer element types the emitter uses.
    fn element_bytes(element: &Type<'_>) -> Option<u32> {
        IntegerType::try_from(*element)
            .ok()
            .map(|integer| integer.width())
    }
}
