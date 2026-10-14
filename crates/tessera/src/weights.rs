//! Weight resolution for the RTL backend.
//!
//! The planner records weight *references* only; the RTL backend asks a
//! [`crate::rtl::WeightResolver`] for the stored bytes of a tensor. This module
//! provides the two resolvers the project needs:
//!
//! * [`GraphWeightResolver`] — the concrete resolver over a persisted
//!   [`GraphArtifact`]. It slices the manifest weight blob by the layer's
//!   declared byte range, so the bytes handed to RTL are exactly the bytes the
//!   manifest shipped (no re-encoding, no rounding), and it verifies the blob
//!   against the artifact's `weights_sha256` provenance before use.
//! * [`NamedBlobResolver`] — a directory/user-supplied resolver for primitive
//!   fixtures that have no manifest behind them.
//!
//! Both fail closed: an unknown tensor, a shape/element mismatch, or a blob that
//! does not match its recorded digest is a [`RtlError`], never a silent
//! substitution.

use crate::frontend::sha256_hex;
use crate::graph::GraphArtifact;
use crate::layout::{ElemType, TensorRef, TensorShape};
use crate::lower::weight_tensor_name;
use crate::rtl::{RtlError, WeightResolver};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Resolves weight bytes from a persisted graph artifact and its weight blob.
pub struct GraphWeightResolver<'a> {
    artifact: &'a GraphArtifact,
    blob: Vec<u8>,
}

impl<'a> GraphWeightResolver<'a> {
    /// Wrap an artifact and its weight blob, checking the blob against the
    /// artifact's recorded length and SHA-256.
    pub fn from_bytes(artifact: &'a GraphArtifact, blob: Vec<u8>) -> Result<Self, RtlError> {
        let provenance = artifact.provenance();
        if blob.len() != provenance.weight_bytes {
            return Err(RtlError::ShapeMismatch {
                layer: "<weights>".to_string(),
                what: format!("weight blob {} size in bytes", provenance.weight_file),
                expected: provenance.weight_bytes as u64,
                actual: blob.len() as u64,
            });
        }
        let digest = sha256_hex(&blob);
        if digest != provenance.weights_sha256 {
            return Err(RtlError::MissingProvenance {
                field: "weights_sha256",
            });
        }
        Ok(GraphWeightResolver { artifact, blob })
    }

    /// Read the weight blob named by the artifact's provenance from `dir`.
    ///
    /// The file name comes from the artifact, not from the caller, and the
    /// contents are checked against the recorded length and digest.
    pub fn load(artifact: &'a GraphArtifact, dir: impl AsRef<Path>) -> Result<Self, RtlError> {
        let name = &artifact.provenance().weight_file;
        let path = dir.as_ref().join(name);
        let blob = std::fs::read(&path).map_err(|source| RtlError::Io {
            context: format!("reading weight blob {}", path.display()),
            source,
        })?;
        Self::from_bytes(artifact, blob)
    }

    /// The artifact this resolver reads.
    pub fn artifact(&self) -> &'a GraphArtifact {
        self.artifact
    }

    /// Byte range of a layer's weights inside the blob.
    fn layer_range(
        &self,
        layer: &str,
        tensor: &TensorRef,
    ) -> Result<std::ops::Range<usize>, RtlError> {
        let model = self.artifact.graph().layer_named(layer).ok_or_else(|| {
            RtlError::WeightsNotResident {
                layer: layer.to_string(),
                tensor: tensor.name.clone(),
                detail: format!("the graph artifact has no layer named {layer:?}"),
            }
        })?;
        let expected_elem = match model.weight_bits.get() {
            8 => ElemType::I8,
            4 => ElemType::I4,
            _ => {
                return Err(RtlError::UnsupportedElementType {
                    layer: layer.to_string(),
                    tensor: tensor.name.clone(),
                    element_type: tensor.elem.into(),
                });
            }
        };
        if tensor.elem != expected_elem {
            return Err(RtlError::UnsupportedElementType {
                layer: layer.to_string(),
                tensor: tensor.name.clone(),
                element_type: tensor.elem.into(),
            });
        }
        let expected_shape = TensorShape {
            n: model.out_channels.get(),
            c: if model.kind == crate::graph::LayerKind::DepthwiseConv2d {
                1
            } else {
                model.in_channels.get() / model.groups.get().max(1)
            },
            h: model.kernel.h.get(),
            w: model.kernel.w.get(),
        };
        if tensor.shape != expected_shape {
            return Err(RtlError::ShapeMismatch {
                layer: layer.to_string(),
                what: format!("weight tensor {} shape", tensor.name),
                expected: expected_shape.elements(),
                actual: tensor.shape.elements(),
            });
        }
        let start = model.weight_byte_offset;
        let end = start.saturating_add(model.weight_bytes);
        if end > self.blob.len() {
            return Err(RtlError::ShapeMismatch {
                layer: layer.to_string(),
                what: format!("weight blob slice for {}", tensor.name),
                expected: end as u64,
                actual: self.blob.len() as u64,
            });
        }
        Ok(start..end)
    }
}

impl WeightResolver for GraphWeightResolver<'_> {
    fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError> {
        let layer =
            tensor
                .name
                .strip_prefix("weights:")
                .ok_or_else(|| RtlError::WeightsNotResident {
                    layer: tensor.name.clone(),
                    tensor: tensor.name.clone(),
                    detail: format!(
                        "the resolver only knows graph weight tensors named {:?}",
                        weight_tensor_name("<layer>")
                    ),
                })?;
        let range = self.layer_range(layer, tensor)?;
        Ok(self.blob[range].to_vec())
    }
}

/// Resolves weight bytes by tensor name from an in-memory table.
///
/// Used for primitive fixtures (see `examples/pointwise`) that are not derived
/// from a manifest, and by tests that want to state weight bytes inline.
#[derive(Debug, Clone, Default)]
pub struct NamedBlobResolver {
    blobs: BTreeMap<String, Vec<u8>>,
}

impl NamedBlobResolver {
    /// An empty resolver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace one tensor's bytes.
    pub fn insert(&mut self, name: impl Into<String>, bytes: Vec<u8>) -> &mut Self {
        self.blobs.insert(name.into(), bytes);
        self
    }

    /// Load one `<tensor-name>.bin` file per tensor.
    ///
    /// The directory is read eagerly, so renaming a file later cannot change the
    /// emitted RTL.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, RtlError> {
        let dir = dir.as_ref();
        let entries = std::fs::read_dir(dir).map_err(|source| RtlError::Io {
            context: format!("reading weight directory {}", dir.display()),
            source,
        })?;
        let mut resolver = NamedBlobResolver::new();
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| RtlError::Io {
                context: format!("reading weight directory {}", dir.display()),
                source,
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                paths.push(path);
            }
        }
        paths.sort();
        for path in paths {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| RtlError::WeightsNotResident {
                    layer: "<weights>".to_string(),
                    tensor: path.display().to_string(),
                    detail: "weight file name is not valid UTF-8".to_string(),
                })?
                .to_string();
            let bytes = std::fs::read(&path).map_err(|source| RtlError::Io {
                context: format!("reading weight file {}", path.display()),
                source,
            })?;
            resolver.blobs.insert(name, bytes);
        }
        if resolver.blobs.is_empty() {
            return Err(RtlError::WeightsNotResident {
                layer: "<weights>".to_string(),
                tensor: dir.display().to_string(),
                detail: "no `*.bin` weight files were found".to_string(),
            });
        }
        Ok(resolver)
    }

    /// Tensor names this resolver can answer for.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.blobs.keys().map(String::as_str)
    }
}

impl WeightResolver for NamedBlobResolver {
    fn tensor_bytes(&self, tensor: &TensorRef) -> Result<Vec<u8>, RtlError> {
        self.blobs
            .get(&tensor.name)
            .cloned()
            .ok_or_else(|| RtlError::WeightsNotResident {
                layer: tensor.name.clone(),
                tensor: tensor.name.clone(),
                detail: format!(
                    "no weight blob named {:?} was supplied (available: {})",
                    tensor.name,
                    self.names().collect::<Vec<_>>().join(", ")
                ),
            })
    }
}
