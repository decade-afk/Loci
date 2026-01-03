//! Model registry for session-model decoupling
//!
//! This module provides a centralized registry for managing multiple models
//! that can be shared across different inference sessions.

use crate::error::{LociError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Unique identifier for a loaded model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(u64);

impl ModelId {
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelId({})", self.0)
    }
}

/// Registry entry for a loaded model
///
/// Note: This is currently a placeholder that tracks model metadata.
/// Full model loading integration will be added in a future update.
struct ModelEntry {
    id: ModelId,
    path: String,
    n_ctx: u32,
    ref_count: AtomicU64,
}

/// Central registry for managing multiple models
///
/// The ModelRegistry allows multiple sessions to share the same underlying model,
/// reducing memory usage and enabling efficient multi-session inference.
///
/// # Examples
///
/// ```no_run
/// use loci::model_registry::ModelRegistry;
///
/// let registry = ModelRegistry::new();
/// let model_id = registry.load_model("qwen-0.5b.gguf", 2048)?;
/// ```
pub struct ModelRegistry {
    models: RwLock<HashMap<ModelId, ModelEntry>>,
    next_id: AtomicU64,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Load a model from file and register it
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF model file
    /// * `n_ctx` - Context size (max tokens)
    ///
    /// # Returns
    ///
    /// ModelId that can be used to create sessions with this model
    ///
    /// # Note
    ///
    /// This is currently a placeholder that registers model metadata.
    /// Full model loading integration will be added in a future update.
    pub fn load_model<P: AsRef<Path>>(&self, path: P, n_ctx: u32) -> Result<ModelId> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| LociError::InvalidModelPath)?
            .to_string();

        // Check if model already loaded (by path)
        {
            let models = self.models.read();
            if let Some(entry) = models.values().find(|e| e.path == path_str) {
                entry.ref_count.fetch_add(1, Ordering::SeqCst);
                return Ok(entry.id);
            }
        }

        // Register model metadata (actual loading deferred)
        let model_id = ModelId(self.next_id.fetch_add(1, Ordering::SeqCst));

        let entry = ModelEntry {
            id: model_id,
            path: path_str,
            n_ctx,
            ref_count: AtomicU64::new(1),
        };

        let mut models = self.models.write();
        models.insert(model_id, entry);

        Ok(model_id)
    }

    /// Check if a model exists
    ///
    /// Returns true if the model ID is valid
    ///
    /// # Note
    ///
    /// In the full implementation, this would return Arc<dyn Model> for actual inference.
    /// Currently this is a placeholder for architecture demonstration.
    #[allow(dead_code)]
    fn model_exists(&self, model_id: ModelId) -> bool {
        self.models.read().contains_key(&model_id)
    }

    /// Get model information
    pub fn get_model_info(&self, model_id: ModelId) -> Option<ModelInfo> {
        let models = self.models.read();
        models.get(&model_id).map(|entry| ModelInfo {
            id: entry.id,
            path: entry.path.clone(),
            ref_count: entry.ref_count.load(Ordering::SeqCst),
        })
    }

    /// Unload a model (decrements reference count)
    ///
    /// The model is only removed from memory when reference count reaches 0
    pub fn unload_model(&self, model_id: ModelId) -> Result<()> {
        let mut models = self.models.write();

        if let Some(entry) = models.get(&model_id) {
            let prev_count = entry.ref_count.fetch_sub(1, Ordering::SeqCst);

            if prev_count == 1 {
                // Last reference, remove from registry
                models.remove(&model_id);
            }

            Ok(())
        } else {
            Err(LociError::ModelNotFound)
        }
    }

    /// Get number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.read().len()
    }

    /// List all loaded models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .map(|entry| ModelInfo {
                id: entry.id,
                path: entry.path.clone(),
                ref_count: entry.ref_count.load(Ordering::SeqCst),
            })
            .collect()
    }

    /// Check if a model is loaded
    pub fn has_model(&self, model_id: ModelId) -> bool {
        self.models.read().contains_key(&model_id)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: ModelId,
    pub path: String,
    pub ref_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new();
        assert_eq!(registry.model_count(), 0);
    }

    #[test]
    fn test_model_id_display() {
        let id = ModelId(42);
        assert_eq!(format!("{}", id), "ModelId(42)");
    }

    #[test]
    fn test_registry_operations() {
        let registry = ModelRegistry::new();

        // Test initial state
        assert_eq!(registry.model_count(), 0);
        assert!(!registry.has_model(ModelId(1)));

        // Test list empty
        let models = registry.list_models();
        assert_eq!(models.len(), 0);
    }
}
