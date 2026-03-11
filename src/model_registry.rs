//! Model registry for session-model decoupling
//!
//! This module provides a centralized registry for managing multiple models
//! that can be shared across different inference sessions.

use crate::error::{LociError, Result};
use crate::inference::{GenerationParams, InferenceEngine};
use crate::model::ModelConfig;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Unique identifier for a loaded model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(u64);

impl ModelId {
    pub fn as_u64(&self) -> u64 {
        self.0
    }

    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ModelId({})", self.0)
    }
}

/// Registry entry for a loaded model
///
/// Tracks model metadata and manages model lifecycle.
struct ModelEntry {
    id: ModelId,
    path: String,
    n_ctx: u32,
    ref_count: AtomicU64,
    engine: Arc<Mutex<Option<InferenceEngine>>>,
}

/// Central registry for managing multiple models
///
/// The ModelRegistry allows multiple sessions to share the same underlying model,
/// reducing memory usage and enabling efficient multi-session inference.
///
/// # Examples
///
/// ```ignore
/// use loci::model_registry::ModelRegistry;
///
/// let registry = ModelRegistry::new();
/// let model_id = registry.load_model("qwen-0.5b.gguf", 2048)?;
/// ```
pub struct ModelRegistry {
    models: RwLock<HashMap<ModelId, ModelEntry>>,
    next_id: AtomicU64,
    round_robin_cursor: AtomicU64,
}

/// Candidate routing strategy for multi-model orchestration.
#[derive(Debug, Clone)]
pub enum ModelRoutingStrategy {
    /// Try candidates in given order, use first successful model.
    FirstHealthy,
    /// Rotate candidate starting point with a global round-robin cursor.
    RoundRobin,
    /// Probe all candidates with a short prompt and route to the fastest successful one.
    FastestProbe {
        probe_prompt: String,
        probe_max_tokens: usize,
    },
}

/// One routing attempt on a model candidate.
#[derive(Debug, Clone)]
pub struct RoutingAttempt {
    pub model_id: ModelId,
    pub latency_ms: u128,
    pub success: bool,
    pub error: Option<String>,
}

/// Routed generation result.
#[derive(Debug, Clone)]
pub struct RoutedGeneration {
    pub selected_model: ModelId,
    pub response: String,
    pub attempts: Vec<RoutingAttempt>,
}

/// Benchmark sample for one model candidate.
#[derive(Debug, Clone)]
pub struct ModelBenchmark {
    pub model_id: ModelId,
    pub latency_ms: u128,
    pub success: bool,
    pub error: Option<String>,
}

/// One successful model answer in ensemble run.
#[derive(Debug, Clone)]
pub struct EnsembleCandidateResponse {
    pub model_id: ModelId,
    pub latency_ms: u128,
    pub response: String,
}

/// Merge strategy for multi-model ensemble responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleMergeStrategy {
    /// Concatenate all successful answers.
    Concatenate,
    /// Select longest successful answer.
    Longest,
    /// Use a judge model to synthesize final answer from candidates.
    Judge,
}

/// Ensemble generation output.
#[derive(Debug, Clone)]
pub struct EnsembleGeneration {
    pub candidates: Vec<EnsembleCandidateResponse>,
    pub final_response: String,
    pub judge_model: Option<ModelId>,
    pub failures: Vec<RoutingAttempt>,
}

impl ModelRegistry {
    /// Create a new empty model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            round_robin_cursor: AtomicU64::new(0),
        }
    }

    fn get_or_init_engine_cell(&self, model_id: ModelId) -> Result<(String, u32, Arc<Mutex<Option<InferenceEngine>>>)> {
        let models = self.models.read();
        let entry = models.get(&model_id).ok_or(LociError::ModelNotFound)?;
        Ok((entry.path.clone(), entry.n_ctx, Arc::clone(&entry.engine)))
    }

    fn generate_once(
        &self,
        model_id: ModelId,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let (path, n_ctx, engine_cell) = self.get_or_init_engine_cell(model_id)?;
        let mut engine_guard = engine_cell.lock();
        if engine_guard.is_none() {
            let config = ModelConfig::new(PathBuf::from(path)).with_context_size(n_ctx);
            *engine_guard = Some(InferenceEngine::new(config)?);
        }

        let params = GenerationParams {
            max_tokens: max_tokens.min(u32::MAX as usize) as u32,
            ..GenerationParams::default()
        };

        let engine = engine_guard.as_mut().ok_or_else(|| {
            LociError::InferenceError("engine initialization failed".to_string())
        })?;
        engine.generate(prompt, params)
    }

    fn round_robin_order(&self, candidates: &[ModelId]) -> Vec<ModelId> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let start = (self.round_robin_cursor.fetch_add(1, Ordering::SeqCst) as usize) % candidates.len();
        (0..candidates.len())
            .map(|offset| candidates[(start + offset) % candidates.len()])
            .collect()
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
            engine: Arc::new(Mutex::new(None)),
        };

        let mut models = self.models.write();
        models.insert(model_id, entry);

        Ok(model_id)
    }

    /// Check if a model exists
    ///
    /// Returns true if the model ID is valid
    ///
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
            n_ctx: entry.n_ctx,
            ref_count: entry.ref_count.load(Ordering::SeqCst),
        })
    }

    /// Unload a model (decrements reference count)
    ///
    /// The model is only removed from memory when reference count reaches 0
    pub fn unload_model(&self, model_id: ModelId) -> Result<()> {
        self.release_model(model_id)
    }

    /// Acquire an additional reference to a loaded model.
    pub fn acquire_model(&self, model_id: ModelId) -> Result<()> {
        let models = self.models.read();
        let entry = models.get(&model_id).ok_or(LociError::ModelNotFound)?;
        entry.ref_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    /// Release a model reference.
    ///
    /// The model is removed when reference count reaches 0.
    pub fn release_model(&self, model_id: ModelId) -> Result<()> {
        let mut models = self.models.write();
        let entry = models.get(&model_id).ok_or(LociError::ModelNotFound)?;
        let prev_count = entry.ref_count.fetch_sub(1, Ordering::SeqCst);
        if prev_count == 1 {
            models.remove(&model_id);
        }
        Ok(())
    }

    /// Generate text with a registered model via a lazily initialized engine.
    pub fn generate(&self, model_id: ModelId, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_once(model_id, prompt, max_tokens)
    }

    /// Benchmark candidate models with a probe prompt.
    ///
    /// This does not rank by quality; it ranks by successful completion latency.
    pub fn benchmark_models(
        &self,
        candidates: &[ModelId],
        probe_prompt: &str,
        probe_max_tokens: usize,
    ) -> Vec<ModelBenchmark> {
        let mut out = Vec::with_capacity(candidates.len());
        for model_id in candidates {
            let start = Instant::now();
            let result = self.generate_once(*model_id, probe_prompt, probe_max_tokens);
            out.push(ModelBenchmark {
                model_id: *model_id,
                latency_ms: start.elapsed().as_millis(),
                success: result.is_ok(),
                error: result.err().map(|e| e.to_string()),
            });
        }
        out
    }

    /// Route generation across multiple models according to strategy.
    ///
    /// The registry attempts models in routed order and returns first success.
    pub fn generate_routed(
        &self,
        candidates: &[ModelId],
        prompt: &str,
        max_tokens: usize,
        strategy: ModelRoutingStrategy,
    ) -> Result<RoutedGeneration> {
        if candidates.is_empty() {
            return Err(LociError::InvalidArgument(
                "No candidate models provided for routing".to_string(),
            ));
        }

        let ordered = match strategy {
            ModelRoutingStrategy::FirstHealthy => candidates.to_vec(),
            ModelRoutingStrategy::RoundRobin => self.round_robin_order(candidates),
            ModelRoutingStrategy::FastestProbe {
                ref probe_prompt,
                probe_max_tokens,
            } => {
                let mut benches =
                    self.benchmark_models(candidates, probe_prompt, probe_max_tokens);
                benches.sort_by(|a, b| {
                    match (a.success, b.success) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => a.latency_ms.cmp(&b.latency_ms),
                    }
                });
                benches.into_iter().map(|b| b.model_id).collect::<Vec<_>>()
            }
        };

        let mut attempts = Vec::with_capacity(ordered.len());
        for model_id in ordered {
            let start = Instant::now();
            match self.generate_once(model_id, prompt, max_tokens) {
                Ok(response) => {
                    attempts.push(RoutingAttempt {
                        model_id,
                        latency_ms: start.elapsed().as_millis(),
                        success: true,
                        error: None,
                    });
                    return Ok(RoutedGeneration {
                        selected_model: model_id,
                        response,
                        attempts,
                    });
                }
                Err(err) => {
                    attempts.push(RoutingAttempt {
                        model_id,
                        latency_ms: start.elapsed().as_millis(),
                        success: false,
                        error: Some(err.to_string()),
                    });
                }
            }
        }

        Err(LociError::InferenceError(format!(
            "All routed model attempts failed: {}",
            attempts
                .iter()
                .map(|a| {
                    format!(
                        "{}({}ms):{}",
                        a.model_id.as_u64(),
                        a.latency_ms,
                        a.error.clone().unwrap_or_else(|| "unknown".to_string())
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        )))
    }

    /// Run ensemble generation across multiple models and merge outputs.
    pub fn generate_ensemble(
        &self,
        candidates: &[ModelId],
        prompt: &str,
        max_tokens: usize,
        merge_strategy: EnsembleMergeStrategy,
        judge_model: Option<ModelId>,
    ) -> Result<EnsembleGeneration> {
        if candidates.is_empty() {
            return Err(LociError::InvalidArgument(
                "No candidate models provided for ensemble".to_string(),
            ));
        }

        let mut successes = Vec::new();
        let mut failures = Vec::new();

        for model_id in candidates {
            let start = Instant::now();
            match self.generate_once(*model_id, prompt, max_tokens) {
                Ok(response) => successes.push(EnsembleCandidateResponse {
                    model_id: *model_id,
                    latency_ms: start.elapsed().as_millis(),
                    response,
                }),
                Err(err) => failures.push(RoutingAttempt {
                    model_id: *model_id,
                    latency_ms: start.elapsed().as_millis(),
                    success: false,
                    error: Some(err.to_string()),
                }),
            }
        }

        if successes.is_empty() {
            return Err(LociError::InferenceError(
                "Ensemble failed: no candidate model produced a valid response".to_string(),
            ));
        }

        let (final_response, resolved_judge_model) = match merge_strategy {
            EnsembleMergeStrategy::Concatenate => {
                let text = successes
                    .iter()
                    .map(|r| format!("[Model {}]\n{}", r.model_id.as_u64(), r.response))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                (text, None)
            }
            EnsembleMergeStrategy::Longest => {
                let text = successes
                    .iter()
                    .max_by_key(|r| r.response.len())
                    .map(|r| r.response.clone())
                    .unwrap_or_default();
                (text, None)
            }
            EnsembleMergeStrategy::Judge => {
                let judge = judge_model.unwrap_or_else(|| successes[0].model_id);
                let candidates_text = successes
                    .iter()
                    .map(|r| format!("[Model {}]\n{}", r.model_id.as_u64(), r.response))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let judge_prompt = format!(
                    "You are an answer synthesizer.\nOriginal user prompt:\n{}\n\nCandidate answers:\n{}\n\nReturn the best final answer only.",
                    prompt, candidates_text
                );
                (self.generate_once(judge, &judge_prompt, max_tokens)?, Some(judge))
            }
        };

        Ok(EnsembleGeneration {
            candidates: successes,
            final_response,
            judge_model: resolved_judge_model,
            failures,
        })
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
                n_ctx: entry.n_ctx,
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
    pub n_ctx: u32,
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

    #[test]
    fn test_lazy_generation_fails_for_missing_model_file() {
        let registry = ModelRegistry::new();
        let model_id = registry.load_model("missing_model.gguf", 2048).unwrap();
        let result = registry.generate(model_id, "hello", 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_round_robin_order_rotates() {
        let registry = ModelRegistry::new();
        let candidates = vec![ModelId::from_u64(1), ModelId::from_u64(2), ModelId::from_u64(3)];

        let first = registry.round_robin_order(&candidates);
        let second = registry.round_robin_order(&candidates);
        let third = registry.round_robin_order(&candidates);

        assert_eq!(first, vec![ModelId::from_u64(1), ModelId::from_u64(2), ModelId::from_u64(3)]);
        assert_eq!(second, vec![ModelId::from_u64(2), ModelId::from_u64(3), ModelId::from_u64(1)]);
        assert_eq!(third, vec![ModelId::from_u64(3), ModelId::from_u64(1), ModelId::from_u64(2)]);
    }

    #[test]
    fn test_generate_routed_rejects_empty_candidates() {
        let registry = ModelRegistry::new();
        let err = registry
            .generate_routed(&[], "hello", 8, ModelRoutingStrategy::FirstHealthy)
            .unwrap_err();
        assert!(format!("{err}").contains("No candidate models"));
    }

    #[test]
    fn test_benchmark_models_reports_failure_for_missing_file() {
        let registry = ModelRegistry::new();
        let model_id = registry.load_model("missing_model_for_benchmark.gguf", 2048).unwrap();

        let benches = registry.benchmark_models(&[model_id], "probe", 4);
        assert_eq!(benches.len(), 1);
        assert!(!benches[0].success);
        assert!(benches[0].error.is_some());
    }

    #[test]
    fn test_generate_ensemble_rejects_empty_candidates() {
        let registry = ModelRegistry::new();
        let err = registry
            .generate_ensemble(&[], "hi", 8, EnsembleMergeStrategy::Concatenate, None)
            .unwrap_err();
        assert!(format!("{err}").contains("No candidate models"));
    }
}
