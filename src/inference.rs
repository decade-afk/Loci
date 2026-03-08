//! Core inference engine with plugin support
//!
//! This module provides the main `InferenceEngine` which orchestrates:
//! - Backend selection and model loading
//! - Plugin management (text processing hooks)
//! - Hot-swappable RAG plugin management
//! - Unified inference API
//! - Batch inference support
//! - Result caching
//! - Timeout control
//! - Resource management
//! - Concurrency control

use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model};
use crate::concurrency_manager::{ConcurrencyConfig, ConcurrencyManager};
use crate::error::{LociError, Result};
use crate::inference_cache::{CacheConfig, InferenceCache};
use crate::model::ModelConfig;
use crate::plugin::PluginManager;
use crate::rag::{InMemoryRagPlugin, RagDocument, RagPlugin};
use crate::resource_manager::{ResourceLimits, ResourceManager};
use crate::timeout_controller::{TimeoutConfig, TimeoutController, TimeoutGuard};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Parameters for text generation (legacy compatibility)
///
/// This is a convenience wrapper around `InferenceParams` for backward compatibility.
/// New code should use `InferenceParams` directly.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Min-p sampling threshold
    pub min_p: f32,
    /// Top-k sampling threshold
    pub top_k: u32,
    /// Repetition penalty
    pub repeat_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

impl From<GenerationParams> for InferenceParams {
    fn from(params: GenerationParams) -> Self {
        InferenceParams {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
            ..Default::default()
        }
    }
}

/// Main inference engine
///
/// Orchestrates backend, model, plugin, and RAG plugin management.
/// Supports caching, timeout control, resource management, and concurrency.
pub struct InferenceEngine {
    model: Box<dyn Model>,
    backend_registry: BackendRegistry,
    backend_name: String,
    plugin_manager: PluginManager,
    rag_plugins: HashMap<String, Box<dyn RagPlugin>>,
    active_rag_plugin: Option<String>,
    // New features
    cache: InferenceCache,
    timeout_controller: TimeoutController,
    resource_manager: Arc<ResourceManager>,
    concurrency_manager: Arc<ConcurrencyManager>,
    cache_enabled: bool,
    timeout_enabled: bool,
    default_inference_params: InferenceParams,
}

impl InferenceEngine {
    /// Create a new inference engine with default backend (`llama.cpp`)
    pub fn new(config: ModelConfig) -> Result<Self> {
        Self::new_with_backend(config, "llama.cpp")
    }

    /// Create a new inference engine with explicit backend name.
    pub fn new_with_backend(config: ModelConfig, backend_name: &str) -> Result<Self> {
        Self::new_with_registry(config, backend_name, BackendRegistry::with_builtin_backends())
    }

    /// Create a new inference engine with custom backend registry.
    pub fn new_with_registry(
        config: ModelConfig,
        backend_name: &str,
        mut backend_registry: BackendRegistry,
    ) -> Result<Self> {
        config.validate()?;

        let mut backend_params = BackendParams {
            n_gpu_layers: config.n_gpu_layers,
            use_gpu: config.use_gpu,
            options: vec![
                ("n_ctx".to_string(), config.n_ctx.to_string()),
                ("n_batch".to_string(), config.n_batch.to_string()),
            ],
        };
        if let Some(n_threads) = config.n_threads {
            backend_params
                .options
                .push(("n_threads".to_string(), n_threads.to_string()));
        }

        let model = backend_registry.load_model(backend_name, &config.model_path, backend_params)?;

        Ok(Self {
            model,
            backend_registry,
            backend_name: backend_name.to_string(),
            plugin_manager: PluginManager::new(),
            rag_plugins: HashMap::new(),
            active_rag_plugin: None,
            // Initialize new features
            cache: InferenceCache::new(),
            timeout_controller: TimeoutController::new(),
            resource_manager: Arc::new(ResourceManager::new()),
            concurrency_manager: Arc::new(ConcurrencyManager::new()),
            cache_enabled: true,
            timeout_enabled: true,
            default_inference_params: InferenceParams {
                n_ctx: config.n_ctx,
                n_batch: config.n_batch,
                n_threads: config.n_threads,
                ..Default::default()
            },
        })
    }

    /// Create a new builder for configuring the engine.
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Get active backend name.
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// List available backends in current registry.
    pub fn available_backends(&self) -> Vec<&str> {
        self.backend_registry.names()
    }

    /// Register a dynamic backend at runtime.
    pub fn register_dynamic_backend<P: AsRef<Path>>(
        &mut self,
        name: impl Into<String>,
        library_path: P,
    ) -> Result<()> {
        self.backend_registry
            .load_dynamic_backend(name.into(), library_path)
    }

    /// Switch backend and reload model.
    pub fn switch_backend(
        &mut self,
        backend_name: &str,
        config: ModelConfig,
    ) -> Result<()> {
        config.validate()?;
        let mut backend_params = BackendParams {
            n_gpu_layers: config.n_gpu_layers,
            use_gpu: config.use_gpu,
            options: vec![
                ("n_ctx".to_string(), config.n_ctx.to_string()),
                ("n_batch".to_string(), config.n_batch.to_string()),
            ],
        };
        if let Some(n_threads) = config.n_threads {
            backend_params
                .options
                .push(("n_threads".to_string(), n_threads.to_string()));
        }

        let model = self
            .backend_registry
            .load_model(backend_name, &config.model_path, backend_params)?;

        self.model = model;
        self.backend_name = backend_name.to_string();
        self.default_inference_params.n_ctx = config.n_ctx;
        self.default_inference_params.n_batch = config.n_batch;
        self.default_inference_params.n_threads = config.n_threads;
        Ok(())
    }

    /// Get plugin manager (mutable).
    pub fn plugin_manager_mut(&mut self) -> &mut PluginManager {
        &mut self.plugin_manager
    }

    /// Get plugin manager (immutable).
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Register a hot-swappable RAG plugin.
    pub fn register_rag_plugin<P: RagPlugin + 'static>(&mut self, plugin: P) -> Result<()> {
        let name = plugin.name().to_string();
        if self.rag_plugins.contains_key(&name) {
            return Err(LociError::InvalidArgument(format!(
                "RAG plugin already registered: {name}"
            )));
        }

        self.rag_plugins.insert(name, Box::new(plugin));
        Ok(())
    }

    /// Register built-in in-memory RAG plugin.
    pub fn add_in_memory_rag_plugin(
        &mut self,
        name: impl Into<String>,
        documents: Vec<RagDocument>,
        top_k: usize,
        instruction: Option<String>,
    ) -> Result<()> {
        let plugin_name = name.into();
        let mut plugin = InMemoryRagPlugin::new(plugin_name.clone());
        plugin.set_top_k(top_k)?;
        plugin.set_instruction(instruction);
        plugin.ingest_documents(documents)?;
        self.register_rag_plugin(plugin)
    }

    /// Ingest documents into an existing in-memory RAG plugin.
    pub fn ingest_rag_documents(
        &mut self,
        name: &str,
        documents: Vec<RagDocument>,
    ) -> Result<usize> {
        let plugin = self.rag_plugins.get_mut(name).ok_or_else(|| {
            LociError::InvalidArgument(format!("RAG plugin not found: {name}"))
        })?;

        if let Some(in_memory) = plugin.as_any_mut().downcast_mut::<InMemoryRagPlugin>() {
            in_memory.ingest_documents(documents)
        } else {
            Err(LociError::UnsupportedOperation(format!(
                "RAG plugin '{name}' does not support document ingestion"
            )))
        }
    }

    /// Remove a RAG plugin.
    pub fn unregister_rag_plugin(&mut self, name: &str) -> Result<()> {
        if self.rag_plugins.remove(name).is_none() {
            return Err(LociError::InvalidArgument(format!(
                "RAG plugin not found: {name}"
            )));
        }

        if self.active_rag_plugin.as_deref() == Some(name) {
            self.active_rag_plugin = None;
        }
        Ok(())
    }

    /// Activate a specific RAG plugin (hot-swap selection).
    pub fn activate_rag_plugin(&mut self, name: &str) -> Result<()> {
        if !self.rag_plugins.contains_key(name) {
            return Err(LociError::InvalidArgument(format!(
                "RAG plugin not found: {name}"
            )));
        }
        self.active_rag_plugin = Some(name.to_string());
        Ok(())
    }

    /// Deactivate any active RAG plugin.
    pub fn deactivate_rag_plugin(&mut self) {
        self.active_rag_plugin = None;
    }

    /// Get active RAG plugin name.
    pub fn active_rag_plugin(&self) -> Option<&str> {
        self.active_rag_plugin.as_deref()
    }

    /// List RAG plugins as `(name, is_active, indexed_chunks)`.
    pub fn list_rag_plugins(&self) -> Vec<(String, bool, usize)> {
        let active = self.active_rag_plugin.as_deref();
        self.rag_plugins
            .iter()
            .map(|(name, plugin)| {
                (
                    name.clone(),
                    active == Some(name.as_str()),
                    plugin.indexed_chunks(),
                )
            })
            .collect()
    }

    fn apply_rag_if_active(&self, prompt: &str) -> Result<String> {
        if let Some(active_name) = self.active_rag_plugin.as_deref() {
            let plugin = self.rag_plugins.get(active_name).ok_or_else(|| {
                LociError::InvalidArgument(format!(
                    "Active RAG plugin not found: {active_name}"
                ))
            })?;
            plugin.augment_prompt(prompt)
        } else {
            Ok(prompt.to_string())
        }
    }

    /// Generate text from a prompt (legacy API).
    pub fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = self.generation_params_to_inference(params);
        self.generate_with_params(prompt, &inference_params)
    }

    /// Generate text from a prompt with full parameter control.
    pub fn generate_with_params(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<String> {
        // Check cache first
        if self.cache_enabled {
            let cache_key = self.cache.generate_key(prompt, params);
            if let Some(cached) = self.cache.get(cache_key) {
                return Ok(cached);
            }
        }

        // Acquire resources
        let _resource_guard = self.resource_manager.acquire()?;

        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;

        let response = self.model.infer_text(&processed_prompt, params)?;
        let final_response = self.plugin_manager.apply_post_generate(&response)?;

        // Cache the result
        if self.cache_enabled {
            let cache_key = self.cache.generate_key(prompt, params);
            self.cache.insert(cache_key, final_response.clone());
        }

        Ok(final_response)
    }

    /// Generate text with streaming output (legacy API).
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        params: GenerationParams,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        let inference_params = self.generation_params_to_inference(params);
        self.generate_stream_with_params(prompt, &inference_params, callback)
    }

    /// Generate text with streaming output.
    pub fn generate_stream_with_params<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        if !self.model.supports_streaming() {
            return Err(LociError::UnsupportedOperation(
                "Streaming not supported by current backend".to_string(),
            ));
        }

        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;

        let plugin_manager = &self.plugin_manager;
        let mut wrapped_callback = |token: &str| -> bool {
            let processed_token = match plugin_manager.apply_on_token(token) {
                Ok(t) => t,
                Err(_) => return false,
            };
            callback(&processed_token)
        };

        self.model
            .infer_stream(&processed_prompt, params, &mut wrapped_callback)?;

        Ok(())
    }

    /// Get model information.
    pub fn model_info(&self) -> ModelInfo {
        let metadata = self.model.metadata();
        ModelInfo {
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
        }
    }

    /// Get detailed model metadata.
    pub fn model_metadata(&self) -> crate::backend::ModelMetadata {
        self.model.metadata()
    }

    // ==================== Cache Management ====================

    /// Enable or disable result caching
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache_enabled = enabled;
    }

    /// Check if caching is enabled
    pub fn is_cache_enabled(&self) -> bool {
        self.cache_enabled
    }

    /// Configure the inference cache
    pub fn configure_cache(&mut self, config: CacheConfig) {
        self.cache = InferenceCache::with_config(config);
    }

    /// Clear the inference cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> &crate::inference_cache::CacheStats {
        self.cache.stats()
    }

    /// Cleanup expired cache entries
    pub fn cleanup_cache(&mut self) {
        self.cache.cleanup_expired();
    }

    // ==================== Timeout Control ====================

    /// Enable or disable timeout control
    pub fn set_timeout_enabled(&mut self, enabled: bool) {
        self.timeout_enabled = enabled;
    }

    /// Check if timeout control is enabled
    pub fn is_timeout_enabled(&self) -> bool {
        self.timeout_enabled
    }

    /// Configure timeout control
    pub fn configure_timeout(&mut self, config: TimeoutConfig) {
        self.timeout_controller.set_config(config);
    }

    /// Get timeout statistics
    pub fn timeout_stats(&self) -> crate::timeout_controller::TimeoutStats {
        self.timeout_controller.stats()
    }

    // ==================== Resource Management ====================

    /// Get resource manager
    pub fn resource_manager(&self) -> &Arc<ResourceManager> {
        &self.resource_manager
    }

    /// Get resource statistics
    pub fn resource_stats(&self) -> crate::resource_manager::ResourceStats {
        self.resource_manager.get_stats()
    }

    /// Check if system is under load
    pub fn is_under_load(&self) -> bool {
        self.resource_manager.is_under_load()
    }

    /// Get resource usage summary
    pub fn resource_summary(&self) -> String {
        self.resource_manager.summary()
    }

    // ==================== Concurrency Management ====================

    /// Get concurrency manager
    pub fn concurrency_manager(&self) -> &Arc<ConcurrencyManager> {
        &self.concurrency_manager
    }

    /// Get concurrency statistics
    pub fn concurrency_stats(&self) -> crate::concurrency_manager::ConcurrencyStats {
        self.concurrency_manager.stats()
    }

    /// Check if system is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.concurrency_manager.is_at_capacity()
    }

    // ==================== Batch Inference ====================

    /// Generate text for multiple prompts (batch inference)
    pub fn generate_batch(&mut self, prompts: &[String], params: &InferenceParams) -> Result<Vec<Result<String>>> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let result = self.generate_with_params(prompt, params);
            results.push(result);
        }

        Ok(results)
    }

    /// Generate text for multiple prompts with concurrent execution
    pub fn generate_batch_concurrent(&mut self, prompts: Vec<String>, params: InferenceParams) -> Result<Vec<Result<String>>> {
        // Use concurrency manager to limit parallel execution
        let results = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(prompts.len())));
        let params = Arc::new(params);

        // Spawn tasks with concurrency control
        for prompt in prompts {
            // Try to acquire concurrency slot
            match self.concurrency_manager.acquire() {
                Ok(_guard) => {
                    let prompt = prompt.clone();
                    let params = params.clone();
                    let results = results.clone();

                    // In a real async implementation, we'd spawn a task here
                    // For now, we'll do sequential execution with concurrency checks
                    let result = self.generate_with_params(&prompt, &params);
                    results.lock().push(result);
                }
                Err(e) => {
                    results.lock().push(Err(e));
                }
            }
        }

        let results = Arc::try_unwrap(results).unwrap().into_inner();
        Ok(results)
    }

    // ==================== Advanced Generation with Controls ====================

    /// Generate text with timeout control
    pub fn generate_with_timeout(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        timeout: Duration,
    ) -> Result<String> {
        if !self.timeout_enabled {
            return self.generate_with_params(prompt, params);
        }

        let timeout_context = self.timeout_controller.create_context(Some(timeout))?;
        let timeout_controller = self.timeout_controller.clone();
        let _guard = TimeoutGuard::new(&timeout_controller);

        // Check cache first
        if self.cache_enabled {
            let cache_key = self.cache.generate_key(prompt, params);
            if let Some(cached) = self.cache.get(cache_key) {
                return Ok(cached);
            }
        }

        // Acquire resources
        let _resource_guard = self.resource_manager.acquire()?;

        // Check timeout before starting
        timeout_context.check()?;

        // Perform inference
        let result = self.generate_with_params_internal(prompt, params, &timeout_context)?;

        // Cache the result
        if self.cache_enabled {
            let cache_key = self.cache.generate_key(prompt, params);
            self.cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Internal generate method with timeout context
    fn generate_with_params_internal(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        timeout_context: &crate::timeout_controller::TimeoutContext,
    ) -> Result<String> {
        // Check timeout periodically (simplified)
        timeout_context.check()?;

        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;

        let response = self.model.infer_text(&processed_prompt, params)?;
        let final_response = self.plugin_manager.apply_post_generate(&response)?;

        Ok(final_response)
    }

    fn generation_params_to_inference(&self, params: GenerationParams) -> InferenceParams {
        InferenceParams {
            n_ctx: self.default_inference_params.n_ctx,
            n_batch: self.default_inference_params.n_batch,
            n_threads: self.default_inference_params.n_threads,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
        }
    }

    // ==================== Enhanced Streaming ====================

    /// Generate text with streaming output and timeout control
    pub fn generate_stream_with_timeout<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        timeout: Duration,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        if !self.model.supports_streaming() {
            return Err(LociError::UnsupportedOperation(
                "Streaming not supported by current backend".to_string(),
            ));
        }

        if !self.timeout_enabled {
            return self.generate_stream_with_params(prompt, params, callback);
        }

        let timeout_context = self.timeout_controller.create_context(Some(timeout))?;
        let timeout_controller = self.timeout_controller.clone();
        let _guard = TimeoutGuard::new(&timeout_controller);

        // Acquire resources
        let _resource_guard = self.resource_manager.acquire()?;

        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;

        let plugin_manager = &self.plugin_manager;
        let timeout_context = &timeout_context;

        let mut wrapped_callback = |token: &str| -> bool {
            // Check timeout
            if timeout_context.is_timeout() || timeout_context.is_cancelled() {
                return false;
            }

            let processed_token = match plugin_manager.apply_on_token(token) {
                Ok(t) => t,
                Err(_) => return false,
            };
            callback(&processed_token)
        };

        self.model
            .infer_stream(&processed_prompt, params, &mut wrapped_callback)?;

        Ok(())
    }
}

/// Information about the loaded model (legacy compatibility).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Vocabulary size
    pub n_vocab: u32,
    /// Training context size
    pub n_ctx_train: u32,
    /// Embedding dimension
    pub n_embd: u32,
}

/// Builder for configuring InferenceEngine.
pub struct InferenceEngineBuilder {
    model_path: Option<PathBuf>,
    n_ctx: u32,
    n_threads: Option<u32>,
    n_batch: u32,
    use_gpu: bool,
    n_gpu_layers: i32,
    backend_name: String,
    backend_registry: Option<BackendRegistry>,
    dynamic_backends: Vec<(String, PathBuf)>,
    // New builder options
    cache_config: Option<CacheConfig>,
    cache_enabled: bool,
    timeout_config: Option<TimeoutConfig>,
    timeout_enabled: bool,
    resource_limits: Option<ResourceLimits>,
    concurrency_config: Option<ConcurrencyConfig>,
}

impl InferenceEngineBuilder {
    fn new() -> Self {
        Self {
            model_path: None,
            n_ctx: 4096,
            n_threads: None,
            n_batch: 512,
            use_gpu: true,
            n_gpu_layers: -1,
            backend_name: "llama.cpp".to_string(),
            backend_registry: None,
            dynamic_backends: Vec::new(),
            cache_config: None,
            cache_enabled: true,
            timeout_config: None,
            timeout_enabled: true,
            resource_limits: None,
            concurrency_config: None,
        }
    }

    /// Set the model path.
    pub fn model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Select backend by name (e.g. `llama.cpp`, `candle`).
    pub fn backend(mut self, backend_name: impl Into<String>) -> Self {
        self.backend_name = backend_name.into();
        self
    }

    /// Provide a custom backend registry.
    pub fn with_backend_registry(mut self, registry: BackendRegistry) -> Self {
        self.backend_registry = Some(registry);
        self
    }

    /// Schedule loading of a dynamic backend before engine build.
    pub fn load_dynamic_backend<P: Into<PathBuf>>(
        mut self,
        name: impl Into<String>,
        library_path: P,
    ) -> Self {
        self.dynamic_backends
            .push((name.into(), library_path.into()));
        self
    }

    /// Set the context size.
    pub fn context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Set the number of threads.
    pub fn threads(mut self, n_threads: u32) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    /// Set the batch size.
    pub fn batch_size(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    /// Disable GPU acceleration.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self.n_gpu_layers = 0;
        self
    }

    /// Set GPU layers to offload.
    pub fn gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Enable or disable caching
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Configure cache
    pub fn with_cache_config(mut self, config: CacheConfig) -> Self {
        self.cache_config = Some(config);
        self
    }

    /// Enable or disable timeout control
    pub fn with_timeout(mut self, enabled: bool) -> Self {
        self.timeout_enabled = enabled;
        self
    }

    /// Configure timeout
    pub fn with_timeout_config(mut self, config: TimeoutConfig) -> Self {
        self.timeout_config = Some(config);
        self
    }

    /// Set resource limits
    pub fn with_resource_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = Some(limits);
        self
    }

    /// Configure concurrency
    pub fn with_concurrency_config(mut self, config: ConcurrencyConfig) -> Self {
        self.concurrency_config = Some(config);
        self
    }

    /// Build the inference engine.
    pub fn build(self) -> Result<InferenceEngine> {
        let model_path = self
            .model_path
            .ok_or_else(|| LociError::ConfigError("Model path not specified".to_string()))?;

        let mut config = ModelConfig::new(model_path)
            .with_context_size(self.n_ctx)
            .with_batch_size(self.n_batch)
            .with_gpu_layers(self.n_gpu_layers);

        if !self.use_gpu {
            config = config.cpu_only();
        }

        if let Some(threads) = self.n_threads {
            config = config.with_threads(threads);
        }

        let mut registry = self
            .backend_registry
            .unwrap_or_else(BackendRegistry::with_builtin_backends);

        for (name, path) in self.dynamic_backends {
            registry.load_dynamic_backend(name, path)?;
        }

        let mut engine = InferenceEngine::new_with_registry(config, &self.backend_name, registry)?;

        // Apply builder configurations
        engine.cache_enabled = self.cache_enabled;
        engine.timeout_enabled = self.timeout_enabled;

        if let Some(cache_config) = self.cache_config {
            engine.cache = InferenceCache::with_config(cache_config);
        }

        if let Some(timeout_config) = self.timeout_config {
            engine.timeout_controller.set_config(timeout_config);
        }

        if let Some(resource_limits) = self.resource_limits {
            Arc::make_mut(&mut engine.resource_manager).set_limits(resource_limits);
        }

        if let Some(concurrency_config) = self.concurrency_config {
            Arc::make_mut(&mut engine.concurrency_manager).set_config(concurrency_config);
        }

        Ok(engine)
    }
}

