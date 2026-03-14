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

use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceParams, Model,
};
use crate::concurrency_manager::{ConcurrencyConfig, ConcurrencyManager};
use crate::error::{LociError, Result};
use crate::function_calling::{
    FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionHandler,
};
use crate::inference_cache::{CacheConfig, InferenceCache};
use crate::mcp::{
    connect_and_register_stdio_server, register_mcp_client_tools, McpClient, McpRegistrationReport,
    McpStdioServerConfig, McpToolRegistrationOptions,
};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::PluginManager;
use crate::rag::{InMemoryRagPlugin, RagDocument, RagPlugin};
use crate::resource_manager::{ResourceLimits, ResourceManager};
use crate::resource_planner::{ResourcePlan, ResourcePlanner};
use crate::skills::{Skill, SkillRegistry};
use crate::timeout_controller::{TimeoutConfig, TimeoutContext, TimeoutController};
use crate::tool_plugin::{
    load_dynamic_tool_plugin as load_dynamic_tool_plugin_impl, LoadedToolPlugin,
    LoadedToolPluginDescriptor,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

fn format_function_definitions_for_prompt(functions: &[&FunctionDefinition]) -> String {
    let mut prompt = String::from("Available functions:\n\n");

    for func in functions {
        prompt.push_str(&format!("Function: {}\n", func.name));
        prompt.push_str(&format!("Description: {}\n", func.description));
        prompt.push_str("Parameters:\n");

        let mut params = func.parameters.iter().collect::<Vec<_>>();
        params.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (param_name, param) in params {
            let required = if func.required.contains(param_name) {
                " (required)"
            } else {
                ""
            };
            prompt.push_str(&format!(
                "  - {}: {}{}\n",
                param_name, param.param_type, required
            ));
            if let Some(desc) = &param.description {
                prompt.push_str(&format!("    {}\n", desc));
            }
        }
        prompt.push('\n');
    }

    prompt.push_str("To call a function, respond with JSON in this format:\n");
    prompt.push_str("{\n");
    prompt.push_str("  \"function\": \"function_name\",\n");
    prompt.push_str("  \"arguments\": {\n");
    prompt.push_str("    \"param1\": \"value1\",\n");
    prompt.push_str("    \"param2\": \"value2\"\n");
    prompt.push_str("  }\n");
    prompt.push_str("}\n");

    prompt
}

fn build_allow_set(allowlist: Option<&[String]>) -> Option<HashSet<String>> {
    allowlist.map(|items| items.iter().map(|s| s.to_string()).collect())
}

fn build_block_set(blocklist: Option<&[String]>) -> HashSet<String> {
    blocklist
        .map(|items| items.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

fn tool_allowed(
    name: &str,
    allow_set: &Option<HashSet<String>>,
    block_set: &HashSet<String>,
) -> bool {
    if block_set.contains(name) {
        return false;
    }
    match allow_set {
        Some(allowed) => allowed.contains(name),
        None => true,
    }
}

/// Pluggable execution policy for inference calls.
///
/// Host applications can replace this to customize scheduling, timeout handling,
/// and resource enforcement around model execution.
pub trait ExecutionPolicy: Send + Sync {
    /// Policy identifier for diagnostics.
    fn name(&self) -> &str;

    /// Execute one non-streaming inference call.
    fn generate_text(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
    ) -> Result<String>;

    /// Execute one streaming inference call.
    fn generate_stream(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()>;
}

/// Built-in default execution policy.
pub struct DefaultExecutionPolicy;

impl DefaultExecutionPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultExecutionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionPolicy for DefaultExecutionPolicy {
    fn name(&self) -> &str {
        "default.execution.policy"
    }

    fn generate_text(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
    ) -> Result<String> {
        if engine.cache_enabled {
            let cache_key = engine.cache.generate_key(prompt, params);
            if let Some(cached) = engine.cache.get(cache_key) {
                return Ok(cached);
            }
        }

        let _concurrency_guard = engine.concurrency_manager.acquire()?;
        let _resource_guard = engine.resource_manager.acquire()?;
        let timeout_context = engine.resolve_timeout_context(timeout_override)?;
        let timeout_started = timeout_context.as_ref().map(|_| Instant::now());

        let result = match timeout_context.as_ref() {
            Some(context) => engine.generate_with_params_internal(prompt, params, context),
            None => engine.generate_text_pipeline(prompt, params),
        };
        if timeout_context.is_some() {
            if let Some(started) = timeout_started {
                engine.record_timeout_completion(started);
            }
            engine.record_timeout_outcome(&result);
        }

        let final_response = result?;
        if engine.cache_enabled {
            let cache_key = engine.cache.generate_key(prompt, params);
            engine.cache.insert(cache_key, final_response.clone());
        }
        Ok(final_response)
    }

    fn generate_stream(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        if !engine.model.supports_streaming() {
            return Err(LociError::UnsupportedOperation(
                "Streaming not supported by current backend".to_string(),
            ));
        }

        let _concurrency_guard = engine.concurrency_manager.acquire()?;
        let _resource_guard = engine.resource_manager.acquire()?;
        let timeout_context = engine.resolve_timeout_context(timeout_override)?;
        let timeout_started = timeout_context.as_ref().map(|_| Instant::now());

        let result =
            engine.generate_stream_pipeline(prompt, params, callback, timeout_context.as_ref());
        if timeout_context.is_some() {
            if let Some(started) = timeout_started {
                engine.record_timeout_completion(started);
            }
            engine.record_timeout_outcome(&result);
        }
        result
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
    function_calling_manager: FunctionCallingManager,
    tool_plugins: Vec<LoadedToolPlugin>,
    skill_registry: SkillRegistry,
    rag_plugins: HashMap<String, Box<dyn RagPlugin>>,
    active_rag_plugin: Option<String>,
    // New features
    cache: InferenceCache,
    timeout_controller: TimeoutController,
    resource_manager: Arc<ResourceManager>,
    concurrency_manager: Arc<ConcurrencyManager>,
    cache_enabled: bool,
    timeout_enabled: bool,
    execution_policy: Arc<dyn ExecutionPolicy>,
    default_inference_params: InferenceParams,
}

impl InferenceEngine {
    fn backend_params_from_config(config: &ModelConfig) -> BackendParams {
        let mut backend_params = BackendParams {
            n_gpu_layers: config.n_gpu_layers,
            use_gpu: config.use_gpu,
            use_mmap: config.use_mmap,
            use_mlock: config.use_mlock,
            kv_offload: config.kv_offload,
            op_offload: config.op_offload,
            split_mode: config.split_mode,
            main_gpu: config.main_gpu,
            tensor_split: config.tensor_split.clone(),
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
        backend_params
    }

    fn config_for_gpu_layer_attempt(config: &ModelConfig, n_gpu_layers: i32) -> ModelConfig {
        if n_gpu_layers <= 0 {
            config.clone().cpu_only()
        } else {
            let mut retry = config.clone();
            retry.n_gpu_layers = n_gpu_layers;
            retry
        }
    }

    fn fallback_gpu_layer_attempts(config: &ModelConfig, step: u32) -> Vec<i32> {
        if !config.use_gpu || config.n_gpu_layers == 0 || step == 0 {
            return Vec::new();
        }

        if config.n_gpu_layers > 0 {
            let step = step as i32;
            let mut attempts = Vec::new();
            let mut next = config.n_gpu_layers - step;
            while next > 0 {
                attempts.push(next);
                next -= step;
            }
            attempts.push(0);
            return attempts;
        }

        let mut attempts = vec![64, 48, 32, 24, 16, 12, 8, 4, 0];
        attempts.dedup();
        attempts
    }

    fn is_retryable_gpu_load_error(error: &LociError) -> bool {
        match error {
            LociError::OutOfMemory(_) | LociError::ResourceExhausted(_) => true,
            LociError::ModelLoadError(message)
            | LociError::BackendError(message)
            | LociError::LlamaCppError(message) => {
                let message = message.to_ascii_lowercase();
                [
                    "out of memory",
                    "failed to allocate",
                    "insufficient",
                    "not enough memory",
                    "cuda",
                    "vram",
                    "oom",
                ]
                .iter()
                .any(|needle| message.contains(needle))
            }
            _ => false,
        }
    }

    fn load_model_with_strategy(
        backend_registry: &mut BackendRegistry,
        backend_name: &str,
        config: &ModelConfig,
    ) -> Result<Box<dyn Model>> {
        let initial_params = Self::backend_params_from_config(config);
        match backend_registry.load_model(backend_name, &config.model_path, initial_params) {
            Ok(model) => Ok(model),
            Err(initial_error) => {
                let ModelLoadStrategy::AutoReduceGpuLayers { step } = config.load_strategy else {
                    return Err(initial_error);
                };

                if !config.use_gpu
                    || config.n_gpu_layers == 0
                    || !Self::is_retryable_gpu_load_error(&initial_error)
                {
                    return Err(initial_error);
                }

                let mut last_error = initial_error;
                for n_gpu_layers in Self::fallback_gpu_layer_attempts(config, step) {
                    let retry_config = Self::config_for_gpu_layer_attempt(config, n_gpu_layers);
                    let retry_params = Self::backend_params_from_config(&retry_config);
                    match backend_registry.load_model(
                        backend_name,
                        &retry_config.model_path,
                        retry_params,
                    ) {
                        Ok(model) => return Ok(model),
                        Err(retry_error) => {
                            if !Self::is_retryable_gpu_load_error(&retry_error) {
                                return Err(retry_error);
                            }
                            last_error = retry_error;
                        }
                    }
                }

                Err(last_error)
            }
        }
    }

    /// Create a new inference engine with default backend (`llama.cpp`)
    pub fn new(config: ModelConfig) -> Result<Self> {
        Self::new_with_backend(config, "llama.cpp")
    }

    /// Create a new inference engine with explicit backend name.
    pub fn new_with_backend(config: ModelConfig, backend_name: &str) -> Result<Self> {
        Self::new_with_registry(
            config,
            backend_name,
            BackendRegistry::with_builtin_backends(),
        )
    }

    /// Create a new inference engine with custom backend registry.
    pub fn new_with_registry(
        config: ModelConfig,
        backend_name: &str,
        mut backend_registry: BackendRegistry,
    ) -> Result<Self> {
        config.validate()?;
        let model = Self::load_model_with_strategy(&mut backend_registry, backend_name, &config)?;

        Ok(Self {
            model,
            backend_registry,
            backend_name: backend_name.to_string(),
            plugin_manager: PluginManager::new(),
            function_calling_manager: FunctionCallingManager::with_builtin_tools(),
            tool_plugins: Vec::new(),
            skill_registry: SkillRegistry::with_builtin_skills(),
            rag_plugins: HashMap::new(),
            active_rag_plugin: None,
            // Initialize new features
            cache: InferenceCache::new(),
            timeout_controller: TimeoutController::new(),
            resource_manager: Arc::new(ResourceManager::new()),
            concurrency_manager: Arc::new(ConcurrencyManager::new()),
            cache_enabled: true,
            timeout_enabled: true,
            execution_policy: Arc::new(DefaultExecutionPolicy::new()),
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

    /// Whether the active backend supports token streaming.
    pub fn supports_streaming(&self) -> bool {
        self.model.supports_streaming()
    }

    /// Whether the active backend supports embedding generation.
    pub fn supports_embeddings(&self) -> bool {
        self.model.supports_embeddings()
    }

    /// Whether the active backend supports multimodal inference.
    pub fn supports_multimodal(&self) -> bool {
        self.model.supports_multimodal()
    }

    /// Declarative backend capabilities for the currently selected backend.
    pub fn backend_capabilities(&self) -> Option<BackendCapabilities> {
        self.backend_registry
            .get(&self.backend_name)
            .map(|backend| backend.capabilities())
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
    pub fn switch_backend(&mut self, backend_name: &str, config: ModelConfig) -> Result<()> {
        config.validate()?;
        let model =
            Self::load_model_with_strategy(&mut self.backend_registry, backend_name, &config)?;

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

    /// Get function/tool manager (mutable).
    pub fn function_calling_manager_mut(&mut self) -> &mut FunctionCallingManager {
        &mut self.function_calling_manager
    }

    /// Get function/tool manager (immutable).
    pub fn function_calling_manager(&self) -> &FunctionCallingManager {
        &self.function_calling_manager
    }

    /// Register a callable tool (function schema + runtime handler).
    pub fn register_tool<H>(&mut self, definition: FunctionDefinition, handler: H) -> Result<()>
    where
        H: FunctionHandler + 'static,
    {
        self.function_calling_manager
            .register_function_with_handler(definition, handler)
    }

    /// Execute a parsed function call against registered tool handlers.
    pub fn execute_function_call(&self, call: &FunctionCall) -> Result<Value> {
        self.function_calling_manager.execute_function_call(call)
    }

    /// Dynamically load one tool plugin (.dll/.so/.dylib) and register its tools.
    pub fn load_dynamic_tool_plugin<P: AsRef<Path>>(
        &mut self,
        library_path: P,
    ) -> Result<(String, Vec<String>)> {
        let loaded =
            load_dynamic_tool_plugin_impl(library_path, &mut self.function_calling_manager)?;
        let name = loaded.name.clone();
        let functions = loaded.function_names.clone();
        self.tool_plugins.push(loaded);
        Ok((name, functions))
    }

    pub fn list_tool_plugins(&self) -> Vec<LoadedToolPluginDescriptor> {
        let mut plugins = self
            .tool_plugins
            .iter()
            .map(|plugin| plugin.descriptor())
            .collect::<Vec<_>>();
        plugins.sort_by(|a, b| a.name.cmp(&b.name));
        plugins
    }

    pub fn unload_dynamic_tool_plugin(&mut self, name: &str) -> Result<()> {
        let index = self
            .tool_plugins
            .iter()
            .position(|plugin| plugin.name == name)
            .ok_or_else(|| {
                crate::error::LociError::PluginError(format!("Tool plugin '{}' not found", name))
            })?;
        if !self.tool_plugins[index].dynamic {
            return Err(crate::error::LociError::PluginError(format!(
                "Static tool plugin '{}' cannot be unloaded at runtime",
                name
            )));
        }

        let plugin = self.tool_plugins.remove(index);
        for function_name in &plugin.function_names {
            self.function_calling_manager
                .unregister_function(function_name);
        }
        Ok(())
    }

    pub fn reload_dynamic_tool_plugin(&mut self, name: &str) -> Result<(String, Vec<String>)> {
        let source = self
            .tool_plugins
            .iter()
            .find(|plugin| plugin.name == name)
            .ok_or_else(|| {
                crate::error::LociError::PluginError(format!("Tool plugin '{}' not found", name))
            })?
            .source
            .clone()
            .ok_or_else(|| {
                crate::error::LociError::PluginError(format!(
                    "Static tool plugin '{}' cannot be reloaded at runtime",
                    name
                ))
            })?;

        self.unload_dynamic_tool_plugin(name)?;
        let (loaded_name, functions) = self.load_dynamic_tool_plugin(&source)?;
        if loaded_name != name {
            return Err(crate::error::LociError::PluginError(format!(
                "Reloaded tool plugin name mismatch: expected '{}', got '{}'",
                name, loaded_name
            )));
        }
        Ok((loaded_name, functions))
    }

    /// Register an MCP client and expose all remote MCP tools to function calling.
    pub fn register_mcp_client(
        &mut self,
        client: Box<dyn McpClient>,
        options: McpToolRegistrationOptions,
    ) -> Result<McpRegistrationReport> {
        register_mcp_client_tools(
            &mut self.function_calling_manager,
            Arc::new(Mutex::new(client)),
            options,
        )
    }

    /// Connect to an MCP stdio server and register all discovered tools.
    pub fn connect_mcp_stdio_server(
        &mut self,
        config: McpStdioServerConfig,
    ) -> Result<McpRegistrationReport> {
        connect_and_register_stdio_server(&mut self.function_calling_manager, config)
    }

    /// Get skill registry (immutable).
    pub fn skill_registry(&self) -> &SkillRegistry {
        &self.skill_registry
    }

    /// Get skill registry (mutable).
    pub fn skill_registry_mut(&mut self) -> &mut SkillRegistry {
        &mut self.skill_registry
    }

    /// Register one skill definition.
    pub fn register_skill(&mut self, skill: Skill) -> Result<()> {
        self.skill_registry.register_skill(skill)
    }

    /// Generate with tool-calling loop enabled.
    ///
    /// The model can emit a function-call JSON payload. The engine executes the
    /// tool and feeds result back into the next round until a final answer is produced.
    pub fn generate_with_tools(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        max_rounds: usize,
    ) -> Result<String> {
        self.generate_with_tools_policy(prompt, params, max_rounds, None, None)
    }

    /// Generate with tool-calling + allowlist policy.
    pub fn generate_with_tools_filtered(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        max_rounds: usize,
        allowed_tools: Option<&[String]>,
    ) -> Result<String> {
        self.generate_with_tools_policy(prompt, params, max_rounds, allowed_tools, None)
    }

    /// Generate with tool-calling + allow/deny policy.
    pub fn generate_with_tools_policy(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        max_rounds: usize,
        allowed_tools: Option<&[String]>,
        blocked_tools: Option<&[String]>,
    ) -> Result<String> {
        let rounds = max_rounds.max(1);
        let allow_set = build_allow_set(allowed_tools);
        let block_set = build_block_set(blocked_tools);
        let selected_functions = self
            .function_calling_manager
            .list_functions()
            .into_iter()
            .filter(|f| tool_allowed(&f.name, &allow_set, &block_set))
            .collect::<Vec<_>>();

        if selected_functions.is_empty() {
            return self.generate_with_params(prompt, params);
        }

        let tools_prompt = format_function_definitions_for_prompt(&selected_functions);
        let mut interaction_log = String::new();
        let mut current_prompt = format!(
            "{tools_prompt}\nUser request:\n{prompt}\n\nIf a tool is needed, output ONLY JSON function call. Otherwise answer directly."
        );

        for round_idx in 0..rounds {
            let response = self.generate_with_params(&current_prompt, params)?;
            let parsed_call = self
                .function_calling_manager
                .parse_function_call(&response)?;
            if let Some(call) = parsed_call {
                let exec_result = if !tool_allowed(&call.name, &allow_set, &block_set) {
                    json!({
                        "ok": false,
                        "tool": call.name,
                        "error": "Tool is not allowed by active policy"
                    })
                } else {
                    match self.function_calling_manager.execute_function_call(&call) {
                        Ok(value) => json!({
                            "ok": true,
                            "tool": call.name,
                            "result": value
                        }),
                        Err(err) => json!({
                            "ok": false,
                            "tool": call.name,
                            "error": err.to_string()
                        }),
                    }
                };

                interaction_log.push_str(&format!(
                    "Round {}:\nFunction call: {}\nExecution: {}\n\n",
                    round_idx + 1,
                    serde_json::to_string(&call).unwrap_or_else(|_| "{}".to_string()),
                    serde_json::to_string(&exec_result).unwrap_or_else(|_| "{}".to_string())
                ));

                current_prompt = format!(
                    "{tools_prompt}\nUser request:\n{prompt}\n\nTool interaction history:\n{interaction_log}\nProvide final answer. If another tool is still needed, output JSON function call."
                );
                continue;
            }

            return Ok(response);
        }

        Err(LociError::ResourceExhausted(format!(
            "Tool-calling exceeded max rounds ({rounds}) without final answer"
        )))
    }

    /// Generate using a named skill (prompt + tool policy + optional rounds override).
    pub fn generate_with_skill(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        skill_name: &str,
        max_rounds: usize,
    ) -> Result<String> {
        let skill = self
            .skill_registry
            .get(skill_name)
            .cloned()
            .ok_or_else(|| LociError::InvalidArgument(format!("Unknown skill: {skill_name}")))?;
        let composed_prompt = skill.compose_prompt(prompt);
        let rounds = skill.max_tool_rounds.unwrap_or(max_rounds).max(1);
        let allowed = if skill.tool_policy.allowed.is_empty() {
            None
        } else {
            Some(skill.tool_policy.allowed.clone())
        };
        let blocked = if skill.tool_policy.blocked.is_empty() {
            None
        } else {
            Some(skill.tool_policy.blocked.clone())
        };
        self.generate_with_tools_policy(
            &composed_prompt,
            params,
            rounds,
            allowed.as_deref(),
            blocked.as_deref(),
        )
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
        let plugin = self
            .rag_plugins
            .get_mut(name)
            .ok_or_else(|| LociError::InvalidArgument(format!("RAG plugin not found: {name}")))?;

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
                LociError::InvalidArgument(format!("Active RAG plugin not found: {active_name}"))
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
        let policy = Arc::clone(&self.execution_policy);
        policy.generate_text(self, prompt, params, None)
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
        let policy = Arc::clone(&self.execution_policy);
        policy.generate_stream(self, prompt, params, None, &mut callback)
    }

    fn generate_text_pipeline(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;
        let response = self.model.infer_text(&processed_prompt, params)?;
        self.plugin_manager.apply_post_generate(&response)
    }

    fn generate_stream_pipeline(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        callback: &mut dyn FnMut(&str) -> bool,
        timeout_context: Option<&TimeoutContext>,
    ) -> Result<()> {
        let rag_prompt = self.apply_rag_if_active(prompt)?;
        let processed_prompt = self.plugin_manager.apply_pre_generate(&rag_prompt)?;

        let plugin_manager = &self.plugin_manager;
        let mut wrapped_callback = |token: &str| -> bool {
            if let Some(ctx) = timeout_context {
                if ctx.is_timeout() || ctx.is_cancelled() {
                    return false;
                }
            }
            let processed_token = match plugin_manager.apply_on_token(token) {
                Ok(t) => t,
                Err(_) => return false,
            };
            callback(&processed_token)
        };

        self.model
            .infer_stream(&processed_prompt, params, &mut wrapped_callback)?;

        if let Some(ctx) = timeout_context {
            ctx.check()?;
        }
        Ok(())
    }

    fn resolve_timeout_context(
        &self,
        timeout_override: Option<Duration>,
    ) -> Result<Option<TimeoutContext>> {
        if timeout_override.is_some() || self.timeout_enabled {
            Ok(Some(
                self.timeout_controller.create_context(timeout_override)?,
            ))
        } else {
            Ok(None)
        }
    }

    fn record_timeout_outcome<T>(&self, result: &Result<T>) {
        if let Err(LociError::Timeout(message)) = result {
            if message.to_ascii_lowercase().contains("cancelled") {
                self.timeout_controller.record_cancellation();
            } else {
                self.timeout_controller.record_timeout();
            }
        }
    }

    fn record_timeout_completion(&self, started_at: Instant) {
        let elapsed_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        self.timeout_controller.record_completion(elapsed_ms);
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

    /// Generate embeddings for input text when supported by the active backend.
    pub fn generate_embeddings(&mut self, text: &str) -> Result<Vec<f32>> {
        let _concurrency_guard = self.concurrency_manager.acquire()?;
        let _resource_guard = self.resource_manager.acquire()?;
        self.model.generate_embeddings(text)
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

    /// Replace execution policy with a custom implementation.
    pub fn set_execution_policy<P>(&mut self, policy: P)
    where
        P: ExecutionPolicy + 'static,
    {
        self.execution_policy = Arc::new(policy);
    }

    /// Replace execution policy with an Arc policy object.
    pub fn set_execution_policy_arc(&mut self, policy: Arc<dyn ExecutionPolicy>) {
        self.execution_policy = policy;
    }

    /// Current execution policy name.
    pub fn execution_policy_name(&self) -> &str {
        self.execution_policy.name()
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
    pub fn generate_batch(
        &mut self,
        prompts: &[String],
        params: &InferenceParams,
    ) -> Result<Vec<Result<String>>> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let result = self.generate_with_params(prompt, params);
            results.push(result);
        }

        Ok(results)
    }

    /// Generate text for multiple prompts with concurrent execution
    pub fn generate_batch_concurrent(
        &mut self,
        prompts: Vec<String>,
        params: InferenceParams,
    ) -> Result<Vec<Result<String>>> {
        // Sequential implementation; each call enforces shared concurrency policy.
        let results = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(prompts.len())));
        let params = Arc::new(params);

        for prompt in prompts {
            let prompt = prompt.clone();
            let params = params.clone();
            let results = results.clone();
            let result = self.generate_with_params(&prompt, &params);
            results.lock().push(result);
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
        let policy = Arc::clone(&self.execution_policy);
        policy.generate_text(self, prompt, params, Some(timeout))
    }

    /// Internal generate method with timeout context
    fn generate_with_params_internal(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        timeout_context: &TimeoutContext,
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
        if !self.timeout_enabled {
            return self.generate_stream_with_params(prompt, params, callback);
        }
        let policy = Arc::clone(&self.execution_policy);
        policy.generate_stream(self, prompt, params, Some(timeout), &mut callback)
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
    use_mmap: bool,
    use_mlock: bool,
    kv_offload: bool,
    op_offload: bool,
    split_mode: GpuSplitMode,
    main_gpu: u32,
    tensor_split: Option<Vec<f32>>,
    load_strategy: ModelLoadStrategy,
    resource_plan: Option<ResourcePlan>,
    auto_resource_plan: bool,
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
    execution_policy: Option<Arc<dyn ExecutionPolicy>>,
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
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            load_strategy: ModelLoadStrategy::Strict,
            resource_plan: None,
            auto_resource_plan: false,
            backend_name: "llama.cpp".to_string(),
            backend_registry: None,
            dynamic_backends: Vec::new(),
            cache_config: None,
            cache_enabled: true,
            timeout_config: None,
            timeout_enabled: true,
            resource_limits: None,
            concurrency_config: None,
            execution_policy: None,
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
        self.kv_offload = false;
        self.op_offload = false;
        self.split_mode = GpuSplitMode::None;
        self.main_gpu = 0;
        self.tensor_split = None;
        self
    }

    /// Set GPU layers to offload.
    pub fn gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Enable or disable memory-mapped model loading.
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// Enable or disable memory locking for model pages.
    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    /// Enable or disable K/Q/V and KV cache offload.
    pub fn with_kv_offload(mut self, kv_offload: bool) -> Self {
        self.kv_offload = kv_offload;
        self
    }

    /// Enable or disable host op offload.
    pub fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.op_offload = op_offload;
        self
    }

    /// Set the multi-GPU split strategy.
    pub fn with_gpu_split_mode(mut self, split_mode: GpuSplitMode) -> Self {
        self.split_mode = split_mode;
        self
    }

    /// Set the primary GPU index used for single-GPU placement.
    pub fn with_main_gpu(mut self, main_gpu: u32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    /// Set relative split weights across multiple GPUs.
    pub fn with_tensor_split(mut self, tensor_split: Vec<f32>) -> Self {
        self.tensor_split = Some(tensor_split);
        self
    }

    /// Apply an explicit resource plan before model load.
    pub fn with_resource_plan(mut self, resource_plan: ResourcePlan) -> Self {
        self.resource_plan = Some(resource_plan);
        self
    }

    /// Derive GPU/CPU placement from the model file and detected hardware.
    pub fn with_auto_resource_plan(mut self, enabled: bool) -> Self {
        self.auto_resource_plan = enabled;
        self
    }

    /// Set the model loading strategy.
    pub fn with_load_strategy(mut self, load_strategy: ModelLoadStrategy) -> Self {
        self.load_strategy = load_strategy;
        self
    }

    /// Retry model loading with progressively fewer GPU layers when placement fails.
    pub fn with_auto_gpu_layer_fallback(mut self, step: u32) -> Self {
        self.load_strategy = ModelLoadStrategy::AutoReduceGpuLayers { step };
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

    /// Replace default execution policy.
    pub fn with_execution_policy<P>(mut self, policy: P) -> Self
    where
        P: ExecutionPolicy + 'static,
    {
        self.execution_policy = Some(Arc::new(policy));
        self
    }

    /// Replace default execution policy using Arc.
    pub fn with_execution_policy_arc(mut self, policy: Arc<dyn ExecutionPolicy>) -> Self {
        self.execution_policy = Some(policy);
        self
    }

    /// Build the inference engine.
    pub fn build(self) -> Result<InferenceEngine> {
        let model_path = self
            .model_path
            .ok_or_else(|| LociError::ConfigError("Model path not specified".to_string()))?;

        let resolved_resource_plan = if let Some(plan) = self.resource_plan.clone() {
            Some(plan)
        } else if self.auto_resource_plan && self.use_gpu {
            Some(ResourcePlanner::plan_for_model(&model_path, self.n_ctx)?)
        } else {
            None
        };

        let (
            use_gpu,
            n_gpu_layers,
            use_mmap,
            use_mlock,
            kv_offload,
            op_offload,
            split_mode,
            main_gpu,
            tensor_split,
        ) = if let Some(plan) = &resolved_resource_plan {
            (
                plan.use_gpu,
                plan.n_gpu_layers,
                plan.use_mmap,
                plan.use_mlock,
                plan.kv_offload,
                plan.op_offload,
                plan.split_mode,
                plan.main_gpu,
                plan.tensor_split.clone(),
            )
        } else {
            (
                self.use_gpu,
                self.n_gpu_layers,
                self.use_mmap,
                self.use_mlock,
                self.kv_offload,
                self.op_offload,
                self.split_mode,
                self.main_gpu,
                self.tensor_split.clone(),
            )
        };

        let mut config = ModelConfig::new(model_path)
            .with_context_size(self.n_ctx)
            .with_batch_size(self.n_batch)
            .with_gpu_layers(n_gpu_layers)
            .with_mmap(use_mmap)
            .with_mlock(use_mlock)
            .with_kv_offload(kv_offload)
            .with_op_offload(op_offload)
            .with_gpu_split_mode(split_mode)
            .with_main_gpu(main_gpu)
            .with_load_strategy(self.load_strategy);

        if let Some(tensor_split) = tensor_split {
            config = config.with_tensor_split(tensor_split);
        }

        if !use_gpu {
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

        if let Some(execution_policy) = self.execution_policy {
            engine.execution_policy = execution_policy;
        }

        Ok(engine)
    }
}
