use crate::types::*;
use loci_core::{
    EmbeddedModelRegistration, EngineConfig, InferenceEngine, InferenceEngineBuilder, Result,
    RoutingStrategy, SessionRequest, SessionResponse, TieredOffloadProfile,
};
use std::path::PathBuf;

/// High-level facade that can be embedded in-process or exposed as a local AI service.
pub struct Loci {
    engine: InferenceEngine,
    runtime_control: RuntimeControlState,
}

#[derive(Debug, Clone)]
struct RuntimeControlState {
    prefetch_window_bytes: Option<u64>,
}

impl Loci {
    /// Creates a new builder for a Loci SDK instance.
    pub fn builder() -> LociBuilder {
        LociBuilder::default()
    }

    /// Registers one local model using path-based inference.
    pub fn register_local_model(
        &mut self,
        path: impl Into<PathBuf>,
        options: EmbeddedModelRegistration,
    ) -> Result<loci_core::ModelDescriptor> {
        self.engine.register_local_model(path, options)
    }

    /// Registers one local model using the stable high-level SDK request.
    pub fn register_model(
        &mut self,
        request: LocalModelRegistrationRequest,
    ) -> Result<RegisteredModelInfo> {
        let (path, options) = request.into_embedded_parts();
        self.engine
            .register_local_model(path, options)
            .map(RegisteredModelInfo::from)
    }

    /// Returns the currently registered models using the stable high-level SDK shape.
    pub fn list_models(&self) -> Vec<RegisteredModelInfo> {
        self.engine
            .models()
            .into_iter()
            .map(RegisteredModelInfo::from)
            .collect()
    }

    /// Removes one registered model by name or alias.
    pub fn unregister_model(&mut self, name: impl Into<String>) -> ModelMutationResult {
        let name = name.into();
        let changed = self.engine.unregister_model(&name);
        ModelMutationResult { name, changed }
    }

    /// Evicts one prepared model from runtime residency while keeping its registration.
    pub fn evict_model(&mut self, name: impl Into<String>) -> ModelMutationResult {
        let name = name.into();
        let changed = self.engine.evict_model(&name);
        ModelMutationResult { name, changed }
    }

    /// Returns readiness information for all currently registered models.
    pub fn inspect_models(&self) -> Vec<ModelInspectionInfo> {
        self.engine
            .inspect_models()
            .into_iter()
            .map(ModelInspectionInfo::from)
            .collect()
    }

    /// Returns readiness information for one registered model.
    pub fn inspect_model(&self, name: impl AsRef<str>) -> Result<ModelInspectionInfo> {
        self.engine
            .inspect_model(name.as_ref())
            .map(ModelInspectionInfo::from)
    }

    /// Runs one inference request directly in-process.
    pub fn infer(&mut self, request: SessionRequest) -> Result<SessionResponse> {
        self.engine.infer(request)
    }

    /// Runs one high-level text generation request directly in-process.
    pub fn generate_text(
        &mut self,
        request: TextGenerationRequest,
    ) -> Result<TextGenerationResponse> {
        self.engine
            .infer(request.into_session_request())
            .map(TextGenerationResponse::from)
    }

    /// Runs one text generation request and exposes the final response as a simple synchronous stream.
    pub fn generate_text_stream(
        &mut self,
        request: TextGenerationRequest,
    ) -> Result<TextGenerationStream> {
        self.generate_text(request)
            .map(TextGenerationStream::from_response)
    }

    /// Runs one generation request and invokes a callback for each produced chunk.
    pub fn generate_text_with_callback<F>(
        &mut self,
        request: TextGenerationRequest,
        mut on_chunk: F,
    ) -> Result<TextGenerationResponse>
    where
        F: FnMut(&TextGenerationChunk),
    {
        let mut stream = self.generate_text_stream(request)?;
        while let Some(chunk) = stream.next() {
            on_chunk(&chunk);
        }
        Ok(stream.response().clone())
    }

    /// Replays a prepared model through the normal generation path so callers can
    /// keep the prepared session key in their own lifecycle layer.
    pub fn generate_with_prepared_model(
        &mut self,
        prepared: &PreparedModelInfo,
        request: TextGenerationRequest,
    ) -> Result<TextGenerationResponse> {
        let mut request = request;
        if request.model.is_none() {
            request.model = Some(prepared.model_name.clone());
        }
        self.generate_text(request)
    }

    /// Opens one in-process text session backed by a prepared model.
    pub fn open_text_session(&mut self, config: TextSessionConfig) -> Result<TextSession> {
        let model = self.resolve_text_session_model(config.model.as_deref())?;
        let warmup = ModelPreparationRequest::new()
            .model(model.clone())
            .prompt("warmup");
        let prepared = self.prepare_model(warmup)?;
        let transcript = config
            .system_prompt
            .into_iter()
            .map(|content| SessionMessage::new(SessionMessageRole::System, content))
            .collect::<Vec<_>>();
        Ok(TextSession::new(
            prepared.clone(),
            TextGenerationRequest::new("")
                .model(model)
                .max_tokens(config.max_tokens)
                .temperature(config.temperature)
                .structured_output(config.structured_output)
                .tool_calling(config.tool_calling),
            transcript,
        ))
    }

    /// Generates one assistant turn inside an existing in-process text session.
    pub fn generate_in_text_session(
        &mut self,
        session: &mut TextSession,
        user_prompt: impl Into<String>,
    ) -> Result<TextGenerationResponse> {
        let user_prompt = user_prompt.into();
        let mut request = session.request_template().clone();
        request.prompt = session.render_prompt_with(&user_prompt);
        request.model = Some(session.prepared().model_name.clone());
        let response = self.generate_with_prepared_model(session.prepared(), request)?;
        session.push_user_message(user_prompt);
        session.push_assistant_message(response.text.clone());
        Ok(response)
    }

    fn resolve_text_session_model(&self, requested: Option<&str>) -> Result<String> {
        if let Some(model) = requested {
            return Ok(model.to_string());
        }

        let models = self.engine.models();
        match models.as_slice() {
            [only] => Ok(only.name.clone()),
            [] => Err(loci_core::LociError::InvalidRequest(
                "open_text_session requires a model when none are registered".to_string(),
            )),
            _ => Err(loci_core::LociError::InvalidRequest(
                "open_text_session requires an explicit model when multiple models are registered"
                    .to_string(),
            )),
        }
    }

    /// Generates one assistant turn inside a session and emits chunk callbacks locally.
    pub fn generate_in_text_session_with_callback<F>(
        &mut self,
        session: &mut TextSession,
        user_prompt: impl Into<String>,
        mut on_chunk: F,
    ) -> Result<TextGenerationResponse>
    where
        F: FnMut(&TextGenerationChunk),
    {
        let user_prompt = user_prompt.into();
        let mut request = session.request_template().clone();
        request.prompt = session.render_prompt_with(&user_prompt);
        request.model = Some(session.prepared().model_name.clone());
        let response = self.generate_text_with_callback(request, |chunk| on_chunk(chunk))?;
        session.push_user_message(user_prompt);
        session.push_assistant_message(response.text.clone());
        Ok(response)
    }

    /// Warms one model without generating text.
    pub fn prepare(&mut self, request: SessionRequest) -> Result<loci_core::PreparedModel> {
        self.engine.prepare(request)
    }

    /// Warms one model using the stable high-level SDK request.
    pub fn prepare_model(&mut self, request: ModelPreparationRequest) -> Result<PreparedModelInfo> {
        self.engine
            .prepare(request.into_session_request())
            .map(PreparedModelInfo::from)
    }

    /// Returns the stable high-level runtime control configuration.
    pub fn runtime_control_config(&self) -> RuntimeControlConfig {
        runtime_control_config_from_snapshot(
            &self.engine.runtime_snapshot(),
            self.runtime_control.prefetch_window_bytes,
        )
    }

    /// Returns the stable high-level runtime control snapshot.
    pub fn runtime_control_snapshot(&self) -> RuntimeControlSnapshot {
        runtime_control_snapshot_from_snapshot(
            self.engine.runtime_snapshot(),
            self.runtime_control.prefetch_window_bytes,
        )
    }

    /// Exposes the underlying runtime snapshot.
    pub fn runtime_snapshot(&self) -> loci_core::RuntimeSnapshot {
        self.engine.runtime_snapshot()
    }

    /// Updates the model residency keep-alive timeout after build.
    pub fn set_model_keep_alive_secs(&mut self, keep_alive_secs: u64) {
        self.engine.set_model_keep_alive_secs(keep_alive_secs);
    }

    /// Updates the active large-model mode after build.
    pub fn set_large_model_mode(&mut self, profile: TieredOffloadProfile) {
        self.engine.set_offload_profile(profile);
    }

    /// Updates the active tiered-offload profile after build.
    pub fn set_offload_profile(&mut self, profile: TieredOffloadProfile) {
        self.set_large_model_mode(profile);
    }

    /// Updates the spill threshold that activates disk-backed tiering after build.
    pub fn set_spill_threshold_bytes(&mut self, spill_threshold_bytes: Option<u64>) {
        self.engine.set_spill_threshold_bytes(spill_threshold_bytes);
    }

    /// Updates the disk budget available to the tiered-offload runtime after build.
    pub fn set_max_disk_bytes(&mut self, max_disk_bytes: Option<u64>) {
        self.engine.set_max_disk_bytes(max_disk_bytes);
    }

    /// Updates the spill prefetch window used by the disk tier after build.
    pub fn set_prefetch_window_bytes(&mut self, prefetch_window_bytes: Option<u64>) {
        self.engine.set_prefetch_window_bytes(prefetch_window_bytes);
        self.runtime_control.prefetch_window_bytes = prefetch_window_bytes;
    }

    /// Updates the planner-facing KV block size after build.
    pub fn set_kv_block_size_tokens(&mut self, block_size_tokens: u32) {
        self.engine.set_kv_block_size_tokens(block_size_tokens);
    }

    /// Enables or disables shared prefix caching in the paged-KV planner after build.
    pub fn set_kv_prefix_cache_enabled(&mut self, enabled: bool) {
        self.engine.set_kv_prefix_cache_enabled(enabled);
    }

    /// Updates the planner-facing KV tensor formats after build.
    pub fn set_kv_types(&mut self, type_k: impl Into<String>, type_v: impl Into<String>) {
        self.engine.set_kv_types(type_k.into(), type_v.into());
    }

    /// Enables or disables routing after build when the feature is compiled in.
    pub fn set_routing_enabled(&mut self, enabled: bool) -> Result<()> {
        self.engine.set_routing_enabled(enabled)
    }

    /// Updates the routing strategy after build when the feature is compiled in.
    pub fn set_routing_strategy(&mut self, strategy: RoutingStrategy) -> Result<()> {
        self.engine.set_routing_strategy(strategy)
    }

    /// Updates the maximum number of resident models after build.
    pub fn set_max_loaded_models(&mut self, max_loaded_models: Option<usize>) {
        self.engine.set_max_loaded_models(max_loaded_models);
    }

    /// Returns a stable high-level planner/runtime configuration snapshot.
    pub fn runtime_config(&self) -> RuntimeConfigInfo {
        let control = self.runtime_control_config();
        RuntimeConfigInfo {
            model_keep_alive_secs: control.model_keep_alive_secs,
            tiered_offload_enabled: control.tiered_offload_enabled,
            tiered_offload_profile: control.large_model_mode,
            spill_threshold_bytes: control.spill_threshold_bytes,
            max_disk_bytes: control.max_disk_bytes,
            prefetch_window_bytes: control.prefetch_window_bytes,
            kv_cache_enabled: control.kv_cache_enabled,
            kv_block_size_tokens: control.kv_block_size_tokens,
            kv_page_size_bytes: control.kv_page_size_bytes,
            kv_prefix_cache_enabled: control.kv_prefix_cache_enabled,
            kv_type_k: control.kv_type_k,
            kv_type_v: control.kv_type_v,
        }
    }

    /// Returns a stable high-level spill runtime snapshot when tiered offload is active.
    pub fn tiered_offload_runtime(&self) -> Option<TieredOffloadRuntimeInfo> {
        let snapshot = self.engine.runtime_snapshot();
        snapshot
            .tiered_offload_runtime
            .as_ref()
            .map(TieredOffloadRuntimeInfo::from)
    }

    /// Exposes the inner engine for advanced callers.
    pub fn into_engine(self) -> InferenceEngine {
        self.engine
    }

    /// Borrows the inner engine for advanced callers.
    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    /// Mutably borrows the inner engine for advanced callers.
    pub fn engine_mut(&mut self) -> &mut InferenceEngine {
        &mut self.engine
    }

    /// Starts the bundled HTTP service around the current engine.
    #[cfg(feature = "service")]
    pub fn run_http(self, bind: impl Into<String>) -> anyhow::Result<()> {
        self.run_service(LociServiceConfig::with_bind(bind))
    }

    /// Starts the bundled HTTP service using a stable SDK service config.
    #[cfg(feature = "service")]
    pub fn run_service(self, config: LociServiceConfig) -> anyhow::Result<()> {
        let Loci {
            engine,
            runtime_control,
        } = self;
        let runtime_snapshot = engine.runtime_snapshot();
        let runtime_control_view = runtime_control_config_from_snapshot(
            &runtime_snapshot,
            runtime_control.prefetch_window_bytes,
        );
        loci_server::run_server_with_runtime_control(
            loci_server::ServerConfig {
                bind: config.bind,
                engine,
            },
            loci_server::RuntimeControlConfig::new(
                runtime_control_view.prefetch_window_bytes,
                runtime_control_view.routing.enabled,
                runtime_control_view.routing.strategy,
                runtime_control_view.routing.max_loaded_models,
                runtime_control_view.model_keep_alive_secs,
                runtime_control_view.tiered_offload_enabled,
                runtime_control_view.large_model_mode,
                runtime_control_view.spill_threshold_bytes,
                runtime_control_view.max_disk_bytes,
                runtime_control_view.kv_cache_enabled,
                runtime_control_view.kv_block_size_tokens,
                runtime_control_view.kv_page_size_bytes,
                runtime_control_view.kv_prefix_cache_enabled,
                runtime_control_view.kv_type_k,
                runtime_control_view.kv_type_v,
            ),
        )
    }
}

/// Builder used to create a Loci SDK instance.
#[derive(Default)]
pub struct LociBuilder {
    config: Option<EngineConfig>,
    preferred_backend: Option<String>,
    models: Vec<(PathBuf, EmbeddedModelRegistration)>,
}

impl LociBuilder {
    /// Overrides the default runtime configuration.
    pub fn config(mut self, config: EngineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Requests a preferred backend when multiple backends are compiled.
    pub fn preferred_backend(mut self, backend: impl Into<String>) -> Self {
        self.preferred_backend = Some(backend.into());
        self
    }

    /// Updates the model keep-alive timeout without requiring direct `EngineConfig` mutation.
    pub fn model_keep_alive_secs(mut self, keep_alive_secs: u64) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .model_keep_alive_secs = keep_alive_secs;
        self
    }

    /// Sets the active tiered-offload profile without requiring direct `EngineConfig` mutation.
    pub fn tiered_offload_profile(mut self, profile: TieredOffloadProfile) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .tiered_offload
            .profile = profile;
        self
    }

    /// Sets the active large-model mode without requiring direct `EngineConfig` mutation.
    pub fn large_model_mode(self, profile: TieredOffloadProfile) -> Self {
        self.tiered_offload_profile(profile)
    }

    /// Enables or disables tiered offload without requiring direct `EngineConfig` mutation.
    pub fn tiered_offload_enabled(mut self, enabled: bool) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .tiered_offload
            .enabled = enabled;
        self
    }

    /// Overrides the spill threshold that activates disk-backed tiering.
    pub fn spill_threshold_bytes(mut self, spill_threshold_bytes: u64) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .tiered_offload
            .spill_threshold_bytes = Some(spill_threshold_bytes);
        self
    }

    /// Caps the total disk budget available to the tiered offload runtime.
    pub fn max_disk_bytes(mut self, max_disk_bytes: u64) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .tiered_offload
            .max_disk_bytes = Some(max_disk_bytes);
        self
    }

    /// Overrides the spill prefetch window used by the disk tier.
    pub fn prefetch_window_bytes(mut self, prefetch_window_bytes: u64) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .tiered_offload
            .prefetch_window_bytes = Some(prefetch_window_bytes);
        self
    }

    /// Overrides the planner-facing paged-KV block size.
    pub fn kv_block_size_tokens(mut self, block_size_tokens: u32) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .paged_kv
            .block_size_tokens = block_size_tokens;
        self
    }

    /// Enables or disables shared prefix caching in the paged-KV planner.
    pub fn kv_prefix_cache_enabled(mut self, enabled: bool) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .paged_kv
            .prefix_cache_enabled = enabled;
        self
    }

    /// Overrides the planner-facing KV tensor formats.
    pub fn kv_types(mut self, type_k: impl Into<String>, type_v: impl Into<String>) -> Self {
        let config = self.config.get_or_insert_with(EngineConfig::default);
        config.paged_kv.type_k = type_k.into();
        config.paged_kv.type_v = type_v.into();
        self
    }

    /// Limits the number of resident models without requiring direct `EngineConfig` mutation.
    pub fn max_loaded_models(mut self, max_loaded_models: usize) -> Self {
        self.config
            .get_or_insert_with(EngineConfig::default)
            .routing
            .max_loaded_models = Some(max_loaded_models);
        self
    }

    /// Adds a local model to be registered during build.
    pub fn local_model(
        mut self,
        path: impl Into<PathBuf>,
        options: EmbeddedModelRegistration,
    ) -> Self {
        self.models.push((path.into(), options));
        self
    }

    /// Builds the SDK facade.
    pub fn build(self) -> Result<Loci> {
        let runtime_control = runtime_control_state_from_config(self.config.as_ref());
        let mut builder = InferenceEngineBuilder::new();
        if let Some(config) = self.config {
            builder = builder.config(config);
        }
        if let Some(backend) = self.preferred_backend {
            builder = builder.preferred_backend(backend);
        }
        for (path, options) in self.models {
            builder = builder.local_model(path, options)?;
        }
        Ok(Loci {
            engine: builder.build()?,
            runtime_control,
        })
    }
}

impl RuntimeControlState {
    fn from_prefetch_window_bytes(prefetch_window_bytes: Option<u64>) -> Self {
        Self {
            prefetch_window_bytes,
        }
    }
}

fn runtime_control_state_from_config(config: Option<&EngineConfig>) -> RuntimeControlState {
    RuntimeControlState::from_prefetch_window_bytes(
        config
            .map(|config| config.tiered_offload.prefetch_window_bytes)
            .unwrap_or_else(|| EngineConfig::default().tiered_offload.prefetch_window_bytes),
    )
}

fn runtime_control_config_from_snapshot(
    snapshot: &loci_core::RuntimeSnapshot,
    prefetch_window_bytes: Option<u64>,
) -> RuntimeControlConfig {
    RuntimeControlConfig {
        model_keep_alive_secs: snapshot.config.model_keep_alive_secs,
        tiered_offload_enabled: snapshot.config.tiered_offload_enabled,
        large_model_mode: snapshot.config.tiered_offload_profile,
        spill_threshold_bytes: snapshot.config.spill_threshold_bytes,
        max_disk_bytes: snapshot.config.max_disk_bytes,
        prefetch_window_bytes,
        kv_cache_enabled: snapshot.config.kv_cache_enabled,
        kv_block_size_tokens: snapshot.config.kv_block_size_tokens,
        kv_page_size_bytes: snapshot.config.kv_page_size_bytes,
        kv_prefix_cache_enabled: snapshot.config.kv_prefix_cache_enabled,
        kv_type_k: snapshot.config.kv_type_k.clone(),
        kv_type_v: snapshot.config.kv_type_v.clone(),
        routing: RuntimeRoutingConfig {
            enabled: snapshot.routing.enabled,
            strategy: snapshot.routing.strategy.clone(),
            max_loaded_models: snapshot.routing.max_loaded_models,
        },
    }
}

fn runtime_control_snapshot_from_snapshot(
    snapshot: loci_core::RuntimeSnapshot,
    prefetch_window_bytes: Option<u64>,
) -> RuntimeControlSnapshot {
    RuntimeControlSnapshot {
        config: runtime_control_config_from_snapshot(&snapshot, prefetch_window_bytes),
        model_pool: snapshot.model_pool,
        tiered_offload_runtime: snapshot.tiered_offload_runtime,
        features: snapshot.features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "gguf")]
    const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");

    #[cfg(feature = "gguf")]
    fn unique_temp_path(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-sdk-{label}-{suffix}.gguf"))
    }

    #[cfg(feature = "gguf")]
    fn write_minimal_gguf(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"llama";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());

        write_tensor_info(&mut bytes, 3, "token_embd.weight", &[4], 0, 0);
        write_tensor_info(&mut bytes, 3, "blk.0.attn_norm.weight", &[4], 0, 16);
        write_tensor_info(&mut bytes, 3, "output.weight", &[4], 0, 32);

        bytes.extend_from_slice(&[0_u8; 32]);
        for value in 1..=12 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }

        fs::write(path, bytes).expect("gguf");
    }

    #[cfg(feature = "gguf")]
    fn write_tensor_info(
        bytes: &mut Vec<u8>,
        version: u32,
        name: &str,
        dimensions: &[u64],
        ggml_dtype: u32,
        offset: u64,
    ) {
        write_sized_string(bytes, version, name.as_bytes());
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions.iter().rev() {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&ggml_dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    #[cfg(feature = "gguf")]
    fn write_sized_string(bytes: &mut Vec<u8>, version: u32, value: &[u8]) {
        match version {
            1 => bytes.extend_from_slice(&(value.len() as u32).to_le_bytes()),
            2 | 3 => bytes.extend_from_slice(&(value.len() as u64).to_le_bytes()),
            other => panic!("unsupported test gguf version: {other}"),
        }
        bytes.extend_from_slice(value);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn builder_registers_local_models() {
        let path = unique_temp_path("builder");
        write_minimal_gguf(&path);

        let loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let snapshot = loci.runtime_snapshot();
        assert_eq!(snapshot.models.len(), 1);
        assert_eq!(snapshot.models[0].name, "demo");

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn generate_text_uses_high_level_sdk_request() {
        let path = unique_temp_path("generate");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let response = loci
            .generate_text(
                TextGenerationRequest::new("Reply in one short sentence.")
                    .model("demo")
                    .max_tokens(24),
            )
            .expect("response");

        assert_eq!(response.model, "demo");
        assert!(matches!(response.backend.as_str(), "candle" | "openvino"));
        assert!(response.generated_tokens > 0);
        assert!(!response.text.is_empty());

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn prepare_model_uses_high_level_sdk_request() {
        let path = unique_temp_path("prepare");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let prepared = loci
            .prepare_model(ModelPreparationRequest::new().model("demo"))
            .expect("prepared");

        assert_eq!(prepared.model_name, "demo");
        assert!(matches!(prepared.backend.as_str(), "candle" | "openvino"));

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn high_level_model_management_hides_protocol_types() {
        let path = unique_temp_path("manage");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder().build().expect("loci");
        let registered = loci
            .register_model(LocalModelRegistrationRequest::new(path.clone()).name("managed"))
            .expect("registered");

        assert_eq!(registered.name, "managed");
        assert_eq!(registered.format, "gguf");

        let listed = loci.list_models();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "managed");

        let inspected = loci.inspect_model("managed").expect("inspection");
        assert_eq!(inspected.model_name, "managed");
        assert_eq!(inspected.format, "gguf");

        let evicted = loci.evict_model("managed");
        assert!(evicted.changed);

        let removed = loci.unregister_model("managed");
        assert!(removed.changed);
        assert!(loci.list_models().is_empty());

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn generate_text_stream_exposes_sync_chunks() {
        let path = unique_temp_path("stream");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let mut stream = loci
            .generate_text_stream(
                TextGenerationRequest::new("Reply in one short sentence.")
                    .model("demo")
                    .max_tokens(24),
            )
            .expect("stream");

        let mut deltas = Vec::new();
        let mut finished = false;
        for chunk in stream.by_ref() {
            finished = chunk.finished;
            deltas.push(chunk.delta);
        }

        assert!(!deltas.is_empty());
        assert!(finished);
        assert_eq!(stream.response().model, "demo");

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn generate_text_with_callback_emits_chunks() {
        let path = unique_temp_path("callback");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let mut seen = 0usize;
        let response = loci
            .generate_text_with_callback(
                TextGenerationRequest::new("Reply in one short sentence.")
                    .model("demo")
                    .max_tokens(24),
                |_| {
                    seen += 1;
                },
            )
            .expect("response");

        assert!(seen > 0);
        assert_eq!(response.model, "demo");

        fs::remove_file(path).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn open_text_session_reuses_prepared_model_and_tracks_transcript() {
        let path = unique_temp_path("session");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let mut session = loci
            .open_text_session(
                TextSessionConfig::new()
                    .model("demo")
                    .system_prompt("you are local")
                    .max_tokens(24),
            )
            .expect("session");
        let response = loci
            .generate_in_text_session(&mut session, "Reply in one short sentence.")
            .expect("response");

        assert_eq!(response.model, "demo");
        assert!(!session.transcript().is_empty());
        assert!(session
            .transcript()
            .iter()
            .any(|message| matches!(message.role, SessionMessageRole::Assistant)));

        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn service_config_defaults_to_localhost() {
        assert_eq!(LociServiceConfig::default().bind, "127.0.0.1:8080");
    }

    #[test]
    fn service_config_can_be_built_from_host_and_port() {
        let config = LociServiceConfig::with_host_port("0.0.0.0", 18081)
            .host("127.0.0.1")
            .port(19090);

        assert_eq!(config.bind, "127.0.0.1:19090");
        assert_eq!(config.host_name(), "127.0.0.1");
        assert_eq!(config.port_number(), 19090);
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn open_text_session_requires_explicit_model_when_multiple_models_are_registered() {
        let path_a = unique_temp_path("session-a");
        let path_b = unique_temp_path("session-b");
        write_minimal_gguf(&path_a);
        write_minimal_gguf(&path_b);

        let mut loci = Loci::builder()
            .local_model(
                path_a.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo-a".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .local_model(
                path_b.clone(),
                EmbeddedModelRegistration {
                    name: Some("demo-b".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let error = loci
            .open_text_session(TextSessionConfig::new().max_tokens(24))
            .expect_err("explicit model should be required");

        assert!(matches!(error, loci_core::LociError::InvalidRequest(_)));

        fs::remove_file(path_a).expect("cleanup");
        fs::remove_file(path_b).expect("cleanup");
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn open_text_session_uses_single_registered_model_by_default() {
        let path = unique_temp_path("single-session");
        write_minimal_gguf(&path);

        let mut loci = Loci::builder()
            .local_model(
                path.clone(),
                EmbeddedModelRegistration {
                    name: Some("solo".to_string()),
                    ..EmbeddedModelRegistration::default()
                },
            )
            .build()
            .expect("loci");

        let session = loci
            .open_text_session(TextSessionConfig::new().max_tokens(16))
            .expect("session");

        assert_eq!(session.prepared().model_name, "solo");

        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn builder_shortcuts_update_engine_config() {
        let loci = Loci::builder()
            .model_keep_alive_secs(45)
            .max_loaded_models(2)
            .tiered_offload_profile(TieredOffloadProfile::GpuResident)
            .spill_threshold_bytes(2 * 1024 * 1024 * 1024)
            .max_disk_bytes(32 * 1024 * 1024 * 1024)
            .prefetch_window_bytes(64 * 1024 * 1024)
            .kv_block_size_tokens(64)
            .kv_prefix_cache_enabled(false)
            .kv_types("q8_0", "q4_0")
            .build()
            .expect("loci");

        let snapshot = loci.runtime_snapshot();
        assert_eq!(snapshot.config.model_keep_alive_secs, 45);
        assert_eq!(snapshot.routing.max_loaded_models, Some(2));
        assert_eq!(
            snapshot.config.tiered_offload_profile,
            TieredOffloadProfile::GpuResident
        );
        assert_eq!(
            snapshot.config.spill_threshold_bytes,
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            snapshot.config.max_disk_bytes,
            Some(32 * 1024 * 1024 * 1024)
        );
        assert_eq!(snapshot.config.kv_block_size_tokens, 64);
        assert!(!snapshot.config.kv_prefix_cache_enabled);
        assert_eq!(snapshot.config.kv_type_k, "q8_0");
        assert_eq!(snapshot.config.kv_type_v, "q4_0");
    }
}
