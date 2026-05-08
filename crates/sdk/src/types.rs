use loci_core::{
    ImageInput, PreparedModel, PreparedResidency, RoutingStrategy, SessionRequest, SessionResponse,
    TieredOffloadProfile,
};
use std::path::PathBuf;

/// Stable high-level local model registration request for SDK callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModelRegistrationRequest {
    pub path: PathBuf,
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub memory_bytes: Option<u64>,
    pub parameter_count: Option<u64>,
    pub context_length: Option<u32>,
    pub preferred_backend: Option<String>,
}

impl LocalModelRegistrationRequest {
    /// Creates a registration request for a local model artifact.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            name: None,
            architecture: None,
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        }
    }

    /// Assigns a stable runtime name for the registered model.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Overrides the inferred architecture.
    pub fn architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = Some(architecture.into());
        self
    }

    /// Provides an expected memory footprint for planning.
    pub fn memory_bytes(mut self, memory_bytes: u64) -> Self {
        self.memory_bytes = Some(memory_bytes);
        self
    }

    /// Provides a known parameter count for routing and diagnostics.
    pub fn parameter_count(mut self, parameter_count: u64) -> Self {
        self.parameter_count = Some(parameter_count);
        self
    }

    /// Overrides the inferred context length.
    pub fn context_length(mut self, context_length: u32) -> Self {
        self.context_length = Some(context_length);
        self
    }

    /// Requests a specific backend for this model when available.
    pub fn preferred_backend(mut self, preferred_backend: impl Into<String>) -> Self {
        self.preferred_backend = Some(preferred_backend.into());
        self
    }

    pub(crate) fn into_embedded_parts(self) -> (PathBuf, loci_core::EmbeddedModelRegistration) {
        (
            self.path,
            loci_core::EmbeddedModelRegistration {
                name: self.name,
                architecture: self.architecture,
                memory_bytes: self.memory_bytes,
                parameter_count: self.parameter_count,
                context_length: self.context_length,
                preferred_backend: self.preferred_backend,
            },
        )
    }
}

/// Stable model summary returned by high-level SDK model management APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub format: String,
    pub memory_bytes: Option<u64>,
    pub parameter_count: Option<u64>,
    pub context_length: Option<u32>,
    pub preferred_backend: Option<String>,
}

impl From<loci_core::ModelDescriptor> for RegisteredModelInfo {
    fn from(value: loci_core::ModelDescriptor) -> Self {
        Self {
            format: value.inferred_format().as_str().to_string(),
            name: value.name,
            path: value.path,
            architecture: value.architecture,
            memory_bytes: value.memory_bytes,
            parameter_count: value.parameter_count,
            context_length: value.context_length,
            preferred_backend: value.preferred_backend,
        }
    }
}

/// Stable response returned when removing or evicting a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMutationResult {
    pub name: String,
    pub changed: bool,
}

/// Stable high-level readiness view returned by SDK inspection APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInspectionInfo {
    pub model_name: String,
    pub architecture: String,
    pub format: String,
    pub asset_layout: String,
    pub ready_for_inference: bool,
    pub recommended_backend: Option<String>,
    pub multimodal: bool,
    pub notes: Vec<String>,
}

impl From<loci_core::ModelReadinessReport> for ModelInspectionInfo {
    fn from(value: loci_core::ModelReadinessReport) -> Self {
        Self {
            model_name: value.model_name,
            architecture: value.architecture,
            format: value.inferred_format.as_str().to_string(),
            asset_layout: value.asset_layout.as_str().to_string(),
            ready_for_inference: value.ready_for_inference,
            recommended_backend: value.recommended_backend,
            multimodal: value.multimodal,
            notes: value.notes,
        }
    }
}

/// Stable high-level text generation request for SDK callers.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub images: Vec<GenerationImage>,
    pub structured_output: bool,
    pub tool_calling: bool,
}

impl TextGenerationRequest {
    /// Creates a request with stable SDK defaults.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            model: None,
            max_tokens: 128,
            temperature: 0.2,
            images: Vec::new(),
            structured_output: false,
            tool_calling: false,
        }
    }

    /// Targets a specific registered model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Overrides the output token budget.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Overrides the sampling temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Appends one image input.
    pub fn image(mut self, image: GenerationImage) -> Self {
        self.images.push(image);
        self
    }

    /// Replaces all image inputs.
    pub fn images(mut self, images: Vec<GenerationImage>) -> Self {
        self.images = images;
        self
    }

    /// Enables structured output mode.
    pub fn structured_output(mut self, structured_output: bool) -> Self {
        self.structured_output = structured_output;
        self
    }

    /// Enables tool-calling mode.
    pub fn tool_calling(mut self, tool_calling: bool) -> Self {
        self.tool_calling = tool_calling;
        self
    }

    pub(crate) fn into_session_request(self) -> SessionRequest {
        SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            target_model: self.model,
            images: self
                .images
                .into_iter()
                .map(GenerationImage::into_protocol)
                .collect(),
            structured_output: self.structured_output,
            tool_calling: self.tool_calling,
        }
    }
}

/// Stable high-level text generation response for SDK callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextGenerationResponse {
    pub text: String,
    pub backend: String,
    pub model: String,
    pub estimated_prefill_ms: u64,
    pub estimated_decode_ms: u64,
    pub generated_tokens: u32,
}

impl From<SessionResponse> for TextGenerationResponse {
    fn from(value: SessionResponse) -> Self {
        Self {
            text: value.text,
            backend: value.backend,
            model: value.model,
            estimated_prefill_ms: value.telemetry.estimated_prefill_ms,
            estimated_decode_ms: value.telemetry.estimated_decode_ms,
            generated_tokens: value.telemetry.generated_tokens,
        }
    }
}

/// Simple synchronous stream facade over a completed text generation response.
#[derive(Debug, Clone)]
pub struct TextGenerationStream {
    response: TextGenerationResponse,
    chunks: Vec<String>,
    index: usize,
}

impl TextGenerationStream {
    /// Builds a chunked stream view from a completed response payload.
    pub(crate) fn from_response(response: TextGenerationResponse) -> Self {
        let chunks = split_text_into_chunks(&response.text);
        Self {
            response,
            chunks,
            index: 0,
        }
    }

    /// Returns the complete response metadata and final text.
    pub fn response(&self) -> &TextGenerationResponse {
        &self.response
    }
}

impl Iterator for TextGenerationStream {
    type Item = TextGenerationChunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunks.len() {
            return None;
        }

        let chunk = TextGenerationChunk {
            delta: self.chunks[self.index].clone(),
            finished: self.index + 1 == self.chunks.len(),
        };
        self.index += 1;
        Some(chunk)
    }
}

/// Stable chunk returned by the synchronous SDK text stream wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextGenerationChunk {
    pub delta: String,
    pub finished: bool,
}

/// Stable high-level model preparation request for SDK callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPreparationRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
}

impl ModelPreparationRequest {
    /// Creates a default warmup request.
    pub fn new() -> Self {
        Self {
            model: None,
            prompt: "warmup".to_string(),
            max_tokens: 1,
        }
    }

    /// Targets a specific registered model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Overrides the warmup prompt.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Overrides the warmup token budget.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub(crate) fn into_session_request(self) -> SessionRequest {
        SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: 0.0,
            target_model: self.model,
            images: Vec::new(),
            structured_output: false,
            tool_calling: false,
        }
    }
}

impl Default for ModelPreparationRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable high-level model preparation response for SDK callers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModelInfo {
    pub model_name: String,
    pub backend: String,
    pub session_key: String,
    pub residency: PreparedResidency,
    pub estimated_memory_bytes: Option<u64>,
}

impl From<PreparedModel> for PreparedModelInfo {
    fn from(value: PreparedModel) -> Self {
        Self {
            model_name: value.model_name,
            backend: value.backend,
            session_key: value.session_key,
            residency: value.residency,
            estimated_memory_bytes: value.estimated_memory_bytes,
        }
    }
}

/// Stable role tag used by the in-process SDK chat/session helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMessageRole {
    System,
    User,
    Assistant,
}

/// Stable session message stored inside the SDK-local transcript helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMessage {
    pub role: SessionMessageRole,
    pub content: String,
}

impl SessionMessage {
    /// Creates a new message with the supplied role and content.
    pub fn new(role: SessionMessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Stable configuration used to open an in-process text session on top of a prepared model.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSessionConfig {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub structured_output: bool,
    pub tool_calling: bool,
}

impl TextSessionConfig {
    /// Creates a session config with default local-chat settings.
    pub fn new() -> Self {
        Self {
            model: None,
            system_prompt: None,
            max_tokens: 128,
            temperature: 0.2,
            structured_output: false,
            tool_calling: false,
        }
    }

    /// Targets a specific registered model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Adds a system prompt that is prepended to future transcript renders.
    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Overrides the default token budget used for each turn.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Overrides the default sampling temperature used for each turn.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Enables structured output mode for the session.
    pub fn structured_output(mut self, structured_output: bool) -> Self {
        self.structured_output = structured_output;
        self
    }

    /// Enables tool-calling mode for the session.
    pub fn tool_calling(mut self, tool_calling: bool) -> Self {
        self.tool_calling = tool_calling;
        self
    }
}

impl Default for TextSessionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable in-process session handle that keeps prepared-model metadata and a local transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSession {
    prepared: PreparedModelInfo,
    request_template: TextGenerationRequest,
    transcript: Vec<SessionMessage>,
}

impl TextSession {
    /// Creates one session handle from prepared-model metadata, a template request, and transcript.
    pub(crate) fn new(
        prepared: PreparedModelInfo,
        request_template: TextGenerationRequest,
        transcript: Vec<SessionMessage>,
    ) -> Self {
        Self {
            prepared,
            request_template,
            transcript,
        }
    }

    /// Returns the prepared model metadata backing this session.
    pub fn prepared(&self) -> &PreparedModelInfo {
        &self.prepared
    }

    /// Returns the immutable request template reused for each generated turn.
    pub(crate) fn request_template(&self) -> &TextGenerationRequest {
        &self.request_template
    }

    /// Returns the current local transcript maintained by this session.
    pub fn transcript(&self) -> &[SessionMessage] {
        &self.transcript
    }

    /// Returns the rendered prompt that will be used on the next turn.
    pub fn render_prompt_with(&self, user_prompt: &str) -> String {
        let mut lines = Vec::new();
        for message in &self.transcript {
            lines.push(format!(
                "{}: {}",
                session_role_label(message.role),
                message.content.trim()
            ));
        }
        lines.push(format!("user: {}", user_prompt.trim()));
        lines.join("\n")
    }

    /// Appends one user message to the local transcript.
    pub fn push_user_message(&mut self, content: impl Into<String>) {
        self.transcript
            .push(SessionMessage::new(SessionMessageRole::User, content));
    }

    /// Appends one assistant message to the local transcript.
    pub fn push_assistant_message(&mut self, content: impl Into<String>) {
        self.transcript
            .push(SessionMessage::new(SessionMessageRole::Assistant, content));
    }

    /// Clears the local session transcript while retaining the prepared session.
    pub fn clear_transcript(&mut self) {
        self.transcript.clear();
    }
}

fn session_role_label(role: SessionMessageRole) -> &'static str {
    match role {
        SessionMessageRole::System => "system",
        SessionMessageRole::User => "user",
        SessionMessageRole::Assistant => "assistant",
    }
}

/// Stable high-level routing controls exposed by the SDK runtime APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRoutingConfig {
    pub enabled: bool,
    pub strategy: RoutingStrategy,
    pub max_loaded_models: Option<usize>,
}

/// Stable high-level runtime configuration exposed by the SDK runtime APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeControlConfig {
    pub model_keep_alive_secs: u64,
    pub tiered_offload_enabled: bool,
    pub large_model_mode: TieredOffloadProfile,
    pub spill_threshold_bytes: Option<u64>,
    pub max_disk_bytes: Option<u64>,
    pub prefetch_window_bytes: Option<u64>,
    pub kv_cache_enabled: bool,
    pub kv_block_size_tokens: u32,
    pub kv_page_size_bytes: u64,
    pub kv_prefix_cache_enabled: bool,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub routing: RuntimeRoutingConfig,
}

/// Stable high-level runtime snapshot exposed by the SDK runtime APIs.
#[derive(Debug, Clone)]
pub struct RuntimeControlSnapshot {
    pub config: RuntimeControlConfig,
    pub model_pool: loci_core::ModelPoolSnapshot,
    pub tiered_offload_runtime: Option<loci_core::TieredOffloadRuntimeSnapshot>,
    pub features: loci_core::EngineFeatureSnapshot,
}

/// Stable planner/runtime configuration view exposed by the SDK.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfigInfo {
    pub model_keep_alive_secs: u64,
    pub tiered_offload_enabled: bool,
    pub tiered_offload_profile: TieredOffloadProfile,
    pub spill_threshold_bytes: Option<u64>,
    pub max_disk_bytes: Option<u64>,
    pub prefetch_window_bytes: Option<u64>,
    pub kv_cache_enabled: bool,
    pub kv_block_size_tokens: u32,
    pub kv_page_size_bytes: u64,
    pub kv_prefix_cache_enabled: bool,
    pub kv_type_k: String,
    pub kv_type_v: String,
}

impl From<&loci_core::RuntimeConfigSnapshot> for RuntimeConfigInfo {
    fn from(value: &loci_core::RuntimeConfigSnapshot) -> Self {
        Self {
            model_keep_alive_secs: value.model_keep_alive_secs,
            tiered_offload_enabled: value.tiered_offload_enabled,
            tiered_offload_profile: value.tiered_offload_profile,
            spill_threshold_bytes: value.spill_threshold_bytes,
            max_disk_bytes: value.max_disk_bytes,
            prefetch_window_bytes: None,
            kv_cache_enabled: value.kv_cache_enabled,
            kv_block_size_tokens: value.kv_block_size_tokens,
            kv_page_size_bytes: value.kv_page_size_bytes,
            kv_prefix_cache_enabled: value.kv_prefix_cache_enabled,
            kv_type_k: value.kv_type_k.clone(),
            kv_type_v: value.kv_type_v.clone(),
        }
    }
}

/// Stable spill-session view exposed by the SDK.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredOffloadSessionInfo {
    pub session_key: String,
    pub model_name: String,
    pub spill_path: String,
    pub mapped_bytes: u64,
    pub prefetched_bytes: u64,
    pub scheduled_prefetch_requests: usize,
    pub completed_prefetch_requests: usize,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activations_bytes: u64,
}

impl From<&loci_core::TieredOffloadSessionSnapshot> for TieredOffloadSessionInfo {
    fn from(value: &loci_core::TieredOffloadSessionSnapshot) -> Self {
        Self {
            session_key: value.session_key.clone(),
            model_name: value.model_name.clone(),
            spill_path: value.spill_path.clone(),
            mapped_bytes: value.mapped_bytes,
            prefetched_bytes: value.prefetched_bytes,
            scheduled_prefetch_requests: value.scheduled_prefetch_requests,
            completed_prefetch_requests: value.completed_prefetch_requests,
            weights_bytes: value.weights_bytes,
            kv_cache_bytes: value.kv_cache_bytes,
            activations_bytes: value.activations_bytes,
        }
    }
}

/// Stable tiered-offload runtime view exposed by the SDK.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredOffloadRuntimeInfo {
    pub root_dir: String,
    pub total_spill_bytes: u64,
    pub total_prefetched_bytes: u64,
    pub sessions: Vec<TieredOffloadSessionInfo>,
}

impl From<&loci_core::TieredOffloadRuntimeSnapshot> for TieredOffloadRuntimeInfo {
    fn from(value: &loci_core::TieredOffloadRuntimeSnapshot) -> Self {
        Self {
            root_dir: value.root_dir.clone(),
            total_spill_bytes: value.total_spill_bytes,
            total_prefetched_bytes: value.total_prefetched_bytes,
            sessions: value
                .sessions
                .iter()
                .map(TieredOffloadSessionInfo::from)
                .collect(),
        }
    }
}

/// Stable image input type for high-level generation requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationImage {
    Path(PathBuf),
    Url(String),
    Base64 {
        data_base64: String,
        media_type: Option<String>,
    },
}

impl GenerationImage {
    pub(crate) fn into_protocol(self) -> ImageInput {
        match self {
            Self::Path(path) => ImageInput::Path { path },
            Self::Url(url) => ImageInput::Url { url },
            Self::Base64 {
                data_base64,
                media_type,
            } => ImageInput::Base64 {
                data_base64,
                media_type,
            },
        }
    }
}

/// Stable high-level service startup configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LociServiceConfig {
    pub bind: String,
}

impl LociServiceConfig {
    /// Creates a config using the default local bind address.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a config with an explicit bind address.
    pub fn with_bind(bind: impl Into<String>) -> Self {
        Self { bind: bind.into() }
    }

    /// Creates a config from a host and port pair.
    pub fn with_host_port(host: impl Into<String>, port: u16) -> Self {
        Self {
            bind: format!("{}:{port}", host.into()),
        }
    }

    /// Overrides the bind address.
    pub fn bind(mut self, bind: impl Into<String>) -> Self {
        self.bind = bind.into();
        self
    }

    /// Overrides only the host portion of the bind address.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        let port = self.port_number();
        self.bind = format!("{}:{port}", host.into());
        self
    }

    /// Overrides only the port portion of the bind address.
    pub fn port(mut self, port: u16) -> Self {
        let host = self.host_name().to_string();
        self.bind = format!("{host}:{port}");
        self
    }

    /// Returns the configured host portion of the bind address.
    pub fn host_name(&self) -> &str {
        self.bind
            .rsplit_once(':')
            .map(|(host, _)| host)
            .unwrap_or("127.0.0.1")
    }

    /// Returns the configured port, defaulting to 8080 for malformed binds.
    pub fn port_number(&self) -> u16 {
        self.bind
            .rsplit_once(':')
            .and_then(|(_, port)| port.parse::<u16>().ok())
            .unwrap_or(8080)
    }
}

impl Default for LociServiceConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8080".to_string(),
        }
    }
}

fn split_text_into_chunks(text: &str) -> Vec<String> {
    let chunks: Vec<String> = text
        .split_whitespace()
        .map(|fragment| format!("{fragment} "))
        .collect();

    if chunks.is_empty() {
        vec![text.to_string()]
    } else {
        chunks
    }
}
