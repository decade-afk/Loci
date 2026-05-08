use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use loci_protocol::{
    ImageInput, ModelDescriptor, RoutingStrategy, SessionRequest, TieredOffloadProfile,
};

/// Request body used to register a new model into the live server runtime.
#[derive(Debug, Deserialize)]
pub(crate) struct RegisterModelRequest {
    pub name: String,
    pub path: PathBuf,
    #[serde(default = "default_architecture")]
    pub architecture: String,
    #[serde(default)]
    pub memory_bytes: Option<u64>,
    #[serde(default)]
    pub parameter_count: Option<u64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub preferred_backend: Option<String>,
}

/// Request body used to unregister a registered model.
#[derive(Debug, Deserialize)]
pub(crate) struct UnregisterModelRequest {
    pub name: String,
}

/// Request body used to evict a prepared model while preserving registration.
#[derive(Debug, Deserialize)]
pub(crate) struct EvictModelRequest {
    pub name: String,
}

/// Request body used to prewarm a model.
#[derive(Debug, Deserialize)]
pub(crate) struct PrewarmModelRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_prewarm_prompt")]
    pub prompt: String,
    #[serde(default = "default_prewarm_tokens")]
    pub max_tokens: u32,
}

/// Request body used to inspect one registered model.
#[derive(Debug, Deserialize)]
pub(crate) struct InspectModelRequest {
    pub name: String,
}

/// One normalized HTTP image input variant accepted by the inference endpoints.
#[derive(Debug, Deserialize)]
pub(crate) struct HttpImageInput {
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub data_base64: Option<String>,
    #[serde(default)]
    pub media_type: Option<String>,
}

/// Request body used to register an alias for an existing model.
#[derive(Debug, Deserialize)]
pub(crate) struct RegisterAliasRequest {
    pub alias: String,
    pub target: String,
}

/// Request body used to remove an alias from the model registry.
#[derive(Debug, Deserialize)]
pub(crate) struct RemoveAliasRequest {
    pub alias: String,
}

/// Request body used to update planner-visible runtime knobs.
#[derive(Debug, Deserialize)]
pub(crate) struct UpdatePlannerConfigRequest {
    #[serde(default)]
    pub keep_alive_secs: Option<u64>,
    #[serde(default)]
    pub tiered_offload_enabled: Option<bool>,
    #[serde(default)]
    pub offload_profile: Option<TieredOffloadProfile>,
    #[serde(default)]
    pub large_model_mode: Option<TieredOffloadProfile>,
    #[serde(default)]
    pub spill_threshold_bytes: Option<u64>,
    #[serde(default)]
    pub max_disk_bytes: Option<u64>,
    #[serde(default)]
    pub prefetch_window_bytes: Option<u64>,
    #[serde(default)]
    pub kv_block_size_tokens: Option<u32>,
    #[serde(default)]
    pub kv_prefix_cache_enabled: Option<bool>,
    #[serde(default)]
    pub kv_type_k: Option<String>,
    #[serde(default)]
    pub kv_type_v: Option<String>,
}

/// Request body used to update dynamic routing controls.
#[derive(Debug, Deserialize)]
pub(crate) struct UpdateRoutingConfigRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub strategy: Option<RoutingStrategy>,
    #[serde(default)]
    pub max_loaded_models: Option<Option<usize>>,
}

/// Request body used to update the complete runtime-control view.
#[derive(Debug, Deserialize)]
pub(crate) struct UpdateRuntimeControlRequest {
    #[serde(default)]
    pub keep_alive_secs: Option<u64>,
    #[serde(default)]
    pub tiered_offload_enabled: Option<bool>,
    #[serde(default)]
    pub offload_profile: Option<TieredOffloadProfile>,
    #[serde(default)]
    pub large_model_mode: Option<TieredOffloadProfile>,
    #[serde(default)]
    pub spill_threshold_bytes: Option<Option<u64>>,
    #[serde(default)]
    pub max_disk_bytes: Option<Option<u64>>,
    #[serde(default)]
    pub prefetch_window_bytes: Option<Option<u64>>,
    #[serde(default)]
    pub kv_block_size_tokens: Option<u32>,
    #[serde(default)]
    pub kv_prefix_cache_enabled: Option<bool>,
    #[serde(default)]
    pub kv_type_k: Option<String>,
    #[serde(default)]
    pub kv_type_v: Option<String>,
    #[serde(default)]
    pub routing_enabled: Option<bool>,
    #[serde(default)]
    pub routing_strategy: Option<RoutingStrategy>,
    #[serde(default)]
    pub max_loaded_models: Option<Option<usize>>,
}

/// Request body used by generic inference endpoints.
#[derive(Debug, Deserialize)]
pub(crate) struct InferenceHttpRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub target_model: Option<String>,
    #[serde(default)]
    pub images: Vec<HttpImageInput>,
    #[serde(default)]
    pub structured_output: bool,
    #[serde(default)]
    pub tool_calling: bool,
    #[serde(default)]
    pub stream: bool,
}

/// Request body used by the planning-only endpoint.
#[derive(Debug, Deserialize)]
pub(crate) struct PlanHttpRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub target_model: Option<String>,
    #[serde(default)]
    pub images: Vec<HttpImageInput>,
    #[serde(default)]
    pub structured_output: bool,
    #[serde(default)]
    pub tool_calling: bool,
}

/// OpenAI-compatible completions request body.
#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

/// OpenAI-compatible chat completions request body.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionsRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub stream: bool,
}

/// One chat message in the OpenAI-compatible chat request.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: ChatMessageContent,
}

/// Permits text-only or multipart chat message bodies.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ChatMessageContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

/// One message part in a multipart OpenAI-compatible chat request.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ChatContentPart {
    Text { text: String },
    ImageUrl { image_url: ChatImageReference },
    InputText { text: String },
    InputImage { image_url: ChatImageReference },
}

/// Normalizes either plain-string or object image references from OpenAI-style chat requests.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ChatImageReference {
    String(String),
    Object {
        url: String,
        #[serde(default)]
        #[serde(rename = "detail")]
        _detail: Option<String>,
    },
}

/// OpenAI-compatible model list response body.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiModelListResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModel>,
}

/// One model entry in the OpenAI-compatible model list response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiModel {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

/// OpenAI-compatible text completion response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompletionChoice>,
}

/// One choice entry in the OpenAI-compatible completion response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
}

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChatCompletionChoice>,
}

/// One choice entry in the OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionChoice {
    pub index: u32,
    pub message: OpenAiChatAssistantMessage,
    pub finish_reason: &'static str,
}

/// Assistant message payload used by the OpenAI-compatible chat response.
#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatAssistantMessage {
    pub role: &'static str,
    pub content: String,
}

impl RegisterModelRequest {
    /// Converts the registration payload into the normalized planner-facing model descriptor.
    pub(crate) fn into_model(self) -> ModelDescriptor {
        #[cfg(feature = "gguf")]
        let gguf_summary = if self
            .path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false)
        {
            loci_core::read_gguf_metadata_summary(&self.path).ok()
        } else {
            None
        };

        #[cfg(not(feature = "gguf"))]
        let gguf_summary: Option<()> = None;

        #[cfg(feature = "gguf")]
        let inferred_architecture = gguf_summary
            .as_ref()
            .and_then(|summary| summary.architecture.clone())
            .unwrap_or_else(|| self.architecture.clone());

        #[cfg(not(feature = "gguf"))]
        let inferred_architecture = self.architecture.clone();

        #[cfg(feature = "gguf")]
        let normalized_architecture = loci_core::resolve_gguf_architecture(&inferred_architecture)
            .map(|spec| spec.canonical_name.to_string())
            .unwrap_or(inferred_architecture);

        #[cfg(not(feature = "gguf"))]
        let normalized_architecture = inferred_architecture;

        #[cfg(feature = "gguf")]
        let inferred_context_length = gguf_summary
            .as_ref()
            .and_then(|summary| summary.context_length);

        #[cfg(not(feature = "gguf"))]
        let inferred_context_length: Option<u32> = None;

        ModelDescriptor {
            name: self.name,
            path: self.path,
            architecture: normalized_architecture,
            memory_bytes: self.memory_bytes,
            parameter_count: self.parameter_count,
            context_length: self.context_length.or(inferred_context_length),
            preferred_backend: self.preferred_backend,
        }
    }
}

impl InferenceHttpRequest {
    /// Converts the inference payload into the normalized internal session request.
    pub(crate) fn into_request(self) -> Result<SessionRequest, String> {
        Ok(SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            target_model: self.target_model,
            images: collect_http_images(self.images)?,
            structured_output: self.structured_output,
            tool_calling: self.tool_calling,
        })
    }
}

impl PlanHttpRequest {
    /// Converts the planning payload into the normalized internal session request.
    pub(crate) fn into_request(self) -> Result<SessionRequest, String> {
        Ok(SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            target_model: self.target_model,
            images: collect_http_images(self.images)?,
            structured_output: self.structured_output,
            tool_calling: self.tool_calling,
        })
    }
}

impl PrewarmModelRequest {
    /// Converts the prewarm payload into a single-token internal warmup request.
    pub(crate) fn into_request(self) -> SessionRequest {
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

impl HttpImageInput {
    /// Converts the external HTTP image payload into the normalized internal image input.
    pub(crate) fn into_protocol(self) -> Result<ImageInput, String> {
        if let Some(path) = self.path {
            return Ok(ImageInput::Path { path });
        }
        if let Some(url) = self.url {
            return Ok(ImageInput::Url { url });
        }
        if let Some(data_base64) = self.data_base64 {
            return Ok(ImageInput::Base64 {
                data_base64,
                media_type: self.media_type,
            });
        }
        Err("image input must provide one of: path, url, data_base64".to_string())
    }
}

impl ChatImageReference {
    /// Returns the canonical URL string regardless of the source chat-image payload shape.
    pub(crate) fn url(&self) -> &str {
        match self {
            ChatImageReference::String(url) => url,
            ChatImageReference::Object { url, _detail: _ } => url,
        }
    }
}

/// Collects and validates a batch of HTTP image payloads into internal image inputs.
pub(crate) fn collect_http_images(images: Vec<HttpImageInput>) -> Result<Vec<ImageInput>, String> {
    images
        .into_iter()
        .map(HttpImageInput::into_protocol)
        .collect()
}

/// Default token budget for generic inference and OpenAI-compatible request bodies.
pub(crate) const fn default_max_tokens() -> u32 {
    128
}

/// Default temperature used by generic inference and OpenAI-compatible request bodies.
pub(crate) const fn default_temperature() -> f32 {
    0.2
}

/// Default architecture fallback used by model registration payloads when no architecture is supplied.
pub(crate) fn default_architecture() -> String {
    "llama".to_string()
}

/// Default prompt used when prewarming a model without an explicit warmup prompt.
pub(crate) fn default_prewarm_prompt() -> String {
    "warmup".to_string()
}

/// Default token budget used when prewarming a model without an explicit warmup token count.
pub(crate) const fn default_prewarm_tokens() -> u32 {
    1
}
