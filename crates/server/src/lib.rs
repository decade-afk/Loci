use anyhow::Context;
use loci_core::{InferenceEngine, InferenceParams, LociError, ModelConfig, ModelLoadStrategy};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ServerConfig {
    pub bind: String,
    pub engine: InferenceEngine,
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct ModelLoadRequest {
    backend_name: String,
    config: ModelLoadConfigRequest,
}

#[derive(Debug, Deserialize)]
struct ModelLoadConfigRequest {
    model_path: PathBuf,
    #[serde(default = "default_n_ctx")]
    n_ctx: u32,
    #[serde(default)]
    n_threads: Option<u32>,
    #[serde(default = "default_n_batch")]
    n_batch: u32,
    #[serde(default = "default_true")]
    use_gpu: bool,
    #[serde(default = "default_n_gpu_layers")]
    n_gpu_layers: i32,
    #[serde(default = "default_true")]
    use_mmap: bool,
    #[serde(default)]
    use_mlock: bool,
    #[serde(default = "default_true")]
    kv_offload: bool,
    #[serde(default = "default_true")]
    op_offload: bool,
    #[serde(default)]
    split_mode: loci_core::GpuSplitMode,
    #[serde(default)]
    main_gpu: u32,
    #[serde(default)]
    tensor_split: Option<Vec<f32>>,
    #[serde(default)]
    load_strategy: Option<ModelLoadStrategyRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ModelLoadStrategyRequest {
    Strict,
    AutoReduceGpuLayers { step: u32 },
}

#[derive(Debug, Deserialize)]
struct OpenAiCompletionsRequest {
    model: String,
    prompt: PromptInput,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatCompletionsRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatMessage {
    role: String,
    content: OpenAiChatContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PromptInput {
    Single(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenAiChatContent {
    Text(String),
    Parts(Vec<OpenAiChatPart>),
}

#[derive(Debug, Deserialize)]
struct OpenAiChatPart {
    #[serde(rename = "type")]
    part_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAiModelListResponse {
    object: &'static str,
    data: Vec<OpenAiModel>,
}

#[derive(Debug, Serialize)]
struct OpenAiModel {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiCompletionChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiCompletionChoice {
    text: String,
    index: u32,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<OpenAiChatCompletionChoice>,
    usage: OpenAiUsage,
}

#[derive(Debug, Serialize)]
struct OpenAiChatCompletionChoice {
    index: u32,
    message: OpenAiChatAssistantMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct OpenAiChatAssistantMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Debug, Serialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'static str,
}

#[derive(Debug)]
struct ServerHttpError {
    status: &'static str,
    message: String,
    error_type: &'static str,
    code: &'static str,
}

impl ServerHttpError {
    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            status: "400 Bad Request",
            message: message.into(),
            error_type: "invalid_request_error",
            code: "invalid_request",
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: "404 Not Found",
            message: message.into(),
            error_type: "not_found_error",
            code: "not_found",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: "500 Internal Server Error",
            message: message.into(),
            error_type: "server_error",
            code: "internal_error",
        }
    }

    fn into_response(self) -> String {
        let body = serde_json::to_string(&OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: self.message,
                error_type: self.error_type,
                code: self.code,
            },
        })
        .unwrap_or_else(|_| {
            r#"{"error":{"message":"serialization failure","type":"server_error","code":"internal_error"}}"#
                .to_string()
        });
        http_response(self.status, "application/json", &body)
    }
}

impl From<serde_json::Error> for ServerHttpError {
    fn from(error: serde_json::Error) -> Self {
        Self::invalid_request(format!("invalid json body: {error}"))
    }
}

pub fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&config.bind)
        .with_context(|| format!("failed to bind server on {}", config.bind))?;
    let mut engine = config.engine;

    for stream in listener.incoming() {
        let mut stream = stream?;
        let mut buffer = [0u8; 64 * 1024];
        let size = stream.read(&mut buffer)?;
        let request = String::from_utf8_lossy(&buffer[..size]);
        let response = handle_request(&mut engine, &request)?;
        stream.write_all(response.as_bytes())?;
        stream.flush()?;
    }

    Ok(())
}

fn handle_request(engine: &mut InferenceEngine, request: &str) -> anyhow::Result<String> {
    Ok(match route_request(engine, request) {
        Ok(response) => response,
        Err(error) => error.into_response(),
    })
}

fn route_request(engine: &mut InferenceEngine, request: &str) -> Result<String, ServerHttpError> {
    let mut lines = request.lines();
    let request_line = lines.next().unwrap_or_default();
    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();

    if request_line.starts_with("GET /health ") {
        return Ok(json_response(r#"{"status":"ok"}"#));
    }

    if request_line.starts_with("GET /v1/runtime ") {
        let json = serde_json::to_string(&engine.runtime_snapshot())
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("POST /v1/inference/generate ") {
        let payload: GenerateRequest = serde_json::from_str(body)?;
        let mut params = InferenceParams::default();
        if let Some(max_tokens) = payload.max_tokens {
            params.max_tokens = max_tokens;
        }
        if let Some(temperature) = payload.temperature {
            params.temperature = temperature;
        }
        let response = engine
            .infer(&payload.prompt, &params)
            .map_err(map_engine_error)?;
        let json = serde_json::to_string(&response)
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("POST /v1/model/load ") {
        let payload: ModelLoadRequest = serde_json::from_str(body)?;
        let config = payload.config.into_model_config();
        engine
            .load_model_config(&payload.backend_name, &config)
            .map_err(map_engine_error)?;
        let json = serde_json::to_string(&engine.runtime_snapshot())
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("POST /v1/model/unload ") {
        let status = engine.unload_model();
        let json = serde_json::to_string(&status)
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("GET /v1/models ") {
        let response = openai_models_response(engine)?;
        let json = serde_json::to_string(&response)
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("POST /v1/completions ") {
        let payload: OpenAiCompletionsRequest = serde_json::from_str(body)?;
        if payload.stream {
            return Err(ServerHttpError::invalid_request(
                "streaming is not supported by loci-server",
            ));
        }
        let model_id = require_active_model(engine, &payload.model)?;
        let prompt = payload.prompt.into_prompt();
        let params = completion_params(payload.max_tokens, payload.temperature, payload.top_p);
        let output = engine
            .generate(&prompt, &params)
            .map_err(map_engine_error)?;
        let response = OpenAiCompletionResponse {
            id: generated_id("cmpl"),
            object: "text_completion",
            created: unix_timestamp(),
            model: model_id,
            choices: vec![OpenAiCompletionChoice {
                text: output.clone(),
                index: 0,
                finish_reason: "stop",
            }],
            usage: usage_for(&prompt, &output),
        };
        let json = serde_json::to_string(&response)
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    if request_line.starts_with("POST /v1/chat/completions ") {
        let payload: OpenAiChatCompletionsRequest = serde_json::from_str(body)?;
        if payload.stream {
            return Err(ServerHttpError::invalid_request(
                "streaming is not supported by loci-server",
            ));
        }
        let model_id = require_active_model(engine, &payload.model)?;
        let prompt = flatten_chat_messages(&payload.messages)?;
        let params = completion_params(payload.max_tokens, payload.temperature, payload.top_p);
        let output = engine
            .generate(&prompt, &params)
            .map_err(map_engine_error)?;
        let response = OpenAiChatCompletionResponse {
            id: generated_id("chatcmpl"),
            object: "chat.completion",
            created: unix_timestamp(),
            model: model_id,
            choices: vec![OpenAiChatCompletionChoice {
                index: 0,
                message: OpenAiChatAssistantMessage {
                    role: "assistant",
                    content: output.clone(),
                },
                finish_reason: "stop",
            }],
            usage: usage_for(&prompt, &output),
        };
        let json = serde_json::to_string(&response)
            .map_err(|error| ServerHttpError::internal(error.to_string()))?;
        return Ok(json_response(&json));
    }

    Err(ServerHttpError::not_found("route not found"))
}

fn json_response(body: &str) -> String {
    http_response("200 OK", "application/json", body)
}

fn http_response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn openai_models_response(
    engine: &InferenceEngine,
) -> Result<OpenAiModelListResponse, ServerHttpError> {
    let data = engine
        .model_path()
        .map(|path| OpenAiModel {
            id: model_identifier(path),
            object: "model",
            created: unix_timestamp(),
            owned_by: "loci",
        })
        .into_iter()
        .collect();
    Ok(OpenAiModelListResponse {
        object: "list",
        data,
    })
}

fn completion_params(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> InferenceParams {
    let mut params = InferenceParams::default();
    if let Some(value) = max_tokens {
        params.max_tokens = value;
    }
    if let Some(value) = temperature {
        params.temperature = value;
    }
    if let Some(value) = top_p {
        params.top_p = value;
    }
    params
}

fn require_active_model(
    engine: &InferenceEngine,
    requested_model: &str,
) -> Result<String, ServerHttpError> {
    let path = engine
        .model_path()
        .ok_or_else(|| ServerHttpError::invalid_request("no active model is loaded"))?;
    if model_matches(path, requested_model) {
        return Ok(model_identifier(path));
    }

    Err(ServerHttpError::invalid_request(format!(
        "requested model `{requested_model}` does not match the active model `{}`",
        path.display()
    )))
}

fn model_matches(path: &Path, requested_model: &str) -> bool {
    requested_model == path.display().to_string()
        || path
            .file_name()
            .and_then(|value| value.to_str())
            .map(|value| value == requested_model)
            .unwrap_or(false)
}

fn model_identifier(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn flatten_chat_messages(messages: &[OpenAiChatMessage]) -> Result<String, ServerHttpError> {
    if messages.is_empty() {
        return Err(ServerHttpError::invalid_request(
            "messages must contain at least one item",
        ));
    }

    let mut prompt = String::new();
    for message in messages {
        let role = message.role.trim();
        if role.is_empty() {
            return Err(ServerHttpError::invalid_request(
                "message role must not be empty",
            ));
        }
        let content = message.content.flatten()?;
        if content.trim().is_empty() {
            return Err(ServerHttpError::invalid_request(
                "message content must not be empty",
            ));
        }
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(content.trim());
        prompt.push_str("\n\n");
    }
    prompt.push_str("assistant:");
    Ok(prompt)
}

fn usage_for(prompt: &str, output: &str) -> OpenAiUsage {
    let prompt_tokens = approximate_token_count(prompt);
    let completion_tokens = approximate_token_count(output);
    OpenAiUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

fn approximate_token_count(text: &str) -> u32 {
    text.split_whitespace().count() as u32
}

fn generated_id(prefix: &str) -> String {
    format!("{prefix}-{}", unix_timestamp())
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn map_engine_error(error: LociError) -> ServerHttpError {
    match error {
        LociError::Other(error) => ServerHttpError::internal(error.to_string()),
        other => ServerHttpError::invalid_request(other.to_string()),
    }
}

impl ModelLoadConfigRequest {
    fn into_model_config(self) -> ModelConfig {
        ModelConfig {
            model_path: self.model_path,
            n_ctx: self.n_ctx,
            n_threads: self.n_threads,
            n_batch: self.n_batch,
            use_gpu: self.use_gpu,
            n_gpu_layers: self.n_gpu_layers,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            kv_offload: self.kv_offload,
            op_offload: self.op_offload,
            split_mode: self.split_mode,
            main_gpu: self.main_gpu,
            tensor_split: self.tensor_split,
            load_strategy: match self
                .load_strategy
                .unwrap_or(ModelLoadStrategyRequest::Strict)
            {
                ModelLoadStrategyRequest::Strict => ModelLoadStrategy::Strict,
                ModelLoadStrategyRequest::AutoReduceGpuLayers { step } => {
                    ModelLoadStrategy::AutoReduceGpuLayers { step }
                }
            },
        }
    }
}

impl PromptInput {
    fn into_prompt(self) -> String {
        match self {
            Self::Single(prompt) => prompt,
            Self::Many(prompts) => prompts.join("\n"),
        }
    }
}

impl OpenAiChatContent {
    fn flatten(&self) -> Result<String, ServerHttpError> {
        match self {
            Self::Text(text) => Ok(text.clone()),
            Self::Parts(parts) => {
                let mut text = String::new();
                for part in parts {
                    if part.part_type == "text" {
                        if let Some(value) = &part.text {
                            text.push_str(value);
                        }
                    }
                }
                if text.is_empty() {
                    return Err(ServerHttpError::invalid_request(
                        "chat message content must contain at least one text part",
                    ));
                }
                Ok(text)
            }
        }
    }
}

const fn default_n_ctx() -> u32 {
    4096
}

const fn default_n_batch() -> u32 {
    512
}

const fn default_true() -> bool {
    true
}

const fn default_n_gpu_layers() -> i32 {
    -1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_model_path(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-server-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("demo.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        path
    }

    #[test]
    fn server_loads_and_unloads_model_via_http_routes() {
        let mut engine = InferenceEngine::builder().build().expect("build");
        let model_path = temp_model_path("load-unload");
        let request = format!(
            "POST /v1/model/load HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{{\"backend_name\":\"mock\",\"config\":{{\"model_path\":\"{}\",\"use_gpu\":false,\"n_gpu_layers\":0,\"kv_offload\":false,\"op_offload\":false,\"split_mode\":\"none\"}}}}",
            model_path.display().to_string().replace('\\', "/")
        );

        let load_response = handle_request(&mut engine, &request).expect("load response");
        assert!(load_response.contains("\"active_backend\":\"mock\""));
        assert!(load_response.contains("\"active_model_path\""));

        let unload_response = handle_request(
            &mut engine,
            "POST /v1/model/unload HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{}",
        )
        .expect("unload response");
        assert!(unload_response.contains("\"unloaded\":true"));
        assert!(engine.active_backend().is_none());
    }

    #[test]
    fn server_lists_active_model_for_openai_models_route() {
        let model_path = temp_model_path("models-route");
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let response = handle_request(&mut engine, "GET /v1/models HTTP/1.1\r\n\r\n")
            .expect("models response");
        assert!(response.contains("\"object\":\"list\""));
        assert!(response.contains("\"id\":\"demo.gguf\""));
    }

    #[test]
    fn server_supports_openai_completions_route() {
        let model_path = temp_model_path("openai-completions");
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let response = handle_request(
            &mut engine,
            "POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo.gguf\",\"prompt\":\"hello\",\"max_tokens\":32}",
        )
        .expect("completion response");

        assert!(response.contains("\"object\":\"text_completion\""));
        assert!(response.contains("\"model\":\"demo.gguf\""));
        assert!(response.contains("mock:hello"));
    }

    #[test]
    fn server_supports_openai_chat_completions_route() {
        let model_path = temp_model_path("openai-chat");
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let response = handle_request(
            &mut engine,
            "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo.gguf\",\"messages\":[{\"role\":\"system\",\"content\":\"be concise\"},{\"role\":\"user\",\"content\":\"hello\"}]}",
        )
        .expect("chat completion response");

        assert!(response.contains("\"object\":\"chat.completion\""));
        assert!(response.contains("\"role\":\"assistant\""));
        assert!(response.contains("mock:system: be concise"));
        assert!(response.contains("user: hello"));
    }

    #[test]
    fn server_rejects_openai_streaming_requests() {
        let model_path = temp_model_path("openai-stream");
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let response = handle_request(
            &mut engine,
            "POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo.gguf\",\"prompt\":\"hello\",\"stream\":true}",
        )
        .expect("stream rejection");

        assert!(response.contains("400 Bad Request"));
        assert!(response.contains("streaming is not supported"));
    }
}
