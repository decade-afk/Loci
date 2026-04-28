use anyhow::Context;
use loci_core::{InferenceEngine, LociError};
use loci_protocol::{ModelDescriptor, SessionRequest, TieredOffloadProfile};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ServerConfig {
    pub bind: String,
    pub engine: InferenceEngine,
}

#[derive(Debug, Deserialize)]
struct RegisterModelRequest {
    name: String,
    path: PathBuf,
    #[serde(default = "default_architecture")]
    architecture: String,
    #[serde(default)]
    memory_bytes: Option<u64>,
    #[serde(default)]
    parameter_count: Option<u64>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    preferred_backend: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UnregisterModelRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
struct EvictModelRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
struct PrewarmModelRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_prewarm_prompt")]
    prompt: String,
    #[serde(default = "default_prewarm_tokens")]
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct RegisterAliasRequest {
    alias: String,
    target: String,
}

#[derive(Debug, Deserialize)]
struct RemoveAliasRequest {
    alias: String,
}

#[derive(Debug, Deserialize)]
struct UpdatePlannerConfigRequest {
    #[serde(default)]
    keep_alive_secs: Option<u64>,
    #[serde(default)]
    offload_profile: Option<TieredOffloadProfile>,
    #[serde(default)]
    kv_block_size_tokens: Option<u32>,
    #[serde(default)]
    kv_prefix_cache_enabled: Option<bool>,
    #[serde(default)]
    kv_type_k: Option<String>,
    #[serde(default)]
    kv_type_v: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceHttpRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    target_model: Option<String>,
    #[serde(default)]
    structured_output: bool,
    #[serde(default)]
    tool_calling: bool,
}

#[derive(Debug, Deserialize)]
struct PlanHttpRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    target_model: Option<String>,
    #[serde(default)]
    structured_output: bool,
    #[serde(default)]
    tool_calling: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
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

pub fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&config.bind)
        .with_context(|| format!("failed to bind server on {}", config.bind))?;
    let mut engine = config.engine;

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("loci-server: accept failed: {error}");
                continue;
            }
        };

        let request = match read_request(&mut stream) {
            Ok(request) => request,
            Err(response) => {
                let _ = stream.write_all(response.as_bytes());
                continue;
            }
        };

        let response = handle_request(&mut engine, &request);
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    }

    Ok(())
}

fn handle_request(engine: &mut InferenceEngine, request: &str) -> String {
    engine.evict_expired_models();
    let (request_line, body) = parse_request_parts(request);

    if request_line.starts_with("GET /health ") {
        return json_response(r#"{"status":"ok"}"#);
    }
    if request_line.starts_with("GET /v1/runtime ") {
        return serialize_response(&engine.runtime_snapshot());
    }
    if request_line.starts_with("GET /v1/config ") {
        return serialize_response(&engine.runtime_snapshot().config);
    }
    if request_line.starts_with("GET /v1/models ") {
        return serialize_response(&OpenAiModelListResponse {
            object: "list",
            data: engine
                .models()
                .into_iter()
                .map(|model| OpenAiModel {
                    id: model.name.clone(),
                    object: "model",
                    created: unix_timestamp(),
                    owned_by: "loci",
                })
                .collect(),
        });
    }
    if request_line.starts_with("POST /v1/models/register ") {
        return match serde_json::from_str::<RegisterModelRequest>(body) {
            Ok(payload) => {
                engine.register_model(payload.into_model());
                serialize_response(&engine.runtime_snapshot())
            }
            Err(error) => bad_request(&format!("invalid model registration payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/models/unregister ") {
        return match serde_json::from_str::<UnregisterModelRequest>(body) {
            Ok(payload) => serialize_response(&serde_json::json!({
                "removed": engine.unregister_model(&payload.name),
                "name": payload.name,
            })),
            Err(error) => bad_request(&format!("invalid model removal payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/models/evict ") {
        return match serde_json::from_str::<EvictModelRequest>(body) {
            Ok(payload) => serialize_response(&serde_json::json!({
                "evicted": engine.evict_model(&payload.name),
                "name": payload.name,
            })),
            Err(error) => bad_request(&format!("invalid model eviction payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/models/prewarm ") {
        return match serde_json::from_str::<PrewarmModelRequest>(body) {
            Ok(payload) => match engine.prepare(payload.into_request()) {
                Ok(prepared) => serialize_response(&prepared),
                Err(error) => map_error(error),
            },
            Err(error) => bad_request(&format!("invalid model prewarm payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/config/aliases/register ") {
        return match serde_json::from_str::<RegisterAliasRequest>(body) {
            Ok(payload) => {
                engine.register_alias(payload.alias, payload.target);
                serialize_response(&engine.runtime_snapshot().config)
            }
            Err(error) => bad_request(&format!("invalid alias registration payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/config/aliases/remove ") {
        return match serde_json::from_str::<RemoveAliasRequest>(body) {
            Ok(payload) => serialize_response(&serde_json::json!({
                "removed": engine.remove_alias(&payload.alias),
                "alias": payload.alias,
                "config": engine.runtime_snapshot().config,
            })),
            Err(error) => bad_request(&format!("invalid alias removal payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/config/planner ") {
        return match serde_json::from_str::<UpdatePlannerConfigRequest>(body) {
            Ok(payload) => {
                apply_planner_config(engine, payload);
                serialize_response(&engine.runtime_snapshot().config)
            }
            Err(error) => bad_request(&format!("invalid planner config payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/plan ") {
        return match serde_json::from_str::<PlanHttpRequest>(body) {
            Ok(payload) => match engine.plan(&payload.into_request()) {
                Ok(plan) => serialize_response(&plan),
                Err(error) => map_error(error),
            },
            Err(error) => bad_request(&format!("invalid plan payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/inference ") {
        return match serde_json::from_str::<InferenceHttpRequest>(body) {
            Ok(payload) => respond_inference(engine, payload.into_request()),
            Err(error) => bad_request(&format!("invalid inference payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/completions ") {
        return match serde_json::from_str::<CompletionRequest>(body) {
            Ok(payload) => {
                let request = SessionRequest {
                    prompt: payload.prompt,
                    max_tokens: payload.max_tokens,
                    temperature: payload.temperature,
                    target_model: payload.model,
                    structured_output: false,
                    tool_calling: false,
                };
                match engine.infer(request) {
                    Ok(response) => serialize_response(&OpenAiCompletionResponse {
                        id: format!("cmpl-{}", unix_timestamp()),
                        object: "text_completion",
                        created: unix_timestamp(),
                        model: response.model,
                        choices: vec![OpenAiCompletionChoice {
                            text: response.text,
                            index: 0,
                            finish_reason: "stop",
                        }],
                    }),
                    Err(error) => map_error(error),
                }
            }
            Err(error) => bad_request(&format!("invalid completion payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/chat/completions ") {
        return match serde_json::from_str::<ChatCompletionsRequest>(body) {
            Ok(payload) => {
                let request = SessionRequest {
                    prompt: flatten_messages(&payload.messages),
                    max_tokens: payload.max_tokens,
                    temperature: payload.temperature,
                    target_model: payload.model,
                    structured_output: false,
                    tool_calling: false,
                };
                match engine.infer(request) {
                    Ok(response) => serialize_response(&OpenAiChatCompletionResponse {
                        id: format!("chatcmpl-{}", unix_timestamp()),
                        object: "chat.completion",
                        created: unix_timestamp(),
                        model: response.model,
                        choices: vec![OpenAiChatCompletionChoice {
                            index: 0,
                            message: OpenAiChatAssistantMessage {
                                role: "assistant",
                                content: response.text,
                            },
                            finish_reason: "stop",
                        }],
                    }),
                    Err(error) => map_error(error),
                }
            }
            Err(error) => bad_request(&format!("invalid chat completion payload: {error}")),
        };
    }

    not_found("route not found")
}

fn respond_inference(engine: &mut InferenceEngine, request: SessionRequest) -> String {
    match engine.infer(request) {
        Ok(response) => serialize_response(&response),
        Err(error) => map_error(error),
    }
}

fn apply_planner_config(engine: &mut InferenceEngine, payload: UpdatePlannerConfigRequest) {
    if let Some(keep_alive_secs) = payload.keep_alive_secs {
        engine.set_model_keep_alive_secs(keep_alive_secs);
    }
    if let Some(offload_profile) = payload.offload_profile {
        engine.set_offload_profile(offload_profile);
    }
    if let Some(block_size_tokens) = payload.kv_block_size_tokens {
        engine.set_kv_block_size_tokens(block_size_tokens);
    }
    if let Some(prefix_cache_enabled) = payload.kv_prefix_cache_enabled {
        engine.set_kv_prefix_cache_enabled(prefix_cache_enabled);
    }
    match (payload.kv_type_k, payload.kv_type_v) {
        (Some(type_k), Some(type_v)) => engine.set_kv_types(type_k, type_v),
        (Some(type_k), None) => {
            let existing = engine.runtime_snapshot().config.kv_type_v;
            engine.set_kv_types(type_k, existing);
        }
        (None, Some(type_v)) => {
            let existing = engine.runtime_snapshot().config.kv_type_k;
            engine.set_kv_types(existing, type_v);
        }
        (None, None) => {}
    }
}

fn serialize_response<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(body) => http_response("200 OK", &body),
        Err(error) => bad_request(&format!("serialization failed: {error}")),
    }
}

fn map_error(error: LociError) -> String {
    match error {
        LociError::NoBackendAvailable => bad_request("no backend available"),
        LociError::NoModelsRegistered => bad_request("no model registered"),
        LociError::RequestedModelMissing(name) => {
            bad_request(&format!("requested model `{name}` is not registered"))
        }
        LociError::NoCompatibleBackend { model, format } => bad_request(&format!(
            "no compatible backend is available for model `{model}` with format `{format}`"
        )),
        LociError::Backend(message) | LociError::InvalidRequest(message) => bad_request(&message),
    }
}

fn flatten_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        prompt.push_str(&message.content);
        prompt.push_str("\n\n");
    }
    prompt.push_str("assistant:");
    prompt
}

fn json_response(body: &str) -> String {
    http_response("200 OK", body)
}

fn bad_request(message: &str) -> String {
    http_response("400 Bad Request", &format!(r#"{{"error":"{}"}}"#, message))
}

fn not_found(message: &str) -> String {
    http_response("404 Not Found", &format!(r#"{{"error":"{}"}}"#, message))
}

fn http_response(status: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn parse_request_parts(request: &str) -> (&str, &str) {
    let (head, body) = request.split_once("\r\n\r\n").unwrap_or((request, ""));
    let request_line = head.lines().next().unwrap_or_default();
    (request_line, body)
}

fn read_request(stream: &mut (impl Read + Write)) -> std::result::Result<String, String> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut sent_continue = false;
    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| bad_request(&format!("read failed: {error}")))?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if !sent_continue {
            if let Some(framing) = request_framing(&buffer) {
                if framing.expect_continue {
                    stream
                        .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                        .map_err(|error| {
                            bad_request(&format!("failed to acknowledge request body: {error}"))
                        })?;
                    stream
                        .flush()
                        .map_err(|error| bad_request(&format!("flush failed: {error}")))?;
                    sent_continue = true;
                }
            }
        }
        if let Some(required_len) = required_request_len(&buffer) {
            if buffer.len() >= required_len {
                break;
            }
        }
    }

    let normalized = normalize_request_body(buffer)?;
    String::from_utf8(normalized).map_err(|_| bad_request("request is not valid utf-8"))
}

fn required_request_len(buffer: &[u8]) -> Option<usize> {
    let framing = request_framing(buffer)?;
    match framing.body {
        BodyFraming::ContentLength(content_length) => Some(framing.header_end + content_length),
        BodyFraming::Chunked => {
            let body = &buffer[framing.header_end..];
            body.windows(5)
                .position(|window| window == b"0\r\n\r\n")
                .map(|offset| framing.header_end + offset + 5)
        }
        BodyFraming::Empty => Some(framing.header_end),
    }
}

fn normalize_request_body(buffer: Vec<u8>) -> std::result::Result<Vec<u8>, String> {
    let Some(framing) = request_framing(&buffer) else {
        return Ok(buffer);
    };

    match framing.body {
        BodyFraming::Chunked => {
            let decoded = decode_chunked_body(&buffer[framing.header_end..])?;
            let mut normalized = buffer[..framing.header_end].to_vec();
            normalized.extend_from_slice(&decoded);
            Ok(normalized)
        }
        _ => Ok(buffer),
    }
}

fn request_framing(buffer: &[u8]) -> Option<RequestFraming> {
    let header_end = buffer.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let headers = std::str::from_utf8(&buffer[..header_end]).ok()?;
    let mut content_length = None;
    let mut chunked = false;
    let mut expect_continue = false;

    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().ok();
        }
        if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
        if name.eq_ignore_ascii_case("expect")
            && value.to_ascii_lowercase().contains("100-continue")
        {
            expect_continue = true;
        }
    }

    let body = if chunked {
        BodyFraming::Chunked
    } else if let Some(content_length) = content_length {
        BodyFraming::ContentLength(content_length)
    } else {
        BodyFraming::Empty
    };

    Some(RequestFraming {
        header_end,
        body,
        expect_continue,
    })
}

fn decode_chunked_body(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut offset = 0usize;
    let mut decoded = Vec::new();

    loop {
        let size_line_end = find_bytes(&body[offset..], b"\r\n")
            .ok_or_else(|| bad_request("malformed chunked request body"))?;
        let size_line = std::str::from_utf8(&body[offset..offset + size_line_end])
            .map_err(|_| bad_request("chunk size is not valid utf-8"))?;
        let size_token = size_line.split(';').next().unwrap_or_default().trim();
        let chunk_size = usize::from_str_radix(size_token, 16)
            .map_err(|_| bad_request("chunk size is not valid hexadecimal"))?;
        offset += size_line_end + 2;

        if chunk_size == 0 {
            return Ok(decoded);
        }

        if body.len() < offset + chunk_size + 2 {
            return Err(bad_request("chunked request body ended unexpectedly"));
        }

        decoded.extend_from_slice(&body[offset..offset + chunk_size]);
        offset += chunk_size;

        if &body[offset..offset + 2] != b"\r\n" {
            return Err(bad_request("chunk delimiter is missing"));
        }
        offset += 2;
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[derive(Clone, Copy)]
struct RequestFraming {
    header_end: usize,
    body: BodyFraming,
    expect_continue: bool,
}

#[derive(Clone, Copy)]
enum BodyFraming {
    Empty,
    ContentLength(usize),
    Chunked,
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

impl RegisterModelRequest {
    fn into_model(self) -> ModelDescriptor {
        ModelDescriptor {
            name: self.name,
            path: self.path,
            architecture: self.architecture,
            memory_bytes: self.memory_bytes,
            parameter_count: self.parameter_count,
            context_length: self.context_length,
            preferred_backend: self.preferred_backend,
        }
    }
}

impl InferenceHttpRequest {
    fn into_request(self) -> SessionRequest {
        SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            target_model: self.target_model,
            structured_output: self.structured_output,
            tool_calling: self.tool_calling,
        }
    }
}

impl PlanHttpRequest {
    fn into_request(self) -> SessionRequest {
        SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            target_model: self.target_model,
            structured_output: self.structured_output,
            tool_calling: self.tool_calling,
        }
    }
}

const fn default_max_tokens() -> u32 {
    128
}

const fn default_temperature() -> f32 {
    0.2
}

fn default_architecture() -> String {
    "llama".to_string()
}

fn default_prewarm_prompt() -> String {
    "warmup".to_string()
}

const fn default_prewarm_tokens() -> u32 {
    1
}

impl PrewarmModelRequest {
    fn into_request(self) -> SessionRequest {
        SessionRequest {
            prompt: self.prompt,
            max_tokens: self.max_tokens,
            temperature: 0.0,
            target_model: self.model,
            structured_output: false,
            tool_calling: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_core::{EngineConfig, InferenceEngine};

    fn engine_with_model() -> InferenceEngine {
        InferenceEngine::builder()
            .model(ModelDescriptor {
                name: "demo".to_string(),
                path: PathBuf::from("D:/models/demo.gguf"),
                architecture: "llama".to_string(),
                memory_bytes: Some(2 * 1024 * 1024 * 1024),
                parameter_count: Some(1_000_000_000),
                context_length: Some(8192),
                preferred_backend: None,
            })
            .build()
            .expect("engine")
    }

    #[test]
    fn plan_route_returns_execution_plan() {
        let mut engine = engine_with_model();
        let response = handle_request(
            &mut engine,
            "POST /v1/plan HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"prompt\":\"hello\",\"max_tokens\":32,\"target_model\":\"demo\"}",
        );

        assert!(response.contains("\"backend\":\"openvino\""));
        assert!(response.contains("\"selected_model\":\"demo\""));
        assert!(response.contains("\"placements\""));
    }

    #[test]
    fn unregister_route_removes_model() {
        let mut engine = engine_with_model();
        let response = handle_request(
            &mut engine,
            "POST /v1/models/unregister HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"name\":\"demo\"}",
        );

        assert!(response.contains("\"removed\":true"));
        assert!(engine.models().is_empty());
    }

    #[test]
    fn evict_route_drops_prepared_state_without_unregistering_model() {
        let mut engine = engine_with_model();
        let _ = handle_request(
            &mut engine,
            "POST /v1/models/prewarm HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo\"}",
        );

        let response = handle_request(
            &mut engine,
            "POST /v1/models/evict HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"name\":\"demo\"}",
        );

        assert!(response.contains("\"evicted\":true"));
        assert_eq!(engine.models().len(), 1);
    }

    #[test]
    fn prewarm_route_prepares_model_session() {
        let mut engine = engine_with_model();
        let response = handle_request(
            &mut engine,
            "POST /v1/models/prewarm HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo\"}",
        );

        assert!(response.contains("\"model_name\":\"demo\""));
        assert!(response.contains("\"backend\":\"openvino\""));
    }

    #[test]
    fn config_routes_update_aliases_and_planner_settings() {
        let mut engine = InferenceEngine::builder()
            .config(EngineConfig::default())
            .model(ModelDescriptor {
                name: "demo".to_string(),
                path: PathBuf::from("D:/models/demo.gguf"),
                architecture: "llama".to_string(),
                memory_bytes: Some(2 * 1024 * 1024 * 1024),
                parameter_count: Some(1_000_000_000),
                context_length: Some(8192),
                preferred_backend: None,
            })
            .build()
            .expect("engine");

        let alias_response = handle_request(
            &mut engine,
            "POST /v1/config/aliases/register HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"alias\":\"tiny\",\"target\":\"demo\"}",
        );
        assert!(alias_response.contains("\"tiny\":\"demo\""));

        let planner_response = handle_request(
            &mut engine,
            "POST /v1/config/planner HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"keep_alive_secs\":12,\"offload_profile\":\"gpu_resident\",\"kv_block_size_tokens\":64,\"kv_prefix_cache_enabled\":false,\"kv_type_k\":\"q8_0\",\"kv_type_v\":\"q4_0\"}",
        );
        assert!(planner_response.contains("\"model_keep_alive_secs\":12"));
        assert!(planner_response.contains("\"tiered_offload_profile\":\"gpu_resident\""));
        assert!(planner_response.contains("\"kv_block_size_tokens\":64"));
        assert!(planner_response.contains("\"kv_type_k\":\"q8_0\""));
        assert!(planner_response.contains("\"kv_type_v\":\"q4_0\""));

        let remove_response = handle_request(
            &mut engine,
            "POST /v1/config/aliases/remove HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"alias\":\"tiny\"}",
        );
        assert!(remove_response.contains("\"removed\":true"));
    }

    #[test]
    fn read_request_honors_content_length_for_post_bodies() {
        let request = b"POST /v1/config/aliases/register HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"alias\":\"tiny\",\"target\":\"demo\"}";
        let mut cursor = std::io::Cursor::new(request.to_vec());
        let parsed = read_request(&mut cursor).expect("request");

        assert!(parsed.ends_with("{\"alias\":\"tiny\",\"target\":\"demo\"}"));
    }

    #[test]
    fn read_request_decodes_chunked_post_bodies() {
        let request = b"POST /v1/config/aliases/register HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n20\r\n{\"alias\":\"tiny\",\"target\":\"demo\"}\r\n0\r\n\r\n";
        let mut cursor = std::io::Cursor::new(request.to_vec());
        let parsed = read_request(&mut cursor).expect("request");

        assert!(parsed.ends_with("{\"alias\":\"tiny\",\"target\":\"demo\"}"));
    }

    #[test]
    fn read_request_acknowledges_expect_continue() {
        let request = b"POST /v1/config/aliases/register HTTP/1.1\r\nExpect: 100-continue\r\nContent-Type: application/json\r\nContent-Length: 32\r\n\r\n{\"alias\":\"tiny\",\"target\":\"demo\"}";
        let mut cursor = std::io::Cursor::new(request.to_vec());
        let parsed = read_request(&mut cursor).expect("request");

        assert!(parsed.ends_with("{\"alias\":\"tiny\",\"target\":\"demo\"}"));
        let written = String::from_utf8(cursor.into_inner()).expect("buffer");
        assert!(written.contains("100 Continue"));
    }
}
