#![cfg(test)]

use super::*;
use crate::openai_compat::chat_request_from_payload;
use loci_core::{EngineConfig, InferenceEngine};
use loci_protocol::{ImageInput, ModelDescriptor};
use std::path::PathBuf;

#[cfg(feature = "gguf")]
use std::fs;
#[cfg(feature = "gguf")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "gguf")]
const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");

#[cfg(feature = "gguf")]
fn unique_temp_path(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("loci-server-{label}-{suffix}.gguf"))
}

#[cfg(feature = "gguf")]
fn write_minimal_gguf(path: &PathBuf) {
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
fn test_model_path(name: &str) -> PathBuf {
    let path = unique_temp_path(name);
    write_minimal_gguf(&path);
    path
}

#[cfg(not(feature = "gguf"))]
fn test_model_path(name: &str) -> PathBuf {
    PathBuf::from(format!("D:/models/{name}.gguf"))
}

fn engine_with_model() -> InferenceEngine {
    InferenceEngine::builder()
        .model(ModelDescriptor {
            name: "demo".to_string(),
            path: test_model_path("demo"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        })
        .build()
        .expect("engine")
}

#[cfg(feature = "gguf")]
#[test]
fn plan_route_returns_execution_plan() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/plan HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"prompt\":\"hello\",\"max_tokens\":32,\"target_model\":\"demo\"}",
    );

    assert!(response.contains(if cfg!(feature = "openvino") {
        "\"backend\":\"openvino\""
    } else {
        "\"backend\":\"candle\""
    }));
    assert!(response.contains("\"selected_model\":\"demo\""));
    assert!(response.contains("\"placements\""));
}

#[cfg(feature = "gguf")]
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

#[cfg(feature = "gguf")]
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

#[cfg(feature = "gguf")]
#[test]
fn prewarm_route_prepares_model_session() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/models/prewarm HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo\"}",
    );

    assert!(response.contains("\"model_name\":\"demo\""));
    assert!(response.contains(if cfg!(feature = "openvino") {
        "\"backend\":\"openvino\""
    } else {
        "\"backend\":\"candle\""
    }));
}

#[cfg(feature = "gguf")]
#[test]
fn inspect_routes_return_model_readiness_reports() {
    let mut engine = engine_with_model();

    let list_response = handle_request(&mut engine, "GET /v1/models/inspect HTTP/1.1\r\n\r\n");
    assert!(list_response.contains("\"model_name\":\"demo\""));
    assert!(list_response.contains("\"backend_readiness\""));

    let single_response = handle_request(
        &mut engine,
        "POST /v1/models/inspect HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"name\":\"demo\"}",
    );
    assert!(single_response.contains("\"model_name\":\"demo\""));
    assert!(single_response.contains("\"ready_for_inference\""));
}

#[cfg(feature = "gguf")]
#[test]
fn config_routes_update_aliases_and_planner_settings() {
    let mut engine = InferenceEngine::builder()
        .config(EngineConfig::default())
        .model(ModelDescriptor {
            name: "demo".to_string(),
            path: test_model_path("demo-config"),
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

#[cfg(feature = "gguf")]
#[test]
fn routing_config_route_updates_model_pool_limits() {
    let mut engine = InferenceEngine::builder()
        .config(EngineConfig::default())
        .model(ModelDescriptor {
            name: "demo".to_string(),
            path: test_model_path("demo-route-a"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        })
        .model(ModelDescriptor {
            name: "demo-2".to_string(),
            path: test_model_path("demo-route-b"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(2_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        })
        .build()
        .expect("engine");

    let routing_response = handle_request(
        &mut engine,
        "POST /v1/config/routing HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"max_loaded_models\":1}",
    );

    assert!(routing_response.contains("\"max_loaded_models\":1"));
    assert_eq!(
        engine.runtime_snapshot().model_pool.resident_models.len(),
        1
    );
}

#[cfg(feature = "gguf")]
#[test]
fn inference_stream_route_returns_sse_events() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/inference/stream HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"prompt\":\"hello stream\",\"max_tokens\":16,\"target_model\":\"demo\"}",
    );

    assert!(response.contains("Content-Type: text/event-stream"));
    assert!(response.contains("\"type\":\"response.delta\""));
    assert!(response.contains("\"type\":\"response.completed\""));
    assert!(response.contains("data: [DONE]"));
}

#[cfg(feature = "gguf")]
#[test]
fn completion_route_supports_sse_streaming() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo\",\"prompt\":\"hello completion stream\",\"stream\":true}",
    );

    assert!(response.contains("Content-Type: text/event-stream"));
    assert!(response.contains("\"object\":\"text_completion.chunk\""));
    assert!(response.contains("\"finish_reason\":\"stop\""));
    assert!(response.contains("data: [DONE]"));
}

#[cfg(feature = "gguf")]
#[test]
fn chat_completion_route_supports_sse_streaming() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"demo\",\"messages\":[{\"role\":\"user\",\"content\":\"hello chat stream\"}],\"stream\":true}",
    );

    assert!(response.contains("Content-Type: text/event-stream"));
    assert!(response.contains("\"object\":\"chat.completion.chunk\""));
    assert!(response.contains("\"role\":\"assistant\""));
    assert!(response.contains("data: [DONE]"));
}

#[test]
fn inference_request_collects_explicit_image_inputs() {
    let request = InferenceHttpRequest {
        prompt: "describe".to_string(),
        max_tokens: 32,
        temperature: 0.2,
        target_model: Some("demo".to_string()),
        images: vec![
            HttpImageInput {
                path: Some(PathBuf::from("D:/images/a.png")),
                url: None,
                data_base64: None,
                media_type: None,
            },
            HttpImageInput {
                path: None,
                url: Some("file:///D:/images/b.png".to_string()),
                data_base64: None,
                media_type: None,
            },
        ],
        structured_output: false,
        tool_calling: false,
        stream: false,
    }
    .into_request()
    .expect("request");

    assert_eq!(request.images.len(), 2);
    assert!(matches!(request.images[0], ImageInput::Path { .. }));
    assert!(matches!(request.images[1], ImageInput::Url { .. }));
}

#[test]
fn chat_request_extracts_images_from_openai_style_content_parts() {
    let request = chat_request_from_payload(ChatCompletionsRequest {
        model: Some("demo".to_string()),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: ChatMessageContent::Parts(vec![
                ChatContentPart::Text {
                    text: "describe".to_string(),
                },
                ChatContentPart::ImageUrl {
                    image_url: ChatImageReference::String(
                        "file:///D:/images/example.png".to_string(),
                    ),
                },
            ]),
        }],
        max_tokens: 48,
        temperature: 0.1,
        stream: false,
    })
    .expect("request");

    assert!(request.prompt.contains("user: describe <image>"));
    assert_eq!(request.images.len(), 1);
    assert!(matches!(request.images[0], ImageInput::Url { .. }));
}

#[cfg(feature = "dynamic-routing")]
#[test]
fn routing_config_route_updates_dynamic_routing_settings() {
    let mut engine = InferenceEngine::builder()
        .config(EngineConfig::default())
        .model(ModelDescriptor {
            name: "small".to_string(),
            path: test_model_path("small"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        })
        .model(ModelDescriptor {
            name: "large".to_string(),
            path: test_model_path("large"),
            architecture: "llama".to_string(),
            memory_bytes: Some(8 * 1024 * 1024 * 1024),
            parameter_count: Some(8_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        })
        .build()
        .expect("engine");

    let routing_response = handle_request(
        &mut engine,
        "POST /v1/config/routing HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"enabled\":true,\"strategy\":\"power_aware\",\"max_loaded_models\":2}",
    );

    assert!(routing_response.contains("\"enabled\":true"));
    assert!(routing_response.contains("\"strategy\":\"power_aware\""));
    assert!(engine.runtime_snapshot().features.dynamic_routing);
}

#[cfg(not(feature = "dynamic-routing"))]
#[test]
fn routing_config_route_rejects_enabling_dynamic_routing_without_feature() {
    let mut engine = engine_with_model();
    let response = handle_request(
        &mut engine,
        "POST /v1/config/routing HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"enabled\":true}",
    );

    assert!(response.contains("dynamic routing is unavailable"));
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
