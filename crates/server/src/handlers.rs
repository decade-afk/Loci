use crate::http_io::{bad_request, http_response, sse_event, sse_response, stream_fragments};
use crate::http_types::{
    ChatCompletionsRequest, OpenAiChatAssistantMessage, OpenAiChatCompletionChoice,
    OpenAiChatCompletionResponse, OpenAiCompletionChoice, OpenAiCompletionResponse,
    UpdatePlannerConfigRequest, UpdateRoutingConfigRequest, UpdateRuntimeControlRequest,
};
use crate::openai_compat::{
    chat_completion_stream_response, chat_request_from_payload, completion_stream_response,
};
use crate::runtime_control::ServerRuntimeControlState;
use loci_core::{InferenceEngine, LociError};
use loci_protocol::SessionRequest;
use serde::Serialize;
use std::time::{SystemTime, UNIX_EPOCH};

/// Runs one internal inference request and serializes the complete response as JSON.
pub(crate) fn respond_inference(engine: &mut InferenceEngine, request: SessionRequest) -> String {
    match engine.infer(request) {
        Ok(response) => serialize_response(&response),
        Err(error) => map_error(error),
    }
}

/// Emits the internal inference result as an SSE stream for clients that expect incremental deltas.
pub(crate) fn respond_inference_stream(
    engine: &mut InferenceEngine,
    request: SessionRequest,
) -> String {
    match engine.infer(request) {
        Ok(response) => {
            let mut events = Vec::new();
            for fragment in stream_fragments(&response.text) {
                events.push(sse_event(&serde_json::json!({
                    "type": "response.delta",
                    "delta": fragment,
                })));
            }
            events.push(sse_event(&serde_json::json!({
                "type": "response.completed",
                "response": response,
            })));
            events.push("data: [DONE]\n\n".to_string());
            sse_response(&events.concat())
        }
        Err(error) => map_error(error),
    }
}

/// Applies planner-facing runtime changes while keeping the mutable control snapshot aligned.
pub(crate) fn apply_planner_config(
    engine: &mut InferenceEngine,
    runtime_state: &mut ServerRuntimeControlState,
    payload: UpdatePlannerConfigRequest,
) {
    if let Some(keep_alive_secs) = payload.keep_alive_secs {
        engine.set_model_keep_alive_secs(keep_alive_secs);
        runtime_state.config.model_keep_alive_secs = keep_alive_secs;
    }
    if let Some(tiered_offload_enabled) = payload.tiered_offload_enabled {
        runtime_state.config.tiered_offload_enabled = tiered_offload_enabled;
    }
    if let Some(offload_profile) = payload.large_model_mode.or(payload.offload_profile) {
        engine.set_offload_profile(offload_profile);
        runtime_state.config.large_model_mode = offload_profile;
    }
    if let Some(spill_threshold_bytes) = payload.spill_threshold_bytes {
        engine.set_spill_threshold_bytes(Some(spill_threshold_bytes));
        runtime_state.config.spill_threshold_bytes = Some(spill_threshold_bytes);
    }
    if let Some(max_disk_bytes) = payload.max_disk_bytes {
        engine.set_max_disk_bytes(Some(max_disk_bytes));
        runtime_state.config.max_disk_bytes = Some(max_disk_bytes);
    }
    if let Some(prefetch_window_bytes) = payload.prefetch_window_bytes {
        engine.set_prefetch_window_bytes(Some(prefetch_window_bytes));
        runtime_state.config.prefetch_window_bytes = Some(prefetch_window_bytes);
    }
    if let Some(block_size_tokens) = payload.kv_block_size_tokens {
        engine.set_kv_block_size_tokens(block_size_tokens);
        runtime_state.config.kv_block_size_tokens = block_size_tokens;
    }
    if let Some(prefix_cache_enabled) = payload.kv_prefix_cache_enabled {
        engine.set_kv_prefix_cache_enabled(prefix_cache_enabled);
        runtime_state.config.kv_prefix_cache_enabled = prefix_cache_enabled;
    }
    match (payload.kv_type_k, payload.kv_type_v) {
        (Some(type_k), Some(type_v)) => {
            engine.set_kv_types(type_k.clone(), type_v.clone());
            runtime_state.config.kv_type_k = type_k;
            runtime_state.config.kv_type_v = type_v;
        }
        (Some(type_k), None) => {
            let existing = engine.runtime_snapshot().config.kv_type_v;
            engine.set_kv_types(type_k.clone(), existing.clone());
            runtime_state.config.kv_type_k = type_k;
            runtime_state.config.kv_type_v = existing;
        }
        (None, Some(type_v)) => {
            let existing = engine.runtime_snapshot().config.kv_type_k;
            engine.set_kv_types(existing.clone(), type_v.clone());
            runtime_state.config.kv_type_k = existing;
            runtime_state.config.kv_type_v = type_v;
        }
        (None, None) => {}
    }
}

/// Applies routing changes and returns backend feature errors unchanged to the HTTP layer.
pub(crate) fn apply_routing_config(
    engine: &mut InferenceEngine,
    runtime_state: &mut ServerRuntimeControlState,
    payload: UpdateRoutingConfigRequest,
) -> Result<(), LociError> {
    if let Some(enabled) = payload.enabled {
        engine.set_routing_enabled(enabled)?;
        runtime_state.config.routing.enabled = enabled;
    }
    if let Some(strategy) = payload.strategy {
        engine.set_routing_strategy(strategy.clone())?;
        runtime_state.config.routing.strategy = strategy;
    }
    if let Some(max_loaded_models) = payload.max_loaded_models {
        engine.set_max_loaded_models(max_loaded_models);
        runtime_state.config.routing.max_loaded_models = max_loaded_models;
    }
    Ok(())
}

/// Applies the full runtime-control surface, including planner and routing knobs, in one request.
pub(crate) fn apply_runtime_control(
    engine: &mut InferenceEngine,
    runtime_state: &mut ServerRuntimeControlState,
    payload: UpdateRuntimeControlRequest,
) -> Result<(), LociError> {
    if let Some(keep_alive_secs) = payload.keep_alive_secs {
        engine.set_model_keep_alive_secs(keep_alive_secs);
        runtime_state.config.model_keep_alive_secs = keep_alive_secs;
    }
    if let Some(tiered_offload_enabled) = payload.tiered_offload_enabled {
        runtime_state.config.tiered_offload_enabled = tiered_offload_enabled;
    }
    if let Some(offload_profile) = payload.large_model_mode.or(payload.offload_profile) {
        engine.set_offload_profile(offload_profile);
        runtime_state.config.large_model_mode = offload_profile;
    }
    if let Some(spill_threshold_bytes) = payload.spill_threshold_bytes {
        engine.set_spill_threshold_bytes(spill_threshold_bytes);
        runtime_state.config.spill_threshold_bytes = spill_threshold_bytes;
    }
    if let Some(max_disk_bytes) = payload.max_disk_bytes {
        engine.set_max_disk_bytes(max_disk_bytes);
        runtime_state.config.max_disk_bytes = max_disk_bytes;
    }
    if let Some(prefetch_window_bytes) = payload.prefetch_window_bytes {
        engine.set_prefetch_window_bytes(prefetch_window_bytes);
        runtime_state.config.prefetch_window_bytes = prefetch_window_bytes;
    }
    if let Some(block_size_tokens) = payload.kv_block_size_tokens {
        engine.set_kv_block_size_tokens(block_size_tokens);
        runtime_state.config.kv_block_size_tokens = block_size_tokens;
    }
    if let Some(prefix_cache_enabled) = payload.kv_prefix_cache_enabled {
        engine.set_kv_prefix_cache_enabled(prefix_cache_enabled);
        runtime_state.config.kv_prefix_cache_enabled = prefix_cache_enabled;
    }
    match (payload.kv_type_k, payload.kv_type_v) {
        (Some(type_k), Some(type_v)) => {
            engine.set_kv_types(type_k.clone(), type_v.clone());
            runtime_state.config.kv_type_k = type_k;
            runtime_state.config.kv_type_v = type_v;
        }
        (Some(type_k), None) => {
            let existing = runtime_state.config.kv_type_v.clone();
            engine.set_kv_types(type_k.clone(), existing.clone());
            runtime_state.config.kv_type_k = type_k;
        }
        (None, Some(type_v)) => {
            let existing = runtime_state.config.kv_type_k.clone();
            engine.set_kv_types(existing.clone(), type_v.clone());
            runtime_state.config.kv_type_v = type_v;
        }
        (None, None) => {}
    }
    if let Some(enabled) = payload.routing_enabled {
        engine.set_routing_enabled(enabled)?;
        runtime_state.config.routing.enabled = enabled;
    }
    if let Some(strategy) = payload.routing_strategy {
        engine.set_routing_strategy(strategy.clone())?;
        runtime_state.config.routing.strategy = strategy;
    }
    if let Some(max_loaded_models) = payload.max_loaded_models {
        engine.set_max_loaded_models(max_loaded_models);
        runtime_state.config.routing.max_loaded_models = max_loaded_models;
    }
    Ok(())
}

/// Handles the OpenAI-compatible completions endpoint on top of the internal inference engine.
pub(crate) fn handle_completion_request(
    engine: &mut InferenceEngine,
    payload: ChatOrCompletionRequest,
) -> String {
    let ChatOrCompletionRequest::Completion(payload) = payload else {
        unreachable!("completion handler received non-completion payload")
    };

    let request = SessionRequest {
        prompt: payload.prompt,
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        target_model: payload.model,
        images: Vec::new(),
        structured_output: false,
        tool_calling: false,
    };
    match engine.infer(request) {
        Ok(response) => {
            let response_id = format!("cmpl-{}", unix_timestamp());
            let created = unix_timestamp();
            if payload.stream {
                completion_stream_response(&response_id, created, &response.model, &response.text)
            } else {
                serialize_response(&OpenAiCompletionResponse {
                    id: response_id,
                    object: "text_completion",
                    created,
                    model: response.model,
                    choices: vec![OpenAiCompletionChoice {
                        text: response.text,
                        index: 0,
                        finish_reason: "stop",
                    }],
                })
            }
        }
        Err(error) => map_error(error),
    }
}

/// Handles the OpenAI-compatible chat completions endpoint on top of the internal inference engine.
pub(crate) fn handle_chat_completion_request(
    engine: &mut InferenceEngine,
    payload: ChatOrCompletionRequest,
) -> String {
    let ChatOrCompletionRequest::Chat(payload) = payload else {
        unreachable!("chat handler received non-chat payload")
    };

    let stream = payload.stream;
    match chat_request_from_payload(payload) {
        Ok(request) => match engine.infer(request) {
            Ok(response) => {
                let response_id = format!("chatcmpl-{}", unix_timestamp());
                let created = unix_timestamp();
                if stream {
                    chat_completion_stream_response(
                        &response_id,
                        created,
                        &response.model,
                        &response.text,
                    )
                } else {
                    serialize_response(&OpenAiChatCompletionResponse {
                        id: response_id,
                        object: "chat.completion",
                        created,
                        model: response.model,
                        choices: vec![OpenAiChatCompletionChoice {
                            index: 0,
                            message: OpenAiChatAssistantMessage {
                                role: "assistant",
                                content: response.text,
                            },
                            finish_reason: "stop",
                        }],
                    })
                }
            }
            Err(error) => map_error(error),
        },
        Err(error) => bad_request(&error),
    }
}

/// Serializes a value into the server's default successful JSON response envelope.
pub(crate) fn serialize_response<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(body) => http_response("200 OK", &body),
        Err(error) => bad_request(&format!("serialization failed: {error}")),
    }
}

/// Maps internal engine errors onto the existing HTTP error contract without changing status semantics.
pub(crate) fn map_error(error: LociError) -> String {
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

/// Generates stable response ids and timestamps using epoch seconds to match the current API shape.
pub(crate) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) enum ChatOrCompletionRequest {
    Chat(ChatCompletionsRequest),
    Completion(crate::http_types::CompletionRequest),
}
