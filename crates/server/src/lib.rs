mod handlers;
mod http_io;
mod http_types;
mod openai_compat;
mod runtime_control;
#[cfg(test)]
mod tests;

use anyhow::Context;
use loci_core::InferenceEngine;
use loci_protocol::TieredOffloadConfig;
use std::io::Write;
use std::net::TcpListener;

use handlers::*;
use http_io::*;
use http_types::*;
use runtime_control::{runtime_control_snapshot, ServerRuntimeControlState};
pub use runtime_control::{RuntimeControlConfig, RuntimeControlSnapshot, RuntimeRoutingConfig};

pub struct ServerConfig {
    pub bind: String,
    pub engine: InferenceEngine,
}

pub fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let snapshot = config.engine.runtime_snapshot();
    let runtime_control = RuntimeControlConfig::from_engine_snapshot(
        &snapshot,
        TieredOffloadConfig::default().prefetch_window_bytes,
    );
    run_server_with_runtime_control(config, runtime_control)
}

pub fn run_server_with_runtime_control(
    config: ServerConfig,
    runtime_control: RuntimeControlConfig,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&config.bind)
        .with_context(|| format!("failed to bind server on {}", config.bind))?;
    let mut engine = config.engine;
    let mut runtime_state = ServerRuntimeControlState {
        config: runtime_control,
    };

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

        let response =
            handle_request_with_runtime_control(&mut engine, &mut runtime_state, &request);
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.flush();
    }

    Ok(())
}

#[cfg(test)]
fn handle_request(engine: &mut InferenceEngine, request: &str) -> String {
    let snapshot = engine.runtime_snapshot();
    let mut runtime_state = ServerRuntimeControlState {
        config: RuntimeControlConfig::from_engine_snapshot(
            &snapshot,
            TieredOffloadConfig::default().prefetch_window_bytes,
        ),
    };
    handle_request_with_runtime_control(engine, &mut runtime_state, request)
}

fn handle_request_with_runtime_control(
    engine: &mut InferenceEngine,
    runtime_state: &mut ServerRuntimeControlState,
    request: &str,
) -> String {
    // This router works on a fully buffered request so the transport/framing layer can stay isolated.
    engine.evict_expired_models();
    let (request_line, body) = parse_request_parts(request);

    if request_line.starts_with("GET /health ") {
        return json_response(r#"{"status":"ok"}"#);
    }
    if request_line.starts_with("GET /v1/runtime ") {
        return serialize_response(&engine.runtime_snapshot());
    }
    if request_line.starts_with("GET /v1/runtime/control ") {
        return serialize_response(&runtime_control_snapshot(engine, runtime_state));
    }
    if request_line.starts_with("GET /v1/config ") {
        return serialize_response(&engine.runtime_snapshot().config);
    }
    if request_line.starts_with("GET /v1/config/runtime ") {
        return serialize_response(&runtime_state.config);
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
    if request_line.starts_with("GET /v1/models/inspect ") {
        return serialize_response(&engine.inspect_models());
    }
    if request_line.starts_with("POST /v1/models/inspect ") {
        return match serde_json::from_str::<InspectModelRequest>(body) {
            Ok(payload) => match engine.inspect_model(&payload.name) {
                Ok(report) => serialize_response(&report),
                Err(error) => map_error(error),
            },
            Err(error) => bad_request(&format!("invalid model inspection payload: {error}")),
        };
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
                apply_planner_config(engine, runtime_state, payload);
                serialize_response(&runtime_state.config)
            }
            Err(error) => bad_request(&format!("invalid planner config payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/config/routing ") {
        return match serde_json::from_str::<UpdateRoutingConfigRequest>(body) {
            Ok(payload) => match apply_routing_config(engine, runtime_state, payload) {
                Ok(()) => serialize_response(&runtime_state.config.routing),
                Err(error) => map_error(error),
            },
            Err(error) => bad_request(&format!("invalid routing config payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/config/runtime ") {
        return match serde_json::from_str::<UpdateRuntimeControlRequest>(body) {
            Ok(payload) => match apply_runtime_control(engine, runtime_state, payload) {
                Ok(()) => serialize_response(&runtime_control_snapshot(engine, runtime_state)),
                Err(error) => map_error(error),
            },
            Err(error) => bad_request(&format!("invalid runtime config payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/plan ") {
        return match serde_json::from_str::<PlanHttpRequest>(body) {
            Ok(payload) => match payload.into_request() {
                Ok(request) => match engine.plan(&request) {
                    Ok(plan) => serialize_response(&plan),
                    Err(error) => map_error(error),
                },
                Err(error) => bad_request(&error),
            },
            Err(error) => bad_request(&format!("invalid plan payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/inference ") {
        return match serde_json::from_str::<InferenceHttpRequest>(body) {
            Ok(payload) => {
                let stream = payload.stream;
                match payload.into_request() {
                    Ok(request) => {
                        if stream {
                            respond_inference_stream(engine, request)
                        } else {
                            respond_inference(engine, request)
                        }
                    }
                    Err(error) => bad_request(&error),
                }
            }
            Err(error) => bad_request(&format!("invalid inference payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/inference/stream ") {
        return match serde_json::from_str::<InferenceHttpRequest>(body) {
            Ok(payload) => match payload.into_request() {
                Ok(request) => respond_inference_stream(engine, request),
                Err(error) => bad_request(&error),
            },
            Err(error) => bad_request(&format!("invalid inference payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/completions ") {
        return match serde_json::from_str::<CompletionRequest>(body) {
            Ok(payload) => {
                handle_completion_request(engine, ChatOrCompletionRequest::Completion(payload))
            }
            Err(error) => bad_request(&format!("invalid completion payload: {error}")),
        };
    }
    if request_line.starts_with("POST /v1/chat/completions ") {
        return match serde_json::from_str::<ChatCompletionsRequest>(body) {
            Ok(payload) => {
                handle_chat_completion_request(engine, ChatOrCompletionRequest::Chat(payload))
            }
            Err(error) => bad_request(&format!("invalid chat completion payload: {error}")),
        };
    }

    not_found("route not found")
}
