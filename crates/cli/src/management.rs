use loci_core::{
    CommandExecutionRequest, CoreRewriterActivationRequest, ManagementService, ModelLoadRequest,
    PluginLoadRequest, TextGenerationRequest,
};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

pub fn run_management_server(bind_addr: &str, service: ManagementService) -> anyhow::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    println!("management API listening on http://{bind_addr}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("management accept error: {error}");
                continue;
            }
        };

        let response = match read_http_request(&mut stream) {
            Ok(request) => handle_management_request(&service, request),
            Err(error) => json_response(400, json!({ "error": error.to_string() })),
        };

        if let Err(error) = write_http_response(&mut stream, response) {
            eprintln!("management response error: {error}");
        }
    }

    Ok(())
}

pub fn handle_management_request(
    service: &ManagementService,
    request: HttpRequest,
) -> HttpResponse {
    let path = request_path(&request.path);

    if request.method == "GET" {
        if let Some(plugin_name) = plugin_name_from_route(path, "/v1/plugins/") {
            return match service.plugin_detail(plugin_name) {
                Ok(Some(detail)) => match serde_json::to_value(detail) {
                    Ok(detail) => json_response(200, detail),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Ok(None) => json_response(404, json!({ "error": "plugin not found" })),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            };
        }
    }

    match (request.method.as_str(), path) {
        ("GET", "/health") => json_response(
            200,
            serde_json::to_value(service.health())
                .unwrap_or_else(|_| json!({ "error": "serialization failure" })),
        ),
        ("GET", "/v1/runtime") => match service.runtime_snapshot() {
            Ok(snapshot) => match serde_json::to_value(snapshot) {
                Ok(snapshot) => json_response(200, snapshot),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/backends") => match service.backend_capabilities() {
            Ok(backends) => match serde_json::to_value(backends) {
                Ok(backends) => json_response(200, backends),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/workflows") => match service.workflow_inventory() {
            Ok(workflows) => match serde_json::to_value(workflows) {
                Ok(workflows) => json_response(200, workflows),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/ui") => match service.ui_inventory() {
            Ok(ui) => match serde_json::to_value(ui) {
                Ok(ui) => json_response(200, ui),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/commands") => match service.command_inventory() {
            Ok(commands) => match serde_json::to_value(commands) {
                Ok(commands) => json_response(200, commands),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/core/rewriters") => match service.configured_core_rewriters() {
            Ok(rewriters) => match serde_json::to_value(rewriters) {
                Ok(rewriters) => json_response(200, rewriters),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/core/rewriters/inventory") => match service.core_rewriter_inventory() {
            Ok(inventory) => match serde_json::to_value(inventory) {
                Ok(inventory) => json_response(200, inventory),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("GET", "/v1/plugins") => match service.plugin_statuses() {
            Ok(plugins) => match serde_json::to_value(plugins) {
                Ok(plugins) => json_response(200, plugins),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(error) => json_response(500, json!({ "error": error.to_string() })),
        },
        ("POST", "/v1/plugins/load") => {
            let load_request = match plugin_load_request_from_body(&request.body) {
                Ok(load_request) => load_request,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.load_plugins(load_request) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/model/load") => {
            let load_request = match model_load_request_from_body(&request.body) {
                Ok(load_request) => load_request,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.load_model(load_request) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/inference/generate") => {
            let generation_request = match text_generation_request_from_body(&request.body) {
                Ok(generation_request) => generation_request,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.generate_text(generation_request) {
                Ok(response) => match serde_json::to_value(response) {
                    Ok(response) => json_response(200, response),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/commands/run") => {
            let command_request = match command_execution_request_from_body(&request.body) {
                Ok(command_request) => command_request,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.run_command(command_request) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/core/rewriters/activate") => {
            let activation_request = match core_rewriter_activation_request_from_body(&request.body)
            {
                Ok(activation_request) => activation_request,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.activate_core_rewriter(activation_request) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/core/inference/activate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.activate_inference_plugin(&plugin_name) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/legacy-text/activate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.activate_legacy_text_plugin(&plugin_name) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", "/v1/legacy-text/deactivate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match service.deactivate_legacy_text_plugin(&plugin_name) {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(status) => json_response(200, status),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                },
                Err(error) => json_response(400, json!({ "error": error.to_string() })),
            }
        }
        ("POST", _) | ("PUT", _) | ("PATCH", _) | ("DELETE", _) => {
            json_response(404, json!({ "error": "route not found" }))
        }
        ("GET", _) => json_response(404, json!({ "error": "route not found" })),
        _ => text_response(405, "method not allowed"),
    }
}

pub fn request_path(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

pub fn plugin_name_from_route<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix).and_then(|plugin_name| {
        if plugin_name.is_empty() || plugin_name.contains('/') {
            None
        } else {
            Some(plugin_name)
        }
    })
}

fn plugin_name_from_body(body: &[u8]) -> anyhow::Result<String> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))?;
    let plugin_name = payload
        .get("plugin_name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("JSON body must contain string field `plugin_name`"))?;
    Ok(plugin_name.to_string())
}

fn model_load_request_from_body(body: &[u8]) -> anyhow::Result<ModelLoadRequest> {
    serde_json::from_slice(body).map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))
}

fn plugin_load_request_from_body(body: &[u8]) -> anyhow::Result<PluginLoadRequest> {
    serde_json::from_slice(body).map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))
}

fn core_rewriter_activation_request_from_body(
    body: &[u8],
) -> anyhow::Result<CoreRewriterActivationRequest> {
    serde_json::from_slice(body).map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))
}

fn text_generation_request_from_body(body: &[u8]) -> anyhow::Result<TextGenerationRequest> {
    serde_json::from_slice(body).map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))
}

fn command_execution_request_from_body(body: &[u8]) -> anyhow::Result<CommandExecutionRequest> {
    serde_json::from_slice(body).map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))
}

fn status_reason(status_code: u16) -> &'static str {
    match status_code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

fn json_response(status_code: u16, value: Value) -> HttpResponse {
    HttpResponse {
        status_code,
        content_type: "application/json; charset=utf-8",
        body: serde_json::to_vec(&value)
            .unwrap_or_else(|_| b"{\"error\":\"serialization failure\"}".to_vec()),
    }
}

fn text_response(status_code: u16, body: impl Into<Vec<u8>>) -> HttpResponse {
    HttpResponse {
        status_code,
        content_type: "text/plain; charset=utf-8",
        body: body.into(),
    }
}

fn write_http_response(stream: &mut TcpStream, response: HttpResponse) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status_code,
        status_reason(response.status_code),
        response.content_type,
        response.body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()?;
    Ok(())
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn read_http_request(stream: &mut TcpStream) -> anyhow::Result<HttpRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut header_end = None;

    while header_end.is_none() {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        header_end = find_header_end(&buffer);
        if buffer.len() > 1024 * 1024 {
            return Err(anyhow::anyhow!("request headers too large"));
        }
    }

    let header_end = header_end
        .ok_or_else(|| anyhow::anyhow!("invalid HTTP request: missing header terminator"))?;
    let header_bytes = &buffer[..header_end];
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| anyhow::anyhow!("invalid HTTP request header encoding"))?;
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing HTTP request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing HTTP method"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing HTTP path"))?
        .to_string();

    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| {
            if name.eq_ignore_ascii_case("content-length") {
                Some(value.trim())
            } else {
                None
            }
        })
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| anyhow::anyhow!("invalid content-length"))?
        .unwrap_or(0);

    let mut body = buffer[(header_end + 4)..].to_vec();
    while body.len() < content_length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }

    if body.len() < content_length {
        return Err(anyhow::anyhow!("incomplete HTTP request body"));
    }
    body.truncate(content_length);

    Ok(HttpRequest { method, path, body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_core::{
        CoreRewriters, InferenceEngine, PlatformTrack, PluginBootstrap, PluginCompatibility,
        PluginManifest, PluginRuntime, RegisteredPlugin, SamplingHook,
    };
    use std::fs;
    use std::net::Shutdown;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::thread;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-cli-management-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        dir
    }

    fn empty_service() -> ManagementService {
        ManagementService::new(InferenceEngine::builder().build().expect("build engine"))
    }

    fn plugin_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        let mut manifest = PluginManifest {
            name: "managed-inference".to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            min_host_version: None,
            max_host_version: None,
            target_tracks: vec![PlatformTrack::AiInfra],
            contributes: Default::default(),
            core_rewriters: CoreRewriters {
                inference: true,
                ..Default::default()
            },
            runtime: PluginRuntime {
                library_path: Some("runtime/plugin.dll".to_string()),
                wasm_path: None,
                sampling_profile: Some("sampling-hook.toml".to_string()),
            },
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility {
                legacy_runtime_path: Some("legacy/compat.dll".to_string()),
                ..Default::default()
            },
        };
        manifest.contributes.inference_hooks = vec!["sampling-profile".to_string()];
        manifest.contributes.workflows = vec!["agent.pipeline".to_string()];
        manifest.contributes.custom_nodes = vec!["node.rewrite".to_string()];
        manifest.contributes.commands = vec!["plugins.reload".to_string()];
        manifest.contributes.ui_contributes.panels = vec!["inspector".to_string()];
        manifest.contributes.ui_contributes.windows = vec!["governance".to_string()];
        manifest.contributes.ui_contributes.widgets = vec!["status-pill".to_string()];
        engine
            .register_plugin(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        ManagementService::new(engine)
    }

    fn workflow_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "workflow-override".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: loci_core::ContributionPoints {
                    workflows: vec!["agent.plan".to_string(), "agent.review".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register plugin");
        ManagementService::new(engine)
    }

    fn ui_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        let mut manifest = PluginManifest {
            name: "ui-shell".to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            min_host_version: None,
            max_host_version: None,
            target_tracks: vec![PlatformTrack::AiAgent],
            contributes: Default::default(),
            core_rewriters: CoreRewriters {
                ui_host: true,
                ..Default::default()
            },
            runtime: PluginRuntime::default(),
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility::default(),
        };
        manifest.contributes.ui_contributes.panels = vec!["inspector".to_string()];
        manifest.contributes.ui_contributes.windows = vec!["governance".to_string()];
        manifest.contributes.ui_contributes.widgets = vec!["status-pill".to_string()];
        engine
            .register_plugin(RegisteredPlugin::new(manifest))
            .expect("register ui plugin");
        ManagementService::new(engine)
    }

    fn combined_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "managed-inference".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: Default::default(),
                core_rewriters: CoreRewriters {
                    inference: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register inference plugin");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "workflow-override".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: Default::default(),
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register workflow plugin");
        ManagementService::new(engine)
    }

    fn command_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "command-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: loci_core::ContributionPoints {
                    commands: vec!["plugins.reload".to_string(), "plugins.audit".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    plugin_manager: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register plugin manager plugin");
        ManagementService::new(engine)
    }

    struct ForceTokenHook;

    impl SamplingHook for ForceTokenHook {}

    fn hooked_plugin_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "hooked-plugin".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: Default::default(),
                core_rewriters: CoreRewriters {
                    inference: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register plugin");
        engine
            .register_sampling_hook("hooked-plugin", Arc::new(ForceTokenHook))
            .expect("register hook");
        ManagementService::new(engine)
    }

    #[test]
    fn request_path_ignores_query_string() {
        assert_eq!(
            request_path("/v1/plugins/demo?verbose=true"),
            "/v1/plugins/demo"
        );
    }

    #[test]
    fn health_route_returns_ok_json() {
        let response = handle_management_request(
            &empty_service(),
            HttpRequest {
                method: "GET".to_string(),
                path: "/health".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["status"], "ok");
    }

    #[test]
    fn runtime_route_returns_snapshot_json() {
        let response = handle_management_request(
            &empty_service(),
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/runtime".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["plugin_count"], 0);
        assert_eq!(value["available_backends"][0]["name"], "mock");
        assert_eq!(value["active_model_path"], Value::Null);
        assert_eq!(value["active_model_info"], Value::Null);
        assert_eq!(value["legacy_text_candidates"], json!([]));
    }

    #[test]
    fn backends_route_returns_available_backend_inventory() {
        let response = handle_management_request(
            &empty_service(),
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/backends".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value[0]["name"], "mock");
        assert_eq!(value[0]["supports_text"], true);
    }

    #[test]
    fn workflows_route_returns_effective_workflow_inventory() {
        let service = workflow_service();
        let before = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/workflows".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(before.status_code, 200);
        let before_value: Value = serde_json::from_slice(&before.body).expect("json");
        assert_eq!(before_value["active_workflow_rewriter"], Value::Null);
        assert_eq!(before_value["workflows"], json!([]));

        let activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"workflow","plugin_name":"workflow-override"}"#.to_vec(),
            },
        );
        assert_eq!(activation.status_code, 200);

        let after = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/workflows".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(after.status_code, 200);
        let after_value: Value = serde_json::from_slice(&after.body).expect("json");
        assert_eq!(after_value["active_workflow_rewriter"], "workflow-override");
        assert_eq!(
            after_value["workflows"],
            json!(["agent.plan", "agent.review"])
        );
    }

    #[test]
    fn ui_route_returns_effective_ui_inventory() {
        let service = ui_service();
        let before = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/ui".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(before.status_code, 200);
        let before_value: Value = serde_json::from_slice(&before.body).expect("json");
        assert_eq!(before_value["active_ui_host"], Value::Null);
        assert_eq!(before_value["ui"]["panels"], json!([]));
        assert_eq!(before_value["ui"]["windows"], json!([]));
        assert_eq!(before_value["ui"]["widgets"], json!([]));

        let activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"ui_host","plugin_name":"ui-shell"}"#.to_vec(),
            },
        );
        assert_eq!(activation.status_code, 200);

        let after = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/ui".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(after.status_code, 200);
        let after_value: Value = serde_json::from_slice(&after.body).expect("json");
        assert_eq!(after_value["active_ui_host"], "ui-shell");
        assert_eq!(after_value["ui"]["panels"], json!(["inspector"]));
        assert_eq!(after_value["ui"]["windows"], json!(["governance"]));
        assert_eq!(after_value["ui"]["widgets"], json!(["status-pill"]));
    }

    #[test]
    fn commands_route_returns_active_plugin_manager_inventory() {
        let service = command_service();
        let before = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/commands".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(before.status_code, 200);
        let before_value: Value = serde_json::from_slice(&before.body).expect("json");
        assert_eq!(before_value["active_plugin_manager"], Value::Null);
        assert_eq!(before_value["commands"], json!([]));

        let activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"plugin_manager","plugin_name":"command-router"}"#.to_vec(),
            },
        );
        assert_eq!(activation.status_code, 200);

        let after = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/commands".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(after.status_code, 200);
        let after_value: Value = serde_json::from_slice(&after.body).expect("json");
        assert_eq!(after_value["active_plugin_manager"], "command-router");
        assert_eq!(
            after_value["commands"],
            json!(["plugins.reload", "plugins.audit"])
        );
    }

    #[test]
    fn command_run_route_requires_active_plugin_manager_and_declared_command() {
        let service = command_service();
        let missing_activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/commands/run".to_string(),
                body: br#"{"command":"plugins.reload"}"#.to_vec(),
            },
        );
        assert_eq!(missing_activation.status_code, 400);
        let missing_value: Value = serde_json::from_slice(&missing_activation.body).expect("json");
        assert!(missing_value["error"]
            .as_str()
            .expect("error")
            .contains("no active plugin manager rewriter"));

        let activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"plugin_manager","plugin_name":"command-router"}"#.to_vec(),
            },
        );
        assert_eq!(activation.status_code, 200);

        let success = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/commands/run".to_string(),
                body: br#"{"command":"plugins.reload"}"#.to_vec(),
            },
        );
        assert_eq!(success.status_code, 200);
        let success_value: Value = serde_json::from_slice(&success.body).expect("json");
        assert_eq!(success_value["status"], "accepted");
        assert_eq!(success_value["command"], "plugins.reload");
        assert_eq!(success_value["routed_plugin_name"], "command-router");

        let undeclared = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/commands/run".to_string(),
                body: br#"{"command":"plugins.missing"}"#.to_vec(),
            },
        );
        assert_eq!(undeclared.status_code, 400);
        let undeclared_value: Value = serde_json::from_slice(&undeclared.body).expect("json");
        assert!(undeclared_value["error"]
            .as_str()
            .expect("error")
            .contains("is not declared by active plugin manager"));
    }

    #[test]
    fn plugin_detail_route_returns_not_found_for_unknown_plugin() {
        let response = handle_management_request(
            &empty_service(),
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/plugins/missing".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 404);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["error"], "plugin not found");
    }

    #[test]
    fn plugin_detail_route_returns_detail_payload() {
        let response = handle_management_request(
            &plugin_service(),
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/plugins/managed-inference?verbose=true".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["status"]["name"], "managed-inference");
        assert_eq!(value["status"]["declares_host_runtime"], true);
        assert_eq!(value["status"]["registered_host_runtime"], false);
        assert_eq!(value["status"]["materialized_host_runtime"], false);
        assert_eq!(value["status"]["host_runtime_kind"], Value::Null);
        assert_eq!(value["declared_core_rewriters"], json!(["inference"]));
        assert_eq!(
            value["runtime_artifacts"]["library_path"],
            "runtime/plugin.dll"
        );
        assert_eq!(value["runtime_artifacts"]["wasm_path"], Value::Null);
        assert_eq!(
            value["runtime_artifacts"]["sampling_profile"],
            "sampling-hook.toml"
        );
        assert_eq!(
            value["runtime_artifacts"]["legacy_runtime_path"],
            "legacy/compat.dll"
        );
        assert_eq!(value["runtime_artifacts"]["host_runtimes"], json!([]));
        assert_eq!(
            value["runtime_artifacts"]["materialized_host_runtime"],
            Value::Null
        );
        assert_eq!(value["workflows"], json!(["agent.pipeline"]));
        assert_eq!(value["custom_nodes"], json!(["node.rewrite"]));
        assert_eq!(value["commands"], json!(["plugins.reload"]));
        assert_eq!(value["ui"]["panels"], json!(["inspector"]));
        assert_eq!(value["ui"]["windows"], json!(["governance"]));
        assert_eq!(value["ui"]["widgets"], json!(["status-pill"]));
    }

    #[test]
    fn plugins_route_reports_declared_registered_and_effective_sampling_states() {
        let service = hooked_plugin_service();
        let before = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/plugins".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(before.status_code, 200);
        let before_value: Value = serde_json::from_slice(&before.body).expect("json");
        assert_eq!(before_value[0]["name"], "hooked-plugin");
        assert_eq!(before_value[0]["declares_sampling_hook"], false);
        assert_eq!(
            before_value[0]["sampling_hook_source"],
            "dynamic_registration"
        );
        assert_eq!(before_value[0]["registered_sampling_hook"], true);
        assert_eq!(before_value[0]["effective_sampling_hook"], false);
        assert_eq!(before_value[0]["active_inference_rewriter"], false);

        let activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/inference/activate".to_string(),
                body: br#"{"plugin_name":"hooked-plugin"}"#.to_vec(),
            },
        );
        assert_eq!(activation.status_code, 200);

        let after = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/plugins/hooked-plugin".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(after.status_code, 200);
        let after_value: Value = serde_json::from_slice(&after.body).expect("json");
        assert_eq!(after_value["status"]["declares_sampling_hook"], false);
        assert_eq!(
            after_value["status"]["sampling_hook_source"],
            "dynamic_registration"
        );
        assert_eq!(after_value["status"]["registered_sampling_hook"], true);
        assert_eq!(after_value["status"]["effective_sampling_hook"], true);
        assert_eq!(after_value["status"]["active_inference_rewriter"], true);
    }

    #[test]
    fn plugin_load_route_loads_manifest_bundle_and_updates_runtime() {
        let service = empty_service();
        let dir = unique_temp_dir("plugin-load");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join("manifest.toml"),
            r#"
name = "route-plugin"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_infra"]
"#,
        )
        .expect("write manifest");

        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/plugins/load".to_string(),
                body: serde_json::to_vec(&json!({
                    "path": dir.join("plugin").join("manifest.toml").display().to_string(),
                    "source_kind": "bundle_file"
                }))
                .expect("serialize request"),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["status"], "loaded");
        assert_eq!(value["loaded_count"], 1);
        assert_eq!(value["loaded_plugin_names"], json!(["route-plugin"]));

        let runtime = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/runtime".to_string(),
                body: Vec::new(),
            },
        );
        let runtime_value: Value = serde_json::from_slice(&runtime.body).expect("json");
        assert_eq!(
            runtime_value["loaded_plugin_names"],
            json!(["route-plugin"])
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_load_route_loads_model_and_returns_runtime_details() {
        let service = empty_service();
        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/model/load".to_string(),
                body: br#"{
                    "backend_name":"mock",
                    "config":{
                        "model_path":"demo.gguf",
                        "use_gpu":false,
                        "n_gpu_layers":0,
                        "kv_offload":false,
                        "op_offload":false,
                        "split_mode":"none"
                    }
                }"#
                .to_vec(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["status"], "loaded");
        assert_eq!(value["backend_name"], "mock");
        assert_eq!(value["model_path"], "demo.gguf");
        assert_eq!(value["active_backend"], "mock");
        assert_eq!(value["active_model_path"], "demo.gguf");
        assert_eq!(value["active_model_info"]["architecture"], "mock");

        let runtime = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/runtime".to_string(),
                body: Vec::new(),
            },
        );
        let runtime_value: Value = serde_json::from_slice(&runtime.body).expect("json");
        assert_eq!(runtime_value["active_backend"], "mock");
        assert_eq!(runtime_value["active_model_path"], "demo.gguf");
        assert_eq!(runtime_value["active_model_info"]["architecture"], "mock");
    }

    #[test]
    fn inference_generate_route_runs_generation_after_model_load() {
        let service = empty_service();
        let load = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/model/load".to_string(),
                body: br#"{
                    "backend_name":"mock",
                    "config":{
                        "model_path":"demo.gguf",
                        "use_gpu":false,
                        "n_gpu_layers":0,
                        "kv_offload":false,
                        "op_offload":false,
                        "split_mode":"none"
                    }
                }"#
                .to_vec(),
            },
        );
        assert_eq!(load.status_code, 200);

        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/inference/generate".to_string(),
                body: serde_json::to_vec(&json!({
                    "prompt": "hello",
                    "params": {
                        "max_tokens": 16,
                        "temperature": 0.2
                    }
                }))
                .expect("serialize request"),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["active_backend"], "mock");
        assert_eq!(value["active_model_path"], "demo.gguf");
        assert_eq!(value["active_model_info"]["architecture"], "mock");
        assert_eq!(value["active_inference"], Value::Null);
        assert!(value["output"]
            .as_str()
            .expect("output string")
            .contains("mock:hello"));
    }

    #[test]
    fn inference_activate_route_switches_active_inference_plugin() {
        let service = plugin_service();
        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/inference/activate".to_string(),
                body: br#"{"plugin_name":"managed-inference"}"#.to_vec(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["component"], "inference");
        assert_eq!(value["active_inference"], "managed-inference");
        assert!(value.get("configured_core_rewriters").is_none());

        let runtime = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/core/rewriters".to_string(),
                body: Vec::new(),
            },
        );
        let rewriters: Value = serde_json::from_slice(&runtime.body).expect("json");
        assert_eq!(
            rewriters,
            json!([{ "component": "inference", "plugin_name": "managed-inference" }])
        );
    }

    #[test]
    fn core_rewriter_activate_route_switches_non_inference_component() {
        let service = workflow_service();
        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"workflow","plugin_name":"workflow-override"}"#.to_vec(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["component"], "workflow");
        assert_eq!(value["plugin_name"], "workflow-override");
        assert_eq!(value["active_inference"], Value::Null);
        assert_eq!(
            value["configured_core_rewriters"],
            json!([{ "component": "workflow", "plugin_name": "workflow-override" }])
        );

        let runtime = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/core/rewriters".to_string(),
                body: Vec::new(),
            },
        );
        let rewriters: Value = serde_json::from_slice(&runtime.body).expect("json");
        assert_eq!(
            rewriters,
            json!([{ "component": "workflow", "plugin_name": "workflow-override" }])
        );
    }

    #[test]
    fn core_rewriter_activate_route_supports_inference_component() {
        let service = plugin_service();
        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"inference","plugin_name":"managed-inference"}"#.to_vec(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["component"], "inference");
        assert_eq!(value["active_inference"], "managed-inference");
        assert_eq!(
            value["configured_core_rewriters"],
            json!([{ "component": "inference", "plugin_name": "managed-inference" }])
        );
    }

    #[test]
    fn core_rewriter_inventory_route_lists_available_plugins() {
        let service = combined_service();
        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/core/rewriters/inventory".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(
            value,
            json!([
                {
                    "component": "inference",
                    "active_plugin_name": null,
                    "available_plugins": ["managed-inference"]
                },
                {
                    "component": "model",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "hardware",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "workflow",
                    "active_plugin_name": null,
                    "available_plugins": ["workflow-override"]
                },
                {
                    "component": "event_bus",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "plugin_manager",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "ui_host",
                    "active_plugin_name": null,
                    "available_plugins": []
                }
            ])
        );
    }

    #[test]
    fn core_rewriter_inventory_route_reflects_activation_state() {
        let service = combined_service();
        let workflow_activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"component":"workflow","plugin_name":"workflow-override"}"#.to_vec(),
            },
        );
        assert_eq!(workflow_activation.status_code, 200);

        let inference_activation = handle_management_request(
            &service,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/inference/activate".to_string(),
                body: br#"{"plugin_name":"managed-inference"}"#.to_vec(),
            },
        );
        assert_eq!(inference_activation.status_code, 200);

        let response = handle_management_request(
            &service,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/core/rewriters/inventory".to_string(),
                body: Vec::new(),
            },
        );
        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(
            value,
            json!([
                {
                    "component": "inference",
                    "active_plugin_name": "managed-inference",
                    "available_plugins": ["managed-inference"]
                },
                {
                    "component": "model",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "hardware",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "workflow",
                    "active_plugin_name": "workflow-override",
                    "available_plugins": ["workflow-override"]
                },
                {
                    "component": "event_bus",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "plugin_manager",
                    "active_plugin_name": null,
                    "available_plugins": []
                },
                {
                    "component": "ui_host",
                    "active_plugin_name": null,
                    "available_plugins": []
                }
            ])
        );
    }

    #[test]
    fn generic_core_rewriter_route_rejects_missing_component() {
        let response = handle_management_request(
            &workflow_service(),
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/rewriters/activate".to_string(),
                body: br#"{"plugin_name":"workflow-override"}"#.to_vec(),
            },
        );

        assert_eq!(response.status_code, 400);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert!(value["error"]
            .as_str()
            .expect("error string")
            .contains("component"));
    }

    #[test]
    fn read_http_request_parses_body_from_tcp_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
        let addr = listener.local_addr().expect("listener addr");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            read_http_request(&mut stream).expect("read request")
        });

        let mut client = TcpStream::connect(addr).expect("connect");
        client
            .write_all(
                b"POST /v1/core/inference/activate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 35\r\n\r\n{\"plugin_name\":\"managed-inference\"}",
            )
            .expect("write request");
        client.shutdown(Shutdown::Write).expect("shutdown write");

        let request = handle.join().expect("join reader");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/core/inference/activate");
        assert_eq!(request.body, br#"{"plugin_name":"managed-inference"}"#);
    }
}
