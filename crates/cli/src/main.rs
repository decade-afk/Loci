use loci_core::{InferenceEngine, PlatformTrack};
use serde_json::{json, Value};
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    plugin_dir: PathBuf,
    activate_legacy_text_plugins: Vec<String>,
    management_bind: Option<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            plugin_dir: PathBuf::from("plugins"),
            activate_legacy_text_plugins: Vec::new(),
            management_bind: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpResponse {
    status_code: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

fn parse_args<I>(args: I) -> anyhow::Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = CliArgs::default();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--plugin-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--plugin-dir requires a path"))?;
                parsed.plugin_dir = PathBuf::from(value);
            }
            "--activate-legacy-text-plugin" => {
                let value = args.next().ok_or_else(|| {
                    anyhow::anyhow!("--activate-legacy-text-plugin requires a plugin name")
                })?;
                parsed.activate_legacy_text_plugins.push(value);
            }
            "--management-bind" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--management-bind requires an address"))?;
                parsed.management_bind = Some(value);
            }
            other => {
                return Err(anyhow::anyhow!("unknown argument: {other}"));
            }
        }
    }

    Ok(parsed)
}

fn comma_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
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

fn plugin_name_from_body(body: &[u8]) -> anyhow::Result<String> {
    let payload: Value = serde_json::from_slice(body)
        .map_err(|error| anyhow::anyhow!("invalid JSON body: {error}"))?;
    let plugin_name = payload
        .get("plugin_name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("JSON body must contain string field `plugin_name`"))?;
    Ok(plugin_name.to_string())
}

fn request_path(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn plugin_name_from_route<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix).and_then(|plugin_name| {
        if plugin_name.is_empty() || plugin_name.contains('/') {
            None
        } else {
            Some(plugin_name)
        }
    })
}

fn handle_management_request(
    engine: &Arc<Mutex<InferenceEngine>>,
    request: HttpRequest,
) -> HttpResponse {
    let path = request_path(&request.path);

    if request.method == "GET" {
        if let Some(plugin_name) = plugin_name_from_route(path, "/v1/plugins/") {
            return match engine.lock() {
                Ok(engine) => match engine.plugin_runtime_detail(plugin_name) {
                    Some(detail) => match serde_json::to_value(detail) {
                        Ok(detail) => json_response(200, detail),
                        Err(error) => json_response(500, json!({ "error": error.to_string() })),
                    },
                    None => json_response(404, json!({ "error": "plugin not found" })),
                },
                Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
            };
        }
    }

    match (request.method.as_str(), path) {
        ("GET", "/health") => json_response(200, json!({ "status": "ok" })),
        ("GET", "/v1/runtime") => match engine.lock() {
            Ok(engine) => match serde_json::to_value(engine.runtime_snapshot()) {
                Ok(snapshot) => json_response(200, snapshot),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
        },
        ("GET", "/v1/core/rewriters") => match engine.lock() {
            Ok(engine) => {
                match serde_json::to_value(engine.runtime_snapshot().configured_core_rewriters) {
                    Ok(rewriters) => json_response(200, rewriters),
                    Err(error) => json_response(500, json!({ "error": error.to_string() })),
                }
            }
            Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
        },
        ("GET", "/v1/plugins") => match engine.lock() {
            Ok(engine) => match serde_json::to_value(engine.runtime_snapshot().plugins) {
                Ok(plugins) => json_response(200, plugins),
                Err(error) => json_response(500, json!({ "error": error.to_string() })),
            },
            Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
        },
        ("POST", "/v1/core/inference/activate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match engine.lock() {
                Ok(mut engine) => match engine.activate_inference_plugin(&plugin_name) {
                    Ok(()) => json_response(
                        200,
                        json!({
                            "status": "activated",
                            "component": "inference",
                            "plugin_name": plugin_name,
                            "active_inference": engine.runtime_snapshot().active_inference,
                        }),
                    ),
                    Err(error) => json_response(400, json!({ "error": error.to_string() })),
                },
                Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
            }
        }
        ("POST", "/v1/legacy-text/activate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match engine.lock() {
                Ok(mut engine) => match engine.activate_legacy_text_plugin(&plugin_name) {
                    Ok(()) => json_response(
                        200,
                        json!({
                            "status": "activated",
                            "plugin_name": plugin_name,
                            "active_legacy_text": engine.active_legacy_text_plugins(),
                        }),
                    ),
                    Err(error) => json_response(400, json!({ "error": error.to_string() })),
                },
                Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
            }
        }
        ("POST", "/v1/legacy-text/deactivate") => {
            let plugin_name = match plugin_name_from_body(&request.body) {
                Ok(plugin_name) => plugin_name,
                Err(error) => return json_response(400, json!({ "error": error.to_string() })),
            };

            match engine.lock() {
                Ok(mut engine) => match engine.deactivate_legacy_text_plugin(&plugin_name) {
                    Ok(()) => json_response(
                        200,
                        json!({
                            "status": "deactivated",
                            "plugin_name": plugin_name,
                            "active_legacy_text": engine.active_legacy_text_plugins(),
                        }),
                    ),
                    Err(error) => json_response(400, json!({ "error": error.to_string() })),
                },
                Err(_) => json_response(500, json!({ "error": "engine mutex poisoned" })),
            }
        }
        ("POST", _) | ("PUT", _) | ("PATCH", _) | ("DELETE", _) => {
            json_response(404, json!({ "error": "route not found" }))
        }
        ("GET", _) => json_response(404, json!({ "error": "route not found" })),
        _ => text_response(405, "method not allowed"),
    }
}

fn run_management_server(
    bind_addr: &str,
    engine: Arc<Mutex<InferenceEngine>>,
) -> anyhow::Result<()> {
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
            Ok(request) => handle_management_request(&engine, request),
            Err(error) => json_response(400, json!({ "error": error.to_string() })),
        };

        if let Err(error) = write_http_response(&mut stream, response) {
            eprintln!("management response error: {error}");
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = parse_args(env::args().skip(1))?;
    let mut engine = InferenceEngine::builder().build()?;
    let loaded = engine.load_plugins_from_dir(&args.plugin_dir)?;

    for plugin_name in &args.activate_legacy_text_plugins {
        engine.activate_legacy_text_plugin(plugin_name)?;
    }

    let snapshot = engine.runtime_snapshot();
    println!(
        "loci-cli ready; plugins={}, loaded_now={}, infra_plugins={}, agent_plugins={}, active_inference={}, legacy_text_candidates={}, active_legacy_text={}, management_bind={}",
        snapshot.plugin_count,
        loaded,
        engine.plugins_for_track(PlatformTrack::AiInfra).len(),
        engine.plugins_for_track(PlatformTrack::AiAgent).len(),
        snapshot.active_inference.as_deref().unwrap_or("none"),
        comma_or_none(&snapshot.legacy_text_candidates),
        comma_or_none(&snapshot.active_legacy_text),
        args.management_bind.as_deref().unwrap_or("none"),
    );

    if let Some(bind_addr) = args.management_bind {
        return run_management_server(&bind_addr, Arc::new(Mutex::new(engine)));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_core::{
        CoreRewriters, PluginBootstrap, PluginCompatibility, PluginManifest, PluginRuntime,
        RegisteredPlugin,
    };

    fn empty_engine() -> InferenceEngine {
        InferenceEngine::builder().build().expect("build engine")
    }

    fn plugin_engine() -> InferenceEngine {
        let mut engine = empty_engine();
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
            runtime: PluginRuntime::default(),
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility::default(),
        };
        manifest.contributes.inference_hooks = vec!["sampling-profile".to_string()];
        engine
            .register_plugin(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        engine
    }

    #[test]
    fn parse_args_supports_management_bind_and_repeated_legacy_activation() {
        let parsed = parse_args([
            "--plugin-dir".to_string(),
            "custom-plugins".to_string(),
            "--management-bind".to_string(),
            "127.0.0.1:8080".to_string(),
            "--activate-legacy-text-plugin".to_string(),
            "legacy-a".to_string(),
            "--activate-legacy-text-plugin".to_string(),
            "legacy-b".to_string(),
        ])
        .expect("parse args");

        assert_eq!(
            parsed,
            CliArgs {
                plugin_dir: PathBuf::from("custom-plugins"),
                activate_legacy_text_plugins: vec!["legacy-a".to_string(), "legacy-b".to_string(),],
                management_bind: Some("127.0.0.1:8080".to_string()),
            }
        );
    }

    #[test]
    fn parse_args_rejects_unknown_argument() {
        let err = parse_args(["--unknown".to_string()]).expect_err("should reject");
        assert!(err.to_string().contains("unknown argument"));
    }

    #[test]
    fn comma_or_none_formats_values() {
        assert_eq!(comma_or_none(&[]), "none");
        assert_eq!(
            comma_or_none(&["a".to_string(), "b".to_string()]),
            "a,b".to_string()
        );
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
        let engine = Arc::new(Mutex::new(empty_engine()));
        let response = handle_management_request(
            &engine,
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
        let engine = Arc::new(Mutex::new(empty_engine()));
        let response = handle_management_request(
            &engine,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/runtime".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["plugin_count"], 0);
        assert_eq!(value["legacy_text_candidates"], json!([]));
    }

    #[test]
    fn plugin_detail_route_returns_not_found_for_unknown_plugin() {
        let engine = Arc::new(Mutex::new(empty_engine()));
        let response = handle_management_request(
            &engine,
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
        let engine = Arc::new(Mutex::new(plugin_engine()));
        let response = handle_management_request(
            &engine,
            HttpRequest {
                method: "GET".to_string(),
                path: "/v1/plugins/managed-inference?verbose=true".to_string(),
                body: Vec::new(),
            },
        );

        assert_eq!(response.status_code, 200);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert_eq!(value["status"]["name"], "managed-inference");
        assert_eq!(value["declared_core_rewriters"], json!(["inference"]));
    }

    #[test]
    fn inference_activate_route_rejects_missing_plugin_name() {
        let engine = Arc::new(Mutex::new(empty_engine()));
        let response = handle_management_request(
            &engine,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/core/inference/activate".to_string(),
                body: b"{}".to_vec(),
            },
        );

        assert_eq!(response.status_code, 400);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert!(value["error"]
            .as_str()
            .expect("error string")
            .contains("plugin_name"));
    }

    #[test]
    fn inference_activate_route_switches_active_inference_plugin() {
        let engine = Arc::new(Mutex::new(plugin_engine()));
        let response = handle_management_request(
            &engine,
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

        let runtime = handle_management_request(
            &engine,
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
    fn legacy_text_activate_route_rejects_invalid_body() {
        let engine = Arc::new(Mutex::new(empty_engine()));
        let response = handle_management_request(
            &engine,
            HttpRequest {
                method: "POST".to_string(),
                path: "/v1/legacy-text/activate".to_string(),
                body: b"{}".to_vec(),
            },
        );

        assert_eq!(response.status_code, 400);
        let value: Value = serde_json::from_slice(&response.body).expect("json");
        assert!(value["error"]
            .as_str()
            .expect("error string")
            .contains("plugin_name"));
    }
}
