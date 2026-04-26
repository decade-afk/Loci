use anyhow::Context;
use loci_core::{InferenceEngine, InferenceParams};
use serde::Deserialize;
use std::io::{Read, Write};
use std::net::TcpListener;

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
    let mut lines = request.lines();
    let request_line = lines.next().unwrap_or_default();
    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();

    if request_line.starts_with("GET /health ") {
        return Ok(json_response(r#"{"status":"ok"}"#));
    }

    if request_line.starts_with("GET /v1/runtime ") {
        let json = serde_json::to_string(&engine.runtime_snapshot())?;
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
        let response = engine.infer(&payload.prompt, &params)?;
        let json = serde_json::to_string(&response)?;
        return Ok(json_response(&json));
    }

    Ok(http_response(
        "404 Not Found",
        "application/json",
        r#"{"error":"not_found"}"#,
    ))
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
