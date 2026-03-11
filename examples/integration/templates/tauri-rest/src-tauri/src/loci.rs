use reqwest::Client;
use serde::{Deserialize, Serialize};

fn base_url() -> String {
    std::env::var("LOCI_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InfoResponse {
    pub status: String,
    pub version: String,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    prompt: &'a str,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
}

#[tauri::command]
pub async fn loci_health() -> Result<HealthResponse, String> {
    let url = format!("{}/v1/health", base_url());
    let client = Client::new();
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("health check failed: {}", resp.status()));
    }
    resp.json::<HealthResponse>().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn loci_info() -> Result<InfoResponse, String> {
    let url = format!("{}/v1/info", base_url());
    let client = Client::new();
    let resp = client.get(url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("info request failed: {}", resp.status()));
    }
    resp.json::<InfoResponse>().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn loci_generate(
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<String, String> {
    let url = format!("{}/v1/generate", base_url());
    let payload = GenerateRequest {
        prompt: &prompt,
        max_tokens: max_tokens.unwrap_or(256),
        temperature: temperature.unwrap_or(0.7),
    };

    let client = Client::new();
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if resp.status().is_success() {
        let ok = resp
            .json::<GenerateResponse>()
            .await
            .map_err(|e| e.to_string())?;
        Ok(ok.response)
    } else {
        let status = resp.status();
        let err = resp
            .json::<ErrorResponse>()
            .await
            .map(|e| e.error)
            .unwrap_or_else(|_| "unknown error".to_string());
        Err(format!("generate failed ({status}): {err}"))
    }
}
