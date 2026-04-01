use crate::backend::InferenceParams;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

impl From<GenerationParams> for InferenceParams {
    fn from(params: GenerationParams) -> Self {
        Self {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginRuntimeStatus {
    pub name: String,
    pub version: String,
    pub supports_ai_infra: bool,
    pub supports_ai_agent: bool,
    pub declares_inference_rewriter: bool,
    pub has_sampling_hook: bool,
    pub is_legacy_compat: bool,
    pub legacy_text_candidate: bool,
    pub active_legacy_text: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    pub plugin_count: usize,
    pub loaded_plugin_names: Vec<String>,
    pub active_backend: Option<String>,
    pub active_inference: Option<String>,
    pub legacy_text_candidates: Vec<String>,
    pub active_legacy_text: Vec<String>,
    pub plugins: Vec<PluginRuntimeStatus>,
}
