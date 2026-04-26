use crate::backend::BackendCapabilities;
use crate::plugin::PluginStatus;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ActiveModelStatus {
    pub architecture: String,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub param_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    pub plugin_count: usize,
    pub plugins: Vec<PluginStatus>,
    pub active_plugins: Vec<String>,
    pub available_backends: Vec<BackendCapabilities>,
    pub active_backend: Option<String>,
    pub active_model_path: Option<String>,
    pub active_model_info: Option<ActiveModelStatus>,
}
