use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub api_version: String,
    #[serde(default)]
    pub contributes: ContributionPoints,
    #[serde(default)]
    pub core_rewriters: CoreRewriters,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContributionPoints {
    #[serde(default)]
    pub model_providers: Vec<String>,
    #[serde(default)]
    pub inference_hooks: Vec<String>,
    #[serde(default)]
    pub workflows: Vec<String>,
    #[serde(default)]
    pub custom_nodes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoreRewriters {
    #[serde(default)]
    pub inference: bool,
    #[serde(default)]
    pub model: bool,
    #[serde(default)]
    pub hardware: bool,
    #[serde(default)]
    pub workflow: bool,
    #[serde(default)]
    pub event_bus: bool,
    #[serde(default)]
    pub plugin_manager: bool,
    #[serde(default)]
    pub ui_host: bool,
}

pub trait Plugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
}
