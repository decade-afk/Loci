use serde::{Deserialize, Serialize};

pub const HOST_PLUGIN_API_VERSION: &str = "1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginKind {
    ModelLoader,
    HardwareBackend,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginCapabilities {
    #[serde(default)]
    pub model_formats: Vec<String>,
    #[serde(default)]
    pub hardware_targets: Vec<String>,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginRuntime {
    #[serde(default)]
    pub library_path: Option<String>,
    #[serde(default)]
    pub wasm_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub api_version: String,
    pub kind: PluginKind,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub auto_activate: bool,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub capabilities: PluginCapabilities,
    #[serde(default)]
    pub runtime: PluginRuntime,
}

impl PluginManifest {
    pub fn supports_model_format(&self, format: &str) -> bool {
        self.capabilities
            .model_formats
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(format))
    }

    pub fn targets_hardware(&self, hardware: &str) -> bool {
        self.capabilities
            .hardware_targets
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(hardware))
    }
}

pub trait Plugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_matches_model_formats_case_insensitively() {
        let manifest = PluginManifest {
            name: "gguf-loader".to_string(),
            version: "0.1.0".to_string(),
            api_version: HOST_PLUGIN_API_VERSION.to_string(),
            kind: PluginKind::ModelLoader,
            description: None,
            auto_activate: true,
            priority: 10,
            capabilities: PluginCapabilities {
                model_formats: vec!["gguf".to_string()],
                hardware_targets: Vec::new(),
                features: vec!["dynamic_load".to_string()],
            },
            runtime: PluginRuntime::default(),
        };

        assert!(manifest.supports_model_format("GGUF"));
        assert!(!manifest.supports_model_format("onnx"));
    }
}
