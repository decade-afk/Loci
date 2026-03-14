use crate::error::{LociError, Result};
use crate::plugin::Plugin;
use crate::sampler::LogitsView;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Minimal WASM plugin configuration retained for serde/API compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPluginConfig {
    pub name: String,
    pub version: String,
    pub wasm_path: PathBuf,
    pub max_memory: usize,
    pub max_fuel: u64,
    pub enable_wasi: bool,
    pub allow_unsafe_wasi: bool,
    pub timeout_ms: u64,
}

impl Default for WasmPluginConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".to_string(),
            wasm_path: PathBuf::new(),
            max_memory: 16 * 1024 * 1024,
            max_fuel: 1_000_000,
            enable_wasi: false,
            allow_unsafe_wasi: false,
            timeout_ms: 5000,
        }
    }
}

/// Stub runtime used on targets where the WASM plugin runtime is disabled.
pub struct WasmPlugin {
    config: WasmPluginConfig,
}

impl WasmPlugin {
    pub fn config(&self) -> &WasmPluginConfig {
        &self.config
    }

    pub fn load(config: WasmPluginConfig) -> Result<Self> {
        Err(LociError::PluginError(format!(
            "WASM plugins are disabled in this build; rebuild with the 'wasm-plugins' feature to load '{}'",
            config.wasm_path.display()
        )))
    }
}

impl Plugin for WasmPlugin {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(prompt.to_string())
    }

    fn transform_logits(&self, _logits: &mut LogitsView, _context: &[i32]) -> Result<()> {
        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        Ok(token_id)
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct WasmPluginManager {
    plugins: Vec<WasmPlugin>,
}

impl WasmPluginManager {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub fn load(&mut self, config: WasmPluginConfig) -> Result<()> {
        let plugin = WasmPlugin::load(config)?;
        self.plugins.push(plugin);
        Ok(())
    }

    pub fn plugins(&self) -> &[WasmPlugin] {
        &self.plugins
    }
}

impl Default for WasmPluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_defaults() {
        let config = WasmPluginConfig::default();
        assert_eq!(config.max_memory, 16 * 1024 * 1024);
        assert_eq!(config.max_fuel, 1_000_000);
        assert!(!config.enable_wasi);
        assert!(!config.allow_unsafe_wasi);
    }

    #[test]
    fn test_wasm_plugin_manager() {
        let manager = WasmPluginManager::new();
        assert_eq!(manager.plugins().len(), 0);
    }
}
