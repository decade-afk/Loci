//! Plugin Registry with Dynamic Loading and Persistence
//!
//! This module provides:
//! - Hot-swappable dynamic plugin loading (via shared libraries)
//! - Persistent plugin configuration (via TOML files)
//! - Centralized plugin management across multiple engines

use crate::error::{LociError, Result};
use crate::plugin::{dynamic_plugin_from_opaque, DynamicPluginOpaque, Plugin};
use crate::sampler::LogitsView;
use crate::wasm_plugin::{WasmPlugin, WasmPluginConfig};
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Plugin type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginType {
    /// Static plugin (compiled into binary)
    Static,
    /// Dynamic plugin (loaded from shared library)
    Dynamic,
    /// WASM plugin (sandboxed execution)
    Wasm,
}

impl PluginType {
    fn as_str(&self) -> &'static str {
        match self {
            PluginType::Static => "static",
            PluginType::Dynamic => "dynamic",
            PluginType::Wasm => "wasm",
        }
    }
}

/// Runtime plugin metadata for management/integration.
#[derive(Debug, Clone, Serialize)]
pub struct PluginRuntimeInfo {
    pub name: String,
    pub version: String,
    pub enabled: bool,
    pub plugin_type: String,
    pub source: Option<String>,
    pub hot_reloadable: bool,
}

/// Plugin configuration that can be serialized to/from TOML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin name (must be unique)
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Whether the plugin is enabled by default
    pub enabled: bool,
    /// Plugin type
    #[serde(default = "default_plugin_type")]
    pub plugin_type: PluginType,
    /// Path to the plugin shared library (for dynamic plugins)
    pub library_path: Option<PathBuf>,
    /// Path to the WASM binary (for WASM plugins)
    pub wasm_path: Option<PathBuf>,
    /// WASM-specific configuration
    #[serde(default)]
    pub wasm_config: Option<WasmPluginConfig>,
    /// Custom configuration for the plugin (JSON-like)
    #[serde(default)]
    pub settings: HashMap<String, String>,
}

fn default_plugin_type() -> PluginType {
    PluginType::Static
}

/// Registry configuration file format
#[derive(Debug, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// List of plugin configurations
    pub plugins: Vec<PluginConfig>,
}

/// Type alias for the plugin constructor function
/// Dynamic plugins must export a function with this signature:
/// ```ignore
/// #[no_mangle]
/// pub extern "C" fn create_plugin_v1() -> DynamicPluginOpaque {
///     dynamic_plugin_into_opaque(Box::new(MyPlugin::new()))
/// }
/// ```
type PluginConstructor = unsafe extern "C" fn() -> DynamicPluginOpaque;

/// Represents a dynamically loaded plugin
pub struct DynamicPlugin {
    plugin: Box<dyn Plugin>,
    #[allow(dead_code)]
    library: Arc<Library>, // Keep library loaded
    config: PluginConfig,
}

/// Global plugin registry with hot-swap and persistence support
pub struct PluginRegistry {
    /// Static plugins (compiled into the binary)
    static_plugins: HashMap<String, Box<dyn Plugin>>,
    /// Dynamic plugins (loaded from shared libraries)
    dynamic_plugins: HashMap<String, DynamicPlugin>,
    /// WASM plugins (sandboxed execution)
    wasm_plugins: HashMap<String, WasmPlugin>,
    /// Plugin enabled states
    enabled_states: HashMap<String, bool>,
    /// Path to the registry configuration file
    config_path: Option<PathBuf>,
}

impl PluginRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            static_plugins: HashMap::new(),
            dynamic_plugins: HashMap::new(),
            wasm_plugins: HashMap::new(),
            enabled_states: HashMap::new(),
            config_path: None,
        }
    }

    /// Create a registry with a configuration file path
    pub fn with_config_path<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            static_plugins: HashMap::new(),
            dynamic_plugins: HashMap::new(),
            wasm_plugins: HashMap::new(),
            enabled_states: HashMap::new(),
            config_path: Some(config_path.as_ref().to_path_buf()),
        }
    }

    /// Load registry configuration from TOML file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            LociError::ConfigError(format!("Failed to read config file: {}", e))
        })?;

        let config: RegistryConfig = toml::from_str(&content).map_err(|e| {
            LociError::ConfigError(format!("Failed to parse TOML: {}", e))
        })?;

        // Load each plugin from the configuration
        for plugin_config in config.plugins {
            match plugin_config.plugin_type {
                PluginType::Dynamic => {
                    if let Some(lib_path) = plugin_config.library_path.clone() {
                        self.load_dynamic_plugin_with_config(lib_path, plugin_config)?;
                    }
                }
                PluginType::Wasm => {
                    if let Some(wasm_path) = plugin_config.wasm_path.clone() {
                        self.load_wasm_plugin_with_config(wasm_path, plugin_config)?;
                    }
                }
                PluginType::Static => {
                    // Static plugin - just update enabled state
                    self.enabled_states
                        .insert(plugin_config.name.clone(), plugin_config.enabled);
                }
            }
        }

        self.config_path = Some(path.to_path_buf());
        Ok(())
    }

    /// Save current registry configuration to TOML file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut configs = Vec::new();

        // Collect static plugin configs
        for (name, plugin) in &self.static_plugins {
            configs.push(PluginConfig {
                name: name.clone(),
                version: plugin.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Static,
                library_path: None,
                wasm_path: None,
                wasm_config: None,
                settings: HashMap::new(),
            });
        }

        // Collect dynamic plugin configs
        for (_name, dynamic) in &self.dynamic_plugins {
            configs.push(dynamic.config.clone());
        }

        // Collect WASM plugin configs
        for (name, wasm) in &self.wasm_plugins {
            configs.push(PluginConfig {
                name: name.clone(),
                version: wasm.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Wasm,
                library_path: None,
                wasm_path: Some(wasm.config().wasm_path.clone()),
                wasm_config: Some(wasm.config().clone()),
                settings: HashMap::new(),
            });
        }

        let registry_config = RegistryConfig { plugins: configs };

        let toml_string = toml::to_string_pretty(&registry_config).map_err(|e| {
            LociError::ConfigError(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path.as_ref(), toml_string).map_err(|e| {
            LociError::ConfigError(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Register a static plugin (compiled into the binary)
    pub fn register_static<P: Plugin + 'static>(&mut self, mut plugin: P) -> Result<()> {
        let name = plugin.name().to_string();

        if self.static_plugins.contains_key(&name)
            || self.dynamic_plugins.contains_key(&name)
            || self.wasm_plugins.contains_key(&name)
        {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' already registered",
                name
            )));
        }

        plugin.init()?;
        self.enabled_states.insert(name.clone(), true);
        self.static_plugins.insert(name, Box::new(plugin));

        Ok(())
    }

    /// Load a dynamic plugin from a shared library (hot-swap)
    pub fn load_dynamic_plugin<P: AsRef<Path>>(&mut self, library_path: P) -> Result<()> {
        let config = PluginConfig {
            name: String::new(), // Will be filled after loading
            version: String::new(),
            enabled: true,
            plugin_type: PluginType::Dynamic,
            library_path: Some(library_path.as_ref().to_path_buf()),
            wasm_path: None,
            wasm_config: None,
            settings: HashMap::new(),
        };

        self.load_dynamic_plugin_with_config(library_path, config)
    }

    /// Load a dynamic plugin with configuration
    fn load_dynamic_plugin_with_config<P: AsRef<Path>>(
        &mut self,
        library_path: P,
        mut config: PluginConfig,
    ) -> Result<()> {
        let lib_path = library_path.as_ref();

        // Load the shared library with error handling
        if !lib_path.exists() {
            return Err(LociError::PluginError(format!(
                "Plugin library not found: {}",
                lib_path.display()
            )));
        }
        let library = unsafe {
            Library::new(lib_path).map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to load library '{}': {}",
                    lib_path.display(),
                    e
                ))
            })?
        };

        // Get the plugin constructor function
        let constructor: Symbol<PluginConstructor> = unsafe {
            // Preferred symbol for explicit opaque ABI.
            match library.get(b"create_plugin_v1") {
                Ok(sym) => sym,
                Err(_) => {
                    // Backward compatibility fallback for old plugin naming.
                    library.get(b"create_plugin").map_err(|e| {
                        LociError::PluginError(format!(
                            "Failed to find dynamic constructor symbol ('create_plugin_v1' or 'create_plugin'): {}",
                            e
                        ))
                    })?
                }
            }
        };

        // Create plugin instance and validate opaque payload.
        let plugin_opaque = unsafe { constructor() };
        let mut plugin = unsafe { dynamic_plugin_from_opaque(plugin_opaque) }.ok_or_else(|| {
            LociError::PluginError(
                "Plugin constructor returned invalid plugin payload".to_string(),
            )
        })?;

        if plugin.name().is_empty() {
            return Err(LociError::PluginError(
                "Plugin returned empty name".to_string(),
            ));
        }

        // Initialize the plugin
        plugin.init()?;

        // Update config with plugin metadata
        let name = plugin.name().to_string();
        let version = plugin.version().to_string();

        if config.name.is_empty() {
            config.name = name.clone();
            config.version = version;
        }

        if self.static_plugins.contains_key(&name)
            || self.dynamic_plugins.contains_key(&name)
            || self.wasm_plugins.contains_key(&name)
        {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' already registered",
                name
            )));
        }

        let enabled = config.enabled;
        self.enabled_states.insert(name.clone(), enabled);

        self.dynamic_plugins.insert(
            name,
            DynamicPlugin {
                plugin,
                library: Arc::new(library),
                config,
            },
        );

        Ok(())
    }

    /// Unload a dynamic plugin (hot-swap)
    pub fn unload_dynamic_plugin(&mut self, name: &str) -> Result<()> {
        if let Some(mut dynamic) = self.dynamic_plugins.remove(name) {
            dynamic.plugin.cleanup()?;
            self.enabled_states.remove(name);
            // Library will be dropped automatically
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Dynamic plugin '{}' not found",
                name
            )))
        }
    }

    /// Reload a dynamic plugin (hot-swap)
    pub fn reload_dynamic_plugin(&mut self, name: &str) -> Result<()> {
        if let Some(dynamic) = self.dynamic_plugins.get(name) {
            let config = dynamic.config.clone();
            let lib_path = config
                .library_path
                .clone()
                .ok_or_else(|| LociError::PluginError("No library path".to_string()))?;

            // Unload old version
            self.unload_dynamic_plugin(name)?;

            // Load new version
            self.load_dynamic_plugin_with_config(lib_path, config)?;

            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Dynamic plugin '{}' not found",
                name
            )))
        }
    }

    /// Load a WASM plugin from a file
    pub fn load_wasm_plugin<P: AsRef<Path>>(&mut self, wasm_path: P) -> Result<()> {
        let config = PluginConfig {
            name: String::new(), // Will be filled after loading
            version: String::new(),
            enabled: true,
            plugin_type: PluginType::Wasm,
            library_path: None,
            wasm_path: Some(wasm_path.as_ref().to_path_buf()),
            wasm_config: Some(WasmPluginConfig::default()),
            settings: HashMap::new(),
        };

        self.load_wasm_plugin_with_config(wasm_path, config)
    }

    /// Load a WASM plugin with configuration
    fn load_wasm_plugin_with_config<P: AsRef<Path>>(
        &mut self,
        wasm_path: P,
        mut plugin_config: PluginConfig,
    ) -> Result<()> {
        let wasm_path = wasm_path.as_ref();

        // Build WASM plugin configuration
        let mut wasm_config = plugin_config
            .wasm_config
            .clone()
            .unwrap_or_default();

        // Set WASM path if not already set
        if wasm_config.wasm_path.as_os_str().is_empty() {
            wasm_config.wasm_path = wasm_path.to_path_buf();
        }

        // Load the WASM plugin
        let plugin = WasmPlugin::load(wasm_config)?;

        // Update config with plugin metadata
        let name = plugin.name().to_string();
        let version = plugin.version().to_string();

        if plugin_config.name.is_empty() {
            plugin_config.name = name.clone();
            plugin_config.version = version;
        }

        if self.static_plugins.contains_key(&name)
            || self.dynamic_plugins.contains_key(&name)
            || self.wasm_plugins.contains_key(&name)
        {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' already registered",
                name
            )));
        }

        let enabled = plugin_config.enabled;
        self.enabled_states.insert(name.clone(), enabled);
        self.wasm_plugins.insert(name, plugin);

        Ok(())
    }

    /// Unload a WASM plugin
    pub fn unload_wasm_plugin(&mut self, name: &str) -> Result<()> {
        if let Some(mut wasm) = self.wasm_plugins.remove(name) {
            wasm.cleanup()?;
            self.enabled_states.remove(name);
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "WASM plugin '{}' not found",
                name
            )))
        }
    }

    /// Reload a WASM plugin
    pub fn reload_wasm_plugin(&mut self, name: &str) -> Result<()> {
        if let Some(wasm) = self.wasm_plugins.get(name) {
            let wasm_config = wasm.config().clone();
            let wasm_path = wasm_config.wasm_path.clone();

            let plugin_config = PluginConfig {
                name: name.to_string(),
                version: wasm.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Wasm,
                library_path: None,
                wasm_path: Some(wasm_path.clone()),
                wasm_config: Some(wasm_config),
                settings: HashMap::new(),
            };

            // Unload old version
            self.unload_wasm_plugin(name)?;

            // Load new version
            self.load_wasm_plugin_with_config(wasm_path, plugin_config)?;

            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "WASM plugin '{}' not found",
                name
            )))
        }
    }

    /// Enable a plugin
    pub fn enable(&mut self, name: &str) -> Result<()> {
        if self.static_plugins.contains_key(name)
            || self.dynamic_plugins.contains_key(name)
            || self.wasm_plugins.contains_key(name)
        {
            self.enabled_states.insert(name.to_string(), true);
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Plugin '{}' not found",
                name
            )))
        }
    }

    /// Disable a plugin
    pub fn disable(&mut self, name: &str) -> Result<()> {
        if self.static_plugins.contains_key(name)
            || self.dynamic_plugins.contains_key(name)
            || self.wasm_plugins.contains_key(name)
        {
            self.enabled_states.insert(name.to_string(), false);
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Plugin '{}' not found",
                name
            )))
        }
    }

    /// Check if a plugin is enabled
    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled_states.get(name).copied().unwrap_or(false)
    }

    /// Get a plugin by name (returns None if not enabled)
    pub fn get(&self, name: &str) -> Option<&dyn Plugin> {
        if !self.is_enabled(name) {
            return None;
        }

        if let Some(plugin) = self.static_plugins.get(name) {
            Some(plugin.as_ref())
        } else if let Some(dynamic) = self.dynamic_plugins.get(name) {
            Some(dynamic.plugin.as_ref())
        } else if let Some(wasm) = self.wasm_plugins.get(name) {
            Some(wasm)
        } else {
            None
        }
    }

    /// List all registered plugins
    /// Returns: (name, version, enabled, plugin_type_str)
    pub fn list(&self) -> Vec<(String, String, bool, String)> {
        let mut result = Vec::new();

        for (name, plugin) in &self.static_plugins {
            result.push((
                name.clone(),
                plugin.version().to_string(),
                self.is_enabled(name),
                "static".to_string(),
            ));
        }

        for (name, dynamic) in &self.dynamic_plugins {
            result.push((
                name.clone(),
                dynamic.plugin.version().to_string(),
                self.is_enabled(name),
                "dynamic".to_string(),
            ));
        }

        for (name, wasm) in &self.wasm_plugins {
            result.push((
                name.clone(),
                wasm.version().to_string(),
                self.is_enabled(name),
                "wasm".to_string(),
            ));
        }

        // Keep list order deterministic for CLI/tests and plugin chain predictability.
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// List plugins with richer runtime metadata for integrations.
    pub fn list_detailed(&self) -> Vec<PluginRuntimeInfo> {
        let mut result = Vec::new();

        for (name, plugin) in &self.static_plugins {
            result.push(PluginRuntimeInfo {
                name: name.clone(),
                version: plugin.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Static.as_str().to_string(),
                source: None,
                hot_reloadable: false,
            });
        }

        for (name, dynamic) in &self.dynamic_plugins {
            result.push(PluginRuntimeInfo {
                name: name.clone(),
                version: dynamic.plugin.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Dynamic.as_str().to_string(),
                source: dynamic
                    .config
                    .library_path
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                hot_reloadable: true,
            });
        }

        for (name, wasm) in &self.wasm_plugins {
            result.push(PluginRuntimeInfo {
                name: name.clone(),
                version: wasm.version().to_string(),
                enabled: self.is_enabled(name),
                plugin_type: PluginType::Wasm.as_str().to_string(),
                source: Some(wasm.config().wasm_path.to_string_lossy().to_string()),
                hot_reloadable: true,
            });
        }

        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Query one plugin by name with rich metadata.
    pub fn get_info(&self, name: &str) -> Option<PluginRuntimeInfo> {
        self.list_detailed().into_iter().find(|p| p.name == name)
    }

    /// Check if plugin exists in current registry.
    pub fn contains(&self, name: &str) -> bool {
        self.static_plugins.contains_key(name)
            || self.dynamic_plugins.contains_key(name)
            || self.wasm_plugins.contains_key(name)
    }

    /// List only enabled plugins
    pub fn list_enabled(&self) -> Vec<(String, String, String)> {
        self.list()
            .into_iter()
            .filter(|(name, _, _, _)| self.is_enabled(name))
            .map(|(name, version, _, plugin_type)| (name, version, plugin_type))
            .collect()
    }

    /// Get total count of plugins
    pub fn count(&self) -> usize {
        self.static_plugins.len() + self.dynamic_plugins.len() + self.wasm_plugins.len()
    }

    /// Get count of enabled plugins
    pub fn count_enabled(&self) -> usize {
        self.list_enabled().len()
    }

    /// Get count of dynamic plugins
    pub fn count_dynamic(&self) -> usize {
        self.dynamic_plugins.len()
    }

    /// Get count of WASM plugins
    pub fn count_wasm(&self) -> usize {
        self.wasm_plugins.len()
    }

    /// Unified unload entrypoint for hot-swappable plugin types.
    pub fn unload(&mut self, name: &str) -> Result<()> {
        if self.dynamic_plugins.contains_key(name) {
            return self.unload_dynamic_plugin(name);
        }
        if self.wasm_plugins.contains_key(name) {
            return self.unload_wasm_plugin(name);
        }
        if self.static_plugins.contains_key(name) {
            return Err(LociError::PluginError(format!(
                "Static plugin '{}' cannot be unloaded at runtime (disable it instead)",
                name
            )));
        }
        Err(LociError::PluginError(format!(
            "Plugin '{}' not found",
            name
        )))
    }

    /// Unified reload entrypoint for hot-swappable plugin types.
    pub fn reload(&mut self, name: &str) -> Result<()> {
        if self.dynamic_plugins.contains_key(name) {
            return self.reload_dynamic_plugin(name);
        }
        if self.wasm_plugins.contains_key(name) {
            return self.reload_wasm_plugin(name);
        }
        if self.static_plugins.contains_key(name) {
            return Err(LociError::PluginError(format!(
                "Static plugin '{}' cannot be hot-reloaded",
                name
            )));
        }
        Err(LociError::PluginError(format!(
            "Plugin '{}' not found",
            name
        )))
    }

    /// Apply pre-generate hooks (only enabled plugins)
    pub fn apply_pre_generate(&self, prompt: &str) -> Result<String> {
        let mut result = prompt.to_string();

        for (name, plugin) in &self.static_plugins {
            if self.is_enabled(name) {
                result = plugin.pre_generate(&result)?;
            }
        }

        for (name, dynamic) in &self.dynamic_plugins {
            if self.is_enabled(name) {
                result = dynamic.plugin.pre_generate(&result)?;
            }
        }

        for (name, wasm) in &self.wasm_plugins {
            if self.is_enabled(name) {
                result = wasm.pre_generate(&result)?;
            }
        }

        Ok(result)
    }

    /// Apply logits transformation hooks (only enabled plugins)
    pub fn apply_transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
        for (name, plugin) in &self.static_plugins {
            if self.is_enabled(name) {
                plugin.transform_logits(logits, context)?;
            }
        }

        for (name, dynamic) in &self.dynamic_plugins {
            if self.is_enabled(name) {
                dynamic.plugin.transform_logits(logits, context)?;
            }
        }

        for (name, wasm) in &self.wasm_plugins {
            if self.is_enabled(name) {
                wasm.transform_logits(logits, context)?;
            }
        }

        Ok(())
    }

    /// Apply post-sample hooks (only enabled plugins)
    pub fn apply_post_sample(&self, token_id: i32) -> Result<i32> {
        let mut result = token_id;

        for (name, plugin) in &self.static_plugins {
            if self.is_enabled(name) {
                result = plugin.post_sample(result)?;
            }
        }

        for (name, dynamic) in &self.dynamic_plugins {
            if self.is_enabled(name) {
                result = dynamic.plugin.post_sample(result)?;
            }
        }

        for (name, wasm) in &self.wasm_plugins {
            if self.is_enabled(name) {
                result = wasm.post_sample(result)?;
            }
        }

        Ok(result)
    }

    /// Apply post-generate hooks (only enabled plugins)
    pub fn apply_post_generate(&self, response: &str) -> Result<String> {
        let mut result = response.to_string();

        for (name, plugin) in &self.static_plugins {
            if self.is_enabled(name) {
                result = plugin.post_generate(&result)?;
            }
        }

        for (name, dynamic) in &self.dynamic_plugins {
            if self.is_enabled(name) {
                result = dynamic.plugin.post_generate(&result)?;
            }
        }

        for (name, wasm) in &self.wasm_plugins {
            if self.is_enabled(name) {
                result = wasm.post_generate(&result)?;
            }
        }

        Ok(result)
    }

    /// Apply on-token hooks (only enabled plugins)
    pub fn apply_on_token(&self, token: &str) -> Result<String> {
        let mut result = token.to_string();

        for (name, plugin) in &self.static_plugins {
            if self.is_enabled(name) {
                result = plugin.on_token(&result)?;
            }
        }

        for (name, dynamic) in &self.dynamic_plugins {
            if self.is_enabled(name) {
                result = dynamic.plugin.on_token(&result)?;
            }
        }

        for (name, wasm) in &self.wasm_plugins {
            if self.is_enabled(name) {
                result = wasm.on_token(&result)?;
            }
        }

        Ok(result)
    }

    /// Persist current configuration to file
    pub fn persist(&self) -> Result<()> {
        if let Some(path) = &self.config_path {
            self.save_to_file(path)
        } else {
            Err(LociError::ConfigError(
                "No config path set for registry".to_string(),
            ))
        }
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PluginRegistry {
    fn drop(&mut self) {
        // Cleanup all plugins
        for plugin in self.static_plugins.values_mut() {
            let _ = plugin.cleanup();
        }

        for dynamic in self.dynamic_plugins.values_mut() {
            let _ = dynamic.plugin.cleanup();
        }

        for wasm in self.wasm_plugins.values_mut() {
            let _ = wasm.cleanup();
        }
    }
}

/// Global shared plugin registry
/// Use this for cross-engine plugin sharing
pub type SharedRegistry = Arc<Mutex<PluginRegistry>>;

/// Create a new shared registry
pub fn create_shared_registry() -> SharedRegistry {
    Arc::new(Mutex::new(PluginRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        name: String,
    }

    impl TestPlugin {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("[{}] {}", self.name, prompt))
        }
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = PluginRegistry::new();

        registry.register_static(TestPlugin::new("test1")).unwrap();
        registry.register_static(TestPlugin::new("test2")).unwrap();

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.count_enabled(), 2);

        registry.disable("test1").unwrap();
        assert_eq!(registry.count_enabled(), 1);

        let result = registry.apply_pre_generate("hello").unwrap();
        assert_eq!(result, "[test2] hello");
    }

    #[test]
    fn test_registry_persistence() {
        let temp_path = "test_registry.toml";

        {
            let mut registry = PluginRegistry::with_config_path(temp_path);
            registry.register_static(TestPlugin::new("test1")).unwrap();
            registry.disable("test1").unwrap();
            registry.save_to_file(temp_path).unwrap();
        }

        {
            let mut registry = PluginRegistry::new();
            registry.register_static(TestPlugin::new("test1")).unwrap();
            registry.load_from_file(temp_path).unwrap();

            assert!(!registry.is_enabled("test1"));
        }

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_registry_detailed_info_and_unified_controls_for_static() {
        let mut registry = PluginRegistry::new();
        registry.register_static(TestPlugin::new("test_static")).unwrap();

        let info = registry.get_info("test_static").expect("info should exist");
        assert_eq!(info.plugin_type, "static");
        assert!(!info.hot_reloadable);
        assert!(info.source.is_none());
        assert!(registry.contains("test_static"));

        let unload_err = registry.unload("test_static").unwrap_err();
        assert!(format!("{unload_err}").contains("cannot be unloaded"));

        let reload_err = registry.reload("test_static").unwrap_err();
        assert!(format!("{reload_err}").contains("cannot be hot-reloaded"));
    }
}
