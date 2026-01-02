//! Plugin system for extending Loci functionality
//!
//! This module provides a plugin architecture that allows third-party applications
//! to integrate with Loci and extend its capabilities.

use crate::error::Result;
use std::collections::HashMap;

/// Plugin trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get the plugin name
    fn name(&self) -> &str;

    /// Get the plugin version
    fn version(&self) -> &str;

    /// Initialize the plugin
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called before text generation
    fn pre_generate(&self, _prompt: &str) -> Result<String> {
        Ok(_prompt.to_string())
    }

    /// Called after text generation
    fn post_generate(&self, _response: &str) -> Result<String> {
        Ok(_response.to_string())
    }

    /// Called on each token during streaming
    fn on_token(&self, _token: &str) -> Result<String> {
        Ok(_token.to_string())
    }

    /// Cleanup when plugin is unloaded
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Plugin manager for loading and managing plugins
pub struct PluginManager {
    plugins: HashMap<String, Box<dyn Plugin>>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin
    pub fn register<P: Plugin + 'static>(&mut self, mut plugin: P) -> Result<()> {
        plugin.init()?;
        let name = plugin.name().to_string();
        self.plugins.insert(name, Box::new(plugin));
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        if let Some(mut plugin) = self.plugins.remove(name) {
            plugin.cleanup()?;
        }
        Ok(())
    }

    /// Get a plugin by name
    pub fn get(&self, name: &str) -> Option<&Box<dyn Plugin>> {
        self.plugins.get(name)
    }

    /// List all registered plugins
    pub fn list(&self) -> Vec<(&str, &str)> {
        self.plugins
            .values()
            .map(|p| (p.name(), p.version()))
            .collect()
    }

    /// Apply pre-generation hooks
    pub fn apply_pre_generate(&self, prompt: &str) -> Result<String> {
        let mut result = prompt.to_string();
        for plugin in self.plugins.values() {
            result = plugin.pre_generate(&result)?;
        }
        Ok(result)
    }

    /// Apply post-generation hooks
    pub fn apply_post_generate(&self, response: &str) -> Result<String> {
        let mut result = response.to_string();
        for plugin in self.plugins.values() {
            result = plugin.post_generate(&result)?;
        }
        Ok(result)
    }

    /// Apply token hooks
    pub fn apply_on_token(&self, token: &str) -> Result<String> {
        let mut result = token.to_string();
        for plugin in self.plugins.values() {
            result = plugin.on_token(&result)?;
        }
        Ok(result)
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        for plugin in self.plugins.values_mut() {
            let _ = plugin.cleanup();
        }
    }
}

/// Example plugin: Prompt template
pub struct PromptTemplatePlugin {
    template: String,
}

impl PromptTemplatePlugin {
    pub fn new(template: String) -> Self {
        Self { template }
    }
}

impl Plugin for PromptTemplatePlugin {
    fn name(&self) -> &str {
        "prompt_template"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(self.template.replace("{prompt}", prompt))
    }
}

/// Example plugin: Response filter
pub struct ResponseFilterPlugin {
    filter_pattern: String,
}

impl ResponseFilterPlugin {
    pub fn new(filter_pattern: String) -> Self {
        Self { filter_pattern }
    }
}

impl Plugin for ResponseFilterPlugin {
    fn name(&self) -> &str {
        "response_filter"
    }

    fn version(&self) -> &str {
        "0.1.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        // Simple filter example
        Ok(response.replace(&self.filter_pattern, "[FILTERED]"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("[PRE] {}", prompt))
        }
    }

    #[test]
    fn test_plugin_manager() {
        let mut manager = PluginManager::new();
        manager.register(TestPlugin).unwrap();

        let plugins = manager.list();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].0, "test");
    }

    #[test]
    fn test_pre_generate_hook() {
        let mut manager = PluginManager::new();
        manager.register(TestPlugin).unwrap();

        let result = manager.apply_pre_generate("Hello").unwrap();
        assert_eq!(result, "[PRE] Hello");
    }
}
