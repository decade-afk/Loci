//! Plugin system for extending Loci functionality
//!
//! This module provides a plugin architecture that allows third-party applications
//! to integrate with Loci and extend its capabilities.
//!
//! # Plugin System Design
//!
//! Loci's plugin system is designed for **text processing hooks** (not inference backends).
//! Plugins can intercept and modify content at multiple stages:
//!
//! - `pre_generate`: Process user input before inference
//! - `transform_logits`: Modify logits before sampling (advanced)
//! - `post_sample`: Process sampled token before decoding
//! - `on_token`: Process each token during streaming inference
//! - `post_generate`: Process model output after inference completes
//!
//! # Example: Creating a Custom Plugin
//!
//! ```rust
//! use loci::plugin::Plugin;
//! use loci::error::Result;
//!
//! struct MyPlugin;
//!
//! impl Plugin for MyPlugin {
//!     fn name(&self) -> &str { "my_plugin" }
//!     fn version(&self) -> &str { "1.0.0" }
//!
//!     fn pre_generate(&self, prompt: &str) -> Result<String> {
//!         Ok(format!("Enhanced: {}", prompt))
//!     }
//! }
//!
//! // Register with InferenceEngine
//! // engine.plugin_manager_mut().register(MyPlugin)?;
//! ```
//!
//! # Third-Party Plugin Integration
//!
//! External applications can integrate plugins in two ways:
//!
//! ## 1. Static Plugin (Compile-Time)
//! ```rust
//! // In your application
//! use loci::prelude::*;
//! use loci::plugin::Plugin;
//!
//! struct ExternalPlugin { /* ... */ }
//! impl Plugin for ExternalPlugin { /* ... */ }
//!
//! let mut engine = InferenceEngine::new(config)?;
//! engine.plugin_manager_mut().register(ExternalPlugin::new())?;
//! # Ok::<(), loci::error::LociError>(())
//! ```
//!
//! ## 2. Dynamic Plugin (Future: Phase 2)
//! ```ignore
//! // Load plugin from shared library
//! let plugin_manager = engine.plugin_manager_mut();
//! plugin_manager.load_dynamic_plugin("path/to/plugin.so")?;
//! ```

use crate::error::{LociError, Result};
use crate::sampler::LogitsView;
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

    /// Called before text generation (text-level hook)
    fn pre_generate(&self, _prompt: &str) -> Result<String> {
        Ok(_prompt.to_string())
    }

    /// Called to transform logits before sampling (advanced hook)
    ///
    /// This is a powerful hook that allows plugins to:
    /// - Ban specific tokens (set logit to -inf)
    /// - Boost certain tokens (add bias)
    /// - Implement custom grammar constraints
    /// - Apply domain-specific sampling logic
    ///
    /// # Arguments
    /// - `logits`: Zero-copy mutable view of model logits
    /// - `context_tokens`: Previously generated tokens (for context-aware transformations)
    ///
    /// # Example
    /// ```ignore
    /// fn transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
    ///     // Ban profanity tokens
    ///     logits.apply_mask(&[123, 456, 789]);
    ///     // Boost technical terms
    ///     logits.apply_bias(999, 2.0)?;
    ///     Ok(())
    /// }
    /// ```
    fn transform_logits(&self, _logits: &mut LogitsView, _context_tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    /// Called after a token is sampled but before decoding (token-level hook)
    ///
    /// This allows plugins to:
    /// - Substitute tokens
    /// - Implement token-level constraints
    /// - Track token statistics
    fn post_sample(&self, _token_id: i32) -> Result<i32> {
        Ok(_token_id)
    }

    /// Called on each token during streaming (text-level hook)
    fn on_token(&self, _token: &str) -> Result<String> {
        Ok(_token.to_string())
    }

    /// Called after text generation completes (text-level hook)
    fn post_generate(&self, _response: &str) -> Result<String> {
        Ok(_response.to_string())
    }

    /// Cleanup when plugin is unloaded
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Plugin state wrapper
struct PluginEntry {
    plugin: Box<dyn Plugin>,
    enabled: bool,
}

/// Plugin manager for loading and managing plugins
///
/// # Thread Safety
/// PluginManager is NOT thread-safe by default. Wrap in Arc<Mutex<>> for concurrent access.
pub struct PluginManager {
    plugins: HashMap<String, PluginEntry>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    /// Register a plugin
    ///
    /// # Errors
    /// Returns error if:
    /// - Plugin with same name already exists
    /// - Plugin initialization fails
    pub fn register<P: Plugin + 'static>(&mut self, mut plugin: P) -> Result<()> {
        let name = plugin.name().to_string();

        // Check for duplicate
        if self.plugins.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' already registered",
                name
            )));
        }

        plugin.init()?;

        self.plugins.insert(
            name,
            PluginEntry {
                plugin: Box::new(plugin),
                enabled: true,
            },
        );
        Ok(())
    }

    /// Unregister a plugin by name
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        if let Some(mut entry) = self.plugins.remove(name) {
            entry.plugin.cleanup()?;
        }
        Ok(())
    }

    /// Enable a plugin
    pub fn enable(&mut self, name: &str) -> Result<()> {
        if let Some(entry) = self.plugins.get_mut(name) {
            entry.enabled = true;
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Plugin '{}' not found",
                name
            )))
        }
    }

    /// Disable a plugin (keeps it registered but inactive)
    pub fn disable(&mut self, name: &str) -> Result<()> {
        if let Some(entry) = self.plugins.get_mut(name) {
            entry.enabled = false;
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
        self.plugins
            .get(name)
            .map(|e| e.enabled)
            .unwrap_or(false)
    }

    /// Get a plugin by name (returns None if disabled)
    pub fn get(&self, name: &str) -> Option<&Box<dyn Plugin>> {
        self.plugins.get(name).and_then(|entry| {
            if entry.enabled {
                Some(&entry.plugin)
            } else {
                None
            }
        })
    }

    /// List all registered plugins (name, version, enabled)
    pub fn list(&self) -> Vec<(&str, &str, bool)> {
        self.plugins
            .values()
            .map(|entry| (entry.plugin.name(), entry.plugin.version(), entry.enabled))
            .collect()
    }

    /// List only enabled plugins
    pub fn list_enabled(&self) -> Vec<(&str, &str)> {
        self.plugins
            .values()
            .filter(|entry| entry.enabled)
            .map(|entry| (entry.plugin.name(), entry.plugin.version()))
            .collect()
    }

    /// Get count of registered plugins
    pub fn count(&self) -> usize {
        self.plugins.len()
    }

    /// Get count of enabled plugins
    pub fn count_enabled(&self) -> usize {
        self.plugins.values().filter(|e| e.enabled).count()
    }

    /// Apply pre-generation hooks (only enabled plugins)
    pub fn apply_pre_generate(&self, prompt: &str) -> Result<String> {
        let mut result = prompt.to_string();
        for entry in self.plugins.values().filter(|e| e.enabled) {
            result = entry.plugin.pre_generate(&result)?;
        }
        Ok(result)
    }

    /// Apply logits transformation hooks (only enabled plugins)
    ///
    /// This is called before sampling to allow plugins to modify logits.
    /// All enabled plugins are applied in registration order.
    pub fn apply_transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
        for entry in self.plugins.values().filter(|e| e.enabled) {
            entry.plugin.transform_logits(logits, context)?;
        }
        Ok(())
    }

    /// Apply post-sample hooks (only enabled plugins)
    ///
    /// This is called after sampling but before token decoding.
    pub fn apply_post_sample(&self, token_id: i32) -> Result<i32> {
        let mut result = token_id;
        for entry in self.plugins.values().filter(|e| e.enabled) {
            result = entry.plugin.post_sample(result)?;
        }
        Ok(result)
    }

    /// Apply post-generation hooks (only enabled plugins)
    pub fn apply_post_generate(&self, response: &str) -> Result<String> {
        let mut result = response.to_string();
        for entry in self.plugins.values().filter(|e| e.enabled) {
            result = entry.plugin.post_generate(&result)?;
        }
        Ok(result)
    }

    /// Apply token hooks (only enabled plugins)
    pub fn apply_on_token(&self, token: &str) -> Result<String> {
        let mut result = token.to_string();
        for entry in self.plugins.values().filter(|e| e.enabled) {
            result = entry.plugin.on_token(&result)?;
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
        for entry in self.plugins.values_mut() {
            let _ = entry.plugin.cleanup();
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
        assert_eq!(plugins[0].2, true); // enabled by default
    }

    #[test]
    fn test_pre_generate_hook() {
        let mut manager = PluginManager::new();
        manager.register(TestPlugin).unwrap();

        let result = manager.apply_pre_generate("Hello").unwrap();
        assert_eq!(result, "[PRE] Hello");
    }

    #[test]
    fn test_plugin_enable_disable() {
        let mut manager = PluginManager::new();
        manager.register(TestPlugin).unwrap();

        // Initially enabled
        assert_eq!(manager.is_enabled("test"), true);
        let result = manager.apply_pre_generate("Hello").unwrap();
        assert_eq!(result, "[PRE] Hello");

        // Disable plugin
        manager.disable("test").unwrap();
        assert_eq!(manager.is_enabled("test"), false);
        let result = manager.apply_pre_generate("Hello").unwrap();
        assert_eq!(result, "Hello"); // No transformation

        // Re-enable plugin
        manager.enable("test").unwrap();
        assert_eq!(manager.is_enabled("test"), true);
        let result = manager.apply_pre_generate("Hello").unwrap();
        assert_eq!(result, "[PRE] Hello");
    }

    #[test]
    fn test_duplicate_plugin() {
        let mut manager = PluginManager::new();
        manager.register(TestPlugin).unwrap();

        // Try to register same name again
        let result = manager.register(TestPlugin);
        assert!(result.is_err());
    }
}
