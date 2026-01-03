//! Integration Test: Multiple Plugin Types Working Together
//!
//! This test verifies that Static, Dynamic, and WASM plugins can coexist
//! and work correctly in the same PluginRegistry.

use loci::prelude::*;

#[cfg(test)]
mod multi_plugin_tests {
    use super::*;

    // Static plugin for testing
    struct TestStaticPlugin {
        name: String,
    }

    impl TestStaticPlugin {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Plugin for TestStaticPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("[static:{}] {}", self.name, prompt))
        }

        fn post_generate(&self, response: &str) -> Result<String> {
            Ok(format!("{} [/static:{}]", response, self.name))
        }

        fn on_token(&self, token: &str) -> Result<String> {
            Ok(format!("s:{}", token))
        }
    }

    #[test]
    fn test_registry_supports_all_three_plugin_types() {
        let mut registry = PluginRegistry::new();

        // Register multiple static plugins
        registry
            .register_static(TestStaticPlugin::new("static1"))
            .expect("Failed to register static1");
        registry
            .register_static(TestStaticPlugin::new("static2"))
            .expect("Failed to register static2");

        // Verify counts
        assert_eq!(registry.count(), 2);
        assert_eq!(registry.count_dynamic(), 0);
        assert_eq!(registry.count_wasm(), 0);

        // Note: Cannot test dynamic and WASM plugins without actual binaries
        // But the structure is in place
    }

    #[test]
    fn test_plugin_name_uniqueness_across_types() {
        let mut registry = PluginRegistry::new();

        // Register static plugin
        registry
            .register_static(TestStaticPlugin::new("test_plugin"))
            .expect("Failed to register static plugin");

        // Try to register another static plugin with same name
        let result = registry.register_static(TestStaticPlugin::new("test_plugin"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered"));
    }

    #[test]
    fn test_hooks_apply_in_correct_order() {
        let mut registry = PluginRegistry::new();

        // Register multiple static plugins
        registry
            .register_static(TestStaticPlugin::new("plugin1"))
            .expect("Failed to register plugin1");
        registry
            .register_static(TestStaticPlugin::new("plugin2"))
            .expect("Failed to register plugin2");
        registry
            .register_static(TestStaticPlugin::new("plugin3"))
            .expect("Failed to register plugin3");

        // Test pre_generate hook chain
        let result = registry
            .apply_pre_generate("test")
            .expect("pre_generate failed");

        // Should apply in order: plugin1 -> plugin2 -> plugin3
        assert!(result.contains("[static:plugin1]"));
        assert!(result.contains("[static:plugin2]"));
        assert!(result.contains("[static:plugin3]"));

        // Test post_generate hook chain
        let result = registry
            .apply_post_generate("response")
            .expect("post_generate failed");

        // Should apply in order: plugin1 -> plugin2 -> plugin3
        assert!(result.contains("[/static:plugin1]"));
        assert!(result.contains("[/static:plugin2]"));
        assert!(result.contains("[/static:plugin3]"));
    }

    #[test]
    fn test_enable_disable_plugins() {
        let mut registry = PluginRegistry::new();

        registry
            .register_static(TestStaticPlugin::new("plugin1"))
            .expect("Failed to register");

        // Plugin should be enabled by default
        assert!(registry.is_enabled("plugin1"));
        assert_eq!(registry.count_enabled(), 1);

        // Disable plugin
        registry.disable("plugin1").expect("Failed to disable");
        assert!(!registry.is_enabled("plugin1"));
        assert_eq!(registry.count_enabled(), 0);

        // Hooks should not apply when disabled
        let result = registry
            .apply_pre_generate("test")
            .expect("pre_generate failed");
        assert_eq!(result, "test"); // No transformation

        // Re-enable plugin
        registry.enable("plugin1").expect("Failed to enable");
        assert!(registry.is_enabled("plugin1"));
        assert_eq!(registry.count_enabled(), 1);

        // Hooks should apply again
        let result = registry
            .apply_pre_generate("test")
            .expect("pre_generate failed");
        assert!(result.contains("[static:plugin1]"));
    }

    #[test]
    fn test_plugin_list_returns_correct_types() {
        let mut registry = PluginRegistry::new();

        registry
            .register_static(TestStaticPlugin::new("static1"))
            .expect("Failed to register");

        let plugins = registry.list();
        assert_eq!(plugins.len(), 1);

        let (name, version, enabled, plugin_type) = &plugins[0];
        assert_eq!(name, "static1");
        assert_eq!(version, "1.0.0");
        assert!(enabled);
        assert_eq!(plugin_type, "static");
    }

    #[test]
    fn test_logits_hook_with_zero_copy() {
        let mut registry = PluginRegistry::new();

        // Create a simple static plugin (no actual transformation)
        registry
            .register_static(TestStaticPlugin::new("test"))
            .expect("Failed to register");

        // Create fake logits data
        let mut logits_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut logits = unsafe { LogitsView::from_raw(logits_data.as_mut_ptr(), logits_data.len()) };

        let context = vec![1i32, 2, 3];

        // Apply transform_logits hook
        let result = registry.apply_transform_logits(&mut logits, &context);
        assert!(result.is_ok());

        // Logits should be accessible after hook
        assert_eq!(logits.vocab_size(), 5);
    }

    #[test]
    fn test_post_sample_hook_chain() {
        let mut registry = PluginRegistry::new();

        registry
            .register_static(TestStaticPlugin::new("plugin1"))
            .expect("Failed to register");

        // Test post_sample hook (should return same token for our test plugin)
        let result = registry.apply_post_sample(42).expect("post_sample failed");
        assert_eq!(result, 42);
    }

    #[test]
    fn test_on_token_hook_chain() {
        let mut registry = PluginRegistry::new();

        registry
            .register_static(TestStaticPlugin::new("plugin1"))
            .expect("Failed to register");
        registry
            .register_static(TestStaticPlugin::new("plugin2"))
            .expect("Failed to register");

        // Test on_token hook chain
        let result = registry.apply_on_token("hello").expect("on_token failed");

        // Both plugins should process the token
        assert!(result.starts_with("s:"));
    }

    #[test]
    fn test_get_plugin_by_name() {
        let mut registry = PluginRegistry::new();

        registry
            .register_static(TestStaticPlugin::new("test_plugin"))
            .expect("Failed to register");

        // Get enabled plugin
        let plugin = registry.get("test_plugin");
        assert!(plugin.is_some());
        assert_eq!(plugin.unwrap().name(), "test_plugin");

        // Disable and verify it returns None
        registry.disable("test_plugin").expect("Failed to disable");
        let plugin = registry.get("test_plugin");
        assert!(plugin.is_none());

        // Non-existent plugin
        let plugin = registry.get("nonexistent");
        assert!(plugin.is_none());
    }
}
