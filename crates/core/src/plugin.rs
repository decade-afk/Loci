use anyhow::{bail, Result};
use loci_plugin_api::{CoreComponent, PlatformTrack, PluginManifest};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct RegisteredPlugin {
    pub manifest: PluginManifest,
}

impl RegisteredPlugin {
    pub fn new(manifest: PluginManifest) -> Self {
        Self { manifest }
    }

    pub fn supports_track(&self, track: PlatformTrack) -> bool {
        self.manifest.supports_track(track)
    }

    pub fn declares_core_rewriter(&self, component: CoreComponent) -> bool {
        self.manifest.declares_core_rewriter(component)
    }
}

#[derive(Default)]
pub struct InMemoryPluginManager {
    plugins: Vec<RegisteredPlugin>,
    plugin_index: BTreeMap<String, usize>,
}

impl crate::core::PluginManager for InMemoryPluginManager {
    fn register(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        if self.plugin_index.contains_key(&plugin.manifest.name) {
            bail!("plugin already registered: {}", plugin.manifest.name);
        }

        let index = self.plugins.len();
        self.plugin_index
            .insert(plugin.manifest.name.clone(), index);
        self.plugins.push(plugin);
        Ok(())
    }

    fn list(&self) -> &[RegisteredPlugin] {
        &self.plugins
    }

    fn get(&self, plugin_name: &str) -> Option<&RegisteredPlugin> {
        self.plugin_index
            .get(plugin_name)
            .and_then(|index| self.plugins.get(*index))
    }

    fn plugins_for_track(&self, track: PlatformTrack) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.supports_track(track))
            .collect()
    }

    fn plugins_for_model_provider(&self, provider: &str) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| {
                plugin
                    .manifest
                    .contributes
                    .model_providers
                    .iter()
                    .any(|candidate| candidate == provider)
            })
            .collect()
    }

    fn plugins_for_core_component(&self, component: CoreComponent) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.declares_core_rewriter(component))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PluginManager;
    use loci_plugin_api::{ContributionPoints, CoreRewriters};

    fn plugin_manifest(name: &str) -> PluginManifest {
        PluginManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            target_tracks: vec![PlatformTrack::AiInfra],
            contributes: ContributionPoints::default(),
            core_rewriters: CoreRewriters::default(),
        }
    }

    #[test]
    fn manager_indexes_plugins_by_track_provider_and_core_component() {
        let mut manager = InMemoryPluginManager::default();
        let mut manifest = plugin_manifest("infra-provider");
        manifest.contributes.model_providers = vec!["private-registry".to_string()];
        manifest.core_rewriters.inference = true;

        manager
            .register(RegisteredPlugin::new(manifest))
            .expect("register");

        assert_eq!(manager.plugins_for_track(PlatformTrack::AiInfra).len(), 1);
        assert_eq!(manager.plugins_for_model_provider("private-registry").len(), 1);
        assert_eq!(
            manager
                .plugins_for_core_component(CoreComponent::Inference)
                .len(),
            1
        );
    }

    #[test]
    fn manager_rejects_duplicate_plugin_names() {
        let mut manager = InMemoryPluginManager::default();
        let plugin = RegisteredPlugin::new(plugin_manifest("duplicate"));

        manager.register(plugin.clone()).expect("first register");
        let err = manager.register(plugin).expect_err("should reject");

        assert!(err.to_string().contains("plugin already registered"));
    }
}
