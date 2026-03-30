use loci_plugin_api::PluginManifest;
use anyhow::{bail, Result};

#[derive(Debug, Clone)]
pub struct RegisteredPlugin {
    pub manifest: PluginManifest,
}

#[derive(Default)]
pub struct InMemoryPluginManager {
    plugins: Vec<RegisteredPlugin>,
}

impl crate::core::PluginManager for InMemoryPluginManager {
    fn register(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        if self
            .plugins
            .iter()
            .any(|existing| existing.manifest.name == plugin.manifest.name)
        {
            bail!("plugin already registered: {}", plugin.manifest.name);
        }
        self.plugins.push(plugin);
        Ok(())
    }

    fn list(&self) -> &[RegisteredPlugin] {
        &self.plugins
    }
}
