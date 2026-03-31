use anyhow::{bail, Context, Result};
use loci_plugin_api::{CoreComponent, PlatformTrack, PluginManifest};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_FILE_NAME: &str = "manifest.toml";

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

pub fn load_plugin_manifest_file(path: impl AsRef<Path>) -> Result<RegisteredPlugin> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read plugin manifest: {}", path.display()))?;
    let manifest: PluginManifest = toml::from_str(&content)
        .with_context(|| format!("failed to parse plugin manifest: {}", path.display()))?;
    Ok(RegisteredPlugin::new(manifest))
}

pub fn discover_plugin_manifest_files(plugin_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let plugin_dir = plugin_dir.as_ref();
    if !plugin_dir.exists() {
        return Ok(Vec::new());
    }

    if plugin_dir.is_file() {
        if plugin_dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case(MANIFEST_FILE_NAME))
            .unwrap_or(false)
        {
            return Ok(vec![plugin_dir.to_path_buf()]);
        }
        return Ok(Vec::new());
    }

    let mut manifests = Vec::new();
    let root_manifest = plugin_dir.join(MANIFEST_FILE_NAME);
    if root_manifest.exists() {
        manifests.push(root_manifest);
    }

    for entry in fs::read_dir(plugin_dir)
        .with_context(|| format!("failed to scan plugin dir: {}", plugin_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let manifest = path.join(MANIFEST_FILE_NAME);
        if manifest.exists() {
            manifests.push(manifest);
        }
    }

    manifests.sort();
    manifests.dedup();
    Ok(manifests)
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
    use std::fs;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!("loci-plugin-test-{name}-{nanos}"));
        path
    }

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
        assert_eq!(
            manager.plugins_for_model_provider("private-registry").len(),
            1
        );
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

    #[test]
    fn discover_and_load_plugin_manifests_from_directory() {
        let dir = unique_temp_dir("discover");
        fs::create_dir_all(dir.join("plugin-a")).expect("mkdir");
        fs::create_dir_all(dir.join("plugin-b")).expect("mkdir");

        fs::write(
            dir.join("plugin-a").join(MANIFEST_FILE_NAME),
            r#"
name = "plugin-a"
version = "1.0.0"
api_version = "1.0"
"#,
        )
        .expect("write");
        fs::write(
            dir.join("plugin-b").join(MANIFEST_FILE_NAME),
            r#"
name = "plugin-b"
version = "1.0.0"
api_version = "1.0"
"#,
        )
        .expect("write");

        let manifests = discover_plugin_manifest_files(&dir).expect("discover");
        assert_eq!(manifests.len(), 2);

        let plugin = load_plugin_manifest_file(&manifests[0]).expect("load");
        assert!(!plugin.manifest.name.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }
}
