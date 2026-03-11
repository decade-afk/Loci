//! Persistent registry for dynamic policy plugin paths and active selection.

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicPolicyRegistryFile {
    #[serde(default)]
    pub active: Option<String>,
    #[serde(default)]
    pub plugins: Vec<PathBuf>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub prefixes: Vec<String>,
}

pub struct DynamicPolicyRegistry {
    file: DynamicPolicyRegistryFile,
    config_path: Option<PathBuf>,
}

impl DynamicPolicyRegistry {
    pub fn new() -> Self {
        Self {
            file: DynamicPolicyRegistryFile::default(),
            config_path: None,
        }
    }

    pub fn with_config_path<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            file: DynamicPolicyRegistryFile::default(),
            config_path: Some(config_path.as_ref().to_path_buf()),
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            LociError::ConfigError(format!(
                "Failed to read policy registry '{}': {}",
                path.display(),
                e
            ))
        })?;
        let file: DynamicPolicyRegistryFile = toml::from_str(&content).map_err(|e| {
            LociError::ConfigError(format!("Failed to parse policy registry TOML: {e}"))
        })?;
        self.file = file;
        self.config_path = Some(path.to_path_buf());
        Ok(())
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = self.file.clone();
        file.plugins.sort();
        file.plugins.dedup();
        let content = toml::to_string_pretty(&file).map_err(|e| {
            LociError::ConfigError(format!("Failed to serialize policy registry: {e}"))
        })?;
        std::fs::write(path.as_ref(), content).map_err(|e| {
            LociError::ConfigError(format!(
                "Failed to write policy registry '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Ok(())
    }

    pub fn persist(&self) -> Result<()> {
        let path = self.config_path.as_ref().ok_or_else(|| {
            LociError::ConfigError("No policy registry path configured".to_string())
        })?;
        self.save_to_file(path)
    }

    pub fn active(&self) -> Option<&str> {
        self.file.active.as_deref()
    }

    pub fn set_active(&mut self, active: Option<String>) {
        self.file.active = active;
    }

    pub fn plugins(&self) -> &[PathBuf] {
        &self.file.plugins
    }

    pub fn scope(&self) -> Option<&str> {
        self.file.scope.as_deref()
    }

    pub fn set_scope(&mut self, scope: Option<String>) {
        self.file.scope = scope;
    }

    pub fn prefixes(&self) -> &[String] {
        &self.file.prefixes
    }

    pub fn set_prefixes(&mut self, prefixes: Vec<String>) {
        self.file.prefixes = prefixes;
    }

    pub fn add_plugin_path<P: Into<PathBuf>>(&mut self, path: P) {
        let path = path.into();
        if !self.file.plugins.iter().any(|existing| existing == &path) {
            self.file.plugins.push(path);
        }
    }

    pub fn remove_plugin_path(&mut self, path: &Path) -> bool {
        let original_len = self.file.plugins.len();
        self.file.plugins.retain(|existing| existing != path);
        original_len != self.file.plugins.len()
    }
}

impl Default for DynamicPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_adds_and_removes_paths() {
        let mut registry = DynamicPolicyRegistry::new();
        registry.add_plugin_path("a.dll");
        registry.add_plugin_path("a.dll");
        registry.add_plugin_path("b.dll");
        assert_eq!(registry.plugins().len(), 2);
        assert!(registry.remove_plugin_path(Path::new("a.dll")));
        assert_eq!(registry.plugins(), &[PathBuf::from("b.dll")]);
    }

    #[test]
    fn registry_tracks_active_name() {
        let mut registry = DynamicPolicyRegistry::new();
        registry.set_active(Some("policy.one".to_string()));
        assert_eq!(registry.active(), Some("policy.one"));
        registry.set_active(None);
        assert_eq!(registry.active(), None);
    }

    #[test]
    fn registry_tracks_scope_and_prefixes() {
        let mut registry = DynamicPolicyRegistry::new();
        registry.set_scope(Some("custom".to_string()));
        registry.set_prefixes(vec!["/tools".to_string(), "/browser".to_string()]);
        assert_eq!(registry.scope(), Some("custom"));
        assert_eq!(
            registry.prefixes(),
            &["/tools".to_string(), "/browser".to_string()]
        );
    }
}
