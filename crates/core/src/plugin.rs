use crate::error::{LociError, Result};
use crate::sampler::LogitsView;
use anyhow::Context;
use loci_plugin_api::{PluginKind, PluginManifest, HOST_PLUGIN_API_VERSION};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MANIFEST_FILE_NAME: &str = "manifest.toml";

pub trait SamplingHook: Send + Sync {
    fn transform_logits(
        &self,
        _logits: &mut LogitsView<'_>,
        _context_tokens: &[i32],
    ) -> Result<()> {
        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        Ok(token_id)
    }
}

#[derive(Clone, Default)]
pub struct PluginSamplingRuntime {
    hooks: Vec<Arc<dyn SamplingHook>>,
}

impl PluginSamplingRuntime {
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    pub fn apply_transform_logits(
        &self,
        logits: &mut LogitsView<'_>,
        context_tokens: &[i32],
    ) -> Result<()> {
        for hook in &self.hooks {
            hook.transform_logits(logits, context_tokens)?;
        }
        Ok(())
    }

    pub fn apply_post_sample(&self, token_id: i32) -> Result<i32> {
        let mut token_id = token_id;
        for hook in &self.hooks {
            token_id = hook.post_sample(token_id)?;
        }
        Ok(token_id)
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredPlugin {
    pub manifest: PluginManifest,
    pub manifest_path: PathBuf,
    pub root_dir: PathBuf,
}

impl RegisteredPlugin {
    pub fn is_kind(&self, kind: PluginKind) -> bool {
        self.manifest.kind == kind
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginStatus {
    pub name: String,
    pub version: String,
    pub kind: PluginKind,
    pub auto_activate: bool,
    pub priority: i32,
    pub model_formats: Vec<String>,
    pub hardware_targets: Vec<String>,
    pub features: Vec<String>,
    pub has_native_runtime: bool,
    pub has_wasm_runtime: bool,
    pub is_active: bool,
}

pub fn discover_plugin_manifest_files(root: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let root = root.as_ref();
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut stack = vec![root.to_path_buf()];
    let mut manifests = Vec::new();

    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)
            .with_context(|| format!("failed to read plugin directory: {}", dir.display()))
            .map_err(LociError::from)?
        {
            let entry = entry.map_err(|error| LociError::from(anyhow::Error::from(error)))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|value| value.to_str())
                .map(|value| value.eq_ignore_ascii_case(MANIFEST_FILE_NAME))
                .unwrap_or(false)
            {
                manifests.push(path);
            }
        }
    }

    manifests.sort();
    Ok(manifests)
}

pub fn load_plugin_manifest_file(path: impl AsRef<Path>) -> Result<RegisteredPlugin> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read plugin manifest: {}", path.display()))
        .map_err(LociError::from)?;
    let manifest = toml::from_str::<PluginManifest>(&content)
        .with_context(|| format!("failed to parse plugin manifest: {}", path.display()))
        .map_err(LociError::from)?;

    validate_manifest(&manifest)?;

    let root_dir = path.parent().ok_or_else(|| {
        LociError::ConfigError(format!(
            "plugin manifest does not have a parent directory: {}",
            path.display()
        ))
    })?;

    validate_runtime_path(root_dir, manifest.runtime.library_path.as_deref())?;
    validate_runtime_path(root_dir, manifest.runtime.wasm_path.as_deref())?;

    Ok(RegisteredPlugin {
        manifest,
        manifest_path: path.to_path_buf(),
        root_dir: root_dir.to_path_buf(),
    })
}

fn validate_manifest(manifest: &PluginManifest) -> Result<()> {
    if manifest.name.trim().is_empty() {
        return Err(LociError::ConfigError(
            "plugin manifest name must not be empty".to_string(),
        ));
    }
    if manifest.version.trim().is_empty() {
        return Err(LociError::ConfigError(format!(
            "plugin `{}` version must not be empty",
            manifest.name
        )));
    }
    if manifest.api_version != HOST_PLUGIN_API_VERSION {
        return Err(LociError::ConfigError(format!(
            "plugin `{}` declares api_version `{}`, host supports `{}`",
            manifest.name, manifest.api_version, HOST_PLUGIN_API_VERSION
        )));
    }
    if manifest.runtime.library_path.is_some() && manifest.runtime.wasm_path.is_some() {
        return Err(LociError::ConfigError(format!(
            "plugin `{}` must not declare both native and wasm runtimes",
            manifest.name
        )));
    }
    Ok(())
}

fn validate_runtime_path(root_dir: &Path, declared: Option<&str>) -> Result<()> {
    let Some(declared) = declared else {
        return Ok(());
    };

    if Path::new(declared)
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(LociError::ConfigError(format!(
            "plugin runtime path escapes plugin root: {}",
            declared
        )));
    }

    let resolved = root_dir.join(declared);
    if !resolved.starts_with(root_dir) {
        return Err(LociError::ConfigError(format!(
            "plugin runtime path escapes plugin root: {}",
            resolved.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-plugin-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        dir
    }

    #[test]
    fn discover_manifests_recursively() {
        let dir = unique_temp_dir("discover");
        fs::create_dir_all(dir.join("a").join("b")).expect("mkdir");
        fs::write(
            dir.join("a").join("b").join("manifest.toml"),
            r#"
name = "demo"
version = "0.1.0"
api_version = "1.0"
kind = "hardware_backend"
"#,
        )
        .expect("write manifest");

        let manifests = discover_plugin_manifest_files(&dir).expect("discover");
        assert_eq!(manifests.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn load_manifest_validates_runtime_boundary() {
        let dir = unique_temp_dir("boundary");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join("manifest.toml"),
            r#"
name = "demo"
version = "0.1.0"
api_version = "1.0"
kind = "hardware_backend"

[runtime]
library_path = "../outside.dll"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join("manifest.toml"))
            .expect_err("should reject");
        assert!(err.to_string().contains("escapes plugin root"));

        let _ = fs::remove_dir_all(dir);
    }
}
