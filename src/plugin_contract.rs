use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const LOCI_PLUGIN_ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginContractKind {
    TextPlugin,
    ToolPlugin,
    ExecutionPolicy,
    ManagementAuthPolicy,
    ModelPullPolicy,
    ModelPullVerifier,
    ServeDispatchPolicy,
    ImageKernel,
    Backend,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginContractManifest {
    pub name: String,
    pub version: String,
    pub kind: PluginContractKind,
    #[serde(default = "default_abi_version")]
    pub abi_version: u32,
    #[serde(default)]
    pub min_host_version: Option<String>,
    #[serde(default)]
    pub max_host_version: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

fn default_abi_version() -> u32 {
    LOCI_PLUGIN_ABI_VERSION
}

fn parse_version_tuple(raw: &str) -> Option<(u64, u64, u64)> {
    let core = raw.split(['-', '+']).next()?.trim();
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

fn build_manifest_candidates(path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        candidates.push(parent.join(format!("{stem}.loci-plugin.json")));
        candidates.push(parent.join(format!("{stem}.loci-plugin.toml")));
        candidates.push(parent.join(format!("{stem}.plugin.json")));
        candidates.push(parent.join(format!("{stem}.plugin.toml")));
    }
    candidates
}

pub fn load_plugin_contract_manifest(path: &Path) -> Result<Option<PluginContractManifest>> {
    for candidate in build_manifest_candidates(path) {
        if !candidate.exists() {
            continue;
        }

        let content = fs::read_to_string(&candidate).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to read plugin manifest '{}': {}",
                candidate.display(),
                e
            ))
        })?;

        let manifest = match candidate
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => serde_json::from_str::<PluginContractManifest>(&content).map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to parse plugin manifest '{}': {}",
                    candidate.display(),
                    e
                ))
            })?,
            "toml" => toml::from_str::<PluginContractManifest>(&content).map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to parse plugin manifest '{}': {}",
                    candidate.display(),
                    e
                ))
            })?,
            other => {
                return Err(LociError::PluginError(format!(
                    "Unsupported plugin manifest format '{}': {}",
                    other,
                    candidate.display()
                )))
            }
        };

        return Ok(Some(manifest));
    }

    Ok(None)
}

pub fn validate_plugin_contract_manifest(
    manifest: &PluginContractManifest,
    expected_kind: PluginContractKind,
) -> Result<()> {
    if manifest.name.trim().is_empty() {
        return Err(LociError::PluginError(
            "Plugin manifest name cannot be empty".to_string(),
        ));
    }
    if manifest.version.trim().is_empty() {
        return Err(LociError::PluginError(format!(
            "Plugin manifest '{}' has empty version",
            manifest.name
        )));
    }
    if manifest.kind != expected_kind {
        return Err(LociError::PluginError(format!(
            "Plugin manifest '{}' has incompatible kind {:?}; expected {:?}",
            manifest.name, manifest.kind, expected_kind
        )));
    }
    if manifest.abi_version != LOCI_PLUGIN_ABI_VERSION {
        return Err(LociError::PluginError(format!(
            "Plugin manifest '{}' requires ABI v{}, host provides v{}",
            manifest.name, manifest.abi_version, LOCI_PLUGIN_ABI_VERSION
        )));
    }

    let host_version = parse_version_tuple(env!("CARGO_PKG_VERSION")).ok_or_else(|| {
        LociError::PluginError("Host package version is not valid semver".to_string())
    })?;

    if let Some(min_version) = &manifest.min_host_version {
        let min_version_tuple = parse_version_tuple(min_version).ok_or_else(|| {
            LociError::PluginError(format!(
                "Plugin manifest '{}' has invalid min_host_version '{}'",
                manifest.name, min_version
            ))
        })?;
        if host_version < min_version_tuple {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' requires host >= {}, current host is {}",
                manifest.name,
                min_version,
                env!("CARGO_PKG_VERSION")
            )));
        }
    }

    if let Some(max_version) = &manifest.max_host_version {
        let max_version_tuple = parse_version_tuple(max_version).ok_or_else(|| {
            LociError::PluginError(format!(
                "Plugin manifest '{}' has invalid max_host_version '{}'",
                manifest.name, max_version
            ))
        })?;
        if host_version > max_version_tuple {
            return Err(LociError::PluginError(format!(
                "Plugin '{}' supports host <= {}, current host is {}",
                manifest.name,
                max_version,
                env!("CARGO_PKG_VERSION")
            )));
        }
    }

    Ok(())
}

pub fn load_and_validate_plugin_contract(
    path: &Path,
    expected_kind: PluginContractKind,
) -> Result<Option<PluginContractManifest>> {
    let manifest = load_plugin_contract_manifest(path)?;
    if let Some(manifest) = &manifest {
        validate_plugin_contract_manifest(manifest, expected_kind)?;
    }
    Ok(manifest)
}

pub fn validate_runtime_plugin_identity(
    manifest: Option<&PluginContractManifest>,
    runtime_name: &str,
    runtime_version: &str,
) -> Result<()> {
    let Some(manifest) = manifest else {
        return Ok(());
    };

    if manifest.name != runtime_name {
        return Err(LociError::PluginError(format!(
            "Plugin manifest name '{}' does not match runtime plugin name '{}'",
            manifest.name, runtime_name
        )));
    }
    if !runtime_version.trim().is_empty() && manifest.version != runtime_version {
        return Err(LociError::PluginError(format!(
            "Plugin manifest version '{}' does not match runtime plugin version '{}'",
            manifest.version, runtime_version
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("loci-plugin-contract-{nonce}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn manifest_sidecar_loads_and_validates() {
        let dir = unique_dir();
        let library = dir.join("demo.dll");
        fs::write(&library, b"binary").expect("write library");
        fs::write(
            dir.join("demo.loci-plugin.json"),
            r#"{
  "name": "demo.plugin",
  "version": "0.1.0",
  "kind": "tool_plugin",
  "abi_version": 1,
  "min_host_version": "0.1.0"
}"#,
        )
        .expect("write manifest");

        let manifest = load_and_validate_plugin_contract(&library, PluginContractKind::ToolPlugin)
            .expect("manifest should validate")
            .expect("manifest should exist");
        assert_eq!(manifest.name, "demo.plugin");

        let _ = fs::remove_file(dir.join("demo.loci-plugin.json"));
        let _ = fs::remove_file(library);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn manifest_rejects_incompatible_kind() {
        let manifest = PluginContractManifest {
            name: "demo".to_string(),
            version: "0.1.0".to_string(),
            kind: PluginContractKind::TextPlugin,
            abi_version: LOCI_PLUGIN_ABI_VERSION,
            min_host_version: None,
            max_host_version: None,
            capabilities: Vec::new(),
        };

        let err = validate_plugin_contract_manifest(&manifest, PluginContractKind::ToolPlugin)
            .expect_err("kind mismatch should fail");
        assert!(err.to_string().contains("incompatible kind"));
    }

    #[test]
    fn runtime_identity_validation_rejects_mismatch() {
        let manifest = PluginContractManifest {
            name: "demo".to_string(),
            version: "0.1.0".to_string(),
            kind: PluginContractKind::TextPlugin,
            abi_version: LOCI_PLUGIN_ABI_VERSION,
            min_host_version: None,
            max_host_version: None,
            capabilities: Vec::new(),
        };

        let err = validate_runtime_plugin_identity(Some(&manifest), "other", "0.1.0")
            .expect_err("name mismatch should fail");
        assert!(err.to_string().contains("does not match"));
    }
}
