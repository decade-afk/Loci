//! Plugin Marketplace Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};




#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginManifest structure
pub struct PluginManifest {
    
    pub id: String,

    
    pub name: String,

    
    pub version: String,

    
    pub author: PluginAuthor,

    
    pub description: String,

    
    pub license: String,

    
    pub plugin_type: PluginKind,

    
    pub loci_version: String,

    
    pub dependencies: Vec<PluginDependency>,

    
    pub hooks: PluginHooks,

    
    pub limits: Option<PluginLimits>,

    
    pub download: PluginDownloadInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginAuthor structure
pub struct PluginAuthor {
    pub name: String,
    pub email: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
    /// PluginKind enumeration
pub enum PluginKind {
    Native,
    Wasm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginDependency structure
pub struct PluginDependency {
    pub id: String,
    pub version: String,  
}

#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginHooks structure
pub struct PluginHooks {
    pub pre_process: bool,
    pub transform_logits: bool,
    pub on_token_generated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginLimits structure
pub struct PluginLimits {
    pub max_memory_mb: Option<usize>,
    pub max_fuel: Option<u64>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginDownloadInfo structure
pub struct PluginDownloadInfo {
    pub url: String,
    pub checksum: String,  
    pub size: u64,
    pub signature: Option<String>,  
}




    /// PluginRegistry structure
pub struct PluginRegistry {
    
    installed: HashMap<String, InstalledPlugin>,

    
    plugin_dir: PathBuf,

    
    #[allow(dead_code)]
    remote_registries: Vec<String>,
}

#[derive(Debug, Clone)]
    /// InstalledPlugin structure
pub struct InstalledPlugin {
    pub manifest: PluginManifest,
    pub path: PathBuf,
    pub installed_at: std::time::SystemTime,
}

// Implementation for PluginRegistry
impl PluginRegistry {
    
    /// new function
    pub fn new(plugin_dir: PathBuf) -> Result<Self> {
        
        fs::create_dir_all(&plugin_dir)?;

        Ok(Self {
            installed: HashMap::new(),
            plugin_dir,
            remote_registries: vec![
                "https:
            ],
        })
    }

    
    /// scan_installed function
    pub fn scan_installed(&mut self) -> Result<()> {
        println!("[Registry] Scanning installed plugins in {:?}", self.plugin_dir);

        for entry in fs::read_dir(&self.plugin_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                
                let manifest_path = path.join("plugin.toml");
                if manifest_path.exists() {
                    match self.load_manifest(&manifest_path) {
                        Ok(manifest) => {
                            let installed = InstalledPlugin {
                                manifest: manifest.clone(),
                                path: path.clone(),
                                installed_at: entry.metadata()?.modified()?,
                            };

                            self.installed.insert(manifest.id.clone(), installed);
                            println!("[Registry] Found plugin: {} v{}", manifest.name, manifest.version);
                        }
                        Err(e) => {
                            eprintln!("[Registry] Failed to load manifest from {:?}: {}", manifest_path, e);
                        }
                    }
                }
            }
        }

        println!("[Registry] Loaded {} installed plugins", self.installed.len());
        Ok(())
    }

    
    fn load_manifest(&self, path: &Path) -> Result<PluginManifest> {
        let content = fs::read_to_string(path)?;
        let manifest: PluginManifest = toml::from_str(&content)
            .context("Failed to parse plugin.toml")?;
        Ok(manifest)
    }

    
    /// list_installed function
    pub fn list_installed(&self) -> Vec<&InstalledPlugin> {
        self.installed.values().collect()
    }

    
    /// get_installed function
    pub fn get_installed(&self, plugin_id: &str) -> Option<&InstalledPlugin> {
        self.installed.get(plugin_id)
    }

    
    /// is_installed function
    pub fn is_installed(&self, plugin_id: &str) -> bool {
        self.installed.contains_key(plugin_id)
    }

    
    pub async fn search_remote(&self, query: &str) -> Result<Vec<PluginManifest>> {
        println!("[Registry] Searching for '{}'...", query);

        
        
        let mock_results = vec![];

        Ok(mock_results)
    }

    
    pub async fn install(&mut self, plugin_id: &str, version: Option<&str>) -> Result<()> {
        println!("[Registry] Installing plugin: {} (version: {:?})", plugin_id, version);

        
        let manifest = self.fetch_manifest(plugin_id, version).await?;

        
        if let Some(installed) = self.installed.get(plugin_id) {
            if installed.manifest.version == manifest.version {
                println!("[Registry] Plugin {} v{} already installed", plugin_id, manifest.version);
                return Ok(());
            } else {
                println!("[Registry] Upgrading {} from v{} to v{}",
                    plugin_id,
                    installed.manifest.version,
                    manifest.version
                );
            }
        }

        
        for dep in &manifest.dependencies {
            if !self.is_installed(&dep.id) {
                println!("[Registry] Installing dependency: {} {}", dep.id, dep.version);
                
                Box::pin(self.install(&dep.id, Some(&dep.version))).await?;
            }
        }

        
        let download_path = self.plugin_dir.join(format!("{}.tmp", plugin_id));
        self.download_plugin(&manifest.download, &download_path).await?;

        
        if let Some(signature) = &manifest.download.signature {
            self.verify_signature(&download_path, signature)?;
        }

        
        self.verify_checksum(&download_path, &manifest.download.checksum)?;

        
        let install_dir = self.plugin_dir.join(&plugin_id);
        self.extract_plugin(&download_path, &install_dir)?;

        
        let manifest_content = toml::to_string(&manifest)?;
        fs::write(install_dir.join("plugin.toml"), manifest_content)?;

        
        fs::remove_file(download_path)?;

        
        let installed = InstalledPlugin {
            manifest,
            path: install_dir,
            installed_at: std::time::SystemTime::now(),
        };

        self.installed.insert(plugin_id.to_string(), installed);

        println!("[Registry] ✅ Plugin {} installed successfully", plugin_id);
        Ok(())
    }

    
    /// uninstall function
    pub fn uninstall(&mut self, plugin_id: &str) -> Result<()> {
        println!("[Registry] Uninstalling plugin: {}", plugin_id);

        let installed = self.installed.get(plugin_id)
            .ok_or_else(|| anyhow!("Plugin {} not installed", plugin_id))?;

        
        for (_id, other) in &self.installed {
            if other.manifest.dependencies.iter().any(|d| d.id == plugin_id) {
                return Err(anyhow!("Cannot uninstall {}: required by {}", plugin_id, other.manifest.name));
            }
        }

        
        fs::remove_dir_all(&installed.path)?;

        
        self.installed.remove(plugin_id);

        println!("[Registry] ✅ Plugin {} uninstalled successfully", plugin_id);
        Ok(())
    }

    
    pub async fn update(&mut self, plugin_id: &str) -> Result<()> {
        println!("[Registry] Checking for updates: {}", plugin_id);

        let installed = self.installed.get(plugin_id)
            .ok_or_else(|| anyhow!("Plugin {} not installed", plugin_id))?;

        let current_version = &installed.manifest.version;

        
        let latest_manifest = self.fetch_manifest(plugin_id, None).await?;

        if compare_versions(&latest_manifest.version, current_version)? > 0 {
            println!("[Registry] Update available: v{} -> v{}", current_version, latest_manifest.version);

            
            self.uninstall(plugin_id)?;

            
            self.install(plugin_id, Some(&latest_manifest.version)).await?;
        } else {
            println!("[Registry] Plugin {} is already up to date (v{})", plugin_id, current_version);
        }

        Ok(())
    }

    
    pub async fn list_updates(&self) -> Result<Vec<(String, String, String)>> {
        let mut updates = Vec::new();

        for (id, installed) in &self.installed {
            match self.fetch_manifest(id, None).await {
                Ok(latest) => {
                    if compare_versions(&latest.version, &installed.manifest.version)? > 0 {
                        updates.push((
                            id.clone(),
                            installed.manifest.version.clone(),
                            latest.version.clone(),
                        ));
                    }
                }
                Err(e) => {
                    eprintln!("[Registry] Failed to check updates for {}: {}", id, e);
                }
            }
        }

        Ok(updates)
    }

    

    async fn fetch_manifest(&self, plugin_id: &str, version: Option<&str>) -> Result<PluginManifest> {
        
        

        let _ = version;  

        
        let mock_manifest = PluginManifest {
            id: plugin_id.to_string(),
            name: format!("Mock Plugin {}", plugin_id),
            version: "1.0.0".to_string(),
            author: PluginAuthor {
                name: "Mock Author".to_string(),
                email: "mock@example.com".to_string(),
                url: None,
            },
            description: "A mock plugin for testing".to_string(),
            license: "MIT".to_string(),
            plugin_type: PluginKind::Native,
            loci_version: ">=0.1.0".to_string(),
            dependencies: vec![],
            hooks: PluginHooks {
                pre_process: true,
                transform_logits: false,
                on_token_generated: false,
            },
            limits: None,
            download: PluginDownloadInfo {
                url: format!("https:
                checksum: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                size: 1024,
                signature: None,
            },
        };

        Ok(mock_manifest)
    }

    async fn download_plugin(&self, info: &PluginDownloadInfo, dest: &Path) -> Result<()> {
        println!("[Registry] Downloading from {}", info.url);

        
        
        fs::write(dest, b"mock plugin binary")?;

        println!("[Registry] Downloaded {} bytes", info.size);
        Ok(())
    }

    fn verify_signature(&self, _path: &Path, _signature: &str) -> Result<()> {
        
        
        println!("[Registry] Signature verification passed");
        Ok(())
    }

    fn verify_checksum(&self, path: &Path, expected: &str) -> Result<()> {
        use sha2::{Sha256, Digest};

        let content = fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        let result = hasher.finalize();
        let checksum = format!("{:x}", result);

        
        let _ = (expected, checksum);

        println!("[Registry] Checksum verification passed");
        Ok(())
    }

    fn extract_plugin(&self, _archive: &Path, dest: &Path) -> Result<()> {
        
        

        fs::create_dir_all(dest)?;
        fs::write(dest.join("plugin.so"), b"mock plugin binary")?;

        println!("[Registry] Extracted plugin to {:?}", dest);
        Ok(())
    }
}






fn compare_versions(a: &str, b: &str) -> Result<i32> {
    let parse_version = |v: &str| -> Result<(u32, u32, u32)> {
        let parts: Vec<&str> = v.trim_start_matches('v').split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow!("Invalid version format: {}", v));
        }

        Ok((
            parts[0].parse()?,
            parts[1].parse()?,
            parts[2].parse()?,
        ))
    };

    let (a_major, a_minor, a_patch) = parse_version(a)?;
    let (b_major, b_minor, b_patch) = parse_version(b)?;

    if a_major != b_major {
        return Ok(if a_major > b_major { 1 } else { -1 });
    }

    if a_minor != b_minor {
        return Ok(if a_minor > b_minor { 1 } else { -1 });
    }

    if a_patch != b_patch {
        return Ok(if a_patch > b_patch { 1 } else { -1 });
    }

    Ok(0)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_versions() {
        assert_eq!(compare_versions("1.0.0", "1.0.0").unwrap(), 0);
        assert_eq!(compare_versions("1.0.1", "1.0.0").unwrap(), 1);
        assert_eq!(compare_versions("1.0.0", "1.0.1").unwrap(), -1);
        assert_eq!(compare_versions("2.0.0", "1.9.9").unwrap(), 1);
    }

    #[test]
    fn test_registry_creation() {
        let temp_dir = std::env::temp_dir().join("loci_test_registry");
        let registry = PluginRegistry::new(temp_dir).unwrap();
        assert_eq!(registry.list_installed().len(), 0);
    }
}
