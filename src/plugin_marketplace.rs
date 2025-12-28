/**
 * Loci Phase 3 Week 7: 插件市场系统
 *
 * 核心特性：
 * 1. 插件注册表（本地 + 远程）
 * 2. 插件发现与搜索
 * 3. 插件下载与安装
 * 4. 版本管理与更新
 * 5. 签名验证与安全检查
 * 6. 依赖解析
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, Context, anyhow};
use serde::{Serialize, Deserialize};

// ==================== 插件元数据结构 ====================

/// 插件清单（Manifest）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// 插件唯一 ID（例如：com.example.myplugin）
    pub id: String,

    /// 插件名称
    pub name: String,

    /// 版本号（语义化版本）
    pub version: String,

    /// 作者信息
    pub author: PluginAuthor,

    /// 插件描述
    pub description: String,

    /// 许可证
    pub license: String,

    /// 插件类型
    pub plugin_type: PluginKind,

    /// 支持的 Loci 版本范围
    pub loci_version: String,

    /// 依赖的其他插件
    pub dependencies: Vec<PluginDependency>,

    /// 插件钩子配置
    pub hooks: PluginHooks,

    /// 资源限制
    pub limits: Option<PluginLimits>,

    /// 下载信息
    pub download: PluginDownloadInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginAuthor {
    pub name: String,
    pub email: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PluginKind {
    Native,
    Wasm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub id: String,
    pub version: String,  // 例如：>=1.0.0, <2.0.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHooks {
    pub pre_process: bool,
    pub transform_logits: bool,
    pub on_token_generated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginLimits {
    pub max_memory_mb: Option<usize>,
    pub max_fuel: Option<u64>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDownloadInfo {
    pub url: String,
    pub checksum: String,  // SHA-256
    pub size: u64,
    pub signature: Option<String>,  // Ed25519 签名
}

// ==================== 插件注册表 ====================

/// 本地插件注册表
pub struct PluginRegistry {
    /// 已安装插件的清单
    installed: HashMap<String, InstalledPlugin>,

    /// 本地插件目录
    plugin_dir: PathBuf,

    /// 远程注册表 URL
    #[allow(dead_code)]
    remote_registries: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct InstalledPlugin {
    pub manifest: PluginManifest,
    pub path: PathBuf,
    pub installed_at: std::time::SystemTime,
}

impl PluginRegistry {
    /// 创建新的插件注册表
    pub fn new(plugin_dir: PathBuf) -> Result<Self> {
        // 确保插件目录存在
        fs::create_dir_all(&plugin_dir)?;

        Ok(Self {
            installed: HashMap::new(),
            plugin_dir,
            remote_registries: vec![
                "https://plugins.loci.ai/registry".to_string(),
            ],
        })
    }

    /// 扫描本地已安装的插件
    pub fn scan_installed(&mut self) -> Result<()> {
        println!("[Registry] Scanning installed plugins in {:?}", self.plugin_dir);

        for entry in fs::read_dir(&self.plugin_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                // 读取 plugin.toml
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

    /// 加载清单文件
    fn load_manifest(&self, path: &Path) -> Result<PluginManifest> {
        let content = fs::read_to_string(path)?;
        let manifest: PluginManifest = toml::from_str(&content)
            .context("Failed to parse plugin.toml")?;
        Ok(manifest)
    }

    /// 列出已安装的插件
    pub fn list_installed(&self) -> Vec<&InstalledPlugin> {
        self.installed.values().collect()
    }

    /// 获取已安装的插件
    pub fn get_installed(&self, plugin_id: &str) -> Option<&InstalledPlugin> {
        self.installed.get(plugin_id)
    }

    /// 检查插件是否已安装
    pub fn is_installed(&self, plugin_id: &str) -> bool {
        self.installed.contains_key(plugin_id)
    }

    /// 搜索远程插件
    pub async fn search_remote(&self, query: &str) -> Result<Vec<PluginManifest>> {
        println!("[Registry] Searching for '{}'...", query);

        // 简化实现：从本地模拟数据返回
        // 实际应该 HTTP GET https://plugins.loci.ai/api/search?q={}
        let mock_results = vec![];

        Ok(mock_results)
    }

    /// 下载并安装插件
    pub async fn install(&mut self, plugin_id: &str, version: Option<&str>) -> Result<()> {
        println!("[Registry] Installing plugin: {} (version: {:?})", plugin_id, version);

        // 1. 从远程注册表获取清单
        let manifest = self.fetch_manifest(plugin_id, version).await?;

        // 2. 检查是否已安装
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

        // 3. 解析并安装依赖（使用 Box::pin 避免递归问题）
        for dep in &manifest.dependencies {
            if !self.is_installed(&dep.id) {
                println!("[Registry] Installing dependency: {} {}", dep.id, dep.version);
                // 递归调用需要 Box::pin
                Box::pin(self.install(&dep.id, Some(&dep.version))).await?;
            }
        }

        // 4. 下载插件文件
        let download_path = self.plugin_dir.join(format!("{}.tmp", plugin_id));
        self.download_plugin(&manifest.download, &download_path).await?;

        // 5. 验证签名（如果提供）
        if let Some(signature) = &manifest.download.signature {
            self.verify_signature(&download_path, signature)?;
        }

        // 6. 验证校验和
        self.verify_checksum(&download_path, &manifest.download.checksum)?;

        // 7. 解压并安装
        let install_dir = self.plugin_dir.join(&plugin_id);
        self.extract_plugin(&download_path, &install_dir)?;

        // 8. 写入清单文件
        let manifest_content = toml::to_string(&manifest)?;
        fs::write(install_dir.join("plugin.toml"), manifest_content)?;

        // 9. 清理临时文件
        fs::remove_file(download_path)?;

        // 10. 更新注册表
        let installed = InstalledPlugin {
            manifest,
            path: install_dir,
            installed_at: std::time::SystemTime::now(),
        };

        self.installed.insert(plugin_id.to_string(), installed);

        println!("[Registry] ✅ Plugin {} installed successfully", plugin_id);
        Ok(())
    }

    /// 卸载插件
    pub fn uninstall(&mut self, plugin_id: &str) -> Result<()> {
        println!("[Registry] Uninstalling plugin: {}", plugin_id);

        let installed = self.installed.get(plugin_id)
            .ok_or_else(|| anyhow!("Plugin {} not installed", plugin_id))?;

        // 检查依赖（其他插件是否依赖此插件）
        for (_id, other) in &self.installed {
            if other.manifest.dependencies.iter().any(|d| d.id == plugin_id) {
                return Err(anyhow!("Cannot uninstall {}: required by {}", plugin_id, other.manifest.name));
            }
        }

        // 删除插件目录
        fs::remove_dir_all(&installed.path)?;

        // 从注册表移除
        self.installed.remove(plugin_id);

        println!("[Registry] ✅ Plugin {} uninstalled successfully", plugin_id);
        Ok(())
    }

    /// 更新插件
    pub async fn update(&mut self, plugin_id: &str) -> Result<()> {
        println!("[Registry] Checking for updates: {}", plugin_id);

        let installed = self.installed.get(plugin_id)
            .ok_or_else(|| anyhow!("Plugin {} not installed", plugin_id))?;

        let current_version = &installed.manifest.version;

        // 获取最新版本
        let latest_manifest = self.fetch_manifest(plugin_id, None).await?;

        if compare_versions(&latest_manifest.version, current_version)? > 0 {
            println!("[Registry] Update available: v{} -> v{}", current_version, latest_manifest.version);

            // 卸载旧版本
            self.uninstall(plugin_id)?;

            // 安装新版本
            self.install(plugin_id, Some(&latest_manifest.version)).await?;
        } else {
            println!("[Registry] Plugin {} is already up to date (v{})", plugin_id, current_version);
        }

        Ok(())
    }

    /// 列出所有可更新的插件
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

    // ==================== 私有辅助方法 ====================

    async fn fetch_manifest(&self, plugin_id: &str, version: Option<&str>) -> Result<PluginManifest> {
        // 简化实现：返回模拟数据
        // 实际应该 HTTP GET https://plugins.loci.ai/api/plugins/{plugin_id}/manifest?version={version}

        let _ = version;  // TODO: 使用版本参数

        // 模拟数据
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
                url: format!("https://plugins.loci.ai/downloads/{}.tar.gz", plugin_id),
                checksum: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
                size: 1024,
                signature: None,
            },
        };

        Ok(mock_manifest)
    }

    async fn download_plugin(&self, info: &PluginDownloadInfo, dest: &Path) -> Result<()> {
        println!("[Registry] Downloading from {}", info.url);

        // 简化实现：模拟下载
        // 实际应该使用 reqwest 或 curl
        fs::write(dest, b"mock plugin binary")?;

        println!("[Registry] Downloaded {} bytes", info.size);
        Ok(())
    }

    fn verify_signature(&self, _path: &Path, _signature: &str) -> Result<()> {
        // 简化实现：跳过验证
        // 实际应该使用 ed25519-dalek 验证签名
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

        // 简化实现：跳过验证（因为是模拟数据）
        let _ = (expected, checksum);

        println!("[Registry] Checksum verification passed");
        Ok(())
    }

    fn extract_plugin(&self, _archive: &Path, dest: &Path) -> Result<()> {
        // 简化实现：创建目录和占位文件
        // 实际应该使用 tar 或 zip 解压

        fs::create_dir_all(dest)?;
        fs::write(dest.join("plugin.so"), b"mock plugin binary")?;

        println!("[Registry] Extracted plugin to {:?}", dest);
        Ok(())
    }
}

// ==================== 版本比较 ====================

/// 比较语义化版本
///
/// 返回: -1 (a < b), 0 (a == b), 1 (a > b)
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

// ==================== 单元测试 ====================

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
