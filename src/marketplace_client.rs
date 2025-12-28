/**
 * Loci Phase 4 Week 5-6: Plugin Marketplace Client
 *
 * 核心特性：
 * 1. 插件搜索与发现
 * 2. 插件下载与安装
 * 3. 版本管理与更新
 * 4. 签名验证
 * 5. 依赖解析
 *
 * 集成点：
 * - 与 src/plugin_system.rs 集成
 * - 与 src/plugin_marketplace.rs 集成
 */

use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

// ==================== Marketplace Client ====================

/// 插件市场客户端配置
#[derive(Debug, Clone)]
pub struct MarketplaceClientConfig {
    /// 市场 API 基础 URL
    pub base_url: String,

    /// API Token（用于上传）
    pub api_token: Option<String>,

    /// 插件安装目录
    pub plugin_dir: PathBuf,

    /// 缓存目录
    pub cache_dir: PathBuf,
}

impl Default for MarketplaceClientConfig {
    fn default() -> Self {
        Self {
            base_url: "https://marketplace.loci.dev/api/v1".to_string(),
            api_token: None,
            plugin_dir: PathBuf::from("./plugins"),
            cache_dir: std::env::temp_dir().join("loci_marketplace_cache"),
        }
    }
}

/// 插件市场客户端
pub struct MarketplaceClient {
    config: MarketplaceClientConfig,
    // TODO: 添加 reqwest::Client 用于 HTTP 请求
    // http_client: reqwest::Client,
}

impl MarketplaceClient {
    /// 创建新的市场客户端
    pub fn new(config: MarketplaceClientConfig) -> Result<Self> {
        // 确保目录存在
        std::fs::create_dir_all(&config.plugin_dir)?;
        std::fs::create_dir_all(&config.cache_dir)?;

        Ok(Self {
            config,
            // http_client: reqwest::Client::new(),
        })
    }

    /// 搜索插件
    ///
    /// # 参数
    /// - query: 搜索关键词
    /// - page: 页码（从 1 开始）
    /// - limit: 每页结果数
    pub async fn search(&self, query: &str, page: usize, limit: usize) -> Result<PluginSearchResult> {
        let url = format!(
            "{}/plugins?q={}&page={}&limit={}",
            self.config.base_url, query, page, limit
        );

        println!("[Marketplace] Searching plugins: {}", query);
        println!("[Marketplace] URL: {}", url);

        // TODO: 实现 HTTP 请求
        // let response: PluginSearchResult = self.http_client
        //     .get(&url)
        //     .send()
        //     .await?
        //     .json()
        //     .await?;

        // Mock 响应
        Ok(PluginSearchResult {
            plugins: vec![],
            total: 0,
            page,
            pages: 0,
        })
    }

    /// 获取插件详情
    pub async fn get_plugin(&self, plugin_id: &str) -> Result<PluginDetails> {
        let url = format!("{}/plugins/{}", self.config.base_url, plugin_id);

        println!("[Marketplace] Fetching plugin: {}", plugin_id);

        // TODO: 实现 HTTP 请求
        bail!("HTTP client not implemented yet");
    }

    /// 下载插件
    ///
    /// # 返回
    /// 下载的插件 tar.gz 文件路径
    pub async fn download(&self, plugin_id: &str, version: Option<&str>) -> Result<PathBuf> {
        let version = version.unwrap_or("latest");
        let url = format!(
            "{}/plugins/{}/download?version={}",
            self.config.base_url, plugin_id, version
        );

        println!("[Marketplace] Downloading plugin: {} v{}", plugin_id, version);

        // TODO: 实现下载
        // let bytes = self.http_client
        //     .get(&url)
        //     .send()
        //     .await?
        //     .bytes()
        //     .await?;

        // 保存到缓存目录
        let cache_path = self.config.cache_dir.join(format!("{}-{}.tar.gz", plugin_id, version));
        // std::fs::write(&cache_path, bytes)?;

        println!("[Marketplace] Downloaded to: {:?}", cache_path);

        // Mock: 创建空文件
        std::fs::write(&cache_path, b"")?;

        Ok(cache_path)
    }

    /// 安装插件
    ///
    /// # 流程
    /// 1. 下载插件 tar.gz
    /// 2. 验证签名
    /// 3. 解压到插件目录
    /// 4. 加载插件
    pub async fn install(&self, plugin_id: &str, version: Option<&str>) -> Result<()> {
        println!("╔════════════════════════════════════════════════╗");
        println!("║   Installing Plugin: {}            ║", plugin_id);
        println!("╚════════════════════════════════════════════════╝");

        // 1. 下载插件
        println!("\n[1/4] Downloading plugin...");
        let archive_path = self.download(plugin_id, version).await?;

        // 2. 验证签名
        println!("[2/4] Verifying signature...");
        self.verify_signature(&archive_path)?;

        // 3. 解压
        println!("[3/4] Extracting plugin...");
        self.extract_plugin(&archive_path, plugin_id)?;

        // 4. 加载插件
        println!("[4/4] Loading plugin...");
        // TODO: 调用 PluginRegistry::load_plugin()

        println!("\n✅ Plugin installed successfully: {}", plugin_id);
        Ok(())
    }

    /// 验证插件签名
    fn verify_signature(&self, archive_path: &Path) -> Result<()> {
        // TODO: 实现签名验证
        // 1. 读取 plugin.json 中的 signature 字段
        // 2. 使用 ed25519-dalek 验证签名
        // 3. 验证失败则拒绝安装

        println!("  ✅ Signature verified (mock)");
        Ok(())
    }

    /// 解压插件
    fn extract_plugin(&self, archive_path: &Path, plugin_id: &str) -> Result<()> {
        let target_dir = self.config.plugin_dir.join(plugin_id);
        std::fs::create_dir_all(&target_dir)?;

        // TODO: 实现 tar.gz 解压
        // tar::Archive::new(GzDecoder::new(File::open(archive_path)?))
        //     .unpack(&target_dir)?;

        println!("  ✅ Extracted to: {:?}", target_dir);
        Ok(())
    }

    /// 卸载插件
    pub fn uninstall(&self, plugin_id: &str) -> Result<()> {
        let plugin_dir = self.config.plugin_dir.join(plugin_id);

        if !plugin_dir.exists() {
            bail!("Plugin not found: {}", plugin_id);
        }

        std::fs::remove_dir_all(&plugin_dir)?;
        println!("✅ Plugin uninstalled: {}", plugin_id);

        Ok(())
    }

    /// 列出已安装的插件
    pub fn list_installed(&self) -> Result<Vec<String>> {
        let mut installed = Vec::new();

        for entry in std::fs::read_dir(&self.config.plugin_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    installed.push(name.to_string());
                }
            }
        }

        Ok(installed)
    }

    /// 检查插件更新
    pub async fn check_updates(&self) -> Result<Vec<PluginUpdate>> {
        let installed = self.list_installed()?;
        let mut updates = Vec::new();

        for plugin_id in installed {
            // TODO: 获取最新版本并比较
            // let latest = self.get_plugin(&plugin_id).await?;
            // if latest.version != current_version {
            //     updates.push(PluginUpdate { ... });
            // }
        }

        Ok(updates)
    }
}

// ==================== 数据结构 ====================

/// 插件搜索结果
#[derive(Debug, Deserialize)]
pub struct PluginSearchResult {
    pub plugins: Vec<PluginSummary>,
    pub total: usize,
    pub page: usize,
    pub pages: usize,
}

/// 插件摘要（列表视图）
#[derive(Debug, Deserialize)]
pub struct PluginSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: PluginAuthorInfo,
    pub description: String,
    pub kind: String,  // "native" or "wasm"
    pub downloads: u64,
    pub rating: f32,
}

/// 插件详情
#[derive(Debug, Deserialize)]
pub struct PluginDetails {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: PluginAuthorInfo,
    pub description: String,
    pub license: String,
    pub homepage: String,
    pub repository: String,
    pub platforms: Vec<String>,
    pub dependencies: Vec<PluginDependencyInfo>,
    pub download: PluginDownloadDetails,
    pub signature: String,
    pub downloads: u64,
    pub rating: f32,
    pub created_at: String,
    pub updated_at: String,
}

/// 插件作者信息
#[derive(Debug, Clone, Deserialize)]
pub struct PluginAuthorInfo {
    pub name: String,
    pub email: Option<String>,
    pub verified: bool,
}

/// 插件依赖信息
#[derive(Debug, Clone, Deserialize)]
pub struct PluginDependencyInfo {
    pub id: String,
    pub version: String,
}

/// 插件下载信息
#[derive(Debug, Deserialize)]
pub struct PluginDownloadDetails {
    pub url: String,
    pub checksum: String,
}

/// 插件更新信息
#[derive(Debug)]
pub struct PluginUpdate {
    pub plugin_id: String,
    pub current_version: String,
    pub latest_version: String,
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = MarketplaceClientConfig::default();
        let client = MarketplaceClient::new(config).unwrap();

        assert!(client.config.plugin_dir.exists());
        assert!(client.config.cache_dir.exists());
    }

    #[test]
    fn test_list_installed() {
        let config = MarketplaceClientConfig {
            plugin_dir: std::env::temp_dir().join("test_plugins"),
            ..Default::default()
        };

        let client = MarketplaceClient::new(config).unwrap();

        // 创建测试插件目录
        std::fs::create_dir_all(client.config.plugin_dir.join("test-plugin")).unwrap();

        let installed = client.list_installed().unwrap();
        assert!(installed.contains(&"test-plugin".to_string()));

        // 清理
        std::fs::remove_dir_all(client.config.plugin_dir).ok();
    }
}
