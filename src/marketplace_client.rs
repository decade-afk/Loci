//! Marketplace Client Module
//!
//! This module provides core functionality for the Loci project.
//!


use anyhow::{Result, bail, Context};
use ed25519_dalek::{VerifyingKey, Signature, Verifier as Ed25519Verifier};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use tar::Archive;
use zip::ZipArchive;




#[derive(Debug, Clone)]
    /// MarketplaceClientConfig structure
pub struct MarketplaceClientConfig {
    
    pub base_url: String,

    
    pub api_token: Option<String>,

    
    pub plugin_dir: PathBuf,

    
    pub cache_dir: PathBuf,

    
    pub official_public_key: VerifyingKey,
}

// Implementation for Default
impl Default for MarketplaceClientConfig {
    fn default() -> Self {
        // 示例公钥 - 实际使用时应从安全配置或环境变量中读取
        // 这是一个示例公钥，生产环境必须使用真实的官方公钥
        let default_public_key_bytes: [u8; 32] = [
            0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7,
            0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07, 0x3a,
            0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25,
            0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07, 0x51, 0x1a,
        ];
        let official_public_key = VerifyingKey::from_bytes(&default_public_key_bytes)
            .expect("Invalid default public key");

        Self {
            base_url: "https://api.loci.ai/v1".to_string(),
            api_token: None,
            plugin_dir: PathBuf::from("./plugins"),
            cache_dir: std::env::temp_dir().join("loci_marketplace_cache"),
            official_public_key,
        }
    }
}


    /// MarketplaceClient structure
pub struct MarketplaceClient {
    config: MarketplaceClientConfig,
    
    
}

// Implementation for MarketplaceClient
impl MarketplaceClient {
    
    /// new function
    pub fn new(config: MarketplaceClientConfig) -> Result<Self> {
        
        std::fs::create_dir_all(&config.plugin_dir)?;
        std::fs::create_dir_all(&config.cache_dir)?;

        Ok(Self {
            config,
            
        })
    }

    
    
    
    
    
    
    pub async fn search(&self, query: &str, page: usize, limit: usize) -> Result<PluginSearchResult> {
        let url = format!(
            "{}/plugins?q={}&page={}&limit={}",
            self.config.base_url, query, page, limit
        );

        println!("[Marketplace] Searching plugins: {}", query);
        println!("[Marketplace] URL: {}", url);

        
        
        
        
        
        
        

        
        Ok(PluginSearchResult {
            plugins: vec![],
            total: 0,
            page,
            pages: 0,
        })
    }

    
    pub async fn get_plugin(&self, plugin_id: &str) -> Result<PluginDetails> {
        let url = format!("{}/plugins/{}", self.config.base_url, plugin_id);

        println!("[Marketplace] Fetching plugin: {}", plugin_id);

        
        bail!("HTTP client not implemented yet");
    }

    
    
    
    
    pub async fn download(&self, plugin_id: &str, version: Option<&str>) -> Result<PathBuf> {
        // Validate plugin_id
        self.validate_plugin_id(plugin_id)?;

        let version = version.unwrap_or("latest");

        // Validate version string
        self.validate_version(version)?;

        // Build URL (validation prevents path traversal)
        let url = format!(
            "{}/plugins/{}/download?version={}",
            self.config.base_url, plugin_id, version
        );

        println!("[Marketplace] Downloading plugin: {} v{}", plugin_id, version);

        
        
        
        
        
        
        

        
        let cache_path = self.config.cache_dir.join(format!("{}-{}.tar.gz", plugin_id, version));
        

        println!("[Marketplace] Downloaded to: {:?}", cache_path);

        
        std::fs::write(&cache_path, b"")?;

        Ok(cache_path)
    }

    
    
    
    
    
    
    
    pub async fn install(&self, plugin_id: &str, version: Option<&str>) -> Result<()> {
        println!("╔════════════════════════════════════════════════╗");
        println!("║   Installing Plugin: {}            ║", plugin_id);
        println!("╚════════════════════════════════════════════════╝");

        
        println!("\n[1/4] Downloading plugin...");
        let archive_path = self.download(plugin_id, version).await?;

        
        println!("[2/4] Verifying signature...");
        self.verify_signature(&archive_path)?;

        
        println!("[3/4] Extracting plugin...");
        self.extract_plugin(&archive_path, plugin_id)?;

        
        println!("[4/4] Loading plugin...");
        

        println!("\n✅ Plugin installed successfully: {}", plugin_id);
        Ok(())
    }

    
    fn verify_signature(&self, archive_path: &Path) -> Result<()> {
        // 读取插件/归档文件内容
        let plugin_data = std::fs::read(archive_path)
            .with_context(|| format!("Failed to read plugin file: {:?}", archive_path))?;

        // 构建签名文件路径（.sig 扩展名）
        let sig_path = archive_path.with_extension("sig");
        if !sig_path.exists() {
            bail!("Signature file not found: {:?}", sig_path);
        }

        // 读取签名文件
        let sig_data = std::fs::read(&sig_path)
            .with_context(|| format!("Failed to read signature file: {:?}", sig_path))?;

        // 验证签名长度（Ed25519 签名必须是 64 字节）
        if sig_data.len() != 64 {
            bail!(
                "Invalid signature length: expected 64 bytes, got {} bytes",
                sig_data.len()
            );
        }

        // 将签名数据转换为 Ed25519 Signature 类型
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&sig_data);
        let signature = Signature::from_bytes(&sig_bytes);

        // 使用官方公钥验证签名
        self.config
            .official_public_key
            .verify(&plugin_data, &signature)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?;

        println!("  ✅ Signature verified successfully");
        Ok(())
    }

    
    fn extract_plugin(&self, archive_path: &Path, plugin_id: &str) -> Result<()> {
        let target_dir = self.config.plugin_dir.join(plugin_id);
        fs::create_dir_all(&target_dir)?;

        let archive_name = archive_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("archive");

        println!("  📦 Extracting {} to {:?}...", archive_name, target_dir);

        // 根据文件扩展名选择解压方法
        if let Some(ext) = archive_path.extension().and_then(|e| e.to_str()) {
            match ext {
                "gz" => self.extract_tar_gz(archive_path, &target_dir)?,
                "zip" => self.extract_zip(archive_path, &target_dir)?,
                _ => bail!("Unsupported archive format: {}", ext),
            }
        } else {
            bail!("Cannot determine archive format");
        }

        println!("  ✅ Extracted to: {:?}", target_dir);
        Ok(())
    }

    /// Extract tar.gz archive with security checks
    fn extract_tar_gz(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        let file = File::open(archive_path)
            .with_context(|| format!("Failed to open archive: {:?}", archive_path))?;

        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);

        // 验证并解压每个文件
        for entry in archive.entries()? {
            let mut entry = entry
                .with_context(|| "Failed to read archive entry")?;

            let path = entry.path()?;
            let path_str = path.to_string_lossy();

            // 安全检查：防止路径遍历攻击（Zip Slip）
            if path_str.contains("..") || path_str.starts_with('/') {
                bail!("Potentially malicious path in archive: {}", path_str);
            }

            // 构建安全的输出路径
            let output_path = target_dir.join(&path);

            // 确保输出路径在目标目录内
            if !output_path.starts_with(target_dir) {
                bail!("Path traversal attempt detected: {}", path_str);
            }

            // 创建父目录
            if let Some(parent) = output_path.parent() {
                fs::create_dir_all(parent)?;
            }

            // 解压文件
            entry.unpack(&output_path)
                .with_context(|| format!("Failed to extract: {}", path_str))?;

            // 设置安全的文件权限
            if output_path.is_file() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&output_path)?.permissions();
                    perms.set_mode(0o644); // rw-r--r--
                    fs::set_permissions(&output_path, perms)?;
                }
            } else if output_path.is_dir() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&output_path)?.permissions();
                    perms.set_mode(0o755); // rwxr-xr-x
                    fs::set_permissions(&output_path, perms)?;
                }
            }
        }

        Ok(())
    }

    /// Extract zip archive with security checks
    fn extract_zip(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        let file = File::open(archive_path)
            .with_context(|| format!("Failed to open archive: {:?}", archive_path))?;

        let mut archive = ZipArchive::new(file)
            .with_context(|| "Failed to open zip archive")?;

        // 解压每个文件
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .with_context(|| format!("Failed to access file at index {}", i))?;

            let path = file.name();
            let path_str = path.to_string_lossy();

            // 安全检查：防止路径遍历攻击（Zip Slip）
            if path_str.contains("..") || path_str.starts_with('/') || path_str.starts_with('\\') {
                bail!("Potentially malicious path in archive: {}", path_str);
            }

            // 构建安全的输出路径
            let output_path = target_dir.join(path);

            // 确保输出路径在目标目录内
            if !output_path.starts_with(target_dir) {
                bail!("Path traversal attempt detected: {}", path_str);
            }

            if file.is_dir() {
                // 创建目录
                fs::create_dir_all(&output_path)?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&output_path)?.permissions();
                    perms.set_mode(0o755); // rwxr-xr-x
                    fs::set_permissions(&output_path, perms)?;
                }
            } else {
                // 创建父目录
                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                // 解压文件
                let mut output_file = File::create(&output_path)
                    .with_context(|| format!("Failed to create file: {:?}", output_path))?;

                io::copy(&mut file, &mut output_file)
                    .with_context(|| format!("Failed to extract: {}", path_str))?;

                // 设置安全的文件权限
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = fs::metadata(&output_path)?.permissions();
                    perms.set_mode(0o644); // rw-r--r--
                    fs::set_permissions(&output_path, perms)?;
                }
            }
        }

        Ok(())
    }

    
    /// uninstall function
    pub fn uninstall(&self, plugin_id: &str) -> Result<()> {
        // Validate plugin_id to prevent path traversal
        self.validate_plugin_id(plugin_id)?;

        let plugin_dir = self.config.plugin_dir.join(plugin_id);

        if !plugin_dir.exists() {
            bail!("Plugin not found: {}", plugin_id);
        }

        std::fs::remove_dir_all(&plugin_dir)?;
        println!("✅ Plugin uninstalled: {}", plugin_id);

        Ok(())
    }

    
    /// list_installed function
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

    
    pub async fn check_updates(&self) -> Result<Vec<PluginUpdate>> {
        let installed = self.list_installed()?;
        let mut updates = Vec::new();

        println!("[Marketplace] Checking for updates for {} plugins...", installed.len());

        for plugin_id in installed {
            // Validate plugin_id before processing
            if let Err(e) = self.validate_plugin_id(&plugin_id) {
                eprintln!("[Marketplace] Warning: Invalid plugin ID '{}': {}", plugin_id, e);
                continue;
            }

            // 获取当前安装的版本（从 plugin.toml 读取）
            let plugin_dir = self.config.plugin_dir.join(&plugin_id);
            let manifest_path = plugin_dir.join("plugin.toml");

            if !manifest_path.exists() {
                eprintln!("[Marketplace] Warning: No manifest found for {}", plugin_id);
                continue;
            }

            let manifest_content = fs::read_to_string(&manifest_path)?;
            let current_version = self.extract_version_from_manifest(&manifest_content)?;

            // Query remote latest version
            match self.fetch_remote_version(&plugin_id).await {
                Ok(latest_version) => {
                    if self.is_newer_version(&latest_version, &current_version) {
                        updates.push(PluginUpdate {
                            plugin_id: plugin_id.clone(),
                            current_version: current_version.clone(),
                            latest_version: latest_version.clone(),
                        });
                        println!(
                            "  📦 Update available: {} {} -> {}",
                            plugin_id, current_version, latest_version
                        );
                    } else {
                        println!("  ✓ {} is up to date (v{})", plugin_id, current_version);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[Marketplace] Failed to check updates for {}: {}",
                        plugin_id, e
                    );
                    // Continue checking other plugins, don't interrupt the whole process
                }
            }
        }

        println!("[Marketplace] Found {} update(s)", updates.len());
        Ok(updates)
    }

    /// Extract version from manifest
    fn extract_version_from_manifest(&self, manifest_content: &str) -> Result<String> {
        // Simple parsing: find version = "x.x.x" line
        for line in manifest_content.lines() {
            let line = line.trim();
            if line.starts_with("version") {
                if let Some(start) = line.find('"') {
                    if let Some(end) = line.rfind('"') {
                        if start < end {
                            return Ok(line[start + 1..end].to_string());
                        }
                    }
                }
            }
        }
        bail!("Failed to extract version from manifest");
    }

    /// Fetch plugin latest version from remote
    async fn fetch_remote_version(&self, plugin_id: &str) -> Result<String> {
        // Build query URL
        let url = format!(
            "{}/plugins/{}/version",
            self.config.base_url, plugin_id
        );

        // TODO: Actual implementation should use HTTP client (like reqwest) to query remote API
        // Here we use mock implementation, return a simulated version number
        // In production, this should be replaced with actual HTTP request

        // Simulate network delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Return a simulated latest version
        // In production, should parse from API response
        Ok("2.0.0".to_string())
    }

    /// 比较版本号，判断是否为新版本
    fn is_newer_version(&self, latest: &str, current: &str) -> bool {
        // Simple version comparison: major.minor.patch
        let parse_version = |v: &str| -> Vec<u32> {
            v.split('.')
                .map(|p| p.parse().unwrap_or(0))
                .collect()
        };

        let latest_parts = parse_version(latest);
        let current_parts = parse_version(current);

        // Compare version numbers level by level
        for (l, c) in latest_parts.iter().zip(current_parts.iter()) {
            if l > c {
                return true;
            } else if l < c {
                return false;
            }
        }

        // If all corresponding parts are equal, compare length
        latest_parts.len() > current_parts.len()
    }

    /// Validates plugin_id to prevent path traversal and injection attacks.
    ///
    /// Plugin ID must:
    /// - Be non-empty
    /// - Only contain alphanumeric characters, hyphens, and underscores
    /// - Not start or end with a hyphen or underscore
    /// - Not contain consecutive hyphens or underscores
    /// - Not contain ".." (path traversal)
    /// - Not exceed 100 characters
    fn validate_plugin_id(&self, plugin_id: &str) -> Result<()> {
        // Check length
        if plugin_id.is_empty() {
            bail!("Plugin ID cannot be empty");
        }
        if plugin_id.len() > 100 {
            bail!("Plugin ID too long (max 100 characters)");
        }

        // Check for path traversal
        if plugin_id.contains("..") {
            bail!("Plugin ID contains invalid sequence '..'");
        }

        // Check for path separators
        if plugin_id.contains('/') || plugin_id.contains('\\') {
            bail!("Plugin ID contains path separator");
        }

        // Check for null bytes
        if plugin_id.contains('\0') {
            bail!("Plugin ID contains null byte");
        }

        // Check for valid characters
        let valid_chars = plugin_id.chars().all(|c| {
            c.is_alphanumeric() || c == '-' || c == '_'
        });
        if !valid_chars {
            bail!("Plugin ID contains invalid characters (only alphanumeric, '-', '_' allowed)");
        }

        // Check for leading/trailing hyphens or underscores
        if plugin_id.starts_with('-') || plugin_id.starts_with('_') ||
           plugin_id.ends_with('-') || plugin_id.ends_with('_') {
            bail!("Plugin ID cannot start or end with '-' or '_'");
        }

        // Check for consecutive hyphens or underscores
        if plugin_id.contains("--") || plugin_id.contains("__") {
            bail!("Plugin ID cannot contain consecutive '-' or '_'");
        }

        Ok(())
    }

    /// Validates version string to prevent injection attacks.
    ///
    /// Version must:
    /// - Be non-empty
    /// - Not exceed 50 characters
    /// - Not contain path separators or null bytes
    fn validate_version(&self, version: &str) -> Result<()> {
        // Check length
        if version.is_empty() {
            bail!("Version cannot be empty");
        }
        if version.len() > 50 {
            bail!("Version too long (max 50 characters)");
        }

        // Check for path traversal
        if version.contains("..") {
            bail!("Version contains invalid sequence '..'");
        }

        // Check for path separators
        if version.contains('/') || version.contains('\\') {
            bail!("Version contains path separator");
        }

        // Check for null bytes
        if version.contains('\0') {
            bail!("Version contains null byte");
        }

        // Allow "latest" as a special version
        if version == "latest" {
            return Ok(());
        }

        // For semantic version, allow dots, hyphens, and plus signs
        let valid_chars = version.chars().all(|c| {
            c.is_alphanumeric() || c == '.' || c == '-' || c == '+'
        });
        if !valid_chars {
            bail!("Version contains invalid characters");
        }

        Ok(())
    }
}




#[derive(Debug, Deserialize)]
    /// PluginSearchResult structure
pub struct PluginSearchResult {
    pub plugins: Vec<PluginSummary>,
    pub total: usize,
    pub page: usize,
    pub pages: usize,
}


#[derive(Debug, Deserialize)]
    /// PluginSummary structure
pub struct PluginSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: PluginAuthorInfo,
    pub description: String,
    pub kind: String,  
    pub downloads: u64,
    pub rating: f32,
}


#[derive(Debug, Deserialize)]
    /// PluginDetails structure
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


#[derive(Debug, Clone, Deserialize)]
    /// PluginAuthorInfo structure
pub struct PluginAuthorInfo {
    pub name: String,
    pub email: Option<String>,
    pub verified: bool,
}


#[derive(Debug, Clone, Deserialize)]
    /// PluginDependencyInfo structure
pub struct PluginDependencyInfo {
    pub id: String,
    pub version: String,
}


#[derive(Debug, Deserialize)]
    /// PluginDownloadDetails structure
pub struct PluginDownloadDetails {
    pub url: String,
    pub checksum: String,
}


#[derive(Debug)]
    /// PluginUpdate structure
pub struct PluginUpdate {
    pub plugin_id: String,
    pub current_version: String,
    pub latest_version: String,
}



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

        
        std::fs::create_dir_all(client.config.plugin_dir.join("test-plugin")).unwrap();

        let installed = client.list_installed().unwrap();
        assert!(installed.contains(&"test-plugin".to_string()));

        
        std::fs::remove_dir_all(client.config.plugin_dir).ok();
    }
}
