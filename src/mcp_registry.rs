//! Persistent MCP server registry for pluginized server management.
//!
//! This mirrors the extension-registry style used by plugin/session-store
//! components: users can register multiple MCP servers and load them by name.

use crate::error::{LociError, Result};
use crate::mcp::{McpStdioServerConfig, DEFAULT_MCP_PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn default_enabled() -> bool {
    true
}

/// Persisted MCP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub working_directory: Option<PathBuf>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub protocol_version: Option<String>,
    #[serde(default)]
    pub client_name: Option<String>,
    #[serde(default)]
    pub tool_prefix: Option<String>,
}

impl McpServerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(LociError::ConfigError(
                "MCP server name cannot be empty".to_string(),
            ));
        }
        if self.command.trim().is_empty() {
            return Err(LociError::ConfigError(format!(
                "MCP server '{}' command cannot be empty",
                self.name
            )));
        }
        Ok(())
    }

    pub fn to_stdio_config(&self) -> McpStdioServerConfig {
        let mut cfg = McpStdioServerConfig::new(self.name.clone(), self.command.clone());
        cfg.args = self.args.clone();
        cfg.working_directory = self.working_directory.clone();
        cfg.env = self.env.clone();
        cfg.protocol_version = self
            .protocol_version
            .clone()
            .unwrap_or_else(|| DEFAULT_MCP_PROTOCOL_VERSION.to_string());
        cfg.client_name = self
            .client_name
            .clone()
            .unwrap_or_else(|| "loci".to_string());
        cfg.tool_prefix = self.tool_prefix.clone();
        cfg
    }
}

impl From<McpStdioServerConfig> for McpServerConfig {
    fn from(value: McpStdioServerConfig) -> Self {
        Self {
            name: value.server_name,
            command: value.command,
            args: value.args,
            enabled: true,
            working_directory: value.working_directory,
            env: value.env,
            protocol_version: Some(value.protocol_version),
            client_name: Some(value.client_name),
            tool_prefix: value.tool_prefix,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct McpRegistryFile {
    #[serde(default)]
    servers: Vec<McpServerConfig>,
}

/// MCP server registry with TOML persistence.
pub struct McpServerRegistry {
    servers: HashMap<String, McpServerConfig>,
    config_path: Option<PathBuf>,
}

impl McpServerRegistry {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            config_path: None,
        }
    }

    pub fn with_config_path<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            servers: HashMap::new(),
            config_path: Some(config_path.as_ref().to_path_buf()),
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            LociError::ConfigError(format!("Failed to read MCP registry '{}': {}", path.display(), e))
        })?;
        let file: McpRegistryFile = toml::from_str(&content)
            .map_err(|e| LociError::ConfigError(format!("Failed to parse MCP registry TOML: {e}")))?;

        self.servers.clear();
        for server in file.servers {
            server.validate()?;
            self.servers.insert(server.name.clone(), server);
        }
        self.config_path = Some(path.to_path_buf());
        Ok(())
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut servers = self.servers.values().cloned().collect::<Vec<_>>();
        servers.sort_by(|a, b| a.name.cmp(&b.name));
        let file = McpRegistryFile { servers };
        let content = toml::to_string_pretty(&file).map_err(|e| {
            LociError::ConfigError(format!("Failed to serialize MCP registry: {e}"))
        })?;
        std::fs::write(path.as_ref(), content).map_err(|e| {
            LociError::ConfigError(format!(
                "Failed to write MCP registry '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Ok(())
    }

    pub fn persist(&self) -> Result<()> {
        let path = self.config_path.as_ref().ok_or_else(|| {
            LociError::ConfigError("No MCP registry path configured".to_string())
        })?;
        self.save_to_file(path)
    }

    pub fn upsert(&mut self, server: McpServerConfig) -> Result<()> {
        server.validate()?;
        self.servers.insert(server.name.clone(), server);
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Result<()> {
        if self.servers.remove(name).is_some() {
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "MCP server '{}' not found",
                name
            )))
        }
    }

    pub fn enable(&mut self, name: &str) -> Result<()> {
        let entry = self.servers.get_mut(name).ok_or_else(|| {
            LociError::PluginError(format!("MCP server '{}' not found", name))
        })?;
        entry.enabled = true;
        Ok(())
    }

    pub fn disable(&mut self, name: &str) -> Result<()> {
        let entry = self.servers.get_mut(name).ok_or_else(|| {
            LociError::PluginError(format!("MCP server '{}' not found", name))
        })?;
        entry.enabled = false;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&McpServerConfig> {
        self.servers.get(name)
    }

    pub fn list(&self) -> Vec<&McpServerConfig> {
        let mut list = self.servers.values().collect::<Vec<_>>();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn list_enabled(&self) -> Vec<&McpServerConfig> {
        let mut list = self
            .servers
            .values()
            .filter(|s| s.enabled)
            .collect::<Vec<_>>();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }
}

impl Default for McpServerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_upsert_and_list() {
        let mut registry = McpServerRegistry::new();
        registry
            .upsert(McpServerConfig {
                name: "fs".to_string(),
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "@modelcontextprotocol/server-filesystem".to_string()],
                enabled: true,
                working_directory: None,
                env: HashMap::new(),
                protocol_version: None,
                client_name: None,
                tool_prefix: Some("mcp.fs.".to_string()),
            })
            .unwrap();
        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.list_enabled().len(), 1);
    }

    #[test]
    fn registry_enable_disable() {
        let mut registry = McpServerRegistry::new();
        registry
            .upsert(McpServerConfig {
                name: "git".to_string(),
                command: "node".to_string(),
                args: vec!["git_server.js".to_string()],
                enabled: true,
                working_directory: None,
                env: HashMap::new(),
                protocol_version: None,
                client_name: None,
                tool_prefix: None,
            })
            .unwrap();
        registry.disable("git").unwrap();
        assert_eq!(registry.list_enabled().len(), 0);
        registry.enable("git").unwrap();
        assert_eq!(registry.list_enabled().len(), 1);
    }
}
