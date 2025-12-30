/**
 * Loci Phase 4 Week 1: 多租户隔离系统
 *
 * 核心特性：
 * 1. 租户级别资源隔离（Session/KV Cache/Plugin）
 * 2. 细粒度资源配额控制
 * 3. 租户生命周期管理
 * 4. 资源使用统计与监控
 *
 * 隔离策略：
 * - Session ID 添加租户前缀
 * - 每个租户独立的 KV Cache 分配器
 * - 每个租户独立的 Plugin Registry
 * - 全局内存预算按租户分配
 */

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use uuid::Uuid;
use anyhow::{Result, Context, bail};

use crate::paged_attention::{SessionManager, SessionId};
use crate::plugin_system::PluginRegistry;
use crate::radix_tree::KVCacheManager;

// ==================== 租户 ID ====================

/// 租户 ID（全局唯一）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantID(pub Uuid);

impl TenantID {
    /// 生成新的租户 ID
    pub fn new() -> Self {
        TenantID(Uuid::new_v4())
    }

    /// 从字符串解析
    pub fn from_str(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .context("Invalid tenant ID format")?;
        Ok(TenantID(uuid))
    }

    /// 转换为字符串
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Default for TenantID {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 租户资源配额 ====================

/// 租户资源配额
#[derive(Debug, Clone)]
pub struct TenantQuota {
    /// 最大会话数
    pub max_sessions: usize,

    /// 最大上下文长度（每个会话）
    pub max_context_length: usize,

    /// 最大内存使用（字节）
    pub max_memory_bytes: u64,

    /// 最大插件数量
    pub max_plugin_count: usize,

    /// 最大并发请求数
    pub max_concurrent_requests: usize,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_sessions: 10,
            max_context_length: 32768,       // 32k tokens
            max_memory_bytes: 4 * 1024 * 1024 * 1024,  // 4GB
            max_plugin_count: 20,
            max_concurrent_requests: 5,
        }
    }
}

impl TenantQuota {
    /// 企业级配额（更高的资源限制）
    pub fn enterprise() -> Self {
        Self {
            max_sessions: 100,
            max_context_length: 131072,      // 128k tokens
            max_memory_bytes: 64 * 1024 * 1024 * 1024,  // 64GB
            max_plugin_count: 100,
            max_concurrent_requests: 50,
        }
    }

    /// 免费级配额（受限资源）
    pub fn free() -> Self {
        Self {
            max_sessions: 3,
            max_context_length: 8192,        // 8k tokens
            max_memory_bytes: 1 * 1024 * 1024 * 1024,  // 1GB
            max_plugin_count: 5,
            max_concurrent_requests: 2,
        }
    }
}

// ==================== 租户资源使用统计 ====================

/// 租户资源使用统计
#[derive(Debug, Clone, Default)]
pub struct TenantResourceUsage {
    /// 当前会话数
    pub active_sessions: usize,

    /// 当前内存使用（字节）
    pub memory_bytes: u64,

    /// 当前加载的插件数
    pub loaded_plugins: usize,

    /// 当前并发请求数
    pub concurrent_requests: usize,

    /// 累计请求数
    pub total_requests: u64,

    /// 累计生成 token 数
    pub total_tokens_generated: u64,
}

impl TenantResourceUsage {
    /// 检查是否超限
    /// 修复：使用 > 而非 >= 以允许使用满配额（例如 max_sessions=10 应允许创建第10个会话）
    pub fn check_quota(&self, quota: &TenantQuota) -> Result<()> {
        if self.active_sessions > quota.max_sessions {
            bail!("Session quota exceeded: {}/{}", self.active_sessions, quota.max_sessions);
        }

        if self.memory_bytes > quota.max_memory_bytes {
            bail!("Memory quota exceeded: {}/{} bytes",
                self.memory_bytes, quota.max_memory_bytes);
        }

        if self.loaded_plugins > quota.max_plugin_count {
            bail!("Plugin quota exceeded: {}/{}", self.loaded_plugins, quota.max_plugin_count);
        }

        if self.concurrent_requests > quota.max_concurrent_requests {
            bail!("Concurrent request quota exceeded: {}/{}",
                self.concurrent_requests, quota.max_concurrent_requests);
        }

        Ok(())
    }
}

// ==================== 租户上下文 ====================

/// 租户上下文（每个租户的独立资源空间）
pub struct TenantContext {
    /// 租户 ID
    pub id: TenantID,

    /// 租户名称
    pub name: String,

    /// 资源配额
    pub quota: TenantQuota,

    /// 资源使用统计
    pub usage: RwLock<TenantResourceUsage>,

    /// Session 管理器
    pub session_manager: Arc<RwLock<SessionManager>>,

    /// Plugin Registry
    pub plugin_registry: Arc<RwLock<PluginRegistry>>,

    /// KV Cache 管理器
    pub kv_cache_manager: Arc<RwLock<KVCacheManager>>,

    /// 租户创建时间
    pub created_at: std::time::SystemTime,

    /// 是否启用
    pub enabled: RwLock<bool>,
}

impl TenantContext {
    /// 创建新的租户上下文
    pub fn new(id: TenantID, name: String, quota: TenantQuota) -> Self {
        Self {
            id,
            name,
            quota: quota.clone(),
            usage: RwLock::new(TenantResourceUsage::default()),
            session_manager: Arc::new(RwLock::new(SessionManager::new(
                4096,  // 4GB VRAM
                8192,  // 8GB RAM
                256,   // 256KB block size
            ))),
            plugin_registry: Arc::new(RwLock::new(
                PluginRegistry::new(crate::plugin_system::ResourceQuota::new(1000, 16 * 1024 * 1024))
            )),
            kv_cache_manager: Arc::new(RwLock::new(KVCacheManager::new())),
            created_at: std::time::SystemTime::now(),
            enabled: RwLock::new(true),
        }
    }

    /// 检查配额是否允许新操作
    pub fn check_quota(&self) -> Result<()> {
        let usage = self.usage.read();
        usage.check_quota(&self.quota)
    }

    /// 增加会话计数
    /// 修复：先检查配额再递增，防止失败时污染计数
    pub fn increment_sessions(&self) -> Result<()> {
        let mut usage = self.usage.write();
        // 预先检查是否会超额（克隆当前状态并模拟递增）
        let mut temp_usage = usage.clone();
        temp_usage.active_sessions += 1;
        temp_usage.check_quota(&self.quota)?;

        // 检查通过后才真正递增
        usage.active_sessions += 1;
        Ok(())
    }

    /// 减少会话计数
    pub fn decrement_sessions(&self) {
        let mut usage = self.usage.write();
        if usage.active_sessions > 0 {
            usage.active_sessions -= 1;
        }
    }

    /// 更新内存使用
    pub fn update_memory_usage(&self, bytes: u64) -> Result<()> {
        let mut usage = self.usage.write();
        usage.memory_bytes = bytes;
        usage.check_quota(&self.quota)
    }

    /// 增加并发请求计数
    /// 修复：先检查配额再递增，防止失败时污染计数
    pub fn increment_concurrent_requests(&self) -> Result<()> {
        let mut usage = self.usage.write();
        // 预先检查是否会超额（克隆当前状态并模拟递增）
        let mut temp_usage = usage.clone();
        temp_usage.concurrent_requests += 1;
        temp_usage.total_requests += 1;
        temp_usage.check_quota(&self.quota)?;

        // 检查通过后才真正递增
        usage.concurrent_requests += 1;
        usage.total_requests += 1;
        Ok(())
    }

    /// 减少并发请求计数
    pub fn decrement_concurrent_requests(&self) {
        let mut usage = self.usage.write();
        if usage.concurrent_requests > 0 {
            usage.concurrent_requests -= 1;
        }
    }

    /// 获取资源使用统计
    pub fn get_usage(&self) -> TenantResourceUsage {
        self.usage.read().clone()
    }

    /// 禁用租户
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    /// 启用租户
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    /// 检查是否启用
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }
}

// ==================== 租户管理器 ====================

/// 租户管理器（全局单例）
pub struct TenantManager {
    /// 租户映射表
    tenants: RwLock<HashMap<TenantID, Arc<TenantContext>>>,

    /// 默认租户（用于单租户模式）
    default_tenant: Arc<TenantContext>,
}

impl TenantManager {
    /// 创建新的租户管理器
    pub fn new() -> Self {
        let default_id = TenantID::new();
        let default_tenant = Arc::new(TenantContext::new(
            default_id,
            "default".to_string(),
            TenantQuota::default(),
        ));

        let mut tenants = HashMap::new();
        tenants.insert(default_id, default_tenant.clone());

        Self {
            tenants: RwLock::new(tenants),
            default_tenant,
        }
    }

    /// 获取全局单例
    pub fn global() -> &'static TenantManager {
        static INSTANCE: once_cell::sync::Lazy<TenantManager> =
            once_cell::sync::Lazy::new(|| TenantManager::new());
        &INSTANCE
    }

    /// 创建新租户
    pub fn create_tenant(&self, name: String, quota: TenantQuota) -> TenantID {
        let id = TenantID::new();
        let context = Arc::new(TenantContext::new(id, name.clone(), quota));

        let mut tenants = self.tenants.write();
        tenants.insert(id, context);

        println!("[TenantManager] Created tenant: {} ({})", name, id.to_string());
        id
    }

    /// 获取租户上下文
    pub fn get_tenant(&self, id: TenantID) -> Result<Arc<TenantContext>> {
        let tenants = self.tenants.read();
        tenants.get(&id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Tenant not found: {}", id.to_string()))
    }

    /// 删除租户（清理所有资源）
    pub fn remove_tenant(&self, id: TenantID) -> Result<()> {
        let mut tenants = self.tenants.write();

        let context = tenants.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Tenant not found: {}", id.to_string()))?;

        // 清理资源
        println!("[TenantManager] Cleaning up tenant: {}", context.name);

        // 清理 Session Manager
        {
            let _session_mgr = context.session_manager.write();
            // TODO: 清理所有会话
        }

        // 清理 Plugin Registry
        {
            let _plugin_registry = context.plugin_registry.write();
            // TODO: 卸载所有插件
        }

        // 清理 KV Cache
        {
            let _kv_cache = context.kv_cache_manager.write();
            // TODO: 释放所有缓存
        }

        println!("[TenantManager] Tenant removed: {}", context.name);
        Ok(())
    }

    /// 列出所有租户
    pub fn list_tenants(&self) -> Vec<TenantID> {
        let tenants = self.tenants.read();
        tenants.keys().copied().collect()
    }

    /// 获取默认租户
    pub fn default_tenant(&self) -> Arc<TenantContext> {
        self.default_tenant.clone()
    }

    /// 获取租户数量
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }

    /// 检查租户配额
    pub fn check_tenant_quota(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.check_quota()
    }

    /// 获取租户资源使用统计
    pub fn get_tenant_usage(&self, id: TenantID) -> Result<TenantResourceUsage> {
        let context = self.get_tenant(id)?;
        Ok(context.get_usage())
    }

    /// 禁用租户
    pub fn disable_tenant(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.disable();
        println!("[TenantManager] Tenant disabled: {}", context.name);
        Ok(())
    }

    /// 启用租户
    pub fn enable_tenant(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.enable();
        println!("[TenantManager] Tenant enabled: {}", context.name);
        Ok(())
    }
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 租户隔离的 Session ID ====================

/// 租户隔离的 Session ID
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantSessionID {
    pub tenant_id: TenantID,
    pub session_id: SessionId,
}

impl TenantSessionID {
    pub fn new(tenant_id: TenantID, session_id: SessionId) -> Self {
        Self { tenant_id, session_id }
    }

    pub fn to_string(&self) -> String {
        format!("{}:{}", self.tenant_id.to_string(), self.session_id.0)
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_creation() {
        let manager = TenantManager::new();

        let id = manager.create_tenant(
            "test-tenant".to_string(),
            TenantQuota::default(),
        );

        let context = manager.get_tenant(id).unwrap();
        assert_eq!(context.name, "test-tenant");
        assert_eq!(context.id, id);
    }

    #[test]
    fn test_quota_enforcement() {
        let manager = TenantManager::new();

        let quota = TenantQuota {
            max_sessions: 2,
            ..Default::default()
        };

        let id = manager.create_tenant("quota-test".to_string(), quota);
        let context = manager.get_tenant(id).unwrap();

        // 第一个会话：成功
        assert!(context.increment_sessions().is_ok());

        // 第二个会话：成功
        assert!(context.increment_sessions().is_ok());

        // 第三个会话：失败（超限）
        assert!(context.increment_sessions().is_err());
    }

    #[test]
    fn test_tenant_removal() {
        let manager = TenantManager::new();

        let id = manager.create_tenant("remove-test".to_string(), TenantQuota::default());
        assert!(manager.get_tenant(id).is_ok());

        manager.remove_tenant(id).unwrap();
        assert!(manager.get_tenant(id).is_err());
    }

    #[test]
    fn test_tenant_enable_disable() {
        let manager = TenantManager::new();

        let id = manager.create_tenant("enable-test".to_string(), TenantQuota::default());
        let context = manager.get_tenant(id).unwrap();

        assert!(context.is_enabled());

        manager.disable_tenant(id).unwrap();
        assert!(!context.is_enabled());

        manager.enable_tenant(id).unwrap();
        assert!(context.is_enabled());
    }

    #[test]
    fn test_resource_usage_tracking() {
        let context = TenantContext::new(
            TenantID::new(),
            "usage-test".to_string(),
            TenantQuota::default(),
        );

        context.increment_sessions().unwrap();
        context.increment_concurrent_requests().unwrap();

        let usage = context.get_usage();
        assert_eq!(usage.active_sessions, 1);
        assert_eq!(usage.concurrent_requests, 1);
        assert_eq!(usage.total_requests, 1);

        context.decrement_sessions();
        context.decrement_concurrent_requests();

        let usage = context.get_usage();
        assert_eq!(usage.active_sessions, 0);
        assert_eq!(usage.concurrent_requests, 0);
        assert_eq!(usage.total_requests, 1);  // 累计请求不变
    }
}
