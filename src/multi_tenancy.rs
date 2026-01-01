//! Multi Tenancy Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use uuid::Uuid;
use anyhow::{Result, Context, bail};

use crate::paged_attention::{SessionManager, SessionId};
use crate::plugin_system::PluginRegistry;
use crate::radix_tree::KVCacheManager;




#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// TenantID structure
pub struct TenantID(pub Uuid);

// Implementation for TenantID
impl TenantID {
    
    /// new function
    pub fn new() -> Self {
        TenantID(Uuid::new_v4())
    }

    
    /// from_str function
    pub fn from_str(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .context("Invalid tenant ID format")?;
        Ok(TenantID(uuid))
    }

    
    /// to_string function
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

// Implementation for Default
impl Default for TenantID {
    fn default() -> Self {
        Self::new()
    }
}




#[derive(Debug, Clone)]
    /// TenantQuota structure
pub struct TenantQuota {
    
    pub max_sessions: usize,

    
    pub max_context_length: usize,

    
    pub max_memory_bytes: u64,

    
    pub max_plugin_count: usize,

    
    pub max_concurrent_requests: usize,
}

// Implementation for Default
impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_sessions: 10,
            max_context_length: 32768,       
            max_memory_bytes: 4 * 1024 * 1024 * 1024,  
            max_plugin_count: 20,
            max_concurrent_requests: 5,
        }
    }
}

// Implementation for TenantQuota
impl TenantQuota {
    
    /// enterprise function
    pub fn enterprise() -> Self {
        Self {
            max_sessions: 100,
            max_context_length: 131072,      
            max_memory_bytes: 64 * 1024 * 1024 * 1024,  
            max_plugin_count: 100,
            max_concurrent_requests: 50,
        }
    }

    
    /// free function
    pub fn free() -> Self {
        Self {
            max_sessions: 3,
            max_context_length: 8192,        
            max_memory_bytes: 1 * 1024 * 1024 * 1024,  
            max_plugin_count: 5,
            max_concurrent_requests: 2,
        }
    }
}




#[derive(Debug, Clone, Default)]
    /// TenantResourceUsage structure
pub struct TenantResourceUsage {
    
    pub active_sessions: usize,

    
    pub memory_bytes: u64,

    
    pub loaded_plugins: usize,

    
    pub concurrent_requests: usize,

    
    pub total_requests: u64,

    
    pub total_tokens_generated: u64,
}

// Implementation for TenantResourceUsage
impl TenantResourceUsage {
    
    
    /// check_quota function
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




    /// TenantContext structure
pub struct TenantContext {
    
    pub id: TenantID,

    
    pub name: String,

    
    pub quota: TenantQuota,

    
    pub usage: RwLock<TenantResourceUsage>,

    
    pub session_manager: Arc<RwLock<SessionManager>>,

    
    pub plugin_registry: Arc<RwLock<PluginRegistry>>,

    
    pub kv_cache_manager: Arc<RwLock<KVCacheManager>>,

    
    pub created_at: std::time::SystemTime,

    
    pub enabled: RwLock<bool>,
}

// Implementation for TenantContext
impl TenantContext {
    
    /// new function
    pub fn new(id: TenantID, name: String, quota: TenantQuota) -> Self {
        Self {
            id,
            name,
            quota: quota.clone(),
            usage: RwLock::new(TenantResourceUsage::default()),
            session_manager: Arc::new(RwLock::new(SessionManager::new(
                4096,  
                8192,  
                256,   
            ))),
            plugin_registry: Arc::new(RwLock::new(
                PluginRegistry::new(crate::plugin_system::ResourceQuota::new(1000, 16 * 1024 * 1024))
            )),
            kv_cache_manager: Arc::new(RwLock::new(KVCacheManager::new())),
            created_at: std::time::SystemTime::now(),
            enabled: RwLock::new(true),
        }
    }

    
    /// check_quota function
    pub fn check_quota(&self) -> Result<()> {
        let usage = self.usage.read();
        usage.check_quota(&self.quota)
    }

    
    
    /// increment_sessions function
    pub fn increment_sessions(&self) -> Result<()> {
        let mut usage = self.usage.write();
        
        let mut temp_usage = usage.clone();
        temp_usage.active_sessions += 1;
        temp_usage.check_quota(&self.quota)?;

        
        usage.active_sessions += 1;
        Ok(())
    }

    
    /// decrement_sessions function
    pub fn decrement_sessions(&self) {
        let mut usage = self.usage.write();
        if usage.active_sessions > 0 {
            usage.active_sessions -= 1;
        }
    }

    
    /// update_memory_usage function
    pub fn update_memory_usage(&self, bytes: u64) -> Result<()> {
        let mut usage = self.usage.write();
        usage.memory_bytes = bytes;
        usage.check_quota(&self.quota)
    }

    
    
    /// increment_concurrent_requests function
    pub fn increment_concurrent_requests(&self) -> Result<()> {
        let mut usage = self.usage.write();
        
        let mut temp_usage = usage.clone();
        temp_usage.concurrent_requests += 1;
        temp_usage.total_requests += 1;
        temp_usage.check_quota(&self.quota)?;

        
        usage.concurrent_requests += 1;
        usage.total_requests += 1;
        Ok(())
    }

    
    /// decrement_concurrent_requests function
    pub fn decrement_concurrent_requests(&self) {
        let mut usage = self.usage.write();
        if usage.concurrent_requests > 0 {
            usage.concurrent_requests -= 1;
        }
    }

    
    /// get_usage function
    pub fn get_usage(&self) -> TenantResourceUsage {
        self.usage.read().clone()
    }

    
    /// disable function
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    
    /// enable function
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    
    /// is_enabled function
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }
}




    /// TenantManager structure
pub struct TenantManager {
    
    tenants: RwLock<HashMap<TenantID, Arc<TenantContext>>>,

    
    default_tenant: Arc<TenantContext>,
}

// Implementation for TenantManager
impl TenantManager {
    
    /// new function
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

    
    /// global function
    pub fn global() -> &'static TenantManager {
        static INSTANCE: once_cell::sync::Lazy<TenantManager> =
            once_cell::sync::Lazy::new(|| TenantManager::new());
        &INSTANCE
    }

    
    /// create_tenant function
    pub fn create_tenant(&self, name: String, quota: TenantQuota) -> TenantID {
        let id = TenantID::new();
        let context = Arc::new(TenantContext::new(id, name.clone(), quota));

        let mut tenants = self.tenants.write();
        tenants.insert(id, context);

        println!("[TenantManager] Created tenant: {} ({})", name, id.to_string());
        id
    }

    
    /// get_tenant function
    pub fn get_tenant(&self, id: TenantID) -> Result<Arc<TenantContext>> {
        let tenants = self.tenants.read();
        tenants.get(&id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Tenant not found: {}", id.to_string()))
    }

    
    /// remove_tenant function
    pub fn remove_tenant(&self, id: TenantID) -> Result<()> {
        let mut tenants = self.tenants.write();

        let context = tenants.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Tenant not found: {}", id.to_string()))?;

        
        println!("[TenantManager] Cleaning up tenant: {}", context.name);

        
        {
            let _session_mgr = context.session_manager.write();
            
        }

        
        {
            let _plugin_registry = context.plugin_registry.write();
            
        }

        
        {
            let _kv_cache = context.kv_cache_manager.write();
            
        }

        println!("[TenantManager] Tenant removed: {}", context.name);
        Ok(())
    }

    
    /// list_tenants function
    pub fn list_tenants(&self) -> Vec<TenantID> {
        let tenants = self.tenants.read();
        tenants.keys().copied().collect()
    }

    
    /// default_tenant function
    pub fn default_tenant(&self) -> Arc<TenantContext> {
        self.default_tenant.clone()
    }

    
    /// tenant_count function
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }

    
    /// check_tenant_quota function
    pub fn check_tenant_quota(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.check_quota()
    }

    
    /// get_tenant_usage function
    pub fn get_tenant_usage(&self, id: TenantID) -> Result<TenantResourceUsage> {
        let context = self.get_tenant(id)?;
        Ok(context.get_usage())
    }

    
    /// disable_tenant function
    pub fn disable_tenant(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.disable();
        println!("[TenantManager] Tenant disabled: {}", context.name);
        Ok(())
    }

    
    /// enable_tenant function
    pub fn enable_tenant(&self, id: TenantID) -> Result<()> {
        let context = self.get_tenant(id)?;
        context.enable();
        println!("[TenantManager] Tenant enabled: {}", context.name);
        Ok(())
    }
}

// Implementation for Default
impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}




#[derive(Debug, Clone, PartialEq, Eq, Hash)]
    /// TenantSessionID structure
pub struct TenantSessionID {
    pub tenant_id: TenantID,
    pub session_id: SessionId,
}

// Implementation for TenantSessionID
impl TenantSessionID {
    /// new function
    pub fn new(tenant_id: TenantID, session_id: SessionId) -> Self {
        Self { tenant_id, session_id }
    }

    /// to_string function
    pub fn to_string(&self) -> String {
        format!("{}:{}", self.tenant_id.to_string(), self.session_id.0)
    }
}



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

        
        assert!(context.increment_sessions().is_ok());

        
        assert!(context.increment_sessions().is_ok());

        
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
        assert_eq!(usage.total_requests, 1);  
    }
}
