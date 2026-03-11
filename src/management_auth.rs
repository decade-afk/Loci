//! Dynamic management-auth policy plugin ABI and registry.
//!
//! This is intended for protecting control-plane endpoints in the serve
//! runtime while keeping the policy replaceable and extensible.

use crate::error::{LociError, Result};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::c_void;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementAuthContext {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub remote_addr: Option<String>,
}

impl ManagementAuthContext {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|value| value.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagementAuthDecision {
    Allow,
    Deny(String),
}

pub trait ManagementAuthPolicyPlugin: Send + Sync {
    fn name(&self) -> &str;

    fn authorize(&self, context: &ManagementAuthContext) -> ManagementAuthDecision;
}

pub struct AllowAllManagementAuthPolicy;

impl ManagementAuthPolicyPlugin for AllowAllManagementAuthPolicy {
    fn name(&self) -> &str {
        "allow-all.management.auth"
    }

    fn authorize(&self, _context: &ManagementAuthContext) -> ManagementAuthDecision {
        ManagementAuthDecision::Allow
    }
}

pub struct BearerTokenManagementAuthPolicy {
    token: String,
}

impl BearerTokenManagementAuthPolicy {
    pub fn new(token: impl Into<String>) -> Result<Self> {
        let token = token.into().trim().to_string();
        if token.is_empty() {
            return Err(LociError::ConfigError(
                "management bearer token cannot be empty".to_string(),
            ));
        }
        Ok(Self { token })
    }
}

impl ManagementAuthPolicyPlugin for BearerTokenManagementAuthPolicy {
    fn name(&self) -> &str {
        "bearer-token.management.auth"
    }

    fn authorize(&self, context: &ManagementAuthContext) -> ManagementAuthDecision {
        let authorized = context
            .header("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(|token| token == self.token)
            .unwrap_or(false)
            || context
                .header("x-api-key")
                .map(|token| token == self.token)
                .unwrap_or(false);

        if authorized {
            ManagementAuthDecision::Allow
        } else {
            ManagementAuthDecision::Deny("missing or invalid management credentials".to_string())
        }
    }
}

pub struct LoopbackOnlyManagementAuthPolicy;

fn remote_addr_is_loopback(remote_addr: &str) -> bool {
    if let Ok(addr) = remote_addr.parse::<SocketAddr>() {
        let ip = addr.ip();
        return ip.is_loopback()
            || matches!(ip, IpAddr::V6(v6) if v6.to_ipv4_mapped().map(|ip| ip.is_loopback()).unwrap_or(false));
    }

    if let Ok(ip) = remote_addr.parse::<IpAddr>() {
        return ip.is_loopback()
            || matches!(ip, IpAddr::V6(v6) if v6.to_ipv4_mapped().map(|ip| ip.is_loopback()).unwrap_or(false));
    }

    false
}

impl ManagementAuthPolicyPlugin for LoopbackOnlyManagementAuthPolicy {
    fn name(&self) -> &str {
        "loopback-only.management.auth"
    }

    fn authorize(&self, context: &ManagementAuthContext) -> ManagementAuthDecision {
        match context.remote_addr.as_deref() {
            Some(remote_addr) if remote_addr_is_loopback(remote_addr) => {
                ManagementAuthDecision::Allow
            }
            Some(_) => ManagementAuthDecision::Deny(
                "management access is restricted to loopback clients".to_string(),
            ),
            None => ManagementAuthDecision::Deny(
                "management access requires a resolvable client address".to_string(),
            ),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicManagementAuthPolicyOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynManagementAuthPolicyPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

pub fn dynamic_management_auth_policy_into_opaque(
    policy: Box<dyn ManagementAuthPolicyPlugin>,
) -> DynamicManagementAuthPolicyOpaque {
    let raw: *mut dyn ManagementAuthPolicyPlugin = Box::into_raw(policy);
    let parts: RawDynManagementAuthPolicyPtr = unsafe { std::mem::transmute(raw) };
    DynamicManagementAuthPolicyOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

pub unsafe fn dynamic_management_auth_policy_from_opaque(
    opaque: DynamicManagementAuthPolicyOpaque,
) -> Option<Box<dyn ManagementAuthPolicyPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynManagementAuthPolicyPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ManagementAuthPolicyPlugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type ManagementAuthPolicyConstructor = unsafe extern "C" fn() -> DynamicManagementAuthPolicyOpaque;

pub struct LoadedManagementAuthPolicy {
    policy: Arc<dyn ManagementAuthPolicyPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

impl LoadedManagementAuthPolicy {
    pub fn policy(&self) -> &Arc<dyn ManagementAuthPolicyPlugin> {
        &self.policy
    }

    pub fn name(&self) -> &str {
        self.policy.name()
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementAuthPolicyDescriptor {
    pub name: String,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

struct ManagementAuthPolicyEntry {
    policy: Arc<dyn ManagementAuthPolicyPlugin>,
    dynamic: Option<DynamicManagementAuthHandle>,
}

struct DynamicManagementAuthHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

pub struct ManagementAuthPolicyRegistry {
    policies: RwLock<HashMap<String, ManagementAuthPolicyEntry>>,
}

impl ManagementAuthPolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_builtin_policies() -> Self {
        let registry = Self::new();
        registry
            .register_policy(AllowAllManagementAuthPolicy)
            .expect("register allow-all management auth policy");
        registry
            .register_policy(LoopbackOnlyManagementAuthPolicy)
            .expect("register loopback-only management auth policy");
        registry
    }

    pub fn with_builtin_and_bearer_token(token: Option<&str>) -> Result<Self> {
        let registry = Self::with_builtin_policies();
        if let Some(token) = token {
            registry.register_policy(BearerTokenManagementAuthPolicy::new(token)?)?;
        }
        Ok(registry)
    }

    pub fn register_policy<P>(&self, policy: P) -> Result<()>
    where
        P: ManagementAuthPolicyPlugin + 'static,
    {
        self.register_policy_arc(Arc::new(policy))
    }

    pub fn register_policy_arc(&self, policy: Arc<dyn ManagementAuthPolicyPlugin>) -> Result<()> {
        let key = policy.name().trim().to_string();
        if key.is_empty() {
            return Err(LociError::PluginError(
                "Management auth policy name cannot be empty".to_string(),
            ));
        }

        let mut policies = self.policies.write();
        if policies.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Management auth policy '{}' already registered",
                key
            )));
        }
        policies.insert(
            key,
            ManagementAuthPolicyEntry {
                policy,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ManagementAuthPolicyPlugin>> {
        self.policies
            .read()
            .get(name)
            .map(|entry| Arc::clone(&entry.policy))
    }

    pub fn describe(&self, name: &str) -> Option<ManagementAuthPolicyDescriptor> {
        self.policies
            .read()
            .get(name)
            .map(|entry| ManagementAuthPolicyDescriptor {
                name: name.to_string(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
    }

    pub fn descriptors(&self) -> Vec<ManagementAuthPolicyDescriptor> {
        let mut descriptors = self
            .policies
            .read()
            .iter()
            .map(|(name, entry)| ManagementAuthPolicyDescriptor {
                name: name.clone(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
            .collect::<Vec<_>>();
        descriptors.sort_by(|a, b| a.name.cmp(&b.name));
        descriptors
    }

    pub fn list_names(&self) -> Vec<String> {
        self.descriptors()
            .into_iter()
            .map(|descriptor| descriptor.name)
            .collect()
    }

    fn load_dynamic_entry<P: AsRef<Path>>(
        library_path: P,
    ) -> Result<(String, ManagementAuthPolicyEntry)> {
        let loaded = load_dynamic_management_auth_policy(&library_path)?;
        let key = loaded.name().to_string();
        let entry = ManagementAuthPolicyEntry {
            policy: Arc::clone(loaded.policy()),
            dynamic: Some(DynamicManagementAuthHandle {
                library: Arc::clone(&loaded.library),
                source: loaded.source().to_path_buf(),
            }),
        };
        Ok((key, entry))
    }

    pub fn load_dynamic_policy<P: AsRef<Path>>(&self, library_path: P) -> Result<String> {
        let (name, entry) = Self::load_dynamic_entry(library_path)?;
        let mut policies = self.policies.write();
        if policies.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Management auth policy '{}' already registered",
                name
            )));
        }
        policies.insert(name.clone(), entry);
        Ok(name)
    }

    pub fn unload_dynamic_policy(&self, name: &str) -> Result<()> {
        let mut policies = self.policies.write();
        match policies.get(name) {
            Some(entry) => {
                if entry.dynamic.is_none() {
                    return Err(LociError::PluginError(format!(
                        "Static management auth policy '{}' cannot be unloaded at runtime",
                        name
                    )));
                }
            }
            None => {
                return Err(LociError::PluginError(format!(
                    "Management auth policy '{}' not found",
                    name
                )));
            }
        }
        policies.remove(name);
        Ok(())
    }

    pub fn reload_dynamic_policy(&self, name: &str) -> Result<()> {
        let source = {
            let policies = self.policies.read();
            let entry = policies.get(name).ok_or_else(|| {
                LociError::PluginError(format!("Management auth policy '{}' not found", name))
            })?;
            let dynamic = entry.dynamic.as_ref().ok_or_else(|| {
                LociError::PluginError(format!(
                    "Static management auth policy '{}' cannot be hot-reloaded",
                    name
                ))
            })?;
            dynamic.source.clone()
        };

        let (loaded_name, entry) = Self::load_dynamic_entry(&source)?;
        if loaded_name != name {
            return Err(LociError::PluginError(format!(
                "Reloaded management auth policy name mismatch: expected '{}', got '{}'",
                name, loaded_name
            )));
        }

        let mut policies = self.policies.write();
        if !policies.contains_key(name) {
            return Err(LociError::PluginError(format!(
                "Management auth policy '{}' not found during reload",
                name
            )));
        }
        policies.insert(name.to_string(), entry);
        Ok(())
    }
}

impl Default for ManagementAuthPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn load_dynamic_management_auth_policy<P: AsRef<Path>>(
    library_path: P,
) -> Result<LoadedManagementAuthPolicy> {
    let lib_path = library_path.as_ref();
    if !lib_path.exists() {
        return Err(LociError::PluginError(format!(
            "Management auth policy plugin library not found: {}",
            lib_path.display()
        )));
    }

    let library = unsafe {
        Library::new(lib_path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load management auth policy plugin '{}': {}",
                lib_path.display(),
                e
            ))
        })?
    };

    let constructor: Symbol<ManagementAuthPolicyConstructor> = unsafe {
        match library.get(b"create_management_auth_policy_v1") {
            Ok(sym) => sym,
            Err(_) => library.get(b"create_management_auth_policy").map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to find management auth policy constructor symbol \
                     ('create_management_auth_policy_v1' or 'create_management_auth_policy'): {}",
                    e
                ))
            })?,
        }
    };

    let policy_opaque = unsafe { constructor() };
    let policy =
        unsafe { dynamic_management_auth_policy_from_opaque(policy_opaque) }.ok_or_else(|| {
            LociError::PluginError(
                "Management auth policy constructor returned invalid policy payload".to_string(),
            )
        })?;
    if policy.name().trim().is_empty() {
        return Err(LociError::PluginError(
            "Management auth policy plugin returned empty name".to_string(),
        ));
    }

    Ok(LoadedManagementAuthPolicy {
        policy: Arc::<dyn ManagementAuthPolicyPlugin>::from(policy),
        library: Arc::new(library),
        source: lib_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct HeaderAuthPolicy;

    impl ManagementAuthPolicyPlugin for HeaderAuthPolicy {
        fn name(&self) -> &str {
            "mock.management.auth"
        }

        fn authorize(&self, context: &ManagementAuthContext) -> ManagementAuthDecision {
            if context.header("x-test-auth") == Some("ok") {
                ManagementAuthDecision::Allow
            } else {
                ManagementAuthDecision::Deny("denied".to_string())
            }
        }
    }

    #[test]
    fn builtin_registry_contains_allow_all() {
        let registry = ManagementAuthPolicyRegistry::with_builtin_policies();
        assert!(registry
            .list_names()
            .contains(&"allow-all.management.auth".to_string()));
        assert!(registry
            .list_names()
            .contains(&"loopback-only.management.auth".to_string()));
    }

    #[test]
    fn dynamic_opaque_roundtrip() {
        let policy: Box<dyn ManagementAuthPolicyPlugin> = Box::new(HeaderAuthPolicy);
        let opaque = dynamic_management_auth_policy_into_opaque(policy);
        let restored = unsafe { dynamic_management_auth_policy_from_opaque(opaque) };
        assert_eq!(
            restored.expect("restore policy").name(),
            "mock.management.auth"
        );
    }

    #[test]
    fn bearer_policy_accepts_authorization_header() {
        let policy = BearerTokenManagementAuthPolicy::new("secret").expect("policy");
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer secret".to_string());
        let context = ManagementAuthContext {
            method: "POST".to_string(),
            path: "/sessions".to_string(),
            headers,
            remote_addr: None,
        };
        assert_eq!(policy.authorize(&context), ManagementAuthDecision::Allow);
    }

    #[test]
    fn loopback_policy_accepts_localhost_clients() {
        let policy = LoopbackOnlyManagementAuthPolicy;
        let context = ManagementAuthContext {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: HashMap::new(),
            remote_addr: Some("127.0.0.1:8080".to_string()),
        };
        assert_eq!(policy.authorize(&context), ManagementAuthDecision::Allow);

        let context = ManagementAuthContext {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: HashMap::new(),
            remote_addr: Some("[::1]:8080".to_string()),
        };
        assert_eq!(policy.authorize(&context), ManagementAuthDecision::Allow);
    }

    #[test]
    fn loopback_policy_rejects_non_local_clients() {
        let policy = LoopbackOnlyManagementAuthPolicy;
        let context = ManagementAuthContext {
            method: "GET".to_string(),
            path: "/sessions".to_string(),
            headers: HashMap::new(),
            remote_addr: Some("10.0.0.2:8080".to_string()),
        };
        assert_eq!(
            policy.authorize(&context),
            ManagementAuthDecision::Deny(
                "management access is restricted to loopback clients".to_string()
            )
        );
    }
}
