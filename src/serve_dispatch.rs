//! Dynamic serve backpressure policy plugin ABI.
//!
//! This module defines a small plugin surface for request dispatch behavior
//! when the serve worker queue is full.

use crate::error::{LociError, Result};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Queue pressure context passed to backpressure policy plugins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueuePressureContext {
    /// Number of retry attempts already performed for current stream.
    pub attempt: u32,
    /// Current queue length snapshot.
    pub queue_len: usize,
    /// Queue capacity (0 when unknown/unbounded).
    pub queue_capacity: usize,
}

/// Backpressure action selected by plugin when queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueFullAction {
    /// Reject request immediately.
    Reject,
    /// Block until queue accepts the request.
    Block,
    /// Sleep for N milliseconds and retry enqueue.
    RetryAfterMillis(u64),
}

/// Dynamic backpressure policy plugin interface for serve dispatch.
pub trait ServeDispatchPolicyPlugin: Send + Sync {
    /// Stable policy name for diagnostics.
    fn name(&self) -> &str;

    /// Decide how to handle a full queue.
    fn on_queue_full(&self, context: &QueuePressureContext) -> QueueFullAction;

    /// Max retries when returning [`QueueFullAction::RetryAfterMillis`].
    fn max_retries(&self) -> u32 {
        0
    }
}

/// Builtin policy that rejects immediately when queue is full.
pub struct RejectServeDispatchPolicyPlugin;

impl ServeDispatchPolicyPlugin for RejectServeDispatchPolicyPlugin {
    fn name(&self) -> &str {
        "reject"
    }

    fn on_queue_full(&self, _context: &QueuePressureContext) -> QueueFullAction {
        QueueFullAction::Reject
    }
}

/// Builtin policy that blocks until queue accepts the request.
pub struct BlockServeDispatchPolicyPlugin;

impl ServeDispatchPolicyPlugin for BlockServeDispatchPolicyPlugin {
    fn name(&self) -> &str {
        "block"
    }

    fn on_queue_full(&self, _context: &QueuePressureContext) -> QueueFullAction {
        QueueFullAction::Block
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicServeDispatchPolicyOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynServeDispatchPolicyPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn ServeDispatchPolicyPlugin>` into ABI-safe opaque payload.
pub fn dynamic_serve_dispatch_policy_into_opaque(
    plugin: Box<dyn ServeDispatchPolicyPlugin>,
) -> DynamicServeDispatchPolicyOpaque {
    let raw: *mut dyn ServeDispatchPolicyPlugin = Box::into_raw(plugin);
    let parts: RawDynServeDispatchPolicyPtr = unsafe { std::mem::transmute(raw) };
    DynamicServeDispatchPolicyOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert opaque payload back into `Box<dyn ServeDispatchPolicyPlugin>`.
///
/// # Safety
/// The payload must come from `dynamic_serve_dispatch_policy_into_opaque`
/// under compatible Rust toolchain/target ABI.
pub unsafe fn dynamic_serve_dispatch_policy_from_opaque(
    opaque: DynamicServeDispatchPolicyOpaque,
) -> Option<Box<dyn ServeDispatchPolicyPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynServeDispatchPolicyPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ServeDispatchPolicyPlugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type ServeDispatchPolicyConstructor = unsafe extern "C" fn() -> DynamicServeDispatchPolicyOpaque;

/// Loaded dynamic serve dispatch policy plugin.
pub struct LoadedServeDispatchPolicy {
    plugin: Arc<dyn ServeDispatchPolicyPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeDispatchPolicyDescriptor {
    pub name: String,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

impl LoadedServeDispatchPolicy {
    pub fn plugin(&self) -> &Arc<dyn ServeDispatchPolicyPlugin> {
        &self.plugin
    }

    pub fn name(&self) -> &str {
        self.plugin.name()
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

struct ServeDispatchPolicyEntry {
    policy: Arc<dyn ServeDispatchPolicyPlugin>,
    dynamic: Option<DynamicServeDispatchHandle>,
}

struct DynamicServeDispatchHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

/// Registry for builtin and dynamically loaded serve dispatch policies.
pub struct ServeDispatchPolicyRegistry {
    policies: RwLock<HashMap<String, ServeDispatchPolicyEntry>>,
}

impl ServeDispatchPolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_builtin_policies() -> Self {
        let registry = Self::new();
        registry
            .register_policy(RejectServeDispatchPolicyPlugin)
            .expect("register reject dispatch policy");
        registry
            .register_policy(BlockServeDispatchPolicyPlugin)
            .expect("register block dispatch policy");
        registry
    }

    pub fn register_policy<P>(&self, policy: P) -> Result<()>
    where
        P: ServeDispatchPolicyPlugin + 'static,
    {
        self.register_policy_arc(Arc::new(policy))
    }

    pub fn register_policy_arc(&self, policy: Arc<dyn ServeDispatchPolicyPlugin>) -> Result<()> {
        let key = policy.name().trim().to_string();
        if key.is_empty() {
            return Err(LociError::PluginError(
                "Serve dispatch policy name cannot be empty".to_string(),
            ));
        }

        let mut policies = self.policies.write();
        if policies.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Serve dispatch policy '{}' already registered",
                key
            )));
        }
        policies.insert(
            key,
            ServeDispatchPolicyEntry {
                policy,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ServeDispatchPolicyPlugin>> {
        self.policies
            .read()
            .get(name)
            .map(|entry| Arc::clone(&entry.policy))
    }

    pub fn describe(&self, name: &str) -> Option<ServeDispatchPolicyDescriptor> {
        self.policies
            .read()
            .get(name)
            .map(|entry| ServeDispatchPolicyDescriptor {
                name: name.to_string(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
    }

    pub fn descriptors(&self) -> Vec<ServeDispatchPolicyDescriptor> {
        let mut descriptors = self
            .policies
            .read()
            .iter()
            .map(|(name, entry)| ServeDispatchPolicyDescriptor {
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

    pub fn list_dynamic_names(&self) -> Vec<String> {
        let mut names = self
            .policies
            .read()
            .iter()
            .filter_map(|(name, entry)| entry.dynamic.as_ref().map(|_| name.clone()))
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    fn load_dynamic_entry<P: AsRef<Path>>(
        library_path: P,
    ) -> Result<(String, ServeDispatchPolicyEntry)> {
        let loaded = load_dynamic_serve_dispatch_policy(&library_path)?;
        let key = loaded.name().to_string();
        let entry = ServeDispatchPolicyEntry {
            policy: Arc::clone(loaded.plugin()),
            dynamic: Some(DynamicServeDispatchHandle {
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
                "Serve dispatch policy '{}' already registered",
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
                        "Static serve dispatch policy '{}' cannot be unloaded at runtime",
                        name
                    )));
                }
            }
            None => {
                return Err(LociError::PluginError(format!(
                    "Serve dispatch policy '{}' not found",
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
                LociError::PluginError(format!("Serve dispatch policy '{}' not found", name))
            })?;
            let dynamic = entry.dynamic.as_ref().ok_or_else(|| {
                LociError::PluginError(format!(
                    "Static serve dispatch policy '{}' cannot be hot-reloaded",
                    name
                ))
            })?;
            dynamic.source.clone()
        };

        let (loaded_name, entry) = Self::load_dynamic_entry(&source)?;
        if loaded_name != name {
            return Err(LociError::PluginError(format!(
                "Reloaded serve dispatch policy name mismatch: expected '{}', got '{}'",
                name, loaded_name
            )));
        }

        let mut policies = self.policies.write();
        if !policies.contains_key(name) {
            return Err(LociError::PluginError(format!(
                "Serve dispatch policy '{}' not found during reload",
                name
            )));
        }
        policies.insert(name.to_string(), entry);
        Ok(())
    }
}

impl Default for ServeDispatchPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Load dynamic serve dispatch policy plugin from shared library.
///
/// Expected exported symbol:
/// - `create_serve_dispatch_policy_v1() -> DynamicServeDispatchPolicyOpaque`
/// Fallback:
/// - `create_serve_dispatch_policy() -> DynamicServeDispatchPolicyOpaque`
pub fn load_dynamic_serve_dispatch_policy<P: AsRef<Path>>(
    library_path: P,
) -> Result<LoadedServeDispatchPolicy> {
    let lib_path = library_path.as_ref();
    if !lib_path.exists() {
        return Err(LociError::PluginError(format!(
            "Serve dispatch policy plugin library not found: {}",
            lib_path.display()
        )));
    }

    let library = unsafe {
        Library::new(lib_path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load serve dispatch policy plugin '{}': {}",
                lib_path.display(),
                e
            ))
        })?
    };

    let constructor: Symbol<ServeDispatchPolicyConstructor> = unsafe {
        match library.get(b"create_serve_dispatch_policy_v1") {
            Ok(sym) => sym,
            Err(_) => library.get(b"create_serve_dispatch_policy").map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to find serve dispatch policy constructor symbol \
                     ('create_serve_dispatch_policy_v1' or 'create_serve_dispatch_policy'): {}",
                    e
                ))
            })?,
        }
    };

    let plugin_opaque = unsafe { constructor() };
    let plugin =
        unsafe { dynamic_serve_dispatch_policy_from_opaque(plugin_opaque) }.ok_or_else(|| {
            LociError::PluginError(
                "Serve dispatch policy constructor returned invalid plugin payload".to_string(),
            )
        })?;
    if plugin.name().trim().is_empty() {
        return Err(LociError::PluginError(
            "Serve dispatch policy plugin returned empty name".to_string(),
        ));
    }

    Ok(LoadedServeDispatchPolicy {
        plugin: Arc::<dyn ServeDispatchPolicyPlugin>::from(plugin),
        library: Arc::new(library),
        source: lib_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockPolicy;

    impl ServeDispatchPolicyPlugin for MockPolicy {
        fn name(&self) -> &str {
            "mock.serve.dispatch"
        }

        fn on_queue_full(&self, _context: &QueuePressureContext) -> QueueFullAction {
            QueueFullAction::Reject
        }
    }

    #[test]
    fn dynamic_opaque_roundtrip() {
        let plugin: Box<dyn ServeDispatchPolicyPlugin> = Box::new(MockPolicy);
        let opaque = dynamic_serve_dispatch_policy_into_opaque(plugin);
        let restored = unsafe { dynamic_serve_dispatch_policy_from_opaque(opaque) };
        let restored = restored.expect("opaque roundtrip should restore plugin");
        assert_eq!(restored.name(), "mock.serve.dispatch");
        assert_eq!(
            restored.on_queue_full(&QueuePressureContext {
                attempt: 0,
                queue_len: 1,
                queue_capacity: 8
            }),
            QueueFullAction::Reject
        );
    }

    #[test]
    fn load_dynamic_missing_library_fails() {
        let err =
            match load_dynamic_serve_dispatch_policy("missing_serve_dispatch_policy_plugin.dll") {
                Ok(_) => panic!("missing plugin should fail"),
                Err(err) => err,
            };
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn registry_contains_builtin_policies() {
        let registry = ServeDispatchPolicyRegistry::with_builtin_policies();
        let names = registry.list_names();
        assert!(names.contains(&"reject".to_string()));
        assert!(names.contains(&"block".to_string()));
        assert!(registry.get("reject").is_some());
        assert!(registry.get("block").is_some());
        assert!(registry.list_dynamic_names().is_empty());
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let registry = ServeDispatchPolicyRegistry::new();
        registry
            .register_policy(RejectServeDispatchPolicyPlugin)
            .expect("first registration should succeed");
        let err = registry
            .register_policy(RejectServeDispatchPolicyPlugin)
            .expect_err("duplicate registration should fail");
        assert!(format!("{err}").contains("already registered"));
    }
}
