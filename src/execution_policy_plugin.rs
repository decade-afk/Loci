//! Dynamic execution policy plugin ABI and registry.
//!
//! This module mirrors the serve-dispatch registry pattern so host
//! applications can swap scheduling/execution behavior without rebuilding the
//! engine core.

use crate::error::{LociError, Result};
use crate::inference::{DefaultExecutionPolicy, ExecutionPolicy};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicExecutionPolicyOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynExecutionPolicyPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn ExecutionPolicy>` into ABI-safe opaque payload.
pub fn dynamic_execution_policy_into_opaque(
    policy: Box<dyn ExecutionPolicy>,
) -> DynamicExecutionPolicyOpaque {
    let raw: *mut dyn ExecutionPolicy = Box::into_raw(policy);
    let parts: RawDynExecutionPolicyPtr = unsafe { std::mem::transmute(raw) };
    DynamicExecutionPolicyOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert opaque payload back into `Box<dyn ExecutionPolicy>`.
///
/// # Safety
/// The payload must come from `dynamic_execution_policy_into_opaque`
/// under compatible Rust toolchain/target ABI.
pub unsafe fn dynamic_execution_policy_from_opaque(
    opaque: DynamicExecutionPolicyOpaque,
) -> Option<Box<dyn ExecutionPolicy>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynExecutionPolicyPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ExecutionPolicy = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type ExecutionPolicyConstructor = unsafe extern "C" fn() -> DynamicExecutionPolicyOpaque;

/// Loaded dynamic execution policy plugin.
pub struct LoadedExecutionPolicy {
    policy: Arc<dyn ExecutionPolicy>,
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

impl LoadedExecutionPolicy {
    pub fn policy(&self) -> &Arc<dyn ExecutionPolicy> {
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
pub struct ExecutionPolicyDescriptor {
    pub name: String,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

struct ExecutionPolicyEntry {
    policy: Arc<dyn ExecutionPolicy>,
    dynamic: Option<DynamicExecutionPolicyHandle>,
}

struct DynamicExecutionPolicyHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

/// Registry for builtin and dynamically loaded execution policies.
pub struct ExecutionPolicyRegistry {
    policies: RwLock<HashMap<String, ExecutionPolicyEntry>>,
}

impl ExecutionPolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_builtin_policies() -> Self {
        let registry = Self::new();
        registry
            .register_policy(DefaultExecutionPolicy::new())
            .expect("register default execution policy");
        registry
    }

    pub fn register_policy<P>(&self, policy: P) -> Result<()>
    where
        P: ExecutionPolicy + 'static,
    {
        self.register_policy_arc(Arc::new(policy))
    }

    pub fn register_policy_arc(&self, policy: Arc<dyn ExecutionPolicy>) -> Result<()> {
        let key = policy.name().trim().to_string();
        if key.is_empty() {
            return Err(LociError::PluginError(
                "Execution policy name cannot be empty".to_string(),
            ));
        }

        let mut policies = self.policies.write();
        if policies.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Execution policy '{}' already registered",
                key
            )));
        }
        policies.insert(
            key,
            ExecutionPolicyEntry {
                policy,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ExecutionPolicy>> {
        self.policies
            .read()
            .get(name)
            .map(|entry| Arc::clone(&entry.policy))
    }

    pub fn describe(&self, name: &str) -> Option<ExecutionPolicyDescriptor> {
        self.policies
            .read()
            .get(name)
            .map(|entry| ExecutionPolicyDescriptor {
                name: name.to_string(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
    }

    pub fn descriptors(&self) -> Vec<ExecutionPolicyDescriptor> {
        let mut descriptors = self
            .policies
            .read()
            .iter()
            .map(|(name, entry)| ExecutionPolicyDescriptor {
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
    ) -> Result<(String, ExecutionPolicyEntry)> {
        let loaded = load_dynamic_execution_policy(&library_path)?;
        let key = loaded.name().to_string();
        let entry = ExecutionPolicyEntry {
            policy: Arc::clone(loaded.policy()),
            dynamic: Some(DynamicExecutionPolicyHandle {
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
                "Execution policy '{}' already registered",
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
                        "Static execution policy '{}' cannot be unloaded at runtime",
                        name
                    )));
                }
            }
            None => {
                return Err(LociError::PluginError(format!(
                    "Execution policy '{}' not found",
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
                LociError::PluginError(format!("Execution policy '{}' not found", name))
            })?;
            let dynamic = entry.dynamic.as_ref().ok_or_else(|| {
                LociError::PluginError(format!(
                    "Static execution policy '{}' cannot be hot-reloaded",
                    name
                ))
            })?;
            dynamic.source.clone()
        };

        let (loaded_name, entry) = Self::load_dynamic_entry(&source)?;
        if loaded_name != name {
            return Err(LociError::PluginError(format!(
                "Reloaded execution policy name mismatch: expected '{}', got '{}'",
                name, loaded_name
            )));
        }

        let mut policies = self.policies.write();
        if !policies.contains_key(name) {
            return Err(LociError::PluginError(format!(
                "Execution policy '{}' not found during reload",
                name
            )));
        }
        policies.insert(name.to_string(), entry);
        Ok(())
    }
}

impl Default for ExecutionPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Load dynamic execution policy plugin from shared library.
///
/// Expected exported symbol:
/// - `create_execution_policy_v1() -> DynamicExecutionPolicyOpaque`
/// Fallback:
/// - `create_execution_policy() -> DynamicExecutionPolicyOpaque`
pub fn load_dynamic_execution_policy<P: AsRef<Path>>(
    library_path: P,
) -> Result<LoadedExecutionPolicy> {
    let lib_path = library_path.as_ref();
    let manifest =
        load_and_validate_plugin_contract(lib_path, PluginContractKind::ExecutionPolicy)?;
    if !lib_path.exists() {
        return Err(LociError::PluginError(format!(
            "Execution policy plugin library not found: {}",
            lib_path.display()
        )));
    }

    let library = unsafe {
        Library::new(lib_path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load execution policy plugin '{}': {}",
                lib_path.display(),
                e
            ))
        })?
    };

    let constructor: Symbol<ExecutionPolicyConstructor> = unsafe {
        match library.get(b"create_execution_policy_v1") {
            Ok(sym) => sym,
            Err(_) => library.get(b"create_execution_policy").map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to find execution policy constructor symbol \
                     ('create_execution_policy_v1' or 'create_execution_policy'): {}",
                    e
                ))
            })?,
        }
    };

    let policy_opaque = unsafe { constructor() };
    let policy =
        unsafe { dynamic_execution_policy_from_opaque(policy_opaque) }.ok_or_else(|| {
            LociError::PluginError(
                "Execution policy constructor returned invalid policy payload".to_string(),
            )
        })?;
    if policy.name().trim().is_empty() {
        return Err(LociError::PluginError(
            "Execution policy plugin returned empty name".to_string(),
        ));
    }
    validate_runtime_plugin_identity(manifest.as_ref(), policy.name(), "")?;

    Ok(LoadedExecutionPolicy {
        policy: Arc::<dyn ExecutionPolicy>::from(policy),
        library: Arc::new(library),
        source: lib_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceParams;
    use crate::inference::InferenceEngine;
    use std::time::Duration;

    struct PrefixExecutionPolicy;

    impl ExecutionPolicy for PrefixExecutionPolicy {
        fn name(&self) -> &str {
            "mock.execution.policy"
        }

        fn generate_text(
            &self,
            _engine: &mut InferenceEngine,
            prompt: &str,
            _params: &InferenceParams,
            _timeout_override: Option<Duration>,
        ) -> Result<String> {
            Ok(format!("prefix:{prompt}"))
        }

        fn generate_stream(
            &self,
            _engine: &mut InferenceEngine,
            prompt: &str,
            _params: &InferenceParams,
            _timeout_override: Option<Duration>,
            callback: &mut dyn FnMut(&str) -> bool,
        ) -> Result<()> {
            callback(prompt);
            Ok(())
        }
    }

    #[test]
    fn dynamic_opaque_roundtrip() {
        let policy: Box<dyn ExecutionPolicy> = Box::new(PrefixExecutionPolicy);
        let opaque = dynamic_execution_policy_into_opaque(policy);
        let restored = unsafe { dynamic_execution_policy_from_opaque(opaque) };
        let restored = restored.expect("opaque roundtrip should restore policy");
        assert_eq!(restored.name(), "mock.execution.policy");
    }

    #[test]
    fn load_missing_dynamic_policy_returns_error() {
        let err = match load_dynamic_execution_policy("missing_execution_policy_plugin.dll") {
            Ok(_) => panic!("missing library should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("not found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn builtin_registry_contains_default_policy() {
        let registry = ExecutionPolicyRegistry::with_builtin_policies();
        let names = registry.list_names();
        assert!(names.contains(&"default.execution.policy".to_string()));
    }

    #[test]
    fn registry_registers_custom_policy() {
        let registry = ExecutionPolicyRegistry::new();
        registry
            .register_policy(PrefixExecutionPolicy)
            .expect("register custom policy");
        let descriptor = registry
            .describe("mock.execution.policy")
            .expect("policy descriptor");
        assert!(!descriptor.dynamic);
        assert!(descriptor.source.is_none());
    }
}
