//! Dynamic model-pull policy plugin ABI and registry.
//!
//! This policy family governs which model asset sources may be imported into
//! the managed store and what integrity constraints must be satisfied.

use crate::error::{LociError, Result};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPullPolicyContext {
    pub source: String,
    pub mirrors: Vec<String>,
    pub requested_id: Option<String>,
    pub requested_name: Option<String>,
    pub expected_sha256: Option<String>,
    pub resume: bool,
    pub tags: Vec<String>,
}

impl ModelPullPolicyContext {
    pub fn new(
        source: impl Into<String>,
        mirrors: Vec<String>,
        requested_id: Option<String>,
        requested_name: Option<String>,
        expected_sha256: Option<String>,
        resume: bool,
        tags: Vec<String>,
    ) -> Self {
        Self {
            source: source.into(),
            mirrors,
            requested_id,
            requested_name,
            expected_sha256,
            resume,
            tags,
        }
    }

    pub fn all_sources(&self) -> impl Iterator<Item = &str> {
        self.mirrors
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(self.source.as_str()))
    }

    pub fn uses_remote_sources(&self) -> bool {
        self.all_sources().any(is_remote_model_source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPullPolicyDecision {
    Allow,
    Deny(String),
}

pub trait ModelPullPolicyPlugin: Send + Sync {
    fn name(&self) -> &str;

    fn authorize(&self, context: &ModelPullPolicyContext) -> ModelPullPolicyDecision;
}

pub struct AllowAllModelPullPolicy;

impl ModelPullPolicyPlugin for AllowAllModelPullPolicy {
    fn name(&self) -> &str {
        "allow-all.model.pull"
    }

    fn authorize(&self, _context: &ModelPullPolicyContext) -> ModelPullPolicyDecision {
        ModelPullPolicyDecision::Allow
    }
}

pub struct LocalOnlyModelPullPolicy;

impl ModelPullPolicyPlugin for LocalOnlyModelPullPolicy {
    fn name(&self) -> &str {
        "local-only.model.pull"
    }

    fn authorize(&self, context: &ModelPullPolicyContext) -> ModelPullPolicyDecision {
        if context.uses_remote_sources() {
            ModelPullPolicyDecision::Deny("remote model sources are disabled by policy".to_string())
        } else {
            ModelPullPolicyDecision::Allow
        }
    }
}

pub struct RequireChecksumForRemoteModelPullPolicy;

impl ModelPullPolicyPlugin for RequireChecksumForRemoteModelPullPolicy {
    fn name(&self) -> &str {
        "checksum-required-remote.model.pull"
    }

    fn authorize(&self, context: &ModelPullPolicyContext) -> ModelPullPolicyDecision {
        if context.uses_remote_sources()
            && context
                .expected_sha256
                .as_deref()
                .map(str::trim)
                .unwrap_or_default()
                .is_empty()
        {
            ModelPullPolicyDecision::Deny(
                "remote model sources require an expected sha256 checksum".to_string(),
            )
        } else {
            ModelPullPolicyDecision::Allow
        }
    }
}

pub fn is_remote_model_source(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

pub fn authorize_model_pull_request(
    policy: &dyn ModelPullPolicyPlugin,
    context: &ModelPullPolicyContext,
) -> std::result::Result<(), String> {
    match policy.authorize(context) {
        ModelPullPolicyDecision::Allow => Ok(()),
        ModelPullPolicyDecision::Deny(reason) => Err(reason),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicModelPullPolicyOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynModelPullPolicyPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

pub fn dynamic_model_pull_policy_into_opaque(
    policy: Box<dyn ModelPullPolicyPlugin>,
) -> DynamicModelPullPolicyOpaque {
    let raw: *mut dyn ModelPullPolicyPlugin = Box::into_raw(policy);
    let parts: RawDynModelPullPolicyPtr = unsafe { std::mem::transmute(raw) };
    DynamicModelPullPolicyOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

pub unsafe fn dynamic_model_pull_policy_from_opaque(
    opaque: DynamicModelPullPolicyOpaque,
) -> Option<Box<dyn ModelPullPolicyPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynModelPullPolicyPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ModelPullPolicyPlugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type ModelPullPolicyConstructor = unsafe extern "C" fn() -> DynamicModelPullPolicyOpaque;

pub struct LoadedModelPullPolicy {
    policy: Arc<dyn ModelPullPolicyPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

impl LoadedModelPullPolicy {
    pub fn policy(&self) -> &Arc<dyn ModelPullPolicyPlugin> {
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
pub struct ModelPullPolicyDescriptor {
    pub name: String,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

struct ModelPullPolicyEntry {
    policy: Arc<dyn ModelPullPolicyPlugin>,
    dynamic: Option<DynamicModelPullPolicyHandle>,
}

struct DynamicModelPullPolicyHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

pub struct ModelPullPolicyRegistry {
    policies: RwLock<HashMap<String, ModelPullPolicyEntry>>,
}

impl ModelPullPolicyRegistry {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_builtin_policies() -> Self {
        let registry = Self::new();
        registry
            .register_policy(AllowAllModelPullPolicy)
            .expect("register allow-all model pull policy");
        registry
            .register_policy(LocalOnlyModelPullPolicy)
            .expect("register local-only model pull policy");
        registry
            .register_policy(RequireChecksumForRemoteModelPullPolicy)
            .expect("register checksum-required-remote model pull policy");
        registry
    }

    pub fn register_policy<P>(&self, policy: P) -> Result<()>
    where
        P: ModelPullPolicyPlugin + 'static,
    {
        self.register_policy_arc(Arc::new(policy))
    }

    pub fn register_policy_arc(&self, policy: Arc<dyn ModelPullPolicyPlugin>) -> Result<()> {
        let key = policy.name().trim().to_string();
        if key.is_empty() {
            return Err(LociError::PluginError(
                "Model pull policy name cannot be empty".to_string(),
            ));
        }

        let mut policies = self.policies.write();
        if policies.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Model pull policy '{}' already registered",
                key
            )));
        }
        policies.insert(
            key,
            ModelPullPolicyEntry {
                policy,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ModelPullPolicyPlugin>> {
        self.policies
            .read()
            .get(name)
            .map(|entry| Arc::clone(&entry.policy))
    }

    pub fn describe(&self, name: &str) -> Option<ModelPullPolicyDescriptor> {
        self.policies
            .read()
            .get(name)
            .map(|entry| ModelPullPolicyDescriptor {
                name: name.to_string(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
    }

    pub fn descriptors(&self) -> Vec<ModelPullPolicyDescriptor> {
        let policies = self.policies.read();
        let mut descriptors = policies
            .iter()
            .map(|(name, entry)| ModelPullPolicyDescriptor {
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
            .map(|entry| entry.name)
            .collect()
    }

    fn load_dynamic_entry(&self, library_path: PathBuf) -> Result<(String, ModelPullPolicyEntry)> {
        let loaded = load_dynamic_model_pull_policy(&library_path)?;
        let name = loaded.name().to_string();
        let entry = ModelPullPolicyEntry {
            policy: Arc::clone(loaded.policy()),
            dynamic: Some(DynamicModelPullPolicyHandle {
                library: Arc::clone(&loaded.library),
                source: loaded.source().to_path_buf(),
            }),
        };
        Ok((name, entry))
    }

    pub fn load_dynamic_policy<P: AsRef<Path>>(&self, library_path: P) -> Result<String> {
        let library_path = library_path.as_ref().to_path_buf();
        let (name, entry) = self.load_dynamic_entry(library_path)?;
        let mut policies = self.policies.write();
        if policies.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Model pull policy '{}' already registered",
                name
            )));
        }
        policies.insert(name.clone(), entry);
        Ok(name)
    }

    pub fn unload_dynamic_policy(&self, name: &str) -> Result<()> {
        let mut policies = self.policies.write();
        let entry = policies.get(name).ok_or_else(|| {
            LociError::PluginError(format!("Model pull policy '{}' not found", name))
        })?;
        if entry.dynamic.is_none() {
            return Err(LociError::PluginError(format!(
                "Model pull policy '{}' is builtin and cannot be unloaded",
                name
            )));
        }
        policies.remove(name);
        Ok(())
    }

    pub fn reload_dynamic_policy(&self, name: &str) -> Result<()> {
        let source = {
            let policies = self.policies.read();
            let entry = policies.get(name).ok_or_else(|| {
                LociError::PluginError(format!("Model pull policy '{}' not found", name))
            })?;
            entry
                .dynamic
                .as_ref()
                .map(|dynamic| dynamic.source.clone())
                .ok_or_else(|| {
                    LociError::PluginError(format!(
                        "Model pull policy '{}' is builtin and cannot be reloaded",
                        name
                    ))
                })?
        };

        let (new_name, entry) = self.load_dynamic_entry(source)?;
        if new_name != name {
            return Err(LociError::PluginError(format!(
                "Reloaded model pull policy renamed from '{}' to '{}'",
                name, new_name
            )));
        }

        self.policies.write().insert(new_name, entry);
        Ok(())
    }
}

impl Default for ModelPullPolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn load_dynamic_model_pull_policy<P: AsRef<Path>>(path: P) -> Result<LoadedModelPullPolicy> {
    let lib_path = path.as_ref();
    let manifest =
        load_and_validate_plugin_contract(lib_path, PluginContractKind::ModelPullPolicy)?;
    let library = unsafe {
        Arc::new(Library::new(lib_path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load model pull policy plugin '{}': {}",
                lib_path.display(),
                e
            ))
        })?)
    };

    let constructor: Symbol<ModelPullPolicyConstructor> = unsafe {
        library.get(b"loci_create_model_pull_policy").map_err(|e| {
            LociError::PluginError(format!(
                "Model pull policy plugin '{}' is missing loci_create_model_pull_policy: {}",
                lib_path.display(),
                e
            ))
        })?
    };
    let opaque = unsafe { constructor() };
    let policy = unsafe { dynamic_model_pull_policy_from_opaque(opaque) }.ok_or_else(|| {
        LociError::PluginError(format!(
            "Model pull policy plugin '{}' returned a null policy",
            lib_path.display()
        ))
    })?;

    validate_runtime_plugin_identity(manifest.as_ref(), policy.name(), "")?;

    Ok(LoadedModelPullPolicy {
        policy: Arc::<dyn ModelPullPolicyPlugin>::from(policy),
        library,
        source: lib_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DenyRemotePolicy;

    impl ModelPullPolicyPlugin for DenyRemotePolicy {
        fn name(&self) -> &str {
            "deny-remote.test"
        }

        fn authorize(&self, context: &ModelPullPolicyContext) -> ModelPullPolicyDecision {
            if context.uses_remote_sources() {
                ModelPullPolicyDecision::Deny("remote denied".to_string())
            } else {
                ModelPullPolicyDecision::Allow
            }
        }
    }

    #[test]
    fn builtin_checksum_policy_rejects_remote_without_sha256() {
        let policy = RequireChecksumForRemoteModelPullPolicy;
        let context = ModelPullPolicyContext::new(
            "https://example.com/model.gguf",
            Vec::new(),
            None,
            None,
            None,
            true,
            vec!["managed".to_string()],
        );
        let result = authorize_model_pull_request(&policy, &context);
        assert_eq!(
            result,
            Err("remote model sources require an expected sha256 checksum".to_string())
        );
    }

    #[test]
    fn registry_registers_and_lists_model_pull_policies() {
        let registry = ModelPullPolicyRegistry::with_builtin_policies();
        registry.register_policy(DenyRemotePolicy).unwrap();
        let names = registry.list_names();
        assert!(names.iter().any(|name| name == "allow-all.model.pull"));
        assert!(names.iter().any(|name| name == "deny-remote.test"));
    }
}
