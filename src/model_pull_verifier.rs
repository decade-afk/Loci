//! Dynamic model-pull verifier plugin ABI and registry.
//!
//! Verifiers run after a model asset has been fetched and checksummed, allowing
//! hosts to enforce sidecar proofs, detached signatures, or other trust rules
//! without coupling that logic to the core model store.

use crate::error::{LociError, Result};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPullVerificationContext {
    pub requested_source: String,
    pub selected_source: String,
    pub mirrors: Vec<String>,
    pub requested_id: Option<String>,
    pub requested_name: Option<String>,
    pub expected_sha256: Option<String>,
    pub tags: Vec<String>,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub checksum_sha256: String,
}

impl ModelPullVerificationContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        requested_source: impl Into<String>,
        selected_source: impl Into<String>,
        mirrors: Vec<String>,
        requested_id: Option<String>,
        requested_name: Option<String>,
        expected_sha256: Option<String>,
        tags: Vec<String>,
        path: PathBuf,
        size_bytes: u64,
        checksum_sha256: impl Into<String>,
    ) -> Self {
        Self {
            requested_source: requested_source.into(),
            selected_source: selected_source.into(),
            mirrors,
            requested_id,
            requested_name,
            expected_sha256,
            tags,
            path,
            size_bytes,
            checksum_sha256: checksum_sha256.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelPullVerifierDecision {
    Allow,
    Deny(String),
}

pub trait ModelPullVerifierPlugin: Send + Sync {
    fn name(&self) -> &str;

    fn verify(&self, context: &ModelPullVerificationContext) -> ModelPullVerifierDecision;
}

pub struct AllowAllModelPullVerifier;

impl ModelPullVerifierPlugin for AllowAllModelPullVerifier {
    fn name(&self) -> &str {
        "allow-all.model.verify"
    }

    fn verify(&self, _context: &ModelPullVerificationContext) -> ModelPullVerifierDecision {
        ModelPullVerifierDecision::Allow
    }
}

pub struct SidecarSha256ModelPullVerifier;

impl ModelPullVerifierPlugin for SidecarSha256ModelPullVerifier {
    fn name(&self) -> &str {
        "sidecar-sha256.model.verify"
    }

    fn verify(&self, context: &ModelPullVerificationContext) -> ModelPullVerifierDecision {
        let expected = match load_sidecar_sha256(&context.selected_source) {
            Ok(expected) => expected,
            Err(err) => return ModelPullVerifierDecision::Deny(err),
        };

        if expected == context.checksum_sha256 {
            ModelPullVerifierDecision::Allow
        } else {
            ModelPullVerifierDecision::Deny(format!(
                "sidecar sha256 mismatch: expected {}, got {}",
                expected, context.checksum_sha256
            ))
        }
    }
}

pub fn verify_model_pull(
    verifier: &dyn ModelPullVerifierPlugin,
    context: &ModelPullVerificationContext,
) -> std::result::Result<(), String> {
    match verifier.verify(context) {
        ModelPullVerifierDecision::Allow => Ok(()),
        ModelPullVerifierDecision::Deny(reason) => Err(reason),
    }
}

fn load_sidecar_sha256(source: &str) -> std::result::Result<String, String> {
    if is_remote_source(source) {
        let url = format!("{}.sha256", strip_url_query_and_fragment(source));
        let body = Client::builder()
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| format!("failed creating HTTP client for sidecar fetch: {e}"))?
            .get(&url)
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(|e| format!("failed fetching sidecar sha256 '{}': {}", url, e))?
            .text()
            .map_err(|e| format!("failed reading sidecar sha256 '{}': {}", url, e))?;
        parse_sidecar_sha256(&body)
    } else {
        let sidecar_path = PathBuf::from(format!("{}.sha256", source));
        let body = fs::read_to_string(&sidecar_path).map_err(|e| {
            format!(
                "failed reading sidecar sha256 '{}': {}",
                sidecar_path.display(),
                e
            )
        })?;
        parse_sidecar_sha256(&body)
    }
}

fn parse_sidecar_sha256(raw: &str) -> std::result::Result<String, String> {
    let candidate = raw
        .split_whitespace()
        .next()
        .ok_or_else(|| "sidecar sha256 file is empty".to_string())?
        .trim()
        .to_ascii_lowercase();
    if candidate.len() != 64 || !candidate.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid sidecar sha256 '{}': expected 64 hex characters",
            candidate
        ));
    }
    Ok(candidate)
}

fn strip_url_query_and_fragment(url: &str) -> &str {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    &url[..end]
}

fn is_remote_source(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicModelPullVerifierOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynModelPullVerifierPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

pub fn dynamic_model_pull_verifier_into_opaque(
    verifier: Box<dyn ModelPullVerifierPlugin>,
) -> DynamicModelPullVerifierOpaque {
    let raw: *mut dyn ModelPullVerifierPlugin = Box::into_raw(verifier);
    let parts: RawDynModelPullVerifierPtr = unsafe { std::mem::transmute(raw) };
    DynamicModelPullVerifierOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

pub unsafe fn dynamic_model_pull_verifier_from_opaque(
    opaque: DynamicModelPullVerifierOpaque,
) -> Option<Box<dyn ModelPullVerifierPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynModelPullVerifierPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ModelPullVerifierPlugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type ModelPullVerifierConstructor = unsafe extern "C" fn() -> DynamicModelPullVerifierOpaque;

pub struct LoadedModelPullVerifier {
    verifier: Arc<dyn ModelPullVerifierPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

impl LoadedModelPullVerifier {
    pub fn verifier(&self) -> &Arc<dyn ModelPullVerifierPlugin> {
        &self.verifier
    }

    pub fn name(&self) -> &str {
        self.verifier.name()
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPullVerifierDescriptor {
    pub name: String,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

struct ModelPullVerifierEntry {
    verifier: Arc<dyn ModelPullVerifierPlugin>,
    dynamic: Option<DynamicModelPullVerifierHandle>,
}

struct DynamicModelPullVerifierHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

pub struct ModelPullVerifierRegistry {
    verifiers: RwLock<HashMap<String, ModelPullVerifierEntry>>,
}

impl ModelPullVerifierRegistry {
    pub fn new() -> Self {
        Self {
            verifiers: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_builtin_verifiers() -> Self {
        let registry = Self::new();
        registry
            .register_verifier(AllowAllModelPullVerifier)
            .expect("register allow-all model verifier");
        registry
            .register_verifier(SidecarSha256ModelPullVerifier)
            .expect("register sidecar sha256 model verifier");
        registry
    }

    pub fn register_verifier<V>(&self, verifier: V) -> Result<()>
    where
        V: ModelPullVerifierPlugin + 'static,
    {
        self.register_verifier_arc(Arc::new(verifier))
    }

    pub fn register_verifier_arc(&self, verifier: Arc<dyn ModelPullVerifierPlugin>) -> Result<()> {
        let key = verifier.name().trim().to_string();
        if key.is_empty() {
            return Err(LociError::PluginError(
                "Model pull verifier name cannot be empty".to_string(),
            ));
        }

        let mut verifiers = self.verifiers.write();
        if verifiers.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Model pull verifier '{}' already registered",
                key
            )));
        }
        verifiers.insert(
            key,
            ModelPullVerifierEntry {
                verifier,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ModelPullVerifierPlugin>> {
        self.verifiers
            .read()
            .get(name)
            .map(|entry| Arc::clone(&entry.verifier))
    }

    pub fn describe(&self, name: &str) -> Option<ModelPullVerifierDescriptor> {
        self.verifiers
            .read()
            .get(name)
            .map(|entry| ModelPullVerifierDescriptor {
                name: name.to_string(),
                dynamic: entry.dynamic.is_some(),
                source: entry.dynamic.as_ref().map(|dynamic| dynamic.source.clone()),
            })
    }

    pub fn descriptors(&self) -> Vec<ModelPullVerifierDescriptor> {
        let verifiers = self.verifiers.read();
        let mut descriptors = verifiers
            .iter()
            .map(|(name, entry)| ModelPullVerifierDescriptor {
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

    fn load_dynamic_entry(
        &self,
        library_path: PathBuf,
    ) -> Result<(String, ModelPullVerifierEntry)> {
        let loaded = load_dynamic_model_pull_verifier(&library_path)?;
        let name = loaded.name().to_string();
        let entry = ModelPullVerifierEntry {
            verifier: Arc::clone(loaded.verifier()),
            dynamic: Some(DynamicModelPullVerifierHandle {
                library: Arc::clone(&loaded.library),
                source: loaded.source().to_path_buf(),
            }),
        };
        Ok((name, entry))
    }

    pub fn load_dynamic_verifier<P: AsRef<Path>>(&self, library_path: P) -> Result<String> {
        let library_path = library_path.as_ref().to_path_buf();
        let (name, entry) = self.load_dynamic_entry(library_path)?;
        let mut verifiers = self.verifiers.write();
        if verifiers.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Model pull verifier '{}' already registered",
                name
            )));
        }
        verifiers.insert(name.clone(), entry);
        Ok(name)
    }

    pub fn unload_dynamic_verifier(&self, name: &str) -> Result<()> {
        let mut verifiers = self.verifiers.write();
        let entry = verifiers.get(name).ok_or_else(|| {
            LociError::PluginError(format!("Model pull verifier '{}' not found", name))
        })?;
        if entry.dynamic.is_none() {
            return Err(LociError::PluginError(format!(
                "Model pull verifier '{}' is builtin and cannot be unloaded",
                name
            )));
        }
        verifiers.remove(name);
        Ok(())
    }

    pub fn reload_dynamic_verifier(&self, name: &str) -> Result<()> {
        let source = {
            let verifiers = self.verifiers.read();
            let entry = verifiers.get(name).ok_or_else(|| {
                LociError::PluginError(format!("Model pull verifier '{}' not found", name))
            })?;
            entry
                .dynamic
                .as_ref()
                .map(|dynamic| dynamic.source.clone())
                .ok_or_else(|| {
                    LociError::PluginError(format!(
                        "Model pull verifier '{}' is builtin and cannot be reloaded",
                        name
                    ))
                })?
        };

        let (new_name, entry) = self.load_dynamic_entry(source)?;
        if new_name != name {
            return Err(LociError::PluginError(format!(
                "Reloaded model pull verifier renamed from '{}' to '{}'",
                name, new_name
            )));
        }
        self.verifiers.write().insert(new_name, entry);
        Ok(())
    }
}

impl Default for ModelPullVerifierRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn load_dynamic_model_pull_verifier<P: AsRef<Path>>(
    path: P,
) -> Result<LoadedModelPullVerifier> {
    let lib_path = path.as_ref();
    let manifest =
        load_and_validate_plugin_contract(lib_path, PluginContractKind::ModelPullVerifier)?;
    let library = unsafe {
        Arc::new(Library::new(lib_path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load model pull verifier plugin '{}': {}",
                lib_path.display(),
                e
            ))
        })?)
    };

    let constructor: Symbol<ModelPullVerifierConstructor> = unsafe {
        library
            .get(b"loci_create_model_pull_verifier")
            .map_err(|e| {
                LociError::PluginError(format!(
                "Model pull verifier plugin '{}' is missing loci_create_model_pull_verifier: {}",
                lib_path.display(),
                e
            ))
            })?
    };
    let opaque = unsafe { constructor() };
    let verifier = unsafe { dynamic_model_pull_verifier_from_opaque(opaque) }.ok_or_else(|| {
        LociError::PluginError(format!(
            "Model pull verifier plugin '{}' returned a null verifier",
            lib_path.display()
        ))
    })?;

    validate_runtime_plugin_identity(manifest.as_ref(), verifier.name(), "")?;

    Ok(LoadedModelPullVerifier {
        verifier: Arc::<dyn ModelPullVerifierPlugin>::from(verifier),
        library,
        source: lib_path.to_path_buf(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "loci-model-pull-verifier-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn sidecar_verifier_accepts_matching_local_sidecar() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("model.gguf");
        fs::write(&source, b"data").unwrap();
        fs::write(
            root.join("model.gguf.sha256"),
            "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7  model.gguf\n",
        )
        .unwrap();

        let verifier = SidecarSha256ModelPullVerifier;
        let context = ModelPullVerificationContext::new(
            source.to_string_lossy(),
            source.to_string_lossy(),
            Vec::new(),
            None,
            None,
            None,
            vec!["managed".to_string()],
            source.clone(),
            4,
            "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
        );

        assert_eq!(verify_model_pull(&verifier, &context), Ok(()));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sidecar_verifier_rejects_missing_remote_sidecar() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request_buf = [0u8; 1024];
                let _ = stream.read(&mut request_buf);
                let response =
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        let source = format!("http://{}/model.gguf", addr);
        let verifier = SidecarSha256ModelPullVerifier;
        let context = ModelPullVerificationContext::new(
            source.clone(),
            source,
            Vec::new(),
            None,
            None,
            None,
            vec!["managed".to_string()],
            PathBuf::from("model.gguf"),
            0,
            "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
        );

        assert!(verify_model_pull(&verifier, &context).is_err());
    }
}
