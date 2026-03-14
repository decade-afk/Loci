use crate::backend::{BackendCapabilities, BackendParams, InferenceBackend, Model};
use crate::error::{LociError, Result};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DynamicBackendOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynBackendPtr {
    data: *mut (),
    vtable: *mut (),
}

pub fn dynamic_backend_into_opaque(backend: Box<dyn InferenceBackend>) -> DynamicBackendOpaque {
    let raw: *mut dyn InferenceBackend = Box::into_raw(backend);
    let parts: RawDynBackendPtr = unsafe { std::mem::transmute(raw) };
    DynamicBackendOpaque {
        data: parts.data.cast::<c_void>(),
        vtable: parts.vtable.cast::<c_void>(),
    }
}

/// Reconstruct a backend from an opaque payload.
///
/// # Safety
/// The payload must be produced by `dynamic_backend_into_opaque` using the same ABI.
pub unsafe fn dynamic_backend_from_opaque(
    opaque: DynamicBackendOpaque,
) -> Option<Box<dyn InferenceBackend>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }
    let parts = RawDynBackendPtr {
        data: opaque.data.cast::<()>(),
        vtable: opaque.vtable.cast::<()>(),
    };
    let raw: *mut dyn InferenceBackend = unsafe { std::mem::transmute(parts) };
    Some(unsafe { Box::from_raw(raw) })
}

type BackendConstructorV1 = unsafe extern "C" fn() -> DynamicBackendOpaque;

#[allow(improper_ctypes_definitions)]
type LegacyBackendConstructor = unsafe extern "C" fn() -> *mut dyn InferenceBackend;

pub struct DynamicBackend {
    backend: Box<dyn InferenceBackend>,
    _library: Arc<Library>,
    source_path: PathBuf,
}

impl DynamicBackend {
    pub fn load<P: AsRef<Path>>(library_path: P) -> Result<Self> {
        let source_path = library_path.as_ref().to_path_buf();
        let manifest =
            load_and_validate_plugin_contract(&source_path, PluginContractKind::Backend)?;

        let library = unsafe { Library::new(&source_path) }.map_err(|e| {
            LociError::BackendError(format!(
                "Failed to load backend library {}: {}",
                source_path.display(),
                e
            ))
        })?;

        let library = Arc::new(library);
        let backend = unsafe {
            if let Ok(constructor_v1) = library.get::<BackendConstructorV1>(b"create_backend_v1") {
                let backend_opaque = constructor_v1();
                dynamic_backend_from_opaque(backend_opaque).ok_or_else(|| {
                    LociError::BackendError(format!(
                        "Dynamic backend constructor returned invalid payload: {}",
                        source_path.display()
                    ))
                })?
            } else {
                let constructor: Symbol<LegacyBackendConstructor> =
                    library.get(b"create_backend").map_err(|e| {
                        LociError::BackendError(format!(
                            "Missing create_backend_v1/create_backend symbol in {}: {}",
                            source_path.display(),
                            e
                        ))
                    })?;

                let backend_ptr = constructor();
                if backend_ptr.is_null() {
                    return Err(LociError::BackendError(format!(
                        "Dynamic backend constructor returned null: {}",
                        source_path.display()
                    )));
                }

                Box::from_raw(backend_ptr)
            }
        };

        let capabilities = backend.capabilities();
        validate_runtime_plugin_identity(
            manifest.as_ref(),
            &capabilities.name,
            &capabilities.version,
        )?;

        Ok(Self {
            backend,
            _library: library,
            source_path,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }
}

impl InferenceBackend for DynamicBackend {
    fn capabilities(&self) -> BackendCapabilities {
        self.backend.capabilities()
    }

    fn load_model(
        &self,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        self.backend.load_model(model_path, backend_params)
    }

    fn init(&mut self) -> Result<()> {
        self.backend.init()
    }

    fn shutdown(&mut self) -> Result<()> {
        self.backend.shutdown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as LociResult;
    use std::path::Path;

    struct MockBackend;

    impl InferenceBackend for MockBackend {
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                name: "mock".to_string(),
                version: "1.0".to_string(),
                supports_text: true,
                supports_multimodal: false,
                supports_embeddings: false,
                supports_streaming: false,
                has_gpu_support: false,
                supported_formats: vec!["mock".to_string()],
            }
        }

        fn load_model(
            &self,
            _model_path: &Path,
            _backend_params: BackendParams,
        ) -> LociResult<Box<dyn Model>> {
            Err(LociError::UnsupportedOperation(
                "mock backend does not implement models".to_string(),
            ))
        }
    }

    #[test]
    fn test_dynamic_backend_load_missing_file() {
        let result = DynamicBackend::load("does_not_exist_backend.dll");
        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_backend_opaque_roundtrip() {
        let backend: Box<dyn InferenceBackend> = Box::new(MockBackend);
        let opaque = dynamic_backend_into_opaque(backend);
        let restored = unsafe { dynamic_backend_from_opaque(opaque) };
        let restored = restored.expect("opaque payload should reconstruct");
        assert_eq!(restored.capabilities().name, "mock");
    }

    #[test]
    fn test_dynamic_backend_opaque_rejects_null() {
        let restored = unsafe {
            dynamic_backend_from_opaque(DynamicBackendOpaque {
                data: std::ptr::null_mut(),
                vtable: std::ptr::null_mut(),
            })
        };
        assert!(restored.is_none());
    }
}
