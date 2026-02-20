use crate::backend::{BackendCapabilities, BackendParams, InferenceBackend, Model};
use crate::error::{LociError, Result};
use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[allow(improper_ctypes_definitions)]
type BackendConstructor = unsafe extern "C" fn() -> *mut dyn InferenceBackend;

pub struct DynamicBackend {
    backend: Box<dyn InferenceBackend>,
    _library: Arc<Library>,
    source_path: PathBuf,
}

impl DynamicBackend {
    pub fn load<P: AsRef<Path>>(library_path: P) -> Result<Self> {
        let source_path = library_path.as_ref().to_path_buf();

        let library = unsafe { Library::new(&source_path) }.map_err(|e| {
            LociError::BackendError(format!(
                "Failed to load backend library {}: {}",
                source_path.display(),
                e
            ))
        })?;

        let library = Arc::new(library);
        let backend = unsafe {
            let constructor: Symbol<BackendConstructor> = library
                .get(b"create_backend")
                .map_err(|e| {
                    LociError::BackendError(format!(
                        "Missing create_backend symbol in {}: {}",
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
        };

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

    #[test]
    fn test_dynamic_backend_load_missing_file() {
        let result = DynamicBackend::load("does_not_exist_backend.dll");
        assert!(result.is_err());
    }
}
