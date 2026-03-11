use loci::backend::{BackendCapabilities, BackendParams, InferenceBackend, Model};
use loci::backends::dynamic::{dynamic_backend_into_opaque, DynamicBackendOpaque};
use loci::backends::LlamaCppBackend;
use loci::error::Result;
use std::path::Path;
use std::sync::Mutex;

pub struct PluginLlamaBackend {
    inner: Mutex<LlamaCppBackend>,
}

impl PluginLlamaBackend {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(LlamaCppBackend::new()),
        }
    }
}

impl InferenceBackend for PluginLlamaBackend {
    fn capabilities(&self) -> BackendCapabilities {
        let mut caps = self
            .inner
            .lock()
            .expect("backend mutex poisoned")
            .capabilities();
        caps.name = "plugin.llama.cpp".to_string();
        caps.version = "1.0.0".to_string();
        caps
    }

    fn load_model(
        &self,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        self.inner
            .lock()
            .expect("backend mutex poisoned")
            .load_model(model_path, backend_params)
    }

    fn init(&mut self) -> Result<()> {
        self.inner.get_mut().expect("backend mutex poisoned").init()
    }

    fn shutdown(&mut self) -> Result<()> {
        self.inner
            .get_mut()
            .expect("backend mutex poisoned")
            .shutdown()
    }
}

#[no_mangle]
pub extern "C" fn create_backend_v1() -> DynamicBackendOpaque {
    dynamic_backend_into_opaque(Box::new(PluginLlamaBackend::new()))
}

#[no_mangle]
pub extern "C" fn create_backend() -> DynamicBackendOpaque {
    create_backend_v1()
}
