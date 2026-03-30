use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model};
use crate::backends::MockBackend;
use crate::core::{CoreRegistry, DefaultCoreRegistry};
use crate::error::{LociError, Result};
use crate::plugin::RegisteredPlugin;
use std::path::{Path, PathBuf};

pub struct InferenceEngine {
    registry: Box<dyn CoreRegistry>,
    backend_registry: BackendRegistry,
    active_backend: Option<String>,
    model: Option<Box<dyn Model>>,
    model_path: Option<PathBuf>,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::default()
    }

    pub fn register_plugin(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        self.registry
            .plugin_manager_mut()
            .register(plugin)
            .map_err(LociError::from)
    }

    pub fn run_command(&self, command: &str) -> Result<String> {
        self.registry.event_bus().publish(command)?;
        Ok(format!("command accepted: {command}"))
    }

    pub fn plugin_count(&self) -> usize {
        self.registry.plugin_manager().list().len()
    }

    pub fn load_model<P: AsRef<Path>>(
        &mut self,
        backend_name: &str,
        model_path: P,
        backend_params: BackendParams,
    ) -> Result<()> {
        let model_path = model_path.as_ref().to_path_buf();
        let model =
            self.backend_registry
                .load_model(backend_name, &model_path, backend_params)?;
        self.active_backend = Some(backend_name.to_string());
        self.model = Some(model);
        self.model_path = Some(model_path);
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        model.infer_text(prompt, params)
    }

    pub fn active_backend(&self) -> Option<&str> {
        self.active_backend.as_deref()
    }

    pub fn model_metadata(&self) -> Option<crate::backend::ModelMetadata> {
        self.model.as_ref().map(|model| model.metadata())
    }
}

#[derive(Default)]
pub struct InferenceEngineBuilder {
    registry: Option<Box<dyn CoreRegistry>>,
    backend_registry: Option<BackendRegistry>,
    backend_name: Option<String>,
    model_path: Option<PathBuf>,
    backend_params: BackendParams,
}

impl InferenceEngineBuilder {
    pub fn with_registry(mut self, registry: Box<dyn CoreRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn with_backend_registry(mut self, backend_registry: BackendRegistry) -> Self {
        self.backend_registry = Some(backend_registry);
        self
    }

    pub fn with_backend_name(mut self, backend_name: impl Into<String>) -> Self {
        self.backend_name = Some(backend_name.into());
        self
    }

    pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    pub fn with_backend_params(mut self, backend_params: BackendParams) -> Self {
        self.backend_params = backend_params;
        self
    }

    pub fn build(self) -> Result<InferenceEngine> {
        let backend_registry = self.backend_registry.unwrap_or_else(|| {
            let mut registry = BackendRegistry::new();
            registry.register("mock".to_string(), Box::new(MockBackend::new()));
            registry
        });

        let mut engine = InferenceEngine {
            registry: self
                .registry
                .unwrap_or_else(|| Box::new(DefaultCoreRegistry::default())),
            backend_registry,
            active_backend: None,
            model: None,
            model_path: None,
        };

        if let (Some(backend_name), Some(model_path)) = (self.backend_name, self.model_path) {
            engine.load_model(&backend_name, model_path, self.backend_params)?;
        }

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_can_preload_model_and_generate() {
        let mut engine = InferenceEngine::builder()
            .with_backend_name("mock")
            .with_model_path("demo.gguf")
            .build()
            .expect("build");

        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("mock:hello"));
        assert_eq!(engine.active_backend(), Some("mock"));
    }
}
