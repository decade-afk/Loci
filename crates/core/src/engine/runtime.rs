use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model, ModelMetadata};
use crate::core::CoreRegistry;
use crate::engine::types::{GenerationParams, ModelInfo};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::RegisteredPlugin;
use std::path::{Path, PathBuf};

pub struct InferenceEngine {
    pub(crate) registry: Box<dyn CoreRegistry>,
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
}

impl InferenceEngine {
    pub fn builder() -> crate::engine::InferenceEngineBuilder {
        crate::engine::InferenceEngineBuilder::new()
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

    pub fn load_model_config(&mut self, backend_name: &str, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        let backend_params = config.to_backend_params();

        let result = self.load_model(backend_name, &config.model_path, backend_params.clone());
        match (result, config.load_strategy) {
            (Ok(()), _) => Ok(()),
            (Err(_), ModelLoadStrategy::AutoReduceGpuLayers { step })
                if config.use_gpu && config.n_gpu_layers > 0 =>
            {
                let mut retry = config.n_gpu_layers.saturating_sub(step as i32);
                while retry >= 0 {
                    let mut reduced = backend_params.clone();
                    reduced.n_gpu_layers = retry;
                    if self
                        .load_model(backend_name, &config.model_path, reduced)
                        .is_ok()
                    {
                        return Ok(());
                    }
                    if retry == 0 {
                        break;
                    }
                    retry = retry.saturating_sub(step as i32);
                }
                Err(LociError::ModelLoadError(
                    "model load failed after GPU fallback attempts".to_string(),
                ))
            }
            (Err(err), _) => Err(err),
        }
    }

    pub fn generate(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        model.infer_text(prompt, params)
    }

    pub fn generate_legacy(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = self.generation_params_to_inference(params);
        self.generate(prompt, &inference_params)
    }

    fn generation_params_to_inference(&self, params: GenerationParams) -> InferenceParams {
        InferenceParams {
            n_ctx: self.default_inference_params.n_ctx,
            n_batch: self.default_inference_params.n_batch,
            n_threads: self.default_inference_params.n_threads,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
        }
    }

    pub fn active_backend(&self) -> Option<&str> {
        self.active_backend.as_deref()
    }

    pub fn model_metadata(&self) -> Option<ModelMetadata> {
        self.model.as_ref().map(|model| model.metadata())
    }

    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model_metadata().map(|metadata| ModelInfo {
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
        })
    }
}
