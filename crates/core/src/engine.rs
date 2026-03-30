use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model};
use crate::backends::MockBackend;
use crate::core::{CoreRegistry, DefaultCoreRegistry};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::RegisteredPlugin;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

impl From<GenerationParams> for InferenceParams {
    fn from(params: GenerationParams) -> Self {
        Self {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
}

pub struct InferenceEngine {
    registry: Box<dyn CoreRegistry>,
    backend_registry: BackendRegistry,
    active_backend: Option<String>,
    model: Option<Box<dyn Model>>,
    model_path: Option<PathBuf>,
    default_inference_params: InferenceParams,
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

    pub fn load_model_config(&mut self, backend_name: &str, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        let backend_params = BackendParams {
            n_gpu_layers: config.n_gpu_layers,
            use_gpu: config.use_gpu,
            use_mmap: config.use_mmap,
            use_mlock: config.use_mlock,
            kv_offload: config.kv_offload,
            op_offload: config.op_offload,
            split_mode: config.split_mode,
            main_gpu: config.main_gpu,
            tensor_split: config.tensor_split.clone(),
            options: vec![
                ("n_ctx".to_string(), config.n_ctx.to_string()),
                ("n_batch".to_string(), config.n_batch.to_string()),
            ],
        };

        let result = self.load_model(backend_name, &config.model_path, backend_params.clone());
        match (result, config.load_strategy) {
            (Ok(()), _) => Ok(()),
            (
                Err(_),
                ModelLoadStrategy::AutoReduceGpuLayers { step },
            ) if config.use_gpu && config.n_gpu_layers > 0 => {
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

    pub fn model_metadata(&self) -> Option<crate::backend::ModelMetadata> {
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

#[derive(Default)]
pub struct InferenceEngineBuilder {
    registry: Option<Box<dyn CoreRegistry>>,
    backend_registry: Option<BackendRegistry>,
    backend_name: Option<String>,
    model_path: Option<PathBuf>,
    backend_params: BackendParams,
    model_config: Option<ModelConfig>,
    n_ctx: u32,
    n_threads: Option<u32>,
    n_batch: u32,
    use_gpu: bool,
    n_gpu_layers: i32,
    use_mmap: bool,
    use_mlock: bool,
    kv_offload: bool,
    op_offload: bool,
    split_mode: crate::backend::GpuSplitMode,
    main_gpu: u32,
    tensor_split: Option<Vec<f32>>,
    load_strategy: ModelLoadStrategy,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            registry: None,
            backend_registry: None,
            backend_name: Some("mock".to_string()),
            model_path: None,
            backend_params: BackendParams::default(),
            model_config: None,
            n_ctx: 4096,
            n_threads: None,
            n_batch: 512,
            use_gpu: true,
            n_gpu_layers: -1,
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: crate::backend::GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            load_strategy: ModelLoadStrategy::Strict,
        }
    }

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

    pub fn model_path(self, model_path: impl Into<PathBuf>) -> Self {
        self.with_model_path(model_path)
    }

    pub fn with_backend_params(mut self, backend_params: BackendParams) -> Self {
        self.backend_params = backend_params;
        self
    }

    pub fn model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    pub fn backend(self, backend_name: impl Into<String>) -> Self {
        self.with_backend_name(backend_name)
    }

    pub fn context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn threads(mut self, n_threads: u32) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    pub fn batch_size(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self.n_gpu_layers = 0;
        self.kv_offload = false;
        self.op_offload = false;
        self.split_mode = crate::backend::GpuSplitMode::None;
        self.main_gpu = 0;
        self.tensor_split = None;
        self
    }

    pub fn gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    pub fn with_kv_offload(mut self, kv_offload: bool) -> Self {
        self.kv_offload = kv_offload;
        self
    }

    pub fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.op_offload = op_offload;
        self
    }

    pub fn with_gpu_split_mode(mut self, split_mode: crate::backend::GpuSplitMode) -> Self {
        self.split_mode = split_mode;
        self
    }

    pub fn with_main_gpu(mut self, main_gpu: u32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    pub fn with_tensor_split(mut self, tensor_split: Vec<f32>) -> Self {
        self.tensor_split = Some(tensor_split);
        self
    }

    pub fn with_load_strategy(mut self, load_strategy: ModelLoadStrategy) -> Self {
        self.load_strategy = load_strategy;
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
            default_inference_params: InferenceParams {
                n_ctx: self.n_ctx,
                n_batch: self.n_batch,
                n_threads: self.n_threads,
                ..Default::default()
            },
        };

        if let Some(model_config) = self.model_config {
            let backend_name = self.backend_name.as_deref().unwrap_or("mock").to_string();
            engine.load_model_config(&backend_name, &model_config)?;
        } else if let (Some(backend_name), Some(model_path)) = (self.backend_name, self.model_path)
        {
            let backend_params = BackendParams {
                n_gpu_layers: self.n_gpu_layers,
                use_gpu: self.use_gpu,
                use_mmap: self.use_mmap,
                use_mlock: self.use_mlock,
                kv_offload: self.kv_offload,
                op_offload: self.op_offload,
                split_mode: self.split_mode,
                main_gpu: self.main_gpu,
                tensor_split: self.tensor_split.clone(),
                options: if self.backend_params.options.is_empty() {
                    vec![
                        ("n_ctx".to_string(), self.n_ctx.to_string()),
                        ("n_batch".to_string(), self.n_batch.to_string()),
                    ]
                } else {
                    self.backend_params.options
                },
            };
            let _ = self.load_strategy;
            engine.load_model(&backend_name, model_path, backend_params)?;
        }

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_can_preload_model_and_generate() {
        let mut engine = InferenceEngineBuilder::new()
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

    #[test]
    fn builder_legacy_generate_uses_default_inference_shape() {
        let mut engine = InferenceEngineBuilder::new()
            .model_path("demo.gguf")
            .context_size(8192)
            .batch_size(1024)
            .build()
            .expect("build");

        let output = engine
            .generate_legacy("hello", GenerationParams::default())
            .expect("generate");
        assert!(output.contains("mock:hello"));
    }

    #[test]
    fn builder_can_load_model_config() {
        let config = ModelConfig::new("demo.gguf")
            .with_context_size(2048)
            .with_batch_size(128)
            .cpu_only();
        let engine = InferenceEngineBuilder::new()
            .backend("mock")
            .model_config(config)
            .build()
            .expect("build");

        assert_eq!(engine.active_backend(), Some("mock"));
    }
}
