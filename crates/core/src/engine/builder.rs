use crate::backend::{BackendRegistry, GpuSplitMode, InferenceParams};
use crate::backends;
use crate::engine::runtime::InferenceEngine;
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::PluginSamplingRuntime;
use crate::Result;
use std::path::PathBuf;

pub struct InferenceEngineBuilder {
    backend_registry: Option<BackendRegistry>,
    backend_name: Option<String>,
    model_path: Option<PathBuf>,
    model_config: Option<ModelConfig>,
    defaults: InferenceParams,
    use_gpu: bool,
    n_gpu_layers: i32,
    use_mmap: bool,
    use_mlock: bool,
    kv_offload: bool,
    op_offload: bool,
    split_mode: GpuSplitMode,
    main_gpu: u32,
    tensor_split: Option<Vec<f32>>,
    load_strategy: ModelLoadStrategy,
}

impl Default for InferenceEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            backend_registry: None,
            backend_name: Some(backends::default_backend_name().to_string()),
            model_path: None,
            model_config: None,
            defaults: InferenceParams::default(),
            use_gpu: true,
            n_gpu_layers: -1,
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            load_strategy: ModelLoadStrategy::Strict,
        }
    }

    pub fn with_backend_registry(mut self, backend_registry: BackendRegistry) -> Self {
        self.backend_registry = Some(backend_registry);
        self
    }

    pub fn with_backend_name(mut self, backend_name: impl Into<String>) -> Self {
        self.backend_name = Some(backend_name.into());
        self
    }

    pub fn backend(self, backend_name: impl Into<String>) -> Self {
        self.with_backend_name(backend_name)
    }

    pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    pub fn model_path(self, model_path: impl Into<PathBuf>) -> Self {
        self.with_model_path(model_path)
    }

    pub fn model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    pub fn context_size(mut self, n_ctx: u32) -> Self {
        self.defaults.n_ctx = n_ctx;
        self
    }

    pub fn batch_size(mut self, n_batch: u32) -> Self {
        self.defaults.n_batch = n_batch;
        self
    }

    pub fn threads(mut self, n_threads: u32) -> Self {
        self.defaults.n_threads = Some(n_threads);
        self
    }

    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self.n_gpu_layers = 0;
        self.kv_offload = false;
        self.op_offload = false;
        self.split_mode = GpuSplitMode::None;
        self.main_gpu = 0;
        self.tensor_split = None;
        self
    }

    pub fn with_load_strategy(mut self, load_strategy: ModelLoadStrategy) -> Self {
        self.load_strategy = load_strategy;
        self
    }

    pub fn build(self) -> Result<InferenceEngine> {
        let mut engine = InferenceEngine {
            backend_registry: self
                .backend_registry
                .unwrap_or_else(BackendRegistry::with_builtin_backends),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: self.defaults,
            plugin_manifests: Vec::new(),
            active_plugins: Default::default(),
            sampling_runtime: PluginSamplingRuntime::default(),
        };

        if let Some(config) = self.model_config {
            let backend_name = self
                .backend_name
                .clone()
                .unwrap_or_else(|| backends::default_backend_name().to_string());
            engine.load_model_config(&backend_name, &config)?;
        } else if let (Some(backend_name), Some(model_path)) = (self.backend_name, self.model_path)
        {
            let config = ModelConfig {
                model_path,
                n_ctx: engine.default_inference_params.n_ctx,
                n_threads: engine.default_inference_params.n_threads,
                n_batch: engine.default_inference_params.n_batch,
                use_gpu: self.use_gpu,
                n_gpu_layers: self.n_gpu_layers,
                use_mmap: self.use_mmap,
                use_mlock: self.use_mlock,
                kv_offload: self.kv_offload,
                op_offload: self.op_offload,
                split_mode: self.split_mode,
                main_gpu: self.main_gpu,
                tensor_split: self.tensor_split,
                load_strategy: self.load_strategy,
            };
            engine.load_model_config(&backend_name, &config)?;
        }

        Ok(engine)
    }
}
