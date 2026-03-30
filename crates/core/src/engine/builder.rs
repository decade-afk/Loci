use crate::backend::{BackendParams, BackendRegistry, GpuSplitMode, InferenceParams};
use crate::backends;
use crate::core::{CoreRegistry, DefaultCoreRegistry};
use crate::engine::runtime::InferenceEngine;
use crate::error::Result;
use crate::model::{ModelConfig, ModelLoadStrategy};
use std::path::PathBuf;

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
            registry: None,
            backend_registry: None,
            backend_name: Some(backends::default_backend_name().to_string()),
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
            split_mode: GpuSplitMode::Layer,
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

    pub fn with_backend_params(mut self, backend_params: BackendParams) -> Self {
        self.backend_params = backend_params;
        self
    }

    pub fn model_config(mut self, config: ModelConfig) -> Self {
        self.model_config = Some(config);
        self
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
        self.split_mode = GpuSplitMode::None;
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

    pub fn with_gpu_split_mode(mut self, split_mode: GpuSplitMode) -> Self {
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
            BackendRegistry::with_builtin_backends()
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
            let backend_name = self
                .backend_name
                .as_deref()
                .unwrap_or(backends::default_backend_name())
                .to_string();
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
    use super::InferenceEngineBuilder;
    use crate::engine::GenerationParams;
    use crate::model::ModelConfig;

    #[test]
    fn builder_can_preload_model_and_generate() {
        let mut engine = InferenceEngineBuilder::new()
            .with_backend_name("mock")
            .with_model_path("demo.gguf")
            .build()
            .expect("build");

        let output = engine
            .generate("hello", &crate::backend::InferenceParams::default())
            .expect("generate");
        assert!(output.contains("mock:hello"));
        assert_eq!(engine.active_backend(), Some("mock"));
    }

    #[test]
    fn builder_legacy_generate_uses_default_inference_shape() {
        let mut engine = InferenceEngineBuilder::new()
            .backend("mock")
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

    #[test]
    fn builder_uses_builtin_registry_by_default() {
        let engine = InferenceEngineBuilder::new().build().expect("build");
        assert_eq!(engine.active_backend(), None);
        assert_eq!(engine.plugin_count(), 0);
    }
}
