use crate::backend::{BackendParams, BackendRegistry, GpuSplitMode, InferenceParams};
use crate::backends;
use crate::core::{CoreRegistry, DefaultCoreRegistry};
use crate::engine::runtime::InferenceEngine;
use crate::error::Result;
use crate::model::{ModelConfig, ModelLoadStrategy};
use std::collections::BTreeMap;
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
        let backend_registry = self
            .backend_registry
            .unwrap_or_else(|| BackendRegistry::with_builtin_backends());

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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: crate::engine::runtime::LegacyTextRuntimeRegistry::default(),
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
            let backend_params = if self.backend_params.options.is_empty() {
                ModelConfig {
                    model_path: model_path.clone(),
                    n_ctx: self.n_ctx,
                    n_threads: self.n_threads,
                    n_batch: self.n_batch,
                    use_gpu: self.use_gpu,
                    n_gpu_layers: self.n_gpu_layers,
                    use_mmap: self.use_mmap,
                    use_mlock: self.use_mlock,
                    kv_offload: self.kv_offload,
                    op_offload: self.op_offload,
                    split_mode: self.split_mode,
                    main_gpu: self.main_gpu,
                    tensor_split: self.tensor_split.clone(),
                    load_strategy: self.load_strategy,
                }
                .to_backend_params()
            } else {
                self.backend_params
            };
            engine.load_model(&backend_name, model_path, backend_params)?;
        }

        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::InferenceEngineBuilder;
    use crate::backend::{
        BackendCapabilities, BackendParams, BackendRegistry, InferenceBackend, InferenceParams,
        Model, ModelMetadata,
    };
    use crate::engine::GenerationParams;
    use crate::error::{LociError, Result as LociResult};
    use crate::model::ModelConfig;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    fn temp_model_path(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-builder-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("demo.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        path
    }

    #[derive(Debug, Clone)]
    struct RecordedLoad {
        params: BackendParams,
    }

    struct RecordingBackend {
        calls: Arc<Mutex<Vec<RecordedLoad>>>,
    }

    impl InferenceBackend for RecordingBackend {
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                name: "recording".to_string(),
                version: "1.0.0".to_string(),
                supports_text: true,
                supports_multimodal: false,
                supports_embeddings: false,
                supports_streaming: false,
                has_gpu_support: true,
                supported_formats: vec!["gguf".to_string()],
            }
        }

        fn load_model(
            &self,
            _model_path: &std::path::Path,
            backend_params: BackendParams,
        ) -> LociResult<Box<dyn Model>> {
            self.calls.lock().expect("calls lock").push(RecordedLoad {
                params: backend_params,
            });
            Ok(Box::new(RecordedModel))
        }
    }

    struct RecordedModel;

    impl Model for RecordedModel {
        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                architecture: "recording".to_string(),
                n_vocab: 0,
                n_ctx_train: 4096,
                n_embd: 0,
                n_layer: 0,
                param_count: None,
            }
        }

        fn infer_text(&mut self, prompt: &str, _params: &InferenceParams) -> LociResult<String> {
            if prompt.trim().is_empty() {
                return Err(LociError::InvalidArgument(
                    "prompt must not be empty".to_string(),
                ));
            }
            Ok(format!("recording:{prompt}"))
        }
    }

    fn recording_backend_registry(calls: Arc<Mutex<Vec<RecordedLoad>>>) -> BackendRegistry {
        let mut registry = BackendRegistry::new();
        registry.register(
            "recording".to_string(),
            Box::new(RecordingBackend { calls }),
        );
        registry
    }

    #[test]
    fn builder_can_preload_model_and_generate() {
        let model_path = temp_model_path("preload");
        let mut engine = InferenceEngineBuilder::new()
            .with_backend_name("mock")
            .with_model_path(&model_path)
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
        let model_path = temp_model_path("legacy");
        let mut engine = InferenceEngineBuilder::new()
            .backend("mock")
            .model_path(&model_path)
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
        let model_path = temp_model_path("config");
        let config = ModelConfig::new(&model_path)
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
    fn builder_model_path_preload_uses_governed_model_load_path() {
        let model_path = temp_model_path("governed");
        let calls = Arc::new(Mutex::new(Vec::<RecordedLoad>::new()));
        let engine = InferenceEngineBuilder::new()
            .backend("recording")
            .with_backend_registry(recording_backend_registry(Arc::clone(&calls)))
            .model_path(&model_path)
            .build()
            .expect("build");

        assert_eq!(engine.active_backend(), Some("recording"));
        let loads = calls.lock().expect("calls lock");
        assert_eq!(loads.len(), 1);
        assert!(!loads[0].params.use_gpu);
        assert_eq!(loads[0].params.n_gpu_layers, 0);
    }

    #[test]
    fn builder_uses_builtin_registry_by_default() {
        let engine = InferenceEngineBuilder::new().build().expect("build");
        assert_eq!(engine.active_backend(), None);
        assert_eq!(engine.plugin_count(), 0);
    }
}
