use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelMetadata,
};
use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

const DEFAULT_N_CTX: u32 = 4096;
const DEFAULT_N_BATCH: u32 = 512;

pub struct LlamaCppBackend {
    initialized: bool,
}

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LlamaCppModel {
    load_plan: LlamaCppLoadPlan,
    runtime_state: LlamaCppRuntimeState,
}

#[derive(Debug, Clone)]
struct LlamaCppRuntimeOptions {
    n_ctx: u32,
    n_batch: u32,
    n_threads: Option<u32>,
}

#[derive(Debug, Clone)]
struct LlamaCppExecutionConfig {
    n_ctx: u32,
    n_batch: u32,
    n_threads: Option<u32>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    top_k: u32,
    repeat_penalty: f32,
}

#[derive(Debug, Clone)]
struct LlamaCppRuntimeState {
    current_n_ctx: u32,
    current_n_batch: u32,
    current_n_threads: Option<u32>,
    kv_offload: bool,
    op_offload: bool,
}

#[derive(Debug, Clone)]
struct LlamaCppLoadPlan {
    model_path: PathBuf,
    runtime: LlamaCppRuntimeOptions,
    gpu_active: bool,
    n_gpu_layers: i32,
    use_mmap: bool,
    use_mlock: bool,
    kv_offload: bool,
    op_offload: bool,
    main_gpu: u32,
    tensor_split: Option<Vec<f32>>,
}

impl LlamaCppLoadPlan {
    fn from_backend_params(model_path: &Path, params: BackendParams) -> Result<Self> {
        validate_model_path(model_path)?;

        let runtime = LlamaCppRuntimeOptions::from_backend_options(&params.options)?;
        let gpu_active = params.use_gpu && params.n_gpu_layers != 0;
        let n_gpu_layers = if gpu_active { params.n_gpu_layers } else { 0 };
        let kv_offload = gpu_active && params.kv_offload;
        let op_offload = gpu_active && params.op_offload;
        let main_gpu = if gpu_active { params.main_gpu } else { 0 };
        let tensor_split = if gpu_active { params.tensor_split } else { None };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            runtime,
            gpu_active,
            n_gpu_layers,
            use_mmap: params.use_mmap,
            use_mlock: params.use_mlock,
            kv_offload,
            op_offload,
            main_gpu,
            tensor_split,
        })
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: 0,
            n_ctx_train: self.runtime.n_ctx,
            n_embd: 0,
            n_layer: 0,
            param_count: None,
        }
    }

    fn create_runtime_state(&self) -> LlamaCppRuntimeState {
        LlamaCppRuntimeState {
            current_n_ctx: self.runtime.n_ctx,
            current_n_batch: self.runtime.n_batch,
            current_n_threads: self.runtime.n_threads,
            kv_offload: self.kv_offload,
            op_offload: self.op_offload,
        }
    }
}

impl LlamaCppRuntimeOptions {
    fn from_backend_options(options: &[(String, String)]) -> Result<Self> {
        let n_ctx = find_option_u32(options, "n_ctx")?.unwrap_or(DEFAULT_N_CTX);
        let n_batch = find_option_u32(options, "n_batch")?.unwrap_or(DEFAULT_N_BATCH);
        let n_threads = find_option_u32(options, "n_threads")?;

        if n_ctx == 0 {
            return Err(LociError::ConfigError(
                "llama.cpp requires n_ctx > 0".to_string(),
            ));
        }
        if n_batch == 0 {
            return Err(LociError::ConfigError(
                "llama.cpp requires n_batch > 0".to_string(),
            ));
        }

        Ok(Self {
            n_ctx,
            n_batch,
            n_threads,
        })
    }

    fn supports(&self, params: &InferenceParams) -> bool {
        self.n_ctx == params.n_ctx
            && self.n_batch == params.n_batch
            && self.n_threads == params.n_threads
    }
}

impl LlamaCppExecutionConfig {
    fn from_inference_params(params: &InferenceParams) -> Result<Self> {
        if params.n_ctx == 0 {
            return Err(LociError::ConfigError(
                "llama.cpp execution requires n_ctx > 0".to_string(),
            ));
        }
        if params.n_batch == 0 {
            return Err(LociError::ConfigError(
                "llama.cpp execution requires n_batch > 0".to_string(),
            ));
        }
        if params.max_tokens == 0 {
            return Err(LociError::ConfigError(
                "llama.cpp execution requires max_tokens > 0".to_string(),
            ));
        }

        Ok(Self {
            n_ctx: params.n_ctx,
            n_batch: params.n_batch,
            n_threads: params.n_threads,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
        })
    }
}

impl LlamaCppRuntimeState {
    fn reconcile(&mut self, config: &LlamaCppExecutionConfig) {
        self.current_n_ctx = config.n_ctx;
        self.current_n_batch = config.n_batch;
        self.current_n_threads = config.n_threads;
    }

    fn summary(&self) -> String {
        format!(
            "runtime[n_ctx={}, n_batch={}, n_threads={}, kv_offload={}, op_offload={}]",
            self.current_n_ctx,
            self.current_n_batch,
            self.current_n_threads
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.kv_offload,
            self.op_offload
        )
    }
}

fn validate_model_path(model_path: &Path) -> Result<()> {
    let ext = model_path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if ext != "gguf" {
        return Err(LociError::ConfigError(format!(
            "llama.cpp backend requires a .gguf model file, got: {}",
            model_path.display()
        )));
    }

    Ok(())
}

fn find_option_u32(options: &[(String, String)], key: &str) -> Result<Option<u32>> {
    options
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| {
            value.parse::<u32>().map_err(|_| {
                LociError::ConfigError(format!(
                    "invalid llama.cpp backend option `{key}`: `{value}`"
                ))
            })
        })
        .transpose()
}

impl Model for LlamaCppModel {
    fn metadata(&self) -> ModelMetadata {
        self.load_plan.metadata()
    }

    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "prompt must not be empty".to_string(),
            ));
        }

        let execution = LlamaCppExecutionConfig::from_inference_params(params)?;
        if !self.load_plan.runtime.supports(params) {
            self.runtime_state.reconcile(&execution);
        }

        Ok(format!(
            "llama.cpp-stub:{prompt} [model={}, gpu_active={}, gpu_layers={}, plan_n_ctx={}, plan_n_batch={}, mmap={}, mlock={}, main_gpu={}, tensor_split={}, {}, exec[max_tokens={}, temperature={}, top_p={}, min_p={}, top_k={}, repeat_penalty={}]]",
            self.load_plan.model_path.display(),
            self.load_plan.gpu_active,
            self.load_plan.n_gpu_layers,
            self.load_plan.runtime.n_ctx,
            self.load_plan.runtime.n_batch,
            self.load_plan.use_mmap,
            self.load_plan.use_mlock,
            self.load_plan.main_gpu,
            self.load_plan
                .tensor_split
                .as_ref()
                .map(|values| values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "none".to_string()),
            self.runtime_state.summary(),
            execution.max_tokens,
            execution.temperature,
            execution.top_p,
            execution.min_p,
            execution.top_k,
            execution.repeat_penalty
        ))
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "llama.cpp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
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
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        if !self.initialized {
            return Err(LociError::BackendError(
                "llama.cpp backend not initialized".to_string(),
            ));
        }

        let load_plan = LlamaCppLoadPlan::from_backend_params(model_path, backend_params)?;
        let runtime_state = load_plan.create_runtime_state();
        Ok(Box::new(LlamaCppModel {
            load_plan,
            runtime_state,
        }))
    }

    fn init(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_plan_requires_gguf_extension() {
        let err = LlamaCppLoadPlan::from_backend_params(Path::new("demo.bin"), BackendParams::default())
            .expect_err("should fail");
        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn load_plan_disables_gpu_specific_flags_in_cpu_mode() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                use_gpu: false,
                n_gpu_layers: 32,
                kv_offload: true,
                op_offload: true,
                main_gpu: 2,
                tensor_split: Some(vec![2.0, 1.0]),
                ..Default::default()
            },
        )
        .expect("plan");

        assert!(!plan.gpu_active);
        assert_eq!(plan.n_gpu_layers, 0);
        assert!(!plan.kv_offload);
        assert!(!plan.op_offload);
        assert_eq!(plan.main_gpu, 0);
        assert!(plan.tensor_split.is_none());
    }

    #[test]
    fn load_plan_reads_runtime_options() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![
                    ("n_ctx".to_string(), "8192".to_string()),
                    ("n_batch".to_string(), "1024".to_string()),
                    ("n_threads".to_string(), "12".to_string()),
                ],
                ..Default::default()
            },
        )
        .expect("plan");

        assert_eq!(plan.runtime.n_ctx, 8192);
        assert_eq!(plan.runtime.n_batch, 1024);
        assert_eq!(plan.runtime.n_threads, Some(12));
    }

    #[test]
    fn load_plan_rejects_invalid_runtime_option() {
        let err = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![("n_ctx".to_string(), "bad".to_string())],
                ..Default::default()
            },
        )
        .expect_err("should fail");

        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn runtime_state_is_seeded_from_load_plan() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![
                    ("n_ctx".to_string(), "4096".to_string()),
                    ("n_batch".to_string(), "256".to_string()),
                    ("n_threads".to_string(), "8".to_string()),
                ],
                ..Default::default()
            },
        )
        .expect("plan");

        let runtime_state = plan.create_runtime_state();
        assert_eq!(runtime_state.current_n_ctx, 4096);
        assert_eq!(runtime_state.current_n_batch, 256);
        assert_eq!(runtime_state.current_n_threads, Some(8));
        assert!(runtime_state.kv_offload);
    }

    #[test]
    fn execution_config_rejects_zero_max_tokens() {
        let err = LlamaCppExecutionConfig::from_inference_params(&InferenceParams {
            max_tokens: 0,
            ..Default::default()
        })
        .expect_err("should fail");

        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn runtime_options_can_compare_execution_shape() {
        let options = LlamaCppRuntimeOptions {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: Some(4),
        };

        assert!(options.supports(&InferenceParams {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: Some(4),
            ..Default::default()
        }));
        assert!(!options.supports(&InferenceParams {
            n_ctx: 8192,
            ..Default::default()
        }));
    }
}
