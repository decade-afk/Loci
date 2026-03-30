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
}

#[derive(Debug, Clone)]
struct LlamaCppRuntimeOptions {
    n_ctx: u32,
    n_batch: u32,
    n_threads: Option<u32>,
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

        Ok(format!(
            "llama.cpp-stub:{prompt} [model={}, gpu_active={}, gpu_layers={}, n_ctx={}, n_batch={}, n_threads={}, mmap={}, mlock={}, kv_offload={}, op_offload={}, main_gpu={}, tensor_split={}, max_tokens={}]",
            self.load_plan.model_path.display(),
            self.load_plan.gpu_active,
            self.load_plan.n_gpu_layers,
            self.load_plan.runtime.n_ctx,
            self.load_plan.runtime.n_batch,
            self.load_plan
                .runtime
                .n_threads
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.load_plan.use_mmap,
            self.load_plan.use_mlock,
            self.load_plan.kv_offload,
            self.load_plan.op_offload,
            self.load_plan.main_gpu,
            self.load_plan
                .tensor_split
                .as_ref()
                .map(|values| values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "none".to_string()),
            params.max_tokens
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
        Ok(Box::new(LlamaCppModel { load_plan }))
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
}
