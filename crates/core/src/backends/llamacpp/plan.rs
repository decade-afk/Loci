use crate::backend::{BackendParams, InferenceParams, ModelMetadata};
use crate::error::{LociError, Result};
use crate::backends::llamacpp::runtime::LlamaCppRuntimeState;
use std::path::{Path, PathBuf};

const DEFAULT_N_CTX: u32 = 4096;
const DEFAULT_N_BATCH: u32 = 512;

#[derive(Debug, Clone)]
pub struct LlamaCppRuntimeOptions {
    n_ctx: u32,
    n_batch: u32,
    n_threads: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppLoadPlan {
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
    pub fn from_backend_params(model_path: &Path, params: BackendParams) -> Result<Self> {
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

    pub fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: 0,
            n_ctx_train: self.runtime.n_ctx,
            n_embd: 0,
            n_layer: 0,
            param_count: None,
        }
    }

    pub fn create_runtime_state(&self) -> LlamaCppRuntimeState {
        LlamaCppRuntimeState::new(
            self.runtime.n_ctx,
            self.runtime.n_batch,
            self.runtime.n_threads,
            self.kv_offload,
            self.op_offload,
        )
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn runtime(&self) -> &LlamaCppRuntimeOptions {
        &self.runtime
    }

    pub fn gpu_active(&self) -> bool {
        self.gpu_active
    }

    pub fn n_gpu_layers(&self) -> i32 {
        self.n_gpu_layers
    }

    pub fn use_mmap(&self) -> bool {
        self.use_mmap
    }

    pub fn use_mlock(&self) -> bool {
        self.use_mlock
    }

    #[cfg(test)]
    pub fn kv_offload(&self) -> bool {
        self.kv_offload
    }

    #[cfg(test)]
    pub fn op_offload(&self) -> bool {
        self.op_offload
    }

    pub fn main_gpu(&self) -> u32 {
        self.main_gpu
    }

    #[cfg(test)]
    pub fn tensor_split(&self) -> Option<&Vec<f32>> {
        self.tensor_split.as_ref()
    }

    pub fn tensor_split_summary(&self) -> String {
        self.tensor_split
            .as_ref()
            .map(|values| values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_else(|| "none".to_string())
    }
}

impl LlamaCppRuntimeOptions {
    #[cfg(test)]
    pub fn new(n_ctx: u32, n_batch: u32, n_threads: Option<u32>) -> Self {
        Self {
            n_ctx,
            n_batch,
            n_threads,
        }
    }

    pub fn from_backend_options(options: &[(String, String)]) -> Result<Self> {
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

    pub fn supports(&self, params: &InferenceParams) -> bool {
        self.n_ctx == params.n_ctx
            && self.n_batch == params.n_batch
            && self.n_threads == params.n_threads
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn n_batch(&self) -> u32 {
        self.n_batch
    }

    #[cfg(test)]
    pub fn n_threads(&self) -> Option<u32> {
        self.n_threads
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
