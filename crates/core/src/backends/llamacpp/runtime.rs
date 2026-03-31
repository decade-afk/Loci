use crate::backend::InferenceParams;
use crate::error::{LociError, Result};

#[derive(Debug, Clone)]
pub struct LlamaCppExecutionConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    n_ctx: u32,
    n_batch: u32,
    n_threads: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppRuntimeState {
    current_n_ctx: u32,
    current_n_batch: u32,
    current_n_threads: Option<u32>,
    kv_offload: bool,
    op_offload: bool,
}

impl LlamaCppExecutionConfig {
    pub fn from_inference_params(params: &InferenceParams) -> Result<Self> {
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

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    pub fn n_batch(&self) -> u32 {
        self.n_batch
    }

    pub fn n_threads(&self) -> Option<u32> {
        self.n_threads
    }
}

impl LlamaCppRuntimeState {
    pub fn new(
        current_n_ctx: u32,
        current_n_batch: u32,
        current_n_threads: Option<u32>,
        kv_offload: bool,
        op_offload: bool,
    ) -> Self {
        Self {
            current_n_ctx,
            current_n_batch,
            current_n_threads,
            kv_offload,
            op_offload,
        }
    }

    pub fn reconcile(&mut self, config: &LlamaCppExecutionConfig) {
        self.current_n_ctx = config.n_ctx();
        self.current_n_batch = config.n_batch();
        self.current_n_threads = config.n_threads();
    }

    pub fn summary(&self) -> String {
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

    #[cfg(test)]
    pub fn current_n_ctx(&self) -> u32 {
        self.current_n_ctx
    }

    #[cfg(test)]
    pub fn current_n_batch(&self) -> u32 {
        self.current_n_batch
    }

    #[cfg(test)]
    pub fn current_n_threads(&self) -> Option<u32> {
        self.current_n_threads
    }

    #[cfg(test)]
    pub fn kv_offload(&self) -> bool {
        self.kv_offload
    }
}
