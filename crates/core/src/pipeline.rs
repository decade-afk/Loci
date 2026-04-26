use crate::backend::InferenceParams;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceResponse {
    pub output: String,
    pub backend: Option<String>,
    pub model_path: Option<String>,
}

pub fn merge_inference_params(
    defaults: &InferenceParams,
    request_params: &InferenceParams,
) -> InferenceParams {
    InferenceParams {
        n_ctx: request_params.n_ctx,
        n_batch: request_params.n_batch,
        n_threads: request_params.n_threads.or(defaults.n_threads),
        max_tokens: request_params.max_tokens,
        temperature: request_params.temperature,
        top_p: request_params.top_p,
        min_p: request_params.min_p,
        top_k: request_params.top_k,
        repeat_penalty: request_params.repeat_penalty,
    }
}
