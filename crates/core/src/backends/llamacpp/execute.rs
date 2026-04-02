use crate::backend::InferenceParams;
use crate::error::{LociError, Result};
use crate::sampler::{sample_token, LogitsView, SamplingParams};

use super::ffi;
use super::runtime::LlamaCppExecutionConfig;
use super::LlamaCppModel;
use std::sync::OnceLock;

const DEFAULT_MAX_PROMPT_BYTES: usize = 24 * 1024;
const TOKENIZE_CHUNK_BYTES: usize = 4096;
const GENERATION_HEADROOM_MIN_TOKENS: usize = 8;
const GENERATION_HEADROOM_MAX_TOKENS: usize = 64;

impl LlamaCppModel {
    pub(super) fn infer_text_native(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<String> {
        let execution = LlamaCppExecutionConfig::from_inference_params(params)?;
        self.ensure_context_shape(&execution)?;

        let max_prompt_bytes = Self::max_prompt_bytes();
        if prompt.len() > max_prompt_bytes {
            return Err(LociError::InferenceError(format!(
                "prompt is too large ({} bytes > {} bytes limit)",
                prompt.len(),
                max_prompt_bytes
            )));
        }

        let mut tokens = {
            let model = self.native_model.require_native_model()?;
            Self::tokenize_stable(model, prompt, false, false)?
        };
        if tokens.is_empty() {
            return Err(LociError::InferenceError(
                "prompt produced no tokens".to_string(),
            ));
        }

        let reserve = Self::generation_headroom(execution.max_tokens);
        Self::enforce_context_window(&mut tokens, execution.n_ctx(), reserve)?;

        {
            let context = self.native_context.require_native_context_mut()?;
            context.kv_cache_clear();
            Self::decode_tokens_chunked(context, &tokens, 0, Self::max_decode_chunk(&execution))?;
        }

        let mut result = String::new();
        let mut current_position = i32::try_from(tokens.len())
            .map_err(|_| LociError::InferenceError("token count exceeds i32 range".to_string()))?;
        let mut remaining_decode_slots = execution.n_ctx() as usize - tokens.len();
        let max_steps = usize::try_from(execution.max_tokens).unwrap_or(usize::MAX);
        let sampling_params = SamplingParams {
            temperature: execution.temperature,
            top_k: execution.top_k,
            top_p: execution.top_p,
            min_p: execution.min_p,
            repeat_penalty: execution.repeat_penalty,
            seed: 0,
        };
        let mut recent_tokens = tokens.iter().copied().rev().take(64).collect::<Vec<_>>();
        recent_tokens.reverse();

        for _ in 0..max_steps {
            let token = {
                let context = self.native_context.require_native_context()?;
                let model = self.native_model.require_native_model()?;
                let logits = context.get_logits_ith(-1);
                if logits.is_null() {
                    return Err(LociError::InferenceError(
                        "failed to read logits from llama.cpp context".to_string(),
                    ));
                }
                let mut logits_view =
                    unsafe { LogitsView::from_raw(logits, model.n_vocab() as usize) };
                self.sampling_runtime
                    .apply_transform_logits(&mut logits_view, &recent_tokens)?;
                let sampled = sample_token(&logits_view, &sampling_params, &recent_tokens);
                self.sampling_runtime.apply_post_sample(sampled)?
            };

            let piece = {
                let model = self.native_model.require_native_model()?;
                if model.is_eog(token) {
                    break;
                }

                model
                    .token_to_str(token)
                    .map_err(LociError::InferenceError)?
            };

            result.push_str(&piece);
            recent_tokens.push(token);
            if recent_tokens.len() > 64 {
                recent_tokens.remove(0);
            }

            if remaining_decode_slots == 0 {
                break;
            }

            let context = self.native_context.require_native_context_mut()?;
            Self::decode_tokens(
                context,
                &[token],
                current_position,
                Self::max_decode_chunk(&execution),
            )?;
            current_position = current_position
                .checked_add(1)
                .ok_or_else(|| LociError::InferenceError("token position overflow".to_string()))?;
            remaining_decode_slots -= 1;
        }

        Ok(result)
    }

    fn max_prompt_bytes() -> usize {
        static MAX_PROMPT_BYTES: OnceLock<usize> = OnceLock::new();
        *MAX_PROMPT_BYTES.get_or_init(|| {
            std::env::var("LOCI_MAX_PROMPT_BYTES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value >= 1024)
                .unwrap_or(DEFAULT_MAX_PROMPT_BYTES)
        })
    }

    fn generation_headroom(max_tokens: u32) -> usize {
        let requested = usize::try_from(max_tokens).unwrap_or(GENERATION_HEADROOM_MAX_TOKENS);
        requested.clamp(
            GENERATION_HEADROOM_MIN_TOKENS,
            GENERATION_HEADROOM_MAX_TOKENS,
        )
    }

    fn max_decode_chunk(execution: &LlamaCppExecutionConfig) -> usize {
        execution.n_batch().max(1) as usize
    }

    fn enforce_context_window(
        tokens: &mut Vec<i32>,
        n_ctx: u32,
        reserve_tokens: usize,
    ) -> Result<()> {
        let n_ctx_usize = n_ctx as usize;
        if n_ctx_usize <= reserve_tokens {
            return Err(LociError::InferenceError(format!(
                "n_ctx is too small for inference (requires > {})",
                reserve_tokens
            )));
        }

        let limit = n_ctx_usize - reserve_tokens;
        if tokens.len() > limit {
            let drop_count = tokens.len() - limit;
            tokens.drain(0..drop_count);
        }

        Ok(())
    }

    fn split_utf8_chunks(text: &str, chunk_bytes: usize) -> Vec<&str> {
        if text.is_empty() {
            return vec![];
        }
        if chunk_bytes == 0 || text.len() <= chunk_bytes {
            return vec![text];
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < text.len() {
            let mut end = (start + chunk_bytes).min(text.len());
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }

            if end == start {
                break;
            }

            chunks.push(&text[start..end]);
            start = end;
        }

        if chunks.is_empty() {
            vec![text]
        } else {
            chunks
        }
    }

    fn tokenize_stable(
        model: &ffi::LlamaModel,
        text: &str,
        add_bos: bool,
        special: bool,
    ) -> Result<Vec<i32>> {
        if text.len() <= TOKENIZE_CHUNK_BYTES {
            return model
                .tokenize(text, add_bos, special)
                .map_err(LociError::InferenceError);
        }

        let chunks = Self::split_utf8_chunks(text, TOKENIZE_CHUNK_BYTES);
        let mut all_tokens = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let mut tokens = model
                .tokenize(chunk, add_bos && index == 0, special)
                .map_err(LociError::InferenceError)?;
            all_tokens.append(&mut tokens);
        }

        Ok(all_tokens)
    }

    fn decode_tokens(
        context: &mut ffi::LlamaContext,
        tokens: &[i32],
        pos_start: i32,
        max_chunk_size: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(LociError::InferenceError(
                "cannot decode an empty token batch".to_string(),
            ));
        }

        if tokens.len() > max_chunk_size {
            return Err(LociError::InferenceError(format!(
                "decode batch too large: {} > n_batch({})",
                tokens.len(),
                max_chunk_size
            )));
        }

        let n_tokens_i32 = i32::try_from(tokens.len()).map_err(|_| {
            LociError::InferenceError("token batch length exceeds i32 range".to_string())
        })?;
        let _end_pos = pos_start
            .checked_add(n_tokens_i32 - 1)
            .ok_or_else(|| LociError::InferenceError("token position overflow".to_string()))?;

        let mut batch =
            ffi::OwnedBatch::new(n_tokens_i32, 0, 1).map_err(LociError::InferenceError)?;
        let batch_ref = batch.as_mut();

        unsafe {
            for (index, &token) in tokens.iter().enumerate() {
                let index_i32 = i32::try_from(index).map_err(|_| {
                    LociError::InferenceError("token position overflow".to_string())
                })?;
                *batch_ref.token.add(index) = token;
                *batch_ref.pos.add(index) = pos_start + index_i32;
                *batch_ref.n_seq_id.add(index) = 1;

                let seq_id_ptr = *batch_ref.seq_id.add(index);
                if seq_id_ptr.is_null() {
                    return Err(LociError::InferenceError(
                        "llama batch seq_id buffer is null".to_string(),
                    ));
                }

                *seq_id_ptr = 0;
                *batch_ref.logits.add(index) = if index + 1 == tokens.len() { 1 } else { 0 };
            }

            batch_ref.n_tokens = n_tokens_i32;
        }

        context.decode(batch_ref).map_err(LociError::InferenceError)
    }

    fn decode_tokens_chunked(
        context: &mut ffi::LlamaContext,
        tokens: &[i32],
        pos_start: i32,
        max_chunk_size: usize,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Err(LociError::InferenceError(
                "cannot decode an empty token batch".to_string(),
            ));
        }

        let mut processed = 0usize;
        while processed < tokens.len() {
            let end = (processed + max_chunk_size).min(tokens.len());
            let chunk = &tokens[processed..end];
            let processed_i32 = i32::try_from(processed)
                .map_err(|_| LociError::InferenceError("token position overflow".to_string()))?;
            let chunk_pos = pos_start
                .checked_add(processed_i32)
                .ok_or_else(|| LociError::InferenceError("token position overflow".to_string()))?;
            Self::decode_tokens(context, chunk, chunk_pos, max_chunk_size)?;
            processed = end;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaCppModel, TOKENIZE_CHUNK_BYTES};

    #[test]
    fn split_utf8_chunks_preserves_data_for_multibyte_text() {
        let text = "深度审查与修复内存安全。".repeat(900);
        let chunks = LlamaCppModel::split_utf8_chunks(&text, TOKENIZE_CHUNK_BYTES);
        assert!(!chunks.is_empty());
        assert!(chunks
            .iter()
            .all(|chunk| chunk.len() <= TOKENIZE_CHUNK_BYTES));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn split_utf8_chunks_handles_empty_input() {
        let chunks = LlamaCppModel::split_utf8_chunks("", TOKENIZE_CHUNK_BYTES);
        assert!(chunks.is_empty());
    }
}
