//! Batch inference support for processing multiple prompts efficiently

use crate::backend::InferenceParams;
use crate::error::{LociError, Result};
use std::sync::{Arc, Mutex};
use std::thread;

/// A batch of prompts to process
#[derive(Debug, Clone)]
pub struct PromptBatch {
    pub prompts: Vec<String>,
    pub params: InferenceParams,
}

impl PromptBatch {
    pub fn new(prompts: Vec<String>, params: InferenceParams) -> Self {
        Self { prompts, params }
    }

    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// Result of batch inference
#[derive(Debug)]
pub struct BatchResult {
    pub responses: Vec<Result<String>>,
    pub total_time_ms: u128,
    pub avg_time_per_prompt_ms: f64,
}

impl BatchResult {
    pub fn successful_count(&self) -> usize {
        self.responses.iter().filter(|r| r.is_ok()).count()
    }

    pub fn failed_count(&self) -> usize {
        self.responses.iter().filter(|r| r.is_err()).count()
    }

    pub fn success_rate(&self) -> f64 {
        self.successful_count() as f64 / self.responses.len() as f64
    }
}

/// Configuration for batch inference
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of concurrent inferences
    pub max_concurrent: usize,
    /// Timeout per prompt in milliseconds
    pub timeout_ms: Option<u64>,
    /// Whether to continue on error
    pub continue_on_error: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1, // Sequential by default
            timeout_ms: None,
            continue_on_error: true,
        }
    }
}

impl BatchConfig {
    pub fn parallel(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            ..Default::default()
        }
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    pub fn stop_on_error(mut self) -> Self {
        self.continue_on_error = false;
        self
    }
}

/// Batch inference processor
pub struct BatchInferenceProcessor<F>
where
    F: Fn(&str, &InferenceParams) -> Result<String> + Send + Sync,
{
    inference_fn: Arc<F>,
    config: BatchConfig,
}

impl<F> BatchInferenceProcessor<F>
where
    F: Fn(&str, &InferenceParams) -> Result<String> + Send + Sync + 'static,
{
    pub fn new(inference_fn: F, config: BatchConfig) -> Self {
        Self {
            inference_fn: Arc::new(inference_fn),
            config,
        }
    }

    /// Process a batch of prompts
    pub fn process_batch(&self, batch: PromptBatch) -> Result<BatchResult> {
        let start_time = std::time::Instant::now();

        if batch.is_empty() {
            return Ok(BatchResult {
                responses: Vec::new(),
                total_time_ms: 0,
                avg_time_per_prompt_ms: 0.0,
            });
        }

        let responses = if self.config.max_concurrent == 1 {
            self.process_sequential(&batch)?
        } else {
            self.process_parallel(&batch)?
        };

        let total_time_ms = start_time.elapsed().as_millis();
        let avg_time_per_prompt_ms = total_time_ms as f64 / batch.len() as f64;

        Ok(BatchResult {
            responses,
            total_time_ms,
            avg_time_per_prompt_ms,
        })
    }

    fn process_sequential(&self, batch: &PromptBatch) -> Result<Vec<Result<String>>> {
        let mut responses = Vec::with_capacity(batch.len());

        for prompt in &batch.prompts {
            let result = (self.inference_fn)(prompt, &batch.params);

            if result.is_err() && !self.config.continue_on_error {
                return Err(result.unwrap_err());
            }

            responses.push(result);
        }

        Ok(responses)
    }

    fn process_parallel(&self, batch: &PromptBatch) -> Result<Vec<Result<String>>> {
        let responses: Arc<Mutex<Vec<Option<Result<String>>>>> =
            Arc::new(Mutex::new((0..batch.len()).map(|_| None).collect()));
        let mut handles = Vec::new();

        // Process in chunks based on max_concurrent
        for (chunk_idx, chunk) in batch.prompts.chunks(self.config.max_concurrent).enumerate() {
            for (idx_in_chunk, prompt) in chunk.iter().enumerate() {
                let global_idx = chunk_idx * self.config.max_concurrent + idx_in_chunk;
                let prompt = prompt.clone();
                let params = batch.params.clone();
                let inference_fn = Arc::clone(&self.inference_fn);
                let responses = Arc::clone(&responses);

                let handle = thread::spawn(move || {
                    let result = inference_fn(&prompt, &params);
                    let mut responses = responses.lock().unwrap();
                    responses[global_idx] = Some(result);
                });

                handles.push(handle);
            }

            // Wait for this chunk to complete before starting next
            for handle in handles.drain(..) {
                handle.join().map_err(|_| {
                    LociError::Other("Thread panicked during batch inference".to_string())
                })?;
            }
        }

        // Extract results
        let responses = Arc::try_unwrap(responses)
            .map_err(|_| LociError::Other("Failed to unwrap responses".to_string()))?
            .into_inner()
            .map_err(|_| LociError::Other("Failed to lock responses".to_string()))?;

        Ok(responses.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Process prompts with progress callback
    pub fn process_with_progress<P>(
        &self,
        batch: PromptBatch,
        mut progress_callback: P,
    ) -> Result<BatchResult>
    where
        P: FnMut(usize, usize),
    {
        let start_time = std::time::Instant::now();
        let mut responses = Vec::with_capacity(batch.len());

        for (idx, prompt) in batch.prompts.iter().enumerate() {
            let result = (self.inference_fn)(prompt, &batch.params);

            if result.is_err() && !self.config.continue_on_error {
                return Err(result.unwrap_err());
            }

            responses.push(result);
            progress_callback(idx + 1, batch.len());
        }

        let total_time_ms = start_time.elapsed().as_millis();
        let avg_time_per_prompt_ms = total_time_ms as f64 / batch.len() as f64;

        Ok(BatchResult {
            responses,
            total_time_ms,
            avg_time_per_prompt_ms,
        })
    }
}

/// Builder for batch inference
pub struct BatchInferenceBuilder {
    config: BatchConfig,
}

impl BatchInferenceBuilder {
    pub fn new() -> Self {
        Self {
            config: BatchConfig::default(),
        }
    }

    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.config.max_concurrent = max;
        self
    }

    pub fn timeout(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = Some(timeout_ms);
        self
    }

    pub fn continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.config.continue_on_error = continue_on_error;
        self
    }

    pub fn build<F>(self, inference_fn: F) -> BatchInferenceProcessor<F>
    where
        F: Fn(&str, &InferenceParams) -> Result<String> + Send + Sync + 'static,
    {
        BatchInferenceProcessor::new(inference_fn, self.config)
    }
}

impl Default for BatchInferenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_inference(prompt: &str, _params: &InferenceParams) -> Result<String> {
        Ok(format!("Response to: {}", prompt))
    }

    #[test]
    fn test_batch_sequential() {
        let config = BatchConfig::default();
        let processor = BatchInferenceProcessor::new(mock_inference, config);

        let batch = PromptBatch::new(
            vec!["Hello".to_string(), "World".to_string()],
            InferenceParams::default(),
        );

        let result = processor.process_batch(batch).unwrap();
        assert_eq!(result.responses.len(), 2);
        assert_eq!(result.successful_count(), 2);
    }

    #[test]
    fn test_batch_parallel() {
        let config = BatchConfig::parallel(2);
        let processor = BatchInferenceProcessor::new(mock_inference, config);

        let batch = PromptBatch::new(
            vec![
                "Prompt 1".to_string(),
                "Prompt 2".to_string(),
                "Prompt 3".to_string(),
            ],
            InferenceParams::default(),
        );

        let result = processor.process_batch(batch).unwrap();
        assert_eq!(result.responses.len(), 3);
        assert_eq!(result.successful_count(), 3);
    }

    #[test]
    fn test_batch_builder() {
        let processor = BatchInferenceBuilder::new()
            .max_concurrent(4)
            .timeout(5000)
            .continue_on_error(true)
            .build(mock_inference);

        let batch = PromptBatch::new(vec!["Test".to_string()], InferenceParams::default());

        let result = processor.process_batch(batch).unwrap();
        assert_eq!(result.successful_count(), 1);
    }
}
