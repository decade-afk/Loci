use loci::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
use loci::error::{LociError, Result};
use loci::prelude::{InferenceEngine, ModelConfig};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

struct RecordingBackend {
    seen: Arc<Mutex<Option<BackendParams>>>,
}

impl InferenceBackend for RecordingBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "recording".to_string(),
            version: "1.0".to_string(),
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
        _model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        *self.seen.lock().expect("lock poisoned") = Some(backend_params);
        Ok(Box::new(RecordingModel))
    }
}

struct RecordingModel;

impl Model for RecordingModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "recording".to_string(),
            n_vocab: 1024,
            n_ctx_train: 4096,
            n_embd: 256,
            n_layer: 8,
            param_count: Some(1_000_000),
        }
    }

    fn infer_text(&mut self, prompt: &str, _params: &InferenceParams) -> Result<String> {
        Ok(prompt.to_string())
    }
}

struct AdaptiveRetryBackend {
    attempts: Arc<Mutex<Vec<BackendParams>>>,
    max_gpu_layers: i32,
    cpu_only_success: bool,
}

impl InferenceBackend for AdaptiveRetryBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "adaptive".to_string(),
            version: "1.0".to_string(),
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
        _model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        self.attempts
            .lock()
            .expect("lock poisoned")
            .push(backend_params.clone());

        if !backend_params.use_gpu {
            if self.cpu_only_success {
                return Ok(Box::new(RecordingModel));
            }
            return Err(LociError::OutOfMemory(
                "cpu-only fallback disabled for test".to_string(),
            ));
        }

        if backend_params.n_gpu_layers > self.max_gpu_layers {
            return Err(LociError::OutOfMemory(
                "insufficient VRAM for requested GPU layer placement".to_string(),
            ));
        }

        Ok(Box::new(RecordingModel))
    }
}

fn temp_model_file() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("monotonic clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("loci-recording-model-{nonce}.gguf"));
    fs::write(&path, b"mock-model").expect("write temp model");
    path
}

#[test]
fn test_inference_engine_builder() {
    let builder = InferenceEngine::builder();

    // Test builder pattern
    let result = builder.model_path("nonexistent.gguf").build();

    // Should fail with model not found
    assert!(result.is_err());
    match result {
        Err(LociError::ConfigError(_))
        | Err(LociError::ModelLoadError(_))
        | Err(LociError::IoError(_)) => {}
        Err(_) => panic!("Expected model load error"),
        Ok(_) => panic!("Expected build error"),
    }
}

#[test]
fn test_generation_params_conversion() {
    use loci::inference::GenerationParams;

    let gen_params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        min_p: 0.0,
        top_k: 50,
        repeat_penalty: 1.2,
    };

    let inference_params: InferenceParams = gen_params.into();

    assert_eq!(inference_params.max_tokens, 100);
    assert_eq!(inference_params.temperature, 0.7);
    assert_eq!(inference_params.top_p, 0.9);
    assert_eq!(inference_params.min_p, 0.0);
    assert_eq!(inference_params.top_k, 50);
    assert_eq!(inference_params.repeat_penalty, 1.2);
}

#[test]
fn test_generation_params_default() {
    use loci::inference::GenerationParams;

    let params = GenerationParams::default();

    assert_eq!(params.max_tokens, 512);
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.95);
    assert_eq!(params.min_p, 0.0);
    assert_eq!(params.top_k, 40);
    assert_eq!(params.repeat_penalty, 1.1);
}

#[test]
fn test_model_config_validation() {
    // Test missing model path
    let config = ModelConfig::new("test.gguf")
        .with_context_size(2048)
        .with_threads(4);

    assert!(config.validate().is_err());

    // Test invalid context size
    let invalid_config = ModelConfig::new("test.gguf").with_context_size(0);

    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_inference_params_default() {
    let params = InferenceParams::default();

    assert_eq!(params.n_ctx, 4096);
    assert_eq!(params.n_batch, 512);
    assert_eq!(params.max_tokens, 512);
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.95);
    assert_eq!(params.min_p, 0.0);
    assert_eq!(params.top_k, 40);
    assert_eq!(params.repeat_penalty, 1.1);
}

#[test]
fn test_backend_params_default() {
    let params = BackendParams::default();

    assert_eq!(params.n_gpu_layers, -1);
    assert_eq!(params.use_gpu, true);
    assert!(params.use_mmap);
    assert!(!params.use_mlock);
    assert!(params.kv_offload);
    assert!(params.op_offload);
    assert_eq!(params.split_mode, GpuSplitMode::Layer);
    assert_eq!(params.main_gpu, 0);
    assert!(params.tensor_split.is_none());
    assert!(params.options.is_empty());
}

#[test]
fn test_builder_propagates_tiered_loading_params() {
    let model_path = temp_model_file();
    let seen = Arc::new(Mutex::new(None));
    let mut registry = BackendRegistry::new();
    registry.register(
        "recording".to_string(),
        Box::new(RecordingBackend {
            seen: Arc::clone(&seen),
        }),
    );

    let engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("recording")
        .with_backend_registry(registry)
        .gpu_layers(24)
        .with_mmap(false)
        .with_mlock(true)
        .with_kv_offload(false)
        .with_op_offload(false)
        .with_gpu_split_mode(GpuSplitMode::Row)
        .with_main_gpu(1)
        .with_tensor_split(vec![3.0, 2.0, 1.0])
        .build();
    assert!(engine.is_ok(), "recording engine should build");

    let params = seen
        .lock()
        .expect("lock poisoned")
        .clone()
        .expect("backend params should be captured");
    assert_eq!(params.n_gpu_layers, 24);
    assert!(params.use_gpu);
    assert!(!params.use_mmap);
    assert!(params.use_mlock);
    assert!(!params.kv_offload);
    assert!(!params.op_offload);
    assert_eq!(params.split_mode, GpuSplitMode::Row);
    assert_eq!(params.main_gpu, 1);
    assert_eq!(params.tensor_split, Some(vec![3.0, 2.0, 1.0]));

    let _ = fs::remove_file(model_path);
}

#[test]
fn test_builder_cpu_only_disables_device_offload() {
    let model_path = temp_model_file();
    let seen = Arc::new(Mutex::new(None));
    let mut registry = BackendRegistry::new();
    registry.register(
        "recording".to_string(),
        Box::new(RecordingBackend {
            seen: Arc::clone(&seen),
        }),
    );

    let engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("recording")
        .with_backend_registry(registry)
        .gpu_layers(24)
        .with_kv_offload(true)
        .with_op_offload(true)
        .cpu_only()
        .build();
    assert!(engine.is_ok(), "recording engine should build");

    let params = seen
        .lock()
        .expect("lock poisoned")
        .clone()
        .expect("backend params should be captured");
    assert_eq!(params.n_gpu_layers, 0);
    assert!(!params.use_gpu);
    assert!(!params.kv_offload);
    assert!(!params.op_offload);
    assert_eq!(params.split_mode, GpuSplitMode::None);
    assert_eq!(params.main_gpu, 0);
    assert!(params.tensor_split.is_none());

    let _ = fs::remove_file(model_path);
}

#[test]
fn test_builder_auto_gpu_fallback_retries_with_lower_gpu_layers() {
    let model_path = temp_model_file();
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let mut registry = BackendRegistry::new();
    registry.register(
        "adaptive".to_string(),
        Box::new(AdaptiveRetryBackend {
            attempts: Arc::clone(&attempts),
            max_gpu_layers: 16,
            cpu_only_success: true,
        }),
    );

    let engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("adaptive")
        .with_backend_registry(registry)
        .gpu_layers(24)
        .with_auto_gpu_layer_fallback(8)
        .build();
    assert!(engine.is_ok(), "adaptive backend should build after retry");

    let attempts = attempts.lock().expect("lock poisoned").clone();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].n_gpu_layers, 24);
    assert!(attempts[0].use_gpu);
    assert_eq!(attempts[1].n_gpu_layers, 16);
    assert!(attempts[1].use_gpu);

    let _ = fs::remove_file(model_path);
}

#[test]
fn test_builder_auto_gpu_fallback_can_drop_to_cpu_only() {
    let model_path = temp_model_file();
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let mut registry = BackendRegistry::new();
    registry.register(
        "adaptive".to_string(),
        Box::new(AdaptiveRetryBackend {
            attempts: Arc::clone(&attempts),
            max_gpu_layers: -1,
            cpu_only_success: true,
        }),
    );

    let engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("adaptive")
        .with_backend_registry(registry)
        .gpu_layers(8)
        .with_gpu_split_mode(GpuSplitMode::Row)
        .with_main_gpu(1)
        .with_tensor_split(vec![2.0, 1.0])
        .with_auto_gpu_layer_fallback(8)
        .build();
    assert!(
        engine.is_ok(),
        "adaptive backend should fall back to cpu-only"
    );

    let attempts = attempts.lock().expect("lock poisoned").clone();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].n_gpu_layers, 8);
    assert!(attempts[0].use_gpu);
    assert_eq!(attempts[1].n_gpu_layers, 0);
    assert!(!attempts[1].use_gpu);
    assert_eq!(attempts[1].split_mode, GpuSplitMode::None);
    assert_eq!(attempts[1].main_gpu, 0);
    assert!(attempts[1].tensor_split.is_none());

    let _ = fs::remove_file(model_path);
}

// Mock tests for when we don't have a real model file
#[cfg(test)]
mod mock_tests {
    use super::*;

    // These tests would run with a mock model file in CI
    #[ignore = "Requires model file"]
    #[test]
    fn test_inference_engine_creation() {
        let config = ModelConfig::new("test_model.gguf")
            .with_context_size(1024)
            .cpu_only();

        let engine = InferenceEngine::new(config);
        assert!(engine.is_ok());
    }

    #[ignore = "Requires model file"]
    #[test]
    fn test_text_generation() {
        let config = ModelConfig::new("test_model.gguf")
            .with_context_size(1024)
            .cpu_only();

        let mut engine = InferenceEngine::new(config).unwrap();
        let params = InferenceParams::default();

        let result = engine.generate_with_params("Hello", &params);
        assert!(result.is_ok());
    }
}
