use loci::prelude::*;
use loci::error::LociError;

#[test]
fn test_inference_engine_builder() {
    let builder = InferenceEngine::builder();
    
    // Test builder pattern
    let result = builder
        .model_path("nonexistent.gguf")
        .build();
    
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
        top_k: 50,
        repeat_penalty: 1.2,
    };
    
    let inference_params: InferenceParams = gen_params.into();
    
    assert_eq!(inference_params.max_tokens, 100);
    assert_eq!(inference_params.temperature, 0.7);
    assert_eq!(inference_params.top_p, 0.9);
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
    let invalid_config = ModelConfig::new("test.gguf")
        .with_context_size(0);
    
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
    assert_eq!(params.top_k, 40);
    assert_eq!(params.repeat_penalty, 1.1);
}

#[test]
fn test_backend_params_default() {
    let params = BackendParams::default();
    
    assert_eq!(params.n_gpu_layers, -1);
    assert_eq!(params.use_gpu, true);
    assert!(params.options.is_empty());
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
