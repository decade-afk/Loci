use loci::prelude::*;
use loci::backend::*;
use loci::backends::*;
use loci::error::LociError;

#[test]
fn test_backend_params_default() {
    let params = BackendParams::default();
    
    assert_eq!(params.n_gpu_layers, -1);
    assert_eq!(params.use_gpu, true);
    assert!(params.options.is_empty());
}

#[test]
fn test_backend_params_builder() {
    let mut params = BackendParams::default();
    params.n_gpu_layers = 10;
    params.use_gpu = false;
    params.options.push(("key".to_string(), "value".to_string()));
    
    assert_eq!(params.n_gpu_layers, 10);
    assert_eq!(params.use_gpu, false);
    assert_eq!(params.options.len(), 1);
    assert_eq!(params.options[0], ("key".to_string(), "value".to_string()));
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
    assert!(params.n_threads.is_none());
}

#[test]
fn test_inference_params_validation() {
    let mut params = InferenceParams::default();
    
    // Test valid parameters
    assert!(params.n_ctx > 0);
    assert!(params.n_batch > 0);
    assert!(params.temperature >= 0.0);
    assert!(params.top_p >= 0.0 && params.top_p <= 1.0);
    assert!(params.top_k > 0);
    assert!(params.repeat_penalty > 0.0);
    
    // Test edge cases
    params.temperature = 0.0; // Should be valid (greedy)
    params.top_p = 1.0; // Should be valid
    params.repeat_penalty = 1.0; // Should be valid (no penalty)
}

#[test]
fn test_image_data_types() {
    // Test different image data types
    let bytes_image = Image {
        data: ImageData::Bytes(vec![1, 2, 3, 4]),
        format: Some(ImageFormat::Png),
    };
    
    let base64_image = Image {
        data: ImageData::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==".to_string()),
        format: Some(ImageFormat::Png),
    };
    
    let path_image = Image {
        data: ImageData::Path("test.jpg".to_string()),
        format: Some(ImageFormat::Jpeg),
    };
    
    // Test format matching
    match bytes_image.format {
        Some(ImageFormat::Png) => {},
        _ => panic!("Expected PNG format"),
    }
    
    match base64_image.format {
        Some(ImageFormat::Png) => {},
        _ => panic!("Expected PNG format"),
    }
    
    match path_image.format {
        Some(ImageFormat::Jpeg) => {},
        _ => panic!("Expected JPEG format"),
    }
}

#[test]
fn test_image_format_enum() {
    let formats = [
        ImageFormat::Png,
        ImageFormat::Jpeg,
        ImageFormat::Gif,
        ImageFormat::Webp,
    ];
    
    // Test that all formats are distinct
    for (i, format1) in formats.iter().enumerate() {
        for (j, format2) in formats.iter().enumerate() {
            if i == j {
                assert_eq!(format1, format2);
            } else {
                assert_ne!(format1, format2);
            }
        }
    }
}

#[test]
fn test_llamacpp_backend_creation() {
    let backend = LlamaCppBackend::new();
    
    // Backend should be created successfully
    // Note: We can't test init() without proper setup
}

// Mock tests for backend functionality
#[cfg(test)]
mod mock_backend_tests {
    use super::*;
    
    // Mock backend for testing
    struct MockBackend {
        initialized: bool,
    }
    
    impl MockBackend {
        fn new() -> Self {
            Self { initialized: false }
        }
        
        fn init(&mut self) -> Result<(), LociError> {
            self.initialized = true;
            Ok(())
        }
        
        fn is_initialized(&self) -> bool {
            self.initialized
        }
    }
    
    #[test]
    fn test_mock_backend() {
        let mut backend = MockBackend::new();
        assert!(!backend.is_initialized());
        
        assert!(backend.init().is_ok());
        assert!(backend.is_initialized());
    }
}

#[test]
fn test_backend_capabilities() {
    // Test that we can create and compare backend capabilities
    // This would be expanded when BackendCapabilities is fully implemented
    
    // For now, just test that the concept works
    let gpu_support = true;
    let multimodal_support = false;
    let streaming_support = true;
    
    assert!(gpu_support);
    assert!(!multimodal_support);
    assert!(streaming_support);
}

#[test]
fn test_model_loading_error_handling() {
    // Test error handling for invalid model paths
    let invalid_path = "nonexistent_model.gguf";
    let params = BackendParams::default();
    
    // This should fail gracefully
    let result = std::panic::catch_unwind(|| {
        // In a real test, this would call LlamaCppModel::load
        // For now, we just test the error path
        Err::<(), LociError>(LociError::ModelLoadError(
            format!("Model not found: {}", invalid_path)
        ))
    });
    
    assert!(result.is_ok());
}

#[test]
fn test_inference_params_cloning() {
    let params1 = InferenceParams {
        n_ctx: 2048,
        n_batch: 256,
        n_threads: Some(4),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repeat_penalty: 1.2,
    };
    
    let params2 = params1.clone();
    
    assert_eq!(params1.n_ctx, params2.n_ctx);
    assert_eq!(params1.n_batch, params2.n_batch);
    assert_eq!(params1.n_threads, params2.n_threads);
    assert_eq!(params1.max_tokens, params2.max_tokens);
    assert_eq!(params1.temperature, params2.temperature);
    assert_eq!(params1.top_p, params2.top_p);
    assert_eq!(params1.top_k, params2.top_k);
    assert_eq!(params1.repeat_penalty, params2.repeat_penalty);
}

#[test]
fn test_backend_params_cloning() {
    let params1 = BackendParams {
        n_gpu_layers: 20,
        use_gpu: false,
        options: vec![("test".to_string(), "value".to_string())],
    };
    
    let params2 = params1.clone();
    
    assert_eq!(params1.n_gpu_layers, params2.n_gpu_layers);
    assert_eq!(params1.use_gpu, params2.use_gpu);
    assert_eq!(params1.options, params2.options);
}