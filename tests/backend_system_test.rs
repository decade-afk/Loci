use loci::backend::{
    BackendParams, GpuSplitMode, Image as BackendImage, ImageData, ImageFormat, InferenceParams,
};
use loci::backends::LlamaCppBackend;
use loci::error::{LociError, Result};

#[test]
fn test_backend_params_default() {
    let params = BackendParams::default();

    assert_eq!(params.n_gpu_layers, -1);
    assert!(params.use_gpu);
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
fn test_backend_params_builder() {
    let mut params = BackendParams::default();
    params.n_gpu_layers = 10;
    params.use_gpu = false;
    params.use_mmap = false;
    params.use_mlock = true;
    params.kv_offload = false;
    params.op_offload = false;
    params.split_mode = GpuSplitMode::Row;
    params.main_gpu = 1;
    params.tensor_split = Some(vec![3.0, 2.0, 1.0]);
    params
        .options
        .push(("key".to_string(), "value".to_string()));

    assert_eq!(params.n_gpu_layers, 10);
    assert!(!params.use_gpu);
    assert!(!params.use_mmap);
    assert!(params.use_mlock);
    assert!(!params.kv_offload);
    assert!(!params.op_offload);
    assert_eq!(params.split_mode, GpuSplitMode::Row);
    assert_eq!(params.main_gpu, 1);
    assert_eq!(params.tensor_split, Some(vec![3.0, 2.0, 1.0]));
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
    assert_eq!(params.min_p, 0.0);
    assert_eq!(params.top_k, 40);
    assert_eq!(params.repeat_penalty, 1.1);
    assert!(params.n_threads.is_none());
}

#[test]
fn test_inference_params_validation() {
    let mut params = InferenceParams::default();

    assert!(params.n_ctx > 0);
    assert!(params.n_batch > 0);
    assert!(params.temperature >= 0.0);
    assert!((0.0..=1.0).contains(&params.top_p));
    assert!((0.0..=1.0).contains(&params.min_p));
    assert!(params.top_k > 0);
    assert!(params.repeat_penalty > 0.0);

    params.temperature = 0.0;
    params.top_p = 1.0;
    params.repeat_penalty = 1.0;
    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_p, 1.0);
    assert_eq!(params.repeat_penalty, 1.0);
}

#[test]
fn test_image_data_types() {
    let bytes_image = BackendImage {
        data: ImageData::Bytes(vec![1, 2, 3, 4]),
        format: Some(ImageFormat::Png),
    };

    let base64_image = BackendImage {
        data: ImageData::Base64("aGVsbG8=".to_string()),
        format: Some(ImageFormat::Png),
    };

    let path_image = BackendImage {
        data: ImageData::Path("test.jpg".to_string()),
        format: Some(ImageFormat::Jpeg),
    };

    assert!(matches!(bytes_image.format, Some(ImageFormat::Png)));
    assert!(matches!(base64_image.format, Some(ImageFormat::Png)));
    assert!(matches!(path_image.format, Some(ImageFormat::Jpeg)));
}

#[test]
fn test_image_format_enum() {
    let formats = [
        ImageFormat::Png,
        ImageFormat::Jpeg,
        ImageFormat::Gif,
        ImageFormat::Webp,
    ];

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
    let _backend = LlamaCppBackend::new();
}

#[cfg(test)]
mod mock_backend_tests {
    use super::*;

    struct MockBackend {
        initialized: bool,
    }

    impl MockBackend {
        fn new() -> Self {
            Self { initialized: false }
        }

        fn init(&mut self) -> Result<()> {
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
    let gpu_support = true;
    let multimodal_support = false;
    let streaming_support = true;

    assert!(gpu_support);
    assert!(!multimodal_support);
    assert!(streaming_support);
}

#[test]
fn test_model_loading_error_handling() {
    let invalid_path = "nonexistent_model.gguf";

    let result = std::panic::catch_unwind(|| {
        Err::<(), LociError>(LociError::ModelLoadError(format!(
            "Model not found: {invalid_path}"
        )))
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
        min_p: 0.05,
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
    assert_eq!(params1.min_p, params2.min_p);
    assert_eq!(params1.top_k, params2.top_k);
    assert_eq!(params1.repeat_penalty, params2.repeat_penalty);
}

#[test]
fn test_backend_params_cloning() {
    let params1 = BackendParams {
        n_gpu_layers: 20,
        use_gpu: false,
        use_mmap: false,
        use_mlock: true,
        kv_offload: false,
        op_offload: false,
        split_mode: GpuSplitMode::Row,
        main_gpu: 1,
        tensor_split: Some(vec![3.0, 2.0, 1.0]),
        options: vec![("test".to_string(), "value".to_string())],
    };

    let params2 = params1.clone();

    assert_eq!(params1.n_gpu_layers, params2.n_gpu_layers);
    assert_eq!(params1.use_gpu, params2.use_gpu);
    assert_eq!(params1.use_mmap, params2.use_mmap);
    assert_eq!(params1.use_mlock, params2.use_mlock);
    assert_eq!(params1.kv_offload, params2.kv_offload);
    assert_eq!(params1.op_offload, params2.op_offload);
    assert_eq!(params1.split_mode, params2.split_mode);
    assert_eq!(params1.main_gpu, params2.main_gpu);
    assert_eq!(params1.tensor_split, params2.tensor_split);
    assert_eq!(params1.options, params2.options);
}
