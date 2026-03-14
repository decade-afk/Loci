use loci::prelude::*;

#[test]
fn test_model_config_default() {
    let config = ModelConfig::default();
    assert_eq!(config.n_ctx, 4096);
    assert_eq!(config.n_batch, 512);
    assert_eq!(config.use_gpu, true);
    assert_eq!(config.n_gpu_layers, -1);
    assert!(config.use_mmap);
    assert!(!config.use_mlock);
    assert!(config.kv_offload);
    assert!(config.op_offload);
    assert_eq!(config.split_mode, GpuSplitMode::Layer);
    assert_eq!(config.main_gpu, 0);
    assert!(config.tensor_split.is_none());
    assert_eq!(config.load_strategy, loci::model::ModelLoadStrategy::Strict);
}

#[test]
fn test_model_config_builder() {
    let config = ModelConfig::new("test.gguf")
        .with_context_size(2048)
        .with_threads(4)
        .with_batch_size(256)
        .with_mmap(false)
        .with_mlock(true)
        .with_kv_offload(true)
        .with_op_offload(true)
        .with_gpu_split_mode(GpuSplitMode::Row)
        .with_main_gpu(1)
        .with_tensor_split(vec![3.0, 2.0, 1.0])
        .with_auto_gpu_layer_fallback(8)
        .cpu_only();

    assert_eq!(config.n_ctx, 2048);
    assert_eq!(config.n_threads, Some(4));
    assert_eq!(config.n_batch, 256);
    assert_eq!(config.use_gpu, false);
    assert_eq!(config.n_gpu_layers, 0);
    assert!(!config.use_mmap);
    assert!(config.use_mlock);
    assert!(!config.kv_offload);
    assert!(!config.op_offload);
    assert_eq!(config.split_mode, GpuSplitMode::None);
    assert_eq!(config.main_gpu, 0);
    assert!(config.tensor_split.is_none());
    assert_eq!(
        config.load_strategy,
        loci::model::ModelLoadStrategy::AutoReduceGpuLayers { step: 8 }
    );
}

#[test]
fn test_generation_params_default() {
    let params = loci::inference::GenerationParams::default();
    assert_eq!(params.max_tokens, 512);
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.95);
    assert_eq!(params.top_k, 40);
}
