use loci::prelude::*;

#[test]
fn test_model_config_default() {
    let config = ModelConfig::default();
    assert_eq!(config.n_ctx, 4096);
    assert_eq!(config.n_batch, 512);
    assert_eq!(config.use_gpu, true);
    assert_eq!(config.n_gpu_layers, -1);
}

#[test]
fn test_model_config_builder() {
    let config = ModelConfig::new("test.gguf")
        .with_context_size(2048)
        .with_threads(4)
        .with_batch_size(256)
        .cpu_only();

    assert_eq!(config.n_ctx, 2048);
    assert_eq!(config.n_threads, Some(4));
    assert_eq!(config.n_batch, 256);
    assert_eq!(config.use_gpu, false);
    assert_eq!(config.n_gpu_layers, 0);
}

#[test]
fn test_generation_params_default() {
    let params = loci::inference::GenerationParams::default();
    assert_eq!(params.max_tokens, 512);
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_p, 0.95);
    assert_eq!(params.top_k, 40);
}
