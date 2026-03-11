use loci::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, InferenceBackend, InferenceParams, Model,
    ModelMetadata,
};
use loci::error::Result;
use loci::inference::ExecutionPolicy;
use loci::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

struct MockBackend;

impl InferenceBackend for MockBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "mock".to_string(),
            version: "1.0".to_string(),
            supports_text: true,
            supports_multimodal: false,
            supports_embeddings: false,
            supports_streaming: true,
            has_gpu_support: false,
            supported_formats: vec!["gguf".to_string()],
        }
    }

    fn load_model(
        &self,
        _model_path: &Path,
        _backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        Ok(Box::new(MockModel))
    }
}

struct MockModel;

impl Model for MockModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "mock".to_string(),
            n_vocab: 1024,
            n_ctx_train: 4096,
            n_embd: 256,
            n_layer: 8,
            param_count: Some(1_000_000),
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn infer_text(&mut self, prompt: &str, _params: &InferenceParams) -> Result<String> {
        Ok(format!("model:{prompt}"))
    }

    fn infer_stream(
        &mut self,
        prompt: &str,
        _params: &InferenceParams,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        if !callback("model:") {
            return Ok(());
        }
        let _ = callback(prompt);
        Ok(())
    }
}

struct PrefixExecutionPolicy {
    prefix: &'static str,
}

impl ExecutionPolicy for PrefixExecutionPolicy {
    fn name(&self) -> &str {
        "test.prefix.policy"
    }

    fn generate_text(
        &self,
        _engine: &mut InferenceEngine,
        prompt: &str,
        _params: &InferenceParams,
        _timeout_override: Option<std::time::Duration>,
    ) -> Result<String> {
        Ok(format!("{}{}", self.prefix, prompt))
    }

    fn generate_stream(
        &self,
        _engine: &mut InferenceEngine,
        prompt: &str,
        _params: &InferenceParams,
        _timeout_override: Option<std::time::Duration>,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        if callback(self.prefix) {
            let _ = callback(prompt);
        }
        Ok(())
    }
}

fn temp_model_file() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("monotonic clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("loci-mock-model-{nonce}.gguf"));
    fs::write(&path, b"mock-model").expect("write temp model");
    path
}

fn build_mock_engine() -> (InferenceEngine, PathBuf) {
    let model_path = temp_model_file();
    let mut registry = BackendRegistry::new();
    registry.register("mock".to_string(), Box::new(MockBackend));

    let engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("mock")
        .with_backend_registry(registry)
        .build()
        .expect("mock engine should build");
    (engine, model_path)
}

#[test]
fn set_execution_policy_replaces_default_behavior() {
    let (mut engine, model_path) = build_mock_engine();
    let params = InferenceParams::default();

    let baseline = engine
        .generate_with_params("hello", &params)
        .expect("default generation should work");
    assert_eq!(baseline, "model:hello");

    engine.set_execution_policy(PrefixExecutionPolicy { prefix: "policy:" });
    assert_eq!(engine.execution_policy_name(), "test.prefix.policy");

    let routed = engine
        .generate_with_params("hello", &params)
        .expect("policy generation should work");
    assert_eq!(routed, "policy:hello");

    let _ = fs::remove_file(model_path);
}

#[test]
fn builder_execution_policy_is_applied() {
    let model_path = temp_model_file();
    let mut registry = BackendRegistry::new();
    registry.register("mock".to_string(), Box::new(MockBackend));

    let mut engine = InferenceEngine::builder()
        .model_path(&model_path)
        .backend("mock")
        .with_backend_registry(registry)
        .with_execution_policy(PrefixExecutionPolicy { prefix: "builder:" })
        .build()
        .expect("mock engine should build");
    let params = InferenceParams::default();

    let generated = engine
        .generate_with_params("hello", &params)
        .expect("policy generation should work");
    assert_eq!(generated, "builder:hello");
    assert_eq!(engine.execution_policy_name(), "test.prefix.policy");

    let mut streamed = String::new();
    engine
        .generate_stream_with_params("world", &params, |token| {
            streamed.push_str(token);
            true
        })
        .expect("policy stream should work");
    assert_eq!(streamed, "builder:world");

    let _ = fs::remove_file(model_path);
}
