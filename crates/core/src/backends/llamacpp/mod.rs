mod adapter;
mod driver;
mod execute;
mod ffi;
mod plan;
mod runtime;
mod session;

use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelMetadata,
};
use crate::error::{LociError, Result};
use crate::plugin::PluginSamplingRuntime;
use adapter::{LlamaCppAdapter, StubLlamaCppAdapter};
use driver::{
    discover_driver, LlamaCppBackendSession, LlamaCppContextCreateRequest, LlamaCppCreatedContext,
    LlamaCppLoadedModel, LlamaCppModelLoadRequest,
};
use plan::LlamaCppLoadPlan;
use runtime::LlamaCppRuntimeState;
use std::path::Path;

pub struct LlamaCppBackend {
    adapter: Box<dyn LlamaCppAdapter>,
    initialized: bool,
}

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self {
            adapter: Box::new(StubLlamaCppAdapter::new()),
            initialized: false,
        }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LlamaCppModel {
    // Drop order matters: context before model before backend session.
    native_context: LlamaCppCreatedContext,
    native_model: LlamaCppLoadedModel,
    #[allow(dead_code)]
    backend_session: LlamaCppBackendSession,
    load_plan: LlamaCppLoadPlan,
    metadata: ModelMetadata,
    runtime_state: LlamaCppRuntimeState,
    sampling_runtime: PluginSamplingRuntime,
}

impl Model for LlamaCppModel {
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "prompt must not be empty".to_string(),
            ));
        }
        self.infer_text_native(prompt, params)
    }

    fn attach_sampling_runtime(&mut self, runtime: PluginSamplingRuntime) -> Result<()> {
        self.sampling_runtime = runtime;
        Ok(())
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "llama.cpp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
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
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        if !self.initialized {
            return Err(LociError::BackendError(
                "llama.cpp backend not initialized".to_string(),
            ));
        }

        let adapter_context = self.adapter.build_context()?;
        let load_plan = LlamaCppLoadPlan::from_backend_params(model_path, backend_params)?;
        let driver = discover_driver();
        let backend_session = driver.init_backend(&adapter_context)?;
        let native_model = driver.load_model(LlamaCppModelLoadRequest {
            model_path,
            load_plan: &load_plan,
        })?;
        let metadata = native_model.metadata().clone();
        let native_context = driver.create_context(LlamaCppContextCreateRequest {
            loaded_model: &native_model,
            load_plan: &load_plan,
            runtime_override: None,
        })?;
        let runtime_state = load_plan.create_runtime_state();

        Ok(Box::new(LlamaCppModel {
            native_context,
            native_model,
            backend_session,
            load_plan,
            metadata,
            runtime_state,
            sampling_runtime: PluginSamplingRuntime::default(),
        }))
    }

    fn init(&mut self) -> Result<()> {
        self.adapter.validate_environment()?;
        self.initialized = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendParams, GpuSplitMode};
    use crate::backends::llamacpp::adapter::{LlamaCppBuildIntegration, LlamaCppSourceLayout};
    use std::path::{Path, PathBuf};

    fn assert_path_ends_with(path: &Path, suffix: &[&str]) {
        let suffix = suffix.iter().fold(PathBuf::new(), |mut acc, segment| {
            acc.push(segment);
            acc
        });
        assert!(
            path.ends_with(&suffix),
            "expected {} to end with {}",
            path.display(),
            suffix.display()
        );
    }

    #[test]
    fn load_plan_requires_gguf_extension() {
        let err =
            LlamaCppLoadPlan::from_backend_params(Path::new("demo.bin"), BackendParams::default())
                .expect_err("should fail");
        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn load_plan_disables_gpu_specific_flags_in_cpu_mode() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                use_gpu: false,
                n_gpu_layers: 32,
                kv_offload: true,
                op_offload: true,
                main_gpu: 2,
                tensor_split: Some(vec![2.0, 1.0]),
                ..Default::default()
            },
        )
        .expect("plan");

        assert!(!plan.gpu_active());
        assert_eq!(plan.n_gpu_layers(), 0);
        assert!(!plan.kv_offload());
        assert!(!plan.op_offload());
        assert_eq!(plan.split_mode(), GpuSplitMode::None);
        assert_eq!(plan.main_gpu(), 0);
        assert!(plan.tensor_split().is_none());
    }

    #[test]
    fn load_plan_reads_runtime_options() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![
                    ("n_ctx".to_string(), "8192".to_string()),
                    ("n_batch".to_string(), "1024".to_string()),
                    ("n_threads".to_string(), "12".to_string()),
                ],
                ..Default::default()
            },
        )
        .expect("plan");

        assert_eq!(plan.runtime().n_ctx(), 8192);
        assert_eq!(plan.runtime().n_batch(), 1024);
        assert_eq!(plan.runtime().n_threads(), Some(12));
    }

    #[test]
    fn load_plan_rejects_invalid_runtime_option() {
        let err = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![("n_ctx".to_string(), "bad".to_string())],
                ..Default::default()
            },
        )
        .expect_err("should fail");

        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn runtime_state_is_seeded_from_load_plan() {
        let plan = LlamaCppLoadPlan::from_backend_params(
            Path::new("demo.gguf"),
            BackendParams {
                options: vec![
                    ("n_ctx".to_string(), "4096".to_string()),
                    ("n_batch".to_string(), "256".to_string()),
                    ("n_threads".to_string(), "8".to_string()),
                ],
                ..Default::default()
            },
        )
        .expect("plan");

        let runtime_state = plan.create_runtime_state();
        assert_eq!(runtime_state.current_n_ctx(), 4096);
        assert_eq!(runtime_state.current_n_batch(), 256);
        assert_eq!(runtime_state.current_n_threads(), Some(8));
        assert!(runtime_state.kv_offload());
    }

    #[test]
    fn execution_config_rejects_zero_max_tokens() {
        let err = runtime::LlamaCppExecutionConfig::from_inference_params(&InferenceParams {
            max_tokens: 0,
            ..Default::default()
        })
        .expect_err("should fail");
        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn runtime_options_can_compare_execution_shape() {
        let options = plan::LlamaCppRuntimeOptions::new(4096, 512, Some(4));
        assert!(options.supports(&InferenceParams {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: Some(4),
            ..Default::default()
        }));
        assert!(!options.supports(&InferenceParams {
            n_ctx: 8192,
            ..Default::default()
        }));
    }

    #[test]
    fn source_layout_matches_cloned_repo_structure() {
        let layout = LlamaCppSourceLayout::discover().expect("layout");
        assert_path_ends_with(&layout.include_dir, &["deps", "llama.cpp", "include"]);
        assert_path_ends_with(
            &layout.llama_header,
            &["deps", "llama.cpp", "include", "llama.h"],
        );
        assert_path_ends_with(
            &layout.ggml_include_dir,
            &["deps", "llama.cpp", "ggml", "include"],
        );
    }

    #[test]
    fn build_integration_matches_workspace_layout() {
        let integration = LlamaCppBuildIntegration::discover().expect("integration");
        assert_path_ends_with(&integration.build_script, &["crates", "core", "build.rs"]);
        assert_path_ends_with(
            &integration.ffi_module,
            &["crates", "core", "src", "backends", "llamacpp", "ffi.rs"],
        );
        assert_path_ends_with(
            &integration.ffi_shim_c,
            &[
                "crates",
                "core",
                "src",
                "backends",
                "llamacpp",
                "ffi_shim.c",
            ],
        );
    }

    #[test]
    fn adapter_context_contains_driver_protocol() {
        let adapter = StubLlamaCppAdapter::new();
        let context = adapter.build_context().expect("context");
        assert_eq!(context.driver_protocol.kind, "native");
        assert_eq!(context.driver_protocol.backend_init_symbol, "backend_init");
        assert_eq!(context.driver_protocol.phases.init.function, "backend_init");
        assert_eq!(
            context
                .driver_protocol
                .phases
                .init
                .companion_free_function
                .as_deref(),
            Some("backend_free")
        );
        assert_eq!(
            context.driver_protocol.phases.load_model.function,
            "LlamaModel::from_file"
        );
        assert_eq!(
            context.driver_protocol.phases.create_context.function,
            "LlamaContext::new"
        );
        assert_path_ends_with(
            Path::new(&context.driver_protocol.ffi_module),
            &["crates", "core", "src", "backends", "llamacpp", "ffi.rs"],
        );
        assert_eq!(context.driver_protocol.lifecycle.model_type, "LlamaModel");
        assert_eq!(
            context.driver_protocol.lifecycle.context_type,
            "LlamaContext"
        );
        assert!(context.driver_protocol.lifecycle.supports_tokenize);
        assert!(context.driver_protocol.lifecycle.supports_token_to_str);
        assert!(context.driver_protocol.lifecycle.supports_decode);
        assert!(context.driver_protocol.lifecycle.supports_logits);
        assert!(context.driver_protocol.lifecycle.supports_kv_cache_clear);
    }

    #[test]
    #[ignore = "requires an opt-in host-native llama.cpp backend smoke check"]
    fn native_driver_executes_backend_init_phase() {
        let adapter = StubLlamaCppAdapter::new();
        let context = adapter.build_context().expect("context");
        let driver = discover_driver();
        let backend = driver.init_backend(&context).expect("backend init");
        assert_eq!(backend.kind(), "native");
        assert!(backend.is_native());
    }

    fn llama_test_model_path() -> PathBuf {
        std::env::var_os("LOCI_LLAMA_TEST_MODEL")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .expect("set LOCI_LLAMA_TEST_MODEL to a decoder gguf model file")
    }

    #[test]
    #[ignore = "requires LOCI_LLAMA_TEST_MODEL=/path/to/decoder-model.gguf"]
    fn real_llama_runtime_smoke_test() {
        let model_path = llama_test_model_path();
        let mut backend = LlamaCppBackend::new();
        backend.init().expect("backend init");

        let mut model = backend
            .load_model(
                &model_path,
                BackendParams {
                    use_gpu: false,
                    n_gpu_layers: 0,
                    options: vec![
                        ("n_ctx".to_string(), "512".to_string()),
                        ("n_batch".to_string(), "64".to_string()),
                    ],
                    ..Default::default()
                },
            )
            .expect("load test model");

        let metadata = model.metadata();
        assert!(metadata.n_vocab > 0);
        assert!(metadata.n_ctx_train > 0);

        let first = model
            .infer_text(
                "Reply with one short greeting.",
                &InferenceParams {
                    n_ctx: 512,
                    n_batch: 64,
                    max_tokens: 8,
                    ..Default::default()
                },
            )
            .expect("first inference");
        let second = model
            .infer_text(
                "Reply with one short word.",
                &InferenceParams {
                    n_ctx: 768,
                    n_batch: 32,
                    max_tokens: 8,
                    ..Default::default()
                },
            )
            .expect("second inference with recreated context");

        assert!(!first.contains("llama.cpp-migrating:"));
        assert!(!second.contains("llama.cpp-migrating:"));
    }
}
