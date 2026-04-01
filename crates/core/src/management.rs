use crate::control_plane::{
    CoreRewriterStatus, InferenceActivationStatus, LegacyTextPluginActivationStatus,
    ManagementHealthStatus, ModelLoadRequest, ModelLoadSplitMode, ModelLoadStatus,
    ModelLoadStrategyRequest, PluginRuntimeDetail, PluginRuntimeStatus, RuntimeSnapshot,
};
use crate::engine::InferenceEngine;
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::GpuSplitMode;
use loci_plugin_api::CoreComponent;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ManagementService {
    engine: Arc<Mutex<InferenceEngine>>,
}

impl ManagementService {
    pub fn new(engine: InferenceEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    pub fn from_shared_engine(engine: Arc<Mutex<InferenceEngine>>) -> Self {
        Self { engine }
    }

    pub fn shared_engine(&self) -> Arc<Mutex<InferenceEngine>> {
        Arc::clone(&self.engine)
    }

    pub fn health(&self) -> ManagementHealthStatus {
        ManagementHealthStatus { status: "ok" }
    }

    pub fn runtime_snapshot(&self) -> Result<RuntimeSnapshot> {
        self.with_engine(|engine| Ok(engine.runtime_snapshot()))
    }

    pub fn plugin_statuses(&self) -> Result<Vec<PluginRuntimeStatus>> {
        self.runtime_snapshot().map(|snapshot| snapshot.plugins)
    }

    pub fn plugin_detail(&self, plugin_name: &str) -> Result<Option<PluginRuntimeDetail>> {
        self.with_engine(|engine| Ok(engine.plugin_runtime_detail(plugin_name)))
    }

    pub fn configured_core_rewriters(&self) -> Result<Vec<CoreRewriterStatus>> {
        self.runtime_snapshot()
            .map(|snapshot| snapshot.configured_core_rewriters)
    }

    pub fn activate_inference_plugin(
        &self,
        plugin_name: &str,
    ) -> Result<InferenceActivationStatus> {
        self.with_engine_mut(|engine| {
            engine.activate_inference_plugin(plugin_name)?;
            Ok(InferenceActivationStatus {
                status: "activated",
                component: CoreComponent::Inference,
                plugin_name: plugin_name.to_string(),
                active_inference: engine.runtime_snapshot().active_inference,
            })
        })
    }

    pub fn load_model(&self, request: ModelLoadRequest) -> Result<ModelLoadStatus> {
        self.with_engine_mut(|engine| {
            let config = model_config_from_request(request.config)?;
            let backend_name = request.backend_name;
            let model_path = config.model_path.display().to_string();
            engine.load_model_config(&backend_name, &config)?;

            Ok(ModelLoadStatus {
                status: "loaded",
                backend_name,
                model_path,
                active_backend: engine.active_backend().map(str::to_string),
                active_model_path: engine
                    .model_path()
                    .map(|model_path| model_path.display().to_string()),
                active_model_info: engine.model_runtime_info(),
            })
        })
    }

    pub fn activate_legacy_text_plugin(
        &self,
        plugin_name: &str,
    ) -> Result<LegacyTextPluginActivationStatus> {
        self.with_engine_mut(|engine| {
            engine.activate_legacy_text_plugin(plugin_name)?;
            Ok(LegacyTextPluginActivationStatus {
                status: "activated",
                plugin_name: plugin_name.to_string(),
                active_legacy_text: engine.active_legacy_text_plugins(),
            })
        })
    }

    pub fn deactivate_legacy_text_plugin(
        &self,
        plugin_name: &str,
    ) -> Result<LegacyTextPluginActivationStatus> {
        self.with_engine_mut(|engine| {
            engine.deactivate_legacy_text_plugin(plugin_name)?;
            Ok(LegacyTextPluginActivationStatus {
                status: "deactivated",
                plugin_name: plugin_name.to_string(),
                active_legacy_text: engine.active_legacy_text_plugins(),
            })
        })
    }

    fn with_engine<T>(&self, f: impl FnOnce(&InferenceEngine) -> Result<T>) -> Result<T> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| LociError::from(anyhow::anyhow!("engine mutex poisoned")))?;
        f(&engine)
    }

    fn with_engine_mut<T>(&self, f: impl FnOnce(&mut InferenceEngine) -> Result<T>) -> Result<T> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| LociError::from(anyhow::anyhow!("engine mutex poisoned")))?;
        f(&mut engine)
    }
}

fn model_config_from_request(
    request: crate::control_plane::ModelLoadConfig,
) -> Result<ModelConfig> {
    let load_strategy = match request.load_strategy {
        ModelLoadStrategyRequest::Strict => ModelLoadStrategy::Strict,
        ModelLoadStrategyRequest::AutoReduceGpuLayers { step } => {
            ModelLoadStrategy::AutoReduceGpuLayers { step }
        }
    };

    let config = ModelConfig {
        model_path: request.model_path.into(),
        n_ctx: request.n_ctx,
        n_threads: request.n_threads,
        n_batch: request.n_batch,
        use_gpu: request.use_gpu,
        n_gpu_layers: request.n_gpu_layers,
        use_mmap: request.use_mmap,
        use_mlock: request.use_mlock,
        kv_offload: request.kv_offload,
        op_offload: request.op_offload,
        split_mode: match request.split_mode {
            ModelLoadSplitMode::None => GpuSplitMode::None,
            ModelLoadSplitMode::Layer => GpuSplitMode::Layer,
            ModelLoadSplitMode::Row => GpuSplitMode::Row,
        },
        main_gpu: request.main_gpu,
        tensor_split: request.tensor_split,
        load_strategy,
    };
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendParams;
    use crate::plugin::registered_legacy_text_plugin_for_tests;
    use crate::{
        CoreRewriters, InferenceEngine, PlatformTrack, PluginBootstrap, PluginCompatibility,
        PluginManifest, PluginRuntime, RegisteredPlugin,
    };

    fn empty_service() -> ManagementService {
        ManagementService::new(InferenceEngine::builder().build().expect("build engine"))
    }

    fn inference_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        let mut manifest = PluginManifest {
            name: "managed-inference".to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            min_host_version: None,
            max_host_version: None,
            target_tracks: vec![PlatformTrack::AiInfra],
            contributes: Default::default(),
            core_rewriters: CoreRewriters {
                inference: true,
                ..Default::default()
            },
            runtime: PluginRuntime::default(),
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility::default(),
        };
        manifest.contributes.inference_hooks = vec!["sampling-profile".to_string()];
        engine
            .register_plugin(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        ManagementService::new(engine)
    }

    fn legacy_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-text",
                &["pre_generate", "post_generate"],
            ))
            .expect("register legacy text plugin");
        ManagementService::new(engine)
    }

    fn legacy_sampling_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-sampler",
                &["transform_logits", "post_sample"],
            ))
            .expect("register legacy sampling plugin");
        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");
        ManagementService::new(engine)
    }

    #[test]
    fn service_reports_health() {
        let service = empty_service();
        assert_eq!(service.health().status, "ok");
    }

    #[test]
    fn service_reports_runtime_snapshot() {
        let service = empty_service();
        let snapshot = service.runtime_snapshot().expect("snapshot");
        assert_eq!(snapshot.plugin_count, 0);
        assert_eq!(snapshot.active_model_path, None);
        assert_eq!(snapshot.active_model_info, None);
        assert!(snapshot.configured_core_rewriters.is_empty());
    }

    #[test]
    fn service_loads_model_from_control_plane_request() {
        let service = empty_service();
        let status = service
            .load_model(ModelLoadRequest {
                backend_name: "mock".to_string(),
                config: crate::ModelLoadConfig {
                    model_path: "demo.gguf".to_string(),
                    use_gpu: false,
                    n_gpu_layers: 0,
                    kv_offload: false,
                    op_offload: false,
                    split_mode: crate::ModelLoadSplitMode::None,
                    ..Default::default()
                },
            })
            .expect("load model");

        assert_eq!(status.status, "loaded");
        assert_eq!(status.backend_name, "mock");
        assert_eq!(status.model_path, "demo.gguf");
        assert_eq!(status.active_backend.as_deref(), Some("mock"));
        assert_eq!(status.active_model_path.as_deref(), Some("demo.gguf"));
        assert_eq!(
            status
                .active_model_info
                .as_ref()
                .map(|info| info.architecture.as_str()),
            Some("mock")
        );

        let snapshot = service.runtime_snapshot().expect("snapshot");
        assert_eq!(snapshot.active_backend.as_deref(), Some("mock"));
        assert_eq!(snapshot.active_model_path.as_deref(), Some("demo.gguf"));
        assert_eq!(
            snapshot
                .active_model_info
                .as_ref()
                .map(|info| info.architecture.as_str()),
            Some("mock")
        );
    }

    #[test]
    fn service_activates_inference_plugin() {
        let service = inference_service();
        let status = service
            .activate_inference_plugin("managed-inference")
            .expect("activate inference");

        assert_eq!(status.status, "activated");
        assert_eq!(status.component, CoreComponent::Inference);
        assert_eq!(
            status.active_inference.as_deref(),
            Some("managed-inference")
        );
        assert_eq!(
            service
                .configured_core_rewriters()
                .expect("configured rewriters"),
            vec![CoreRewriterStatus {
                component: CoreComponent::Inference,
                plugin_name: "managed-inference".to_string(),
            }]
        );
    }

    #[test]
    fn service_reports_plugin_detail() {
        let service = inference_service();
        let detail = service
            .plugin_detail("managed-inference")
            .expect("detail query")
            .expect("plugin detail");

        assert_eq!(detail.status.name, "managed-inference");
        assert_eq!(
            detail.status.sampling_hook_source,
            crate::SamplingHookSource::None
        );
        assert!(!detail.status.declares_sampling_hook);
        assert!(!detail.status.registered_sampling_hook);
        assert!(!detail.status.effective_sampling_hook);
        assert!(!detail.status.active_inference_rewriter);
        assert_eq!(detail.inference_hooks, vec!["sampling-profile".to_string()]);
        assert_eq!(
            detail.declared_core_rewriters,
            vec![CoreComponent::Inference]
        );
    }

    #[test]
    fn service_activates_and_deactivates_legacy_text_plugin() {
        let service = legacy_service();
        let active = service
            .activate_legacy_text_plugin("legacy-text")
            .expect("activate legacy text");
        assert_eq!(active.status, "activated");
        assert_eq!(active.active_legacy_text, vec!["legacy-text".to_string()]);

        let inactive = service
            .deactivate_legacy_text_plugin("legacy-text")
            .expect("deactivate legacy text");
        assert_eq!(inactive.status, "deactivated");
        assert!(inactive.active_legacy_text.is_empty());
    }

    #[test]
    fn service_reports_legacy_sampling_materialization_after_activation() {
        let service = legacy_sampling_service();
        let before = service
            .plugin_detail("legacy-sampler")
            .expect("detail query")
            .expect("plugin detail");
        assert!(before.status.declares_sampling_hook);
        assert_eq!(
            before.status.sampling_hook_source,
            crate::SamplingHookSource::LegacyCompat
        );
        assert!(!before.status.registered_sampling_hook);
        assert!(!before.status.effective_sampling_hook);
        assert!(!before.status.materialized_legacy_runtime);
        assert!(!before.status.active_inference_rewriter);
        assert!(!before.status.has_sampling_hook);
        assert!(before.active_core_rewriters.is_empty());

        let activation = service
            .activate_inference_plugin("legacy-sampler")
            .expect("activate legacy sampler");
        assert_eq!(
            activation.active_inference.as_deref(),
            Some("legacy-sampler")
        );

        let after = service
            .plugin_detail("legacy-sampler")
            .expect("detail query")
            .expect("plugin detail");
        assert!(after.status.declares_sampling_hook);
        assert_eq!(
            after.status.sampling_hook_source,
            crate::SamplingHookSource::LegacyCompat
        );
        assert!(after.status.registered_sampling_hook);
        assert!(after.status.effective_sampling_hook);
        assert!(after.status.materialized_legacy_runtime);
        assert!(after.status.active_inference_rewriter);
        assert!(after.status.has_sampling_hook);
        assert_eq!(after.active_core_rewriters, vec![CoreComponent::Inference]);
    }
}
