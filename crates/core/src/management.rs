use crate::backend::BackendCapabilities;
use crate::control_plane::{
    CommandExecutionRequest, CommandExecutionStatus, CommandInventoryStatus,
    CoreRewriterActivationRequest, CoreRewriterActivationStatus, CoreRewriterInventoryStatus,
    CoreRewriterStatus, InferenceActivationStatus, LegacyTextPluginActivationStatus,
    ManagementHealthStatus, ModelLoadRequest, ModelLoadSplitMode, ModelLoadStatus,
    ModelLoadStrategyRequest, PluginLoadRequest, PluginLoadSourceKind, PluginLoadStatus,
    PluginRuntimeDetail, PluginRuntimeStatus, RuntimeSnapshot, TextGenerationRequest,
    TextGenerationResponse, WorkflowInventoryStatus,
};
use crate::engine::InferenceEngine;
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::{GpuSplitMode, InferenceParams};
use loci_plugin_api::CoreComponent;
use std::collections::BTreeSet;
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

    pub fn core_rewriter_inventory(&self) -> Result<Vec<CoreRewriterInventoryStatus>> {
        self.with_engine(|engine| {
            Ok(all_core_components()
                .into_iter()
                .map(|component| {
                    let mut available_plugins = engine.plugins_for_core_component(component);
                    available_plugins.sort();
                    CoreRewriterInventoryStatus {
                        component,
                        active_plugin_name: engine
                            .active_core_rewriter(component)
                            .map(str::to_string),
                        available_plugins,
                    }
                })
                .collect())
        })
    }

    pub fn backend_capabilities(&self) -> Result<Vec<BackendCapabilities>> {
        self.with_engine(|engine| Ok(engine.backend_capabilities()))
    }

    pub fn workflow_inventory(&self) -> Result<WorkflowInventoryStatus> {
        self.with_engine(|engine| Ok(engine.workflow_inventory()))
    }

    pub fn command_inventory(&self) -> Result<CommandInventoryStatus> {
        self.with_engine(|engine| Ok(engine.command_inventory()))
    }

    pub fn run_command(&self, request: CommandExecutionRequest) -> Result<CommandExecutionStatus> {
        self.with_engine(|engine| {
            let command = request.command.trim().to_string();
            if command.is_empty() {
                return Err(LociError::InvalidArgument(
                    "command must not be empty".to_string(),
                ));
            }

            let routed_plugin_name = engine
                .active_core_rewriter(CoreComponent::PluginManager)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!(
                        "no active plugin manager rewriter is configured"
                    ))
                })?
                .to_string();

            engine.run_command(&command)?;

            Ok(CommandExecutionStatus {
                status: "accepted",
                command,
                routed_plugin_name,
            })
        })
    }

    pub fn activate_inference_plugin(
        &self,
        plugin_name: &str,
    ) -> Result<InferenceActivationStatus> {
        let status = self.activate_core_rewriter(CoreRewriterActivationRequest {
            component: CoreComponent::Inference,
            plugin_name: plugin_name.to_string(),
        })?;

        Ok(InferenceActivationStatus {
            status: status.status,
            component: status.component,
            plugin_name: status.plugin_name,
            active_inference: status.active_inference,
        })
    }

    pub fn activate_core_rewriter(
        &self,
        request: CoreRewriterActivationRequest,
    ) -> Result<CoreRewriterActivationStatus> {
        self.with_engine_mut(|engine| {
            let component = request.component;
            let plugin_name = request.plugin_name;

            match component {
                CoreComponent::Inference => engine.activate_inference_plugin(&plugin_name)?,
                _ => engine.activate_core_rewriter(component, &plugin_name)?,
            }

            let snapshot = engine.runtime_snapshot();

            Ok(CoreRewriterActivationStatus {
                status: "activated",
                component,
                plugin_name,
                active_inference: snapshot.active_inference,
                configured_core_rewriters: snapshot.configured_core_rewriters,
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

    pub fn load_plugins(&self, request: PluginLoadRequest) -> Result<PluginLoadStatus> {
        self.with_engine_mut(|engine| {
            let before_names = engine.plugin_names();
            let path = request.path;
            let source_kind = request.source_kind;

            let loaded_count = match source_kind {
                PluginLoadSourceKind::BundleFile => {
                    engine.load_plugin_bundle_file(&path)?;
                    1
                }
                PluginLoadSourceKind::Directory => engine.load_plugins_from_dir(&path)?,
            };

            let loaded_plugin_names =
                newly_loaded_plugin_names(&before_names, &engine.plugin_names());

            Ok(PluginLoadStatus {
                status: "loaded",
                path,
                source_kind,
                loaded_count,
                loaded_plugin_names,
                plugin_count_after: engine.plugin_count(),
                active_inference: engine.runtime_snapshot().active_inference,
            })
        })
    }

    pub fn generate_text(&self, request: TextGenerationRequest) -> Result<TextGenerationResponse> {
        self.with_engine_mut(|engine| {
            if request.prompt.trim().is_empty() {
                return Err(LociError::InvalidArgument(
                    "prompt must not be empty".to_string(),
                ));
            }

            let params = inference_params_from_request(request.params);
            let output = engine.generate(&request.prompt, &params)?;
            let snapshot = engine.runtime_snapshot();

            Ok(TextGenerationResponse {
                output,
                active_backend: snapshot.active_backend,
                active_model_path: snapshot.active_model_path,
                active_model_info: snapshot.active_model_info,
                active_inference: snapshot.active_inference,
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

fn newly_loaded_plugin_names(before: &[String], after: &[String]) -> Vec<String> {
    let before_set = before.iter().map(String::as_str).collect::<BTreeSet<_>>();
    after
        .iter()
        .filter(|name| !before_set.contains(name.as_str()))
        .cloned()
        .collect()
}

fn all_core_components() -> [CoreComponent; 7] {
    [
        CoreComponent::Inference,
        CoreComponent::Model,
        CoreComponent::Hardware,
        CoreComponent::Workflow,
        CoreComponent::EventBus,
        CoreComponent::PluginManager,
        CoreComponent::UiHost,
    ]
}

fn inference_params_from_request(
    request: crate::control_plane::TextGenerationParams,
) -> InferenceParams {
    InferenceParams {
        n_ctx: request.n_ctx,
        n_batch: request.n_batch,
        n_threads: request.n_threads,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        min_p: request.min_p,
        top_k: request.top_k,
        repeat_penalty: request.repeat_penalty,
    }
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
    use std::fs;
    use std::path::PathBuf;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-management-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        dir
    }

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
            runtime: PluginRuntime {
                library_path: Some("runtime/plugin.dll".to_string()),
                wasm_path: None,
                sampling_profile: Some("sampling-hook.toml".to_string()),
            },
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility {
                legacy_runtime_path: Some("legacy/compat.dll".to_string()),
                ..Default::default()
            },
        };
        manifest.contributes.inference_hooks = vec!["sampling-profile".to_string()];
        manifest.contributes.workflows = vec!["agent.pipeline".to_string()];
        manifest.contributes.custom_nodes = vec!["node.rewrite".to_string()];
        manifest.contributes.commands = vec!["plugins.reload".to_string()];
        manifest.contributes.ui_contributes.panels = vec!["inspector".to_string()];
        manifest.contributes.ui_contributes.windows = vec!["governance".to_string()];
        manifest.contributes.ui_contributes.widgets = vec!["status-pill".to_string()];
        engine
            .register_plugin(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        ManagementService::new(engine)
    }

    fn workflow_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "workflow-override".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: crate::ContributionPoints {
                    workflows: vec!["agent.plan".to_string(), "agent.review".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
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

    fn combined_rewriter_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
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
            }))
            .expect("register inference plugin");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "workflow-override".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: Default::default(),
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register workflow plugin");
        ManagementService::new(engine)
    }

    fn plugin_manager_service() -> ManagementService {
        let mut engine = InferenceEngine::builder().build().expect("build engine");
        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "command-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: crate::ContributionPoints {
                    commands: vec!["plugins.reload".to_string(), "plugins.audit".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    plugin_manager: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register plugin manager plugin");
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
        assert_eq!(snapshot.available_backends.len(), 1);
        assert_eq!(snapshot.available_backends[0].name, "mock");
        assert_eq!(snapshot.active_model_path, None);
        assert_eq!(snapshot.active_model_info, None);
        assert!(snapshot.configured_core_rewriters.is_empty());
    }

    #[test]
    fn service_lists_available_backends() {
        let service = empty_service();
        let backends = service.backend_capabilities().expect("backends");
        assert_eq!(backends.len(), 1);
        assert_eq!(backends[0].name, "mock");
        assert!(backends[0].supports_text);
    }

    #[test]
    fn service_reports_core_rewriter_inventory() {
        let service = combined_rewriter_service();
        assert_eq!(
            service.core_rewriter_inventory().expect("inventory"),
            vec![
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Inference,
                    active_plugin_name: None,
                    available_plugins: vec!["managed-inference".to_string()],
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Model,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Hardware,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Workflow,
                    active_plugin_name: None,
                    available_plugins: vec!["workflow-override".to_string()],
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::EventBus,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::PluginManager,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::UiHost,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
            ]
        );
    }

    #[test]
    fn service_reports_workflow_inventory() {
        let service = workflow_service();
        assert_eq!(
            service.workflow_inventory().expect("workflow inventory"),
            WorkflowInventoryStatus {
                active_workflow_rewriter: None,
                workflows: Vec::new(),
            }
        );

        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            })
            .expect("activate workflow");

        assert_eq!(
            service.workflow_inventory().expect("workflow inventory"),
            WorkflowInventoryStatus {
                active_workflow_rewriter: Some("workflow-override".to_string()),
                workflows: vec!["agent.plan".to_string(), "agent.review".to_string()],
            }
        );
    }

    #[test]
    fn service_reports_command_inventory_from_active_plugin_manager() {
        let service = plugin_manager_service();
        assert_eq!(
            service.command_inventory().expect("command inventory"),
            CommandInventoryStatus {
                active_plugin_manager: None,
                commands: Vec::new(),
            }
        );

        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::PluginManager,
                plugin_name: "command-router".to_string(),
            })
            .expect("activate plugin manager");

        assert_eq!(
            service.command_inventory().expect("command inventory"),
            CommandInventoryStatus {
                active_plugin_manager: Some("command-router".to_string()),
                commands: vec!["plugins.reload".to_string(), "plugins.audit".to_string()],
            }
        );
    }

    #[test]
    fn service_routes_commands_only_through_active_plugin_manager() {
        let service = plugin_manager_service();

        let missing_activation = service
            .run_command(CommandExecutionRequest {
                command: "plugins.reload".to_string(),
            })
            .expect_err("command should require active plugin manager");
        assert!(missing_activation
            .to_string()
            .contains("no active plugin manager rewriter"));

        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::PluginManager,
                plugin_name: "command-router".to_string(),
            })
            .expect("activate plugin manager");

        let status = service
            .run_command(CommandExecutionRequest {
                command: "plugins.reload".to_string(),
            })
            .expect("run declared command");
        assert_eq!(status.status, "accepted");
        assert_eq!(status.command, "plugins.reload");
        assert_eq!(status.routed_plugin_name, "command-router");

        let undeclared = service
            .run_command(CommandExecutionRequest {
                command: "plugins.missing".to_string(),
            })
            .expect_err("undeclared command should fail");
        assert!(undeclared
            .to_string()
            .contains("is not declared by active plugin manager"));
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
    fn service_generates_text_from_control_plane_request() {
        let service = empty_service();
        service
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

        let response = service
            .generate_text(TextGenerationRequest {
                prompt: "hello".to_string(),
                params: crate::TextGenerationParams {
                    max_tokens: 16,
                    temperature: 0.2,
                    ..Default::default()
                },
            })
            .expect("generate");

        assert!(response.output.contains("mock:hello"));
        assert!(response.output.contains("max_tokens=16"));
        assert_eq!(response.active_backend.as_deref(), Some("mock"));
        assert_eq!(response.active_model_path.as_deref(), Some("demo.gguf"));
        assert_eq!(
            response
                .active_model_info
                .as_ref()
                .map(|info| info.architecture.as_str()),
            Some("mock")
        );
        assert_eq!(response.active_inference, None);
    }

    #[test]
    fn service_loads_plugin_bundle_from_control_plane_request() {
        let service = empty_service();
        let dir = unique_temp_dir("plugin-bundle");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join("manifest.toml"),
            r#"
name = "managed-plugin"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_infra"]
"#,
        )
        .expect("write manifest");

        let status = service
            .load_plugins(PluginLoadRequest {
                path: dir
                    .join("plugin")
                    .join("manifest.toml")
                    .display()
                    .to_string(),
                source_kind: PluginLoadSourceKind::BundleFile,
            })
            .expect("load plugin");

        assert_eq!(status.status, "loaded");
        assert_eq!(status.loaded_count, 1);
        assert_eq!(
            status.loaded_plugin_names,
            vec!["managed-plugin".to_string()]
        );
        assert_eq!(status.plugin_count_after, 1);
        assert_eq!(status.active_inference, None);

        let snapshot = service.runtime_snapshot().expect("snapshot");
        assert_eq!(
            snapshot.loaded_plugin_names,
            vec!["managed-plugin".to_string()]
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn service_loads_plugin_directory_from_control_plane_request() {
        let service = empty_service();
        let dir = unique_temp_dir("plugin-dir");
        fs::create_dir_all(dir.join("plugin-a")).expect("mkdir a");
        fs::create_dir_all(dir.join("plugin-b")).expect("mkdir b");
        fs::write(
            dir.join("plugin-a").join("manifest.toml"),
            r#"
name = "plugin-a"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_infra"]
"#,
        )
        .expect("write manifest a");
        fs::write(
            dir.join("plugin-b").join("manifest.toml"),
            r#"
name = "plugin-b"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_agent"]
"#,
        )
        .expect("write manifest b");

        let status = service
            .load_plugins(PluginLoadRequest {
                path: dir.display().to_string(),
                source_kind: PluginLoadSourceKind::Directory,
            })
            .expect("load plugins");

        assert_eq!(status.status, "loaded");
        assert_eq!(status.loaded_count, 2);
        assert_eq!(
            status.loaded_plugin_names,
            vec!["plugin-a".to_string(), "plugin-b".to_string()]
        );
        assert_eq!(status.plugin_count_after, 2);

        let _ = fs::remove_dir_all(&dir);
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
    fn service_activates_generic_workflow_rewriter() {
        let service = workflow_service();
        let status = service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            })
            .expect("activate workflow");

        assert_eq!(status.status, "activated");
        assert_eq!(status.component, CoreComponent::Workflow);
        assert_eq!(status.plugin_name, "workflow-override");
        assert_eq!(status.active_inference, None);
        assert_eq!(
            status.configured_core_rewriters,
            vec![CoreRewriterStatus {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            }]
        );
        assert_eq!(
            service
                .configured_core_rewriters()
                .expect("configured rewriters"),
            vec![CoreRewriterStatus {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            }]
        );
    }

    #[test]
    fn service_generic_inference_activation_preserves_legacy_sampling_materialization() {
        let service = legacy_sampling_service();
        let activation = service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Inference,
                plugin_name: "legacy-sampler".to_string(),
            })
            .expect("activate legacy sampler through generic seam");

        assert_eq!(activation.status, "activated");
        assert_eq!(activation.component, CoreComponent::Inference);
        assert_eq!(activation.plugin_name, "legacy-sampler");
        assert_eq!(
            activation.active_inference.as_deref(),
            Some("legacy-sampler")
        );
        assert_eq!(
            activation.configured_core_rewriters,
            vec![CoreRewriterStatus {
                component: CoreComponent::Inference,
                plugin_name: "legacy-sampler".to_string(),
            }]
        );

        let detail = service
            .plugin_detail("legacy-sampler")
            .expect("detail query")
            .expect("plugin detail");
        assert!(detail.status.materialized_legacy_runtime);
        assert!(detail.status.registered_sampling_hook);
        assert!(detail.status.effective_sampling_hook);
    }

    #[test]
    fn service_generic_activation_status_reports_complete_rewriter_snapshot() {
        let service = combined_rewriter_service();
        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            })
            .expect("activate workflow");

        let activation = service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Inference,
                plugin_name: "managed-inference".to_string(),
            })
            .expect("activate inference");

        assert_eq!(
            activation.configured_core_rewriters,
            vec![
                CoreRewriterStatus {
                    component: CoreComponent::Inference,
                    plugin_name: "managed-inference".to_string(),
                },
                CoreRewriterStatus {
                    component: CoreComponent::Workflow,
                    plugin_name: "workflow-override".to_string(),
                },
            ]
        );
    }

    #[test]
    fn service_core_rewriter_inventory_reflects_activation_state() {
        let service = combined_rewriter_service();
        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Workflow,
                plugin_name: "workflow-override".to_string(),
            })
            .expect("activate workflow");
        service
            .activate_core_rewriter(CoreRewriterActivationRequest {
                component: CoreComponent::Inference,
                plugin_name: "managed-inference".to_string(),
            })
            .expect("activate inference");

        assert_eq!(
            service.core_rewriter_inventory().expect("inventory"),
            vec![
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Inference,
                    active_plugin_name: Some("managed-inference".to_string()),
                    available_plugins: vec!["managed-inference".to_string()],
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Model,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Hardware,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::Workflow,
                    active_plugin_name: Some("workflow-override".to_string()),
                    available_plugins: vec!["workflow-override".to_string()],
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::EventBus,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::PluginManager,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
                CoreRewriterInventoryStatus {
                    component: CoreComponent::UiHost,
                    active_plugin_name: None,
                    available_plugins: Vec::new(),
                },
            ]
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
        assert!(detail.status.declares_host_runtime);
        assert!(!detail.status.registered_host_runtime);
        assert!(!detail.status.materialized_host_runtime);
        assert_eq!(detail.status.host_runtime_kind, None);
        assert!(!detail.status.active_inference_rewriter);
        assert_eq!(detail.inference_hooks, vec!["sampling-profile".to_string()]);
        assert_eq!(detail.workflows, vec!["agent.pipeline".to_string()]);
        assert_eq!(detail.custom_nodes, vec!["node.rewrite".to_string()]);
        assert_eq!(detail.commands, vec!["plugins.reload".to_string()]);
        assert_eq!(
            detail.runtime_artifacts.library_path.as_deref(),
            Some("runtime/plugin.dll")
        );
        assert_eq!(detail.runtime_artifacts.wasm_path, None);
        assert_eq!(
            detail.runtime_artifacts.sampling_profile.as_deref(),
            Some("sampling-hook.toml")
        );
        assert_eq!(
            detail.runtime_artifacts.legacy_runtime_path.as_deref(),
            Some("legacy/compat.dll")
        );
        assert!(detail.runtime_artifacts.host_runtimes.is_empty());
        assert!(detail.runtime_artifacts.materialized_host_runtime.is_none());
        assert_eq!(detail.ui.panels, vec!["inspector".to_string()]);
        assert_eq!(detail.ui.windows, vec!["governance".to_string()]);
        assert_eq!(detail.ui.widgets, vec!["status-pill".to_string()]);
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
