use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, InferenceParams, Model, ModelMetadata,
};
use crate::control_plane::{
    CoreRewriterStatus, ModelRuntimeInfo, PluginRuntimeArtifacts, PluginRuntimeDetail,
    PluginRuntimeStatus, PluginUiContributionStatus, RuntimeSnapshot, SamplingHookSource,
};
use crate::core::CoreRegistry;
use crate::engine::types::{GenerationParams, ModelInfo};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::{
    discover_plugin_bundle_files, legacy_sampling_hook_from_compat, load_plugin_bundle_file,
    load_plugin_manifest_file, RegisteredPlugin, SamplingHook,
};
use loci_legacy_plugin_compat::{load_legacy_text_plugin_compat, LegacyTextCompat};
use loci_plugin_api::{CoreComponent, PlatformTrack};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct InferenceEngine {
    pub(crate) registry: Box<dyn CoreRegistry>,
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
    pub(crate) active_legacy_text_plugins: BTreeSet<String>,
    pub(crate) legacy_text_plugin_runtimes: BTreeMap<String, Arc<dyn LegacyTextCompat>>,
}

impl InferenceEngine {
    pub fn builder() -> crate::engine::InferenceEngineBuilder {
        crate::engine::InferenceEngineBuilder::new()
    }

    pub fn register_plugin(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        let plugin_name = plugin.manifest.name.clone();
        let auto_activate = plugin.auto_activate_components().to_vec();
        for component in &auto_activate {
            if !plugin.declares_core_rewriter(*component) {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` requests auto activation for `{component:?}` without declaring the core rewriter capability"
                )));
            }

            if let Some(active_plugin_name) = self.registry.active_core_rewriter(*component) {
                if active_plugin_name != plugin_name {
                    return Err(LociError::from(anyhow::anyhow!(
                        "plugin `{plugin_name}` requests auto activation for `{component:?}`, but `{active_plugin_name}` is already active; explicit activation is required"
                    )));
                }
            }
        }

        self.registry
            .plugin_manager_mut()
            .register(plugin)
            .map_err(LociError::from)?;

        for component in auto_activate {
            match component {
                CoreComponent::Inference => self.activate_inference_plugin(&plugin_name)?,
                _ => self.activate_core_rewriter(component, &plugin_name)?,
            }
        }

        self.refresh_model_sampling_runtime()
    }

    pub fn register_sampling_hook(
        &mut self,
        plugin_name: &str,
        hook: Arc<dyn SamplingHook>,
    ) -> Result<()> {
        self.registry
            .plugin_manager_mut()
            .register_sampling_hook(plugin_name, hook)
            .map_err(LociError::from)?;
        self.refresh_model_sampling_runtime()
    }

    pub fn run_command(&self, command: &str) -> Result<String> {
        self.registry.event_bus().publish(command)?;
        Ok(format!("command accepted: {command}"))
    }

    pub fn plugin_count(&self) -> usize {
        self.registry.plugin_manager().list().len()
    }

    pub fn sampling_hook_count(&self) -> usize {
        self.registry
            .plugin_manager()
            .sampling_runtime_for_inference(
                self.registry.active_core_rewriter(CoreComponent::Inference),
            )
            .hook_count()
    }

    pub fn plugin_names(&self) -> Vec<String> {
        self.registry
            .plugin_manager()
            .list()
            .iter()
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn plugins_for_track(&self, track: PlatformTrack) -> Vec<String> {
        self.registry
            .plugin_manager()
            .plugins_for_track(track)
            .into_iter()
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn plugins_for_model_provider(&self, provider: &str) -> Vec<String> {
        self.registry
            .plugin_manager()
            .plugins_for_model_provider(provider)
            .into_iter()
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn plugins_for_core_component(&self, component: CoreComponent) -> Vec<String> {
        self.registry
            .plugin_manager()
            .plugins_for_core_component(component)
            .into_iter()
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn activate_core_rewriter(
        &mut self,
        component: CoreComponent,
        plugin_name: &str,
    ) -> Result<()> {
        self.registry
            .activate_core_rewriter(component, plugin_name)
            .map_err(LociError::from)?;
        self.refresh_model_sampling_runtime()
    }

    pub fn active_core_rewriter(&self, component: CoreComponent) -> Option<&str> {
        self.registry.active_core_rewriter(component)
    }

    pub fn activate_inference_plugin(&mut self, plugin_name: &str) -> Result<()> {
        self.ensure_legacy_sampling_hook(plugin_name)?;
        self.activate_core_rewriter(CoreComponent::Inference, plugin_name)
    }

    pub fn activate_legacy_text_plugin(&mut self, plugin_name: &str) -> Result<()> {
        {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
                })?;

            if !plugin.is_legacy_compat_bundle() {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` is not a legacy compatibility bundle"
                )));
            }

            if plugin.declares_legacy_capability("on_token") {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` declares legacy `on_token`, but streaming compat is not implemented"
                )));
            }

            if !plugin.supports_legacy_pre_generate() && !plugin.supports_legacy_post_generate() {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` does not expose supported legacy pre/post text hooks"
                )));
            }
        }

        if self.active_legacy_text_plugins.contains(plugin_name) {
            return Ok(());
        }

        let runtime = self.materialize_legacy_text_runtime(plugin_name)?;

        self.legacy_text_plugin_runtimes
            .insert(plugin_name.to_string(), runtime);
        self.active_legacy_text_plugins
            .insert(plugin_name.to_string());
        Ok(())
    }

    pub fn deactivate_legacy_text_plugin(&mut self, plugin_name: &str) -> Result<()> {
        let existed = self.active_legacy_text_plugins.remove(plugin_name);
        if existed {
            return Ok(());
        }

        Err(LociError::from(anyhow::anyhow!(
            "legacy text plugin not active: {plugin_name}"
        )))
    }

    pub fn active_legacy_text_plugins(&self) -> Vec<String> {
        self.registry
            .plugin_manager()
            .list()
            .iter()
            .filter(|plugin| {
                self.active_legacy_text_plugins
                    .contains(&plugin.manifest.name)
            })
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn legacy_text_plugin_candidates(&self) -> Vec<String> {
        self.registry
            .plugin_manager()
            .list()
            .iter()
            .filter(|plugin| {
                plugin.is_legacy_compat_bundle()
                    && !plugin.declares_legacy_capability("on_token")
                    && (plugin.supports_legacy_pre_generate()
                        || plugin.supports_legacy_post_generate())
            })
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn plugin_runtime_detail(&self, plugin_name: &str) -> Option<PluginRuntimeDetail> {
        let plugin = self.registry.plugin_manager().get(plugin_name)?;
        let status = self.plugin_runtime_status(plugin);
        let active_core_rewriters = self
            .registry
            .configured_core_rewriters()
            .into_iter()
            .filter_map(|(component, active_plugin_name)| {
                if active_plugin_name == plugin.manifest.name {
                    Some(component)
                } else {
                    None
                }
            })
            .collect();

        Some(PluginRuntimeDetail {
            status,
            declared_core_rewriters: plugin.manifest.core_rewriters.declared_components(),
            auto_activate_components: plugin.auto_activate_components().to_vec(),
            active_core_rewriters,
            runtime_artifacts: PluginRuntimeArtifacts {
                library_path: plugin.manifest.runtime.library_path.clone(),
                wasm_path: plugin.manifest.runtime.wasm_path.clone(),
                sampling_profile: plugin.manifest.runtime.sampling_profile.clone(),
                legacy_runtime_path: plugin.manifest.compatibility.legacy_runtime_path.clone(),
            },
            model_providers: plugin.manifest.contributes.model_providers.clone(),
            inference_hooks: plugin.manifest.contributes.inference_hooks.clone(),
            workflows: plugin.manifest.contributes.workflows.clone(),
            custom_nodes: plugin.manifest.contributes.custom_nodes.clone(),
            commands: plugin.manifest.contributes.commands.clone(),
            ui: PluginUiContributionStatus {
                panels: plugin.manifest.contributes.ui_contributes.panels.clone(),
                windows: plugin.manifest.contributes.ui_contributes.windows.clone(),
                widgets: plugin.manifest.contributes.ui_contributes.widgets.clone(),
            },
            legacy_capabilities: plugin.manifest.compatibility.legacy_capabilities.clone(),
        })
    }

    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        let active_inference = self
            .active_core_rewriter(CoreComponent::Inference)
            .map(str::to_string);
        let available_backends = self.backend_capabilities();
        let active_backend = self.active_backend().map(str::to_string);
        let active_model_path = self
            .model_path()
            .map(|model_path| model_path.display().to_string());
        let active_model_info = self.model_runtime_info();
        let loaded_plugin_names = self.plugin_names();
        let configured_core_rewriters = self
            .registry
            .configured_core_rewriters()
            .into_iter()
            .map(|(component, plugin_name)| CoreRewriterStatus {
                component,
                plugin_name,
            })
            .collect();
        let legacy_text_candidates = self.legacy_text_plugin_candidates();
        let active_legacy_text = self.active_legacy_text_plugins();
        let plugins = self
            .registry
            .plugin_manager()
            .list()
            .iter()
            .map(|plugin| self.plugin_runtime_status(plugin))
            .collect();

        RuntimeSnapshot {
            plugin_count: loaded_plugin_names.len(),
            loaded_plugin_names,
            available_backends,
            active_backend,
            active_model_path,
            active_model_info,
            active_inference,
            configured_core_rewriters,
            legacy_text_candidates,
            active_legacy_text,
            plugins,
        }
    }

    pub fn load_plugin_manifest_file<P: AsRef<Path>>(&mut self, manifest_path: P) -> Result<()> {
        let plugin = load_plugin_manifest_file(manifest_path).map_err(LociError::from)?;
        self.register_plugin(plugin)
    }

    pub fn load_plugin_bundle_file<P: AsRef<Path>>(&mut self, bundle_path: P) -> Result<()> {
        let plugin = load_plugin_bundle_file(bundle_path).map_err(LociError::from)?;
        self.register_plugin(plugin)
    }

    pub fn load_plugins_from_dir<P: AsRef<Path>>(&mut self, plugin_dir: P) -> Result<usize> {
        let manifests = discover_plugin_bundle_files(plugin_dir).map_err(LociError::from)?;
        let mut loaded = 0usize;
        for manifest in manifests {
            self.load_plugin_bundle_file(&manifest)?;
            loaded += 1;
        }
        Ok(loaded)
    }

    pub fn load_model<P: AsRef<Path>>(
        &mut self,
        backend_name: &str,
        model_path: P,
        backend_params: BackendParams,
    ) -> Result<()> {
        let model_path = model_path.as_ref().to_path_buf();
        let model = self
            .backend_registry
            .load_model(backend_name, &model_path, backend_params)?;
        self.active_backend = Some(backend_name.to_string());
        self.model = Some(model);
        self.model_path = Some(model_path);
        self.refresh_model_sampling_runtime()?;
        Ok(())
    }

    pub fn load_model_config(&mut self, backend_name: &str, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        let backend_params = config.to_backend_params();

        let result = self.load_model(backend_name, &config.model_path, backend_params.clone());
        match (result, config.load_strategy) {
            (Ok(()), _) => Ok(()),
            (Err(_), ModelLoadStrategy::AutoReduceGpuLayers { step })
                if config.use_gpu && config.n_gpu_layers > 0 =>
            {
                let mut retry = config.n_gpu_layers.saturating_sub(step as i32);
                while retry >= 0 {
                    let mut reduced = backend_params.clone();
                    reduced.n_gpu_layers = retry;
                    if self
                        .load_model(backend_name, &config.model_path, reduced)
                        .is_ok()
                    {
                        return Ok(());
                    }
                    if retry == 0 {
                        break;
                    }
                    retry = retry.saturating_sub(step as i32);
                }
                Err(LociError::ModelLoadError(
                    "model load failed after GPU fallback attempts".to_string(),
                ))
            }
            (Err(err), _) => Err(err),
        }
    }

    pub fn generate(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let prompt = self.apply_legacy_pre_generate(prompt)?;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        let response = model.infer_text(&prompt, params)?;
        self.apply_legacy_post_generate(&response)
    }

    pub fn generate_legacy(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = self.generation_params_to_inference(params);
        self.generate(prompt, &inference_params)
    }

    fn ensure_legacy_sampling_hook(&mut self, plugin_name: &str) -> Result<()> {
        let requires_legacy_sampling = {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
                })?;

            plugin.is_legacy_compat_bundle() && plugin.supports_legacy_sampling()
        };

        if !requires_legacy_sampling {
            return Ok(());
        }

        let hook_registered = self
            .registry
            .plugin_manager()
            .sampling_runtime_for_inference(Some(plugin_name))
            .hook_count()
            > 0;
        if hook_registered {
            return Ok(());
        }

        let compat = self.materialize_legacy_text_runtime(plugin_name)?;
        self.register_sampling_hook(plugin_name, legacy_sampling_hook_from_compat(compat))
    }

    fn materialize_legacy_text_runtime(
        &mut self,
        plugin_name: &str,
    ) -> Result<Arc<dyn LegacyTextCompat>> {
        if let Some(runtime) = self.legacy_text_plugin_runtimes.get(plugin_name) {
            return Ok(Arc::clone(runtime));
        }

        let (manifest_name, manifest_version, legacy_runtime_path, legacy_capabilities, runtime) = {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
                })?;

            if !plugin.is_legacy_compat_bundle() {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` is not a legacy compatibility bundle"
                )));
            }

            (
                plugin.manifest.name.clone(),
                plugin.manifest.version.clone(),
                plugin.legacy_runtime_path().map(str::to_string),
                plugin.manifest.compatibility.legacy_capabilities.clone(),
                plugin.legacy_text_compat_runtime(),
            )
        };

        let runtime = if let Some(runtime) = runtime {
            runtime
        } else {
            let runtime_path = legacy_runtime_path.ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` does not declare a legacy runtime path"
                ))
            })?;
            load_legacy_text_plugin_compat(
                Path::new(&runtime_path),
                &manifest_name,
                &manifest_version,
                &legacy_capabilities,
            )
            .map_err(LociError::from)?
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` does not provide a supported legacy compat runtime"
                ))
            })?
        };

        self.legacy_text_plugin_runtimes
            .insert(plugin_name.to_string(), Arc::clone(&runtime));
        Ok(runtime)
    }

    fn refresh_model_sampling_runtime(&mut self) -> Result<()> {
        if let Some(model) = self.model.as_mut() {
            let runtime = self
                .registry
                .plugin_manager()
                .sampling_runtime_for_inference(
                    self.registry.active_core_rewriter(CoreComponent::Inference),
                );
            model.attach_sampling_runtime(runtime)?;
        }
        Ok(())
    }

    fn apply_legacy_pre_generate(&self, prompt: &str) -> Result<String> {
        let mut prompt = prompt.to_string();
        for plugin in self.registry.plugin_manager().list() {
            if !self
                .active_legacy_text_plugins
                .contains(&plugin.manifest.name)
                || !plugin.supports_legacy_pre_generate()
            {
                continue;
            }

            let runtime = self
                .legacy_text_plugin_runtimes
                .get(&plugin.manifest.name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!(
                        "legacy text runtime missing for active plugin `{}`",
                        plugin.manifest.name
                    ))
                })?;
            prompt = runtime.pre_generate(&prompt).map_err(LociError::from)?;
        }
        Ok(prompt)
    }

    fn apply_legacy_post_generate(&self, response: &str) -> Result<String> {
        let mut response = response.to_string();
        for plugin in self.registry.plugin_manager().list() {
            if !self
                .active_legacy_text_plugins
                .contains(&plugin.manifest.name)
                || !plugin.supports_legacy_post_generate()
            {
                continue;
            }

            let runtime = self
                .legacy_text_plugin_runtimes
                .get(&plugin.manifest.name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!(
                        "legacy text runtime missing for active plugin `{}`",
                        plugin.manifest.name
                    ))
                })?;
            response = runtime.post_generate(&response).map_err(LociError::from)?;
        }
        Ok(response)
    }

    fn generation_params_to_inference(&self, params: GenerationParams) -> InferenceParams {
        InferenceParams {
            n_ctx: self.default_inference_params.n_ctx,
            n_batch: self.default_inference_params.n_batch,
            n_threads: self.default_inference_params.n_threads,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            min_p: params.min_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
        }
    }

    pub fn active_backend(&self) -> Option<&str> {
        self.active_backend.as_deref()
    }

    pub fn backend_capabilities(&self) -> Vec<BackendCapabilities> {
        self.backend_registry.list()
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    pub fn model_metadata(&self) -> Option<ModelMetadata> {
        self.model.as_ref().map(|model| model.metadata())
    }

    pub fn model_runtime_info(&self) -> Option<ModelRuntimeInfo> {
        self.model_metadata().map(|metadata| ModelRuntimeInfo {
            architecture: metadata.architecture,
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
            n_layer: metadata.n_layer,
            param_count: metadata.param_count,
        })
    }

    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model_metadata().map(|metadata| ModelInfo {
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
        })
    }

    fn plugin_runtime_status(&self, plugin: &RegisteredPlugin) -> PluginRuntimeStatus {
        let active_inference_rewriter = self.active_core_rewriter(CoreComponent::Inference)
            == Some(plugin.manifest.name.as_str());
        let registered_sampling_hook = self
            .registry
            .plugin_manager()
            .sampling_runtime_for_inference(Some(&plugin.manifest.name))
            .hook_count()
            > 0;
        let declares_sampling_hook = plugin.has_sampling_hook()
            || (plugin.is_legacy_compat_bundle() && plugin.supports_legacy_sampling());
        let sampling_hook_source =
            if plugin.is_legacy_compat_bundle() && plugin.supports_legacy_sampling() {
                SamplingHookSource::LegacyCompat
            } else if plugin.has_sampling_hook() {
                SamplingHookSource::NativeRuntime
            } else if registered_sampling_hook {
                SamplingHookSource::DynamicRegistration
            } else {
                SamplingHookSource::None
            };

        PluginRuntimeStatus {
            name: plugin.manifest.name.clone(),
            version: plugin.manifest.version.clone(),
            supports_ai_infra: plugin.supports_track(PlatformTrack::AiInfra),
            supports_ai_agent: plugin.supports_track(PlatformTrack::AiAgent),
            source_format: plugin.manifest.compatibility.source_format,
            runtime_bridge: plugin.manifest.compatibility.runtime_bridge,
            declares_inference_rewriter: plugin.declares_core_rewriter(CoreComponent::Inference),
            declares_sampling_hook,
            sampling_hook_source,
            registered_sampling_hook,
            effective_sampling_hook: active_inference_rewriter && registered_sampling_hook,
            materialized_legacy_runtime: self
                .legacy_text_plugin_runtimes
                .contains_key(&plugin.manifest.name),
            active_inference_rewriter,
            has_sampling_hook: registered_sampling_hook,
            is_legacy_compat: plugin.is_legacy_compat_bundle(),
            legacy_text_candidate: self
                .legacy_text_plugin_candidates()
                .iter()
                .any(|candidate| candidate == &plugin.manifest.name),
            active_legacy_text: self
                .active_legacy_text_plugins()
                .iter()
                .any(|active| active == &plugin.manifest.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DefaultCoreRegistry;
    use crate::plugin::registered_legacy_text_plugin_for_tests;
    use crate::sampler::LogitsView;
    use loci_plugin_api::{
        ContributionPoints, CoreRewriters, PluginBootstrap, PluginCompatibility, PluginManifest,
        PluginRuntime,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::sync::Arc;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!("loci-engine-test-{name}-{nanos}"));
        path
    }

    #[test]
    fn engine_exposes_plugin_indexes_and_core_rewriter_activation() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "agent-workflow".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: ContributionPoints {
                    model_providers: vec!["rag-local".to_string()],
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
            .expect("register");

        assert_eq!(engine.plugin_names(), vec!["agent-workflow".to_string()]);
        assert_eq!(
            engine.plugins_for_track(PlatformTrack::AiAgent),
            vec!["agent-workflow".to_string()]
        );
        assert_eq!(
            engine.plugins_for_model_provider("rag-local"),
            vec!["agent-workflow".to_string()]
        );
        assert_eq!(
            engine.plugins_for_core_component(CoreComponent::Workflow),
            vec!["agent-workflow".to_string()]
        );
        assert_eq!(
            engine.workflow_inventory(),
            WorkflowInventoryStatus {
                active_workflow_rewriter: None,
                workflows: Vec::new(),
            }
        );

        engine
            .activate_core_rewriter(CoreComponent::Workflow, "agent-workflow")
            .expect("activate");
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Workflow),
            Some("agent-workflow")
        );
        assert_eq!(
            engine.workflow_inventory(),
            WorkflowInventoryStatus {
                active_workflow_rewriter: Some("agent-workflow".to_string()),
                workflows: vec!["agent.plan".to_string(), "agent.review".to_string()],
            }
        );
    }

    #[test]
    fn engine_loads_plugins_from_manifest_directory() {
        let dir = unique_temp_dir("plugins");
        fs::create_dir_all(dir.join("agent-plugin")).expect("mkdir");
        fs::write(
            dir.join("agent-plugin").join("manifest.toml"),
            r#"
name = "agent-plugin"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_agent"]

[contributes]
model_providers = ["rag-local"]
workflows = ["agent.plan", "agent.review"]

[core_rewriters]
workflow = true
"#,
        )
        .expect("write manifest");

        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(engine.plugin_count(), 1);
        assert_eq!(
            engine.plugins_for_track(PlatformTrack::AiAgent),
            vec!["agent-plugin".to_string()]
        );
        assert_eq!(
            engine.workflow_inventory(),
            WorkflowInventoryStatus {
                active_workflow_rewriter: None,
                workflows: Vec::new(),
            }
        );

        engine
            .activate_core_rewriter(CoreComponent::Workflow, "agent-plugin")
            .expect("activate workflow rewriter");
        assert_eq!(
            engine.workflow_inventory(),
            WorkflowInventoryStatus {
                active_workflow_rewriter: Some("agent-plugin".to_string()),
                workflows: vec!["agent.plan".to_string(), "agent.review".to_string()],
            }
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_loads_plugin_runtime_bundle_and_auto_activates_inference() {
        let dir = unique_temp_dir("runtime-bundle");
        fs::create_dir_all(dir.join("inference-plugin")).expect("mkdir");
        fs::write(
            dir.join("inference-plugin").join("manifest.toml"),
            r#"
name = "inference-plugin"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_infra"]

[contributes]
inference_hooks = ["sampling-profile"]

[core_rewriters]
inference = true

[runtime]
sampling_profile = "sampling-hook.toml"

[bootstrap]
activate_on_load = ["inference"]
"#,
        )
        .expect("write manifest");
        fs::write(
            dir.join("inference-plugin").join("sampling-hook.toml"),
            r#"
post_sample_override = 9

[[logit_biases]]
token_id = 4
logit = 42.0
"#,
        )
        .expect("write profile");

        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Inference),
            Some("inference-plugin")
        );
        assert_eq!(engine.sampling_hook_count(), 1);

        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");
        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("hooks=1"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_loads_legacy_plugin_bundle_as_compat_metadata() {
        let dir = unique_temp_dir("legacy-bundle");
        fs::create_dir_all(dir.join("legacy-plugin")).expect("mkdir");
        fs::write(dir.join("legacy-plugin").join("rot13.dll"), b"binary").expect("write runtime");
        fs::write(
            dir.join("legacy-plugin").join("rot13.loci-plugin.json"),
            r#"{
  "name": "rot13_dynamic",
  "version": "1.0.0",
  "kind": "text_plugin",
  "abi_version": 1,
  "capabilities": ["post_generate"]
}"#,
        )
        .expect("write contract");

        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(engine.plugin_names(), vec!["rot13_dynamic".to_string()]);
        assert_eq!(engine.sampling_hook_count(), 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_loads_legacy_sampling_bundle_as_metadata_until_activation() {
        let dir = unique_temp_dir("legacy-sampling-bundle");
        fs::create_dir_all(dir.join("legacy-sampler")).expect("mkdir");
        fs::write(dir.join("legacy-sampler").join("sampler.dll"), b"binary")
            .expect("write runtime");
        fs::write(
            dir.join("legacy-sampler").join("sampler.loci-plugin.json"),
            r#"{
  "name": "legacy_sampler",
  "version": "1.0.0",
  "kind": "text_plugin",
  "abi_version": 1,
  "capabilities": ["transform_logits", "post_sample"]
}"#,
        )
        .expect("write contract");

        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(engine.plugin_names(), vec!["legacy_sampler".to_string()]);
        assert_eq!(engine.sampling_hook_count(), 0);
        let status = &engine.runtime_snapshot().plugins[0];
        assert!(status.declares_sampling_hook);
        assert_eq!(
            status.sampling_hook_source,
            SamplingHookSource::LegacyCompat
        );
        assert!(!status.registered_sampling_hook);
        assert!(!status.effective_sampling_hook);
        assert!(!status.materialized_legacy_runtime);
        assert!(!status.active_inference_rewriter);
        assert!(!status.has_sampling_hook);

        let err = engine
            .activate_inference_plugin("legacy_sampler")
            .expect_err("activation should materialize runtime and fail on invalid binary");
        assert!(err
            .to_string()
            .contains("failed to load legacy plugin library"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_legacy_text_plugin_requires_explicit_activation() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-text",
                &["pre_generate", "post_generate"],
            ))
            .expect("register legacy text plugin");
        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");

        let inactive_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate without activation");
        assert!(inactive_output.starts_with("mock:hello"));
        assert_eq!(engine.active_legacy_text_plugins(), Vec::<String>::new());

        engine
            .activate_legacy_text_plugin("legacy-text")
            .expect("activate legacy text plugin");
        assert_eq!(
            engine.active_legacy_text_plugins(),
            vec!["legacy-text".to_string()]
        );

        let active_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate with activation");
        assert!(active_output.starts_with("mock:[legacy-text:pre]hello"));
        assert!(active_output.ends_with("[legacy-text:post]"));

        engine
            .deactivate_legacy_text_plugin("legacy-text")
            .expect("deactivate legacy text plugin");
        let deactivated_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate after deactivation");
        assert!(deactivated_output.starts_with("mock:hello"));
    }

    #[test]
    fn engine_rejects_legacy_text_activation_when_on_token_is_declared() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-stream",
                &["pre_generate", "on_token"],
            ))
            .expect("register legacy stream plugin");

        let err = engine
            .activate_legacy_text_plugin("legacy-stream")
            .expect_err("activation should fail");
        assert!(err
            .to_string()
            .contains("streaming compat is not implemented"));
    }

    #[test]
    fn engine_lists_only_supported_legacy_text_activation_candidates() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-prepost",
                &["pre_generate", "post_generate"],
            ))
            .expect("register pre/post legacy plugin");
        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-stream",
                &["pre_generate", "on_token"],
            ))
            .expect("register on_token legacy plugin");

        assert_eq!(
            engine.legacy_text_plugin_candidates(),
            vec!["legacy-prepost".to_string()]
        );
    }

    #[test]
    fn engine_runtime_snapshot_reports_plugin_and_activation_state() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-prepost",
                &["pre_generate", "post_generate"],
            ))
            .expect("register legacy plugin");
        engine
            .activate_legacy_text_plugin("legacy-prepost")
            .expect("activate legacy plugin");

        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.plugin_count, 1);
        assert_eq!(
            snapshot.loaded_plugin_names,
            vec!["legacy-prepost".to_string()]
        );
        assert_eq!(
            snapshot.legacy_text_candidates,
            vec!["legacy-prepost".to_string()]
        );
        assert_eq!(
            snapshot.active_legacy_text,
            vec!["legacy-prepost".to_string()]
        );
        assert_eq!(snapshot.plugins.len(), 1);
        assert_eq!(
            snapshot.configured_core_rewriters,
            Vec::<CoreRewriterStatus>::new()
        );
        assert!(snapshot.plugins[0].is_legacy_compat);
        assert!(snapshot.plugins[0].legacy_text_candidate);
        assert!(snapshot.plugins[0].active_legacy_text);
        assert!(snapshot.plugins[0].materialized_legacy_runtime);
        assert!(!snapshot.plugins[0].declares_inference_rewriter);
        assert!(!snapshot.plugins[0].declares_sampling_hook);
        assert_eq!(
            snapshot.plugins[0].sampling_hook_source,
            SamplingHookSource::None
        );
        assert!(!snapshot.plugins[0].registered_sampling_hook);
        assert!(!snapshot.plugins[0].effective_sampling_hook);
        assert!(!snapshot.plugins[0].active_inference_rewriter);
    }

    #[test]
    fn engine_plugin_runtime_detail_reports_governance_metadata() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "governed-inference".to_string(),
                version: "1.2.3".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    model_providers: vec!["private-registry".to_string()],
                    inference_hooks: vec!["sampling-profile".to_string()],
                    workflows: vec!["agent.pipeline".to_string()],
                    custom_nodes: vec!["node.rewrite".to_string()],
                    commands: vec!["plugins.reload".to_string()],
                    ui_contributes: loci_plugin_api::UiContributionPoints {
                        panels: vec!["inspector".to_string()],
                        windows: vec!["governance".to_string()],
                        widgets: vec!["status-pill".to_string()],
                    },
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    inference: true,
                    workflow: true,
                    ..Default::default()
                },
                runtime: PluginRuntime {
                    library_path: Some("runtime/plugin.dll".to_string()),
                    wasm_path: Some("runtime/plugin.wasm".to_string()),
                    sampling_profile: Some("sampling-hook.toml".to_string()),
                },
                bootstrap: PluginBootstrap {
                    activate_on_load: vec![CoreComponent::Inference],
                },
                compatibility: PluginCompatibility {
                    legacy_runtime_path: Some("legacy/compat.dll".to_string()),
                    ..Default::default()
                },
            }))
            .expect("register plugin");

        let detail = engine
            .plugin_runtime_detail("governed-inference")
            .expect("detail should exist");

        assert_eq!(detail.status.name, "governed-inference");
        assert!(!detail.status.declares_sampling_hook);
        assert_eq!(detail.status.sampling_hook_source, SamplingHookSource::None);
        assert!(!detail.status.registered_sampling_hook);
        assert!(!detail.status.effective_sampling_hook);
        assert!(detail.status.active_inference_rewriter);
        assert_eq!(
            detail.declared_core_rewriters,
            vec![CoreComponent::Inference, CoreComponent::Workflow]
        );
        assert_eq!(
            detail.auto_activate_components,
            vec![CoreComponent::Inference]
        );
        assert_eq!(detail.active_core_rewriters, vec![CoreComponent::Inference]);
        assert_eq!(
            detail.runtime_artifacts.library_path.as_deref(),
            Some("runtime/plugin.dll")
        );
        assert_eq!(
            detail.runtime_artifacts.wasm_path.as_deref(),
            Some("runtime/plugin.wasm")
        );
        assert_eq!(
            detail.runtime_artifacts.sampling_profile.as_deref(),
            Some("sampling-hook.toml")
        );
        assert_eq!(
            detail.runtime_artifacts.legacy_runtime_path.as_deref(),
            Some("legacy/compat.dll")
        );
        assert_eq!(detail.model_providers, vec!["private-registry".to_string()]);
        assert_eq!(detail.inference_hooks, vec!["sampling-profile".to_string()]);
        assert_eq!(detail.workflows, vec!["agent.pipeline".to_string()]);
        assert_eq!(detail.custom_nodes, vec!["node.rewrite".to_string()]);
        assert_eq!(detail.commands, vec!["plugins.reload".to_string()]);
        assert_eq!(detail.ui.panels, vec!["inspector".to_string()]);
        assert_eq!(detail.ui.windows, vec!["governance".to_string()]);
        assert_eq!(detail.ui.widgets, vec!["status-pill".to_string()]);
    }

    struct ForceTokenHook;

    impl SamplingHook for ForceTokenHook {
        fn transform_logits(
            &self,
            logits: &mut LogitsView<'_>,
            _context_tokens: &[i32],
        ) -> crate::Result<()> {
            logits.set_usize(0, 123.0)?;
            Ok(())
        }
    }

    #[test]
    fn engine_only_activates_sampling_runtime_after_inference_rewriter_activation() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "hooked-plugin".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters {
                    inference: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");
        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");
        engine
            .register_sampling_hook("hooked-plugin", Arc::new(ForceTokenHook))
            .expect("register hook");

        let inactive_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(inactive_output.contains("hooks=0"));
        assert_eq!(engine.sampling_hook_count(), 0);
        let before = &engine.runtime_snapshot().plugins[0];
        assert_eq!(
            before.sampling_hook_source,
            SamplingHookSource::DynamicRegistration
        );
        assert!(before.registered_sampling_hook);
        assert!(before.has_sampling_hook);
        assert!(!before.effective_sampling_hook);
        assert!(!before.active_inference_rewriter);

        engine
            .activate_inference_plugin("hooked-plugin")
            .expect("activate inference rewriter");

        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("hooks=1"));
        assert_eq!(engine.sampling_hook_count(), 1);
        assert_eq!(
            engine.runtime_snapshot().configured_core_rewriters,
            vec![CoreRewriterStatus {
                component: CoreComponent::Inference,
                plugin_name: "hooked-plugin".to_string(),
            }]
        );
        let after = &engine.runtime_snapshot().plugins[0];
        assert!(!after.declares_sampling_hook);
        assert_eq!(
            after.sampling_hook_source,
            SamplingHookSource::DynamicRegistration
        );
        assert!(after.registered_sampling_hook);
        assert!(after.effective_sampling_hook);
        assert!(after.active_inference_rewriter);
    }

    #[test]
    fn engine_materializes_legacy_sampling_runtime_on_inference_activation() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-sampler",
                &["transform_logits", "post_sample"],
            ))
            .expect("register legacy sampler");
        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");

        let before = engine.runtime_snapshot();
        assert!(before.plugins[0].declares_sampling_hook);
        assert_eq!(
            before.plugins[0].sampling_hook_source,
            SamplingHookSource::LegacyCompat
        );
        assert!(!before.plugins[0].registered_sampling_hook);
        assert!(!before.plugins[0].effective_sampling_hook);
        assert!(!before.plugins[0].materialized_legacy_runtime);
        assert!(!before.plugins[0].active_inference_rewriter);
        assert!(!before.plugins[0].has_sampling_hook);
        assert_eq!(engine.sampling_hook_count(), 0);

        let inactive_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate without activation");
        assert!(inactive_output.contains("hooks=0"));

        engine
            .activate_inference_plugin("legacy-sampler")
            .expect("activate legacy sampler");

        let after = engine.runtime_snapshot();
        assert!(after.plugins[0].declares_sampling_hook);
        assert_eq!(
            after.plugins[0].sampling_hook_source,
            SamplingHookSource::LegacyCompat
        );
        assert!(after.plugins[0].registered_sampling_hook);
        assert!(after.plugins[0].effective_sampling_hook);
        assert!(after.plugins[0].materialized_legacy_runtime);
        assert!(after.plugins[0].active_inference_rewriter);
        assert!(after.plugins[0].has_sampling_hook);
        assert_eq!(
            after.configured_core_rewriters,
            vec![CoreRewriterStatus {
                component: CoreComponent::Inference,
                plugin_name: "legacy-sampler".to_string(),
            }]
        );

        let active_output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate with activation");
        assert!(active_output.contains("hooks=1"));
        assert_eq!(engine.sampling_hook_count(), 1);
        assert!(engine.active_legacy_text_plugins().is_empty());
    }

    #[test]
    fn engine_auto_activation_uses_legacy_inference_materialization_path() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let mut plugin = registered_legacy_text_plugin_for_tests(
            "legacy-auto-sampler",
            &["transform_logits", "post_sample"],
        );
        plugin.manifest.bootstrap.activate_on_load = vec![CoreComponent::Inference];

        engine.register_plugin(plugin).expect("register plugin");
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Inference),
            Some("legacy-auto-sampler")
        );

        let snapshot = engine.runtime_snapshot();
        assert_eq!(
            snapshot.configured_core_rewriters,
            vec![CoreRewriterStatus {
                component: CoreComponent::Inference,
                plugin_name: "legacy-auto-sampler".to_string(),
            }]
        );
        assert!(snapshot.plugins[0].declares_sampling_hook);
        assert_eq!(
            snapshot.plugins[0].sampling_hook_source,
            SamplingHookSource::LegacyCompat
        );
        assert!(snapshot.plugins[0].registered_sampling_hook);
        assert!(snapshot.plugins[0].effective_sampling_hook);
        assert!(snapshot.plugins[0].materialized_legacy_runtime);
        assert!(snapshot.plugins[0].active_inference_rewriter);
        assert!(snapshot.plugins[0].has_sampling_hook);

        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");
        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("hooks=1"));
    }

    #[test]
    fn engine_rejects_auto_activation_for_undeclared_component() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        let err = engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "broken-bootstrap".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters::default(),
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap {
                    activate_on_load: vec![CoreComponent::Inference],
                },
                compatibility: PluginCompatibility::default(),
            }))
            .expect_err("register should fail");

        assert!(err
            .to_string()
            .contains("requests auto activation for `Inference`"));
    }

    #[test]
    fn engine_rejects_conflicting_auto_activation_for_same_component() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            active_legacy_text_plugins: BTreeSet::new(),
            legacy_text_plugin_runtimes: BTreeMap::new(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "first-inference".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters {
                    inference: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap {
                    activate_on_load: vec![CoreComponent::Inference],
                },
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register first");

        let err = engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "second-inference".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters {
                    inference: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap {
                    activate_on_load: vec![CoreComponent::Inference],
                },
                compatibility: PluginCompatibility::default(),
            }))
            .expect_err("second register should fail");

        assert!(err.to_string().contains("explicit activation is required"));
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Inference),
            Some("first-inference")
        );
        assert_eq!(engine.plugin_names(), vec!["first-inference".to_string()]);
    }
}
