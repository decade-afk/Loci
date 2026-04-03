use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, InferenceParams, Model, ModelMetadata,
};
use crate::control_plane::{
    CommandInventoryStatus, CoreRewriterStatus, EventInventoryStatus, ModelRuntimeInfo,
    PluginCompatibilityDetail, PluginCompatibilityStatus, PluginHostRuntimeKind,
    PluginHostRuntimeMaterialization, PluginHostRuntimeRegistration, PluginRuntimeArtifacts,
    PluginRuntimeDetail, PluginRuntimeStatus, PluginUiContributionStatus,
    RuntimeCompatibilitySnapshot, RuntimeSnapshot, SamplingHookSource, UiInventoryStatus,
    UiSurfaceKind, WorkflowInventoryStatus,
};
use crate::core::CoreRegistry;
use crate::engine::types::{GenerationParams, ModelInfo};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::{
    discover_plugin_bundle_files, legacy_sampling_hook_from_compat, load_plugin_bundle_file,
    load_plugin_manifest_file, LegacyCapability, RegisteredHostRuntimeKind, RegisteredPlugin,
    SamplingHook,
};
use loci_legacy_plugin_compat::{load_legacy_text_plugin_compat, LegacyTextCompat};
use loci_plugin_api::{CoreComponent, PlatformTrack};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializedHostRuntime {
    kind: RegisteredHostRuntimeKind,
    resolved_path: PathBuf,
    file_size_bytes: u64,
}

#[derive(Debug, Clone)]
struct GovernedModelLoad {
    model_path: PathBuf,
    backend_params: BackendParams,
}

#[derive(Default)]
struct LegacyTextRuntimeRegistry {
    active_plugins: BTreeSet<String>,
    runtimes: BTreeMap<String, Arc<dyn LegacyTextCompat>>,
}

impl LegacyTextRuntimeRegistry {
    fn is_active(&self, plugin_name: &str) -> bool {
        self.active_plugins.contains(plugin_name)
    }

    fn activate(&mut self, plugin_name: &str, runtime: Arc<dyn LegacyTextCompat>) -> bool {
        self.runtimes.insert(plugin_name.to_string(), runtime);
        self.active_plugins.insert(plugin_name.to_string())
    }

    fn deactivate(&mut self, plugin_name: &str) -> bool {
        self.active_plugins.remove(plugin_name)
    }

    fn runtime(&self, plugin_name: &str) -> Option<Arc<dyn LegacyTextCompat>> {
        self.runtimes.get(plugin_name).map(Arc::clone)
    }

    fn materialize(
        &mut self,
        plugin_name: &str,
        runtime: Arc<dyn LegacyTextCompat>,
    ) -> Arc<dyn LegacyTextCompat> {
        let runtime = self
            .runtimes
            .entry(plugin_name.to_string())
            .or_insert(runtime);
        Arc::clone(runtime)
    }

    fn has_materialized_runtime(&self, plugin_name: &str) -> bool {
        self.runtimes.contains_key(plugin_name)
    }

    fn active_plugin_names(&self, plugins: &[RegisteredPlugin]) -> Vec<String> {
        plugins
            .iter()
            .filter(|plugin| self.is_active(&plugin.manifest.name))
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }
}

#[derive(Default)]
pub(crate) struct LegacyTextCompatCoordinator {
    runtime_registry: LegacyTextRuntimeRegistry,
}

impl LegacyTextCompatCoordinator {
    fn activate_plugin(&mut self, plugin_name: &str, plugin: &RegisteredPlugin) -> Result<()> {
        self.validate_text_activation(plugin_name, plugin)?;
        if self.runtime_registry.is_active(plugin_name) {
            return Ok(());
        }

        let runtime = self.materialize_runtime(plugin_name, plugin)?;
        self.runtime_registry.activate(plugin_name, runtime);
        Ok(())
    }

    fn deactivate_plugin(&mut self, plugin_name: &str) -> bool {
        self.runtime_registry.deactivate(plugin_name)
    }

    fn active_plugin_names(&self, plugins: &[RegisteredPlugin]) -> Vec<String> {
        self.runtime_registry.active_plugin_names(plugins)
    }

    fn candidate_plugin_names(&self, plugins: &[RegisteredPlugin]) -> Vec<String> {
        plugins
            .iter()
            .filter(|plugin| {
                plugin.is_legacy_compat_bundle()
                    && !plugin.declares_legacy_capability(LegacyCapability::OnToken)
                    && (plugin.supports_legacy_pre_generate()
                        || plugin.supports_legacy_post_generate())
            })
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    fn ensure_sampling_hook(
        &mut self,
        plugin_name: &str,
        plugin: &RegisteredPlugin,
        hook_registered: bool,
    ) -> Result<Option<Arc<dyn SamplingHook>>> {
        if !plugin.is_legacy_compat_bundle() || !plugin.supports_legacy_sampling() {
            return Ok(None);
        }
        if hook_registered {
            return Ok(None);
        }

        let compat = self.materialize_runtime(plugin_name, plugin)?;
        Ok(Some(legacy_sampling_hook_from_compat(compat)))
    }

    fn has_materialized_runtime(&self, plugin_name: &str) -> bool {
        self.runtime_registry.has_materialized_runtime(plugin_name)
    }

    fn apply_pre_generate(&self, prompt: &str, plugins: &[RegisteredPlugin]) -> Result<String> {
        let mut prompt = prompt.to_string();
        for plugin in plugins {
            if !self.runtime_registry.is_active(&plugin.manifest.name)
                || !plugin.supports_legacy_pre_generate()
            {
                continue;
            }

            let runtime = self
                .runtime_registry
                .runtime(&plugin.manifest.name)
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

    fn apply_post_generate(&self, response: &str, plugins: &[RegisteredPlugin]) -> Result<String> {
        let mut response = response.to_string();
        for plugin in plugins {
            if !self.runtime_registry.is_active(&plugin.manifest.name)
                || !plugin.supports_legacy_post_generate()
            {
                continue;
            }

            let runtime = self
                .runtime_registry
                .runtime(&plugin.manifest.name)
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

    fn validate_text_activation(&self, plugin_name: &str, plugin: &RegisteredPlugin) -> Result<()> {
        if !plugin.is_legacy_compat_bundle() {
            return Err(LociError::from(anyhow::anyhow!(
                "plugin `{plugin_name}` is not a legacy compatibility bundle"
            )));
        }

        if plugin.declares_legacy_capability(LegacyCapability::OnToken) {
            return Err(LociError::from(anyhow::anyhow!(
                "plugin `{plugin_name}` declares legacy `on_token`, but streaming compat is not implemented"
            )));
        }

        if !plugin.supports_legacy_pre_generate() && !plugin.supports_legacy_post_generate() {
            return Err(LociError::from(anyhow::anyhow!(
                "plugin `{plugin_name}` does not expose supported legacy pre/post text hooks"
            )));
        }

        Ok(())
    }

    fn materialize_runtime(
        &mut self,
        plugin_name: &str,
        plugin: &RegisteredPlugin,
    ) -> Result<Arc<dyn LegacyTextCompat>> {
        if let Some(runtime) = self.runtime_registry.runtime(plugin_name) {
            return Ok(runtime);
        }

        if !plugin.is_legacy_compat_bundle() {
            return Err(LociError::from(anyhow::anyhow!(
                "plugin `{plugin_name}` is not a legacy compatibility bundle"
            )));
        }

        let runtime = if let Some(runtime) = plugin.legacy_text_compat_runtime() {
            runtime
        } else {
            let runtime_path = plugin.legacy_runtime_path().ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` does not declare a legacy runtime path"
                ))
            })?;
            load_legacy_text_plugin_compat(
                Path::new(runtime_path),
                &plugin.manifest.name,
                &plugin.manifest.version,
                &plugin.manifest.compatibility.legacy_capabilities,
            )
            .map_err(LociError::from)?
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` does not provide a supported legacy compat runtime"
                ))
            })?
        };

        Ok(self.runtime_registry.materialize(plugin_name, runtime))
    }
}

pub struct InferenceEngine {
    pub(crate) registry: Box<dyn CoreRegistry>,
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
    pub(crate) host_plugin_runtimes: BTreeMap<String, MaterializedHostRuntime>,
    pub(crate) legacy_text_runtime: LegacyTextCompatCoordinator,
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
        let command = command.trim();
        if command.is_empty() {
            return Err(LociError::InvalidArgument(
                "command must not be empty".to_string(),
            ));
        }

        let plugin_name = self
            .active_core_rewriter(CoreComponent::PluginManager)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "no active plugin manager rewriter is configured"
                ))
            })?;
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "active plugin manager rewriter `{plugin_name}` is not registered"
                ))
            })?;

        if !plugin
            .manifest
            .contributes
            .commands
            .iter()
            .any(|candidate| candidate == command)
        {
            return Err(LociError::from(anyhow::anyhow!(
                "command `{command}` is not declared by active plugin manager `{plugin_name}`"
            )));
        }

        self.registry
            .event_bus()
            .publish(&format!("plugin_manager/{plugin_name}/{command}"))?;
        Ok(format!("command accepted by {plugin_name}: {command}"))
    }

    pub fn publish_event(&self, event: &str) -> Result<String> {
        let event = event.trim();
        if event.is_empty() {
            return Err(LociError::InvalidArgument(
                "event must not be empty".to_string(),
            ));
        }

        let plugin_name = self
            .active_core_rewriter(CoreComponent::EventBus)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "no active event bus rewriter is configured"
                ))
            })?;
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "active event bus rewriter `{plugin_name}` is not registered"
                ))
            })?;

        if !plugin
            .manifest
            .contributes
            .events
            .iter()
            .any(|candidate| candidate == event)
        {
            return Err(LociError::from(anyhow::anyhow!(
                "event `{event}` is not declared by active event bus rewriter `{plugin_name}`"
            )));
        }

        self.registry
            .event_bus()
            .publish(&format!("event_bus/{plugin_name}/{event}"))?;
        Ok(format!("event published by {plugin_name}: {event}"))
    }

    pub fn command_inventory(&self) -> CommandInventoryStatus {
        let active_plugin_manager = self
            .active_core_rewriter(CoreComponent::PluginManager)
            .map(str::to_string);
        let commands = active_plugin_manager
            .as_deref()
            .and_then(|plugin_name| self.registry.plugin_manager().get(plugin_name))
            .map(|plugin| plugin.manifest.contributes.commands.clone())
            .unwrap_or_default();

        CommandInventoryStatus {
            active_plugin_manager,
            commands,
        }
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
        self.ensure_host_runtime_materialized(plugin_name)?;
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
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .cloned()
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
            })?;
        self.legacy_text_runtime
            .activate_plugin(plugin_name, &plugin)
    }

    pub fn deactivate_legacy_text_plugin(&mut self, plugin_name: &str) -> Result<()> {
        let existed = self.legacy_text_runtime.deactivate_plugin(plugin_name);
        if existed {
            return Ok(());
        }

        Err(LociError::from(anyhow::anyhow!(
            "legacy text plugin not active: {plugin_name}"
        )))
    }

    pub fn active_legacy_text_plugins(&self) -> Vec<String> {
        self.legacy_text_runtime
            .active_plugin_names(self.registry.plugin_manager().list())
    }

    pub fn legacy_text_plugin_candidates(&self) -> Vec<String> {
        self.legacy_text_runtime
            .candidate_plugin_names(self.registry.plugin_manager().list())
    }

    pub fn plugin_runtime_detail(&self, plugin_name: &str) -> Option<PluginRuntimeDetail> {
        let plugin = self.registry.plugin_manager().get(plugin_name)?;
        let status = self.plugin_runtime_status(plugin);
        let compat_status = status.compat.clone();
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
                host_runtimes: plugin
                    .registered_host_runtimes()
                    .iter()
                    .map(|runtime| PluginHostRuntimeRegistration {
                        kind: match runtime.kind() {
                            RegisteredHostRuntimeKind::DynamicLibrary => {
                                PluginHostRuntimeKind::DynamicLibrary
                            }
                            RegisteredHostRuntimeKind::WasmModule => {
                                PluginHostRuntimeKind::WasmModule
                            }
                        },
                        declared_path: runtime.declared_path().to_string(),
                        resolved_path: runtime.resolved_path().display().to_string(),
                    })
                    .collect(),
                materialized_host_runtime: self.host_plugin_runtimes.get(plugin_name).map(
                    |runtime| PluginHostRuntimeMaterialization {
                        kind: host_runtime_kind_to_control_plane(runtime.kind),
                        resolved_path: runtime.resolved_path.display().to_string(),
                        file_size_bytes: runtime.file_size_bytes,
                    },
                ),
            },
            model_providers: plugin.manifest.contributes.model_providers.clone(),
            accelerators: plugin.manifest.contributes.accelerators.clone(),
            inference_hooks: plugin.manifest.contributes.inference_hooks.clone(),
            events: plugin.manifest.contributes.events.clone(),
            workflows: plugin.manifest.contributes.workflows.clone(),
            custom_nodes: plugin.manifest.contributes.custom_nodes.clone(),
            commands: plugin.manifest.contributes.commands.clone(),
            ui: PluginUiContributionStatus {
                panels: plugin.manifest.contributes.ui_contributes.panels.clone(),
                windows: plugin.manifest.contributes.ui_contributes.windows.clone(),
                widgets: plugin.manifest.contributes.ui_contributes.widgets.clone(),
            },
            compat: PluginCompatibilityDetail {
                status: compat_status,
                legacy_runtime_path: plugin.manifest.compatibility.legacy_runtime_path.clone(),
                legacy_capabilities: plugin.manifest.compatibility.legacy_capabilities.clone(),
            },
        })
    }

    pub fn workflow_inventory(&self) -> WorkflowInventoryStatus {
        let active_workflow_rewriter = self
            .active_core_rewriter(CoreComponent::Workflow)
            .map(str::to_string);
        let workflows = active_workflow_rewriter
            .as_deref()
            .and_then(|plugin_name| self.registry.plugin_manager().get(plugin_name))
            .map(|plugin| plugin.manifest.contributes.workflows.clone())
            .unwrap_or_default();

        WorkflowInventoryStatus {
            active_workflow_rewriter,
            workflows,
        }
    }

    pub fn run_workflow(&self, workflow: &str) -> Result<String> {
        let workflow = workflow.trim();
        if workflow.is_empty() {
            return Err(LociError::InvalidArgument(
                "workflow must not be empty".to_string(),
            ));
        }

        let plugin_name = self
            .active_core_rewriter(CoreComponent::Workflow)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!("no active workflow rewriter is configured"))
            })?;
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "active workflow rewriter `{plugin_name}` is not registered"
                ))
            })?;

        if !plugin
            .manifest
            .contributes
            .workflows
            .iter()
            .any(|candidate| candidate == workflow)
        {
            return Err(LociError::from(anyhow::anyhow!(
                "workflow `{workflow}` is not declared by active workflow rewriter `{plugin_name}`"
            )));
        }

        self.registry
            .event_bus()
            .publish(&format!("workflow/{plugin_name}/{workflow}"))?;
        Ok(format!("workflow accepted by {plugin_name}: {workflow}"))
    }

    pub fn ui_inventory(&self) -> UiInventoryStatus {
        let active_ui_host = self
            .active_core_rewriter(CoreComponent::UiHost)
            .map(str::to_string);
        let ui = active_ui_host
            .as_deref()
            .and_then(|plugin_name| self.registry.plugin_manager().get(plugin_name))
            .map(|plugin| PluginUiContributionStatus {
                panels: plugin.manifest.contributes.ui_contributes.panels.clone(),
                windows: plugin.manifest.contributes.ui_contributes.windows.clone(),
                widgets: plugin.manifest.contributes.ui_contributes.widgets.clone(),
            })
            .unwrap_or(PluginUiContributionStatus {
                panels: Vec::new(),
                windows: Vec::new(),
                widgets: Vec::new(),
            });

        UiInventoryStatus { active_ui_host, ui }
    }

    pub fn present_ui_surface(&self, surface_kind: UiSurfaceKind, surface: &str) -> Result<String> {
        let surface = surface.trim();
        if surface.is_empty() {
            return Err(LociError::InvalidArgument(
                "surface must not be empty".to_string(),
            ));
        }

        let plugin_name = self
            .active_core_rewriter(CoreComponent::UiHost)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!("no active ui host rewriter is configured"))
            })?;
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!(
                    "active ui host rewriter `{plugin_name}` is not registered"
                ))
            })?;

        let declared = match surface_kind {
            UiSurfaceKind::Panel => &plugin.manifest.contributes.ui_contributes.panels,
            UiSurfaceKind::Window => &plugin.manifest.contributes.ui_contributes.windows,
            UiSurfaceKind::Widget => &plugin.manifest.contributes.ui_contributes.widgets,
        };
        if !declared.iter().any(|candidate| candidate == surface) {
            return Err(LociError::from(anyhow::anyhow!(
                "ui {:?} `{surface}` is not declared by active ui host `{plugin_name}`",
                surface_kind
            )));
        }

        self.registry.event_bus().publish(&format!(
            "ui_host/{plugin_name}/{}/{surface}",
            ui_surface_kind_segment(surface_kind)
        ))?;
        Ok(format!(
            "ui {:?} accepted by {plugin_name}: {surface}",
            surface_kind
        ))
    }

    pub fn event_inventory(&self) -> EventInventoryStatus {
        let active_event_bus_rewriter = self
            .active_core_rewriter(CoreComponent::EventBus)
            .map(str::to_string);
        let events = active_event_bus_rewriter
            .as_deref()
            .and_then(|plugin_name| self.registry.plugin_manager().get(plugin_name))
            .map(|plugin| plugin.manifest.contributes.events.clone())
            .unwrap_or_default();

        EventInventoryStatus {
            active_event_bus_rewriter,
            events,
            recent_events: self.registry.event_bus().recent_events(),
        }
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
        let compat = RuntimeCompatibilitySnapshot {
            text_generation_candidates: self.legacy_text_plugin_candidates(),
            active_text_generation_plugins: self.active_legacy_text_plugins(),
        };
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
            compat,
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
        let governed = self.govern_model_load(backend_name, model_path.as_ref(), backend_params)?;
        self.load_model_unchecked(backend_name, &governed.model_path, governed.backend_params)
    }

    fn load_model_unchecked(
        &mut self,
        backend_name: &str,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<()> {
        let model_path = model_path.to_path_buf();
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
        let governed =
            self.govern_model_load(backend_name, &config.model_path, config.to_backend_params())?;
        let result = self.load_model_unchecked(
            backend_name,
            &governed.model_path,
            governed.backend_params.clone(),
        );
        match (result, config.load_strategy) {
            (Ok(()), _) => Ok(()),
            (Err(_), ModelLoadStrategy::AutoReduceGpuLayers { step })
                if governed.backend_params.use_gpu && governed.backend_params.n_gpu_layers > 0 =>
            {
                let mut retry = governed
                    .backend_params
                    .n_gpu_layers
                    .saturating_sub(step as i32);
                while retry >= 0 {
                    let reduced = with_gpu_layer_retry(governed.backend_params.clone(), retry);
                    if self
                        .load_model_unchecked(backend_name, &governed.model_path, reduced)
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

    fn govern_model_load(
        &self,
        backend_name: &str,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<GovernedModelLoad> {
        let backend_capabilities = self
            .backend_registry
            .capabilities(backend_name)
            .ok_or_else(|| {
                LociError::BackendNotAvailable(format!("backend not found: {backend_name}"))
            })?;
        let model_path = model_path.to_path_buf();
        let model_ref = model_path.display().to_string();
        let model_provider = model_provider_for_reference(&model_ref);

        self.ensure_model_load_is_admitted(&model_ref, model_provider.as_str())?;
        let backend_params =
            self.normalize_backend_params_for_hardware(backend_params, &backend_capabilities)?;

        Ok(GovernedModelLoad {
            model_path,
            backend_params,
        })
    }

    fn ensure_model_load_is_admitted(&self, model_ref: &str, provider: &str) -> Result<()> {
        if let Some(plugin_name) = self.active_core_rewriter(CoreComponent::Model) {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!(
                        "active model rewriter `{plugin_name}` is not registered"
                    ))
                })?;

            if !plugin
                .manifest
                .contributes
                .model_providers
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(provider))
            {
                return Err(LociError::ModelLoadError(format!(
                    "model provider `{provider}` is not declared by active model rewriter `{plugin_name}`"
                )));
            }
        } else if provider != "local" {
            return Err(LociError::ModelLoadError(format!(
                "model provider `{provider}` requires an active model rewriter"
            )));
        }

        if provider == "local" && !self.registry.model_repository().has_model(model_ref) {
            return Err(LociError::ModelLoadError(format!(
                "model repository does not admit local model `{model_ref}`"
            )));
        }

        Ok(())
    }

    fn normalize_backend_params_for_hardware(
        &self,
        backend_params: BackendParams,
        backend_capabilities: &BackendCapabilities,
    ) -> Result<BackendParams> {
        let mut available_accelerators = normalize_accelerators(
            self.registry
                .hardware_abstraction()
                .available_accelerators(),
        );

        if let Some(plugin_name) = self.active_core_rewriter(CoreComponent::Hardware) {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!(
                        "active hardware rewriter `{plugin_name}` is not registered"
                    ))
                })?;
            let declared = normalize_accelerators(plugin.manifest.contributes.accelerators.clone());
            if declared.is_empty() {
                return Err(LociError::ModelLoadError(format!(
                    "active hardware rewriter `{plugin_name}` does not declare any accelerators"
                )));
            }

            available_accelerators.retain(|accelerator| {
                accelerator == "cpu" || declared.iter().any(|candidate| candidate == accelerator)
            });
            if !available_accelerators
                .iter()
                .any(|candidate| candidate == "cpu")
            {
                available_accelerators.push("cpu".to_string());
            }
        }

        let gpu_available = backend_capabilities.has_gpu_support
            && available_accelerators
                .iter()
                .any(|accelerator| is_gpu_accelerator(accelerator));
        if gpu_available && backend_params.use_gpu {
            Ok(backend_params)
        } else {
            Ok(cpu_only_backend_params(backend_params))
        }
    }

    pub fn generate(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let prompt = self
            .legacy_text_runtime
            .apply_pre_generate(prompt, self.registry.plugin_manager().list())?;
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        let response = model.infer_text(&prompt, params)?;
        self.legacy_text_runtime
            .apply_post_generate(&response, self.registry.plugin_manager().list())
    }

    pub fn generate_legacy(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = self.generation_params_to_inference(params);
        self.generate(prompt, &inference_params)
    }

    fn ensure_legacy_sampling_hook(&mut self, plugin_name: &str) -> Result<()> {
        let plugin = self
            .registry
            .plugin_manager()
            .get(plugin_name)
            .cloned()
            .ok_or_else(|| {
                LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
            })?;
        let hook_registered = self
            .registry
            .plugin_manager()
            .sampling_runtime_for_inference(Some(plugin_name))
            .hook_count()
            > 0;
        if let Some(hook) =
            self.legacy_text_runtime
                .ensure_sampling_hook(plugin_name, &plugin, hook_registered)?
        {
            self.register_sampling_hook(plugin_name, hook)?;
        }
        Ok(())
    }

    fn ensure_host_runtime_materialized(&mut self, plugin_name: &str) -> Result<()> {
        if self.host_plugin_runtimes.contains_key(plugin_name) {
            return Ok(());
        }

        let host_runtime = {
            let plugin = self
                .registry
                .plugin_manager()
                .get(plugin_name)
                .ok_or_else(|| {
                    LociError::from(anyhow::anyhow!("plugin not registered: {plugin_name}"))
                })?;

            plugin.registered_host_runtimes().first().cloned()
        };

        let Some(host_runtime) = host_runtime else {
            return Ok(());
        };

        let metadata = fs::metadata(host_runtime.resolved_path()).map_err(|err| {
            LociError::from(anyhow::anyhow!(
                "failed to materialize host runtime for plugin `{plugin_name}`: {}",
                err
            ))
        })?;

        if !metadata.is_file() {
            return Err(LociError::from(anyhow::anyhow!(
                "host runtime artifact for plugin `{plugin_name}` is not a file: {}",
                host_runtime.resolved_path().display()
            )));
        }

        if metadata.len() == 0 {
            return Err(LociError::from(anyhow::anyhow!(
                "host runtime artifact for plugin `{plugin_name}` is empty: {}",
                host_runtime.resolved_path().display()
            )));
        }

        if host_runtime.kind() == RegisteredHostRuntimeKind::WasmModule {
            let bytes = fs::read(host_runtime.resolved_path()).map_err(|err| {
                LociError::from(anyhow::anyhow!(
                    "failed to read wasm host runtime for plugin `{plugin_name}`: {}",
                    err
                ))
            })?;
            if bytes.len() < 4 || &bytes[..4] != b"\0asm" {
                return Err(LociError::from(anyhow::anyhow!(
                    "plugin `{plugin_name}` declares a wasm host runtime, but the artifact does not start with the wasm magic header"
                )));
            }
        }

        self.host_plugin_runtimes.insert(
            plugin_name.to_string(),
            MaterializedHostRuntime {
                kind: host_runtime.kind(),
                resolved_path: host_runtime.resolved_path().to_path_buf(),
                file_size_bytes: metadata.len(),
            },
        );
        Ok(())
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
        let declares_host_runtime = plugin.manifest.runtime.library_path.is_some()
            || plugin.manifest.runtime.wasm_path.is_some();
        let registered_host_runtime = !plugin.registered_host_runtimes().is_empty();
        let materialized_host_runtime = self
            .host_plugin_runtimes
            .contains_key(&plugin.manifest.name);
        let host_runtime_kind = plugin
            .registered_host_runtimes()
            .first()
            .map(|runtime| host_runtime_kind_to_control_plane(runtime.kind()));

        PluginRuntimeStatus {
            name: plugin.manifest.name.clone(),
            version: plugin.manifest.version.clone(),
            supports_ai_infra: plugin.supports_track(PlatformTrack::AiInfra),
            supports_ai_agent: plugin.supports_track(PlatformTrack::AiAgent),
            source_format: plugin.manifest.compatibility.source_format,
            declares_inference_rewriter: plugin.declares_core_rewriter(CoreComponent::Inference),
            declares_sampling_hook,
            sampling_hook_source,
            registered_sampling_hook,
            effective_sampling_hook: active_inference_rewriter && registered_sampling_hook,
            declares_host_runtime,
            registered_host_runtime,
            materialized_host_runtime,
            host_runtime_kind,
            active_inference_rewriter,
            has_sampling_hook: registered_sampling_hook,
            compat: PluginCompatibilityStatus {
                runtime_bridge: plugin.manifest.compatibility.runtime_bridge,
                legacy_bundle: plugin.is_legacy_compat_bundle(),
                text_generation_candidate: self
                    .legacy_text_plugin_candidates()
                    .iter()
                    .any(|candidate| candidate == &plugin.manifest.name),
                active_text_generation: self
                    .active_legacy_text_plugins()
                    .iter()
                    .any(|active| active == &plugin.manifest.name),
                materialized_runtime: self
                    .legacy_text_runtime
                    .has_materialized_runtime(&plugin.manifest.name),
            },
        }
    }
}

fn host_runtime_kind_to_control_plane(kind: RegisteredHostRuntimeKind) -> PluginHostRuntimeKind {
    match kind {
        RegisteredHostRuntimeKind::DynamicLibrary => PluginHostRuntimeKind::DynamicLibrary,
        RegisteredHostRuntimeKind::WasmModule => PluginHostRuntimeKind::WasmModule,
    }
}

fn model_provider_for_reference(model_ref: &str) -> String {
    if let Some((provider, _)) = model_ref.split_once("://") {
        return provider.to_ascii_lowercase();
    }
    if let Some((provider, _)) = model_ref.split_once("::") {
        return provider.to_ascii_lowercase();
    }
    "local".to_string()
}

fn normalize_accelerators(accelerators: Vec<String>) -> Vec<String> {
    let mut normalized = accelerators
        .into_iter()
        .map(|accelerator| accelerator.trim().to_ascii_lowercase())
        .filter(|accelerator| !accelerator.is_empty())
        .collect::<Vec<_>>();
    if !normalized.iter().any(|accelerator| accelerator == "cpu") {
        normalized.push("cpu".to_string());
    }
    normalized.sort();
    normalized.dedup();
    normalized
}

fn is_gpu_accelerator(accelerator: &str) -> bool {
    accelerator != "cpu"
}

fn cpu_only_backend_params(mut backend_params: BackendParams) -> BackendParams {
    backend_params.use_gpu = false;
    backend_params.n_gpu_layers = 0;
    backend_params.kv_offload = false;
    backend_params.op_offload = false;
    backend_params.split_mode = crate::backend::GpuSplitMode::None;
    backend_params.main_gpu = 0;
    backend_params.tensor_split = None;
    backend_params
}

fn with_gpu_layer_retry(mut backend_params: BackendParams, n_gpu_layers: i32) -> BackendParams {
    if n_gpu_layers <= 0 {
        cpu_only_backend_params(backend_params)
    } else {
        backend_params.use_gpu = true;
        backend_params.n_gpu_layers = n_gpu_layers;
        backend_params
    }
}

fn ui_surface_kind_segment(surface_kind: UiSurfaceKind) -> &'static str {
    match surface_kind {
        UiSurfaceKind::Panel => "panel",
        UiSurfaceKind::Window => "window",
        UiSurfaceKind::Widget => "widget",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{GpuSplitMode, InferenceBackend};
    use crate::core::{
        CoreRegistry, DefaultCoreRegistry, DefaultModelRepository, DefaultWorkflowEngine,
        HeadlessUiHost, PluginManager as _, RecordingEventBus,
    };
    use crate::error::{LociError, Result as LociResult};
    use crate::plugin::registered_legacy_text_plugin_for_tests;
    use crate::plugin::InMemoryPluginManager;
    use crate::sampler::LogitsView;
    use anyhow::{bail, Result as AnyhowResult};
    use loci_plugin_api::{
        ContributionPoints, CoreRewriters, PluginBootstrap, PluginCompatibility, PluginManifest,
        PluginRuntime,
    };
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!("loci-engine-test-{name}-{nanos}"));
        path
    }

    fn temp_model_path(name: &str) -> PathBuf {
        let dir = unique_temp_dir(name);
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("demo.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        path
    }

    #[derive(Debug, Clone)]
    struct RecordedModelLoad {
        model_path: PathBuf,
        backend_params: BackendParams,
    }

    struct RecordingBackend {
        calls: Arc<Mutex<Vec<RecordedModelLoad>>>,
        has_gpu_support: bool,
    }

    impl RecordingBackend {
        fn new(calls: Arc<Mutex<Vec<RecordedModelLoad>>>, has_gpu_support: bool) -> Self {
            Self {
                calls,
                has_gpu_support,
            }
        }
    }

    impl InferenceBackend for RecordingBackend {
        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                name: "recording".to_string(),
                version: "1.0.0".to_string(),
                supports_text: true,
                supports_multimodal: false,
                supports_embeddings: false,
                supports_streaming: false,
                has_gpu_support: self.has_gpu_support,
                supported_formats: vec!["gguf".to_string()],
            }
        }

        fn load_model(
            &self,
            model_path: &Path,
            backend_params: BackendParams,
        ) -> LociResult<Box<dyn Model>> {
            self.calls
                .lock()
                .expect("calls lock")
                .push(RecordedModelLoad {
                    model_path: model_path.to_path_buf(),
                    backend_params,
                });
            Ok(Box::new(RecordedModel))
        }
    }

    struct RecordedModel;

    impl Model for RecordedModel {
        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                architecture: "recording".to_string(),
                n_vocab: 0,
                n_ctx_train: 4096,
                n_embd: 0,
                n_layer: 0,
                param_count: None,
            }
        }

        fn infer_text(&mut self, prompt: &str, _params: &InferenceParams) -> LociResult<String> {
            if prompt.trim().is_empty() {
                return Err(LociError::InvalidArgument(
                    "prompt must not be empty".to_string(),
                ));
            }
            Ok(format!("recording:{prompt}"))
        }
    }

    struct FixedHardwareAbstraction {
        accelerators: Vec<String>,
    }

    impl crate::core::HardwareAbstraction for FixedHardwareAbstraction {
        fn available_accelerators(&self) -> Vec<String> {
            self.accelerators.clone()
        }
    }

    struct TestRegistry {
        model_repository: DefaultModelRepository,
        workflow_engine: DefaultWorkflowEngine,
        event_bus: RecordingEventBus,
        hardware_abstraction: FixedHardwareAbstraction,
        ui_host: HeadlessUiHost,
        plugin_manager: InMemoryPluginManager,
        active_core_rewriters: BTreeMap<CoreComponent, String>,
    }

    impl TestRegistry {
        fn with_accelerators(accelerators: &[&str]) -> Self {
            Self {
                model_repository: DefaultModelRepository,
                workflow_engine: DefaultWorkflowEngine,
                event_bus: RecordingEventBus::default(),
                hardware_abstraction: FixedHardwareAbstraction {
                    accelerators: accelerators
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect(),
                },
                ui_host: HeadlessUiHost,
                plugin_manager: InMemoryPluginManager::default(),
                active_core_rewriters: BTreeMap::new(),
            }
        }
    }

    impl CoreRegistry for TestRegistry {
        fn model_repository(&self) -> &dyn crate::core::ModelRepository {
            &self.model_repository
        }

        fn workflow_engine(&self) -> &dyn crate::core::WorkflowEngine {
            &self.workflow_engine
        }

        fn event_bus(&self) -> &dyn crate::core::EventBus {
            &self.event_bus
        }

        fn hardware_abstraction(&self) -> &dyn crate::core::HardwareAbstraction {
            &self.hardware_abstraction
        }

        fn ui_host(&self) -> &dyn crate::core::UiHost {
            &self.ui_host
        }

        fn plugin_manager(&self) -> &dyn crate::core::PluginManager {
            &self.plugin_manager
        }

        fn plugin_manager_mut(&mut self) -> &mut dyn crate::core::PluginManager {
            &mut self.plugin_manager
        }

        fn activate_core_rewriter(
            &mut self,
            component: CoreComponent,
            plugin_name: &str,
        ) -> AnyhowResult<()> {
            let plugin = self
                .plugin_manager
                .get(plugin_name)
                .ok_or_else(|| anyhow::anyhow!("plugin not registered: {plugin_name}"))?;

            if !plugin.declares_core_rewriter(component) {
                bail!(
                    "plugin `{}` does not declare core rewriter capability for `{component:?}`",
                    plugin.manifest.name
                );
            }

            self.active_core_rewriters
                .insert(component, plugin.manifest.name.clone());
            Ok(())
        }

        fn active_core_rewriter(&self, component: CoreComponent) -> Option<&str> {
            self.active_core_rewriters
                .get(&component)
                .map(String::as_str)
        }

        fn configured_core_rewriters(&self) -> Vec<(CoreComponent, String)> {
            self.active_core_rewriters
                .iter()
                .map(|(component, plugin_name)| (*component, plugin_name.clone()))
                .collect()
        }
    }

    fn recording_backend_registry(
        calls: Arc<Mutex<Vec<RecordedModelLoad>>>,
        has_gpu_support: bool,
    ) -> BackendRegistry {
        let mut registry = BackendRegistry::new();
        registry.register(
            "recording".to_string(),
            Box::new(RecordingBackend::new(calls, has_gpu_support)),
        );
        registry
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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

        let missing_activation = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        }
        .run_workflow("agent.plan")
        .expect_err("workflow should require activation");
        assert!(missing_activation
            .to_string()
            .contains("no active workflow rewriter"));

        let status = engine
            .run_workflow("agent.plan")
            .expect("run declared workflow");
        assert_eq!(status, "workflow accepted by agent-workflow: agent.plan");
        assert_eq!(
            engine.event_inventory().recent_events,
            vec!["workflow/agent-workflow/agent.plan".to_string()]
        );

        let undeclared = engine
            .run_workflow("agent.missing")
            .expect_err("undeclared workflow should fail");
        assert!(undeclared
            .to_string()
            .contains("is not declared by active workflow rewriter"));
    }

    #[test]
    fn engine_rejects_remote_model_provider_without_active_model_rewriter() {
        let calls = Arc::new(Mutex::new(Vec::<RecordedModelLoad>::new()));
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: recording_backend_registry(Arc::clone(&calls), true),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        let err = engine
            .load_model(
                "recording",
                Path::new("hf://meta/llama-3"),
                BackendParams::default(),
            )
            .expect_err("remote provider should require model rewriter");
        assert!(err
            .to_string()
            .contains("requires an active model rewriter"));
        assert!(calls.lock().expect("calls lock").is_empty());
    }

    #[test]
    fn engine_active_model_rewriter_must_admit_local_provider_before_backend_load() {
        let calls = Arc::new(Mutex::new(Vec::<RecordedModelLoad>::new()));
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: recording_backend_registry(Arc::clone(&calls), true),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "model-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    model_providers: vec!["private-registry".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    model: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");
        engine
            .activate_core_rewriter(CoreComponent::Model, "model-router")
            .expect("activate");

        let model_path = temp_model_path("model-router-local");
        let err = engine
            .load_model("recording", &model_path, BackendParams::default())
            .expect_err("local provider should be governed by active model rewriter");
        assert!(err
            .to_string()
            .contains("is not declared by active model rewriter"));
        assert!(calls.lock().expect("calls lock").is_empty());
    }

    #[test]
    fn engine_hardware_rewriter_clips_gpu_request_to_cpu_before_backend_load() {
        let calls = Arc::new(Mutex::new(Vec::<RecordedModelLoad>::new()));
        let mut engine = InferenceEngine {
            registry: Box::new(TestRegistry::with_accelerators(&["cpu", "cuda"])),
            backend_registry: recording_backend_registry(Arc::clone(&calls), true),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "hardware-policy".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    accelerators: vec!["cpu".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    hardware: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");
        engine
            .activate_core_rewriter(CoreComponent::Hardware, "hardware-policy")
            .expect("activate");

        let model_path = temp_model_path("hardware-clip");
        engine
            .load_model(
                "recording",
                &model_path,
                BackendParams {
                    use_gpu: true,
                    n_gpu_layers: 42,
                    kv_offload: true,
                    op_offload: true,
                    split_mode: GpuSplitMode::Layer,
                    main_gpu: 2,
                    tensor_split: Some(vec![0.5, 0.5]),
                    ..Default::default()
                },
            )
            .expect("load model");

        let loads = calls.lock().expect("calls lock");
        assert_eq!(loads.len(), 1);
        assert_eq!(loads[0].model_path, model_path);
        assert!(!loads[0].backend_params.use_gpu);
        assert_eq!(loads[0].backend_params.n_gpu_layers, 0);
        assert!(!loads[0].backend_params.kv_offload);
        assert!(!loads[0].backend_params.op_offload);
        assert_eq!(loads[0].backend_params.split_mode, GpuSplitMode::None);
        assert_eq!(loads[0].backend_params.main_gpu, 0);
        assert_eq!(loads[0].backend_params.tensor_split, None);
    }

    #[test]
    fn engine_reports_ui_inventory_from_active_ui_host() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "ui-shell".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: ContributionPoints {
                    ui_contributes: loci_plugin_api::UiContributionPoints {
                        panels: vec!["inspector".to_string()],
                        windows: vec!["governance".to_string()],
                        widgets: vec!["status-pill".to_string()],
                    },
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    ui_host: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");

        assert_eq!(
            engine.ui_inventory(),
            UiInventoryStatus {
                active_ui_host: None,
                ui: PluginUiContributionStatus {
                    panels: Vec::new(),
                    windows: Vec::new(),
                    widgets: Vec::new(),
                },
            }
        );

        engine
            .activate_core_rewriter(CoreComponent::UiHost, "ui-shell")
            .expect("activate ui host");

        assert_eq!(
            engine.ui_inventory(),
            UiInventoryStatus {
                active_ui_host: Some("ui-shell".to_string()),
                ui: PluginUiContributionStatus {
                    panels: vec!["inspector".to_string()],
                    windows: vec!["governance".to_string()],
                    widgets: vec!["status-pill".to_string()],
                },
            }
        );

        let accepted = engine
            .present_ui_surface(UiSurfaceKind::Panel, "inspector")
            .expect("present declared panel");
        assert_eq!(accepted, "ui Panel accepted by ui-shell: inspector");
        assert_eq!(
            engine.event_inventory().recent_events,
            vec!["ui_host/ui-shell/panel/inspector".to_string()]
        );

        let undeclared = engine
            .present_ui_surface(UiSurfaceKind::Widget, "missing")
            .expect_err("undeclared widget should fail");
        assert!(undeclared
            .to_string()
            .contains("is not declared by active ui host"));
    }

    #[test]
    fn engine_ui_surface_requires_active_ui_host() {
        let engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        let err = engine
            .present_ui_surface(UiSurfaceKind::Panel, "inspector")
            .expect_err("ui action should require activation");
        assert!(err.to_string().contains("no active ui host rewriter"));
    }

    #[test]
    fn engine_reports_event_inventory_from_active_event_bus() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "event-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    events: vec!["models.loaded".to_string(), "plugins.synced".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    event_bus: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");

        assert_eq!(
            engine.event_inventory(),
            EventInventoryStatus {
                active_event_bus_rewriter: None,
                events: Vec::new(),
                recent_events: Vec::new(),
            }
        );

        engine
            .activate_core_rewriter(CoreComponent::EventBus, "event-router")
            .expect("activate event bus");

        assert_eq!(
            engine.event_inventory(),
            EventInventoryStatus {
                active_event_bus_rewriter: Some("event-router".to_string()),
                events: vec!["models.loaded".to_string(), "plugins.synced".to_string()],
                recent_events: Vec::new(),
            }
        );
    }

    #[test]
    fn engine_publishes_only_declared_events_through_active_event_bus() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "event-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    events: vec!["models.loaded".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    event_bus: true,
                    ..Default::default()
                },
                runtime: PluginRuntime::default(),
                bootstrap: PluginBootstrap::default(),
                compatibility: PluginCompatibility::default(),
            }))
            .expect("register");

        let missing_activation = engine
            .publish_event("models.loaded")
            .expect_err("event bus activation should be required");
        assert!(missing_activation
            .to_string()
            .contains("no active event bus rewriter"));

        engine
            .activate_core_rewriter(CoreComponent::EventBus, "event-router")
            .expect("activate event bus");

        let published = engine
            .publish_event("models.loaded")
            .expect("publish declared event");
        assert_eq!(published, "event published by event-router: models.loaded");
        assert_eq!(
            engine.event_inventory().recent_events,
            vec!["event_bus/event-router/models.loaded".to_string()]
        );

        let undeclared = engine
            .publish_event("models.missing")
            .expect_err("undeclared event should fail");
        assert!(undeclared
            .to_string()
            .contains("is not declared by active event bus rewriter"));
    }

    #[test]
    fn engine_reports_command_inventory_from_active_plugin_manager() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "command-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
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
            .expect("register");

        assert_eq!(
            engine.command_inventory(),
            CommandInventoryStatus {
                active_plugin_manager: None,
                commands: Vec::new(),
            }
        );

        engine
            .activate_core_rewriter(CoreComponent::PluginManager, "command-router")
            .expect("activate plugin manager");

        assert_eq!(
            engine.command_inventory(),
            CommandInventoryStatus {
                active_plugin_manager: Some("command-router".to_string()),
                commands: vec!["plugins.reload".to_string(), "plugins.audit".to_string()],
            }
        );
    }

    #[test]
    fn engine_runs_only_declared_commands_through_active_plugin_manager() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "command-router".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                min_host_version: None,
                max_host_version: None,
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    commands: vec!["plugins.reload".to_string()],
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
            .expect("register");

        let missing_activation = engine
            .run_command("plugins.reload")
            .expect_err("active plugin manager should be required");
        assert!(missing_activation
            .to_string()
            .contains("no active plugin manager rewriter"));

        engine
            .activate_core_rewriter(CoreComponent::PluginManager, "command-router")
            .expect("activate plugin manager");

        let accepted = engine
            .run_command("plugins.reload")
            .expect("run declared command");
        assert_eq!(
            accepted,
            "command accepted by command-router: plugins.reload"
        );

        let undeclared = engine
            .run_command("plugins.missing")
            .expect_err("undeclared command should fail");
        assert!(undeclared
            .to_string()
            .contains("is not declared by active plugin manager"));
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Inference),
            Some("inference-plugin")
        );
        assert_eq!(engine.sampling_hook_count(), 1);

        let model_path = temp_model_path("bundle-auto-inference");
        engine
            .load_model("mock", &model_path, BackendParams::default())
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
        assert!(!status.compat.materialized_runtime);
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-text",
                &["pre_generate", "post_generate"],
            ))
            .expect("register legacy text plugin");
        let model_path = temp_model_path("legacy-text");
        engine
            .load_model("mock", &model_path, BackendParams::default())
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            snapshot.compat.text_generation_candidates,
            vec!["legacy-prepost".to_string()]
        );
        assert_eq!(
            snapshot.compat.active_text_generation_plugins,
            vec!["legacy-prepost".to_string()]
        );
        assert_eq!(snapshot.plugins.len(), 1);
        assert_eq!(
            snapshot.configured_core_rewriters,
            Vec::<CoreRewriterStatus>::new()
        );
        assert!(snapshot.plugins[0].compat.legacy_bundle);
        assert!(snapshot.plugins[0].compat.text_generation_candidate);
        assert!(snapshot.plugins[0].compat.active_text_generation);
        assert!(snapshot.plugins[0].compat.materialized_runtime);
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
        let dir = unique_temp_dir("governed-runtime-detail");
        fs::create_dir_all(dir.join("governed-inference").join("runtime")).expect("mkdir");
        fs::write(
            dir.join("governed-inference")
                .join("runtime")
                .join("plugin.dll"),
            b"binary",
        )
        .expect("write runtime");
        fs::write(
            dir.join("governed-inference").join("sampling-hook.toml"),
            "post_sample_override = 4\n",
        )
        .expect("write profile");
        fs::write(
            dir.join("governed-inference").join("manifest.toml"),
            r#"
name = "governed-inference"
version = "1.2.3"
api_version = "1.0"
target_tracks = ["ai_infra"]

[contributes]
model_providers = ["private-registry"]
inference_hooks = ["sampling-profile"]
workflows = ["agent.pipeline"]
custom_nodes = ["node.rewrite"]
commands = ["plugins.reload"]

[contributes.ui_contributes]
panels = ["inspector"]
windows = ["governance"]
widgets = ["status-pill"]

[core_rewriters]
inference = true
workflow = true

[runtime]
library_path = "runtime/plugin.dll"
sampling_profile = "sampling-hook.toml"

[bootstrap]
activate_on_load = ["inference"]
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .load_plugin_manifest_file(dir.join("governed-inference").join("manifest.toml"))
            .expect("register plugin");

        let detail = engine
            .plugin_runtime_detail("governed-inference")
            .expect("detail should exist");

        assert_eq!(detail.status.name, "governed-inference");
        assert!(detail.status.declares_sampling_hook);
        assert_eq!(
            detail.status.sampling_hook_source,
            SamplingHookSource::NativeRuntime
        );
        assert!(detail.status.registered_sampling_hook);
        assert!(detail.status.effective_sampling_hook);
        assert!(detail.status.declares_host_runtime);
        assert!(detail.status.registered_host_runtime);
        assert!(detail.status.materialized_host_runtime);
        assert_eq!(
            detail.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::DynamicLibrary)
        );
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
        assert_eq!(detail.runtime_artifacts.wasm_path, None);
        assert_eq!(
            detail.runtime_artifacts.sampling_profile.as_deref(),
            Some("sampling-hook.toml")
        );
        assert_eq!(detail.compat.legacy_runtime_path, None);
        assert_eq!(detail.runtime_artifacts.host_runtimes.len(), 1);
        assert!(detail.runtime_artifacts.materialized_host_runtime.is_some());
        assert_eq!(
            detail.runtime_artifacts.host_runtimes[0].kind,
            PluginHostRuntimeKind::DynamicLibrary
        );
        assert_eq!(
            detail.runtime_artifacts.host_runtimes[0].declared_path,
            "runtime/plugin.dll"
        );
        assert!(
            Path::new(&detail.runtime_artifacts.host_runtimes[0].resolved_path)
                .ends_with(Path::new("runtime").join("plugin.dll"))
        );
        assert_eq!(
            detail
                .runtime_artifacts
                .materialized_host_runtime
                .as_ref()
                .expect("materialized host runtime")
                .file_size_bytes,
            6
        );
        assert_eq!(detail.model_providers, vec!["private-registry".to_string()]);
        assert_eq!(detail.inference_hooks, vec!["sampling-profile".to_string()]);
        assert_eq!(detail.workflows, vec!["agent.pipeline".to_string()]);
        assert_eq!(detail.custom_nodes, vec!["node.rewrite".to_string()]);
        assert_eq!(detail.commands, vec!["plugins.reload".to_string()]);
        assert_eq!(detail.ui.panels, vec!["inspector".to_string()]);
        assert_eq!(detail.ui.windows, vec!["governance".to_string()]);
        assert_eq!(detail.ui.widgets, vec!["status-pill".to_string()]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_materializes_host_runtime_on_activation_and_reuses_cache() {
        let dir = unique_temp_dir("materialized-host-runtime");
        fs::create_dir_all(dir.join("hosted-workflow").join("runtime")).expect("mkdir");
        fs::write(
            dir.join("hosted-workflow")
                .join("runtime")
                .join("plugin.dll"),
            b"binary",
        )
        .expect("write runtime");
        fs::write(
            dir.join("hosted-workflow").join("manifest.toml"),
            r#"
name = "hosted-workflow"
version = "1.0.0"
api_version = "1.0"

[core_rewriters]
workflow = true
plugin_manager = true

[runtime]
library_path = "runtime/plugin.dll"
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .load_plugin_manifest_file(dir.join("hosted-workflow").join("manifest.toml"))
            .expect("load plugin");

        let before = engine
            .plugin_runtime_detail("hosted-workflow")
            .expect("detail before activation");
        assert!(before.status.declares_host_runtime);
        assert!(before.status.registered_host_runtime);
        assert!(!before.status.materialized_host_runtime);
        assert_eq!(
            before.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::DynamicLibrary)
        );
        assert!(before.runtime_artifacts.materialized_host_runtime.is_none());

        engine
            .activate_core_rewriter(CoreComponent::Workflow, "hosted-workflow")
            .expect("activate workflow");

        let after = engine
            .plugin_runtime_detail("hosted-workflow")
            .expect("detail after activation");
        assert!(after.status.materialized_host_runtime);
        assert_eq!(
            after.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::DynamicLibrary)
        );
        let materialized = after
            .runtime_artifacts
            .materialized_host_runtime
            .as_ref()
            .expect("materialized host runtime");
        assert_eq!(materialized.kind, PluginHostRuntimeKind::DynamicLibrary);
        assert_eq!(materialized.file_size_bytes, 6);
        assert!(Path::new(&materialized.resolved_path).ends_with("plugin.dll"));

        fs::remove_file(
            dir.join("hosted-workflow")
                .join("runtime")
                .join("plugin.dll"),
        )
        .expect("remove runtime after materialization");

        engine
            .activate_core_rewriter(CoreComponent::PluginManager, "hosted-workflow")
            .expect("reuse materialized runtime");

        let reused = engine
            .plugin_runtime_detail("hosted-workflow")
            .expect("detail after reuse");
        assert!(reused.status.materialized_host_runtime);
        assert_eq!(
            reused
                .runtime_artifacts
                .materialized_host_runtime
                .as_ref()
                .expect("materialized runtime")
                .file_size_bytes,
            6
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn engine_defers_invalid_wasm_host_runtime_failure_until_activation() {
        let dir = unique_temp_dir("invalid-wasm-runtime");
        fs::create_dir_all(dir.join("hosted-wasm").join("runtime")).expect("mkdir");
        fs::write(
            dir.join("hosted-wasm").join("runtime").join("plugin.wasm"),
            b"not-wasm",
        )
        .expect("write runtime");
        fs::write(
            dir.join("hosted-wasm").join("manifest.toml"),
            r#"
name = "hosted-wasm"
version = "1.0.0"
api_version = "1.0"

[core_rewriters]
workflow = true

[runtime]
wasm_path = "runtime/plugin.wasm"
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .load_plugin_manifest_file(dir.join("hosted-wasm").join("manifest.toml"))
            .expect("load plugin");

        let before = engine
            .plugin_runtime_detail("hosted-wasm")
            .expect("detail before activation");
        assert!(before.status.declares_host_runtime);
        assert!(before.status.registered_host_runtime);
        assert!(!before.status.materialized_host_runtime);
        assert_eq!(
            before.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::WasmModule)
        );

        let err = engine
            .activate_core_rewriter(CoreComponent::Workflow, "hosted-wasm")
            .expect_err("invalid wasm should fail during materialization");
        assert!(err.to_string().contains("wasm magic header"));
        assert_eq!(engine.active_core_rewriter(CoreComponent::Workflow), None);

        let after = engine
            .plugin_runtime_detail("hosted-wasm")
            .expect("detail after failed activation");
        assert!(!after.status.materialized_host_runtime);
        assert!(after.runtime_artifacts.materialized_host_runtime.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn workspace_example_ui_shell_materializes_host_runtime_on_activation() {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let manifest_path = workspace_root
            .join("plugins")
            .join("example-ui-shell")
            .join("manifest.toml");

        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .load_plugin_manifest_file(&manifest_path)
            .expect("load workspace example ui shell");

        let before = engine
            .plugin_runtime_detail("example-ui-shell")
            .expect("detail before activation");
        assert!(before.status.declares_host_runtime);
        assert!(before.status.registered_host_runtime);
        assert!(!before.status.materialized_host_runtime);
        assert_eq!(
            before.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::DynamicLibrary)
        );
        assert!(before.runtime_artifacts.materialized_host_runtime.is_none());

        engine
            .activate_core_rewriter(CoreComponent::UiHost, "example-ui-shell")
            .expect("activate ui host from workspace example");

        let after = engine
            .plugin_runtime_detail("example-ui-shell")
            .expect("detail after activation");
        assert!(after.status.materialized_host_runtime);
        assert_eq!(
            after.status.host_runtime_kind,
            Some(PluginHostRuntimeKind::DynamicLibrary)
        );
        assert_eq!(
            after.ui.panels,
            vec![
                "workspace-overview".to_string(),
                "model-catalog".to_string()
            ]
        );
        assert_eq!(after.ui.windows, vec!["operations-console".to_string()]);
        assert_eq!(after.ui.widgets, vec!["runtime-status".to_string()]);
        let materialized = after
            .runtime_artifacts
            .materialized_host_runtime
            .as_ref()
            .expect("materialized host runtime");
        assert_eq!(materialized.kind, PluginHostRuntimeKind::DynamicLibrary);
        assert!(materialized.file_size_bytes > 0);
        assert!(Path::new(&materialized.resolved_path).ends_with(
            Path::new("plugins")
                .join("example-ui-shell")
                .join("runtime")
                .join("plugin.dll")
        ));
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
        let model_path = temp_model_path("dynamic-hook");
        engine
            .load_model("mock", &model_path, BackendParams::default())
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
        };

        engine
            .register_plugin(registered_legacy_text_plugin_for_tests(
                "legacy-sampler",
                &["transform_logits", "post_sample"],
            ))
            .expect("register legacy sampler");
        let model_path = temp_model_path("legacy-sampler");
        engine
            .load_model("mock", &model_path, BackendParams::default())
            .expect("load model");

        let before = engine.runtime_snapshot();
        assert!(before.plugins[0].declares_sampling_hook);
        assert_eq!(
            before.plugins[0].sampling_hook_source,
            SamplingHookSource::LegacyCompat
        );
        assert!(!before.plugins[0].registered_sampling_hook);
        assert!(!before.plugins[0].effective_sampling_hook);
        assert!(!before.plugins[0].compat.materialized_runtime);
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
        assert!(after.plugins[0].compat.materialized_runtime);
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
        assert!(snapshot.plugins[0].compat.materialized_runtime);
        assert!(snapshot.plugins[0].active_inference_rewriter);
        assert!(snapshot.plugins[0].has_sampling_hook);

        let model_path = temp_model_path("legacy-auto-activation");
        engine
            .load_model("mock", &model_path, BackendParams::default())
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
            host_plugin_runtimes: BTreeMap::new(),
            legacy_text_runtime: LegacyTextCompatCoordinator::default(),
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
