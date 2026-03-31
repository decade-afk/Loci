use crate::plugin::RegisteredPlugin;
use anyhow::{bail, Result as AnyhowResult};
use loci_plugin_api::{CoreComponent, PlatformTrack};
use std::collections::BTreeMap;
use std::sync::Arc;

pub trait ModelRepository: Send + Sync {
    fn has_model(&self, model_id: &str) -> bool;
}

pub trait WorkflowEngine: Send + Sync {
    fn workflow_count(&self) -> usize;
}

pub trait EventBus: Send + Sync {
    fn publish(&self, event: &str) -> AnyhowResult<()>;
}

pub trait HardwareAbstraction: Send + Sync {
    fn available_accelerators(&self) -> Vec<String>;
}

pub trait UiHost: Send + Sync {
    fn is_headless(&self) -> bool;
}

pub trait PluginManager: Send + Sync {
    fn register(&mut self, plugin: RegisteredPlugin) -> AnyhowResult<()>;
    fn register_sampling_hook(
        &mut self,
        plugin_name: &str,
        hook: Arc<dyn crate::plugin::SamplingHook>,
    ) -> AnyhowResult<()>;
    fn list(&self) -> &[RegisteredPlugin];
    fn get(&self, plugin_name: &str) -> Option<&RegisteredPlugin>;
    fn plugins_for_track(&self, track: PlatformTrack) -> Vec<&RegisteredPlugin>;
    fn plugins_for_model_provider(&self, provider: &str) -> Vec<&RegisteredPlugin>;
    fn plugins_for_core_component(&self, component: CoreComponent) -> Vec<&RegisteredPlugin>;
    fn sampling_runtime_for_inference(
        &self,
        active_plugin_name: Option<&str>,
    ) -> crate::plugin::PluginSamplingRuntime;
}

pub trait CoreRegistry: Send + Sync {
    fn model_repository(&self) -> &dyn ModelRepository;
    fn workflow_engine(&self) -> &dyn WorkflowEngine;
    fn event_bus(&self) -> &dyn EventBus;
    fn hardware_abstraction(&self) -> &dyn HardwareAbstraction;
    fn ui_host(&self) -> &dyn UiHost;
    fn plugin_manager(&self) -> &dyn PluginManager;
    fn plugin_manager_mut(&mut self) -> &mut dyn PluginManager;
    fn activate_core_rewriter(
        &mut self,
        component: CoreComponent,
        plugin_name: &str,
    ) -> AnyhowResult<()>;
    fn active_core_rewriter(&self, component: CoreComponent) -> Option<&str>;
    fn configured_core_rewriters(&self) -> Vec<(CoreComponent, String)>;
}

#[derive(Default)]
pub struct DefaultModelRepository;

impl ModelRepository for DefaultModelRepository {
    fn has_model(&self, _model_id: &str) -> bool {
        false
    }
}

#[derive(Default)]
pub struct DefaultWorkflowEngine;

impl WorkflowEngine for DefaultWorkflowEngine {
    fn workflow_count(&self) -> usize {
        0
    }
}

#[derive(Default)]
pub struct NoopEventBus;

impl EventBus for NoopEventBus {
    fn publish(&self, _event: &str) -> AnyhowResult<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct DefaultHardwareAbstraction;

impl HardwareAbstraction for DefaultHardwareAbstraction {
    fn available_accelerators(&self) -> Vec<String> {
        vec!["cpu".to_string()]
    }
}

#[derive(Default)]
pub struct HeadlessUiHost;

impl UiHost for HeadlessUiHost {
    fn is_headless(&self) -> bool {
        true
    }
}

pub struct DefaultCoreRegistry {
    model_repository: DefaultModelRepository,
    workflow_engine: DefaultWorkflowEngine,
    event_bus: NoopEventBus,
    hardware_abstraction: DefaultHardwareAbstraction,
    ui_host: HeadlessUiHost,
    plugin_manager: crate::plugin::InMemoryPluginManager,
    active_core_rewriters: BTreeMap<CoreComponent, String>,
}

impl Default for DefaultCoreRegistry {
    fn default() -> Self {
        Self {
            model_repository: DefaultModelRepository,
            workflow_engine: DefaultWorkflowEngine,
            event_bus: NoopEventBus,
            hardware_abstraction: DefaultHardwareAbstraction,
            ui_host: HeadlessUiHost,
            plugin_manager: crate::plugin::InMemoryPluginManager::default(),
            active_core_rewriters: BTreeMap::new(),
        }
    }
}

impl CoreRegistry for DefaultCoreRegistry {
    fn model_repository(&self) -> &dyn ModelRepository {
        &self.model_repository
    }

    fn workflow_engine(&self) -> &dyn WorkflowEngine {
        &self.workflow_engine
    }

    fn event_bus(&self) -> &dyn EventBus {
        &self.event_bus
    }

    fn hardware_abstraction(&self) -> &dyn HardwareAbstraction {
        &self.hardware_abstraction
    }

    fn ui_host(&self) -> &dyn UiHost {
        &self.ui_host
    }

    fn plugin_manager(&self) -> &dyn PluginManager {
        &self.plugin_manager
    }

    fn plugin_manager_mut(&mut self) -> &mut dyn PluginManager {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::RegisteredPlugin;
    use loci_plugin_api::{ContributionPoints, CoreRewriters, PluginManifest};

    #[test]
    fn registry_can_activate_declared_core_rewriter() {
        let mut registry = DefaultCoreRegistry::default();
        registry
            .plugin_manager_mut()
            .register(RegisteredPlugin::new(PluginManifest {
                name: "workflow-override".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
            }))
            .expect("register");

        registry
            .activate_core_rewriter(CoreComponent::Workflow, "workflow-override")
            .expect("activate");

        assert_eq!(
            registry.active_core_rewriter(CoreComponent::Workflow),
            Some("workflow-override")
        );
    }

    #[test]
    fn registry_rejects_undeclared_core_rewriter() {
        let mut registry = DefaultCoreRegistry::default();
        registry
            .plugin_manager_mut()
            .register(RegisteredPlugin::new(PluginManifest {
                name: "model-provider".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints {
                    model_providers: vec!["private-registry".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters::default(),
            }))
            .expect("register");

        let err = registry
            .activate_core_rewriter(CoreComponent::Inference, "model-provider")
            .expect_err("should reject");

        assert!(err
            .to_string()
            .contains("does not declare core rewriter capability"));
    }
}
