use crate::plugin::RegisteredPlugin;
use anyhow::Result as AnyhowResult;

pub trait ModelRepository: Send + Sync {
    fn has_model(&self, model_id: &str) -> bool;
}

pub trait WorkflowEngine: Send + Sync {
    fn workflow_count(&self) -> usize;
}

pub trait EventBus: Send + Sync {
    fn publish(&self, event: &str) -> AnyhowResult<()>;
}

pub trait PluginManager: Send + Sync {
    fn register(&mut self, plugin: RegisteredPlugin) -> AnyhowResult<()>;
    fn list(&self) -> &[RegisteredPlugin];
}

pub trait CoreRegistry: Send + Sync {
    fn model_repository(&self) -> &dyn ModelRepository;
    fn workflow_engine(&self) -> &dyn WorkflowEngine;
    fn event_bus(&self) -> &dyn EventBus;
    fn plugin_manager(&self) -> &dyn PluginManager;
    fn plugin_manager_mut(&mut self) -> &mut dyn PluginManager;
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

pub struct DefaultCoreRegistry {
    model_repository: DefaultModelRepository,
    workflow_engine: DefaultWorkflowEngine,
    event_bus: NoopEventBus,
    plugin_manager: crate::plugin::InMemoryPluginManager,
}

impl Default for DefaultCoreRegistry {
    fn default() -> Self {
        Self {
            model_repository: DefaultModelRepository,
            workflow_engine: DefaultWorkflowEngine,
            event_bus: NoopEventBus,
            plugin_manager: crate::plugin::InMemoryPluginManager::default(),
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

    fn plugin_manager(&self) -> &dyn PluginManager {
        &self.plugin_manager
    }

    fn plugin_manager_mut(&mut self) -> &mut dyn PluginManager {
        &mut self.plugin_manager
    }
}
