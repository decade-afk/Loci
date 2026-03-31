use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlatformTrack {
    AiInfra,
    AiAgent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreComponent {
    Inference,
    Model,
    Hardware,
    Workflow,
    EventBus,
    PluginManager,
    UiHost,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UiContributionPoints {
    #[serde(default)]
    pub panels: Vec<String>,
    #[serde(default)]
    pub windows: Vec<String>,
    #[serde(default)]
    pub widgets: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContributionPoints {
    #[serde(default)]
    pub model_providers: Vec<String>,
    #[serde(default)]
    pub inference_hooks: Vec<String>,
    #[serde(default)]
    pub workflows: Vec<String>,
    #[serde(default)]
    pub custom_nodes: Vec<String>,
    #[serde(default)]
    pub commands: Vec<String>,
    #[serde(default)]
    pub ui_contributes: UiContributionPoints,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CoreRewriters {
    #[serde(default)]
    pub inference: bool,
    #[serde(default)]
    pub model: bool,
    #[serde(default)]
    pub hardware: bool,
    #[serde(default)]
    pub workflow: bool,
    #[serde(default)]
    pub event_bus: bool,
    #[serde(default)]
    pub plugin_manager: bool,
    #[serde(default)]
    pub ui_host: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub api_version: String,
    #[serde(default)]
    pub target_tracks: Vec<PlatformTrack>,
    #[serde(default)]
    pub contributes: ContributionPoints,
    #[serde(default)]
    pub core_rewriters: CoreRewriters,
}

impl ContributionPoints {
    pub fn is_empty(&self) -> bool {
        self.model_providers.is_empty()
            && self.inference_hooks.is_empty()
            && self.workflows.is_empty()
            && self.custom_nodes.is_empty()
            && self.commands.is_empty()
            && self.ui_contributes.panels.is_empty()
            && self.ui_contributes.windows.is_empty()
            && self.ui_contributes.widgets.is_empty()
    }
}

impl CoreRewriters {
    pub fn supports(&self, component: CoreComponent) -> bool {
        match component {
            CoreComponent::Inference => self.inference,
            CoreComponent::Model => self.model,
            CoreComponent::Hardware => self.hardware,
            CoreComponent::Workflow => self.workflow,
            CoreComponent::EventBus => self.event_bus,
            CoreComponent::PluginManager => self.plugin_manager,
            CoreComponent::UiHost => self.ui_host,
        }
    }

    pub fn declared_components(&self) -> Vec<CoreComponent> {
        [
            CoreComponent::Inference,
            CoreComponent::Model,
            CoreComponent::Hardware,
            CoreComponent::Workflow,
            CoreComponent::EventBus,
            CoreComponent::PluginManager,
            CoreComponent::UiHost,
        ]
        .into_iter()
        .filter(|component| self.supports(*component))
        .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.declared_components().is_empty()
    }
}

impl PluginManifest {
    pub fn supports_track(&self, track: PlatformTrack) -> bool {
        self.target_tracks.is_empty() || self.target_tracks.contains(&track)
    }

    pub fn declares_core_rewriter(&self, component: CoreComponent) -> bool {
        self.core_rewriters.supports(component)
    }
}

pub trait Plugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_defaults_to_all_tracks_when_unspecified() {
        let manifest = PluginManifest {
            name: "demo".to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            target_tracks: Vec::new(),
            contributes: ContributionPoints::default(),
            core_rewriters: CoreRewriters::default(),
        };

        assert!(manifest.supports_track(PlatformTrack::AiInfra));
        assert!(manifest.supports_track(PlatformTrack::AiAgent));
    }

    #[test]
    fn core_rewriters_report_declared_components() {
        let rewriters = CoreRewriters {
            inference: true,
            workflow: true,
            ui_host: true,
            ..Default::default()
        };

        assert_eq!(
            rewriters.declared_components(),
            vec![
                CoreComponent::Inference,
                CoreComponent::Workflow,
                CoreComponent::UiHost,
            ]
        );
    }
}
