use crate::engine::{
    CoreRewriterStatus, InferenceEngine, PluginRuntimeDetail, PluginRuntimeStatus, RuntimeSnapshot,
};
use crate::error::{LociError, Result};
use loci_plugin_api::CoreComponent;
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ManagementHealthStatus {
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InferenceActivationStatus {
    pub status: &'static str,
    pub component: CoreComponent,
    pub plugin_name: String,
    pub active_inference: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LegacyTextPluginActivationStatus {
    pub status: &'static str,
    pub plugin_name: String,
    pub active_legacy_text: Vec<String>,
}

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

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(snapshot.configured_core_rewriters.is_empty());
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
}
