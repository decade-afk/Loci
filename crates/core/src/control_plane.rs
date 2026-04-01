use loci_plugin_api::{CoreComponent, LegacyRuntimeBridge, PluginSourceFormat};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SamplingHookSource {
    None,
    NativeRuntime,
    LegacyCompat,
    DynamicRegistration,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginRuntimeStatus {
    pub name: String,
    pub version: String,
    pub supports_ai_infra: bool,
    pub supports_ai_agent: bool,
    pub source_format: PluginSourceFormat,
    pub runtime_bridge: LegacyRuntimeBridge,
    pub declares_inference_rewriter: bool,
    pub declares_sampling_hook: bool,
    pub sampling_hook_source: SamplingHookSource,
    pub registered_sampling_hook: bool,
    pub effective_sampling_hook: bool,
    pub materialized_legacy_runtime: bool,
    pub active_inference_rewriter: bool,
    pub has_sampling_hook: bool,
    pub is_legacy_compat: bool,
    pub legacy_text_candidate: bool,
    pub active_legacy_text: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginRuntimeDetail {
    pub status: PluginRuntimeStatus,
    pub declared_core_rewriters: Vec<CoreComponent>,
    pub auto_activate_components: Vec<CoreComponent>,
    pub active_core_rewriters: Vec<CoreComponent>,
    pub model_providers: Vec<String>,
    pub inference_hooks: Vec<String>,
    pub commands: Vec<String>,
    pub legacy_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoreRewriterStatus {
    pub component: CoreComponent,
    pub plugin_name: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    pub plugin_count: usize,
    pub loaded_plugin_names: Vec<String>,
    pub active_backend: Option<String>,
    pub active_inference: Option<String>,
    pub configured_core_rewriters: Vec<CoreRewriterStatus>,
    pub legacy_text_candidates: Vec<String>,
    pub active_legacy_text: Vec<String>,
    pub plugins: Vec<PluginRuntimeStatus>,
}

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
