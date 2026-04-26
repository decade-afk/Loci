use crate::backend::{BackendRegistry, InferenceParams, Model};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::pipeline::{merge_inference_params, InferenceResponse};
use crate::plugin::{
    discover_plugin_manifest_files, load_plugin_manifest_file, PluginSamplingRuntime, PluginStatus,
    RegisteredPlugin,
};
use crate::runtime::{ActiveModelStatus, RuntimeSnapshot};
use crate::{PluginKind, PluginManifest};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct InferenceEngine {
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
    pub(crate) plugin_manifests: Vec<RegisteredPlugin>,
    pub(crate) active_plugins: BTreeMap<PluginKind, String>,
    pub(crate) sampling_runtime: PluginSamplingRuntime,
}

impl InferenceEngine {
    pub fn builder() -> crate::engine::InferenceEngineBuilder {
        crate::engine::InferenceEngineBuilder::new()
    }

    pub fn backend_capabilities(&self) -> Vec<crate::BackendCapabilities> {
        self.backend_registry.list()
    }

    pub fn active_backend(&self) -> Option<&str> {
        self.active_backend.as_deref()
    }

    pub fn model_path(&self) -> Option<&Path> {
        self.model_path.as_deref()
    }

    pub fn load_model(&mut self, backend_name: &str, model_path: impl AsRef<Path>) -> Result<()> {
        let config = ModelConfig::new(model_path.as_ref());
        self.load_model_config(backend_name, &config)
    }

    pub fn load_model_config(&mut self, backend_name: &str, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        let governed = self.govern_model_config(config);
        let mut model = self.backend_registry.load_model(
            backend_name,
            &governed.model_path,
            governed.to_backend_params(),
        )?;
        model.attach_sampling_runtime(self.sampling_runtime.clone())?;

        self.active_backend = Some(backend_name.to_string());
        self.model_path = Some(governed.model_path.clone());
        self.model = Some(model);
        Ok(())
    }

    pub fn generate(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        if prompt.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "prompt must not be empty".to_string(),
            ));
        }

        let params = merge_inference_params(&self.default_inference_params, params);
        model.infer_text(prompt, &params)
    }

    pub fn infer(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<InferenceResponse> {
        let output = self.generate(prompt, params)?;
        Ok(InferenceResponse {
            output,
            backend: self.active_backend.clone(),
            model_path: self.model_path.as_ref().map(|path| path.display().to_string()),
        })
    }

    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            plugin_count: self.plugin_manifests.len(),
            plugins: self.plugin_statuses(),
            active_plugins: self
                .active_plugins
                .values()
                .cloned()
                .collect::<Vec<_>>(),
            available_backends: self.backend_registry.list(),
            active_backend: self.active_backend.clone(),
            active_model_path: self.model_path.as_ref().map(|path| path.display().to_string()),
            active_model_info: self.model.as_ref().map(|model| {
                let metadata = model.metadata();
                ActiveModelStatus {
                    architecture: metadata.architecture,
                    n_vocab: metadata.n_vocab,
                    n_ctx_train: metadata.n_ctx_train,
                    n_embd: metadata.n_embd,
                    n_layer: metadata.n_layer,
                    param_count: metadata.param_count,
                }
            }),
        }
    }

    pub fn plugin_count(&self) -> usize {
        self.plugin_manifests.len()
    }

    pub fn plugin_names(&self) -> Vec<String> {
        self.plugin_manifests
            .iter()
            .map(|plugin| plugin.manifest.name.clone())
            .collect()
    }

    pub fn plugin_manifests(&self) -> &[RegisteredPlugin] {
        &self.plugin_manifests
    }

    pub fn load_plugin_manifest_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let plugin = load_plugin_manifest_file(path)?;
        self.register_plugin(plugin)
    }

    pub fn load_plugins_from_dir(&mut self, root: impl AsRef<Path>) -> Result<usize> {
        let manifests = discover_plugin_manifest_files(root)?;
        let mut loaded = 0usize;
        for manifest_path in manifests {
            self.load_plugin_manifest_file(manifest_path)?;
            loaded += 1;
        }
        Ok(loaded)
    }

    pub fn register_plugin(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        if self
            .plugin_manifests
            .iter()
            .any(|existing| existing.manifest.name == plugin.manifest.name)
        {
            return Err(LociError::ConfigError(format!(
                "plugin already registered: {}",
                plugin.manifest.name
            )));
        }

        if plugin.manifest.auto_activate {
            self.active_plugins
                .insert(plugin.manifest.kind, plugin.manifest.name.clone());
        }
        self.plugin_manifests.push(plugin);
        self.plugin_manifests.sort_by(|left, right| {
            right
                .manifest
                .priority
                .cmp(&left.manifest.priority)
                .then_with(|| left.manifest.name.cmp(&right.manifest.name))
        });
        Ok(())
    }

    pub fn activate_plugin(&mut self, plugin_name: &str) -> Result<()> {
        let manifest = self
            .plugin_manifests
            .iter()
            .find(|plugin| plugin.manifest.name == plugin_name)
            .map(|plugin| plugin.manifest.clone())
            .ok_or_else(|| {
                LociError::ConfigError(format!("plugin not registered: {plugin_name}"))
            })?;
        self.active_plugins
            .insert(manifest.kind, manifest.name.clone());
        Ok(())
    }

    pub fn active_plugin(&self, kind: PluginKind) -> Option<&str> {
        self.active_plugins.get(&kind).map(String::as_str)
    }

    pub fn plugins_for_kind(&self, kind: PluginKind) -> Vec<&PluginManifest> {
        self.plugin_manifests
            .iter()
            .filter(|plugin| plugin.manifest.kind == kind)
            .map(|plugin| &plugin.manifest)
            .collect()
    }

    fn plugin_statuses(&self) -> Vec<PluginStatus> {
        self.plugin_manifests
            .iter()
            .map(|plugin| PluginStatus {
                name: plugin.manifest.name.clone(),
                version: plugin.manifest.version.clone(),
                kind: plugin.manifest.kind,
                auto_activate: plugin.manifest.auto_activate,
                priority: plugin.manifest.priority,
                model_formats: plugin.manifest.capabilities.model_formats.clone(),
                hardware_targets: plugin.manifest.capabilities.hardware_targets.clone(),
                features: plugin.manifest.capabilities.features.clone(),
                has_native_runtime: plugin.manifest.runtime.library_path.is_some(),
                has_wasm_runtime: plugin.manifest.runtime.wasm_path.is_some(),
                is_active: self.active_plugin(plugin.manifest.kind)
                    == Some(plugin.manifest.name.as_str()),
            })
            .collect()
    }

    fn govern_model_config(&self, config: &ModelConfig) -> ModelConfig {
        let hardware_plugin = self.active_plugin(PluginKind::HardwareBackend);
        let allows_gpu = hardware_plugin.is_some();

        if config.use_gpu && !allows_gpu {
            let mut cpu_fallback = config.clone();
            cpu_fallback.use_gpu = false;
            cpu_fallback.n_gpu_layers = 0;
            cpu_fallback.kv_offload = false;
            cpu_fallback.op_offload = false;
            cpu_fallback.split_mode = crate::GpuSplitMode::None;
            cpu_fallback.main_gpu = 0;
            cpu_fallback.tensor_split = None;
            return cpu_fallback;
        }

        match config.load_strategy {
            ModelLoadStrategy::Strict => config.clone(),
            ModelLoadStrategy::AutoReduceGpuLayers { step } if config.use_gpu && step > 0 => {
                let mut adjusted = config.clone();
                if adjusted.n_gpu_layers > 0 {
                    adjusted.n_gpu_layers = (adjusted.n_gpu_layers - step as i32).max(0);
                }
                adjusted
            }
            _ => config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_model_path(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-engine-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("demo.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        path
    }

    #[test]
    fn engine_can_load_model_and_generate() {
        let model_path = temp_model_path("generate");
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("mock:hello"));
    }

    #[test]
    fn engine_auto_activates_plugin() {
        let plugin_dir = temp_model_path("plugin-dir")
            .parent()
            .expect("parent")
            .join("plugins");
        fs::create_dir_all(plugin_dir.join("cpu")).expect("mkdir");
        fs::write(
            plugin_dir.join("cpu").join("manifest.toml"),
            r#"
name = "cpu"
version = "0.1.0"
api_version = "1.0"
kind = "hardware_backend"
auto_activate = true
"#,
        )
        .expect("write manifest");

        let mut engine = InferenceEngine::builder().build().expect("build");
        engine
            .load_plugins_from_dir(&plugin_dir)
            .expect("load plugins");

        assert_eq!(engine.active_plugin(PluginKind::HardwareBackend), Some("cpu"));
    }
}
