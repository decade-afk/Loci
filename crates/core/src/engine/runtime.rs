use crate::backend::{BackendRegistry, InferenceParams, Model};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::pipeline::{merge_inference_params, InferenceResponse};
use crate::plugin::{
    discover_plugin_manifest_files, load_plugin_manifest_file, PluginRuntimeKind,
    PluginSamplingRuntime, PluginStatus, RegisteredPlugin,
};
use crate::runtime::{ActiveModelStatus, ModelUnloadStatus, RuntimeSnapshot};
use crate::{PluginKind, PluginManifest};
use libloading::Library;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d];

#[allow(dead_code)]
enum LoadedPluginRuntime {
    Native { path: PathBuf, library: Library },
    Wasm { path: PathBuf, bytes: Vec<u8> },
}

impl LoadedPluginRuntime {
    fn kind(&self) -> PluginRuntimeKind {
        match self {
            Self::Native { .. } => PluginRuntimeKind::Native,
            Self::Wasm { .. } => PluginRuntimeKind::Wasm,
        }
    }
}

#[derive(Default)]
pub struct InferenceEngine {
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
    pub(crate) plugin_manifests: Vec<RegisteredPlugin>,
    pub(crate) active_plugins: BTreeMap<PluginKind, String>,
    loaded_plugin_runtimes: BTreeMap<String, LoadedPluginRuntime>,
    pub(crate) sampling_runtime: PluginSamplingRuntime,
}

impl InferenceEngine {
    pub fn builder() -> crate::engine::InferenceEngineBuilder {
        crate::engine::InferenceEngineBuilder::new()
    }

    pub fn backend_capabilities(&self) -> Vec<crate::BackendCapabilities> {
        self.backend_registry.list()
    }

    pub(crate) fn new(
        backend_registry: BackendRegistry,
        default_inference_params: InferenceParams,
    ) -> Self {
        Self {
            backend_registry,
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params,
            plugin_manifests: Vec::new(),
            active_plugins: BTreeMap::new(),
            loaded_plugin_runtimes: BTreeMap::new(),
            sampling_runtime: PluginSamplingRuntime::default(),
        }
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

    pub fn unload_model(&mut self) -> ModelUnloadStatus {
        let previous_backend = self.active_backend.take();
        let previous_model_path = self
            .model_path
            .take()
            .map(|path| path.display().to_string());
        let unloaded =
            previous_backend.is_some() || previous_model_path.is_some() || self.model.is_some();
        self.model = None;

        ModelUnloadStatus {
            unloaded,
            previous_backend,
            previous_model_path,
        }
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

    pub fn infer(&mut self, prompt: &str, params: &InferenceParams) -> Result<InferenceResponse> {
        let output = self.generate(prompt, params)?;
        Ok(InferenceResponse {
            output,
            backend: self.active_backend.clone(),
            model_path: self
                .model_path
                .as_ref()
                .map(|path| path.display().to_string()),
        })
    }

    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            plugin_count: self.plugin_manifests.len(),
            plugins: self.plugin_statuses(),
            active_plugins: self.active_plugins.values().cloned().collect::<Vec<_>>(),
            available_backends: self.backend_registry.list(),
            active_backend: self.active_backend.clone(),
            active_model_path: self
                .model_path
                .as_ref()
                .map(|path| path.display().to_string()),
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

        let plugin_name = plugin.manifest.name.clone();
        let auto_activate = plugin.manifest.auto_activate;
        self.plugin_manifests.push(plugin);
        self.plugin_manifests.sort_by(|left, right| {
            right
                .manifest
                .priority
                .cmp(&left.manifest.priority)
                .then_with(|| left.manifest.name.cmp(&right.manifest.name))
        });
        if auto_activate {
            if let Err(error) = self.activate_plugin(&plugin_name) {
                self.plugin_manifests
                    .retain(|existing| existing.manifest.name != plugin_name);
                return Err(error);
            }
        }
        Ok(())
    }

    pub fn activate_plugin(&mut self, plugin_name: &str) -> Result<()> {
        let plugin = self
            .plugin_manifests
            .iter()
            .find(|plugin| plugin.manifest.name == plugin_name)
            .cloned()
            .ok_or_else(|| {
                LociError::ConfigError(format!("plugin not registered: {plugin_name}"))
            })?;
        self.materialize_plugin_runtime(&plugin)?;
        self.active_plugins
            .insert(plugin.manifest.kind, plugin.manifest.name.clone());
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
                runtime_kind: self
                    .loaded_plugin_runtimes
                    .get(plugin.manifest.name.as_str())
                    .map(LoadedPluginRuntime::kind)
                    .or_else(|| plugin.declared_runtime_kind()),
                name: plugin.manifest.name.clone(),
                version: plugin.manifest.version.clone(),
                kind: plugin.manifest.kind,
                auto_activate: plugin.manifest.auto_activate,
                priority: plugin.manifest.priority,
                model_formats: plugin.manifest.capabilities.model_formats.clone(),
                hardware_targets: plugin.manifest.capabilities.hardware_targets.clone(),
                features: plugin.manifest.capabilities.features.clone(),
                declares_runtime: plugin.declared_runtime_kind().is_some(),
                runtime_loaded: self
                    .loaded_plugin_runtimes
                    .contains_key(plugin.manifest.name.as_str()),
                has_native_runtime: plugin.manifest.runtime.library_path.is_some(),
                has_wasm_runtime: plugin.manifest.runtime.wasm_path.is_some(),
                is_active: self.active_plugin(plugin.manifest.kind)
                    == Some(plugin.manifest.name.as_str()),
            })
            .collect()
    }

    fn materialize_plugin_runtime(&mut self, plugin: &RegisteredPlugin) -> Result<()> {
        if self
            .loaded_plugin_runtimes
            .contains_key(plugin.manifest.name.as_str())
        {
            return Ok(());
        }

        let Some(runtime) = self.load_plugin_runtime(plugin)? else {
            return Ok(());
        };

        self.loaded_plugin_runtimes
            .insert(plugin.manifest.name.clone(), runtime);
        Ok(())
    }

    fn load_plugin_runtime(
        &self,
        plugin: &RegisteredPlugin,
    ) -> Result<Option<LoadedPluginRuntime>> {
        let Some(runtime_kind) = plugin.declared_runtime_kind() else {
            return Ok(None);
        };
        let runtime_path = plugin.resolved_runtime_path().ok_or_else(|| {
            LociError::ConfigError(format!(
                "plugin `{}` declared a runtime without a runtime path",
                plugin.manifest.name
            ))
        })?;

        match runtime_kind {
            PluginRuntimeKind::Native => {
                if !runtime_path.is_file() {
                    return Err(LociError::ConfigError(format!(
                        "plugin `{}` native runtime not found: {}",
                        plugin.manifest.name,
                        runtime_path.display()
                    )));
                }

                let library = unsafe { Library::new(&runtime_path) }.map_err(|error| {
                    LociError::ConfigError(format!(
                        "failed to load native runtime for plugin `{}` from {}: {error}",
                        plugin.manifest.name,
                        runtime_path.display()
                    ))
                })?;

                Ok(Some(LoadedPluginRuntime::Native {
                    path: runtime_path,
                    library,
                }))
            }
            PluginRuntimeKind::Wasm => {
                let bytes = fs::read(&runtime_path).map_err(|error| {
                    LociError::ConfigError(format!(
                        "failed to read wasm runtime for plugin `{}` from {}: {error}",
                        plugin.manifest.name,
                        runtime_path.display()
                    ))
                })?;
                if bytes.len() < WASM_MAGIC.len() || bytes[..WASM_MAGIC.len()] != WASM_MAGIC {
                    return Err(LociError::ConfigError(format!(
                        "plugin `{}` runtime is not a valid wasm module: {}",
                        plugin.manifest.name,
                        runtime_path.display()
                    )));
                }

                Ok(Some(LoadedPluginRuntime::Wasm {
                    path: runtime_path,
                    bytes,
                }))
            }
        }
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
    fn engine_can_unload_model() {
        let model_path = temp_model_path("unload");
        let model_path_text = model_path.display().to_string();
        let mut engine = InferenceEngine::builder()
            .backend("mock")
            .model_path(&model_path)
            .build()
            .expect("build");

        let status = engine.unload_model();
        assert!(status.unloaded);
        assert_eq!(status.previous_backend.as_deref(), Some("mock"));
        assert_eq!(
            status.previous_model_path.as_deref(),
            Some(model_path_text.as_str())
        );
        assert!(engine.active_backend().is_none());
        assert!(engine.model_path().is_none());
        assert!(engine.model.is_none());
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

        assert_eq!(
            engine.active_plugin(PluginKind::HardwareBackend),
            Some("cpu")
        );
    }

    #[test]
    fn engine_activation_materializes_wasm_runtime() {
        let root_dir = temp_model_path("plugin-wasm")
            .parent()
            .expect("parent")
            .join("plugins");
        fs::create_dir_all(root_dir.join("sampler")).expect("mkdir");
        fs::write(root_dir.join("sampler").join("runtime.wasm"), WASM_MAGIC).expect("write wasm");
        fs::write(
            root_dir.join("sampler").join("manifest.toml"),
            r#"
name = "sampler"
version = "0.1.0"
api_version = "1.0"
kind = "model_loader"

[runtime]
wasm_path = "runtime.wasm"
"#,
        )
        .expect("write manifest");

        let mut engine = InferenceEngine::builder().build().expect("build");
        engine
            .load_plugin_manifest_file(root_dir.join("sampler").join("manifest.toml"))
            .expect("register plugin");
        engine.activate_plugin("sampler").expect("activate plugin");

        let status = engine
            .runtime_snapshot()
            .plugins
            .into_iter()
            .find(|plugin| plugin.name == "sampler")
            .expect("status");
        assert!(status.declares_runtime);
        assert!(status.runtime_loaded);
        assert_eq!(status.runtime_kind, Some(PluginRuntimeKind::Wasm));
    }

    #[test]
    fn engine_rejects_invalid_wasm_runtime() {
        let root_dir = temp_model_path("plugin-invalid-wasm")
            .parent()
            .expect("parent")
            .join("plugins");
        fs::create_dir_all(root_dir.join("broken")).expect("mkdir");
        fs::write(root_dir.join("broken").join("runtime.wasm"), b"not-wasm").expect("write wasm");
        fs::write(
            root_dir.join("broken").join("manifest.toml"),
            r#"
name = "broken"
version = "0.1.0"
api_version = "1.0"
kind = "hardware_backend"

[runtime]
wasm_path = "runtime.wasm"
"#,
        )
        .expect("write manifest");

        let mut engine = InferenceEngine::builder().build().expect("build");
        engine
            .load_plugin_manifest_file(root_dir.join("broken").join("manifest.toml"))
            .expect("register plugin");

        let err = engine.activate_plugin("broken").expect_err("should reject");
        assert!(err
            .to_string()
            .contains("runtime is not a valid wasm module"));
    }
}
