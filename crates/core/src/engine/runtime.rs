use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model, ModelMetadata};
use crate::core::CoreRegistry;
use crate::engine::types::{GenerationParams, ModelInfo};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::{
    discover_plugin_manifest_files, load_plugin_manifest_file, RegisteredPlugin, SamplingHook,
};
use loci_plugin_api::{CoreComponent, PlatformTrack};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct InferenceEngine {
    pub(crate) registry: Box<dyn CoreRegistry>,
    pub(crate) backend_registry: BackendRegistry,
    pub(crate) active_backend: Option<String>,
    pub(crate) model: Option<Box<dyn Model>>,
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) default_inference_params: InferenceParams,
}

impl InferenceEngine {
    pub fn builder() -> crate::engine::InferenceEngineBuilder {
        crate::engine::InferenceEngineBuilder::new()
    }

    pub fn register_plugin(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        self.registry
            .plugin_manager_mut()
            .register(plugin)
            .map_err(LociError::from)?;
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
            .sampling_runtime()
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
            .map_err(LociError::from)
    }

    pub fn active_core_rewriter(&self, component: CoreComponent) -> Option<&str> {
        self.registry.active_core_rewriter(component)
    }

    pub fn load_plugin_manifest_file<P: AsRef<Path>>(&mut self, manifest_path: P) -> Result<()> {
        let plugin = load_plugin_manifest_file(manifest_path).map_err(LociError::from)?;
        self.register_plugin(plugin)
    }

    pub fn load_plugins_from_dir<P: AsRef<Path>>(&mut self, plugin_dir: P) -> Result<usize> {
        let manifests = discover_plugin_manifest_files(plugin_dir).map_err(LociError::from)?;
        let mut loaded = 0usize;
        for manifest in manifests {
            self.load_plugin_manifest_file(&manifest)?;
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
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| LociError::InferenceError("no model loaded".to_string()))?;
        model.infer_text(prompt, params)
    }

    pub fn generate_legacy(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = self.generation_params_to_inference(params);
        self.generate(prompt, &inference_params)
    }

    fn refresh_model_sampling_runtime(&mut self) -> Result<()> {
        if let Some(model) = self.model.as_mut() {
            let runtime = self.registry.plugin_manager().sampling_runtime();
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

    pub fn model_metadata(&self) -> Option<ModelMetadata> {
        self.model.as_ref().map(|model| model.metadata())
    }

    pub fn model_info(&self) -> Option<ModelInfo> {
        self.model_metadata().map(|metadata| ModelInfo {
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DefaultCoreRegistry;
    use crate::sampler::LogitsView;
    use loci_plugin_api::{ContributionPoints, CoreRewriters, PluginManifest};
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
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "agent-workflow".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                target_tracks: vec![PlatformTrack::AiAgent],
                contributes: ContributionPoints {
                    model_providers: vec!["rag-local".to_string()],
                    ..Default::default()
                },
                core_rewriters: CoreRewriters {
                    workflow: true,
                    ..Default::default()
                },
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

        engine
            .activate_core_rewriter(CoreComponent::Workflow, "agent-workflow")
            .expect("activate");
        assert_eq!(
            engine.active_core_rewriter(CoreComponent::Workflow),
            Some("agent-workflow")
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
        };

        let loaded = engine.load_plugins_from_dir(&dir).expect("load plugins");
        assert_eq!(loaded, 1);
        assert_eq!(engine.plugin_count(), 1);
        assert_eq!(
            engine.plugins_for_track(PlatformTrack::AiAgent),
            vec!["agent-plugin".to_string()]
        );

        let _ = fs::remove_dir_all(&dir);
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
    fn engine_propagates_sampling_runtime_into_loaded_model() {
        let mut engine = InferenceEngine {
            registry: Box::new(DefaultCoreRegistry::default()),
            backend_registry: BackendRegistry::with_builtin_backends(),
            active_backend: None,
            model: None,
            model_path: None,
            default_inference_params: InferenceParams::default(),
        };

        engine
            .register_plugin(RegisteredPlugin::new(PluginManifest {
                name: "hooked-plugin".to_string(),
                version: "1.0.0".to_string(),
                api_version: "1.0".to_string(),
                target_tracks: vec![PlatformTrack::AiInfra],
                contributes: ContributionPoints::default(),
                core_rewriters: CoreRewriters::default(),
            }))
            .expect("register");
        engine
            .load_model("mock", "demo.gguf", BackendParams::default())
            .expect("load model");
        engine
            .register_sampling_hook("hooked-plugin", Arc::new(ForceTokenHook))
            .expect("register hook");

        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("hooks=1"));
        assert_eq!(engine.sampling_hook_count(), 1);
    }
}
