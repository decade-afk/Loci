use crate::backend::{BackendParams, BackendRegistry, InferenceParams, Model, ModelMetadata};
use crate::core::CoreRegistry;
use crate::engine::types::{GenerationParams, ModelInfo};
use crate::error::{LociError, Result};
use crate::model::{ModelConfig, ModelLoadStrategy};
use crate::plugin::{
    discover_plugin_bundle_files, load_plugin_bundle_file, load_plugin_manifest_file,
    RegisteredPlugin, SamplingHook,
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
            self.registry
                .activate_core_rewriter(component, &plugin_name)
                .map_err(LociError::from)?;
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

    pub fn activate_legacy_text_plugin(&mut self, plugin_name: &str) -> Result<()> {
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

            (
                plugin.manifest.name.clone(),
                plugin.manifest.version.clone(),
                plugin.legacy_runtime_path().map(str::to_string),
                plugin.manifest.compatibility.legacy_capabilities.clone(),
                plugin.legacy_text_compat_runtime(),
            )
        };

        if self.active_legacy_text_plugins.contains(plugin_name) {
            return Ok(());
        }

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
                    "plugin `{plugin_name}` does not provide a supported legacy text compat runtime"
                ))
            })?
        };

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

        engine
            .activate_core_rewriter(CoreComponent::Inference, "hooked-plugin")
            .expect("activate inference rewriter");

        let output = engine
            .generate("hello", &InferenceParams::default())
            .expect("generate");
        assert!(output.contains("hooks=1"));
        assert_eq!(engine.sampling_hook_count(), 1);
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
