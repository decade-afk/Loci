use crate::config::EngineConfig;
use crate::error::{LociError, Result};
use crate::model_registry::ModelRegistry;
use crate::planner::{build_plan, choose_backend, merge_topologies};
use crate::router::select_model;
use crate::snapshot::{
    EngineFeatureSnapshot, ModelPoolSnapshot, RoutingSnapshot, RuntimeConfigSnapshot,
    RuntimeSnapshot,
};
use loci_protocol::{
    Backend, BackendExecutionProfile, BackendOutput, ExecutionPlan, HardwareTopology, KvCachePlan,
    ModelDescriptor, PreparedModel, SessionRequest, SessionResponse, TieredOffloadPlan,
    TieredOffloadProfile,
};
#[cfg(test)]
use std::time::{Duration, Instant};

pub struct InferenceEngineBuilder {
    config: EngineConfig,
    preferred_backend: Option<String>,
    models: Vec<ModelDescriptor>,
}

impl InferenceEngineBuilder {
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
            preferred_backend: None,
            models: Vec::new(),
        }
    }

    pub fn config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn preferred_backend(mut self, backend: impl Into<String>) -> Self {
        self.preferred_backend = Some(backend.into());
        self
    }

    pub fn model(mut self, model: ModelDescriptor) -> Self {
        self.models.push(model);
        self
    }

    pub fn build(self) -> Result<InferenceEngine> {
        let config = self.config;
        let backends = builtin_backends();
        if backends.is_empty() {
            return Err(LociError::NoBackendAvailable);
        }

        let topology = merge_topologies(&backends);
        let features = EngineFeatureSnapshot {
            openvino: cfg!(feature = "openvino"),
            candle: cfg!(feature = "candle"),
            tiered_offload: cfg!(feature = "tiered-offload") && config.tiered_offload.enabled,
            paged_kv: cfg!(feature = "paged-kv") && config.paged_kv.enabled,
            power_aware: cfg!(feature = "power-aware"),
            dynamic_routing: cfg!(feature = "dynamic-routing") && config.routing.enabled,
        };

        Ok(InferenceEngine {
            config: config.clone(),
            preferred_backend: self.preferred_backend,
            registry: ModelRegistry::new(
                self.models,
                config.routing.max_loaded_models,
                config.model_keep_alive_secs,
                config.model_aliases.clone(),
            ),
            backends,
            topology,
            features,
            last_routed_model: None,
        })
    }
}

pub struct InferenceEngine {
    config: EngineConfig,
    preferred_backend: Option<String>,
    registry: ModelRegistry,
    backends: Vec<Box<dyn Backend>>,
    topology: HardwareTopology,
    features: EngineFeatureSnapshot,
    last_routed_model: Option<String>,
}

impl InferenceEngine {
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    pub fn register_model(&mut self, model: ModelDescriptor) {
        self.evict_expired_models();
        let model_name = model.name.clone();
        self.registry.register(model);
        self.touch_model_pool(&model_name);
    }

    pub fn unregister_model(&mut self, name: &str) -> bool {
        let resolved_name = self
            .resolve_registered_name(name)
            .unwrap_or_else(|| name.to_string());
        let removed = self.registry.unregister(&resolved_name);
        self.evict_model(&resolved_name);
        if self.last_routed_model.as_deref() == Some(resolved_name.as_str()) {
            self.last_routed_model = None;
        }
        removed
    }

    pub fn evict_model(&mut self, name: &str) -> bool {
        let resolved_name = self
            .resolve_registered_name(name)
            .unwrap_or_else(|| name.to_string());
        self.registry.evict(&resolved_name)
    }

    pub fn evict_expired_models(&mut self) -> Vec<String> {
        self.registry.evict_expired()
    }

    pub fn models(&self) -> Vec<ModelDescriptor> {
        self.registry.descriptors()
    }

    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        let alias = alias.into();
        let target = target.into();
        self.config
            .model_aliases
            .insert(alias.clone(), target.clone());
        self.registry.register_alias(alias, target);
    }

    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.config.model_aliases.remove(alias).is_some() || self.registry.remove_alias(alias)
    }

    pub fn set_model_keep_alive_secs(&mut self, keep_alive_secs: u64) {
        self.config.model_keep_alive_secs = keep_alive_secs;
        self.registry.set_keep_alive_secs(keep_alive_secs);
    }

    pub fn set_offload_profile(&mut self, profile: TieredOffloadProfile) {
        self.config.tiered_offload.profile = profile;
    }

    pub fn set_kv_block_size_tokens(&mut self, block_size_tokens: u32) {
        self.config.paged_kv.block_size_tokens = block_size_tokens;
    }

    pub fn set_kv_prefix_cache_enabled(&mut self, enabled: bool) {
        self.config.paged_kv.prefix_cache_enabled = enabled;
    }

    pub fn set_kv_types(&mut self, type_k: String, type_v: String) {
        self.config.paged_kv.type_k = type_k;
        self.config.paged_kv.type_v = type_v;
    }

    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            backends: self
                .backends
                .iter()
                .map(|backend| backend.descriptor())
                .collect(),
            topology: self.topology.clone(),
            models: self.registry.descriptors(),
            preferred_backend: self.preferred_backend.clone(),
            config: RuntimeConfigSnapshot {
                model_keep_alive_secs: self.config.model_keep_alive_secs,
                model_aliases: self.registry.aliases(),
                tiered_offload_enabled: self.config.tiered_offload.enabled,
                tiered_offload_profile: self.config.tiered_offload.profile,
                spill_threshold_bytes: self.config.tiered_offload.spill_threshold_bytes,
                max_disk_bytes: self.config.tiered_offload.max_disk_bytes,
                kv_cache_enabled: self.config.paged_kv.enabled,
                kv_block_size_tokens: self.config.paged_kv.block_size_tokens,
                kv_page_size_bytes: self.config.paged_kv.page_size_bytes,
                kv_prefix_cache_enabled: self.config.paged_kv.prefix_cache_enabled,
                kv_type_k: self.config.paged_kv.type_k.clone(),
                kv_type_v: self.config.paged_kv.type_v.clone(),
            },
            routing: RoutingSnapshot {
                enabled: self.config.routing.enabled,
                strategy: self.config.routing.strategy.clone(),
                max_loaded_models: self.config.routing.max_loaded_models,
            },
            model_pool: ModelPoolSnapshot {
                registered_models: self.registry.model_count(),
                resident_models: self.registry.resident_models(),
                prepared_models: self.registry.prepared_models(),
                resident_memory_bytes: self.registry.resident_memory_bytes(),
                resident_budget_bytes: self.resident_budget_bytes(),
                keep_alive_secs: self.registry.keep_alive_secs(),
                max_loaded_models: self.config.routing.max_loaded_models,
                last_routed_model: self.last_routed_model.clone(),
            },
            features: self.features.clone(),
        }
    }

    pub fn plan(&self, request: &SessionRequest) -> Result<ExecutionPlan> {
        let resolved_request = self.resolve_request_model(request)?;
        let models = self.registry.descriptors();
        let (model, route) = select_model(
            &models,
            &resolved_request,
            &self.config.routing,
            &self.topology,
        )?;
        let backend = choose_backend(&self.backends, model, self.preferred_backend.as_deref())?;
        let kv_cache = self.build_kv_cache_plan(model);
        let tiered_offload = self.build_tiered_offload_plan(model);

        Ok(build_plan(
            &self.config,
            &backend.descriptor(),
            &self.topology,
            model,
            &resolved_request,
            route,
            kv_cache,
            tiered_offload,
        ))
    }

    pub fn infer(&mut self, request: SessionRequest) -> Result<SessionResponse> {
        self.evict_expired_models();
        let (plan, model, backend_index, prepared) = self.prepare_request(&request)?;
        let model_name = model.name.clone();
        let backend = self.backends[backend_index].as_ref();

        let BackendOutput { text, telemetry } = backend
            .execute(&prepared, &model, &request, &plan)
            .map_err(|error| LociError::Backend(error.message))?;

        Ok(SessionResponse {
            text,
            backend: plan.backend.clone(),
            model: model_name,
            plan,
            telemetry,
        })
    }

    pub fn prepare(&mut self, request: SessionRequest) -> Result<PreparedModel> {
        self.evict_expired_models();
        let (_plan, _model, _backend_index, prepared) = self.prepare_request(&request)?;
        Ok(prepared)
    }

    fn build_tiered_offload_plan(&self, model: &ModelDescriptor) -> Option<TieredOffloadPlan> {
        #[cfg(feature = "tiered-offload")]
        {
            if self.config.tiered_offload.enabled {
                let manager = loci_tiered_offload::TieredOffloadManager::new(
                    self.config.tiered_offload.clone(),
                );
                return manager.plan(model, &self.topology);
            }
        }

        let _ = model;
        None
    }

    fn build_kv_cache_plan(&self, model: &ModelDescriptor) -> KvCachePlan {
        #[cfg(feature = "paged-kv")]
        {
            if self.config.paged_kv.enabled {
                let planner = loci_paged_kv::PagedKvPlanner::new(self.config.paged_kv.clone());
                return planner.plan(model, &self.topology, self.registry.model_count());
            }
        }

        KvCachePlan {
            strategy: "contiguous".to_string(),
            shared_across_models: false,
            page_size_bytes: None,
            block_size_tokens: None,
            max_cache_bytes: model.memory_bytes.map(|value| value / 8),
            type_k: None,
            type_v: None,
            tiered: false,
        }
    }

    fn touch_model_pool(&mut self, model_name: &str) {
        self.registry.touch(model_name);
        self.enforce_resident_memory_budget();
    }

    fn ensure_prepared_model(
        &mut self,
        backend_index: usize,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> Result<PreparedModel> {
        let backend_name = &plan.backend;
        let session_key = session_key(plan);

        if let Some(existing) = self
            .registry
            .prepared(&model.name, backend_name, session_key)
        {
            return Ok(existing);
        }

        let prepared = self.backends[backend_index]
            .prepare(model, plan)
            .map_err(|error| LociError::Backend(error.message))?;
        self.registry.set_prepared(prepared.clone());
        Ok(prepared)
    }

    fn prepare_request(
        &mut self,
        request: &SessionRequest,
    ) -> Result<(ExecutionPlan, ModelDescriptor, usize, PreparedModel)> {
        let plan = self.plan(request)?;
        let model = self
            .registry
            .find(&plan.route.selected_model)
            .cloned()
            .ok_or_else(|| LociError::RequestedModelMissing(plan.route.selected_model.clone()))?;
        let model_name = model.name.clone();
        self.touch_model_pool(&model_name);
        self.last_routed_model = Some(model_name);
        let backend_index = self
            .backends
            .iter()
            .position(|candidate| candidate.descriptor().name == plan.backend)
            .ok_or(LociError::NoBackendAvailable)?;
        let prepared = self.ensure_prepared_model(backend_index, &model, &plan)?;
        Ok((plan, model, backend_index, prepared))
    }

    fn resolve_request_model(&self, request: &SessionRequest) -> Result<SessionRequest> {
        let Some(target) = &request.target_model else {
            return Ok(request.clone());
        };

        let resolved = self
            .resolve_registered_name(target)
            .ok_or_else(|| LociError::RequestedModelMissing(target.clone()))?;

        let mut normalized = request.clone();
        normalized.target_model = Some(resolved);
        Ok(normalized)
    }

    fn resolve_registered_name(&self, name: &str) -> Option<String> {
        self.registry.resolve_name(name)
    }

    fn resident_budget_bytes(&self) -> u64 {
        self.topology
            .devices
            .iter()
            .filter(|device| device.kind != loci_protocol::AcceleratorKind::Disk)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>()
            .saturating_mul(3)
            / 4
    }

    fn enforce_resident_memory_budget(&mut self) {
        let _ = self.registry.enforce_limits(self.resident_budget_bytes());
    }
}

fn builtin_backends() -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = Vec::new();

    #[cfg(feature = "openvino")]
    backends.push(loci_backend_openvino::boxed_backend());

    #[cfg(feature = "candle")]
    backends.push(loci_backend_candle::boxed_backend());

    backends
}

fn session_key(plan: &ExecutionPlan) -> &str {
    match &plan.backend_profile {
        BackendExecutionProfile::OpenVino(profile) => &profile.session_key,
        BackendExecutionProfile::Candle(profile) => &profile.session_key,
        BackendExecutionProfile::Generic(profile) => &profile.session_key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "dynamic-routing")]
    use loci_protocol::{PowerState, RoutingConfig};
    use loci_protocol::{SessionRequest, ThermalState};
    use std::path::PathBuf;

    fn demo_model(name: &str, memory_bytes: u64, parameter_count: u64) -> ModelDescriptor {
        ModelDescriptor {
            name: name.to_string(),
            path: PathBuf::from(format!("D:/models/{name}.gguf")),
            architecture: "llama".to_string(),
            memory_bytes: Some(memory_bytes),
            parameter_count: Some(parameter_count),
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    #[test]
    fn engine_prefers_npu_decode_when_openvino_is_available() {
        let engine = InferenceEngine::builder()
            .model(demo_model("tiny", 2 * 1024 * 1024 * 1024, 1_000_000_000))
            .build()
            .expect("engine");

        let plan = engine
            .plan(&SessionRequest {
                prompt: "hello".to_string(),
                max_tokens: 64,
                temperature: 0.2,
                target_model: None,
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(plan.backend, "openvino");
        assert!(plan.placements.iter().any(|placement| {
            placement.stage == loci_protocol::PipelineStage::Decode
                && placement.target == loci_protocol::AcceleratorKind::Npu
        }));
        assert_eq!(
            engine.runtime_snapshot().topology.power.thermal_state,
            ThermalState::Nominal
        );
    }

    #[test]
    fn register_model_replaces_existing_descriptor_with_same_name() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        engine.register_model(demo_model("demo", 2, 3));

        let models = engine.models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].memory_bytes, Some(2));
        assert_eq!(models[0].parameter_count, Some(3));
        assert_eq!(
            engine.runtime_snapshot().model_pool.resident_models,
            vec!["demo"]
        );
        assert!(engine
            .runtime_snapshot()
            .model_pool
            .prepared_models
            .is_empty());
        assert!(engine.runtime_snapshot().model_pool.resident_budget_bytes > 0);
    }

    #[test]
    fn runtime_snapshot_exposes_alias_and_planner_configuration() {
        let mut config = EngineConfig::default();
        config.model_keep_alive_secs = 42;
        config
            .model_aliases
            .insert("tiny".to_string(), "demo".to_string());
        config.tiered_offload.profile = loci_protocol::TieredOffloadProfile::DiskHeavy;
        config.paged_kv.block_size_tokens = 32;
        config.paged_kv.type_k = "q8_0".to_string();
        config.paged_kv.type_v = "q4_0".to_string();

        let engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.config.model_keep_alive_secs, 42);
        assert_eq!(
            snapshot
                .config
                .model_aliases
                .get("tiny")
                .map(String::as_str),
            Some("demo")
        );
        assert_eq!(
            snapshot.config.tiered_offload_profile,
            loci_protocol::TieredOffloadProfile::DiskHeavy
        );
        assert_eq!(snapshot.config.kv_block_size_tokens, 32);
        assert_eq!(snapshot.config.kv_type_k, "q8_0");
        assert_eq!(snapshot.config.kv_type_v, "q4_0");
    }

    #[test]
    fn runtime_config_can_be_updated_after_build() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        engine.register_alias("tiny", "demo");
        engine.set_model_keep_alive_secs(77);
        engine.set_offload_profile(TieredOffloadProfile::GpuResident);
        engine.set_kv_block_size_tokens(64);
        engine.set_kv_prefix_cache_enabled(false);
        engine.set_kv_types("q8_0".to_string(), "q4_0".to_string());

        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.config.model_keep_alive_secs, 77);
        assert_eq!(
            snapshot
                .config
                .model_aliases
                .get("tiny")
                .map(String::as_str),
            Some("demo")
        );
        assert_eq!(
            snapshot.config.tiered_offload_profile,
            TieredOffloadProfile::GpuResident
        );
        assert_eq!(snapshot.config.kv_block_size_tokens, 64);
        assert!(!snapshot.config.kv_prefix_cache_enabled);
        assert_eq!(snapshot.config.kv_type_k, "q8_0");
        assert_eq!(snapshot.config.kv_type_v, "q4_0");
    }

    #[test]
    fn unregister_model_removes_existing_entry() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        assert!(engine.unregister_model("demo"));
        assert!(engine.models().is_empty());
        assert!(!engine.unregister_model("missing"));
    }

    #[test]
    fn evict_and_unregister_accept_alias_resolution() {
        let mut config = EngineConfig::default();
        config
            .model_aliases
            .insert("tiny".to_string(), "demo".to_string());

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        engine
            .prepare(SessionRequest {
                prompt: "warmup".to_string(),
                max_tokens: 1,
                temperature: 0.0,
                target_model: Some("tiny".to_string()),
                structured_output: false,
                tool_calling: false,
            })
            .expect("prepared");

        assert!(engine.evict_model("tiny"));
        assert!(engine.unregister_model("tiny"));
        assert!(engine.models().is_empty());
    }

    #[test]
    fn model_pool_tracks_recent_models_with_capacity_limit() {
        let mut config = EngineConfig::default();
        config.routing.max_loaded_models = Some(2);

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("a", 1, 1))
            .model(demo_model("b", 1, 1))
            .build()
            .expect("engine");

        engine.register_model(demo_model("c", 1, 1));

        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.model_pool.resident_models, vec!["b", "c"]);
        assert!(snapshot.model_pool.prepared_models.is_empty());
    }

    #[test]
    fn infer_prepares_and_tracks_backend_session() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        let response = engine
            .infer(SessionRequest {
                prompt: "hello".to_string(),
                max_tokens: 32,
                temperature: 0.2,
                target_model: Some("demo".to_string()),
                structured_output: false,
                tool_calling: false,
            })
            .expect("response");

        assert_eq!(response.backend, "openvino");
        let prepared = &engine.runtime_snapshot().model_pool.prepared_models;
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].model_name, "demo");
        assert_eq!(prepared[0].backend, "openvino");
        assert!(engine.runtime_snapshot().model_pool.resident_memory_bytes > 0);
    }

    #[test]
    fn prepare_warms_model_without_running_inference() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        let prepared = engine
            .prepare(SessionRequest {
                prompt: "warmup".to_string(),
                max_tokens: 1,
                temperature: 0.0,
                target_model: Some("demo".to_string()),
                structured_output: false,
                tool_calling: false,
            })
            .expect("prepared");

        assert_eq!(prepared.model_name, "demo");
        assert_eq!(
            engine.runtime_snapshot().model_pool.prepared_models.len(),
            1
        );
    }

    #[test]
    fn evict_model_drops_resident_and_prepared_state_but_keeps_registration() {
        let mut engine = InferenceEngine::builder()
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        engine
            .prepare(SessionRequest {
                prompt: "warmup".to_string(),
                max_tokens: 1,
                temperature: 0.0,
                target_model: Some("demo".to_string()),
                structured_output: false,
                tool_calling: false,
            })
            .expect("prepared");

        assert!(engine.evict_model("demo"));
        assert_eq!(engine.models().len(), 1);
        assert!(engine
            .runtime_snapshot()
            .model_pool
            .resident_models
            .is_empty());
        assert!(engine
            .runtime_snapshot()
            .model_pool
            .prepared_models
            .is_empty());
    }

    #[test]
    fn expired_models_are_evicted_using_keep_alive_policy() {
        let mut config = EngineConfig::default();
        config.model_keep_alive_secs = 1;

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        engine.register_model(demo_model("demo", 1, 1));
        engine
            .registry
            .mark_last_used_for_test("demo", Instant::now() - Duration::from_secs(5));

        let evicted = engine.evict_expired_models();
        assert_eq!(evicted, vec!["demo".to_string()]);
        assert!(engine
            .runtime_snapshot()
            .model_pool
            .resident_models
            .is_empty());
    }

    #[cfg(feature = "dynamic-routing")]
    #[test]
    fn engine_routes_simple_prompts_to_smaller_models() {
        let mut config = EngineConfig::default();
        config.routing = RoutingConfig {
            enabled: true,
            max_loaded_models: Some(2),
            strategy: loci_protocol::RoutingStrategy::PromptComplexity,
        };

        let engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("small", 1, 1))
            .model(demo_model("large", 10, 10))
            .build()
            .expect("engine");

        let plan = engine
            .plan(&SessionRequest {
                prompt: "hi".to_string(),
                max_tokens: 8,
                temperature: 0.2,
                target_model: None,
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(plan.route.selected_model, "small");
    }

    #[test]
    fn engine_resolves_target_model_alias_before_planning() {
        let mut config = EngineConfig::default();
        config
            .model_aliases
            .insert("tiny".to_string(), "demo".to_string());

        let engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("demo", 1, 1))
            .build()
            .expect("engine");

        let plan = engine
            .plan(&SessionRequest {
                prompt: "hello".to_string(),
                max_tokens: 8,
                temperature: 0.2,
                target_model: Some("tiny".to_string()),
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(plan.route.selected_model, "demo");
    }

    #[cfg(feature = "dynamic-routing")]
    #[test]
    fn power_aware_routing_prefers_smaller_model_under_thermal_pressure() {
        let mut config = EngineConfig::default();
        config.routing = RoutingConfig {
            enabled: true,
            max_loaded_models: Some(2),
            strategy: loci_protocol::RoutingStrategy::PowerAware,
        };

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("small", 1, 1))
            .model(demo_model("large", 10, 10))
            .build()
            .expect("engine");

        engine.topology.power = PowerState {
            battery_powered: true,
            battery_percent: Some(10),
            thermal_state: ThermalState::Hot,
            power_budget_watts: Some(15),
        };

        let plan = engine
            .plan(&SessionRequest {
                prompt: "summarize".to_string(),
                max_tokens: 256,
                temperature: 0.2,
                target_model: None,
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(plan.route.selected_model, "small");
    }
}
