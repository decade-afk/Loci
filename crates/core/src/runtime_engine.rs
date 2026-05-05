//! Main runtime entry point for planning, preparation, and inference.

use crate::config::EngineConfig;
use crate::embedded::{infer_model_descriptor_from_path, EmbeddedModelRegistration};
use crate::error::{LociError, Result};
use crate::host_profiler::profile_host_capabilities;
use crate::kernel_registry::KernelRegistry;
use crate::model_inspector::{inspect_model, inspect_models};
use crate::model_registry::ModelRegistry;
use crate::planner::{build_plan, choose_backend, merge_topologies};
use crate::router::select_model;
use crate::snapshot::{
    EngineFeatureSnapshot, HostCapabilitySnapshot, ModelPoolSnapshot, RoutingSnapshot,
    RuntimeConfigSnapshot, RuntimeSnapshot, TieredOffloadRuntimeSnapshot,
    TieredOffloadSessionSnapshot,
};
use loci_protocol::{
    Backend, BackendExecutionProfile, BackendOutput, ExecutionPlan, HardwareTopology, KvCachePlan,
    ModelDescriptor, ModelReadinessReport, PreparedModel, SessionRequest, SessionResponse,
    TieredOffloadPlan, TieredOffloadProfile,
};
#[cfg(feature = "tiered-offload")]
use loci_tiered_offload::{HostTieringHints, SpillTensorKind, TieredOffloadRuntime};
use std::collections::HashMap;
#[cfg(test)]
use std::time::{Duration, Instant};

/// Builder for constructing an [`InferenceEngine`] with static backends.
pub struct InferenceEngineBuilder {
    config: EngineConfig,
    preferred_backend: Option<String>,
    models: Vec<ModelDescriptor>,
}

impl InferenceEngineBuilder {
    /// Starts a builder with default runtime configuration.
    pub fn new() -> Self {
        Self {
            config: EngineConfig::default(),
            preferred_backend: None,
            models: Vec::new(),
        }
    }

    /// Replaces the default runtime configuration.
    pub fn config(mut self, config: EngineConfig) -> Self {
        self.config = config;
        self
    }

    /// Requests that planning prefer a specific backend when possible.
    pub fn preferred_backend(mut self, backend: impl Into<String>) -> Self {
        self.preferred_backend = Some(backend.into());
        self
    }

    /// Registers a model that should be available when the engine starts.
    pub fn model(mut self, model: ModelDescriptor) -> Self {
        self.models.push(model);
        self
    }

    /// Registers a local model path and infers missing descriptor fields.
    pub fn local_model(
        mut self,
        path: impl Into<std::path::PathBuf>,
        options: EmbeddedModelRegistration,
    ) -> Result<Self> {
        let model = infer_model_descriptor_from_path(path, options)?;
        self.models.push(model);
        Ok(self)
    }

    /// Finalizes the engine and discovers the merged backend topology.
    pub fn build(self) -> Result<InferenceEngine> {
        let config = self.config;
        if config.routing.enabled && !cfg!(feature = "dynamic-routing") {
            return Err(LociError::InvalidRequest(
                "dynamic routing was requested, but the `dynamic-routing` feature is not enabled"
                    .to_string(),
            ));
        }
        let backends = builtin_backends();
        if backends.is_empty() {
            return Err(LociError::NoBackendAvailable);
        }

        let kernel_registry = KernelRegistry::from_backends(&backends);
        let topology = merge_topologies(&backends);
        let host = profile_host_capabilities();
        let features = feature_snapshot(&config);

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
            kernel_registry,
            host,
            topology,
            features,
            #[cfg(feature = "tiered-offload")]
            tiered_offload_runtime: TieredOffloadRuntime::default(),
            last_routed_model: None,
        })
    }
}

/// Owns the Loci control plane for model management and heterogeneous planning.
pub struct InferenceEngine {
    config: EngineConfig,
    preferred_backend: Option<String>,
    registry: ModelRegistry,
    backends: Vec<Box<dyn Backend>>,
    kernel_registry: KernelRegistry,
    host: HostCapabilitySnapshot,
    topology: HardwareTopology,
    features: EngineFeatureSnapshot,
    #[cfg(feature = "tiered-offload")]
    tiered_offload_runtime: TieredOffloadRuntime,
    last_routed_model: Option<String>,
}

impl InferenceEngine {
    /// Creates a builder for a new inference engine.
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Registers a model and marks it as recently used.
    pub fn register_model(&mut self, model: ModelDescriptor) {
        self.evict_expired_models();
        let model_name = model.name.clone();
        self.registry.register(model);
        self.touch_model_pool(&model_name);
    }

    /// Registers a local model path and infers missing descriptor fields.
    pub fn register_local_model(
        &mut self,
        path: impl Into<std::path::PathBuf>,
        options: EmbeddedModelRegistration,
    ) -> Result<ModelDescriptor> {
        let model = infer_model_descriptor_from_path(path, options)?;
        self.register_model(model.clone());
        Ok(model)
    }

    /// Removes a model registration, accepting aliases and fuzzy names.
    pub fn unregister_model(&mut self, name: &str) -> bool {
        let resolved_name = self
            .resolve_registered_name(name)
            .unwrap_or_else(|| name.to_string());
        self.drop_tiered_sessions_for_model(&resolved_name);
        let removed = self.registry.unregister(&resolved_name);
        if self.last_routed_model.as_deref() == Some(resolved_name.as_str()) {
            self.last_routed_model = None;
        }
        removed
    }

    /// Evicts runtime state for a model while keeping its registration.
    pub fn evict_model(&mut self, name: &str) -> bool {
        let resolved_name = self
            .resolve_registered_name(name)
            .unwrap_or_else(|| name.to_string());
        self.drop_tiered_sessions_for_model(&resolved_name);
        self.registry.evict(&resolved_name)
    }

    /// Evicts resident models that exceeded the configured keep-alive window.
    pub fn evict_expired_models(&mut self) -> Vec<String> {
        let prepared_sessions = self.prepared_sessions_by_model();
        let evicted = self.registry.evict_expired();
        self.drop_tiered_sessions(&prepared_sessions, &evicted);
        evicted
    }

    /// Returns the registered model descriptors.
    pub fn models(&self) -> Vec<ModelDescriptor> {
        self.registry.descriptors()
    }

    /// Registers a case-insensitive alias for a model name.
    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        let alias = alias.into();
        let target = target.into();
        self.config
            .model_aliases
            .insert(alias.clone(), target.clone());
        self.registry.register_alias(alias, target);
    }

    /// Removes an alias from both the config snapshot and the registry index.
    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.config.model_aliases.remove(alias).is_some() || self.registry.remove_alias(alias)
    }

    /// Updates the model residency keep-alive timeout.
    pub fn set_model_keep_alive_secs(&mut self, keep_alive_secs: u64) {
        self.config.model_keep_alive_secs = keep_alive_secs;
        self.registry.set_keep_alive_secs(keep_alive_secs);
    }

    /// Updates the active tiered-offload profile.
    pub fn set_offload_profile(&mut self, profile: TieredOffloadProfile) {
        self.config.tiered_offload.profile = profile;
    }

    /// Updates the spill threshold that activates disk-backed tiering.
    pub fn set_spill_threshold_bytes(&mut self, spill_threshold_bytes: Option<u64>) {
        self.config.tiered_offload.spill_threshold_bytes = spill_threshold_bytes;
    }

    /// Updates the maximum disk budget available to the spill runtime.
    pub fn set_max_disk_bytes(&mut self, max_disk_bytes: Option<u64>) {
        self.config.tiered_offload.max_disk_bytes = max_disk_bytes;
    }

    /// Updates the spill prefetch window used by the disk tier.
    pub fn set_prefetch_window_bytes(&mut self, prefetch_window_bytes: Option<u64>) {
        self.config.tiered_offload.prefetch_window_bytes = prefetch_window_bytes;
    }

    /// Updates the planner-facing KV block size.
    pub fn set_kv_block_size_tokens(&mut self, block_size_tokens: u32) {
        self.config.paged_kv.block_size_tokens = block_size_tokens;
    }

    /// Enables or disables prefix-cache sharing in the paged KV planner.
    pub fn set_kv_prefix_cache_enabled(&mut self, enabled: bool) {
        self.config.paged_kv.prefix_cache_enabled = enabled;
    }

    /// Updates the planner-facing KV tensor formats.
    pub fn set_kv_types(&mut self, type_k: String, type_v: String) {
        self.config.paged_kv.type_k = type_k;
        self.config.paged_kv.type_v = type_v;
    }

    /// Enables or disables dynamic routing when the feature is compiled in.
    pub fn set_routing_enabled(&mut self, enabled: bool) -> Result<()> {
        if enabled && !cfg!(feature = "dynamic-routing") {
            return Err(LociError::InvalidRequest(
                "dynamic routing is unavailable because the `dynamic-routing` feature is not enabled"
                    .to_string(),
            ));
        }
        self.config.routing.enabled = enabled;
        self.refresh_feature_snapshot();
        Ok(())
    }

    /// Updates the routing strategy used when dynamic routing is enabled.
    pub fn set_routing_strategy(&mut self, strategy: loci_protocol::RoutingStrategy) -> Result<()> {
        if self.config.routing.enabled && !cfg!(feature = "dynamic-routing") {
            return Err(LociError::InvalidRequest(
                "routing strategy cannot be changed because the `dynamic-routing` feature is not enabled"
                    .to_string(),
            ));
        }
        self.config.routing.strategy = strategy;
        self.refresh_feature_snapshot();
        Ok(())
    }

    /// Updates the maximum number of resident models the registry may keep loaded.
    pub fn set_max_loaded_models(&mut self, max_loaded_models: Option<usize>) {
        self.config.routing.max_loaded_models = max_loaded_models;
        self.registry.set_max_loaded_models(max_loaded_models);
        self.enforce_resident_memory_budget();
    }

    /// Produces a serializable snapshot of the current runtime state.
    pub fn runtime_snapshot(&self) -> RuntimeSnapshot {
        let models = self.registry.descriptors();
        RuntimeSnapshot {
            backends: self
                .backends
                .iter()
                .map(|backend| backend.descriptor())
                .collect(),
            backend_assets: self
                .backends
                .iter()
                .map(|backend| backend.asset_capabilities())
                .collect(),
            backend_lowering: self
                .backends
                .iter()
                .map(|backend| backend.lowering_capabilities())
                .collect(),
            backend_kernels: self.kernel_registry.catalogs().to_vec(),
            host: self.host.clone(),
            topology: self.topology.clone(),
            model_diagnostics: inspect_models(&models, &self.backends),
            models,
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
            tiered_offload_runtime: self.tiered_offload_runtime_snapshot(),
            features: self.features.clone(),
        }
    }

    /// Produces readiness diagnostics for every registered model.
    pub fn inspect_models(&self) -> Vec<ModelReadinessReport> {
        inspect_models(&self.registry.descriptors(), &self.backends)
    }

    /// Produces readiness diagnostics for one registered model.
    pub fn inspect_model(&self, name: &str) -> Result<ModelReadinessReport> {
        let resolved_name = self
            .resolve_registered_name(name)
            .ok_or_else(|| LociError::RequestedModelMissing(name.to_string()))?;
        let model = self
            .registry
            .find(&resolved_name)
            .ok_or_else(|| LociError::RequestedModelMissing(resolved_name.clone()))?;
        Ok(inspect_model(model, &self.backends))
    }

    /// Computes an execution plan without preparing backend state or running inference.
    pub fn plan(&self, request: &SessionRequest) -> Result<ExecutionPlan> {
        let resolved_request = self.resolve_request_model(request)?;
        let models = self.registry.descriptors();
        let (model, route) = select_model(
            &models,
            &resolved_request,
            &self.config.routing,
            &self.topology,
        )?;
        let backend = choose_backend(
            &self.backends,
            model,
            &resolved_request,
            self.preferred_backend.as_deref(),
        )?;
        let kv_cache = self.build_kv_cache_plan(model);
        let tiered_offload = self.build_tiered_offload_plan(model);

        Ok(build_plan(
            &self.config,
            &backend.descriptor(),
            &backend.lowering_capabilities(),
            &self.topology,
            &self.host,
            model,
            &resolved_request,
            route,
            kv_cache,
            tiered_offload,
        ))
    }

    /// Plans and executes one inference request.
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

    /// Plans and prepares backend state without running generation.
    pub fn prepare(&mut self, request: SessionRequest) -> Result<PreparedModel> {
        self.evict_expired_models();
        let (_plan, _model, _backend_index, prepared) = self.prepare_request(&request)?;
        Ok(prepared)
    }

    /// Builds the tiered-offload policy layer when the feature is active.
    fn build_tiered_offload_plan(&self, model: &ModelDescriptor) -> Option<TieredOffloadPlan> {
        #[cfg(feature = "tiered-offload")]
        {
            if self.config.tiered_offload.enabled {
                let manager = loci_tiered_offload::TieredOffloadManager::new(
                    self.config.tiered_offload.clone(),
                );
                return manager.plan(model, &self.topology, Some(&self.tiering_hints()));
            }
        }

        let _ = model;
        None
    }

    /// Builds the planner-facing KV layout description for a model.
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

    /// Marks the model as resident and enforces global residency limits.
    fn touch_model_pool(&mut self, model_name: &str) {
        self.registry.touch(model_name);
        self.enforce_resident_memory_budget();
    }

    /// Reuses an existing prepared backend session or prepares a new one.
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

        self.prepare_tiered_offload_session(model, plan)?;
        let prepared = self.backends[backend_index]
            .prepare(model, plan)
            .map_err(|error| LociError::Backend(error.message))?;
        self.registry.set_prepared(prepared.clone());
        Ok(prepared)
    }

    /// Resolves the request into a concrete plan, model, backend, and prepared state.
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

    /// Normalizes an explicit target model through the registry alias layer.
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

    /// Resolves a user-facing model identifier into a registered name.
    fn resolve_registered_name(&self, name: &str) -> Option<String> {
        self.registry.resolve_name(name)
    }

    /// Estimates the total non-disk residency budget available to the runtime.
    fn resident_budget_bytes(&self) -> u64 {
        let topology_budget = self
            .topology
            .devices
            .iter()
            .filter(|device| device.kind != loci_protocol::AcceleratorKind::Disk)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>()
            .saturating_mul(3)
            / 4;
        let host_budget = self.host.available_memory_bytes.saturating_mul(3) / 4;
        if host_budget > 0 {
            topology_budget.min(host_budget)
        } else {
            topology_budget
        }
    }

    /// Applies the resident-memory budget by evicting old model state as needed.
    fn enforce_resident_memory_budget(&mut self) {
        let prepared_sessions = self.prepared_sessions_by_model();
        let evicted = self.registry.enforce_limits(self.resident_budget_bytes());
        self.drop_tiered_sessions(&prepared_sessions, &evicted);
    }

    /// Recomputes feature activation flags after runtime configuration changes.
    fn refresh_feature_snapshot(&mut self) {
        self.features = feature_snapshot(&self.config);
    }

    /// Converts the current host snapshot into tiered-offload planning hints.
    #[cfg(feature = "tiered-offload")]
    fn tiering_hints(&self) -> HostTieringHints {
        HostTieringHints {
            total_memory_bytes: self.host.total_memory_bytes,
            available_memory_bytes: self.host.available_memory_bytes,
            free_disk_bytes: self
                .host
                .disks
                .iter()
                .find(|disk| disk.mount_point.to_ascii_lowercase().starts_with('d'))
                .or_else(|| self.host.disks.first())
                .map(|disk| disk.available_bytes),
            disk_read_mbps: self.host.probe.disk_read_mbps,
            disk_write_mbps: self.host.probe.disk_write_mbps,
        }
    }

    /// Materializes the spill session for a prepared model when disk tiering is active.
    fn prepare_tiered_offload_session(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> Result<()> {
        #[cfg(feature = "tiered-offload")]
        {
            if let Some(tiered_plan) = &plan.tiered_offload {
                let assets = inspect_model(model, &self.backends).asset_inventory;
                self.tiered_offload_runtime
                    .prepare_session(session_key(plan), model, &assets, tiered_plan)
                    .map_err(|error| {
                        LociError::Backend(format!("tiered offload prepare failed: {error}"))
                    })?;
            }
        }

        let _ = (model, plan);
        Ok(())
    }

    /// Builds a snapshot of the active spill runtime when the feature is compiled in.
    fn tiered_offload_runtime_snapshot(&self) -> Option<TieredOffloadRuntimeSnapshot> {
        #[cfg(feature = "tiered-offload")]
        {
            let snapshot = self.tiered_offload_runtime.snapshot().ok()?;
            return Some(TieredOffloadRuntimeSnapshot {
                root_dir: snapshot.root_dir.display().to_string(),
                total_spill_bytes: snapshot.total_spill_bytes,
                total_prefetched_bytes: snapshot.total_prefetched_bytes,
                sessions: snapshot
                    .active_sessions
                    .into_iter()
                    .map(|session| {
                        let weights_bytes = segment_bytes(&session, SpillTensorKind::Weights);
                        let kv_cache_bytes = segment_bytes(&session, SpillTensorKind::KvCache);
                        let activations_bytes =
                            segment_bytes(&session, SpillTensorKind::Activations);

                        TieredOffloadSessionSnapshot {
                            session_key: session.session_key,
                            model_name: session.model_name,
                            spill_path: session.spill_path.display().to_string(),
                            mapped_bytes: session.mapped_bytes,
                            prefetched_bytes: session.prefetched_bytes,
                            scheduled_prefetch_requests: session.scheduled_prefetch_requests,
                            completed_prefetch_requests: session.completed_prefetch_requests,
                            weights_bytes,
                            kv_cache_bytes,
                            activations_bytes,
                        }
                    })
                    .collect(),
            });
        }

        #[cfg(not(feature = "tiered-offload"))]
        None
    }

    /// Captures the currently prepared session keys by model before registry eviction occurs.
    fn prepared_sessions_by_model(&self) -> HashMap<String, Vec<String>> {
        let mut sessions = HashMap::<String, Vec<String>>::new();
        for prepared in self.registry.prepared_models() {
            sessions
                .entry(prepared.model_name.clone())
                .or_default()
                .push(prepared.session_key);
        }
        sessions
    }

    /// Removes spill sessions for models that the registry just evicted.
    fn drop_tiered_sessions(
        &self,
        prepared_sessions: &HashMap<String, Vec<String>>,
        model_names: &[String],
    ) {
        #[cfg(feature = "tiered-offload")]
        {
            for model_name in model_names {
                if let Some(session_keys) = prepared_sessions.get(model_name) {
                    for session_key in session_keys {
                        let _ = self.tiered_offload_runtime.evict_session(session_key);
                    }
                }
            }
        }
    }

    /// Removes spill sessions for a single model before explicit eviction or unregister.
    fn drop_tiered_sessions_for_model(&self, model_name: &str) {
        let prepared_sessions = self.prepared_sessions_by_model();
        self.drop_tiered_sessions(&prepared_sessions, &[model_name.to_string()]);
    }
}

/// Instantiates the statically compiled backend set for this build.
fn builtin_backends() -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = Vec::new();

    #[cfg(feature = "candle")]
    backends.push(loci_backend_candle::boxed_backend());

    #[cfg(feature = "openvino")]
    backends.push(loci_backend_openvino::boxed_backend());

    backends
}

/// Extracts the backend-specific session key from an execution plan.
fn session_key(plan: &ExecutionPlan) -> &str {
    match &plan.backend_profile {
        BackendExecutionProfile::OpenVino(profile) => &profile.session_key,
        BackendExecutionProfile::Candle(profile) => &profile.session_key,
        BackendExecutionProfile::Generic(profile) => &profile.session_key,
    }
}

/// Computes the feature snapshot for the current configuration.
fn feature_snapshot(config: &EngineConfig) -> EngineFeatureSnapshot {
    EngineFeatureSnapshot {
        openvino: cfg!(feature = "openvino"),
        candle: cfg!(feature = "candle"),
        gguf: cfg!(feature = "gguf"),
        kernels_llama: cfg!(feature = "kernels-llama"),
        tiered_offload: cfg!(feature = "tiered-offload") && config.tiered_offload.enabled,
        paged_kv: cfg!(feature = "paged-kv") && config.paged_kv.enabled,
        power_aware: cfg!(feature = "power-aware"),
        dynamic_routing: cfg!(feature = "dynamic-routing") && config.routing.enabled,
        mobile: cfg!(feature = "mobile"),
        neon: cfg!(feature = "neon"),
        coreml: cfg!(feature = "coreml"),
        qnn: cfg!(feature = "qnn"),
    }
}

#[cfg(feature = "tiered-offload")]
fn segment_bytes(
    session: &loci_tiered_offload::TieredSessionSnapshot,
    tensor: SpillTensorKind,
) -> u64 {
    session
        .segments
        .iter()
        .filter(|segment| segment.tensor == tensor)
        .map(|segment| segment.length_bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "dynamic-routing")]
    use loci_protocol::{PowerState, RoutingConfig};
    use loci_protocol::{SessionRequest, ThermalState};
    #[cfg(feature = "gguf")]
    use loci_gguf::GGUF_MAGIC;
    #[cfg(feature = "gguf")]
    use std::fs;
    use std::path::PathBuf;
    #[cfg(feature = "gguf")]
    use std::time::{SystemTime, UNIX_EPOCH};

    fn demo_model(name: &str, memory_bytes: u64, parameter_count: u64) -> ModelDescriptor {
        ModelDescriptor {
            name: name.to_string(),
            path: demo_model_path(name),
            architecture: "llama".to_string(),
            memory_bytes: Some(memory_bytes),
            parameter_count: Some(parameter_count),
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    #[cfg(feature = "gguf")]
    fn demo_model_path(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-runtime-{name}-{suffix}.gguf"));
        write_minimal_gguf(&path);
        path
    }

    #[cfg(not(feature = "gguf"))]
    fn demo_model_path(name: &str) -> PathBuf {
        PathBuf::from(format!("D:/models/{name}.gguf"))
    }

    #[cfg(feature = "gguf")]
    fn write_minimal_gguf(path: &PathBuf) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"llama";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());

        write_tensor_info(&mut bytes, 3, "token_embd.weight", &[4], 0, 0);
        write_tensor_info(&mut bytes, 3, "blk.0.attn_norm.weight", &[4], 0, 16);
        write_tensor_info(&mut bytes, 3, "output.weight", &[4], 0, 32);

        bytes.extend_from_slice(&[0_u8; 32]);
        for value in 1..=12 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }

        fs::write(path, bytes).expect("gguf");
    }

    #[cfg(feature = "gguf")]
    fn write_tensor_info(
        bytes: &mut Vec<u8>,
        version: u32,
        name: &str,
        dimensions: &[u64],
        ggml_dtype: u32,
        offset: u64,
    ) {
        write_sized_string(bytes, version, name.as_bytes());
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions.iter().rev() {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&ggml_dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    #[cfg(feature = "gguf")]
    fn write_sized_string(bytes: &mut Vec<u8>, version: u32, value: &[u8]) {
        match version {
            1 => bytes.extend_from_slice(&(value.len() as u32).to_le_bytes()),
            2 | 3 => bytes.extend_from_slice(&(value.len() as u64).to_le_bytes()),
            other => panic!("unsupported test gguf version: {other}"),
        }
        bytes.extend_from_slice(value);
    }

    #[test]
    fn engine_prefers_best_available_decode_device_for_available_backend() {
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
                images: Vec::new(),
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(
            plan.backend,
            if cfg!(feature = "openvino") {
                "openvino"
            } else {
                "candle"
            }
        );
        let expected_target = if engine
            .runtime_snapshot()
            .topology
            .devices
            .iter()
            .any(|device| device.kind == loci_protocol::AcceleratorKind::Npu)
        {
            loci_protocol::AcceleratorKind::Npu
        } else {
            loci_protocol::AcceleratorKind::Gpu
        };
        assert!(plan.placements.iter().any(|placement| {
            placement.stage == loci_protocol::PipelineStage::Decode
                && placement.target == expected_target
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
        assert!(!snapshot.backend_assets.is_empty());
        assert!(!snapshot.backend_lowering.is_empty());
        assert!(snapshot.host.logical_cores >= 1);
        assert!(snapshot.host.total_memory_bytes >= snapshot.host.available_memory_bytes);
        assert_eq!(snapshot.config.model_keep_alive_secs, 42);
        assert_eq!(snapshot.model_diagnostics.len(), 1);
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
        engine.set_max_loaded_models(Some(2));

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
        assert_eq!(snapshot.routing.max_loaded_models, Some(2));
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
                images: Vec::new(),
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
    fn max_loaded_models_can_be_reduced_after_build() {
        let mut config = EngineConfig::default();
        config.routing.max_loaded_models = Some(3);

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model("a", 1, 1))
            .model(demo_model("b", 1, 1))
            .model(demo_model("c", 1, 1))
            .build()
            .expect("engine");

        engine.set_max_loaded_models(Some(1));

        let snapshot = engine.runtime_snapshot();
        assert_eq!(snapshot.routing.max_loaded_models, Some(1));
        assert_eq!(snapshot.model_pool.resident_models.len(), 1);
    }

    #[cfg(not(feature = "dynamic-routing"))]
    #[test]
    fn build_rejects_enabled_routing_without_feature() {
        let mut config = EngineConfig::default();
        config.routing.enabled = true;

        let error = match InferenceEngine::builder()
            .config(config)
            .model(demo_model("demo", 1, 1))
            .build()
        {
            Ok(_) => panic!("routing should be rejected"),
            Err(error) => error,
        };

        assert!(matches!(error, LociError::InvalidRequest(_)));
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
                images: Vec::new(),
                structured_output: false,
                tool_calling: false,
            })
            .expect("response");

        assert_eq!(
            response.backend,
            if cfg!(feature = "openvino") {
                "openvino"
            } else {
                "candle"
            }
        );
        let prepared = &engine.runtime_snapshot().model_pool.prepared_models;
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].model_name, "demo");
        assert_eq!(prepared[0].backend, response.backend);
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
                images: Vec::new(),
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
    fn prepare_materializes_tiered_offload_runtime_for_disk_backed_models() {
        let mut config = EngineConfig::default();
        config.tiered_offload.spill_threshold_bytes = Some(1);
        config.tiered_offload.max_disk_bytes = Some(16 * 1024 * 1024);
        config.tiered_offload.prefetch_window_bytes = Some(512 * 1024);

        let mut engine = InferenceEngine::builder()
            .config(config)
            .model(demo_model(
                "oversized",
                40 * 1024 * 1024 * 1024,
                20_000_000_000,
            ))
            .build()
            .expect("engine");

        engine
            .prepare(SessionRequest {
                prompt: "warmup".to_string(),
                max_tokens: 1,
                temperature: 0.0,
                target_model: Some("oversized".to_string()),
                images: Vec::new(),
                structured_output: false,
                tool_calling: false,
            })
            .expect("prepared");

        let snapshot = engine.runtime_snapshot();
        let runtime = snapshot
            .tiered_offload_runtime
            .expect("tiered offload runtime snapshot");
        assert_eq!(runtime.sessions.len(), 1);
        assert_eq!(runtime.sessions[0].model_name, "oversized");
        assert!(runtime.sessions[0].mapped_bytes > 0);
        assert!(runtime.sessions[0].weights_bytes > 0);
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
                images: Vec::new(),
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
                images: Vec::new(),
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
                images: Vec::new(),
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
                images: Vec::new(),
                structured_output: false,
                tool_calling: false,
            })
            .expect("plan");

        assert_eq!(plan.route.selected_model, "small");
    }
}
