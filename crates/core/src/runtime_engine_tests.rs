use super::*;
#[cfg(feature = "gguf")]
use loci_gguf::GGUF_MAGIC;
#[cfg(feature = "dynamic-routing")]
use loci_protocol::{PowerState, RoutingConfig};
use loci_protocol::{SessionRequest, ThermalState};
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
fn infer_triggers_execution_time_tiered_prefetch_for_disk_backed_models() {
    let mut config = EngineConfig::default();
    config.tiered_offload.spill_threshold_bytes = Some(1);
    config.tiered_offload.max_disk_bytes = Some(16 * 1024 * 1024);
    config.tiered_offload.prefetch_window_bytes = Some(128 * 1024);

    let mut engine = InferenceEngine::builder()
        .config(config)
        .model(demo_model(
            "oversized-infer",
            40 * 1024 * 1024 * 1024,
            20_000_000_000,
        ))
        .build()
        .expect("engine");

    let before = engine.runtime_snapshot();
    let before_runtime = before
        .tiered_offload_runtime
        .expect("tiered offload runtime snapshot before infer");
    assert!(before_runtime.sessions.is_empty());

    engine
        .infer(SessionRequest {
            prompt: "hello".to_string(),
            max_tokens: 64,
            temperature: 0.0,
            target_model: Some("oversized-infer".to_string()),
            images: Vec::new(),
            structured_output: false,
            tool_calling: false,
        })
        .expect("response");

    std::thread::sleep(std::time::Duration::from_millis(50));

    let after = engine.runtime_snapshot();
    let runtime = after
        .tiered_offload_runtime
        .expect("tiered offload runtime snapshot after infer");
    assert_eq!(runtime.sessions.len(), 1);
    assert_eq!(runtime.sessions[0].model_name, "oversized-infer");
    assert!(runtime.sessions[0].mapped_bytes > 0);
    assert!(runtime.sessions[0].scheduled_prefetch_requests >= 2);
    assert!(runtime.sessions[0].completed_prefetch_requests >= 2);
    assert!(runtime.sessions[0].prefetched_bytes > 0);
    assert!(runtime.total_prefetched_bytes >= runtime.sessions[0].prefetched_bytes);
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
            "oversized-prepare",
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
            target_model: Some("oversized-prepare".to_string()),
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
    assert_eq!(runtime.sessions[0].model_name, "oversized-prepare");
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
