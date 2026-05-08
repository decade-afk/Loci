use super::*;
use crate::config::EngineConfig;
use crate::snapshot::{HostCapabilitySnapshot, HostDiskSnapshot, HostProbeSnapshot};
use loci_protocol::{
    AcceleratorKind, Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
    BackendExecutionProfile, BackendLoweringCapabilities, BackendOutput, BackendResult,
    ChipOperatorClass, ExecutionArtifactKind, ExecutionPlan, KvCachePlan, ModelAssetLayout,
    OpenVinoExecutionProfile, PipelineStage, PreparedModel, RouteDecision, TieredOffloadPolicy,
    TieredOffloadProfile, TieredPlacementPercentages,
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
struct MockBackend {
    descriptor: BackendDescriptor,
    supports_model: bool,
}

impl Backend for MockBackend {
    fn descriptor(&self) -> BackendDescriptor {
        self.descriptor.clone()
    }

    fn asset_capabilities(&self) -> BackendAssetCapabilities {
        match self.descriptor.runtime_family {
            loci_protocol::BackendRuntimeFamily::OpenVino => BackendAssetCapabilities {
                backend: self.descriptor.name.clone(),
                runtime_family: self.descriptor.runtime_family,
                directly_supported_layouts: vec![
                    ModelAssetLayout::OpenVinoGenAiExport,
                    ModelAssetLayout::OpenVinoIr,
                    ModelAssetLayout::OpenVinoBlob,
                ],
                ingestible_layouts: vec![
                    ModelAssetLayout::OnnxModel,
                    ModelAssetLayout::GgufFile,
                    ModelAssetLayout::GgufDirectory,
                    ModelAssetLayout::SafeTensorsFile,
                    ModelAssetLayout::SafeTensorsDirectory,
                    ModelAssetLayout::PytorchBinFile,
                    ModelAssetLayout::PytorchCheckpointDirectory,
                    ModelAssetLayout::TransformersCheckpoint,
                    ModelAssetLayout::UnknownDirectory,
                    ModelAssetLayout::UnknownFile,
                ],
                preferred_artifact: ExecutionArtifactKind::OpenVinoIr,
                requires_lowering_for_execution: true,
                notes: Vec::new(),
            },
            loci_protocol::BackendRuntimeFamily::Candle => BackendAssetCapabilities {
                backend: self.descriptor.name.clone(),
                runtime_family: self.descriptor.runtime_family,
                directly_supported_layouts: vec![
                    ModelAssetLayout::GgufFile,
                    ModelAssetLayout::GgufDirectory,
                    ModelAssetLayout::SafeTensorsFile,
                    ModelAssetLayout::SafeTensorsDirectory,
                    ModelAssetLayout::PytorchBinFile,
                    ModelAssetLayout::PytorchCheckpointDirectory,
                    ModelAssetLayout::TransformersCheckpoint,
                ],
                ingestible_layouts: vec![
                    ModelAssetLayout::UnknownDirectory,
                    ModelAssetLayout::UnknownFile,
                ],
                preferred_artifact: ExecutionArtifactKind::NativeCheckpoint,
                requires_lowering_for_execution: false,
                notes: Vec::new(),
            },
            _ => BackendAssetCapabilities {
                backend: self.descriptor.name.clone(),
                runtime_family: self.descriptor.runtime_family,
                directly_supported_layouts: Vec::new(),
                ingestible_layouts: Vec::new(),
                preferred_artifact: ExecutionArtifactKind::RuntimeDefined,
                requires_lowering_for_execution: false,
                notes: Vec::new(),
            },
        }
    }

    fn discover_topology(&self) -> HardwareTopology {
        HardwareTopology::default()
    }

    fn supports_model(&self, _model: &ModelDescriptor) -> bool {
        self.supports_model
    }

    fn prepare(
        &self,
        _model: &ModelDescriptor,
        _plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel> {
        Err(BackendError {
            message: "unused in planner tests".to_string(),
        })
    }

    fn execute(
        &self,
        _prepared: &PreparedModel,
        _model: &ModelDescriptor,
        _request: &SessionRequest,
        _plan: &ExecutionPlan,
    ) -> BackendResult<BackendOutput> {
        Err(BackendError {
            message: "unused in planner tests".to_string(),
        })
    }
}

fn backend(name: &str, supports_npu: bool) -> BackendDescriptor {
    BackendDescriptor {
        name: name.to_string(),
        runtime_family: match name {
            "openvino" => loci_protocol::BackendRuntimeFamily::OpenVino,
            "candle" => loci_protocol::BackendRuntimeFamily::Candle,
            _ => loci_protocol::BackendRuntimeFamily::Generic,
        },
        supports_cpu: true,
        supports_gpu: true,
        supports_npu,
        supports_disk_tiering: true,
        supports_paged_kv: true,
        supports_multimodal: name == "openvino",
    }
}

fn lowering(name: &str, supports_npu: bool) -> BackendLoweringCapabilities {
    BackendLoweringCapabilities {
        backend: name.to_string(),
        runtime_family: match name {
            "openvino" => loci_protocol::BackendRuntimeFamily::OpenVino,
            "candle" => loci_protocol::BackendRuntimeFamily::Candle,
            _ => loci_protocol::BackendRuntimeFamily::Generic,
        },
        granularity: if supports_npu {
            loci_protocol::LoweringGranularity::Subgraph
        } else {
            loci_protocol::LoweringGranularity::Graph
        },
        supports_real_execution: name == "openvino",
        supports_graph_partitioning: supports_npu,
        supports_layer_affinity: false,
        supports_dynamic_reoffload: supports_npu,
        supports_custom_operators: false,
        operator_classes: vec![
            loci_protocol::ChipOperatorClass::Attention,
            loci_protocol::ChipOperatorClass::Mlp,
            loci_protocol::ChipOperatorClass::KvCache,
        ],
        notes: Vec::new(),
    }
}

fn topology() -> HardwareTopology {
    HardwareTopology {
        devices: vec![
            DeviceDescriptor {
                id: "cpu:0".to_string(),
                name: "cpu".to_string(),
                kind: AcceleratorKind::Cpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(16 * 1024 * 1024 * 1024),
                compute_units: Some(16),
                power_watts: Some(20.0),
            },
            DeviceDescriptor {
                id: "gpu:0".to_string(),
                name: "gpu".to_string(),
                kind: AcceleratorKind::Gpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(8 * 1024 * 1024 * 1024),
                compute_units: Some(128),
                power_watts: Some(30.0),
            },
            DeviceDescriptor {
                id: "npu:0".to_string(),
                name: "npu".to_string(),
                kind: AcceleratorKind::Npu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(2 * 1024 * 1024 * 1024),
                compute_units: Some(1),
                power_watts: Some(5.0),
            },
            DeviceDescriptor {
                id: "disk:0".to_string(),
                name: "disk".to_string(),
                kind: AcceleratorKind::Disk,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(256 * 1024 * 1024 * 1024),
                compute_units: None,
                power_watts: None,
            },
        ],
        power: PowerState {
            battery_powered: false,
            battery_percent: None,
            thermal_state: ThermalState::Nominal,
            power_budget_watts: Some(45),
        },
    }
}

fn model() -> ModelDescriptor {
    ModelDescriptor {
        name: "demo".to_string(),
        path: PathBuf::from("D:/models/demo.gguf"),
        architecture: "llama".to_string(),
        memory_bytes: Some(16 * 1024 * 1024 * 1024),
        parameter_count: Some(8_000_000_000),
        context_length: Some(8192),
        preferred_backend: None,
    }
}

fn request() -> SessionRequest {
    SessionRequest {
        prompt: "hello".to_string(),
        max_tokens: 128,
        temperature: 0.2,
        target_model: Some("demo".to_string()),
        images: Vec::new(),
        structured_output: false,
        tool_calling: false,
    }
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("loci-planner-{label}-{suffix}"))
}

fn route() -> RouteDecision {
    RouteDecision {
        selected_model: "demo".to_string(),
        reason: "explicit".to_string(),
        alternatives: Vec::new(),
    }
}

fn host() -> HostCapabilitySnapshot {
    HostCapabilitySnapshot {
        target_family: std::env::consts::FAMILY.to_string(),
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        mobile_class: false,
        host_name: Some("test-host".to_string()),
        os_name: Some("test-os".to_string()),
        os_version: Some("1".to_string()),
        kernel_version: Some("1".to_string()),
        cpu_brand: Some("cpu".to_string()),
        cpu_vendor: Some("vendor".to_string()),
        cpu_frequency_mhz: Some(1000),
        physical_cores: Some(4),
        logical_cores: 8,
        total_memory_bytes: 16 * 1024 * 1024 * 1024,
        available_memory_bytes: 8 * 1024 * 1024 * 1024,
        total_swap_bytes: 0,
        free_swap_bytes: 0,
        uptime_secs: 1,
        load_average_one: 0.0,
        load_average_five: 0.0,
        load_average_fifteen: 0.0,
        disks: vec![HostDiskSnapshot {
            name: "disk".to_string(),
            mount_point: "D:\\".to_string(),
            file_system: "NTFS".to_string(),
            total_bytes: 256 * 1024 * 1024 * 1024,
            available_bytes: 128 * 1024 * 1024 * 1024,
            is_removable: false,
        }],
        probe: HostProbeSnapshot {
            cpu_scalar_gops: 1.0,
            memory_bandwidth_gbps: 10.0,
            disk_read_mbps: 2000.0,
            disk_write_mbps: 1500.0,
            probe_bytes: 16 * 1024 * 1024,
            probe_duration_ms: 10,
        },
    }
}

#[test]
fn build_plan_uses_disk_for_kv_when_disk_heavy_policy_demands_it() {
    let plan = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        Some(TieredOffloadPlan {
            spill_bytes: 8 << 30,
            prefetch_window_bytes: 128 << 20,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::DiskHeavy,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 10,
                    cpu_percent: 30,
                    disk_percent: 60,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 10,
                    cpu_percent: 30,
                    disk_percent: 60,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 40,
                    cpu_percent: 60,
                    disk_percent: 0,
                },
                cpu_cache_compute: true,
                compress_weights: true,
                compress_kv_cache: true,
            },
        }),
    );

    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::KvCache
            && placement.target == AcceleratorKind::Disk
            && placement.device_id.as_deref() == Some("disk:0")
    }));
    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::Weights && placement.target == AcceleratorKind::Disk
    }));
}

#[test]
fn build_plan_keeps_kv_on_cpu_when_balanced_policy_avoids_disk_dominance() {
    let plan = build_plan(
        &EngineConfig::default(),
        &backend("candle", false),
        &lowering("candle", false),
        &topology(),
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        Some(TieredOffloadPlan {
            spill_bytes: 4 << 30,
            prefetch_window_bytes: 128 << 20,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::Balanced,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 30,
                    cpu_percent: 50,
                    disk_percent: 20,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 60,
                    disk_percent: 20,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 60,
                    cpu_percent: 40,
                    disk_percent: 0,
                },
                cpu_cache_compute: true,
                compress_weights: false,
                compress_kv_cache: false,
            },
        }),
    );

    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::KvCache
            && placement.target == AcceleratorKind::Cpu
            && placement.device_id.as_deref() == Some("cpu:0")
    }));
    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::Weights
            && placement.target == AcceleratorKind::Cpu
            && placement.device_id.as_deref() == Some("cpu:0")
    }));
    assert!(matches!(
        plan.backend_profile,
        BackendExecutionProfile::Candle(_)
    ));
}

#[test]
fn build_plan_assigns_weight_placement_without_disk_tiering() {
    let plan = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: false,
        },
        None,
    );

    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::Weights
            && placement.target == AcceleratorKind::Gpu
            && placement.device_id.as_deref() == Some("gpu:0")
            && placement.memory_bytes == model().memory_bytes
    }));
}

#[test]
fn openvino_session_key_is_scoped_per_model_name() {
    let mut model_a = model();
    model_a.name = "demo-a".to_string();
    let mut model_b = model();
    model_b.name = "demo-b".to_string();

    let plan_a = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &host(),
        &model_a,
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged".to_string(),
            shared_across_models: false,
            page_size_bytes: None,
            block_size_tokens: None,
            max_cache_bytes: None,
            type_k: None,
            type_v: None,
            tiered: false,
        },
        None,
    );
    let plan_b = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &host(),
        &model_b,
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged".to_string(),
            shared_across_models: false,
            page_size_bytes: None,
            block_size_tokens: None,
            max_cache_bytes: None,
            type_k: None,
            type_v: None,
            tiered: false,
        },
        None,
    );

    let BackendExecutionProfile::OpenVino(profile_a) = plan_a.backend_profile else {
        panic!("expected openvino profile for model a");
    };
    let BackendExecutionProfile::OpenVino(profile_b) = plan_b.backend_profile else {
        panic!("expected openvino profile for model b");
    };

    assert_ne!(profile_a.session_key, profile_b.session_key);
    assert!(profile_a.session_key.contains("demo-a"));
    assert!(profile_b.session_key.contains("demo-b"));
}

#[test]
fn build_plan_rebalances_kv_and_weights_under_power_pressure() {
    let mut constrained_topology = topology();
    constrained_topology.power = PowerState {
        battery_powered: true,
        battery_percent: Some(10),
        thermal_state: ThermalState::Hot,
        power_budget_watts: Some(15),
    };

    let plan = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &constrained_topology,
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        Some(TieredOffloadPlan {
            spill_bytes: 4 << 30,
            prefetch_window_bytes: 128 << 20,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::Balanced,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 45,
                    cpu_percent: 35,
                    disk_percent: 20,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 60,
                    cpu_percent: 30,
                    disk_percent: 10,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 50,
                    cpu_percent: 50,
                    disk_percent: 0,
                },
                cpu_cache_compute: false,
                compress_weights: false,
                compress_kv_cache: false,
            },
        }),
    );

    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::KvCache
            && placement.target == AcceleratorKind::Cpu
            && placement.device_id.as_deref() == Some("cpu:0")
    }));
    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::Weights
            && placement.target == AcceleratorKind::Cpu
            && placement.device_id.as_deref() == Some("cpu:0")
    }));
    assert!(matches!(
        plan.backend_profile,
        BackendExecutionProfile::OpenVino(OpenVinoExecutionProfile {
            dynamic_reoffload: true,
            ..
        })
    ));
}

#[test]
fn build_plan_prefers_disk_for_cold_state_when_host_memory_is_tight() {
    let mut tight_host = host();
    tight_host.available_memory_bytes = 1024 * 1024 * 1024;

    let plan = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &tight_host,
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        Some(TieredOffloadPlan {
            spill_bytes: 8 << 30,
            prefetch_window_bytes: 128 << 20,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::Balanced,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 35,
                    cpu_percent: 35,
                    disk_percent: 30,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 50,
                    disk_percent: 30,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 60,
                    cpu_percent: 40,
                    disk_percent: 0,
                },
                cpu_cache_compute: true,
                compress_weights: true,
                compress_kv_cache: true,
            },
        }),
    );

    assert!(plan.placements.iter().any(|placement| {
        placement.stage == PipelineStage::Weights && placement.target == AcceleratorKind::Disk
    }));
}

#[test]
fn build_plan_emits_backend_lowering_guidance() {
    let plan = build_plan(
        &EngineConfig::default(),
        &backend("openvino", true),
        &lowering("openvino", true),
        &topology(),
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        None,
    );

    let lowering = plan.lowering_plan.expect("lowering plan");
    assert_eq!(lowering.backend, "openvino");
    assert!(lowering.subgraphs.iter().any(|subgraph| {
        subgraph.id == "decode_attention_block" && subgraph.affinity_tag.as_deref() == Some("NPU")
    }));
    assert!(!lowering.partitions.is_empty());
    assert!(!lowering.operators.is_empty());
    assert!(lowering.partitions.iter().any(|partition| {
        partition.affinity_tag.as_deref() == Some("NPU")
            && partition
                .operator_classes
                .contains(&ChipOperatorClass::Attention)
    }));
    assert!(lowering.operators.iter().any(|operator| {
        operator.subgraph == "decode_attention_block"
            && operator.partition.starts_with("partition-")
    }));
    assert!(lowering.subgraphs.iter().any(|subgraph| {
        subgraph.id == "kv_state_region" && subgraph.operator_class == ChipOperatorClass::KvCache
    }));
}

#[test]
fn build_plan_dispatches_backend_profile_by_runtime_family() {
    let plan = build_plan(
        &EngineConfig::default(),
        &BackendDescriptor {
            name: "intel-openvino-main".to_string(),
            runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: true,
            supports_disk_tiering: true,
            supports_paged_kv: true,
            supports_multimodal: true,
        },
        &BackendLoweringCapabilities {
            backend: "intel-openvino-main".to_string(),
            runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
            granularity: loci_protocol::LoweringGranularity::Subgraph,
            supports_real_execution: true,
            supports_graph_partitioning: true,
            supports_layer_affinity: false,
            supports_dynamic_reoffload: true,
            supports_custom_operators: false,
            operator_classes: vec![
                loci_protocol::ChipOperatorClass::Attention,
                loci_protocol::ChipOperatorClass::Mlp,
            ],
            notes: Vec::new(),
        },
        &topology(),
        &host(),
        &model(),
        &request(),
        route(),
        KvCachePlan {
            strategy: "paged-prefix-cache".to_string(),
            shared_across_models: true,
            page_size_bytes: Some(1 << 20),
            block_size_tokens: Some(16),
            max_cache_bytes: Some(512 << 20),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            tiered: true,
        },
        None,
    );

    assert!(matches!(
        plan.backend_profile,
        BackendExecutionProfile::OpenVino(_)
    ));
}

#[test]
fn choose_backend_skips_non_multimodal_backends_for_image_requests() {
    let dir = unique_temp_dir("multimodal-openvino-ready");
    fs::create_dir_all(&dir).expect("dir");
    fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

    let backends: Vec<Box<dyn Backend>> = vec![
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "candle".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: false,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: false,
            },
            supports_model: true,
        }),
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "openvino".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: true,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: true,
            },
            supports_model: true,
        }),
    ];
    let multimodal_model = ModelDescriptor {
        name: "demo".to_string(),
        path: dir.clone(),
        architecture: "minicpm-v".to_string(),
        memory_bytes: Some(1),
        parameter_count: Some(1),
        context_length: Some(128),
        preferred_backend: Some("candle".to_string()),
    };

    let mut multimodal_request = request();
    multimodal_request
        .images
        .push(loci_protocol::ImageInput::Path {
            path: PathBuf::from("D:/images/demo.png"),
        });

    let selected = choose_backend(
        &backends,
        &multimodal_model,
        &multimodal_request,
        Some("candle"),
    )
    .expect("backend");

    assert_eq!(selected.descriptor().name, "openvino");
    fs::remove_dir_all(dir).expect("cleanup");
}

#[test]
fn choose_backend_prefers_ready_backend_over_partial_fallbacks() {
    let dir = unique_temp_dir("openvino-ready");
    fs::create_dir_all(&dir).expect("dir");
    fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

    let model = ModelDescriptor {
        name: "demo".to_string(),
        path: dir.clone(),
        architecture: "llama".to_string(),
        memory_bytes: Some(1),
        parameter_count: Some(1),
        context_length: Some(128),
        preferred_backend: Some("candle".to_string()),
    };

    let backends: Vec<Box<dyn Backend>> = vec![
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "candle".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: false,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: false,
            },
            supports_model: true,
        }),
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "openvino".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: true,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: true,
            },
            supports_model: true,
        }),
    ];

    let selected = choose_backend(&backends, &model, &request(), Some("candle")).expect("backend");

    assert_eq!(selected.descriptor().name, "openvino");
    fs::remove_dir_all(dir).expect("cleanup");
}

#[test]
fn choose_backend_uses_readiness_instead_of_supports_model_heuristics() {
    let dir = unique_temp_dir("readiness-over-supports-model");
    fs::create_dir_all(&dir).expect("dir");
    fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

    let model = ModelDescriptor {
        name: "demo".to_string(),
        path: dir.clone(),
        architecture: "llama".to_string(),
        memory_bytes: Some(1),
        parameter_count: Some(1),
        context_length: Some(128),
        preferred_backend: None,
    };

    let backends: Vec<Box<dyn Backend>> = vec![
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "openvino".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: true,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: true,
            },
            supports_model: false,
        }),
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "candle".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: false,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: false,
            },
            supports_model: true,
        }),
    ];

    let selected = choose_backend(&backends, &model, &request(), None).expect("backend");
    assert_eq!(selected.descriptor().name, "openvino");

    fs::remove_dir_all(dir).expect("cleanup");
}

#[test]
fn choose_backend_rejects_non_ready_candle_checkpoint_paths() {
    let file = unique_temp_dir("torch-checkpoint").with_extension("pt");
    fs::write(&file, "weights").expect("weights");

    let model = ModelDescriptor {
        name: "demo".to_string(),
        path: file.clone(),
        architecture: "llama".to_string(),
        memory_bytes: Some(1),
        parameter_count: Some(1),
        context_length: Some(128),
        preferred_backend: Some("candle".to_string()),
    };

    let backends: Vec<Box<dyn Backend>> = vec![Box::new(MockBackend {
        descriptor: BackendDescriptor {
            name: "candle".to_string(),
            runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: false,
            supports_disk_tiering: true,
            supports_paged_kv: true,
            supports_multimodal: false,
        },
        supports_model: true,
    })];

    let error = match choose_backend(&backends, &model, &request(), Some("candle")) {
        Ok(_) => panic!("non-ready backend should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, LociError::NoCompatibleBackend { .. }));

    fs::remove_file(file).expect("cleanup");
}
