use loci_gguf::{
    read_gguf_header, read_gguf_metadata_summary, support_profile as gguf_support_profile,
    GgufMetadataSummary, GgufTensorInfoSummary,
};
use loci_kernels_llama::{curated_kernel_descriptors, rms_norm_f32, rope_f32};
use loci_protocol::{
    AcceleratorKind, Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
    BackendExecutionProfile, BackendKernelCatalog, BackendLoweringCapabilities, BackendOutput,
    BackendResult, BackendRuntimeFamily, BackendTelemetry, CandleExecutionProfile,
    CandleTensorResidency, ChipOperatorClass, DeviceDescriptor, ExecutionArtifactKind,
    ExecutionPlan, HardwareTopology, KernelDescriptor, KernelImplementationKind, KernelMaturity,
    KernelOrigin, LoweringGranularity, ModelAssetLayout, ModelDescriptor, ModelFormat,
    PipelineStage, PlacementDecision, PowerState, PreparedModel, PreparedResidency, SessionRequest,
    ThermalState,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::Mutex;

pub fn boxed_backend() -> Box<dyn Backend> {
    Box::new(CandleBackend::default())
}

#[derive(Default)]
struct CandleBackend {
    runtime: CandleRuntime,
}

#[derive(Default)]
struct CandleRuntime {
    prepared_sessions: Mutex<HashMap<String, PreparedSessionArtifact>>,
}

#[derive(Debug, Clone)]
struct PreparedSessionArtifact {
    prepared_summary: String,
    prepared_fingerprint: String,
    architecture: Option<String>,
    general_name: Option<String>,
    context_length: Option<u32>,
    tensor_count: Option<u64>,
    metadata_count: Option<u64>,
    alignment: Option<u64>,
    tensor_data_offset: Option<u64>,
    preview_tensors: Vec<String>,
    first_tensor: Option<TensorProbe>,
    last_tensor: Option<TensorProbe>,
    max_tensor_rank: u32,
    attention_tensor_count: u32,
    ffn_tensor_count: u32,
    norm_tensor_count: u32,
    contains_output_weight: bool,
    contains_token_embedding: bool,
    file_probe: FileProbe,
}

#[derive(Debug, Clone)]
struct TensorProbe {
    name: String,
    rank: u32,
    elements: u64,
    ggml_dtype: u32,
    offset: u64,
}

#[derive(Debug, Clone)]
struct FileProbe {
    prefix_len: usize,
    rolling_checksum: u64,
    byte_histogram: [u32; 4],
}

impl Default for FileProbe {
    fn default() -> Self {
        Self {
            prefix_len: 0,
            rolling_checksum: 0,
            byte_histogram: [0; 4],
        }
    }
}

impl Backend for CandleBackend {
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "candle".to_string(),
            runtime_family: BackendRuntimeFamily::Candle,
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: false,
            supports_disk_tiering: true,
            supports_paged_kv: true,
            supports_multimodal: false,
        }
    }

    fn asset_capabilities(&self) -> BackendAssetCapabilities {
        BackendAssetCapabilities {
            backend: "candle".to_string(),
            runtime_family: BackendRuntimeFamily::Candle,
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
            notes: vec![
                "Candle should eventually execute native checkpoint-style assets without an OpenVINO export step".to_string(),
                "the current backend shape is still partial even for directly supported layouts".to_string(),
            ],
        }
    }

    fn lowering_capabilities(&self) -> BackendLoweringCapabilities {
        BackendLoweringCapabilities {
            backend: "candle".to_string(),
            runtime_family: BackendRuntimeFamily::Candle,
            granularity: LoweringGranularity::Tensor,
            supports_real_execution: false,
            supports_graph_partitioning: false,
            supports_layer_affinity: false,
            supports_dynamic_reoffload: false,
            supports_custom_operators: false,
            operator_classes: vec![
                ChipOperatorClass::Attention,
                ChipOperatorClass::Matmul,
                ChipOperatorClass::Embedding,
                ChipOperatorClass::RmsNorm,
                ChipOperatorClass::Mlp,
                ChipOperatorClass::KvCache,
                ChipOperatorClass::Sampling,
            ],
            notes: vec![
                "the intended fallback shape is per-tensor placement in a pure Rust runtime"
                    .to_string(),
                "the current backend does not yet execute real Candle models".to_string(),
            ],
        }
    }

    fn kernel_catalog(&self) -> BackendKernelCatalog {
        let gguf_profile = gguf_support_profile();
        BackendKernelCatalog {
            backend: "candle".to_string(),
            runtime_family: BackendRuntimeFamily::Candle,
            kernels: {
                let mut kernels = vec![KernelDescriptor {
                    backend: "candle".to_string(),
                    kernel_name: "candle_attention_decode".to_string(),
                    operator_class: ChipOperatorClass::Attention,
                    implementation: KernelImplementationKind::Rust,
                    maturity: KernelMaturity::Planned,
                    origin: KernelOrigin {
                        project: "candle".to_string(),
                        component: "attention".to_string(),
                        license: Some("MIT OR Apache-2.0".to_string()),
                        notes: vec![
                            "default pure Rust decode path".to_string(),
                            "intended to cooperate with imported llama.cpp hotspots".to_string(),
                        ],
                    },
                    supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
                    supported_formats: vec![
                        ModelFormat::Gguf,
                        ModelFormat::SafeTensors,
                        ModelFormat::PytorchBin,
                        ModelFormat::Directory,
                    ],
                    supported_architectures: gguf_profile
                        .supported_architectures
                        .iter()
                        .map(|arch| (*arch).to_string())
                        .collect(),
                    dispatch_keys: vec!["decode".to_string(), "causal".to_string()],
                    notes: vec!["portable baseline path retained in Candle backend".to_string()],
                }];
                kernels.extend(curated_kernel_descriptors());
                kernels
            },
            notes: vec![
                "Candle is the default pure Rust execution path".to_string(),
                "llama.cpp-inspired hotspot metadata is sourced from loci-kernels-llama"
                    .to_string(),
            ],
        }
    }

    fn discover_topology(&self) -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "host-cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    compute_units: Some(16),
                    power_watts: Some(25.0),
                },
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "generic-gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(12 * 1024 * 1024 * 1024),
                    compute_units: Some(96),
                    power_watts: Some(35.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "nvme-tier".to_string(),
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
                power_budget_watts: Some(60),
            },
        }
    }

    fn supports_model(&self, model: &ModelDescriptor) -> bool {
        matches!(
            model.inferred_format(),
            ModelFormat::Gguf
                | ModelFormat::SafeTensors
                | ModelFormat::PytorchBin
                | ModelFormat::Directory
        )
    }

    fn prepare(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel> {
        let profile = candle_profile(plan)?;
        self.runtime.prepare_session(model, plan, profile)
    }

    fn execute(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
    ) -> BackendResult<BackendOutput> {
        if request.prompt.trim().is_empty() {
            return Err(BackendError {
                message: "prompt must not be empty".to_string(),
            });
        }

        let profile = candle_profile(plan)?;
        if prepared.session_key != profile.session_key {
            return Err(BackendError {
                message: format!(
                    "prepared Candle session `{}` does not match plan `{}`",
                    prepared.session_key, profile.session_key
                ),
            });
        }

        self.runtime
            .run_session(prepared, model, request, plan, profile)
    }
}

impl CandleRuntime {
    fn prepare_session(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
        profile: &CandleExecutionProfile,
    ) -> BackendResult<PreparedModel> {
        validate_candle_plan(plan, profile)?;
        let residency = derive_residency(plan, profile);
        let session_artifact = self.build_prepared_artifact(model)?;
        self.prepared_sessions
            .lock()
            .map_err(|_| BackendError {
                message: "Candle prepared session cache is poisoned".to_string(),
            })?
            .insert(profile.session_key.clone(), session_artifact);

        Ok(PreparedModel {
            model_name: model.name.clone(),
            backend: "candle".to_string(),
            session_key: profile.session_key.clone(),
            residency,
            estimated_memory_bytes: estimate_resident_memory_bytes(model, plan, residency),
        })
    }

    fn run_session(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
        profile: &CandleExecutionProfile,
    ) -> BackendResult<BackendOutput> {
        validate_candle_plan(plan, profile)?;
        if !request.images.is_empty() {
            return Err(BackendError {
                message: "Candle fallback does not yet support multimodal image inputs".to_string(),
            });
        }
        let spill = plan
            .tiered_offload
            .as_ref()
            .map(|tier| {
                format!(
                    "spill={}B profile={}",
                    tier.spill_bytes,
                    offload_profile_label(plan)
                )
            })
            .unwrap_or_else(|| "spill=0B".to_string());
        let prefill = placement_summary(plan, PipelineStage::Prefill);
        let decode = placement_summary(plan, PipelineStage::Decode);
        let kv = placement_summary(plan, PipelineStage::KvCache);
        let weights = placement_summary(plan, PipelineStage::Weights);
        let session_artifact = self
            .lookup_prepared_artifact(&prepared.session_key)?
            .ok_or_else(|| BackendError {
                message: format!(
                    "Candle execution is missing a prepared artifact for session `{}`",
                    prepared.session_key
                ),
            })?;
        let prompt_embedding =
            prompt_probe_embedding(request.prompt.trim(), model, Some(&session_artifact));
        let probe_weight = probe_weight(&prompt_embedding, Some(&session_artifact));
        let probe_norm =
            rms_norm_f32(&prompt_embedding, &probe_weight, 1e-5).map_err(|error| BackendError {
                message: format!("Candle RMSNorm probe failed: {error}"),
            })?;
        let rope_input = rope_probe_input(&prompt_embedding, Some(&session_artifact));
        let probe_rope = rope_f32(
            &rope_input,
            request.prompt.len(),
            10_000.0,
            rope_input.len(),
        )
        .map_err(|error| BackendError {
            message: format!("Candle RoPE probe failed: {error}"),
        })?;
        let norm_signature = probe_norm
            .iter()
            .take(4)
            .map(|value| format!("{value:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        let rope_signature = probe_rope
            .iter()
            .take(4)
            .map(|value| format!("{value:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        let gguf_probe = session_artifact.prepared_summary.clone();
        let tensor_probe = format!(
            "fingerprint={} preview_tensors={} first_tensor={} last_tensor={}",
            session_artifact.prepared_fingerprint,
            session_artifact.preview_tensors.join("|"),
            format_tensor_probe(session_artifact.first_tensor.as_ref()),
            format_tensor_probe(session_artifact.last_tensor.as_ref()),
        );
        let session_name = session_artifact
            .general_name
            .as_deref()
            .unwrap_or(model.name.as_str());

        Ok(BackendOutput {
            text: format!(
                "[candle:{}] prefill={} decode={} kv={} weights={} residency={:?} prepared={} {} {} {} rmsnorm=[{}] rope=[{}] prompt=`{}`",
                session_name,
                prefill,
                decode,
                kv,
                weights,
                profile.tensor_residency,
                prepared.session_key,
                spill,
                gguf_probe,
                tensor_probe,
                norm_signature,
                rope_signature,
                request.prompt.trim()
            ),
            telemetry: BackendTelemetry {
                estimated_prefill_ms: estimate_prefill_ms(profile, plan),
                estimated_decode_ms: estimate_decode_ms(profile, plan),
                generated_tokens: request.max_tokens.min(128),
            },
        })
    }

    fn build_prepared_artifact(
        &self,
        model: &ModelDescriptor,
    ) -> BackendResult<PreparedSessionArtifact> {
        let metadata = if model.inferred_format() == ModelFormat::Gguf {
            read_gguf_metadata_summary(&model.path).ok()
        } else {
            None
        };
        let file_probe = probe_model_file(&model.path).unwrap_or_default();
        Ok(build_prepared_artifact(model, metadata, file_probe))
    }

    fn lookup_prepared_artifact(
        &self,
        session_key: &str,
    ) -> BackendResult<Option<PreparedSessionArtifact>> {
        self.prepared_sessions
            .lock()
            .map_err(|_| BackendError {
                message: "Candle prepared session cache is poisoned".to_string(),
            })
            .map(|sessions| sessions.get(session_key).cloned())
    }
}

fn candle_profile(plan: &ExecutionPlan) -> BackendResult<&CandleExecutionProfile> {
    match &plan.backend_profile {
        BackendExecutionProfile::Candle(profile) => Ok(profile),
        _ => Err(BackendError {
            message: "execution plan is missing a Candle backend profile".to_string(),
        }),
    }
}

fn validate_candle_plan(
    plan: &ExecutionPlan,
    profile: &CandleExecutionProfile,
) -> BackendResult<()> {
    if plan
        .placements
        .iter()
        .any(|placement| placement.target == AcceleratorKind::Npu)
    {
        return Err(BackendError {
            message: "Candle fallback does not support NPU placements".to_string(),
        });
    }

    if profile.prefill_device.starts_with("npu:")
        || profile.decode_device.starts_with("npu:")
        || profile.kv_cache_device.starts_with("npu:")
    {
        return Err(BackendError {
            message: "Candle execution profile cannot reference NPU devices".to_string(),
        });
    }

    let uses_disk = plan
        .placements
        .iter()
        .any(|placement| placement.target == AcceleratorKind::Disk);
    if !uses_disk && matches!(profile.tensor_residency, CandleTensorResidency::Hybrid) {
        return Err(BackendError {
            message: "Candle hybrid residency requires a disk-backed placement".to_string(),
        });
    }

    if let Some(lowering_plan) = &plan.lowering_plan {
        if lowering_plan.backend != "candle" {
            return Err(BackendError {
                message: "Candle execution received a lowering plan for a different backend"
                    .to_string(),
            });
        }
        if lowering_plan
            .partitions
            .iter()
            .any(|partition| partition.target == AcceleratorKind::Npu)
        {
            return Err(BackendError {
                message: "Candle lowering guidance cannot target NPU partitions".to_string(),
            });
        }
        if lowering_plan
            .subgraphs
            .iter()
            .any(|subgraph| subgraph.target == AcceleratorKind::Npu)
        {
            return Err(BackendError {
                message: "Candle lowering guidance cannot target NPU regions".to_string(),
            });
        }
        if lowering_plan
            .operators
            .iter()
            .any(|operator| operator.target == AcceleratorKind::Npu)
        {
            return Err(BackendError {
                message: "Candle lowering guidance cannot target NPU operators".to_string(),
            });
        }
        if lowering_plan.operators.iter().any(|operator| {
            !lowering_plan
                .partitions
                .iter()
                .any(|partition| partition.id == operator.partition)
        }) {
            return Err(BackendError {
                message: "Candle lowering operator references a partition that does not exist"
                    .to_string(),
            });
        }
    }

    Ok(())
}

fn derive_residency(plan: &ExecutionPlan, profile: &CandleExecutionProfile) -> PreparedResidency {
    let weights_on_disk =
        placement_target(plan, PipelineStage::Weights) == Some(AcceleratorKind::Disk);
    let kv_on_disk = placement_target(plan, PipelineStage::KvCache) == Some(AcceleratorKind::Disk);

    match profile.tensor_residency {
        CandleTensorResidency::MemoryOnly => PreparedResidency::Memory,
        CandleTensorResidency::Hybrid => {
            if weights_on_disk && kv_on_disk {
                PreparedResidency::DiskBacked
            } else {
                PreparedResidency::Hybrid
            }
        }
    }
}

fn estimate_resident_memory_bytes(
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
    residency: PreparedResidency,
) -> Option<u64> {
    let model_bytes = model.memory_bytes?;
    let resident_weights = if let Some(tier) = &plan.tiered_offload {
        let memory_percent = 100u64.saturating_sub(tier.policy.weights.disk_percent as u64);
        model_bytes.saturating_mul(memory_percent) / 100
    } else {
        model_bytes
    };
    let kv_bytes = plan.kv_cache.max_cache_bytes.unwrap_or(0);
    let resident_kv = match placement_target(plan, PipelineStage::KvCache) {
        Some(AcceleratorKind::Disk) => kv_bytes / 3,
        Some(_) => kv_bytes,
        None => 0,
    };

    Some(match residency {
        PreparedResidency::Memory => resident_weights.saturating_add(resident_kv),
        PreparedResidency::Hybrid => resident_weights
            .saturating_mul(9)
            .saturating_div(10)
            .saturating_add(resident_kv),
        PreparedResidency::DiskBacked => resident_weights / 2 + resident_kv / 2,
    })
}

fn estimate_prefill_ms(profile: &CandleExecutionProfile, plan: &ExecutionPlan) -> u64 {
    if profile.prefill_device.starts_with("gpu:") {
        14 + weight_penalty(plan)
    } else {
        22 + weight_penalty(plan)
    }
}

fn estimate_decode_ms(profile: &CandleExecutionProfile, plan: &ExecutionPlan) -> u64 {
    let kv_penalty = match placement_target(plan, PipelineStage::KvCache) {
        Some(AcceleratorKind::Disk) => 6,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    };
    if profile.decode_device.starts_with("gpu:") {
        9 + kv_penalty
    } else {
        13 + kv_penalty
    }
}

fn weight_penalty(plan: &ExecutionPlan) -> u64 {
    match placement_target(plan, PipelineStage::Weights) {
        Some(AcceleratorKind::Disk) => 5,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    }
}

fn placement_target(plan: &ExecutionPlan, stage: PipelineStage) -> Option<AcceleratorKind> {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(|placement| placement.target)
}

fn placement_summary(plan: &ExecutionPlan, stage: PipelineStage) -> String {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(format_placement)
        .unwrap_or_else(|| "unassigned".to_string())
}

fn format_placement(placement: &PlacementDecision) -> String {
    let device = placement.device_id.as_deref().unwrap_or("none");
    format!("{}@{}", accelerator_label(placement.target), device)
}

fn accelerator_label(kind: AcceleratorKind) -> &'static str {
    match kind {
        AcceleratorKind::Cpu => "cpu",
        AcceleratorKind::Gpu => "gpu",
        AcceleratorKind::Npu => "npu",
        AcceleratorKind::Disk => "disk",
    }
}

fn offload_profile_label(plan: &ExecutionPlan) -> &'static str {
    match plan.tiered_offload.as_ref().map(|tier| tier.profile) {
        Some(loci_protocol::TieredOffloadProfile::Auto) => "auto",
        Some(loci_protocol::TieredOffloadProfile::GpuResident) => "gpu_resident",
        Some(loci_protocol::TieredOffloadProfile::Balanced) => "balanced",
        Some(loci_protocol::TieredOffloadProfile::DiskHeavy) => "disk_heavy",
        None => "none",
    }
}

fn build_prepared_artifact(
    model: &ModelDescriptor,
    metadata: Option<GgufMetadataSummary>,
    file_probe: FileProbe,
) -> PreparedSessionArtifact {
    let prepared_summary = if let Some(metadata) = metadata.as_ref() {
        format!(
            "gguf=v{} tensors={} metadata={} arch={} ctx={} align={} data_offset={} attn={} ffn={} norm={} checksum={:016x}",
            metadata.header.version,
            metadata.header.tensor_count,
            metadata.header.metadata_count,
            metadata
                .architecture
                .as_deref()
                .unwrap_or(model.architecture.as_str()),
            metadata
                .context_length
                .or(model.context_length)
                .unwrap_or_default(),
            metadata.tensor_table.alignment,
            metadata.tensor_table.tensor_data_offset,
            metadata.tensor_table.attention_tensor_count,
            metadata.tensor_table.ffn_tensor_count,
            metadata.tensor_table.norm_tensor_count,
            file_probe.rolling_checksum,
        )
    } else if model.inferred_format() == ModelFormat::Gguf {
        match read_gguf_header(&model.path) {
            Ok(header) => format!(
                "gguf=v{} tensors={} metadata={} checksum={:016x}",
                header.version,
                header.tensor_count,
                header.metadata_count,
                file_probe.rolling_checksum
            ),
            Err(error) => format!(
                "gguf=unreadable({error}) checksum={:016x}",
                file_probe.rolling_checksum
            ),
        }
    } else {
        format!(
            "format={:?} checksum={:016x}",
            model.inferred_format(),
            file_probe.rolling_checksum
        )
    };
    let tensor_table = metadata.as_ref().map(|summary| &summary.tensor_table);
    let prepared_fingerprint = prepared_fingerprint(model, metadata.as_ref(), &file_probe);

    PreparedSessionArtifact {
        prepared_summary,
        prepared_fingerprint,
        architecture: metadata
            .as_ref()
            .and_then(|summary| summary.architecture.clone())
            .or_else(|| Some(model.architecture.clone())),
        general_name: metadata
            .as_ref()
            .and_then(|summary| summary.general_name.clone()),
        context_length: metadata
            .as_ref()
            .and_then(|summary| summary.context_length)
            .or(model.context_length),
        tensor_count: metadata.as_ref().map(|summary| summary.header.tensor_count),
        metadata_count: metadata
            .as_ref()
            .map(|summary| summary.header.metadata_count),
        alignment: metadata.as_ref().and_then(|summary| summary.alignment),
        tensor_data_offset: tensor_table.map(|summary| summary.tensor_data_offset),
        preview_tensors: tensor_table
            .map(|summary| summary.preview_names.clone())
            .unwrap_or_default(),
        first_tensor: tensor_table
            .and_then(|summary| summary.first_tensor.as_ref())
            .map(tensor_probe_from_summary),
        last_tensor: tensor_table
            .and_then(|summary| summary.last_tensor.as_ref())
            .map(tensor_probe_from_summary),
        max_tensor_rank: tensor_table
            .map(|summary| summary.max_rank)
            .unwrap_or_default(),
        attention_tensor_count: tensor_table
            .map(|summary| summary.attention_tensor_count)
            .unwrap_or_default(),
        ffn_tensor_count: tensor_table
            .map(|summary| summary.ffn_tensor_count)
            .unwrap_or_default(),
        norm_tensor_count: tensor_table
            .map(|summary| summary.norm_tensor_count)
            .unwrap_or_default(),
        contains_output_weight: tensor_table
            .map(|summary| summary.contains_output_weight)
            .unwrap_or(false),
        contains_token_embedding: tensor_table
            .map(|summary| summary.contains_token_embedding)
            .unwrap_or(false),
        file_probe,
    }
}

fn prompt_probe_embedding(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: Option<&PreparedSessionArtifact>,
) -> Vec<f32> {
    let mut values = prompt
        .bytes()
        .take(16)
        .map(|byte| (byte as f32) / 255.0)
        .collect::<Vec<_>>();
    if let Some(artifact) = artifact {
        values.extend(session_embedding_features(artifact));
    } else if let Some(context_length) = model.context_length {
        values.push(((context_length % 4096) as f32) / 4096.0);
    }
    if values.is_empty() {
        values.push(0.0);
    }
    values
}

fn session_embedding_features(artifact: &PreparedSessionArtifact) -> Vec<f32> {
    let mut values = Vec::with_capacity(16);
    if let Some(context_length) = artifact.context_length {
        values.push(((context_length % 4096) as f32) / 4096.0);
    }
    if let Some(tensor_count) = artifact.tensor_count {
        values.push(((tensor_count % 10_000) as f32) / 10_000.0);
    }
    if let Some(metadata_count) = artifact.metadata_count {
        values.push(((metadata_count % 1_000) as f32) / 1_000.0);
    }
    if let Some(alignment) = artifact.alignment {
        values.push(((alignment % 512) as f32) / 512.0);
    }
    if let Some(data_offset) = artifact.tensor_data_offset {
        values.push(((data_offset % 8192) as f32) / 8192.0);
    }
    values.push((artifact.max_tensor_rank as f32) / 8.0);
    values.push((artifact.attention_tensor_count as f32) / 256.0);
    values.push((artifact.ffn_tensor_count as f32) / 256.0);
    values.push((artifact.norm_tensor_count as f32) / 256.0);
    values.push(if artifact.contains_output_weight {
        1.0
    } else {
        0.0
    });
    values.push(if artifact.contains_token_embedding {
        1.0
    } else {
        0.0
    });
    values.push((artifact.file_probe.prefix_len as f32) / 256.0);
    values.push(((artifact.file_probe.rolling_checksum & 0xffff) as f32) / 65535.0);
    values.extend(
        artifact
            .file_probe
            .byte_histogram
            .iter()
            .map(|value| (*value as f32) / 64.0),
    );
    values
}

fn probe_weight(prompt_embedding: &[f32], artifact: Option<&PreparedSessionArtifact>) -> Vec<f32> {
    let mut weights = vec![1.0_f32; prompt_embedding.len()];
    if let Some(artifact) = artifact {
        let arch_scale = artifact
            .architecture
            .as_deref()
            .map(architecture_weight_scale)
            .unwrap_or(1.0);
        let checksum_scale = 1.0 + ((artifact.file_probe.rolling_checksum & 0xff) as f32) / 2048.0;
        let tensor_scale = 1.0
            + (artifact.attention_tensor_count as f32) / 4096.0
            + (artifact.norm_tensor_count as f32) / 8192.0;
        for weight in &mut weights {
            *weight *= arch_scale * checksum_scale * tensor_scale;
        }
    }
    weights
}

fn architecture_weight_scale(architecture: &str) -> f32 {
    match architecture.to_ascii_lowercase().as_str() {
        "llama" => 1.00,
        "mistral" => 1.05,
        "qwen" | "qwen2" | "qwen2.5" | "qwen3" => 1.08,
        _ => 1.02,
    }
}

fn rope_probe_input(
    prompt_embedding: &[f32],
    artifact: Option<&PreparedSessionArtifact>,
) -> Vec<f32> {
    let mut values = prompt_embedding.iter().copied().take(8).collect::<Vec<_>>();
    if let Some(artifact) = artifact {
        values.push(((artifact.file_probe.rolling_checksum >> 8) & 0xff) as f32 / 255.0);
        values.push(((artifact.file_probe.rolling_checksum >> 16) & 0xff) as f32 / 255.0);
        values.push((artifact.max_tensor_rank as f32) / 8.0);
        values.push((artifact.attention_tensor_count.min(255) as f32) / 255.0);
    }
    if values.len() % 2 != 0 {
        values.push(0.0);
    }
    if values.is_empty() {
        values.extend_from_slice(&[0.0, 0.0]);
    }
    values
}

fn probe_model_file(path: &std::path::Path) -> Result<FileProbe, std::io::Error> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 256];
    let read = file.read(&mut buffer)?;
    let mut rolling_checksum = 0xcbf29ce484222325_u64;
    let mut byte_histogram = [0_u32; 4];
    for byte in &buffer[..read] {
        rolling_checksum ^= *byte as u64;
        rolling_checksum = rolling_checksum.wrapping_mul(0x100000001b3);
        let bucket = (*byte as usize) / 64;
        byte_histogram[bucket] += 1;
    }
    Ok(FileProbe {
        prefix_len: read,
        rolling_checksum,
        byte_histogram,
    })
}

fn prepared_fingerprint(
    model: &ModelDescriptor,
    metadata: Option<&GgufMetadataSummary>,
    file_probe: &FileProbe,
) -> String {
    let mut parts = vec![
        format!("fmt={:?}", model.inferred_format()),
        format!("arch={}", model.architecture),
        format!("sum={:016x}", file_probe.rolling_checksum),
    ];
    if let Some(metadata) = metadata {
        parts.push(format!("v={}", metadata.header.version));
        parts.push(format!("t={}", metadata.header.tensor_count));
        parts.push(format!("m={}", metadata.header.metadata_count));
        parts.push(format!("a={}", metadata.tensor_table.alignment));
        parts.push(format!("do={}", metadata.tensor_table.tensor_data_offset));
        parts.push(format!("mr={}", metadata.tensor_table.max_rank));
    }
    parts.join(";")
}

fn tensor_probe_from_summary(summary: &GgufTensorInfoSummary) -> TensorProbe {
    TensorProbe {
        name: summary.name.clone(),
        rank: summary.dimensions.len() as u32,
        elements: tensor_element_count(&summary.dimensions),
        ggml_dtype: summary.ggml_dtype,
        offset: summary.offset,
    }
}

fn tensor_element_count(dimensions: &[u64]) -> u64 {
    if dimensions.is_empty() {
        return 0;
    }
    dimensions
        .iter()
        .copied()
        .fold(1_u64, |acc, value| acc.saturating_mul(value.max(1)))
}

fn format_tensor_probe(probe: Option<&TensorProbe>) -> String {
    match probe {
        Some(probe) => format!(
            "{}:rank{}:el{}:dt{}:off{}",
            probe.name, probe.rank, probe.elements, probe.ggml_dtype, probe.offset
        ),
        None => "none".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{
        BackendExecutionProfile, ExecutionPlan, GenericExecutionProfile, KvCachePlan,
        PlacementDecision, RouteDecision, TieredOffloadPlan, TieredOffloadPolicy,
        TieredPlacementPercentages,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn demo_model() -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: Some("candle".to_string()),
        }
    }

    fn write_demo_gguf() -> PathBuf {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::from_le_bytes(*b"GGUF").to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&4_u64.to_le_bytes());

        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"qwen2.5";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let key = b"qwen2.context_length";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32768_u32.to_le_bytes());

        let key = b"general.name";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"Demo Qwen";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());

        write_tensor_info(&mut bytes, 3, "token_embd.weight", &[4096, 32000], 1, 0);
        write_tensor_info(&mut bytes, 3, "blk.0.attn_norm.weight", &[4096], 0, 1024);
        write_tensor_info(&mut bytes, 3, "output.weight", &[32000, 4096], 1, 2048);

        bytes.extend_from_slice(&[0_u8; 32]);
        bytes.extend_from_slice(&[0x13, 0x37, 0x42, 0x99, 0xde, 0xad, 0xbe, 0xef]);

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-candle-demo-{unique}.gguf"));
        fs::write(&path, bytes).expect("write gguf");
        path
    }

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

    fn write_sized_string(bytes: &mut Vec<u8>, version: u32, value: &[u8]) {
        match version {
            1 => bytes.extend_from_slice(&(value.len() as u32).to_le_bytes()),
            2 | 3 => bytes.extend_from_slice(&(value.len() as u64).to_le_bytes()),
            other => panic!("unsupported test gguf version: {other}"),
        }
        bytes.extend_from_slice(value);
    }

    fn candle_plan() -> ExecutionPlan {
        ExecutionPlan {
            backend: "candle".to_string(),
            route: RouteDecision {
                selected_model: "demo".to_string(),
                reason: "fallback".to_string(),
                alternatives: Vec::new(),
            },
            placements: vec![
                PlacementDecision {
                    stage: PipelineStage::Prefill,
                    target: AcceleratorKind::Gpu,
                    device_id: Some("gpu:0".to_string()),
                    memory_bytes: None,
                    rationale: "prefill".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::Decode,
                    target: AcceleratorKind::Cpu,
                    device_id: Some("cpu:0".to_string()),
                    memory_bytes: None,
                    rationale: "decode".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::KvCache,
                    target: AcceleratorKind::Disk,
                    device_id: Some("disk:0".to_string()),
                    memory_bytes: None,
                    rationale: "kv".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::Weights,
                    target: AcceleratorKind::Disk,
                    device_id: Some("disk:0".to_string()),
                    memory_bytes: None,
                    rationale: "weights".to_string(),
                },
            ],
            lowering_plan: None,
            kv_cache: KvCachePlan {
                strategy: "paged".to_string(),
                shared_across_models: false,
                page_size_bytes: Some(1 << 20),
                block_size_tokens: Some(16),
                max_cache_bytes: Some(128 << 20),
                type_k: Some("f16".to_string()),
                type_v: Some("f16".to_string()),
                tiered: true,
            },
            tiered_offload: Some(TieredOffloadPlan {
                spill_bytes: 512 << 20,
                prefetch_window_bytes: 64 << 20,
                target_device: "disk:0".to_string(),
                profile: loci_protocol::TieredOffloadProfile::Balanced,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 20,
                        cpu_percent: 40,
                        disk_percent: 40,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 0,
                        cpu_percent: 50,
                        disk_percent: 50,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 50,
                        cpu_percent: 50,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: true,
                    compress_weights: true,
                    compress_kv_cache: true,
                },
            }),
            backend_profile: BackendExecutionProfile::Candle(CandleExecutionProfile {
                session_key: "candle:gpu:0:cpu:0".to_string(),
                prefill_device: "gpu:0".to_string(),
                decode_device: "cpu:0".to_string(),
                kv_cache_device: "disk:0".to_string(),
                tensor_residency: CandleTensorResidency::Hybrid,
                fallback_reason: "fallback".to_string(),
            }),
        }
    }

    #[test]
    fn prepare_uses_plan_profile() {
        let backend = CandleBackend::default();
        let prepared = backend
            .prepare(&demo_model(), &candle_plan())
            .expect("prepared");

        assert_eq!(prepared.backend, "candle");
        assert_eq!(prepared.session_key, "candle:gpu:0:cpu:0");
        assert_eq!(prepared.residency, PreparedResidency::DiskBacked);
    }

    #[test]
    fn execute_rejects_non_candle_profile() {
        let backend = CandleBackend::default();
        let prepared = PreparedModel {
            model_name: "demo".to_string(),
            backend: "candle".to_string(),
            session_key: "candle:gpu:0:cpu:0".to_string(),
            residency: PreparedResidency::Memory,
            estimated_memory_bytes: None,
        };
        let error = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 16,
                    temperature: 0.2,
                    target_model: None,
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &ExecutionPlan {
                    backend: "candle".to_string(),
                    route: RouteDecision {
                        selected_model: "demo".to_string(),
                        reason: "generic".to_string(),
                        alternatives: Vec::new(),
                    },
                    placements: Vec::new(),
                    lowering_plan: None,
                    kv_cache: KvCachePlan {
                        strategy: "contiguous".to_string(),
                        shared_across_models: false,
                        page_size_bytes: None,
                        block_size_tokens: None,
                        max_cache_bytes: None,
                        type_k: None,
                        type_v: None,
                        tiered: false,
                    },
                    tiered_offload: None,
                    backend_profile: BackendExecutionProfile::Generic(GenericExecutionProfile {
                        session_key: "generic".to_string(),
                        summary: "generic".to_string(),
                    }),
                },
            )
            .expect_err("error");

        assert!(error.message.contains("Candle backend profile"));
    }

    #[test]
    fn prepare_rejects_npu_placements() {
        let backend = CandleBackend::default();
        let mut plan = candle_plan();
        plan.placements[1].target = AcceleratorKind::Npu;

        let error = backend.prepare(&demo_model(), &plan).expect_err("error");
        assert!(error.message.contains("NPU"));
    }

    #[test]
    fn execute_rejects_multimodal_requests() {
        let backend = CandleBackend::default();
        let prepared = backend
            .prepare(&demo_model(), &candle_plan())
            .expect("prepared");

        let error = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "describe".to_string(),
                    max_tokens: 16,
                    temperature: 0.2,
                    target_model: Some("demo".to_string()),
                    images: vec![loci_protocol::ImageInput::Path {
                        path: PathBuf::from("D:/images/demo.png"),
                    }],
                    structured_output: false,
                    tool_calling: false,
                },
                &candle_plan(),
            )
            .expect_err("error");

        assert!(error.message.contains("multimodal"));
    }

    #[test]
    fn execute_reports_rmsnorm_probe_signature() {
        let backend = CandleBackend::default();
        let path = write_demo_gguf();
        let mut model = demo_model();
        model.path = path.clone();
        model.architecture = "qwen".to_string();
        let prepared = backend.prepare(&model, &candle_plan()).expect("prepared");

        let output = backend
            .execute(
                &prepared,
                &model,
                &SessionRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 8,
                    temperature: 0.2,
                    target_model: Some("demo".to_string()),
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &candle_plan(),
            )
            .expect("output");

        assert!(output.text.contains("rmsnorm=["));
        assert!(output.text.contains("rope=["));
        assert!(output
            .text
            .contains("gguf=v3 tensors=3 metadata=4 arch=qwen2.5"));
        assert!(output.text.contains("align=32 data_offset="));
        assert!(output.text.contains("attn=1 ffn=0 norm=1"));
        assert!(output
            .text
            .contains("preview_tensors=token_embd.weight|blk.0.attn_norm.weight|output.weight"));
        assert!(output
            .text
            .contains("first_tensor=token_embd.weight:rank2:el131072000:dt1:off0"));
        assert!(output
            .text
            .contains("last_tensor=output.weight:rank2:el131072000:dt1:off2048"));
        assert!(output.text.contains("fingerprint=fmt=Gguf;arch=qwen;sum="));
        assert!(output.text.contains("checksum="));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn prepare_captures_tensor_table_facts() {
        let backend = CandleBackend::default();
        let path = write_demo_gguf();
        let mut model = demo_model();
        model.path = path.clone();
        model.architecture = "qwen".to_string();
        let profile = match &candle_plan().backend_profile {
            BackendExecutionProfile::Candle(profile) => profile.clone(),
            _ => panic!("expected candle profile"),
        };

        let artifact = backend
            .runtime
            .build_prepared_artifact(&model)
            .expect("artifact");
        assert_eq!(artifact.alignment, Some(32));
        assert!(artifact.tensor_data_offset.is_some());
        assert_eq!(
            artifact.preview_tensors,
            vec![
                "token_embd.weight".to_string(),
                "blk.0.attn_norm.weight".to_string(),
                "output.weight".to_string()
            ]
        );
        assert_eq!(artifact.max_tensor_rank, 2);
        assert_eq!(artifact.attention_tensor_count, 1);
        assert_eq!(artifact.ffn_tensor_count, 0);
        assert_eq!(artifact.norm_tensor_count, 1);
        assert!(artifact.contains_output_weight);
        assert!(artifact.contains_token_embedding);
        assert!(artifact
            .prepared_fingerprint
            .starts_with("fmt=Gguf;arch=qwen;sum="));

        backend.prepare(&model, &candle_plan()).expect("prepared");
        let cached = backend
            .runtime
            .lookup_prepared_artifact(&profile.session_key)
            .expect("lookup")
            .expect("cached");
        assert_eq!(cached.preview_tensors.len(), 3);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn execute_requires_prepared_session_artifact() {
        let backend = CandleBackend::default();
        let path = write_demo_gguf();
        let mut model = demo_model();
        model.path = path.clone();
        let prepared = PreparedModel {
            model_name: "demo".to_string(),
            backend: "candle".to_string(),
            session_key: "candle:gpu:0:cpu:0".to_string(),
            residency: PreparedResidency::Memory,
            estimated_memory_bytes: None,
        };

        let error = backend
            .execute(
                &prepared,
                &model,
                &SessionRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 8,
                    temperature: 0.2,
                    target_model: Some("demo".to_string()),
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &candle_plan(),
            )
            .expect_err("missing prepared artifact");

        assert!(error.message.contains("prepared artifact"));

        let _ = fs::remove_file(path);
    }
}
