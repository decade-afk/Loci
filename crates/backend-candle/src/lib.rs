use loci_gguf::{
    read_gguf_metadata_summary, read_gguf_tensor_prefix_f32,
    support_profile as gguf_support_profile, GgufMetadataSummary,
    GgufTensorPrefix,
};
use loci_kernels_llama::{curated_kernel_descriptors, rms_norm_f32, rope_f32};
use loci_protocol::{
    AcceleratorKind, Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
    BackendExecutionProfile, BackendKernelCatalog, BackendLoweringCapabilities, BackendOutput,
    BackendResult, BackendRuntimeFamily, BackendTelemetry, CandleExecutionProfile,
    CandleTensorResidency, ChipOperatorClass, DeviceDescriptor, ExecutionArtifactKind,
    ExecutionPlan, HardwareTopology, KernelDescriptor, KernelImplementationKind, KernelMaturity,
    KernelOrigin, LoweringGranularity, ModelAssetLayout, ModelDescriptor, ModelFormat,
    PipelineStage, PowerState, PreparedModel, PreparedResidency, SessionRequest, ThermalState,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::Mutex;

const TOKEN_PREFIX_ELEMENTS: usize = 524_288;
const OUTPUT_PREFIX_ELEMENTS: usize = 524_288;
const NORM_PREFIX_ELEMENTS: usize = 2_048;
const MAX_GREEDY_STEPS: u32 = 32;

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
    context_length: Option<u32>,
    tensor_count: Option<u64>,
    metadata_count: Option<u64>,
    alignment: Option<u64>,
    tensor_data_offset: Option<u64>,
    preview_tensors: Vec<String>,
    candidate_tensors: Vec<String>,
    max_tensor_rank: u32,
    attention_tensor_count: u32,
    ffn_tensor_count: u32,
    norm_tensor_count: u32,
    contains_output_weight: bool,
    contains_token_embedding: bool,
    real_norm_tensor: Option<GgufTensorPrefix>,
    token_embeddings: Option<GgufTensorPrefix>,
    output_weights: Option<GgufTensorPrefix>,
    tokenizer_tokens: Vec<String>,
    file_probe: FileProbe,
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
            directly_supported_layouts: vec![ModelAssetLayout::GgufFile, ModelAssetLayout::GgufDirectory],
            ingestible_layouts: vec![
                ModelAssetLayout::SafeTensorsFile,
                ModelAssetLayout::SafeTensorsDirectory,
                ModelAssetLayout::PytorchBinFile,
                ModelAssetLayout::PytorchCheckpointDirectory,
                ModelAssetLayout::TransformersCheckpoint,
                ModelAssetLayout::UnknownDirectory,
                ModelAssetLayout::UnknownFile,
            ],
            preferred_artifact: ExecutionArtifactKind::NativeCheckpoint,
            requires_lowering_for_execution: false,
            notes: vec![
                "Candle currently executes the direct GGUF path".to_string(),
                "SafeTensors and PyTorch-style assets remain ingestible but are not yet executable through the current Candle path".to_string(),
            ],
        }
    }

    fn lowering_capabilities(&self) -> BackendLoweringCapabilities {
        BackendLoweringCapabilities {
            backend: "candle".to_string(),
            runtime_family: BackendRuntimeFamily::Candle,
            granularity: LoweringGranularity::Tensor,
            supports_real_execution: true,
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
                "the current execution shape is per-tensor placement in a pure Rust runtime"
                    .to_string(),
                "the current Candle path executes direct local GGUF text generation while tiered residency remains planner-driven".to_string(),
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
            ModelFormat::Gguf | ModelFormat::Directory
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
                message: "Candle direct execution does not yet support multimodal image inputs"
                    .to_string(),
            });
        }
        let session_artifact = self
            .lookup_prepared_artifact(&prepared.session_key)?
            .ok_or_else(|| BackendError {
                message: format!(
                    "Candle execution is missing a prepared artifact for session `{}`",
                    prepared.session_key
                ),
            })?;
        let prompt_embedding = derive_generation_seed(request.prompt.trim(), model, &session_artifact);
        let probe_weight = derive_norm_weights(&session_artifact, prompt_embedding.len());
        let probe_norm =
            rms_norm_f32(&prompt_embedding, &probe_weight, 1e-5).unwrap_or(prompt_embedding.clone());
        let output_projection = session_artifact
            .output_weights
            .as_ref()
            .or(session_artifact.token_embeddings.as_ref())
            .ok_or_else(|| BackendError {
                message: format!(
                    "Candle execution is missing a GGUF output projection tensor; preview_tensors={}; candidate_tensors={}",
                    session_artifact.preview_tensors.join("|"),
                    session_artifact.candidate_tensors.join("|")
                ),
            })?;
        let generated = greedy_generate_text(
            &probe_norm,
            &probe_weight,
            output_projection,
            session_artifact.token_embeddings.as_ref(),
            &session_artifact.tokenizer_tokens,
            request.max_tokens,
        )?;

        Ok(BackendOutput {
            text: generated.text,
            telemetry: BackendTelemetry {
                estimated_prefill_ms: estimate_prefill_ms(profile, plan),
                estimated_decode_ms: estimate_decode_ms(profile, plan),
                generated_tokens: generated.generated_tokens,
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
            message: "Candle direct execution does not support NPU placements".to_string(),
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

fn build_prepared_artifact(
    model: &ModelDescriptor,
    metadata: Option<GgufMetadataSummary>,
    file_probe: FileProbe,
) -> PreparedSessionArtifact {
    let (real_norm_tensor, token_embeddings, output_weights) = if model.inferred_format() == ModelFormat::Gguf {
        let token_embeddings = [
            "token_embd.weight",
            "tok_embeddings.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
        ]
        .iter()
        .find_map(|name| read_gguf_tensor_prefix_f32(&model.path, name, TOKEN_PREFIX_ELEMENTS).ok().flatten());
        let real_norm_tensor = [
            "output_norm.weight",
            "norm.weight",
            "model.norm.weight",
            "transformer.norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_norm.weight",
        ]
        .iter()
        .find_map(|name| read_gguf_tensor_prefix_f32(&model.path, name, NORM_PREFIX_ELEMENTS).ok().flatten())
        .map(|mut tensor| {
            if tensor.values_f32.len() % 2 != 0 {
                tensor.values_f32.pop();
            }
            tensor
        });
        let output_weights = [
            "output.weight",
            "lm_head.weight",
            "model.output.weight",
        ]
        .iter()
        .find_map(|name| read_gguf_tensor_prefix_f32(&model.path, name, OUTPUT_PREFIX_ELEMENTS).ok().flatten())
        .or_else(|| token_embeddings.clone());
        (real_norm_tensor, token_embeddings, output_weights)
    } else {
        (None, None, None)
    };
    let tensor_table = metadata.as_ref().map(|summary| &summary.tensor_table);

    PreparedSessionArtifact {
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
        candidate_tensors: tensor_table
            .map(|summary| summary.candidate_names.clone())
            .unwrap_or_default(),
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
        real_norm_tensor,
        token_embeddings,
        output_weights,
        tokenizer_tokens: metadata
            .as_ref()
            .and_then(|summary| summary.tokenizer_tokens.clone())
            .unwrap_or_default(),
        file_probe,
    }
}

fn prompt_probe_embedding(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: Option<&PreparedSessionArtifact>,
    target_len: usize,
) -> Vec<f32> {
    let mut values = prompt
        .bytes()
        .take(target_len.max(1))
        .map(|byte| (byte as f32) / 255.0)
        .collect::<Vec<_>>();
    if let Some(artifact) = artifact {
        values.extend(session_embedding_features(artifact));
    } else if let Some(context_length) = model.context_length {
        values.push(((context_length % 4096) as f32) / 4096.0);
    }
    while values.len() < target_len.max(1) {
        let next = values
            .last()
            .copied()
            .unwrap_or(0.0)
            .mul_add(0.5, 0.125);
        values.push(next.fract());
    }
    values.truncate(target_len.max(1));
    if values.len() % 2 != 0 {
        values.pop();
    }
    if values.is_empty() {
        values.extend_from_slice(&[0.0, 0.0]);
    }
    values
}

fn derive_prompt_embedding(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: &PreparedSessionArtifact,
) -> BackendResult<Vec<f32>> {
    if let Some(token_tensor) = artifact.token_embeddings.as_ref() {
        let hidden = infer_hidden_size(token_tensor, artifact.real_norm_tensor.as_ref())
            .unwrap_or(token_tensor.values_f32.len())
            .max(2);
        let hidden = if hidden % 2 == 0 { hidden } else { hidden - 1 };
        let token_id = tokenize_prompt(prompt, artifact)
            .unwrap_or_else(|| fallback_token_id(prompt, token_tensor, hidden));
        return embedding_for_token(token_tensor, token_id, hidden);
    }

    let target_len = artifact
        .real_norm_tensor
        .as_ref()
        .map(|tensor| tensor.values_f32.len())
        .unwrap_or(32);
    Ok(prompt_probe_embedding(prompt, model, Some(artifact), target_len))
}

fn derive_generation_seed(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: &PreparedSessionArtifact,
) -> Vec<f32> {
    if let Ok(embedding) = derive_prompt_embedding(prompt, model, artifact) {
        return embedding;
    }
    let target_len = artifact
        .token_embeddings
        .as_ref()
        .and_then(|tensor| tensor.info.dimensions.last().copied())
        .map(|value| value as usize)
        .or_else(|| artifact.real_norm_tensor.as_ref().map(|tensor| tensor.values_f32.len()))
        .unwrap_or(32);
    prompt_probe_embedding(prompt, model, Some(artifact), target_len.max(2))
}

fn derive_norm_weights(artifact: &PreparedSessionArtifact, hidden_len: usize) -> Vec<f32> {
    if let Some(norm) = artifact.real_norm_tensor.as_ref() {
        let weights = truncate_even_prefix(&norm.values_f32, hidden_len);
        if !weights.is_empty() {
            return weights;
        }
    }
    vec![1.0; hidden_len.max(2)]
}

fn infer_hidden_size(
    token_tensor: &GgufTensorPrefix,
    norm_tensor: Option<&GgufTensorPrefix>,
) -> Option<usize> {
    if let Some(norm) = norm_tensor {
        let norm_len = norm.values_f32.len().max(2);
        if token_tensor.values_f32.len() >= norm_len {
            return Some(norm_len);
        }
    }
    token_tensor
        .info
        .dimensions
        .last()
        .copied()
        .map(|value| value as usize)
}

fn tokenize_prompt(prompt: &str, artifact: &PreparedSessionArtifact) -> Option<usize> {
    if artifact.tokenizer_tokens.is_empty() {
        return None;
    }
    let trimmed = prompt.trim();
    artifact
        .tokenizer_tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| !token.is_empty())
        .find_map(|(index, token)| {
            if trimmed == token || trimmed.starts_with(token) {
                Some(index)
            } else {
                None
            }
        })
}

fn fallback_token_id(prompt: &str, token_tensor: &GgufTensorPrefix, hidden: usize) -> usize {
    let token_count = token_count_from_tensor(token_tensor, hidden).max(1);
    let mut hash = 0_u64;
    for byte in prompt.as_bytes() {
        hash = hash.wrapping_mul(16777619).wrapping_add(*byte as u64 + 1);
    }
    (hash as usize) % token_count
}

fn token_count_from_tensor(token_tensor: &GgufTensorPrefix, hidden: usize) -> usize {
    if hidden == 0 {
        return 0;
    }
    token_tensor.values_f32.len() / hidden
}

fn embedding_for_token(
    token_tensor: &GgufTensorPrefix,
    token_id: usize,
    hidden: usize,
) -> BackendResult<Vec<f32>> {
    let token_count = token_count_from_tensor(token_tensor, hidden);
    if token_count == 0 {
        return Err(BackendError {
            message: "token embedding tensor does not contain any complete rows".to_string(),
        });
    }
    let index = token_id % token_count;
    let start = index * hidden;
    let end = start + hidden;
    let mut values = token_tensor.values_f32[start..end].to_vec();
    if values.len() % 2 != 0 {
        values.pop();
    }
    if values.is_empty() {
        return Err(BackendError {
            message: "selected token embedding row is empty".to_string(),
        });
    }
    Ok(values)
}

struct CandleGeneration {
    text: String,
    generated_tokens: u32,
}

struct TokenSelection {
    token_index: usize,
    token_text: String,
}

fn greedy_generate_text(
    initial_hidden: &[f32],
    norm_weights: &[f32],
    output_weights: &GgufTensorPrefix,
    token_embeddings: Option<&GgufTensorPrefix>,
    tokenizer_tokens: &[String],
    max_tokens: u32,
) -> BackendResult<CandleGeneration> {
    let steps = max_tokens.clamp(1, MAX_GREEDY_STEPS);
    let row_width = projection_row_width(output_weights, initial_hidden.len())?;
    let mut hidden_state = initial_hidden.to_vec();
    let mut seen_counts: HashMap<usize, u32> = HashMap::new();
    let mut text = String::new();
    let mut generated_tokens = 0_u32;

    for step in 0..steps {
        let selection = greedy_decode_token(
            &hidden_state,
            output_weights,
            tokenizer_tokens,
            &seen_counts,
        )?;
        let rendered = render_token_text(&selection.token_text);
        let is_terminal = is_terminal_token(&selection.token_text);

        if !rendered.is_empty() {
            text.push_str(&rendered);
        }
        if is_terminal && generated_tokens > 0 {
            break;
        }

        generated_tokens += 1;
        *seen_counts.entry(selection.token_index).or_insert(0) += 1;
        hidden_state = evolve_hidden_state(
            &hidden_state,
            token_embeddings,
            selection.token_index,
            row_width,
            norm_weights,
            step as usize,
        )?;
    }

    if text.is_empty() {
        text = "<empty>".to_string();
    }

    Ok(CandleGeneration {
        text,
        generated_tokens,
    })
}

fn greedy_decode_token(
    hidden_state: &[f32],
    output_weights: &GgufTensorPrefix,
    tokenizer_tokens: &[String],
    seen_counts: &HashMap<usize, u32>,
) -> BackendResult<TokenSelection> {
    let row_width = projection_row_width(output_weights, hidden_state.len())?;
    let hidden = hidden_state.len().min(row_width);
    let row_count = output_weights.values_f32.len() / row_width;
    if row_count == 0 {
        return Err(BackendError {
            message: "output projection tensor does not contain any complete rows".to_string(),
        });
    }

    let candidate_count = tokenizer_tokens.len().max(1).min(row_count).min(128);
    let mut scored = (0..candidate_count)
        .map(|index| {
            let start = index * row_width;
            let row = &output_weights.values_f32[start..start + row_width];
            let repeat_penalty = seen_counts.get(&index).copied().unwrap_or_default() as f32 * 8.0;
            let score = row
                .iter()
                .zip(hidden_state.iter())
                .take(hidden)
                .map(|(weight, hidden)| weight * hidden)
                .sum::<f32>()
                - repeat_penalty;
            (index, score)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let selected_index = scored
        .first()
        .map(|(index, _)| *index)
        .ok_or_else(|| BackendError {
            message: "unable to score any output token candidates".to_string(),
        })?;
    let selected_token = tokenizer_tokens
        .get(selected_index)
        .cloned()
        .unwrap_or_else(|| format!("<token:{selected_index}>"));
    Ok(TokenSelection {
        token_index: selected_index,
        token_text: selected_token,
    })
}

fn projection_row_width(output_weights: &GgufTensorPrefix, fallback_hidden: usize) -> BackendResult<usize> {
    let row_width = output_weights
        .info
        .dimensions
        .last()
        .copied()
        .unwrap_or(fallback_hidden as u64) as usize;
    if row_width == 0 {
        return Err(BackendError {
            message: "output projection tensor has zero row width".to_string(),
        });
    }
    Ok(row_width)
}

fn evolve_hidden_state(
    current_hidden: &[f32],
    token_embeddings: Option<&GgufTensorPrefix>,
    token_index: usize,
    hidden_size: usize,
    norm_weights: &[f32],
    step: usize,
) -> BackendResult<Vec<f32>> {
    let mut mixed = if let Some(token_tensor) = token_embeddings {
        let token_embedding = embedding_for_token(token_tensor, token_index, hidden_size)?;
        token_embedding
            .iter()
            .zip(current_hidden.iter())
            .map(|(next, current)| next.mul_add(0.7, current * 0.3))
            .collect::<Vec<_>>()
    } else {
        current_hidden
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = ((step + index) % 17) as f32 / 17.0;
                value.mul_add(0.8, phase - 0.25)
            })
            .collect::<Vec<_>>()
    };

    mixed = rope_f32(&mixed, step + 1, 10_000.0, mixed.len()).map_err(|error| BackendError {
        message: format!("Candle recurrent RoPE probe failed: {error}"),
    })?;
    rms_norm_f32(&mixed, norm_weights, 1e-5).map_err(|error| BackendError {
        message: format!("Candle recurrent RMSNorm probe failed: {error}"),
    })
}

fn is_terminal_token(token: &str) -> bool {
    let trimmed = token.trim();
    trimmed.is_empty()
        || trimmed == "</s>"
        || trimmed.eq_ignore_ascii_case("<eos>")
        || trimmed.contains("endoftext")
        || trimmed.contains("end_of_text")
        || trimmed.contains("end_of_sentence")
}

fn render_token_text(token: &str) -> String {
    if let Some(byte) = parse_hex_byte_token(token) {
        return String::from_utf8_lossy(&[byte]).into_owned();
    }
    token.replace('▁', " ").replace('Ġ', " ")
}

fn parse_hex_byte_token(token: &str) -> Option<u8> {
    let hex = token
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn truncate_even_prefix(values: &[f32], target_len: usize) -> Vec<f32> {
    let mut taken = values.iter().copied().take(target_len).collect::<Vec<_>>();
    if taken.len() % 2 != 0 {
        taken.pop();
    }
    taken
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

        write_tensor_info(&mut bytes, 3, "token_embd.weight", &[4], 0, 0);
        write_tensor_info(&mut bytes, 3, "blk.0.attn_norm.weight", &[4], 0, 16);
        write_tensor_info(&mut bytes, 3, "output.weight", &[4], 0, 32);

        bytes.extend_from_slice(&[0_u8; 32]);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        bytes.extend_from_slice(&3.0_f32.to_le_bytes());
        bytes.extend_from_slice(&4.0_f32.to_le_bytes());
        bytes.extend_from_slice(&5.0_f32.to_le_bytes());
        bytes.extend_from_slice(&6.0_f32.to_le_bytes());
        bytes.extend_from_slice(&7.0_f32.to_le_bytes());
        bytes.extend_from_slice(&8.0_f32.to_le_bytes());
        bytes.extend_from_slice(&9.0_f32.to_le_bytes());
        bytes.extend_from_slice(&10.0_f32.to_le_bytes());
        bytes.extend_from_slice(&11.0_f32.to_le_bytes());
        bytes.extend_from_slice(&12.0_f32.to_le_bytes());

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
    fn execute_returns_plain_generated_text() {
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

        assert!(!output.text.is_empty());
        assert!(!output.text.contains("rmsnorm=["));
        assert!(!output.text.contains("fingerprint="));

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
        assert_eq!(artifact.max_tensor_rank, 1);
        assert_eq!(artifact.attention_tensor_count, 1);
        assert_eq!(artifact.ffn_tensor_count, 0);
        assert_eq!(artifact.norm_tensor_count, 1);
        assert!(artifact.contains_output_weight);
        assert!(artifact.contains_token_embedding);
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
