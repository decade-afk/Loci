use crate::planning::{
    estimate_decode_ms, estimate_prefill_ms, lowering_summary, offload_profile_label,
    placement_summary,
};
use crate::{FallbackSession, ModelDescriptor};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use loci_protocol::{
    BackendError, BackendOutput, BackendResult, BackendTelemetry, ExecutionPlan, ImageInput,
    OpenVinoExecutionMode, OpenVinoExecutionProfile, PipelineStage, SessionRequest,
};
use openvino::{ElementType, Shape, Tensor};
use openvino_genai::{DecodedResults, GenerationConfig, VlmDecodedResults};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub(super) fn build_generation_config(request: &SessionRequest) -> BackendResult<GenerationConfig> {
    let mut config = GenerationConfig::new().map_err(inference_error)?;
    config
        .set_max_new_tokens(request.max_tokens as usize)
        .map_err(inference_error)?;
    config
        .set_temperature(request.temperature.max(0.0))
        .map_err(inference_error)?;
    config
        .set_do_sample(request.temperature > 0.0)
        .map_err(inference_error)?;
    config.set_num_beams(1).map_err(inference_error)?;
    config.validate().map_err(inference_error)?;
    Ok(config)
}

pub(super) fn telemetry_from_llm_results(
    results: &DecodedResults,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let metrics = results.get_perf_metrics().map_err(inference_error);
    telemetry_from_metrics(metrics, request, profile, model, plan)
}

pub(super) fn telemetry_from_vlm_results(
    results: &VlmDecodedResults,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let metrics = results.get_perf_metrics().map_err(inference_error);
    telemetry_from_metrics(metrics, request, profile, model, plan)
}

pub(super) fn fallback_behavior(
    session: &FallbackSession,
    model: &ModelDescriptor,
    request: &SessionRequest,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> BackendResult<BackendOutput> {
    let allow_fallback = env::var("LOCI_OPENVINO_ALLOW_FALLBACK")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);

    if allow_fallback {
        return Ok(run_fallback_session(session, model, request, plan, profile));
    }

    Err(BackendError {
        message: format!(
            "OpenVINO real execution is unavailable for model `{}`: {}. Set LOCI_OPENVINO_ALLOW_FALLBACK=1 to re-enable diagnostic fallback output.",
            model.name, session.reason
        ),
    })
}

pub(super) fn setup_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub(super) fn openvino_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub(super) fn inference_error(error: impl std::fmt::Display) -> BackendError {
    BackendError {
        message: error.to_string(),
    }
}

pub(super) fn load_image_tensors(images: &[ImageInput]) -> BackendResult<Vec<Tensor>> {
    images.iter().map(load_image_tensor).collect()
}

fn telemetry_from_metrics(
    metrics: Result<openvino_genai::PerfMetrics, BackendError>,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let fallback = BackendTelemetry {
        estimated_prefill_ms: estimate_prefill_ms(profile, model, plan),
        estimated_decode_ms: estimate_decode_ms(profile, plan),
        generated_tokens: request.max_tokens.min(128),
    };

    let Ok(metrics) = metrics else {
        return fallback;
    };

    let generated_tokens = metrics
        .get_num_generation_tokens()
        .ok()
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(fallback.generated_tokens);
    let estimated_prefill_ms = metrics
        .get_ttft()
        .ok()
        .map(|(mean, _)| f32_ms_to_u64(mean))
        .unwrap_or(fallback.estimated_prefill_ms);
    let estimated_decode_ms = metrics
        .get_tpot()
        .ok()
        .map(|(mean, _)| f32_ms_to_u64(mean))
        .unwrap_or(fallback.estimated_decode_ms);

    BackendTelemetry {
        estimated_prefill_ms,
        estimated_decode_ms,
        generated_tokens,
    }
}

fn run_fallback_session(
    session: &FallbackSession,
    model: &ModelDescriptor,
    request: &SessionRequest,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> BackendOutput {
    let mode = match profile.execution_mode {
        OpenVinoExecutionMode::Hetero => "hetero",
        OpenVinoExecutionMode::NpuFirst => "npu-first",
    };
    let devices = profile.hetero_devices.join(">");
    let prefill = placement_summary(plan, PipelineStage::Prefill);
    let decode = placement_summary(plan, PipelineStage::Decode);
    let kv = placement_summary(plan, PipelineStage::KvCache);
    let weights = placement_summary(plan, PipelineStage::Weights);
    let lowering = lowering_summary(plan);
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
    let model_root = session
        .model_root
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "unresolved".to_string());
    let lowering_diagnostics = session.lowering_diagnostics.as_deref().unwrap_or("none");

    BackendOutput {
        text: format!(
            "[openvino-fallback:{}] mode={} device={} devices={} model_root={} prefill={} decode={} kv={} weights={} lowering={} route={} {} image_count={} reason={} lowering_diagnostics={} prompt=`{}`",
            model.name,
            mode,
            session.device_name,
            devices,
            model_root,
            prefill,
            decode,
            kv,
            weights,
            lowering,
            plan.route.reason,
            spill,
            request.images.len(),
            session.reason,
            lowering_diagnostics,
            request.prompt.trim()
        ),
        telemetry: BackendTelemetry {
            estimated_prefill_ms: estimate_prefill_ms(profile, model, plan),
            estimated_decode_ms: estimate_decode_ms(profile, plan),
            generated_tokens: request.max_tokens.min(128),
        },
    }
}

fn f32_ms_to_u64(value: f32) -> u64 {
    if value.is_finite() && value >= 0.0 {
        value.round() as u64
    } else {
        0
    }
}

fn load_image_tensor(image: &ImageInput) -> BackendResult<Tensor> {
    match image {
        ImageInput::Path { path } => {
            let bytes = fs::read(path).map_err(io_error)?;
            decode_image_tensor(&bytes, Some(path))
        }
        ImageInput::Url { url } => {
            if let Some(path) = file_url_to_path(url) {
                let bytes = fs::read(&path).map_err(io_error)?;
                decode_image_tensor(&bytes, Some(&path))
            } else if let Some(bytes) = decode_data_url(url)? {
                decode_image_tensor(&bytes, None)
            } else {
                Err(BackendError {
                    message: format!(
                        "unsupported image URL `{url}`; only file:// URLs and data URLs are supported"
                    ),
                })
            }
        }
        ImageInput::Base64 {
            data_base64,
            media_type: _,
        } => {
            let bytes = BASE64_STANDARD
                .decode(data_base64)
                .map_err(|error| BackendError {
                    message: format!("invalid base64 image payload: {error}"),
                })?;
            decode_image_tensor(&bytes, None)
        }
    }
}

fn decode_image_tensor(bytes: &[u8], source_path: Option<&Path>) -> BackendResult<Tensor> {
    let decoded = image::load_from_memory(bytes).map_err(|error| BackendError {
        message: match source_path {
            Some(path) => format!("failed to decode image `{}`: {error}", path.display()),
            None => format!("failed to decode image payload: {error}"),
        },
    })?;
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    let shape = Shape::new(&[height as i64, width as i64, 3]).map_err(inference_error)?;
    let mut tensor = Tensor::new(ElementType::U8, &shape).map_err(inference_error)?;
    tensor
        .get_raw_data_mut()
        .map_err(inference_error)?
        .copy_from_slice(&rgb.into_raw());
    Ok(tensor)
}

pub(crate) fn file_url_to_path(url: &str) -> Option<PathBuf> {
    let normalized = url.strip_prefix("file://")?;
    let trimmed = normalized.strip_prefix('/').unwrap_or(normalized);
    Some(PathBuf::from(trimmed))
}

pub(crate) fn decode_data_url(url: &str) -> BackendResult<Option<Vec<u8>>> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Ok(None);
    };
    let Some((metadata, encoded)) = payload.split_once(',') else {
        return Err(BackendError {
            message: "data URL image payload is malformed".to_string(),
        });
    };
    if !metadata.ends_with(";base64") {
        return Err(BackendError {
            message: "data URL image payload must use base64 encoding".to_string(),
        });
    }
    let bytes = BASE64_STANDARD
        .decode(encoded)
        .map_err(|error| BackendError {
            message: format!("invalid base64 image payload: {error}"),
        })?;
    Ok(Some(bytes))
}

fn io_error(error: std::io::Error) -> BackendError {
    BackendError {
        message: error.to_string(),
    }
}
