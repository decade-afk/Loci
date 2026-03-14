use crate::backend::GpuSplitMode;
use crate::device::{DeviceInfo, DeviceSelector, DeviceType};
use crate::error::{LociError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_VERSION_V2: u32 = 2;
const GGUF_VERSION_V3: u32 = 3;
const GGUF_MAX_STRING_BYTES: usize = 8 * 1024 * 1024;
const MIN_KV_CACHE_BYTES: u64 = 64 * 1024 * 1024;

const GGUF_VALUE_TYPE_UINT8: u32 = 0;
const GGUF_VALUE_TYPE_INT8: u32 = 1;
const GGUF_VALUE_TYPE_UINT16: u32 = 2;
const GGUF_VALUE_TYPE_INT16: u32 = 3;
const GGUF_VALUE_TYPE_UINT32: u32 = 4;
const GGUF_VALUE_TYPE_INT32: u32 = 5;
const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
const GGUF_VALUE_TYPE_BOOL: u32 = 7;
const GGUF_VALUE_TYPE_STRING: u32 = 8;
const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
const GGUF_VALUE_TYPE_UINT64: u32 = 10;
const GGUF_VALUE_TYPE_INT64: u32 = 11;
const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum EstimateMetadataSource {
    FileSizeOnly,
    GgufMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct GgufMetadataSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub embedding_length: Option<u32>,
    pub block_count: Option<u32>,
    pub vocab_size: Option<u32>,
    pub attention_head_count: Option<u32>,
    pub attention_head_count_kv: Option<u32>,
    pub feed_forward_length: Option<u32>,
    pub expert_count: Option<u32>,
    pub file_type: Option<u32>,
}

impl GgufMetadataSummary {
    pub fn layer_count(&self) -> Option<u32> {
        self.block_count.filter(|value| *value > 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelResourceEstimate {
    pub model_bytes: u64,
    pub kv_cache_bytes: u64,
    pub working_set_bytes: u64,
    pub total_bytes: u64,
    pub context_size: u32,
    pub metadata_source: EstimateMetadataSource,
    pub gguf_metadata: Option<GgufMetadataSummary>,
}

impl ModelResourceEstimate {
    pub fn layer_count(&self) -> Option<u32> {
        self.gguf_metadata
            .as_ref()
            .and_then(GgufMetadataSummary::layer_count)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResourcePlan {
    pub use_gpu: bool,
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub kv_offload: bool,
    pub op_offload: bool,
    pub split_mode: GpuSplitMode,
    pub main_gpu: u32,
    pub tensor_split: Option<Vec<f32>>,
    pub rationale: String,
}

pub struct ResourcePlanner;

impl ResourcePlanner {
    pub fn estimate_model_requirements(
        model_path: &Path,
        context_size: u32,
    ) -> Result<ModelResourceEstimate> {
        let metadata = fs::metadata(model_path).map_err(|e| {
            LociError::ConfigError(format!(
                "failed to inspect model '{}': {}",
                model_path.display(),
                e
            ))
        })?;
        let model_bytes = metadata.len();
        if model_bytes == 0 {
            return Err(LociError::ConfigError(format!(
                "model '{}' is empty",
                model_path.display()
            )));
        }

        let gguf_metadata = read_gguf_metadata(model_path).ok().flatten();
        let metadata_source = if gguf_metadata.is_some() {
            EstimateMetadataSource::GgufMetadata
        } else {
            EstimateMetadataSource::FileSizeOnly
        };

        let kv_cache_bytes = gguf_metadata
            .as_ref()
            .and_then(|metadata| estimate_kv_cache_bytes(metadata, context_size))
            .unwrap_or_else(|| {
                (context_size as u64)
                    .saturating_mul(16 * 1024)
                    .max(MIN_KV_CACHE_BYTES)
            });
        let working_set_bytes = model_bytes.saturating_div(4).saturating_add(kv_cache_bytes);
        let total_bytes = model_bytes.saturating_add(working_set_bytes);

        Ok(ModelResourceEstimate {
            model_bytes,
            kv_cache_bytes,
            working_set_bytes,
            total_bytes,
            context_size,
            metadata_source,
            gguf_metadata,
        })
    }

    pub fn plan_for_model(model_path: &Path, context_size: u32) -> Result<ResourcePlan> {
        let estimate = Self::estimate_model_requirements(model_path, context_size)?;
        let selector = DeviceSelector::new();
        Ok(Self::plan_for_estimate(&estimate, selector.devices()))
    }

    pub fn plan_for_estimate(
        estimate: &ModelResourceEstimate,
        devices: &[DeviceInfo],
    ) -> ResourcePlan {
        let mut gpus = devices
            .iter()
            .filter(|device| device.available && device.device_type != DeviceType::CPU)
            .cloned()
            .collect::<Vec<_>>();
        gpus.sort_by(|left, right| right.memory_bytes.cmp(&left.memory_bytes));

        let best_gpu = gpus.first();
        let total_gpu_bytes = gpus
            .iter()
            .fold(0u64, |acc, device| acc.saturating_add(device.memory_bytes));
        let cpu_memory_bytes = devices
            .iter()
            .find(|device| device.available && device.device_type == DeviceType::CPU)
            .map(|device| device.memory_bytes)
            .unwrap_or_default();
        let max_layers = estimate.layer_count().unwrap_or(32).clamp(1, 4096);

        if let Some(best_gpu) = best_gpu {
            if best_gpu.memory_bytes >= estimate.total_bytes {
                return ResourcePlan {
                    use_gpu: true,
                    n_gpu_layers: -1,
                    use_mmap: estimate.model_bytes >= 2 * 1024 * 1024 * 1024,
                    use_mlock: false,
                    kv_offload: true,
                    op_offload: true,
                    split_mode: GpuSplitMode::None,
                    main_gpu: best_gpu.id.max(0) as u32,
                    tensor_split: None,
                    rationale: format!(
                        "single GPU '{}' has enough memory for the estimated working set",
                        best_gpu.name
                    ),
                };
            }

            if gpus.len() > 1 && total_gpu_bytes >= estimate.total_bytes {
                let mut weights = gpus
                    .iter()
                    .map(|device| device.memory_gb())
                    .collect::<Vec<_>>();
                if !weights.iter().any(|value| *value > 0.0) {
                    weights = vec![1.0; gpus.len()];
                }

                return ResourcePlan {
                    use_gpu: true,
                    n_gpu_layers: -1,
                    use_mmap: true,
                    use_mlock: false,
                    kv_offload: true,
                    op_offload: true,
                    split_mode: GpuSplitMode::Layer,
                    main_gpu: best_gpu.id.max(0) as u32,
                    tensor_split: Some(weights),
                    rationale: format!(
                        "combined GPU memory across {} devices can hold the estimated working set",
                        gpus.len()
                    ),
                };
            }

            let partial_ratio = (best_gpu.memory_bytes as f64) / (estimate.total_bytes as f64);
            if partial_ratio >= 0.15 {
                let min_layers = max_layers.min(4) as i32;
                let n_gpu_layers = ((partial_ratio * f64::from(max_layers)).round() as i32)
                    .clamp(min_layers, max_layers as i32);
                return ResourcePlan {
                    use_gpu: true,
                    n_gpu_layers,
                    use_mmap: true,
                    use_mlock: false,
                    kv_offload: true,
                    op_offload: true,
                    split_mode: GpuSplitMode::None,
                    main_gpu: best_gpu.id.max(0) as u32,
                    tensor_split: None,
                    rationale: format!(
                        "GPU '{}' cannot fit the full model, using partial offload with approximately {} GPU layers",
                        best_gpu.name, n_gpu_layers
                    ),
                };
            }
        }

        ResourcePlan {
            use_gpu: false,
            n_gpu_layers: 0,
            use_mmap: true,
            use_mlock: cpu_memory_bytes >= estimate.total_bytes.saturating_mul(2),
            kv_offload: false,
            op_offload: false,
            split_mode: GpuSplitMode::None,
            main_gpu: 0,
            tensor_split: None,
            rationale: "falling back to CPU + mmap because no suitable GPU placement was found"
                .to_string(),
        }
    }
}

#[derive(Debug, Clone)]
enum GgufMetadataValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Skipped,
}

fn read_gguf_metadata(path: &Path) -> Result<Option<GgufMetadataSummary>> {
    let file = File::open(path).map_err(|e| {
        LociError::ConfigError(format!("failed to open '{}': {}", path.display(), e))
    })?;
    let mut reader = BufReader::new(file);

    let magic = read_u32_le(&mut reader)?;
    if magic != GGUF_MAGIC {
        return Ok(None);
    }

    let version = read_u32_le(&mut reader)?;
    if !matches!(version, GGUF_VERSION_V2 | GGUF_VERSION_V3) {
        return Ok(None);
    }

    let tensor_count = read_u64_le(&mut reader)?;
    let kv_count = read_u64_le(&mut reader)?;
    let mut values = HashMap::new();

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value_type = read_u32_le(&mut reader)?;
        let value = read_gguf_value(&mut reader, value_type)?;
        if !matches!(value, GgufMetadataValue::Skipped) {
            values.insert(key, value);
        }
    }

    let architecture = match values.get("general.architecture") {
        Some(GgufMetadataValue::String(value)) => Some(value.clone()),
        _ => None,
    };

    let metadata = GgufMetadataSummary {
        version,
        tensor_count,
        architecture: architecture.clone(),
        context_length: find_arch_u32(&values, architecture.as_deref(), "context_length"),
        embedding_length: find_arch_u32(&values, architecture.as_deref(), "embedding_length"),
        block_count: find_arch_u32(&values, architecture.as_deref(), "block_count"),
        vocab_size: find_arch_u32(&values, architecture.as_deref(), "vocab_size"),
        attention_head_count: find_arch_u32(
            &values,
            architecture.as_deref(),
            "attention.head_count",
        ),
        attention_head_count_kv: find_arch_u32(
            &values,
            architecture.as_deref(),
            "attention.head_count_kv",
        ),
        feed_forward_length: find_arch_u32(&values, architecture.as_deref(), "feed_forward_length"),
        expert_count: find_arch_u32(&values, architecture.as_deref(), "expert_count"),
        file_type: find_direct_u32(&values, "general.file_type"),
    };

    if metadata.architecture.is_some()
        || metadata.context_length.is_some()
        || metadata.embedding_length.is_some()
        || metadata.block_count.is_some()
        || metadata.file_type.is_some()
    {
        Ok(Some(metadata))
    } else {
        Ok(None)
    }
}

fn estimate_kv_cache_bytes(metadata: &GgufMetadataSummary, context_size: u32) -> Option<u64> {
    let layers = metadata.block_count? as u64;
    let embedding_length = metadata.embedding_length? as u64;
    if layers == 0 || embedding_length == 0 {
        return None;
    }

    let head_count = metadata.attention_head_count.unwrap_or(0).max(1) as u64;
    let head_count_kv = metadata
        .attention_head_count_kv
        .unwrap_or(metadata.attention_head_count.unwrap_or(0).max(1))
        .max(1) as u64;
    let kv_hidden = embedding_length
        .saturating_mul(head_count_kv)
        .checked_div(head_count)
        .unwrap_or(embedding_length)
        .max(1);
    let kv_cache_bytes = (context_size as u64)
        .saturating_mul(layers)
        .saturating_mul(kv_hidden)
        .saturating_mul(2)
        .saturating_mul(2);

    Some(kv_cache_bytes.max(MIN_KV_CACHE_BYTES))
}

fn find_arch_u32(
    values: &HashMap<String, GgufMetadataValue>,
    architecture: Option<&str>,
    suffix: &str,
) -> Option<u32> {
    if let Some(architecture) = architecture {
        let direct = format!("{architecture}.{suffix}");
        if let Some(value) = find_direct_u32(values, &direct) {
            return Some(value);
        }
    }

    let tail = format!(".{suffix}");
    values.iter().find_map(|(key, value)| {
        if key.ends_with(&tail) {
            numeric_value_to_u32(value)
        } else {
            None
        }
    })
}

fn find_direct_u32(values: &HashMap<String, GgufMetadataValue>, key: &str) -> Option<u32> {
    values.get(key).and_then(numeric_value_to_u32)
}

fn numeric_value_to_u32(value: &GgufMetadataValue) -> Option<u32> {
    match value {
        GgufMetadataValue::Unsigned(value) => (*value).try_into().ok(),
        GgufMetadataValue::Signed(value) => (*value).try_into().ok(),
        GgufMetadataValue::Float(value) if *value >= 0.0 => (*value as u64).try_into().ok(),
        GgufMetadataValue::Bool(value) => Some(u32::from(*value)),
        GgufMetadataValue::String(_) | GgufMetadataValue::Skipped => None,
        GgufMetadataValue::Float(_) => None,
    }
}

fn read_gguf_value<R: Read>(reader: &mut R, value_type: u32) -> Result<GgufMetadataValue> {
    match value_type {
        GGUF_VALUE_TYPE_UINT8 => Ok(GgufMetadataValue::Unsigned(read_u8(reader)? as u64)),
        GGUF_VALUE_TYPE_INT8 => Ok(GgufMetadataValue::Signed(read_i8(reader)? as i64)),
        GGUF_VALUE_TYPE_UINT16 => Ok(GgufMetadataValue::Unsigned(read_u16_le(reader)? as u64)),
        GGUF_VALUE_TYPE_INT16 => Ok(GgufMetadataValue::Signed(read_i16_le(reader)? as i64)),
        GGUF_VALUE_TYPE_UINT32 => Ok(GgufMetadataValue::Unsigned(read_u32_le(reader)? as u64)),
        GGUF_VALUE_TYPE_INT32 => Ok(GgufMetadataValue::Signed(read_i32_le(reader)? as i64)),
        GGUF_VALUE_TYPE_FLOAT32 => Ok(GgufMetadataValue::Float(read_f32_le(reader)? as f64)),
        GGUF_VALUE_TYPE_BOOL => Ok(GgufMetadataValue::Bool(read_u8(reader)? != 0)),
        GGUF_VALUE_TYPE_STRING => Ok(GgufMetadataValue::String(read_gguf_string(reader)?)),
        GGUF_VALUE_TYPE_ARRAY => {
            let inner_type = read_u32_le(reader)?;
            let len = read_u64_le(reader)?;
            skip_gguf_array(reader, inner_type, len)?;
            Ok(GgufMetadataValue::Skipped)
        }
        GGUF_VALUE_TYPE_UINT64 => Ok(GgufMetadataValue::Unsigned(read_u64_le(reader)?)),
        GGUF_VALUE_TYPE_INT64 => Ok(GgufMetadataValue::Signed(read_i64_le(reader)?)),
        GGUF_VALUE_TYPE_FLOAT64 => Ok(GgufMetadataValue::Float(read_f64_le(reader)?)),
        other => Err(LociError::ConfigError(format!(
            "unsupported GGUF metadata value type: {other}"
        ))),
    }
}

fn skip_gguf_array<R: Read>(reader: &mut R, value_type: u32, len: u64) -> Result<()> {
    match value_type {
        GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
            skip_bytes(reader, len)?
        }
        GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
            skip_bytes(reader, len.saturating_mul(2))?
        }
        GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
            skip_bytes(reader, len.saturating_mul(4))?
        }
        GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
            skip_bytes(reader, len.saturating_mul(8))?
        }
        GGUF_VALUE_TYPE_STRING => {
            for _ in 0..len {
                let _ = read_gguf_string(reader)?;
            }
        }
        GGUF_VALUE_TYPE_ARRAY => {
            return Err(LociError::ConfigError(
                "nested GGUF arrays are not supported".to_string(),
            ));
        }
        other => {
            return Err(LociError::ConfigError(format!(
                "unsupported GGUF array value type: {other}"
            )));
        }
    }

    Ok(())
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64_le(reader)? as usize;
    if len > GGUF_MAX_STRING_BYTES {
        return Err(LociError::ConfigError(format!(
            "GGUF string too large: {len} bytes"
        )));
    }
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF string: {e}")))?;
    String::from_utf8(bytes)
        .map_err(|e| LociError::ConfigError(format!("invalid UTF-8 in GGUF metadata: {e}")))
}

fn skip_bytes<R: Read>(reader: &mut R, len: u64) -> Result<()> {
    let mut remaining = len;
    let mut buffer = [0u8; 8192];
    while remaining > 0 {
        let chunk = remaining.min(buffer.len() as u64) as usize;
        reader
            .read_exact(&mut buffer[..chunk])
            .map_err(|e| LociError::ConfigError(format!("failed to skip GGUF bytes: {e}")))?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF u8: {e}")))?;
    Ok(bytes[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16> {
    let mut bytes = [0u8; 2];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF u16: {e}")))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_i16_le<R: Read>(reader: &mut R) -> Result<i16> {
    let mut bytes = [0u8; 2];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF i16: {e}")))?;
    Ok(i16::from_le_bytes(bytes))
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF u32: {e}")))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32_le<R: Read>(reader: &mut R) -> Result<i32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF i32: {e}")))?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF u64: {e}")))?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64_le<R: Read>(reader: &mut R) -> Result<i64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF i64: {e}")))?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f32_le<R: Read>(reader: &mut R) -> Result<f32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF f32: {e}")))?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_f64_le<R: Read>(reader: &mut R) -> Result<f64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| LociError::ConfigError(format!("failed to read GGUF f64: {e}")))?;
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    enum MockGgufValue<'a> {
        String(&'a str),
        U32(u32),
    }

    fn cpu(memory_gb: u64) -> DeviceInfo {
        DeviceInfo {
            id: 0,
            name: "CPU".to_string(),
            memory_bytes: memory_gb * 1024 * 1024 * 1024,
            device_type: DeviceType::CPU,
            compute_capability: 0.0,
            available: true,
        }
    }

    fn gpu(id: i32, name: &str, memory_gb: u64) -> DeviceInfo {
        DeviceInfo {
            id,
            name: name.to_string(),
            memory_bytes: memory_gb * 1024 * 1024 * 1024,
            device_type: DeviceType::CUDA,
            compute_capability: 8.0,
            available: true,
        }
    }

    fn write_mock_gguf(entries: &[(&str, MockGgufValue<'_>)]) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-resource-plan-{nonce}.gguf"));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());

        for (key, value) in entries {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            match value {
                MockGgufValue::String(value) => {
                    bytes.extend_from_slice(&GGUF_VALUE_TYPE_STRING.to_le_bytes());
                    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
                    bytes.extend_from_slice(value.as_bytes());
                }
                MockGgufValue::U32(value) => {
                    bytes.extend_from_slice(&GGUF_VALUE_TYPE_UINT32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        fs::write(&path, bytes).expect("write mock gguf");
        path
    }

    #[test]
    fn planner_prefers_single_gpu_when_it_fits() {
        let estimate = ModelResourceEstimate {
            model_bytes: 3 * 1024 * 1024 * 1024,
            kv_cache_bytes: 128 * 1024 * 1024,
            working_set_bytes: 1024 * 1024 * 1024,
            total_bytes: 4 * 1024 * 1024 * 1024,
            context_size: 4096,
            metadata_source: EstimateMetadataSource::FileSizeOnly,
            gguf_metadata: None,
        };

        let plan = ResourcePlanner::plan_for_estimate(&estimate, &[gpu(0, "GPU0", 8), cpu(32)]);
        assert!(plan.use_gpu);
        assert_eq!(plan.n_gpu_layers, -1);
        assert_eq!(plan.split_mode, GpuSplitMode::None);
    }

    #[test]
    fn planner_uses_multi_gpu_split_when_combined_memory_fits() {
        let estimate = ModelResourceEstimate {
            model_bytes: 10 * 1024 * 1024 * 1024,
            kv_cache_bytes: 256 * 1024 * 1024,
            working_set_bytes: 2 * 1024 * 1024 * 1024,
            total_bytes: 12 * 1024 * 1024 * 1024,
            context_size: 8192,
            metadata_source: EstimateMetadataSource::FileSizeOnly,
            gguf_metadata: None,
        };

        let plan = ResourcePlanner::plan_for_estimate(
            &estimate,
            &[gpu(0, "GPU0", 8), gpu(1, "GPU1", 8), cpu(64)],
        );
        assert!(plan.use_gpu);
        assert_eq!(plan.n_gpu_layers, -1);
        assert_eq!(plan.split_mode, GpuSplitMode::Layer);
        assert_eq!(plan.tensor_split, Some(vec![8.0, 8.0]));
    }

    #[test]
    fn planner_falls_back_to_cpu_when_gpu_is_too_small() {
        let estimate = ModelResourceEstimate {
            model_bytes: 12 * 1024 * 1024 * 1024,
            kv_cache_bytes: 256 * 1024 * 1024,
            working_set_bytes: 4 * 1024 * 1024 * 1024,
            total_bytes: 16 * 1024 * 1024 * 1024,
            context_size: 4096,
            metadata_source: EstimateMetadataSource::FileSizeOnly,
            gguf_metadata: None,
        };

        let plan = ResourcePlanner::plan_for_estimate(&estimate, &[gpu(0, "GPU0", 1), cpu(64)]);
        assert!(!plan.use_gpu);
        assert_eq!(plan.n_gpu_layers, 0);
        assert!(plan.use_mmap);
    }

    #[test]
    fn estimate_model_requirements_reads_gguf_metadata() {
        let path = write_mock_gguf(&[
            ("general.architecture", MockGgufValue::String("llama")),
            ("general.file_type", MockGgufValue::U32(15)),
            ("llama.context_length", MockGgufValue::U32(8192)),
            ("llama.embedding_length", MockGgufValue::U32(1024)),
            ("llama.block_count", MockGgufValue::U32(24)),
            ("llama.attention.head_count", MockGgufValue::U32(16)),
            ("llama.attention.head_count_kv", MockGgufValue::U32(8)),
        ]);

        let estimate = ResourcePlanner::estimate_model_requirements(&path, 4096)
            .expect("estimate should succeed");

        assert_eq!(
            estimate.metadata_source,
            EstimateMetadataSource::GgufMetadata
        );
        assert_eq!(
            estimate
                .gguf_metadata
                .as_ref()
                .and_then(|metadata| metadata.architecture.as_deref()),
            Some("llama")
        );
        assert_eq!(estimate.layer_count(), Some(24));
        assert!(estimate.kv_cache_bytes > MIN_KV_CACHE_BYTES);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn planner_uses_detected_layer_count_for_partial_gpu_offload() {
        let estimate = ModelResourceEstimate {
            model_bytes: 10 * 1024 * 1024 * 1024,
            kv_cache_bytes: 2 * 1024 * 1024 * 1024,
            working_set_bytes: 2 * 1024 * 1024 * 1024,
            total_bytes: 14 * 1024 * 1024 * 1024,
            context_size: 8192,
            metadata_source: EstimateMetadataSource::GgufMetadata,
            gguf_metadata: Some(GgufMetadataSummary {
                block_count: Some(80),
                ..GgufMetadataSummary::default()
            }),
        };

        let plan = ResourcePlanner::plan_for_estimate(&estimate, &[gpu(0, "GPU0", 6), cpu(64)]);
        assert!(plan.use_gpu);
        assert_eq!(plan.n_gpu_layers, 34);
    }
}
