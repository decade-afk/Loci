//! GGUF-first helpers and lightweight architecture registry for Loci.
//!
//! This crate is intentionally small but concrete: it centralizes the MVP
//! architecture support policy and exposes a registry-like API that other
//! crates can query without duplicating GGUF-specific logic.

use loci_protocol::{ModelDescriptor, ModelFormat};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

pub const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");

/// Minimal GGUF header used by the MVP readiness checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

/// Minimal GGUF metadata probe used by the MVP to self-configure model registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadataSummary {
    pub header: GgufHeader,
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub general_name: Option<String>,
    pub alignment: Option<u64>,
    pub selected_metadata: BTreeMap<String, String>,
    pub tokenizer_tokens: Option<Vec<String>>,
    pub tokenizer_types: Option<Vec<u32>>,
    pub tensor_table: GgufTensorTableSummary,
}

/// Minimal tensor-info probe that preserves enough structure for backend planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfoSummary {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_dtype: u32,
    pub offset: u64,
}

/// Minimal tensor-table summary extracted from the GGUF preamble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorTableSummary {
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub preview_names: Vec<String>,
    pub candidate_names: Vec<String>,
    pub tensor_infos: Vec<GgufTensorInfoSummary>,
    pub first_tensor: Option<GgufTensorInfoSummary>,
    pub last_tensor: Option<GgufTensorInfoSummary>,
    pub max_rank: u32,
    pub attention_tensor_count: u32,
    pub ffn_tensor_count: u32,
    pub norm_tensor_count: u32,
    pub contains_output_weight: bool,
    pub contains_token_embedding: bool,
}

/// A small prefix of a real GGUF tensor payload.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensorPrefix {
    pub info: GgufTensorInfoSummary,
    pub values_f32: Vec<f32>,
}

#[derive(Debug)]
pub enum GgufHeaderError {
    Io(std::io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    UnexpectedEof(&'static str),
}

impl fmt::Display for GgufHeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic: 0x{magic:08x}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version: {version}")
            }
            Self::UnexpectedEof(field) => {
                write!(f, "unexpected end of file while reading GGUF {field}")
            }
        }
    }
}

impl std::error::Error for GgufHeaderError {}

impl From<std::io::Error> for GgufHeaderError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Minimal architecture descriptor for the GGUF-first MVP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufArchitectureSpec {
    pub canonical_name: &'static str,
    pub aliases: &'static [&'static str],
    pub supports_kv_cache: bool,
    pub supports_sliding_window: bool,
    pub default_context_length: u32,
}

/// Architectural support policy for the GGUF-first MVP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSupportProfile {
    pub supported_architectures: Vec<&'static str>,
    pub primary_format: ModelFormat,
}

const LLAMA_SPEC: GgufArchitectureSpec = GgufArchitectureSpec {
    canonical_name: "llama",
    aliases: &["llama2", "llama3", "llama-2", "llama-3"],
    supports_kv_cache: true,
    supports_sliding_window: false,
    default_context_length: 8192,
};

const MISTRAL_SPEC: GgufArchitectureSpec = GgufArchitectureSpec {
    canonical_name: "mistral",
    aliases: &["mixtral", "ministral"],
    supports_kv_cache: true,
    supports_sliding_window: true,
    default_context_length: 32768,
};

const QWEN_SPEC: GgufArchitectureSpec = GgufArchitectureSpec {
    canonical_name: "qwen",
    aliases: &["qwen2", "qwen2.5", "qwen3"],
    supports_kv_cache: true,
    supports_sliding_window: true,
    default_context_length: 32768,
};

const REGISTRY: &[GgufArchitectureSpec] = &[LLAMA_SPEC, MISTRAL_SPEC, QWEN_SPEC];

/// Returns the default GGUF support profile for the current MVP.
pub fn support_profile() -> GgufSupportProfile {
    GgufSupportProfile {
        supported_architectures: REGISTRY.iter().map(|spec| spec.canonical_name).collect(),
        primary_format: ModelFormat::Gguf,
    }
}

/// Returns the architecture spec matching the provided name or alias.
pub fn resolve_architecture(name: &str) -> Option<&'static GgufArchitectureSpec> {
    REGISTRY.iter().find(|spec| {
        spec.canonical_name.eq_ignore_ascii_case(name)
            || spec
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    })
}

/// Returns whether a model falls inside the current GGUF-first support window.
pub fn is_primary_supported_model(model: &ModelDescriptor) -> bool {
    if model.inferred_format() != ModelFormat::Gguf {
        return false;
    }

    resolve_architecture(&model.architecture).is_some()
}

/// Returns the canonical architecture name when a model is recognized.
pub fn canonical_architecture_name(model: &ModelDescriptor) -> Option<&'static str> {
    resolve_architecture(&model.architecture).map(|spec| spec.canonical_name)
}

/// Returns the default context length the registry would use for a model.
pub fn suggested_context_length(model: &ModelDescriptor) -> Option<u32> {
    resolve_architecture(&model.architecture).map(|spec| spec.default_context_length)
}

/// Reads only the GGUF file header needed by the current MVP path.
pub fn read_gguf_header(path: impl AsRef<Path>) -> Result<GgufHeader, GgufHeaderError> {
    let mut file = File::open(path)?;
    let mut prefix = [0_u8; 24];
    let read = file.read(&mut prefix)?;
    parse_gguf_header_bytes(&prefix[..read])
}

/// Reads a small subset of GGUF metadata used by the current runtime bootstrap path.
pub fn read_gguf_metadata_summary(
    path: impl AsRef<Path>,
) -> Result<GgufMetadataSummary, GgufHeaderError> {
    let bytes = std::fs::read(path)?;
    parse_gguf_metadata_summary_bytes(&bytes)
}

/// Reads a named GGUF tensor prefix and converts the first values into f32.
pub fn read_gguf_tensor_prefix_f32(
    path: impl AsRef<Path>,
    tensor_name: &str,
    max_elements: usize,
) -> Result<Option<GgufTensorPrefix>, GgufHeaderError> {
    let bytes = std::fs::read(path)?;
    parse_gguf_tensor_prefix_f32_bytes(&bytes, tensor_name, max_elements)
}

/// Parses a GGUF header from an in-memory buffer.
pub fn parse_gguf_header_bytes(bytes: &[u8]) -> Result<GgufHeader, GgufHeaderError> {
    let mut cursor = Cursor::new(bytes);
    let magic = read_u32(&mut cursor, "magic")?;
    if magic != GGUF_MAGIC {
        return Err(GgufHeaderError::InvalidMagic(magic));
    }

    let version = read_u32(&mut cursor, "version")?;
    let (tensor_count, metadata_count) = match version {
        1 => (
            read_u32(&mut cursor, "tensor_count")? as u64,
            read_u32(&mut cursor, "metadata_count")? as u64,
        ),
        2 | 3 => (
            read_u64(&mut cursor, "tensor_count")?,
            read_u64(&mut cursor, "metadata_count")?,
        ),
        _ => return Err(GgufHeaderError::UnsupportedVersion(version)),
    };

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_count,
    })
}

/// Parses the GGUF header and selected metadata keys from an in-memory buffer.
pub fn parse_gguf_metadata_summary_bytes(
    bytes: &[u8],
) -> Result<GgufMetadataSummary, GgufHeaderError> {
    let mut cursor = Cursor::new(bytes);
    let header = parse_header_from_cursor(&mut cursor)?;
    let mut architecture = None;
    let mut context_length = None;
    let mut general_name = None;
    let mut alignment = None;
    let mut selected_metadata = BTreeMap::new();
    let mut tokenizer_tokens = None;
    let mut tokenizer_types = None;

    for _ in 0..header.metadata_count {
        let key = read_sized_string(&mut cursor, header.version, "metadata_key")?;
        let value_type = read_u32(&mut cursor, "metadata_value_type")?;
        let value = read_metadata_value(&mut cursor, header.version, value_type)?;

        if key == "general.architecture" {
            if let GgufMetadataValue::String(text) = &value {
                architecture = Some(text.clone());
            }
        } else if key == "general.name" {
            if let GgufMetadataValue::String(text) = &value {
                general_name = Some(text.clone());
            }
        } else if key.ends_with(".context_length") {
            if let Some(length) = value.as_u32() {
                context_length = Some(length);
            }
        } else if key == "general.alignment" {
            alignment = value.as_u64();
        } else if key == "tokenizer.ggml.tokens" {
            tokenizer_tokens = value.as_string_array();
        } else if key == "tokenizer.ggml.token_type" {
            tokenizer_types = value.as_u32_array();
        }

        if should_capture_metadata_key(&key) {
            selected_metadata.insert(key.clone(), value.to_compact_string());
        }
    }

    let tensor_table = read_tensor_table_summary(
        &mut cursor,
        header.version,
        header.tensor_count,
        alignment.unwrap_or(32),
    )?;

    Ok(GgufMetadataSummary {
        header,
        architecture,
        context_length,
        general_name,
        alignment,
        selected_metadata,
        tokenizer_tokens,
        tokenizer_types,
        tensor_table,
    })
}

/// Parses an in-memory GGUF buffer and returns a named tensor prefix when available.
pub fn parse_gguf_tensor_prefix_f32_bytes(
    bytes: &[u8],
    tensor_name: &str,
    max_elements: usize,
) -> Result<Option<GgufTensorPrefix>, GgufHeaderError> {
    let mut cursor = Cursor::new(bytes);
    let header = parse_header_from_cursor(&mut cursor)?;
    let mut alignment = None;

    for _ in 0..header.metadata_count {
        let key = read_sized_string(&mut cursor, header.version, "metadata_key")?;
        let value_type = read_u32(&mut cursor, "metadata_value_type")?;
        let value = read_metadata_value(&mut cursor, header.version, value_type)?;
        if key == "general.alignment" {
            alignment = value.as_u64();
        }
    }

    let tensor_table = read_tensor_table_summary(
        &mut cursor,
        header.version,
        header.tensor_count,
        alignment.unwrap_or(32),
    )?;
    let info = match tensor_table
        .tensor_infos
        .iter()
        .find(|tensor| tensor.name == tensor_name)
        .cloned()
    {
        Some(info) => info,
        None => return Ok(None),
    };
    let values_f32 = read_tensor_prefix_values_f32(bytes, &tensor_table, &info, max_elements)?;
    Ok(Some(GgufTensorPrefix { info, values_f32 }))
}

fn parse_header_from_cursor(cursor: &mut Cursor<&[u8]>) -> Result<GgufHeader, GgufHeaderError> {
    let magic = read_u32(cursor, "magic")?;
    if magic != GGUF_MAGIC {
        return Err(GgufHeaderError::InvalidMagic(magic));
    }

    let version = read_u32(cursor, "version")?;
    let (tensor_count, metadata_count) = match version {
        1 => (
            read_u32(cursor, "tensor_count")? as u64,
            read_u32(cursor, "metadata_count")? as u64,
        ),
        2 | 3 => (
            read_u64(cursor, "tensor_count")?,
            read_u64(cursor, "metadata_count")?,
        ),
        _ => return Err(GgufHeaderError::UnsupportedVersion(version)),
    };

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_count,
    })
}

#[derive(Debug, Clone, PartialEq)]
enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufMetadataValue {
    fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(value) => Some(*value as u32),
            Self::U16(value) => Some(*value as u32),
            Self::U32(value) => Some(*value),
            Self::U64(value) => (*value).try_into().ok(),
            Self::I8(value) if *value >= 0 => Some(*value as u32),
            Self::I16(value) if *value >= 0 => Some(*value as u32),
            Self::I32(value) if *value >= 0 => Some(*value as u32),
            Self::I64(value) if *value >= 0 => (*value as u64).try_into().ok(),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(*value as u64),
            Self::U16(value) => Some(*value as u64),
            Self::U32(value) => Some(*value as u64),
            Self::U64(value) => Some(*value),
            Self::I8(value) if *value >= 0 => Some(*value as u64),
            Self::I16(value) if *value >= 0 => Some(*value as u64),
            Self::I32(value) if *value >= 0 => Some(*value as u64),
            Self::I64(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    fn to_compact_string(&self) -> String {
        match self {
            Self::U8(value) => value.to_string(),
            Self::I8(value) => value.to_string(),
            Self::U16(value) => value.to_string(),
            Self::I16(value) => value.to_string(),
            Self::U32(value) => value.to_string(),
            Self::I32(value) => value.to_string(),
            Self::F32(value) => format!("{value}"),
            Self::Bool(value) => value.to_string(),
            Self::String(value) => value.clone(),
            Self::Array(values) => format!("array[len={}]", values.len()),
            Self::U64(value) => value.to_string(),
            Self::I64(value) => value.to_string(),
            Self::F64(value) => format!("{value}"),
        }
    }

    fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(values) => Some(
                values
                    .iter()
                    .filter_map(|value| match value {
                        Self::String(text) => Some(text.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            Self::Array(values) => Some(
                values
                    .iter()
                    .filter_map(|value| value.as_u32())
                    .collect(),
            ),
            _ => None,
        }
    }
}

fn should_capture_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "general.architecture" | "general.name" | "general.alignment"
    ) || key.ends_with(".context_length")
        || key.ends_with(".block_count")
        || key.ends_with(".embedding_length")
        || key.ends_with(".feed_forward_length")
        || key.ends_with(".attention.head_count")
        || key.ends_with(".attention.head_count_kv")
        || key.ends_with(".rope.dimension_count")
}

fn read_tensor_table_summary(
    cursor: &mut Cursor<&[u8]>,
    version: u32,
    tensor_count: u64,
    alignment: u64,
) -> Result<GgufTensorTableSummary, GgufHeaderError> {
    let mut tensor_infos = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_sized_string(cursor, version, "tensor_name")?;
        let dimension_count = read_u32(cursor, "tensor_dimension_count")?;
        let mut dimensions = Vec::with_capacity(dimension_count as usize);
        for _ in 0..dimension_count {
            dimensions.push(read_length(cursor, version, "tensor_dimension")?);
        }
        dimensions.reverse();
        let ggml_dtype = read_u32(cursor, "tensor_ggml_dtype")?;
        let offset = read_u64(cursor, "tensor_offset")?;
        tensor_infos.push(GgufTensorInfoSummary {
            name,
            dimensions,
            ggml_dtype,
            offset,
        });
    }

    let alignment = alignment.max(1);
    let position = cursor.position();
    let tensor_data_offset = position.div_ceil(alignment) * alignment;
    let preview_names = tensor_infos
        .iter()
        .take(6)
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let candidate_names = tensor_infos
        .iter()
        .filter(|tensor| {
            tensor.name.contains("norm")
                || tensor.name.contains("output")
                || tensor.name.contains("embd")
                || tensor.name.contains("embed")
                || tensor.name.contains("lm_head")
        })
        .take(32)
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let attention_tensor_count = tensor_infos
        .iter()
        .filter(|tensor| tensor.name.contains("attn"))
        .count() as u32;
    let ffn_tensor_count = tensor_infos
        .iter()
        .filter(|tensor| tensor.name.contains("ffn"))
        .count() as u32;
    let norm_tensor_count = tensor_infos
        .iter()
        .filter(|tensor| tensor.name.contains("norm"))
        .count() as u32;
    let contains_output_weight = tensor_infos
        .iter()
        .any(|tensor| tensor.name == "output.weight");
    let contains_token_embedding = tensor_infos.iter().any(|tensor| {
        tensor.name == "token_embd.weight"
            || tensor.name == "tok_embeddings.weight"
            || tensor.name.ends_with("embed_tokens.weight")
    });
    let max_rank = tensor_infos
        .iter()
        .map(|tensor| tensor.dimensions.len() as u32)
        .max()
        .unwrap_or_default();

    Ok(GgufTensorTableSummary {
        alignment,
        tensor_data_offset,
        preview_names,
        candidate_names,
        tensor_infos: tensor_infos.clone(),
        first_tensor: tensor_infos.first().cloned(),
        last_tensor: tensor_infos.last().cloned(),
        max_rank,
        attention_tensor_count,
        ffn_tensor_count,
        norm_tensor_count,
        contains_output_weight,
        contains_token_embedding,
    })
}

fn read_tensor_prefix_values_f32(
    bytes: &[u8],
    table: &GgufTensorTableSummary,
    info: &GgufTensorInfoSummary,
    max_elements: usize,
) -> Result<Vec<f32>, GgufHeaderError> {
    let available = tensor_element_count(&info.dimensions) as usize;
    let start = table.tensor_data_offset.saturating_add(info.offset) as usize;
    if start >= bytes.len() {
        return Ok(Vec::new());
    }
    match info.ggml_dtype {
        0 => read_plain_tensor_prefix_values_f32(bytes, start, available, max_elements, 4, |chunk| {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        }),
        1 => read_plain_tensor_prefix_values_f32(bytes, start, available, max_elements, 2, |chunk| {
            f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
        }),
        2 => read_q4_0_tensor_prefix_values_f32(bytes, start, available, max_elements),
        8 => read_q8_0_tensor_prefix_values_f32(bytes, start, available, max_elements),
        _ => Err(GgufHeaderError::UnexpectedEof("unsupported_tensor_dtype")),
    }
}

fn read_plain_tensor_prefix_values_f32(
    bytes: &[u8],
    start: usize,
    available: usize,
    max_elements: usize,
    bytes_per_elem: usize,
    decode: impl Fn(&[u8]) -> f32,
) -> Result<Vec<f32>, GgufHeaderError> {
    let available_by_file = (bytes.len() - start) / bytes_per_elem;
    let elements = available.min(max_elements).min(available_by_file);
    if elements == 0 {
        return Ok(Vec::new());
    }
    let end = start.saturating_add(elements.saturating_mul(bytes_per_elem));
    if end > bytes.len() {
        return Err(GgufHeaderError::UnexpectedEof("tensor_payload"));
    }

    Ok(bytes[start..end]
        .chunks_exact(bytes_per_elem)
        .map(decode)
        .collect())
}

fn read_q4_0_tensor_prefix_values_f32(
    bytes: &[u8],
    start: usize,
    available: usize,
    max_elements: usize,
) -> Result<Vec<f32>, GgufHeaderError> {
    read_quantized_tensor_prefix_values_f32(
        bytes,
        start,
        available,
        max_elements,
        32,
        18,
        dequantize_block_q4_0,
    )
}

fn read_q8_0_tensor_prefix_values_f32(
    bytes: &[u8],
    start: usize,
    available: usize,
    max_elements: usize,
) -> Result<Vec<f32>, GgufHeaderError> {
    read_quantized_tensor_prefix_values_f32(
        bytes,
        start,
        available,
        max_elements,
        32,
        34,
        dequantize_block_q8_0,
    )
}

fn read_quantized_tensor_prefix_values_f32(
    bytes: &[u8],
    start: usize,
    available: usize,
    max_elements: usize,
    block_elements: usize,
    block_bytes: usize,
    decode_block: impl Fn(&[u8], &mut Vec<f32>) -> Result<(), GgufHeaderError>,
) -> Result<Vec<f32>, GgufHeaderError> {
    let requested = available.min(max_elements);
    if requested == 0 {
        return Ok(Vec::new());
    }

    let blocks_by_tensor = available.div_ceil(block_elements);
    let blocks_by_request = requested.div_ceil(block_elements);
    let blocks_by_file = (bytes.len() - start) / block_bytes;
    let blocks = blocks_by_tensor.min(blocks_by_request).min(blocks_by_file);
    if blocks == 0 {
        return Ok(Vec::new());
    }

    let end = start.saturating_add(blocks.saturating_mul(block_bytes));
    if end > bytes.len() {
        return Err(GgufHeaderError::UnexpectedEof("tensor_payload"));
    }

    let mut values = Vec::with_capacity(blocks.saturating_mul(block_elements));
    for block in bytes[start..end].chunks_exact(block_bytes) {
        decode_block(block, &mut values)?;
    }
    values.truncate(requested);
    Ok(values)
}

fn dequantize_block_q4_0(block: &[u8], out: &mut Vec<f32>) -> Result<(), GgufHeaderError> {
    if block.len() != 18 {
        return Err(GgufHeaderError::UnexpectedEof("q4_0_block"));
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..];
    for byte in qs {
        let low = ((byte & 0x0f) as i16 - 8) as f32;
        out.push(low * d);
    }
    for byte in qs {
        let high = ((byte >> 4) as i16 - 8) as f32;
        out.push(high * d);
    }
    Ok(())
}

fn dequantize_block_q8_0(block: &[u8], out: &mut Vec<f32>) -> Result<(), GgufHeaderError> {
    if block.len() != 34 {
        return Err(GgufHeaderError::UnexpectedEof("q8_0_block"));
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for value in &block[2..] {
        out.push((*value as i8) as f32 * d);
    }
    Ok(())
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

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let fbits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut frac = frac;
            let mut exp = 127 - 15 + 1;
            while (frac & 0x0400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x03ff;
            (sign << 31) | (exp << 23) | (frac << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        let exp = exp + (127 - 15);
        (sign << 31) | (exp << 23) | (frac << 13)
    };
    f32::from_bits(fbits)
}

fn read_metadata_value(
    cursor: &mut Cursor<&[u8]>,
    version: u32,
    value_type: u32,
) -> Result<GgufMetadataValue, GgufHeaderError> {
    match value_type {
        0 => Ok(GgufMetadataValue::U8(read_u8(cursor, "metadata_u8")?)),
        1 => Ok(GgufMetadataValue::I8(read_i8(cursor, "metadata_i8")?)),
        2 => Ok(GgufMetadataValue::U16(read_u16(cursor, "metadata_u16")?)),
        3 => Ok(GgufMetadataValue::I16(read_i16(cursor, "metadata_i16")?)),
        4 => Ok(GgufMetadataValue::U32(read_u32(cursor, "metadata_u32")?)),
        5 => Ok(GgufMetadataValue::I32(read_i32(cursor, "metadata_i32")?)),
        6 => Ok(GgufMetadataValue::F32(read_f32(cursor, "metadata_f32")?)),
        7 => Ok(GgufMetadataValue::Bool(read_bool(cursor, "metadata_bool")?)),
        8 => Ok(GgufMetadataValue::String(read_sized_string(
            cursor,
            version,
            "metadata_string",
        )?)),
        9 => {
            let element_type = read_u32(cursor, "metadata_array_element_type")?;
            let count = read_length(cursor, version, "metadata_array_len")?;
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_metadata_value(cursor, version, element_type)?);
            }
            Ok(GgufMetadataValue::Array(values))
        }
        10 => Ok(GgufMetadataValue::U64(read_u64(cursor, "metadata_u64")?)),
        11 => Ok(GgufMetadataValue::I64(read_i64(cursor, "metadata_i64")?)),
        12 => Ok(GgufMetadataValue::F64(read_f64(cursor, "metadata_f64")?)),
        other => Err(GgufHeaderError::UnexpectedEof(match other {
            _ => "unsupported_metadata_value",
        })),
    }
}

fn read_length(
    reader: &mut Cursor<&[u8]>,
    version: u32,
    field: &'static str,
) -> Result<u64, GgufHeaderError> {
    match version {
        1 => Ok(read_u32(reader, field)? as u64),
        2 | 3 => read_u64(reader, field),
        _ => Err(GgufHeaderError::UnsupportedVersion(version)),
    }
}

fn read_sized_string(
    reader: &mut Cursor<&[u8]>,
    version: u32,
    field: &'static str,
) -> Result<String, GgufHeaderError> {
    let len = read_length(reader, version, field)?;
    let mut bytes = vec![0_u8; len as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn read_u8(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<u8, GgufHeaderError> {
    let mut bytes = [0_u8; 1];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(bytes[0])
}

fn read_i8(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<i8, GgufHeaderError> {
    Ok(read_u8(reader, field)? as i8)
}

fn read_u16(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<u16, GgufHeaderError> {
    let mut bytes = [0_u8; 2];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_i16(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<i16, GgufHeaderError> {
    Ok(read_u16(reader, field)? as i16)
}

fn read_u32(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<u32, GgufHeaderError> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<u64, GgufHeaderError> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i32(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<i32, GgufHeaderError> {
    Ok(read_u32(reader, field)? as i32)
}

fn read_i64(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<i64, GgufHeaderError> {
    Ok(read_u64(reader, field)? as i64)
}

fn read_f32(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<f32, GgufHeaderError> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_f64(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<f64, GgufHeaderError> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| match error.kind() {
            std::io::ErrorKind::UnexpectedEof => GgufHeaderError::UnexpectedEof(field),
            _ => GgufHeaderError::Io(error),
        })?;
    Ok(f64::from_le_bytes(bytes))
}

fn read_bool(reader: &mut Cursor<&[u8]>, field: &'static str) -> Result<bool, GgufHeaderError> {
    Ok(read_u8(reader, field)? != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn model(path: &str, architecture: &str) -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from(path),
            architecture: architecture.to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        }
    }

    #[test]
    fn profile_is_gguf_first() {
        let profile = support_profile();
        assert_eq!(profile.primary_format, ModelFormat::Gguf);
        assert!(profile.supported_architectures.contains(&"llama"));
        assert!(profile.supported_architectures.contains(&"mistral"));
        assert!(profile.supported_architectures.contains(&"qwen"));
    }

    #[test]
    fn resolves_aliases_to_canonical_architecture() {
        let spec = resolve_architecture("qwen2.5").expect("spec");
        assert_eq!(spec.canonical_name, "qwen");
        assert!(spec.supports_sliding_window);
    }

    #[test]
    fn primary_support_requires_gguf_and_supported_architecture() {
        assert!(is_primary_supported_model(&model(
            "D:/models/demo.gguf",
            "llama"
        )));
        assert!(is_primary_supported_model(&model(
            "D:/models/demo.gguf",
            "qwen2.5"
        )));
        assert!(!is_primary_supported_model(&model(
            "D:/models/demo.onnx",
            "llama"
        )));
        assert!(!is_primary_supported_model(&model(
            "D:/models/demo.gguf",
            "phi"
        )));
    }

    #[test]
    fn suggests_context_length_from_registry() {
        let model = model("D:/models/demo.gguf", "mistral");
        assert_eq!(canonical_architecture_name(&model), Some("mistral"));
        assert_eq!(suggested_context_length(&model), Some(32768));
    }

    #[test]
    fn parses_v3_header_counts() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        bytes.extend_from_slice(&11_u64.to_le_bytes());

        let header = parse_gguf_header_bytes(&bytes).expect("header");
        assert_eq!(
            header,
            GgufHeader {
                version: 3,
                tensor_count: 7,
                metadata_count: 11,
            }
        );
    }

    #[test]
    fn parses_v1_header_counts() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&5_u32.to_le_bytes());

        let header = parse_gguf_header_bytes(&bytes).expect("header");
        assert_eq!(header.version, 1);
        assert_eq!(header.tensor_count, 3);
        assert_eq!(header.metadata_count, 5);
    }

    #[test]
    fn rejects_invalid_magic() {
        let error = parse_gguf_header_bytes(&0_u32.to_le_bytes()).expect_err("error");
        assert!(matches!(error, GgufHeaderError::InvalidMagic(0)));
    }

    #[test]
    fn rejects_short_headers() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());

        let error = parse_gguf_header_bytes(&bytes).expect_err("error");
        assert!(matches!(
            error,
            GgufHeaderError::UnexpectedEof("tensor_count")
        ));
    }

    #[test]
    fn reads_header_from_file() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&9_u64.to_le_bytes());
        bytes.extend_from_slice(&13_u64.to_le_bytes());

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-gguf-header-{unique}.gguf"));
        fs::write(&path, bytes).expect("write");

        let header = read_gguf_header(&path).expect("header");
        assert_eq!(header.tensor_count, 9);
        assert_eq!(header.metadata_count, 13);

        fs::remove_file(path).expect("cleanup");
    }

    #[test]
    fn parses_selected_metadata_summary() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());

        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"qwen2";
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
        let value = b"Qwen Demo";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let summary = parse_gguf_metadata_summary_bytes(&bytes).expect("summary");
        assert_eq!(summary.architecture.as_deref(), Some("qwen2"));
        assert_eq!(summary.context_length, Some(32768));
        assert_eq!(summary.general_name.as_deref(), Some("Qwen Demo"));
        assert_eq!(
            summary.selected_metadata.get("qwen2.context_length"),
            Some(&"32768".to_string())
        );
    }

    #[test]
    fn parses_and_skips_metadata_arrays() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

        let key = b"tokenizer.ggml.tokens";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        let value = b"ab";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
        let value = b"cd";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let key = b"general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        let value = b"llama";
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);

        let summary = parse_gguf_metadata_summary_bytes(&bytes).expect("summary");
        assert_eq!(summary.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn reads_q8_0_tensor_prefix_values() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x3800_u16.to_le_bytes());
        payload.extend((0..32).map(|value| value as i8 as u8));

        let bytes = write_single_tensor_gguf("output.weight", &[32], 8, &payload);
        let tensor = parse_gguf_tensor_prefix_f32_bytes(&bytes, "output.weight", 6)
            .expect("tensor")
            .expect("prefix");

        assert_eq!(tensor.info.ggml_dtype, 8);
        assert_eq!(tensor.values_f32, vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn reads_q4_0_tensor_prefix_values() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x3400_u16.to_le_bytes());
        payload.extend(std::iter::repeat_n(0x98_u8, 16));

        let bytes = write_single_tensor_gguf("output.weight", &[32], 2, &payload);
        let tensor = parse_gguf_tensor_prefix_f32_bytes(&bytes, "output.weight", 20)
            .expect("tensor")
            .expect("prefix");

        assert_eq!(tensor.info.ggml_dtype, 2);
        assert_eq!(tensor.values_f32.len(), 20);
        assert!(tensor.values_f32[..16].iter().all(|value| *value == 0.0));
        assert!(tensor.values_f32[16..].iter().all(|value| *value == 0.25));
    }

    #[test]
    fn reads_real_qwen_fp16_gguf_metadata_when_available() {
        let path = PathBuf::from(
            "D:/Code/Loci/tmp/models/qwen2.5-0.5b-instruct-gguf-ms/qwen2.5-0.5b-instruct-fp16.gguf",
        );
        if !path.exists() {
            return;
        }

        let bytes = fs::read(&path).expect("read");
        if bytes.starts_with(b"version https://git-lfs.github.com/spec") {
            return;
        }

        let summary = parse_gguf_metadata_summary_bytes(&bytes).expect("summary");
        assert!(summary.header.tensor_count > 0);
        assert!(summary.header.metadata_count > 0);
        assert!(summary.tensor_table.tensor_data_offset > 0);
        assert!(!summary.tensor_table.tensor_infos.is_empty());
    }

    fn write_single_tensor_gguf(
        name: &str,
        dimensions: &[u64],
        ggml_dtype: u32,
        payload: &[u8],
    ) -> Vec<u8> {
        let version = 3_u32;
        let alignment = 32_u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());

        let key = b"general.alignment";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&alignment.to_le_bytes());

        write_tensor_info(&mut bytes, version, name, dimensions, ggml_dtype, 0);
        while bytes.len() % alignment as usize != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(payload);
        bytes
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
}
