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
    pub first_tensor: Option<GgufTensorInfoSummary>,
    pub last_tensor: Option<GgufTensorInfoSummary>,
    pub max_rank: u32,
    pub attention_tensor_count: u32,
    pub ffn_tensor_count: u32,
    pub norm_tensor_count: u32,
    pub contains_output_weight: bool,
    pub contains_token_embedding: bool,
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
        tensor_table,
    })
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
}
