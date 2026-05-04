//! Embedded integration helpers for direct in-process use.

use crate::error::{LociError, Result};
use loci_protocol::ModelDescriptor;
use std::path::PathBuf;

/// Options used when registering a local model directly inside a host process.
#[derive(Debug, Clone, Default)]
pub struct EmbeddedModelRegistration {
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub memory_bytes: Option<u64>,
    pub parameter_count: Option<u64>,
    pub context_length: Option<u32>,
    pub preferred_backend: Option<String>,
}

/// Infers a usable [`ModelDescriptor`] from a local model path.
pub fn infer_model_descriptor_from_path(
    path: impl Into<PathBuf>,
    options: EmbeddedModelRegistration,
) -> Result<ModelDescriptor> {
    let path = path.into();
    if !path.exists() {
        return Err(LociError::InvalidRequest(format!(
            "model path `{}` does not exist",
            path.display()
        )));
    }

    #[cfg(feature = "gguf")]
    let gguf_summary = if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
    {
        loci_gguf::read_gguf_metadata_summary(&path).ok()
    } else {
        None
    };

    #[cfg(not(feature = "gguf"))]
    let gguf_summary: Option<()> = None;

    let name = options.name.unwrap_or_else(|| {
        #[cfg(feature = "gguf")]
        if let Some(summary) = &gguf_summary {
            if let Some(general_name) = &summary.general_name {
                return general_name.clone();
            }
        }

        path.file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("default-model")
            .to_string()
    });

    let requested_architecture = options.architecture.unwrap_or_else(|| {
        #[cfg(feature = "gguf")]
        if let Some(summary) = &gguf_summary {
            if let Some(architecture) = &summary.architecture {
                return architecture.clone();
            }
        }

        "llama".to_string()
    });

    #[cfg(feature = "gguf")]
    let architecture = loci_gguf::resolve_architecture(&requested_architecture)
        .map(|spec| spec.canonical_name.to_string())
        .unwrap_or(requested_architecture);

    #[cfg(not(feature = "gguf"))]
    let architecture = requested_architecture;

    #[cfg(feature = "gguf")]
    let inferred_context_length = gguf_summary
        .as_ref()
        .and_then(|summary| summary.context_length);

    #[cfg(not(feature = "gguf"))]
    let inferred_context_length: Option<u32> = None;

    Ok(ModelDescriptor {
        name,
        path,
        architecture,
        memory_bytes: options.memory_bytes,
        parameter_count: options.parameter_count,
        context_length: options
            .context_length
            .or(inferred_context_length)
            .or(Some(8192)),
        preferred_backend: options.preferred_backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gguf")]
    use loci_gguf::GGUF_MAGIC;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(label: &str, extension: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-embedded-{label}-{suffix}.{extension}"))
    }

    #[cfg(feature = "gguf")]
    fn write_minimal_gguf(path: &PathBuf) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

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

        fs::write(path, bytes).expect("gguf");
    }

    #[test]
    fn rejects_missing_model_path() {
        let error = infer_model_descriptor_from_path(
            "D:/definitely-missing-model.gguf",
            EmbeddedModelRegistration::default(),
        )
        .expect_err("error");

        assert!(matches!(error, LociError::InvalidRequest(_)));
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn infers_descriptor_from_gguf_metadata() {
        let path = unique_temp_path("qwen", "gguf");
        write_minimal_gguf(&path);

        let descriptor =
            infer_model_descriptor_from_path(&path, EmbeddedModelRegistration::default())
                .expect("descriptor");

        assert_eq!(descriptor.architecture, "qwen");
        assert_eq!(descriptor.context_length, Some(32768));

        fs::remove_file(path).expect("cleanup");
    }
}
