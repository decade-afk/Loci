use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use image::ImageReader;
use loci_protocol::{BackendError, BackendResult, ImageInput};
use std::fs;
use std::path::{Path, PathBuf};

/// Converts image payloads into compact numeric features for the local generation chain.
pub(super) fn collect_image_features(images: &[ImageInput]) -> BackendResult<Vec<f32>> {
    let mut features = Vec::new();
    for image in images {
        let bytes = load_image_bytes(image)?;
        let decoded = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| BackendError {
                message: format!("failed to inspect image payload: {error}"),
            })?
            .decode()
            .map_err(|error| BackendError {
                message: format!("failed to decode image payload: {error}"),
            })?;
        let rgb = decoded.to_rgb8();
        let (width, height) = rgb.dimensions();
        features.push(width as f32 / 4096.0);
        features.push(height as f32 / 4096.0);
        let checksum =
            rgb.into_raw()
                .iter()
                .take(2048)
                .enumerate()
                .fold(0u64, |acc, (index, byte)| {
                    acc.wrapping_mul(16777619)
                        .wrapping_add(*byte as u64 + index as u64 + 1)
                });
        features.push((checksum % 10_000) as f32 / 10_000.0);
    }
    Ok(features)
}

fn load_image_bytes(image: &ImageInput) -> BackendResult<Vec<u8>> {
    match image {
        ImageInput::Path { path } => fs::read(path).map_err(|error| BackendError {
            message: format!("failed to read image `{}`: {error}", path.display()),
        }),
        ImageInput::Url { url } => {
            if let Some(path) = file_url_to_path(url) {
                fs::read(&path).map_err(|error| BackendError {
                    message: format!("failed to read image `{}`: {error}", path.display()),
                })
            } else if let Some(bytes) = decode_data_url(url)? {
                Ok(bytes)
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
        } => BASE64_STANDARD
            .decode(data_base64)
            .map_err(|error| BackendError {
                message: format!("invalid base64 image payload: {error}"),
            }),
    }
}

fn file_url_to_path(url: &str) -> Option<PathBuf> {
    if let Some(rest) = url.strip_prefix("file:///") {
        Some(Path::new("/").join(rest.trim_start_matches('/')))
    } else if let Some(rest) = url.strip_prefix("file://") {
        Some(PathBuf::from(rest))
    } else {
        None
    }
}

fn decode_data_url(url: &str) -> BackendResult<Option<Vec<u8>>> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Ok(None);
    };
    let Some((_, encoded)) = payload.split_once("base64,") else {
        return Ok(None);
    };
    BASE64_STANDARD
        .decode(encoded)
        .map(Some)
        .map_err(|error| BackendError {
            message: format!("invalid data URL payload: {error}"),
        })
}
