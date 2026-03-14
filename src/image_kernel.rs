//! Dynamic image kernel plugin support.
//!
//! This module defines a plugin trait for text-to-image inference kernels and
//! provides dynamic loading helpers for runtime integration.

use crate::error::{LociError, Result};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

/// Request passed into an image generation kernel.
#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub model_id: String,
    pub steps: u32,
    pub guidance_scale: f32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub seed: Option<u64>,
    pub device: String,
}

/// Output returned by an image generation kernel.
#[derive(Debug, Clone)]
pub struct ImageGenerationResult {
    pub image_bytes: Vec<u8>,
    pub format: String,
}

/// Image generation kernel plugin trait.
pub trait ImageGenerationPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResult>;

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicImagePluginOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynImagePluginPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn ImageGenerationPlugin>` into an opaque ABI payload.
pub fn dynamic_image_plugin_into_opaque(
    plugin: Box<dyn ImageGenerationPlugin>,
) -> DynamicImagePluginOpaque {
    let raw: *mut dyn ImageGenerationPlugin = Box::into_raw(plugin);
    let parts: RawDynImagePluginPtr = unsafe { std::mem::transmute(raw) };
    DynamicImagePluginOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert an opaque payload back into `Box<dyn ImageGenerationPlugin>`.
///
/// # Safety
/// The payload must come from `dynamic_image_plugin_into_opaque` under a
/// compatible Rust toolchain/target ABI.
pub unsafe fn dynamic_image_plugin_from_opaque(
    opaque: DynamicImagePluginOpaque,
) -> Option<Box<dyn ImageGenerationPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynImagePluginPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ImageGenerationPlugin = unsafe { std::mem::transmute(parts) };
    if raw.is_null() {
        None
    } else {
        Some(unsafe { Box::from_raw(raw) })
    }
}

type ImagePluginConstructor = unsafe extern "C" fn() -> DynamicImagePluginOpaque;

#[allow(improper_ctypes_definitions)]
type LegacyImagePluginConstructor = unsafe extern "C" fn() -> *mut dyn ImageGenerationPlugin;

/// Loaded dynamic image plugin handle.
pub struct DynamicImageKernel {
    plugin: Box<dyn ImageGenerationPlugin>,
    #[allow(dead_code)]
    library: Arc<Library>,
}

impl DynamicImageKernel {
    pub fn plugin(&self) -> &dyn ImageGenerationPlugin {
        self.plugin.as_ref()
    }

    pub fn plugin_mut(&mut self) -> &mut dyn ImageGenerationPlugin {
        self.plugin.as_mut()
    }
}

/// Load a dynamic image kernel plugin from shared library.
pub fn load_dynamic_image_plugin<P: AsRef<Path>>(library_path: P) -> Result<DynamicImageKernel> {
    let path = library_path.as_ref();
    let manifest = load_and_validate_plugin_contract(path, PluginContractKind::ImageKernel)?;
    if !path.exists() {
        return Err(LociError::PluginError(format!(
            "Image kernel library not found: {}",
            path.display()
        )));
    }

    let library = unsafe {
        Library::new(path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load image kernel library '{}': {}",
                path.display(),
                e
            ))
        })?
    };

    let library = Arc::new(library);
    let mut plugin = unsafe {
        if let Ok(constructor_v1) = library.get::<ImagePluginConstructor>(b"create_image_plugin_v1")
        {
            let plugin_opaque = constructor_v1();
            dynamic_image_plugin_from_opaque(plugin_opaque).ok_or_else(|| {
                LociError::PluginError(
                    "Image kernel constructor returned invalid payload".to_string(),
                )
            })?
        } else {
            let constructor: Symbol<LegacyImagePluginConstructor> = library
                .get(b"create_image_plugin")
                .map_err(|e| {
                    LociError::PluginError(format!(
                        "Failed to find image kernel constructor symbol ('create_image_plugin_v1' or 'create_image_plugin'): {}",
                        e
                    ))
                })?;
            let raw = constructor();
            if raw.is_null() {
                return Err(LociError::PluginError(
                    "Image kernel constructor returned null".to_string(),
                ));
            }
            Box::from_raw(raw)
        }
    };

    if plugin.name().is_empty() {
        return Err(LociError::PluginError(
            "Image kernel plugin returned empty name".to_string(),
        ));
    }

    plugin.init()?;
    validate_runtime_plugin_identity(manifest.as_ref(), plugin.name(), plugin.version())?;

    Ok(DynamicImageKernel { plugin, library })
}

impl Drop for DynamicImageKernel {
    fn drop(&mut self) {
        let _ = self.plugin.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockImagePlugin;

    impl ImageGenerationPlugin for MockImagePlugin {
        fn name(&self) -> &str {
            "mock-image"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn generate(&self, _request: &ImageGenerationRequest) -> Result<ImageGenerationResult> {
            Ok(ImageGenerationResult {
                image_bytes: b"P3\n1 1\n255\n255 0 0\n".to_vec(),
                format: "ppm".to_string(),
            })
        }
    }

    #[test]
    fn dynamic_image_plugin_opaque_roundtrip() {
        let plugin: Box<dyn ImageGenerationPlugin> = Box::new(MockImagePlugin);
        let opaque = dynamic_image_plugin_into_opaque(plugin);
        let restored = unsafe { dynamic_image_plugin_from_opaque(opaque) };
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.name(), "mock-image");
    }

    #[test]
    fn dynamic_image_plugin_opaque_rejects_null() {
        let restored = unsafe {
            dynamic_image_plugin_from_opaque(DynamicImagePluginOpaque {
                data: std::ptr::null_mut(),
                vtable: std::ptr::null_mut(),
            })
        };
        assert!(restored.is_none());
    }
}
