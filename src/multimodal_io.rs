//! Pluginized multimodal I/O bridge for Loci.
//!
//! This module focuses on I/O orchestration instead of backbone modeling:
//! - Convert multimodal user request into LLM-friendly prompt context
//! - Interpret LLM response into structured multimodal output plan
//! - Support built-in and dynamic plugins

use crate::error::{LociError, Result};
use crate::multimodal::{Audio, Image};
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Output modality requested by user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputModality {
    Text,
    Image,
    Audio,
}

/// User multimodal request passed into I/O plugins.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultimodalRequest {
    pub prompt: String,
    #[serde(default)]
    pub image_inputs: Vec<PathBuf>,
    #[serde(default)]
    pub audio_inputs: Vec<PathBuf>,
    #[serde(default)]
    pub output_modalities: Vec<OutputModality>,
}

impl MultimodalRequest {
    pub fn wants_image_output(&self) -> bool {
        self.output_modalities
            .iter()
            .any(|m| matches!(m, OutputModality::Image))
    }

    pub fn wants_audio_output(&self) -> bool {
        self.output_modalities
            .iter()
            .any(|m| matches!(m, OutputModality::Audio))
    }
}

/// Structured multimodal output plan parsed from model response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultimodalOutputPlan {
    pub text_response: String,
    #[serde(default)]
    pub image_prompts: Vec<String>,
    #[serde(default)]
    pub audio_prompts: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Multimodal I/O plugin trait.
pub trait MultimodalIoPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Turn structured multimodal request into model prompt.
    fn prepare_prompt(&self, request: &MultimodalRequest) -> Result<String>;

    /// Interpret model output into multimodal output plan.
    fn interpret_response(
        &self,
        request: &MultimodalRequest,
        response: &str,
    ) -> Result<MultimodalOutputPlan>;

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicMultimodalIoPluginOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynMultimodalIoPluginPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn MultimodalIoPlugin>` into opaque ABI payload.
pub fn dynamic_multimodal_io_plugin_into_opaque(
    plugin: Box<dyn MultimodalIoPlugin>,
) -> DynamicMultimodalIoPluginOpaque {
    let raw: *mut dyn MultimodalIoPlugin = Box::into_raw(plugin);
    let parts: RawDynMultimodalIoPluginPtr = unsafe { std::mem::transmute(raw) };
    DynamicMultimodalIoPluginOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert opaque ABI payload back into plugin object.
///
/// # Safety
/// Payload must come from `dynamic_multimodal_io_plugin_into_opaque`.
pub unsafe fn dynamic_multimodal_io_plugin_from_opaque(
    opaque: DynamicMultimodalIoPluginOpaque,
) -> Option<Box<dyn MultimodalIoPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }
    let parts = RawDynMultimodalIoPluginPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn MultimodalIoPlugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type MultimodalIoPluginConstructor =
    unsafe extern "C" fn() -> DynamicMultimodalIoPluginOpaque;

struct MultimodalIoPluginEntry {
    plugin: Box<dyn MultimodalIoPlugin>,
    enabled: bool,
    dynamic: Option<Arc<Library>>,
}

/// Registry for multimodal I/O plugins.
pub struct MultimodalIoRegistry {
    plugins: HashMap<String, MultimodalIoPluginEntry>,
}

impl MultimodalIoRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn with_builtin_plugins() -> Self {
        let mut registry = Self::new();
        registry
            .register(DescriptorMultimodalIoPlugin::new())
            .expect("register descriptor multimodal plugin");
        registry
    }

    pub fn register<P: MultimodalIoPlugin + 'static>(&mut self, mut plugin: P) -> Result<()> {
        let name = plugin.name().to_string();
        if name.trim().is_empty() {
            return Err(LociError::PluginError(
                "Multimodal I/O plugin name cannot be empty".to_string(),
            ));
        }
        if self.plugins.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Multimodal I/O plugin '{}' already registered",
                name
            )));
        }
        plugin.init()?;
        self.plugins.insert(
            name,
            MultimodalIoPluginEntry {
                plugin: Box::new(plugin),
                enabled: true,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn load_dynamic_plugin<P: AsRef<Path>>(&mut self, library_path: P) -> Result<String> {
        let path = library_path.as_ref();
        if !path.exists() {
            return Err(LociError::PluginError(format!(
                "Multimodal I/O plugin library not found: {}",
                path.display()
            )));
        }

        let library = unsafe {
            Library::new(path).map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to load multimodal I/O plugin library '{}': {}",
                    path.display(),
                    e
                ))
            })?
        };

        let constructor: Symbol<MultimodalIoPluginConstructor> = unsafe {
            library.get(b"create_multimodal_io_plugin_v1").map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to find multimodal I/O constructor symbol 'create_multimodal_io_plugin_v1': {}",
                    e
                ))
            })?
        };

        let mut plugin = unsafe {
            let opaque = constructor();
            dynamic_multimodal_io_plugin_from_opaque(opaque).ok_or_else(|| {
                LociError::PluginError(
                    "Multimodal I/O plugin constructor returned invalid payload".to_string(),
                )
            })?
        };
        if plugin.name().trim().is_empty() {
            return Err(LociError::PluginError(
                "Multimodal I/O plugin returned empty name".to_string(),
            ));
        }
        let name = plugin.name().to_string();
        if self.plugins.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Multimodal I/O plugin '{}' already registered",
                name
            )));
        }
        plugin.init()?;
        self.plugins.insert(
            name.clone(),
            MultimodalIoPluginEntry {
                plugin,
                enabled: true,
                dynamic: Some(Arc::new(library)),
            },
        );
        Ok(name)
    }

    pub fn unregister(&mut self, name: &str) -> Result<()> {
        let mut entry = self.plugins.remove(name).ok_or_else(|| {
            LociError::PluginError(format!("Multimodal I/O plugin '{}' not found", name))
        })?;
        entry.plugin.cleanup()?;
        Ok(())
    }

    pub fn enable(&mut self, name: &str) -> Result<()> {
        let entry = self.plugins.get_mut(name).ok_or_else(|| {
            LociError::PluginError(format!("Multimodal I/O plugin '{}' not found", name))
        })?;
        entry.enabled = true;
        Ok(())
    }

    pub fn disable(&mut self, name: &str) -> Result<()> {
        let entry = self.plugins.get_mut(name).ok_or_else(|| {
            LociError::PluginError(format!("Multimodal I/O plugin '{}' not found", name))
        })?;
        entry.enabled = false;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn MultimodalIoPlugin> {
        self.plugins.get(name).and_then(|entry| {
            if entry.enabled {
                Some(entry.plugin.as_ref())
            } else {
                None
            }
        })
    }

    pub fn list(&self) -> Vec<(String, String, bool, bool)> {
        let mut out = self
            .plugins
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    entry.plugin.version().to_string(),
                    entry.enabled,
                    entry.dynamic.is_some(),
                )
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

impl Default for MultimodalIoRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MultimodalIoRegistry {
    fn drop(&mut self) {
        for entry in self.plugins.values_mut() {
            let _ = entry.plugin.cleanup();
        }
    }
}

/// Built-in multimodal I/O plugin.
///
/// It enriches prompt with multimodal descriptors and expects optional
/// `IMAGE_PROMPT:` / `AUDIO_PROMPT:` lines in model response.
pub struct DescriptorMultimodalIoPlugin;

impl DescriptorMultimodalIoPlugin {
    pub fn new() -> Self {
        Self
    }
}

impl MultimodalIoPlugin for DescriptorMultimodalIoPlugin {
    fn name(&self) -> &str {
        "descriptor"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn prepare_prompt(&self, request: &MultimodalRequest) -> Result<String> {
        let mut prompt = String::new();
        prompt.push_str("Multimodal request context:\n");
        prompt.push_str(&format!("User prompt: {}\n", request.prompt));

        if request.image_inputs.is_empty() {
            prompt.push_str("Images: none\n");
        } else {
            prompt.push_str("Images:\n");
            for (idx, path) in request.image_inputs.iter().enumerate() {
                if !path.exists() {
                    return Err(LociError::InvalidArgument(format!(
                        "Image input file not found: {}",
                        path.display()
                    )));
                }
                let img = Image::load(path)?;
                prompt.push_str(&format!(
                    "- [{}] path={}, size={}x{}, channels={}\n",
                    idx,
                    path.display(),
                    img.width,
                    img.height,
                    img.channels
                ));
            }
        }

        if request.audio_inputs.is_empty() {
            prompt.push_str("Audio: none\n");
        } else {
            prompt.push_str("Audio:\n");
            for (idx, path) in request.audio_inputs.iter().enumerate() {
                if !path.exists() {
                    return Err(LociError::InvalidArgument(format!(
                        "Audio input file not found: {}",
                        path.display()
                    )));
                }
                let audio = Audio::load(path)?;
                prompt.push_str(&format!(
                    "- [{}] path={}, samples={}, sample_rate={}, channels={}\n",
                    idx,
                    path.display(),
                    audio.samples.len(),
                    audio.sample_rate,
                    audio.channels
                ));
            }
        }

        if !request.output_modalities.is_empty() {
            let wants = request
                .output_modalities
                .iter()
                .map(|m| match m {
                    OutputModality::Text => "text",
                    OutputModality::Image => "image",
                    OutputModality::Audio => "audio",
                })
                .collect::<Vec<_>>()
                .join(", ");
            prompt.push_str(&format!("Requested output modalities: {}\n", wants));
        }

        if request.wants_image_output() || request.wants_audio_output() {
            prompt.push_str("If non-text outputs are needed, append one or more planning lines:\n");
            if request.wants_image_output() {
                prompt.push_str("IMAGE_PROMPT: <image generation prompt>\n");
            }
            if request.wants_audio_output() {
                prompt.push_str("AUDIO_PROMPT: <audio generation prompt>\n");
            }
        }

        Ok(prompt)
    }

    fn interpret_response(
        &self,
        _request: &MultimodalRequest,
        response: &str,
    ) -> Result<MultimodalOutputPlan> {
        let mut plan = MultimodalOutputPlan::default();
        let mut text_lines = Vec::new();

        for line in response.lines() {
            let trimmed = line.trim();
            if let Some(value) = trimmed.strip_prefix("IMAGE_PROMPT:") {
                let value = value.trim();
                if !value.is_empty() {
                    plan.image_prompts.push(value.to_string());
                }
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("AUDIO_PROMPT:") {
                let value = value.trim();
                if !value.is_empty() {
                    plan.audio_prompts.push(value.to_string());
                }
                continue;
            }
            text_lines.push(line);
        }

        plan.text_response = text_lines.join("\n").trim().to_string();
        if plan.text_response.is_empty() {
            plan.text_response = response.trim().to_string();
        }
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OpaqueTestPlugin;

    impl MultimodalIoPlugin for OpaqueTestPlugin {
        fn name(&self) -> &str {
            "opaque_mm"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn prepare_prompt(&self, request: &MultimodalRequest) -> Result<String> {
            Ok(request.prompt.clone())
        }

        fn interpret_response(
            &self,
            _request: &MultimodalRequest,
            response: &str,
        ) -> Result<MultimodalOutputPlan> {
            Ok(MultimodalOutputPlan {
                text_response: response.to_string(),
                image_prompts: vec![],
                audio_prompts: vec![],
                metadata: HashMap::new(),
            })
        }
    }

    #[test]
    fn descriptor_plugin_interprets_planning_lines() {
        let plugin = DescriptorMultimodalIoPlugin::new();
        let request = MultimodalRequest {
            prompt: "draw and narrate".to_string(),
            image_inputs: vec![],
            audio_inputs: vec![],
            output_modalities: vec![OutputModality::Text, OutputModality::Image],
        };
        let response = "Here is the answer.\nIMAGE_PROMPT: a blue robot in rain\n";
        let plan = plugin.interpret_response(&request, response).unwrap();
        assert_eq!(plan.text_response, "Here is the answer.");
        assert_eq!(plan.image_prompts, vec!["a blue robot in rain".to_string()]);
    }

    #[test]
    fn opaque_roundtrip_for_multimodal_io_plugin() {
        let plugin: Box<dyn MultimodalIoPlugin> = Box::new(OpaqueTestPlugin);
        let opaque = dynamic_multimodal_io_plugin_into_opaque(plugin);
        let restored = unsafe { dynamic_multimodal_io_plugin_from_opaque(opaque) };
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.name(), "opaque_mm");
    }

    #[test]
    fn registry_has_builtin_descriptor_plugin() {
        let registry = MultimodalIoRegistry::with_builtin_plugins();
        let plugins = registry.list();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].0, "descriptor");
    }
}
