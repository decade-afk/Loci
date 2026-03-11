use loci::error::{LociError, Result};
use loci::image_kernel::{
    dynamic_image_plugin_from_opaque, dynamic_image_plugin_into_opaque, DynamicImagePluginOpaque,
    ImageGenerationPlugin, ImageGenerationRequest, ImageGenerationResult,
};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct DiffusersKernelPlugin;

impl DiffusersKernelPlugin {
    pub fn new() -> Self {
        Self
    }

    fn python_bin() -> String {
        std::env::var("LOCI_T2I_PYTHON").unwrap_or_else(|_| "python".to_string())
    }

    fn script_path() -> PathBuf {
        if let Ok(path) = std::env::var("LOCI_T2I_SCRIPT") {
            return PathBuf::from(path);
        }
        PathBuf::from("scripts").join("t2i_generate.py")
    }

    fn temp_output_path() -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        path.push(format!("loci_kernel_t2i_{ts}.png"));
        path
    }
}

impl ImageGenerationPlugin for DiffusersKernelPlugin {
    fn name(&self) -> &str {
        "diffusers_kernel_plugin"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn generate(&self, request: &ImageGenerationRequest) -> Result<ImageGenerationResult> {
        let python = Self::python_bin();
        let script = Self::script_path();
        if !script.exists() {
            return Err(LociError::PluginError(format!(
                "t2i script not found: {}",
                script.display()
            )));
        }

        let out_path = Self::temp_output_path();
        let mut cmd = Command::new(python);
        cmd.arg(&script)
            .arg("--prompt")
            .arg(&request.prompt)
            .arg("--model-id")
            .arg(&request.model_id)
            .arg("--output")
            .arg(&out_path)
            .arg("--steps")
            .arg(request.steps.to_string())
            .arg("--guidance-scale")
            .arg(request.guidance_scale.to_string())
            .arg("--device")
            .arg(&request.device);

        let fallback_enabled = std::env::var("LOCI_T2I_FALLBACK")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        if fallback_enabled {
            cmd.arg("--fallback-placeholder");
        }

        if let Some(width) = request.width {
            cmd.arg("--width").arg(width.to_string());
        }
        if let Some(height) = request.height {
            cmd.arg("--height").arg(height.to_string());
        }
        if let Some(seed) = request.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }

        let output = cmd
            .output()
            .map_err(|e| LociError::PluginError(format!("failed to run t2i script: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(LociError::PluginError(format!(
                "t2i script failed with status: {} stdout: {} stderr: {}",
                output.status, stdout, stderr
            )));
        }

        let image_bytes = fs::read(&out_path).map_err(|e| {
            LociError::PluginError(format!(
                "failed to read generated image '{}': {e}",
                out_path.display()
            ))
        })?;
        let _ = fs::remove_file(&out_path);

        Ok(ImageGenerationResult {
            image_bytes,
            format: "png".to_string(),
        })
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn create_image_plugin_v1() -> DynamicImagePluginOpaque {
    dynamic_image_plugin_into_opaque(Box::new(DiffusersKernelPlugin::new()))
}

#[no_mangle]
pub extern "C" fn create_image_plugin() -> DynamicImagePluginOpaque {
    create_image_plugin_v1()
}

#[no_mangle]
pub extern "C" fn destroy_image_plugin_v1(opaque: DynamicImagePluginOpaque) {
    if let Some(_plugin) = unsafe { dynamic_image_plugin_from_opaque(opaque) } {
        // drop plugin
    }
}
