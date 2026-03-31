use crate::error::Result;
use super::adapter::{LlamaCppAdapterContext, LlamaCppBuildIntegration};
use crate::error::LociError;
use std::fs;

pub trait LlamaCppDriver: Send + Sync {
    fn kind(&self) -> &'static str;
    fn validate(&self, context: &LlamaCppAdapterContext) -> Result<()>;
    fn protocol(&self, context: &LlamaCppAdapterContext) -> LlamaCppDriverProtocol;
}

#[derive(Debug, Clone)]
pub struct LlamaCppDriverProtocol {
    pub kind: String,
    pub backend_init_symbol: String,
    pub model_default_params_symbol: String,
    pub context_default_params_symbol: String,
    pub ffi_module: String,
    pub ffi_shim_c: String,
    pub lifecycle: LlamaCppLifecycleContract,
}

#[derive(Debug, Clone)]
pub struct LlamaCppLifecycleContract {
    pub model_type: String,
    pub context_type: String,
    pub supports_backend_init: bool,
    pub supports_model_defaults: bool,
    pub supports_context_defaults: bool,
    pub supports_tokenize: bool,
    pub supports_token_to_str: bool,
    pub supports_decode: bool,
    pub supports_logits: bool,
    pub supports_kv_cache_clear: bool,
}

impl LlamaCppDriverProtocol {
    pub fn summary(&self) -> String {
        format!(
            "driver[kind={}, backend_init={}, model_default_params={}, context_default_params={}, ffi={}, shim={}, lifecycle={}]",
            self.kind,
            self.backend_init_symbol,
            self.model_default_params_symbol,
            self.context_default_params_symbol,
            self.ffi_module,
            self.ffi_shim_c,
            self.lifecycle.summary()
        )
    }
}

impl LlamaCppLifecycleContract {
    pub fn summary(&self) -> String {
        format!(
            "contract[model={}, context={}, init={}, model_defaults={}, context_defaults={}, tokenize={}, token_to_str={}, decode={}, logits={}, kv_clear={}]",
            self.model_type,
            self.context_type,
            self.supports_backend_init,
            self.supports_model_defaults,
            self.supports_context_defaults,
            self.supports_tokenize,
            self.supports_token_to_str,
            self.supports_decode,
            self.supports_logits,
            self.supports_kv_cache_clear
        )
    }
}

pub struct StubLlamaCppDriver;
pub struct NativeLlamaCppDriver;

impl StubLlamaCppDriver {
    pub fn new() -> Self {
        Self
    }
}

impl NativeLlamaCppDriver {
    pub fn new() -> Self {
        Self
    }
}

impl LlamaCppDriver for StubLlamaCppDriver {
    fn kind(&self) -> &'static str {
        "stub"
    }

    fn validate(&self, context: &LlamaCppAdapterContext) -> Result<()> {
        let _ = context.build_integration.summary();
        Ok(())
    }

    fn protocol(&self, context: &LlamaCppAdapterContext) -> LlamaCppDriverProtocol {
        protocol_from_build_integration(self.kind(), &context.build_integration)
    }
}

impl LlamaCppDriver for NativeLlamaCppDriver {
    fn kind(&self) -> &'static str {
        "native"
    }

    fn validate(&self, context: &LlamaCppAdapterContext) -> Result<()> {
        let ffi_source = fs::read_to_string(&context.build_integration.ffi_module).map_err(|err| {
            LociError::ConfigError(format!(
                "failed to read ffi module for native llama driver: {err}"
            ))
        })?;

        for required in required_native_markers() {
            if !ffi_source.contains(required) {
                return Err(LociError::ConfigError(format!(
                    "native llama driver missing required ffi symbol declaration: {required}"
                )));
            }
        }

        Ok(())
    }

    fn protocol(&self, context: &LlamaCppAdapterContext) -> LlamaCppDriverProtocol {
        protocol_from_build_integration(self.kind(), &context.build_integration)
    }
}

fn protocol_from_build_integration(
    kind: &str,
    integration: &LlamaCppBuildIntegration,
) -> LlamaCppDriverProtocol {
    let ffi_source = fs::read_to_string(&integration.ffi_module).unwrap_or_default();
    LlamaCppDriverProtocol {
        kind: kind.to_string(),
        backend_init_symbol: "backend_init".to_string(),
        model_default_params_symbol: "model_default_params".to_string(),
        context_default_params_symbol: "context_default_params".to_string(),
        ffi_module: integration.ffi_module.display().to_string(),
        ffi_shim_c: integration.ffi_shim_c.display().to_string(),
        lifecycle: lifecycle_from_ffi_source(&ffi_source),
    }
}

pub fn discover_driver(integration: &LlamaCppBuildIntegration) -> Box<dyn LlamaCppDriver> {
    if let Ok(ffi_source) = fs::read_to_string(&integration.ffi_module) {
        if required_native_markers()
            .iter()
            .all(|marker| ffi_source.contains(marker))
        {
            return Box::new(NativeLlamaCppDriver::new());
        }
    }

    Box::new(StubLlamaCppDriver::new())
}

fn required_native_markers() -> [&'static str; 5] {
    [
        "pub fn backend_init()",
        "pub fn model_default_params()",
        "pub fn context_default_params()",
        "pub struct LlamaModel",
        "pub struct LlamaContext",
    ]
}

fn lifecycle_from_ffi_source(ffi_source: &str) -> LlamaCppLifecycleContract {
    LlamaCppLifecycleContract {
        model_type: if ffi_source.contains("pub struct LlamaModel") {
            "LlamaModel".to_string()
        } else {
            "unknown".to_string()
        },
        context_type: if ffi_source.contains("pub struct LlamaContext") {
            "LlamaContext".to_string()
        } else {
            "unknown".to_string()
        },
        supports_backend_init: ffi_source.contains("pub fn backend_init()"),
        supports_model_defaults: ffi_source.contains("pub fn model_default_params()"),
        supports_context_defaults: ffi_source.contains("pub fn context_default_params()"),
        supports_tokenize: ffi_source.contains("pub fn tokenize("),
        supports_token_to_str: ffi_source.contains("pub fn token_to_str("),
        supports_decode: ffi_source.contains("pub fn decode("),
        supports_logits: ffi_source.contains("pub fn get_logits_ith("),
        supports_kv_cache_clear: ffi_source.contains("pub fn kv_cache_clear("),
    }
}
