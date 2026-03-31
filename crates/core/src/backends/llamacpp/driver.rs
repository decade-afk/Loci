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
}

impl LlamaCppDriverProtocol {
    pub fn summary(&self) -> String {
        format!(
            "driver[kind={}, backend_init={}, model_default_params={}, context_default_params={}, ffi={}, shim={}]",
            self.kind,
            self.backend_init_symbol,
            self.model_default_params_symbol,
            self.context_default_params_symbol,
            self.ffi_module,
            self.ffi_shim_c
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

        for required in [
            "pub fn backend_init()",
            "pub fn model_default_params()",
            "pub fn context_default_params()",
            "pub struct LlamaModel",
            "pub struct LlamaContext",
        ] {
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
    LlamaCppDriverProtocol {
        kind: kind.to_string(),
        backend_init_symbol: "backend_init".to_string(),
        model_default_params_symbol: "model_default_params".to_string(),
        context_default_params_symbol: "context_default_params".to_string(),
        ffi_module: integration.ffi_module.display().to_string(),
        ffi_shim_c: integration.ffi_shim_c.display().to_string(),
    }
}

pub fn discover_driver(integration: &LlamaCppBuildIntegration) -> Box<dyn LlamaCppDriver> {
    if let Ok(ffi_source) = fs::read_to_string(&integration.ffi_module) {
        let native_markers = [
            "pub fn backend_init()",
            "pub fn model_default_params()",
            "pub fn context_default_params()",
            "pub struct LlamaModel",
            "pub struct LlamaContext",
        ];
        if native_markers
            .iter()
            .all(|marker| ffi_source.contains(marker))
        {
            return Box::new(NativeLlamaCppDriver::new());
        }
    }

    Box::new(StubLlamaCppDriver::new())
}
