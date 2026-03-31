use crate::backend::{GpuSplitMode, ModelMetadata};
use crate::error::{LociError, Result};
use std::fs;
use std::path::Path;

use super::adapter::{LlamaCppAdapterContext, LlamaCppBuildIntegration};
use super::ffi;
use super::plan::LlamaCppLoadPlan;
use super::runtime::LlamaCppExecutionConfig;

pub trait LlamaCppDriver: Send + Sync {
    fn kind(&self) -> &'static str;
    fn validate(&self, context: &LlamaCppAdapterContext) -> Result<()>;
    fn protocol(&self, context: &LlamaCppAdapterContext) -> LlamaCppDriverProtocol;
    fn init_backend(&self, context: &LlamaCppAdapterContext) -> Result<LlamaCppBackendSession>;
    fn load_model(&self, request: LlamaCppModelLoadRequest<'_>) -> Result<LlamaCppLoadedModel>;
    fn create_context(
        &self,
        request: LlamaCppContextCreateRequest<'_>,
    ) -> Result<LlamaCppCreatedContext>;
}

pub struct LlamaCppBackendSession {
    kind: String,
    native: Option<ffi::LlamaBackendHandle>,
}

pub struct LlamaCppLoadedModel {
    metadata: ModelMetadata,
    native: Option<ffi::LlamaModel>,
}

pub struct LlamaCppCreatedContext {
    native: Option<ffi::LlamaContext>,
}

pub struct LlamaCppModelLoadRequest<'a> {
    pub model_path: &'a Path,
    pub load_plan: &'a LlamaCppLoadPlan,
}

pub struct LlamaCppContextCreateRequest<'a> {
    pub loaded_model: &'a LlamaCppLoadedModel,
    pub load_plan: &'a LlamaCppLoadPlan,
    pub runtime_override: Option<&'a LlamaCppExecutionConfig>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppDriverProtocol {
    pub kind: String,
    pub backend_init_symbol: String,
    pub model_default_params_symbol: String,
    pub context_default_params_symbol: String,
    pub ffi_module: String,
    pub ffi_shim_c: String,
    pub phases: LlamaCppDriverPhases,
    pub lifecycle: LlamaCppLifecycleContract,
}

#[derive(Debug, Clone)]
pub struct LlamaCppDriverPhases {
    pub init: LlamaCppInitPhase,
    pub load_model: LlamaCppLoadModelPhase,
    pub create_context: LlamaCppCreateContextPhase,
}

#[derive(Debug, Clone)]
pub struct LlamaCppInitPhase {
    pub function: String,
    pub companion_free_function: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlamaCppLoadModelPhase {
    pub model_type: String,
    pub function: String,
    pub params_function: String,
}

#[derive(Debug, Clone)]
pub struct LlamaCppCreateContextPhase {
    pub context_type: String,
    pub function: String,
    pub params_function: String,
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
            "driver[kind={}, backend_init={}, model_default_params={}, context_default_params={}, ffi={}, shim={}, phases={}, lifecycle={}]",
            self.kind,
            self.backend_init_symbol,
            self.model_default_params_symbol,
            self.context_default_params_symbol,
            self.ffi_module,
            self.ffi_shim_c,
            self.phases.summary(),
            self.lifecycle.summary()
        )
    }
}

impl LlamaCppDriverPhases {
    pub fn summary(&self) -> String {
        format!(
            "phases[init={}, load_model={}, create_context={}]",
            self.init.summary(),
            self.load_model.summary(),
            self.create_context.summary()
        )
    }
}

impl LlamaCppInitPhase {
    pub fn summary(&self) -> String {
        format!(
            "init(function={}, free={})",
            self.function,
            self.companion_free_function
                .clone()
                .unwrap_or_else(|| "none".to_string())
        )
    }
}

impl LlamaCppLoadModelPhase {
    pub fn summary(&self) -> String {
        format!(
            "load_model(type={}, function={}, params={})",
            self.model_type, self.function, self.params_function
        )
    }
}

impl LlamaCppCreateContextPhase {
    pub fn summary(&self) -> String {
        format!(
            "create_context(type={}, function={}, params={})",
            self.context_type, self.function, self.params_function
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

impl LlamaCppBackendSession {
    pub fn kind(&self) -> &str {
        &self.kind
    }
    pub fn is_native(&self) -> bool {
        self.native.is_some()
    }
}

impl LlamaCppLoadedModel {
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    pub fn native_model(&self) -> Option<&ffi::LlamaModel> {
        self.native.as_ref()
    }

    pub fn require_native_model(&self) -> Result<&ffi::LlamaModel> {
        self.native_model().ok_or_else(|| {
            LociError::BackendError("llama.cpp driver missing native model handle".to_string())
        })
    }
}

impl LlamaCppCreatedContext {
    pub fn is_native(&self) -> bool {
        self.native.is_some()
    }

    pub fn native_context(&self) -> Option<&ffi::LlamaContext> {
        self.native.as_ref()
    }
    pub fn native_context_mut(&mut self) -> Option<&mut ffi::LlamaContext> {
        self.native.as_mut()
    }

    pub fn require_native_context(&self) -> Result<&ffi::LlamaContext> {
        self.native_context().ok_or_else(|| {
            LociError::BackendError("llama.cpp driver missing native context handle".to_string())
        })
    }

    pub fn require_native_context_mut(&mut self) -> Result<&mut ffi::LlamaContext> {
        self.native_context_mut().ok_or_else(|| {
            LociError::BackendError("llama.cpp driver missing native context handle".to_string())
        })
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
    fn init_backend(&self, _context: &LlamaCppAdapterContext) -> Result<LlamaCppBackendSession> {
        Ok(LlamaCppBackendSession {
            kind: self.kind().to_string(),
            native: None,
        })
    }
    fn load_model(&self, request: LlamaCppModelLoadRequest<'_>) -> Result<LlamaCppLoadedModel> {
        Ok(LlamaCppLoadedModel {
            metadata: request.load_plan.metadata(),
            native: None,
        })
    }
    fn create_context(
        &self,
        _request: LlamaCppContextCreateRequest<'_>,
    ) -> Result<LlamaCppCreatedContext> {
        Ok(LlamaCppCreatedContext { native: None })
    }
}

impl LlamaCppDriver for NativeLlamaCppDriver {
    fn kind(&self) -> &'static str {
        "native"
    }

    fn validate(&self, context: &LlamaCppAdapterContext) -> Result<()> {
        let ffi_source =
            fs::read_to_string(&context.build_integration.ffi_module).map_err(|err| {
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

    fn init_backend(&self, _context: &LlamaCppAdapterContext) -> Result<LlamaCppBackendSession> {
        Ok(LlamaCppBackendSession {
            kind: self.kind().to_string(),
            native: Some(ffi::LlamaBackendHandle::acquire()),
        })
    }

    fn load_model(&self, request: LlamaCppModelLoadRequest<'_>) -> Result<LlamaCppLoadedModel> {
        let model_path = request
            .model_path
            .to_str()
            .ok_or_else(|| LociError::ConfigError("invalid llama.cpp model path".to_string()))?;
        let params = model_params_from_load_plan(request.load_plan);
        let model =
            ffi::LlamaModel::from_file(model_path, &params).map_err(LociError::ModelLoadError)?;

        if !model.has_decoder() {
            return Err(LociError::ModelLoadError(
                "model does not expose a decoder path required for text generation".to_string(),
            ));
        }

        let metadata = ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: model.n_vocab() as u32,
            n_ctx_train: model.n_ctx_train() as u32,
            n_embd: model.n_embd() as u32,
            n_layer: 0,
            param_count: None,
        };

        Ok(LlamaCppLoadedModel {
            metadata,
            native: Some(model),
        })
    }

    fn create_context(
        &self,
        request: LlamaCppContextCreateRequest<'_>,
    ) -> Result<LlamaCppCreatedContext> {
        let model = request.loaded_model.require_native_model()?;
        let params = context_params_from_request(&request);
        let context = ffi::LlamaContext::new(model, &params).map_err(LociError::InferenceError)?;
        Ok(LlamaCppCreatedContext {
            native: Some(context),
        })
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
        phases: phases_from_ffi_source(&ffi_source),
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

fn phases_from_ffi_source(ffi_source: &str) -> LlamaCppDriverPhases {
    LlamaCppDriverPhases {
        init: LlamaCppInitPhase {
            function: if ffi_source.contains("pub fn backend_init()") {
                "backend_init".to_string()
            } else {
                "missing".to_string()
            },
            companion_free_function: if ffi_source.contains("pub fn backend_free()") {
                Some("backend_free".to_string())
            } else {
                None
            },
        },
        load_model: LlamaCppLoadModelPhase {
            model_type: if ffi_source.contains("pub struct LlamaModel") {
                "LlamaModel".to_string()
            } else {
                "unknown".to_string()
            },
            function: if ffi_source.contains("pub fn from_file(") {
                "LlamaModel::from_file".to_string()
            } else {
                "missing".to_string()
            },
            params_function: if ffi_source.contains("pub fn model_default_params()") {
                "model_default_params".to_string()
            } else {
                "missing".to_string()
            },
        },
        create_context: LlamaCppCreateContextPhase {
            context_type: if ffi_source.contains("pub struct LlamaContext") {
                "LlamaContext".to_string()
            } else {
                "unknown".to_string()
            },
            function: if ffi_source.contains("pub fn new(model: &LlamaModel") {
                "LlamaContext::new".to_string()
            } else {
                "missing".to_string()
            },
            params_function: if ffi_source.contains("pub fn context_default_params()") {
                "context_default_params".to_string()
            } else {
                "missing".to_string()
            },
        },
    }
}

fn model_params_from_load_plan(load_plan: &LlamaCppLoadPlan) -> ffi::llama_model_params {
    let mut params = ffi::model_default_params();
    params.n_gpu_layers = load_plan.n_gpu_layers();
    params.split_mode = match load_plan.split_mode() {
        GpuSplitMode::None => ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE,
        GpuSplitMode::Layer => ffi::llama_split_mode_LLAMA_SPLIT_MODE_LAYER,
        GpuSplitMode::Row => ffi::llama_split_mode_LLAMA_SPLIT_MODE_ROW,
    };
    params.main_gpu = if load_plan.gpu_active() && load_plan.split_mode() == GpuSplitMode::None {
        load_plan.main_gpu() as i32
    } else {
        0
    };
    params.tensor_split = load_plan
        .tensor_split()
        .map(|values| values.as_ptr())
        .unwrap_or(std::ptr::null());
    params.use_mmap = load_plan.use_mmap();
    params.use_mlock = load_plan.use_mlock();
    params
}

fn context_params_from_request(
    request: &LlamaCppContextCreateRequest<'_>,
) -> ffi::llama_context_params {
    let mut params = ffi::context_default_params();
    params.n_ctx = request
        .runtime_override
        .map(|runtime| runtime.n_ctx())
        .unwrap_or_else(|| request.load_plan.runtime().n_ctx());
    params.n_batch = request
        .runtime_override
        .map(|runtime| runtime.n_batch())
        .unwrap_or_else(|| request.load_plan.runtime().n_batch());
    params.offload_kqv = request.load_plan.kv_offload();
    params.op_offload = request.load_plan.op_offload();
    params.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if let Some(n_threads) = request
        .runtime_override
        .and_then(|runtime| runtime.n_threads())
        .or_else(|| request.load_plan.runtime().n_threads())
    {
        params.n_threads = n_threads as i32;
    }
    params
}
