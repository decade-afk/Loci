use loci_core::{InferenceEngine, InferenceParams, LociError, ModelConfig, ModelLoadStrategy};
use serde::Serialize;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

const LOCI_ABI_VERSION: u32 = 1;
static LOCI_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

thread_local! {
    static LAST_ERROR: RefCell<LastErrorState> = RefCell::new(LastErrorState::default());
}

pub struct LociEngine {
    engine: InferenceEngine,
}

#[derive(Debug, Clone)]
struct LastErrorState {
    code: LociStatusCode,
    message: CString,
}

impl Default for LastErrorState {
    fn default() -> Self {
        Self {
            code: LociStatusCode::Ok,
            message: CString::new("").expect("empty cstring"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct LociErrorInfo {
    code: &'static str,
    message: String,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LociStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    ConfigError = 2,
    BackendNotAvailable = 3,
    BackendError = 4,
    ModelLoadError = 5,
    InferenceError = 6,
    UnsupportedOperation = 7,
    InternalError = 8,
}

impl LociStatusCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::InvalidArgument => "invalid_argument",
            Self::ConfigError => "config_error",
            Self::BackendNotAvailable => "backend_not_available",
            Self::BackendError => "backend_error",
            Self::ModelLoadError => "model_load_error",
            Self::InferenceError => "inference_error",
            Self::UnsupportedOperation => "unsupported_operation",
            Self::InternalError => "internal_error",
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LociGpuSplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LociModelLoadStrategyKind {
    Strict = 0,
    AutoReduceGpuLayers = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LociModelLoadOptions {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub has_n_threads: u8,
    pub n_threads: u32,
    pub use_gpu: u8,
    pub n_gpu_layers: i32,
    pub use_mmap: u8,
    pub use_mlock: u8,
    pub kv_offload: u8,
    pub op_offload: u8,
    pub split_mode: LociGpuSplitMode,
    pub main_gpu: u32,
    pub tensor_split: *const f32,
    pub tensor_split_len: u32,
    pub load_strategy: LociModelLoadStrategyKind,
    pub load_strategy_step: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LociGenerationOptions {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub has_n_threads: u8,
    pub n_threads: u32,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}

impl Default for LociModelLoadOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            has_n_threads: 0,
            n_threads: 0,
            use_gpu: 1,
            n_gpu_layers: -1,
            use_mmap: 1,
            use_mlock: 0,
            kv_offload: 1,
            op_offload: 1,
            split_mode: LociGpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: ptr::null(),
            tensor_split_len: 0,
            load_strategy: LociModelLoadStrategyKind::Strict,
            load_strategy_step: 0,
        }
    }
}

impl Default for LociGenerationOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            has_n_threads: 0,
            n_threads: 0,
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

fn set_last_error(code: LociStatusCode, message: &str) {
    let sanitized = message.replace('\0', " ");
    let value = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("error message contains interior NUL").expect("cstring"));
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = LastErrorState {
            code,
            message: value,
        }
    });
}

fn clear_last_error() {
    set_last_error(LociStatusCode::Ok, "");
}

fn set_last_core_error(error: &LociError) {
    let code = match error {
        LociError::ConfigError(_) => LociStatusCode::ConfigError,
        LociError::InvalidArgument(_) => LociStatusCode::InvalidArgument,
        LociError::BackendNotAvailable(_) => LociStatusCode::BackendNotAvailable,
        LociError::BackendError(_) => LociStatusCode::BackendError,
        LociError::ModelLoadError(_) => LociStatusCode::ModelLoadError,
        LociError::InferenceError(_) => LociStatusCode::InferenceError,
        LociError::UnsupportedOperation(_) => LociStatusCode::UnsupportedOperation,
        LociError::Other(_) => LociStatusCode::InternalError,
    };
    set_last_error(code, &error.to_string());
}

fn bool_from_u8(value: u8) -> bool {
    value != 0
}

fn string_into_raw(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(value) => value.into_raw(),
        Err(_) => {
            set_last_error(
                LociStatusCode::InternalError,
                "result contains interior NUL byte",
            );
            ptr::null_mut()
        }
    }
}

fn json_into_raw<T: serde::Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(json) => string_into_raw(json),
        Err(error) => {
            set_last_error(
                LociStatusCode::InternalError,
                &format!("serialization failed: {error}"),
            );
            ptr::null_mut()
        }
    }
}

unsafe fn required_cstr(value: *const c_char, field: &str) -> Result<String, ()> {
    if value.is_null() {
        set_last_error(
            LociStatusCode::InvalidArgument,
            &format!("{field} must not be null"),
        );
        return Err(());
    }

    let text = match CStr::from_ptr(value).to_str() {
        Ok(text) => text.trim(),
        Err(_) => {
            set_last_error(
                LociStatusCode::InvalidArgument,
                &format!("{field} must be valid UTF-8"),
            );
            return Err(());
        }
    };

    if text.is_empty() {
        set_last_error(
            LociStatusCode::InvalidArgument,
            &format!("{field} must not be empty"),
        );
        return Err(());
    }

    Ok(text.to_string())
}

unsafe fn prompt_from_ptr_len(prompt: *const c_char, prompt_len: u32) -> Result<String, ()> {
    if prompt_len == 0 {
        return Ok(String::new());
    }
    if prompt.is_null() {
        set_last_error(
            LociStatusCode::InvalidArgument,
            "prompt must not be null when prompt_len > 0",
        );
        return Err(());
    }
    let bytes = slice::from_raw_parts(prompt as *const u8, prompt_len as usize);
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_owned()),
        Err(_) => {
            set_last_error(
                LociStatusCode::InvalidArgument,
                "prompt must be valid UTF-8",
            );
            Err(())
        }
    }
}

unsafe fn model_load_options(options: *const LociModelLoadOptions) -> LociModelLoadOptions {
    if options.is_null() {
        LociModelLoadOptions::default()
    } else {
        *options
    }
}

unsafe fn generation_options(options: *const LociGenerationOptions) -> LociGenerationOptions {
    if options.is_null() {
        LociGenerationOptions::default()
    } else {
        *options
    }
}

unsafe fn tensor_split_from_options(
    options: &LociModelLoadOptions,
) -> Result<Option<Vec<f32>>, ()> {
    if options.tensor_split_len == 0 {
        return Ok(None);
    }
    if options.tensor_split.is_null() {
        set_last_error(
            LociStatusCode::InvalidArgument,
            "tensor_split must not be null when tensor_split_len > 0",
        );
        return Err(());
    }

    Ok(Some(
        slice::from_raw_parts(options.tensor_split, options.tensor_split_len as usize).to_vec(),
    ))
}

unsafe fn with_engine<T>(
    engine: *mut LociEngine,
    f: impl FnOnce(&mut LociEngine) -> loci_core::Result<T>,
) -> Result<T, ()> {
    if engine.is_null() {
        set_last_error(LociStatusCode::InvalidArgument, "engine must not be null");
        return Err(());
    }

    match f(&mut *engine) {
        Ok(value) => Ok(value),
        Err(error) => {
            set_last_core_error(&error);
            Err(())
        }
    }
}

fn build_model_config(
    model_path: String,
    options: LociModelLoadOptions,
    tensor_split: Option<Vec<f32>>,
) -> ModelConfig {
    ModelConfig {
        model_path: model_path.into(),
        n_ctx: options.n_ctx,
        n_threads: if bool_from_u8(options.has_n_threads) {
            Some(options.n_threads)
        } else {
            None
        },
        n_batch: options.n_batch,
        use_gpu: bool_from_u8(options.use_gpu),
        n_gpu_layers: options.n_gpu_layers,
        use_mmap: bool_from_u8(options.use_mmap),
        use_mlock: bool_from_u8(options.use_mlock),
        kv_offload: bool_from_u8(options.kv_offload),
        op_offload: bool_from_u8(options.op_offload),
        split_mode: match options.split_mode {
            LociGpuSplitMode::None => loci_core::GpuSplitMode::None,
            LociGpuSplitMode::Layer => loci_core::GpuSplitMode::Layer,
            LociGpuSplitMode::Row => loci_core::GpuSplitMode::Row,
        },
        main_gpu: options.main_gpu,
        tensor_split,
        load_strategy: match options.load_strategy {
            LociModelLoadStrategyKind::Strict => ModelLoadStrategy::Strict,
            LociModelLoadStrategyKind::AutoReduceGpuLayers => {
                ModelLoadStrategy::AutoReduceGpuLayers {
                    step: options.load_strategy_step,
                }
            }
        },
    }
}

fn build_generation_params(options: LociGenerationOptions) -> InferenceParams {
    InferenceParams {
        n_ctx: options.n_ctx,
        n_batch: options.n_batch,
        n_threads: if bool_from_u8(options.has_n_threads) {
            Some(options.n_threads)
        } else {
            None
        },
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        min_p: options.min_p,
        top_k: options.top_k,
        repeat_penalty: options.repeat_penalty,
    }
}

#[no_mangle]
pub extern "C" fn loci_abi_version() -> u32 {
    LOCI_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn loci_default_model_load_options() -> LociModelLoadOptions {
    LociModelLoadOptions::default()
}

#[no_mangle]
pub extern "C" fn loci_default_generation_options() -> LociGenerationOptions {
    LociGenerationOptions::default()
}

#[no_mangle]
pub unsafe extern "C" fn loci_version() -> *const c_char {
    LOCI_VERSION.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn loci_get_last_status_code() -> LociStatusCode {
    LAST_ERROR.with(|slot| slot.borrow().code)
}

#[no_mangle]
pub unsafe extern "C" fn loci_get_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().message.as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn loci_get_last_error_json() -> *mut c_char {
    let payload = LAST_ERROR.with(|slot| {
        let state = slot.borrow();
        LociErrorInfo {
            code: state.code.as_str(),
            message: state.message.to_string_lossy().into_owned(),
        }
    });
    json_into_raw(&payload)
}

#[no_mangle]
pub unsafe extern "C" fn loci_free_string(value: *mut c_char) {
    if !value.is_null() {
        let _ = CString::from_raw(value);
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_new() -> *mut LociEngine {
    match InferenceEngine::builder().build() {
        Ok(engine) => Box::into_raw(Box::new(LociEngine { engine })),
        Err(error) => {
            set_last_core_error(&error);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_free(engine: *mut LociEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_load_model_json(
    engine: *mut LociEngine,
    backend_name: *const c_char,
    model_path: *const c_char,
    options: *const LociModelLoadOptions,
) -> *mut c_char {
    let backend_name = match required_cstr(backend_name, "backend_name") {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let model_path = match required_cstr(model_path, "model_path") {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let options = model_load_options(options);
    let tensor_split = match tensor_split_from_options(&options) {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let config = build_model_config(model_path, options, tensor_split);

    match with_engine(engine, |engine| {
        engine.engine.load_model_config(&backend_name, &config)?;
        Ok(engine.engine.runtime_snapshot())
    }) {
        Ok(snapshot) => {
            clear_last_error();
            json_into_raw(&snapshot)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_generate_with_len_and_options(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    options: *const LociGenerationOptions,
) -> *mut c_char {
    let prompt = match prompt_from_ptr_len(prompt, prompt_len) {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let options = generation_options(options);
    let params = build_generation_params(options);

    match with_engine(engine, |engine| engine.engine.generate(&prompt, &params)) {
        Ok(output) => {
            clear_last_error();
            string_into_raw(output)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_unload_model_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| Ok(engine.engine.unload_model())) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_infer_with_len_and_options_json(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    options: *const LociGenerationOptions,
) -> *mut c_char {
    let prompt = match prompt_from_ptr_len(prompt, prompt_len) {
        Ok(value) => value,
        Err(_) => return ptr::null_mut(),
    };
    let options = generation_options(options);
    let params = build_generation_params(options);

    match with_engine(engine, |engine| engine.engine.infer(&prompt, &params)) {
        Ok(response) => {
            clear_last_error();
            json_into_raw(&response)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_runtime_snapshot_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| Ok(engine.engine.runtime_snapshot())) {
        Ok(snapshot) => {
            clear_last_error();
            json_into_raw(&snapshot)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs;
    use std::path::PathBuf;

    fn temp_model_path(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-ffi-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("demo.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        path
    }

    unsafe fn take_owned_string(value: *mut c_char) -> String {
        assert!(!value.is_null(), "ffi returned null");
        let text = CStr::from_ptr(value).to_str().expect("utf8").to_owned();
        loci_free_string(value);
        text
    }

    #[test]
    fn ffi_can_infer_and_unload_model_as_json() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null(), "engine");
            assert_eq!(loci_get_last_status_code(), LociStatusCode::Ok);

            let backend = CString::new("mock").expect("backend");
            let model_path = temp_model_path("infer-unload");
            let model_path_cstr =
                CString::new(model_path.display().to_string()).expect("model_path");

            let load_json = take_owned_string(loci_engine_load_model_json(
                engine,
                backend.as_ptr(),
                model_path_cstr.as_ptr(),
                ptr::null(),
            ));
            assert!(load_json.contains("\"active_backend\":\"mock\""));
            assert_eq!(loci_get_last_status_code(), LociStatusCode::Ok);

            let prompt = b"hello";
            let infer_json = take_owned_string(loci_infer_with_len_and_options_json(
                engine,
                prompt.as_ptr() as *const c_char,
                prompt.len() as u32,
                ptr::null(),
            ));
            assert!(infer_json.contains("\"output\":\"mock:hello"));
            assert!(infer_json.contains("\"backend\":\"mock\""));
            assert_eq!(loci_get_last_status_code(), LociStatusCode::Ok);

            let unload_json = take_owned_string(loci_engine_unload_model_json(engine));
            assert!(unload_json.contains("\"unloaded\":true"));
            assert!(unload_json.contains("\"previous_backend\":\"mock\""));
            assert_eq!(loci_get_last_status_code(), LociStatusCode::Ok);

            loci_engine_free(engine);
        }
    }

    #[test]
    fn ffi_reports_errors_for_null_prompt_pointer() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null(), "engine");

            let result = loci_infer_with_len_and_options_json(engine, ptr::null(), 5, ptr::null());
            assert!(result.is_null(), "call should fail");
            assert_eq!(loci_get_last_status_code(), LociStatusCode::InvalidArgument);

            let error = CStr::from_ptr(loci_get_last_error())
                .to_str()
                .expect("utf8");
            assert!(error.contains("prompt must not be null"));

            let error_json = take_owned_string(loci_get_last_error_json());
            assert!(error_json.contains("\"code\":\"invalid_argument\""));
            assert!(error_json.contains("prompt must not be null"));

            loci_engine_free(engine);
        }
    }

    #[test]
    fn ffi_classifies_backend_errors() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null(), "engine");

            let backend = CString::new("missing-backend").expect("backend");
            let model_path = temp_model_path("missing-backend");
            let model_path_cstr =
                CString::new(model_path.display().to_string()).expect("model_path");

            let result = loci_engine_load_model_json(
                engine,
                backend.as_ptr(),
                model_path_cstr.as_ptr(),
                ptr::null(),
            );
            assert!(result.is_null(), "call should fail");
            assert_eq!(
                loci_get_last_status_code(),
                LociStatusCode::BackendNotAvailable
            );

            let error_json = take_owned_string(loci_get_last_error_json());
            assert!(error_json.contains("\"code\":\"backend_not_available\""));
            assert!(error_json.contains("backend not available"));

            loci_engine_free(engine);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_backend_capabilities_json(
    engine: *mut LociEngine,
) -> *mut c_char {
    match with_engine(engine, |engine| Ok(engine.engine.backend_capabilities())) {
        Ok(backends) => {
            clear_last_error();
            json_into_raw(&backends)
        }
        Err(_) => ptr::null_mut(),
    }
}
