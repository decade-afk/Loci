use loci_core::{InferenceEngine, InferenceParams, ModelConfig, ModelLoadStrategy};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

const LOCI_ABI_VERSION: u32 = 1;
static LOCI_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty cstring"));
}

pub struct LociEngine {
    engine: InferenceEngine,
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

fn set_last_error(message: &str) {
    let sanitized = message.replace('\0', " ");
    let value = CString::new(sanitized)
        .unwrap_or_else(|_| CString::new("error message contains interior NUL").expect("cstring"));
    LAST_ERROR.with(|slot| *slot.borrow_mut() = value);
}

fn clear_last_error() {
    set_last_error("");
}

fn bool_from_u8(value: u8) -> bool {
    value != 0
}

fn string_into_raw(value: String) -> *mut c_char {
    match CString::new(value) {
        Ok(value) => value.into_raw(),
        Err(_) => {
            set_last_error("result contains interior NUL byte");
            ptr::null_mut()
        }
    }
}

fn json_into_raw<T: serde::Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(json) => string_into_raw(json),
        Err(error) => {
            set_last_error(&format!("serialization failed: {error}"));
            ptr::null_mut()
        }
    }
}

unsafe fn required_cstr(value: *const c_char, field: &str) -> Result<String, ()> {
    if value.is_null() {
        set_last_error(&format!("{field} must not be null"));
        return Err(());
    }

    let text = match CStr::from_ptr(value).to_str() {
        Ok(text) => text.trim(),
        Err(_) => {
            set_last_error(&format!("{field} must be valid UTF-8"));
            return Err(());
        }
    };

    if text.is_empty() {
        set_last_error(&format!("{field} must not be empty"));
        return Err(());
    }

    Ok(text.to_string())
}

unsafe fn prompt_from_ptr_len(prompt: *const c_char, prompt_len: u32) -> Result<String, ()> {
    if prompt_len == 0 {
        return Ok(String::new());
    }
    if prompt.is_null() {
        set_last_error("prompt must not be null when prompt_len > 0");
        return Err(());
    }
    let bytes = slice::from_raw_parts(prompt as *const u8, prompt_len as usize);
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_owned()),
        Err(_) => {
            set_last_error("prompt must be valid UTF-8");
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
        set_last_error("tensor_split must not be null when tensor_split_len > 0");
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
        set_last_error("engine must not be null");
        return Err(());
    }

    match f(&mut *engine) {
        Ok(value) => Ok(value),
        Err(error) => {
            set_last_error(&error.to_string());
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
pub unsafe extern "C" fn loci_get_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
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
            set_last_error(&error.to_string());
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
pub unsafe extern "C" fn loci_engine_runtime_snapshot_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| Ok(engine.engine.runtime_snapshot())) {
        Ok(snapshot) => {
            clear_last_error();
            json_into_raw(&snapshot)
        }
        Err(_) => ptr::null_mut(),
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
