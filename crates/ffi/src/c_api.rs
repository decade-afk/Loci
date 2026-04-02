use loci_core::{
    CoreComponent, CoreRewriterActivationRequest, InferenceEngine, ManagementService,
    ModelLoadConfig, ModelLoadRequest, ModelLoadSplitMode, ModelLoadStatus,
    ModelLoadStrategyRequest, PluginLoadRequest, PluginLoadSourceKind, TextGenerationParams,
    TextGenerationRequest,
};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

const LOCI_ABI_VERSION: u32 = 1;
const C_API_DEFAULT_MAX_PROMPT_BYTES: usize = 24 * 1024;
static LOCI_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty cstring"));
}

pub struct LociEngine {
    service: ManagementService,
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
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = value;
    });
}

fn clear_last_error() {
    set_last_error("");
}

fn c_api_max_prompt_bytes() -> usize {
    std::env::var("LOCI_MAX_PROMPT_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1024)
        .unwrap_or(C_API_DEFAULT_MAX_PROMPT_BYTES)
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

unsafe fn prompt_from_cstr_bounded(prompt: *const c_char) -> Result<String, ()> {
    if prompt.is_null() {
        set_last_error("prompt must not be null");
        return Err(());
    }

    let max_prompt_bytes = c_api_max_prompt_bytes();
    let mut nul_pos = None;
    for idx in 0..=(max_prompt_bytes + 1) {
        if *prompt.add(idx) == 0 {
            nul_pos = Some(idx);
            break;
        }
    }

    let len = match nul_pos {
        Some(len) if len <= max_prompt_bytes => len,
        _ => {
            set_last_error("prompt is too large for current native safety limit");
            return Err(());
        }
    };

    let bytes = slice::from_raw_parts(prompt as *const u8, len);
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_owned()),
        Err(_) => {
            set_last_error("prompt must be valid UTF-8");
            Err(())
        }
    }
}

unsafe fn prompt_from_ptr_len(prompt: *const c_char, prompt_len: u32) -> Result<String, ()> {
    let len = prompt_len as usize;
    if len > c_api_max_prompt_bytes() {
        set_last_error("prompt is too large for current native safety limit");
        return Err(());
    }

    if len == 0 {
        return Ok(String::new());
    }

    if prompt.is_null() {
        set_last_error("prompt must not be null when prompt_len > 0");
        return Err(());
    }

    let bytes = slice::from_raw_parts(prompt as *const u8, len);
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

fn parse_component(component: &str) -> Result<CoreComponent, ()> {
    match component.trim().to_ascii_lowercase().as_str() {
        "inference" => Ok(CoreComponent::Inference),
        "model" => Ok(CoreComponent::Model),
        "hardware" => Ok(CoreComponent::Hardware),
        "workflow" => Ok(CoreComponent::Workflow),
        "event_bus" => Ok(CoreComponent::EventBus),
        "plugin_manager" => Ok(CoreComponent::PluginManager),
        "ui_host" => Ok(CoreComponent::UiHost),
        _ => {
            set_last_error(
                "component must be one of: inference, model, hardware, workflow, event_bus, plugin_manager, ui_host",
            );
            Err(())
        }
    }
}

unsafe fn with_engine<T>(
    engine: *mut LociEngine,
    f: impl FnOnce(&LociEngine) -> loci_core::Result<T>,
) -> Result<T, ()> {
    if engine.is_null() {
        set_last_error("engine must not be null");
        return Err(());
    }

    match f(&*engine) {
        Ok(value) => Ok(value),
        Err(error) => {
            set_last_error(&error.to_string());
            Err(())
        }
    }
}

unsafe fn create_engine() -> Result<*mut LociEngine, ()> {
    match InferenceEngine::builder().build() {
        Ok(engine) => {
            clear_last_error();
            Ok(Box::into_raw(Box::new(LociEngine {
                service: ManagementService::new(engine),
            })))
        }
        Err(error) => {
            set_last_error(&error.to_string());
            Err(())
        }
    }
}

unsafe fn build_model_load_request(
    backend_name: *const c_char,
    model_path: *const c_char,
    options: *const LociModelLoadOptions,
) -> Result<ModelLoadRequest, ()> {
    let backend_name = required_cstr(backend_name, "backend_name")?;
    let model_path = required_cstr(model_path, "model_path")?;
    let options = model_load_options(options);
    let tensor_split = tensor_split_from_options(&options)?;

    Ok(ModelLoadRequest {
        backend_name,
        config: ModelLoadConfig {
            model_path,
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
                LociGpuSplitMode::None => ModelLoadSplitMode::None,
                LociGpuSplitMode::Layer => ModelLoadSplitMode::Layer,
                LociGpuSplitMode::Row => ModelLoadSplitMode::Row,
            },
            main_gpu: options.main_gpu,
            tensor_split,
            load_strategy: match options.load_strategy {
                LociModelLoadStrategyKind::Strict => ModelLoadStrategyRequest::Strict,
                LociModelLoadStrategyKind::AutoReduceGpuLayers => {
                    ModelLoadStrategyRequest::AutoReduceGpuLayers {
                        step: options.load_strategy_step,
                    }
                }
            },
        },
    })
}

unsafe fn build_generation_request(
    prompt: String,
    options: *const LociGenerationOptions,
) -> TextGenerationRequest {
    let options = generation_options(options);
    TextGenerationRequest {
        prompt,
        params: TextGenerationParams {
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
        },
    }
}

unsafe fn load_model_status(
    engine: *mut LociEngine,
    backend_name: *const c_char,
    model_path: *const c_char,
    options: *const LociModelLoadOptions,
) -> Result<ModelLoadStatus, ()> {
    let request = build_model_load_request(backend_name, model_path, options)?;
    with_engine(engine, |engine| engine.service.load_model(request))
}

unsafe fn generate_text_response(
    engine: *mut LociEngine,
    prompt: String,
    options: *const LociGenerationOptions,
) -> Result<loci_core::TextGenerationResponse, ()> {
    let request = build_generation_request(prompt, options);
    with_engine(engine, |engine| engine.service.generate_text(request))
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
    match create_engine() {
        Ok(engine) => engine,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_new_with_model(
    backend_name: *const c_char,
    model_path: *const c_char,
    options: *const LociModelLoadOptions,
) -> *mut LociEngine {
    let engine = match create_engine() {
        Ok(engine) => engine,
        Err(_) => return ptr::null_mut(),
    };

    if load_model_status(engine, backend_name, model_path, options).is_err() {
        loci_engine_free(engine);
        return ptr::null_mut();
    }

    engine
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_free(engine: *mut LociEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_free_safe(engine: *mut *mut LociEngine) {
    if engine.is_null() {
        return;
    }

    let current = *engine;
    if !current.is_null() {
        loci_engine_free(current);
        *engine = ptr::null_mut();
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_load_model_json(
    engine: *mut LociEngine,
    backend_name: *const c_char,
    model_path: *const c_char,
    options: *const LociModelLoadOptions,
) -> *mut c_char {
    match load_model_status(engine, backend_name, model_path, options) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_generate(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char {
    let prompt = match prompt_from_cstr_bounded(prompt) {
        Ok(prompt) => prompt,
        Err(_) => return ptr::null_mut(),
    };
    let options = LociGenerationOptions {
        max_tokens,
        temperature,
        ..LociGenerationOptions::default()
    };

    match generate_text_response(engine, prompt, &options) {
        Ok(response) => {
            clear_last_error();
            string_into_raw(response.output)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_generate_with_len(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char {
    let prompt = match prompt_from_ptr_len(prompt, prompt_len) {
        Ok(prompt) => prompt,
        Err(_) => return ptr::null_mut(),
    };
    let options = LociGenerationOptions {
        max_tokens,
        temperature,
        ..LociGenerationOptions::default()
    };

    match generate_text_response(engine, prompt, &options) {
        Ok(response) => {
            clear_last_error();
            string_into_raw(response.output)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_generate_with_options(
    engine: *mut LociEngine,
    prompt: *const c_char,
    options: *const LociGenerationOptions,
) -> *mut c_char {
    let prompt = match prompt_from_cstr_bounded(prompt) {
        Ok(prompt) => prompt,
        Err(_) => return ptr::null_mut(),
    };

    match generate_text_response(engine, prompt, options) {
        Ok(response) => {
            clear_last_error();
            string_into_raw(response.output)
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
        Ok(prompt) => prompt,
        Err(_) => return ptr::null_mut(),
    };

    match generate_text_response(engine, prompt, options) {
        Ok(response) => {
            clear_last_error();
            string_into_raw(response.output)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_runtime_snapshot_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.runtime_snapshot()) {
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
    match with_engine(engine, |engine| engine.service.backend_capabilities()) {
        Ok(backends) => {
            clear_last_error();
            json_into_raw(&backends)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_plugin_statuses_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.plugin_statuses()) {
        Ok(statuses) => {
            clear_last_error();
            json_into_raw(&statuses)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_plugin_detail_json(
    engine: *mut LociEngine,
    plugin_name: *const c_char,
) -> *mut c_char {
    let plugin_name = match required_cstr(plugin_name, "plugin_name") {
        Ok(plugin_name) => plugin_name,
        Err(_) => return ptr::null_mut(),
    };

    match with_engine(engine, |engine| engine.service.plugin_detail(&plugin_name)) {
        Ok(Some(detail)) => {
            clear_last_error();
            json_into_raw(&detail)
        }
        Ok(None) => {
            set_last_error("plugin not found");
            ptr::null_mut()
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_core_rewriter_inventory_json(
    engine: *mut LociEngine,
) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.core_rewriter_inventory()) {
        Ok(inventory) => {
            clear_last_error();
            json_into_raw(&inventory)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_workflow_inventory_json(
    engine: *mut LociEngine,
) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.workflow_inventory()) {
        Ok(inventory) => {
            clear_last_error();
            json_into_raw(&inventory)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_ui_inventory_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.ui_inventory()) {
        Ok(inventory) => {
            clear_last_error();
            json_into_raw(&inventory)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_event_inventory_json(engine: *mut LociEngine) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.event_inventory()) {
        Ok(inventory) => {
            clear_last_error();
            json_into_raw(&inventory)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_command_inventory_json(
    engine: *mut LociEngine,
) -> *mut c_char {
    match with_engine(engine, |engine| engine.service.command_inventory()) {
        Ok(inventory) => {
            clear_last_error();
            json_into_raw(&inventory)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_load_plugin_bundle_json(
    engine: *mut LociEngine,
    manifest_path: *const c_char,
) -> *mut c_char {
    let manifest_path = match required_cstr(manifest_path, "manifest_path") {
        Ok(path) => path,
        Err(_) => return ptr::null_mut(),
    };

    let request = PluginLoadRequest {
        path: manifest_path,
        source_kind: PluginLoadSourceKind::BundleFile,
    };

    match with_engine(engine, |engine| engine.service.load_plugins(request)) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_load_plugin_dir_json(
    engine: *mut LociEngine,
    plugin_dir: *const c_char,
) -> *mut c_char {
    let plugin_dir = match required_cstr(plugin_dir, "plugin_dir") {
        Ok(path) => path,
        Err(_) => return ptr::null_mut(),
    };

    let request = PluginLoadRequest {
        path: plugin_dir,
        source_kind: PluginLoadSourceKind::Directory,
    };

    match with_engine(engine, |engine| engine.service.load_plugins(request)) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_activate_core_rewriter_json(
    engine: *mut LociEngine,
    component: *const c_char,
    plugin_name: *const c_char,
) -> *mut c_char {
    let component = match required_cstr(component, "component") {
        Ok(component) => component,
        Err(_) => return ptr::null_mut(),
    };
    let component = match parse_component(&component) {
        Ok(component) => component,
        Err(_) => return ptr::null_mut(),
    };
    let plugin_name = match required_cstr(plugin_name, "plugin_name") {
        Ok(plugin_name) => plugin_name,
        Err(_) => return ptr::null_mut(),
    };

    let request = CoreRewriterActivationRequest {
        component,
        plugin_name,
    };

    match with_engine(engine, |engine| {
        engine.service.activate_core_rewriter(request)
    }) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_activate_legacy_text_plugin_json(
    engine: *mut LociEngine,
    plugin_name: *const c_char,
) -> *mut c_char {
    let plugin_name = match required_cstr(plugin_name, "plugin_name") {
        Ok(plugin_name) => plugin_name,
        Err(_) => return ptr::null_mut(),
    };

    match with_engine(engine, |engine| {
        engine.service.activate_legacy_text_plugin(&plugin_name)
    }) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn loci_engine_deactivate_legacy_text_plugin_json(
    engine: *mut LociEngine,
    plugin_name: *const c_char,
) -> *mut c_char {
    let plugin_name = match required_cstr(plugin_name, "plugin_name") {
        Ok(plugin_name) => plugin_name,
        Err(_) => return ptr::null_mut(),
    };

    match with_engine(engine, |engine| {
        engine.service.deactivate_legacy_text_plugin(&plugin_name)
    }) {
        Ok(status) => {
            clear_last_error();
            json_into_raw(&status)
        }
        Err(_) => ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "loci-ffi-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        dir
    }

    fn write_model_file(name: &str) -> CString {
        let dir = temp_dir(name);
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("model.gguf");
        fs::write(&path, b"mock-model").expect("write model");
        CString::new(path.display().to_string()).expect("path cstring")
    }

    fn write_plugin_dir() -> CString {
        let dir = temp_dir("plugins");
        let plugin = dir.join("plugin-a");
        fs::create_dir_all(&plugin).expect("mkdir plugin");
        fs::write(
            plugin.join("manifest.toml"),
            r#"
name = "plugin-a"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_agent"]

[contributes]
workflows = ["agent.workflow"]

[core_rewriters]
workflow = true
"#,
        )
        .expect("write manifest");
        CString::new(dir.display().to_string()).expect("dir cstring")
    }

    unsafe fn take_string(ptr: *mut c_char) -> String {
        assert!(!ptr.is_null(), "expected non-null string pointer");
        let value = CStr::from_ptr(ptr).to_string_lossy().into_owned();
        loci_free_string(ptr);
        value
    }

    unsafe fn parse_json(ptr: *mut c_char) -> Value {
        serde_json::from_str(&take_string(ptr)).expect("json")
    }

    #[test]
    fn version_and_abi_are_exposed() {
        unsafe {
            let version = CStr::from_ptr(loci_version())
                .to_str()
                .expect("utf8 version");
            assert_eq!(loci_abi_version(), 1);
            assert_eq!(version, env!("CARGO_PKG_VERSION"));
        }
    }

    #[test]
    fn runtime_snapshot_and_backends_are_exposed_as_json() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null());

            let snapshot = parse_json(loci_engine_runtime_snapshot_json(engine));
            assert_eq!(snapshot["plugin_count"], 0);

            let backends = parse_json(loci_engine_backend_capabilities_json(engine));
            assert!(backends
                .as_array()
                .expect("array")
                .iter()
                .any(|backend| { backend["name"] == Value::String("mock".to_string()) }));

            loci_engine_free(engine);
        }
    }

    #[test]
    fn load_model_and_generate_through_ffi() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null());

            let backend = CString::new("mock").expect("backend");
            let model_path = write_model_file("load-and-generate");
            let status = parse_json(loci_engine_load_model_json(
                engine,
                backend.as_ptr(),
                model_path.as_ptr(),
                ptr::null(),
            ));
            assert_eq!(status["status"], "loaded");
            assert_eq!(status["backend_name"], "mock");

            let prompt = CString::new("hello from ffi").expect("prompt");
            let output = take_string(loci_generate(engine, prompt.as_ptr(), 32, 0.5));
            assert!(output.contains("mock:hello from ffi"));
            assert!(output.contains("max_tokens=32"));

            loci_engine_free(engine);
        }
    }

    #[test]
    fn plugin_directory_loading_and_activation_are_visible() {
        unsafe {
            let engine = loci_engine_new();
            assert!(!engine.is_null());

            let plugin_dir = write_plugin_dir();
            let load_status = parse_json(loci_engine_load_plugin_dir_json(
                engine,
                plugin_dir.as_ptr(),
            ));
            assert_eq!(load_status["status"], "loaded");
            assert_eq!(load_status["loaded_count"], 1);

            let component = CString::new("workflow").expect("component");
            let plugin_name = CString::new("plugin-a").expect("plugin");
            let activation = parse_json(loci_engine_activate_core_rewriter_json(
                engine,
                component.as_ptr(),
                plugin_name.as_ptr(),
            ));
            assert_eq!(activation["status"], "activated");

            let inventory = parse_json(loci_engine_core_rewriter_inventory_json(engine));
            assert!(inventory.as_array().expect("array").iter().any(|entry| {
                entry["component"] == Value::String("workflow".to_string())
                    && entry["active_plugin_name"] == Value::String("plugin-a".to_string())
            }));

            loci_engine_free(engine);
        }
    }

    #[test]
    fn null_engine_sets_last_error() {
        unsafe {
            let snapshot = loci_engine_runtime_snapshot_json(ptr::null_mut());
            assert!(snapshot.is_null());
            let err = CStr::from_ptr(loci_get_last_error())
                .to_str()
                .expect("utf8 error");
            assert!(err.contains("engine must not be null"));
        }
    }

    #[test]
    fn free_safe_nulls_pointer() {
        unsafe {
            let mut engine = loci_engine_new();
            assert!(!engine.is_null());
            loci_engine_free_safe(&mut engine);
            assert!(engine.is_null());
        }
    }
}
