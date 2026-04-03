use crate::plugin::Plugin;
use crate::sampler::LogitsView;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr;

pub const LEGACY_PLUGIN_ABI_V1: u32 = 1;
pub const LEGACY_PLUGIN_ABI_V2: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LegacyPluginBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

impl LegacyPluginBuffer {
    pub const fn empty() -> Self {
        Self {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        }
    }

    pub fn from_vec(buffer: Vec<u8>) -> Self {
        let mut buffer = ManuallyDrop::new(buffer);
        Self {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
            cap: buffer.capacity(),
        }
    }

    pub fn from_string(value: String) -> Self {
        Self::from_vec(value.into_bytes())
    }

    pub unsafe fn into_vec(self) -> Vec<u8> {
        if self.ptr.is_null() {
            debug_assert_eq!(self.len, 0);
            debug_assert_eq!(self.cap, 0);
            Vec::new()
        } else {
            Vec::from_raw_parts(self.ptr, self.len, self.cap)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LegacyPluginStringView {
    pub ptr: *const u8,
    pub len: usize,
}

impl LegacyPluginStringView {
    pub const fn empty() -> Self {
        Self {
            ptr: ptr::null(),
            len: 0,
        }
    }

    pub fn from_str(value: &str) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }

    pub unsafe fn as_bytes(self) -> &'static [u8] {
        if self.ptr.is_null() {
            debug_assert_eq!(self.len, 0);
            &[]
        } else {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    pub unsafe fn to_str(self) -> std::result::Result<&'static str, std::str::Utf8Error> {
        std::str::from_utf8(self.as_bytes())
    }
}

pub type LegacyPluginConstructorV2 = unsafe extern "C" fn() -> LegacyPluginApiV2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LegacyPluginApiV2 {
    pub abi_version: u32,
    pub instance: *mut c_void,
    pub destroy: unsafe extern "C" fn(instance: *mut c_void),
    pub free_buffer: unsafe extern "C" fn(buffer: LegacyPluginBuffer),
    pub name: unsafe extern "C" fn(instance: *const c_void) -> LegacyPluginStringView,
    pub version: unsafe extern "C" fn(instance: *const c_void) -> LegacyPluginStringView,
    pub init: unsafe extern "C" fn(instance: *mut c_void, error: *mut LegacyPluginBuffer) -> bool,
    pub cleanup:
        unsafe extern "C" fn(instance: *mut c_void, error: *mut LegacyPluginBuffer) -> bool,
    pub pre_generate: Option<
        unsafe extern "C" fn(
            instance: *const c_void,
            prompt: LegacyPluginStringView,
            output: *mut LegacyPluginBuffer,
            error: *mut LegacyPluginBuffer,
        ) -> bool,
    >,
    pub transform_logits: Option<
        unsafe extern "C" fn(
            instance: *const c_void,
            logits: *mut f32,
            logits_len: usize,
            context_tokens: *const i32,
            context_tokens_len: usize,
            error: *mut LegacyPluginBuffer,
        ) -> bool,
    >,
    pub post_sample: Option<
        unsafe extern "C" fn(
            instance: *const c_void,
            token_id: i32,
            output: *mut i32,
            error: *mut LegacyPluginBuffer,
        ) -> bool,
    >,
    pub post_generate: Option<
        unsafe extern "C" fn(
            instance: *const c_void,
            response: LegacyPluginStringView,
            output: *mut LegacyPluginBuffer,
            error: *mut LegacyPluginBuffer,
        ) -> bool,
    >,
}

pub fn export_plugin_v2<T>(plugin: T) -> LegacyPluginApiV2
where
    T: Plugin + 'static,
{
    LegacyPluginApiV2 {
        abi_version: LEGACY_PLUGIN_ABI_V2,
        instance: Box::into_raw(Box::new(plugin)).cast::<c_void>(),
        destroy: destroy_plugin::<T>,
        free_buffer: free_plugin_buffer,
        name: plugin_name::<T>,
        version: plugin_version::<T>,
        init: plugin_init::<T>,
        cleanup: plugin_cleanup::<T>,
        pre_generate: Some(plugin_pre_generate::<T>),
        transform_logits: Some(plugin_transform_logits::<T>),
        post_sample: Some(plugin_post_sample::<T>),
        post_generate: Some(plugin_post_generate::<T>),
    }
}

#[macro_export]
macro_rules! export_legacy_plugin_v2 {
    ($plugin:expr) => {
        #[no_mangle]
        pub extern "C" fn loci_legacy_plugin_create_v2() -> $crate::abi::LegacyPluginApiV2 {
            $crate::abi::export_plugin_v2($plugin)
        }
    };
}

unsafe extern "C" fn destroy_plugin<T>(instance: *mut c_void)
where
    T: Plugin + 'static,
{
    if !instance.is_null() {
        let _ = Box::from_raw(instance.cast::<T>());
    }
}

unsafe extern "C" fn free_plugin_buffer(buffer: LegacyPluginBuffer) {
    let _ = buffer.into_vec();
}

unsafe extern "C" fn plugin_name<T>(instance: *const c_void) -> LegacyPluginStringView
where
    T: Plugin + 'static,
{
    instance
        .cast::<T>()
        .as_ref()
        .map(|plugin| LegacyPluginStringView::from_str(plugin.name()))
        .unwrap_or_else(LegacyPluginStringView::empty)
}

unsafe extern "C" fn plugin_version<T>(instance: *const c_void) -> LegacyPluginStringView
where
    T: Plugin + 'static,
{
    instance
        .cast::<T>()
        .as_ref()
        .map(|plugin| LegacyPluginStringView::from_str(plugin.version()))
        .unwrap_or_else(LegacyPluginStringView::empty)
}

unsafe extern "C" fn plugin_init<T>(instance: *mut c_void, error: *mut LegacyPluginBuffer) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(error);
    let Some(plugin) = instance.cast::<T>().as_mut() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };

    match plugin.init() {
        Ok(()) => true,
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe extern "C" fn plugin_cleanup<T>(
    instance: *mut c_void,
    error: *mut LegacyPluginBuffer,
) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(error);
    let Some(plugin) = instance.cast::<T>().as_mut() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };

    match plugin.cleanup() {
        Ok(()) => true,
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe extern "C" fn plugin_pre_generate<T>(
    instance: *const c_void,
    prompt: LegacyPluginStringView,
    output: *mut LegacyPluginBuffer,
    error: *mut LegacyPluginBuffer,
) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(output);
    clear_buffer(error);

    let Some(plugin) = instance.cast::<T>().as_ref() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };
    let prompt = match prompt.to_str() {
        Ok(prompt) => prompt,
        Err(_) => {
            write_error(error, "prompt must be valid UTF-8");
            return false;
        }
    };

    match plugin.pre_generate(prompt) {
        Ok(value) => {
            write_buffer(output, LegacyPluginBuffer::from_string(value));
            true
        }
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe extern "C" fn plugin_transform_logits<T>(
    instance: *const c_void,
    logits: *mut f32,
    logits_len: usize,
    context_tokens: *const i32,
    context_tokens_len: usize,
    error: *mut LegacyPluginBuffer,
) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(error);

    let Some(plugin) = instance.cast::<T>().as_ref() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };
    if logits.is_null() && logits_len > 0 {
        write_error(error, "logits pointer must not be null when logits_len > 0");
        return false;
    }
    if context_tokens.is_null() && context_tokens_len > 0 {
        write_error(
            error,
            "context_tokens pointer must not be null when context_tokens_len > 0",
        );
        return false;
    }

    let mut logits = LogitsView::from_raw(logits, logits_len);
    let context_tokens = if context_tokens.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(context_tokens, context_tokens_len)
    };

    match plugin.transform_logits(&mut logits, context_tokens) {
        Ok(()) => true,
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe extern "C" fn plugin_post_sample<T>(
    instance: *const c_void,
    token_id: i32,
    output: *mut i32,
    error: *mut LegacyPluginBuffer,
) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(error);

    let Some(plugin) = instance.cast::<T>().as_ref() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };
    let Some(output) = output.as_mut() else {
        write_error(error, "post-sample output pointer must not be null");
        return false;
    };

    match plugin.post_sample(token_id) {
        Ok(value) => {
            *output = value;
            true
        }
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe extern "C" fn plugin_post_generate<T>(
    instance: *const c_void,
    response: LegacyPluginStringView,
    output: *mut LegacyPluginBuffer,
    error: *mut LegacyPluginBuffer,
) -> bool
where
    T: Plugin + 'static,
{
    clear_buffer(output);
    clear_buffer(error);

    let Some(plugin) = instance.cast::<T>().as_ref() else {
        write_error(error, "plugin instance must not be null");
        return false;
    };
    let response = match response.to_str() {
        Ok(response) => response,
        Err(_) => {
            write_error(error, "response must be valid UTF-8");
            return false;
        }
    };

    match plugin.post_generate(response) {
        Ok(value) => {
            write_buffer(output, LegacyPluginBuffer::from_string(value));
            true
        }
        Err(err) => {
            write_error(error, err.to_string());
            false
        }
    }
}

unsafe fn write_error(slot: *mut LegacyPluginBuffer, message: impl Into<String>) {
    write_buffer(slot, LegacyPluginBuffer::from_string(message.into()));
}

unsafe fn clear_buffer(slot: *mut LegacyPluginBuffer) {
    write_buffer(slot, LegacyPluginBuffer::empty());
}

unsafe fn write_buffer(slot: *mut LegacyPluginBuffer, buffer: LegacyPluginBuffer) {
    if let Some(slot) = slot.as_mut() {
        *slot = buffer;
    }
}
