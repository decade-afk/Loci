use crate::error::Result;
use crate::sampler::LogitsView;
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicPluginOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynPluginPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

pub fn dynamic_plugin_into_opaque(plugin: Box<dyn Plugin>) -> DynamicPluginOpaque {
    let raw: *mut dyn Plugin = Box::into_raw(plugin);
    let parts: RawDynPluginPtr = unsafe { std::mem::transmute(raw) };
    DynamicPluginOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

pub unsafe fn dynamic_plugin_from_opaque(opaque: DynamicPluginOpaque) -> Option<Box<dyn Plugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynPluginPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn Plugin = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(prompt.to_string())
    }

    fn transform_logits(&self, logits: &mut LogitsView, context_tokens: &[i32]) -> Result<()> {
        let _ = logits;
        let _ = context_tokens;
        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        Ok(token_id)
    }

    fn on_token(&self, token: &str) -> Result<String> {
        Ok(token.to_string())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(response.to_string())
    }

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
