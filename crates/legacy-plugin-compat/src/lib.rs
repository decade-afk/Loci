use anyhow::{bail, Context, Result};
use libloading::{Library, Symbol};
use loci_legacy_plugin_api::abi::{
    LegacyPluginApiV2, LegacyPluginBuffer, LegacyPluginConstructorV2, LegacyPluginStringView,
    LEGACY_PLUGIN_ABI_V2,
};
use loci_legacy_plugin_api::plugin::{dynamic_plugin_from_opaque, DynamicPluginOpaque, Plugin};
use std::path::Path;
use std::sync::{Arc, Mutex};

type PluginConstructorV1 = unsafe extern "C" fn() -> DynamicPluginOpaque;

pub trait LegacyTextCompat: Send + Sync {
    fn pre_generate(&self, prompt: &str) -> Result<String>;
    fn post_generate(&self, response: &str) -> Result<String>;
    fn transform_logits(&self, logits: &mut [f32], context_tokens: &[i32]) -> Result<()>;
    fn post_sample(&self, token_id: i32) -> Result<i32>;
}

enum LoadedLegacyPluginRuntime {
    V1(Box<dyn Plugin>),
    V2(LegacyPluginApiV2),
}

// The legacy plugin contract has always required implementations to be
// thread-safe (`Plugin: Send + Sync`). The stable v2 ABI preserves that
// contract, but the raw function-table pointers cannot express it directly.
unsafe impl Send for LoadedLegacyPluginRuntime {}
unsafe impl Sync for LoadedLegacyPluginRuntime {}

struct LoadedLegacyTextPlugin {
    runtime: Mutex<Option<LoadedLegacyPluginRuntime>>,
    _library: Option<Arc<Library>>,
    supports_pre_generate: bool,
    supports_post_generate: bool,
    supports_transform_logits: bool,
    supports_post_sample: bool,
}

impl LoadedLegacyTextPlugin {
    fn with_runtime<T>(
        &self,
        f: impl FnOnce(&LoadedLegacyPluginRuntime) -> Result<T>,
    ) -> Result<T> {
        let guard = self
            .runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("legacy plugin mutex poisoned"))?;
        let runtime = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("legacy plugin already released"))?;
        f(runtime)
    }
}

impl Drop for LoadedLegacyTextPlugin {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.runtime.lock() {
            if let Some(runtime) = guard.take() {
                match runtime {
                    LoadedLegacyPluginRuntime::V1(mut plugin) => {
                        let _ = plugin.cleanup();
                    }
                    LoadedLegacyPluginRuntime::V2(api) => unsafe {
                        let mut error = LegacyPluginBuffer::empty();
                        let _ = (api.cleanup)(api.instance, &mut error as *mut _);
                        free_plugin_buffer(api.free_buffer, error);
                        (api.destroy)(api.instance);
                    },
                }
            }
        }
    }
}

impl LegacyTextCompat for LoadedLegacyTextPlugin {
    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if !self.supports_pre_generate {
            return Ok(prompt.to_string());
        }

        self.with_runtime(|runtime| match runtime {
            LoadedLegacyPluginRuntime::V1(plugin) => plugin
                .pre_generate(prompt)
                .map_err(|error| anyhow::anyhow!(error.to_string())),
            LoadedLegacyPluginRuntime::V2(api) => unsafe {
                call_v2_text_hook(
                    *api,
                    api.pre_generate,
                    prompt,
                    "pre_generate",
                    prompt.to_string(),
                )
            },
        })
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        if !self.supports_post_generate {
            return Ok(response.to_string());
        }

        self.with_runtime(|runtime| match runtime {
            LoadedLegacyPluginRuntime::V1(plugin) => plugin
                .post_generate(response)
                .map_err(|error| anyhow::anyhow!(error.to_string())),
            LoadedLegacyPluginRuntime::V2(api) => unsafe {
                call_v2_text_hook(
                    *api,
                    api.post_generate,
                    response,
                    "post_generate",
                    response.to_string(),
                )
            },
        })
    }

    fn transform_logits(&self, logits: &mut [f32], context_tokens: &[i32]) -> Result<()> {
        if !self.supports_transform_logits {
            return Ok(());
        }

        self.with_runtime(|runtime| match runtime {
            LoadedLegacyPluginRuntime::V1(plugin) => {
                let mut legacy_logits = loci_legacy_plugin_api::sampler::LogitsView::new(logits);
                plugin
                    .transform_logits(&mut legacy_logits, context_tokens)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))
            }
            LoadedLegacyPluginRuntime::V2(api) => unsafe {
                call_v2_transform_logits(*api, logits, context_tokens)
            },
        })
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        if !self.supports_post_sample {
            return Ok(token_id);
        }

        self.with_runtime(|runtime| match runtime {
            LoadedLegacyPluginRuntime::V1(plugin) => plugin
                .post_sample(token_id)
                .map_err(|error| anyhow::anyhow!(error.to_string())),
            LoadedLegacyPluginRuntime::V2(api) => unsafe { call_v2_post_sample(*api, token_id) },
        })
    }
}

pub fn load_legacy_text_plugin_compat(
    runtime_artifact_path: &Path,
    expected_name: &str,
    expected_version: &str,
    capabilities: &[String],
) -> Result<Option<Arc<dyn LegacyTextCompat>>> {
    let supports_pre_generate = capabilities.iter().any(|cap| cap == "pre_generate");
    let supports_post_generate = capabilities.iter().any(|cap| cap == "post_generate");
    let supports_transform_logits = capabilities.iter().any(|cap| cap == "transform_logits");
    let supports_post_sample = capabilities.iter().any(|cap| cap == "post_sample");
    if !supports_pre_generate
        && !supports_post_generate
        && !supports_transform_logits
        && !supports_post_sample
    {
        return Ok(None);
    }

    if !runtime_artifact_path.exists() {
        bail!(
            "legacy runtime artifact not found: {}",
            runtime_artifact_path.display()
        );
    }

    let library = Arc::new(unsafe {
        Library::new(runtime_artifact_path).with_context(|| {
            format!(
                "failed to load legacy plugin library: {}",
                runtime_artifact_path.display()
            )
        })?
    });

    if let Some(constructor) = unsafe { load_plugin_constructor_v2(&library)? } {
        let api = unsafe { constructor() };
        return load_legacy_text_plugin_compat_v2(
            api,
            expected_name,
            expected_version,
            supports_pre_generate,
            supports_post_generate,
            supports_transform_logits,
            supports_post_sample,
            Some(Arc::clone(&library)),
        );
    }

    let constructor = unsafe { load_plugin_constructor_v1(&library, runtime_artifact_path)? };
    let plugin = unsafe {
        dynamic_plugin_from_opaque(constructor())
            .ok_or_else(|| anyhow::anyhow!("legacy plugin constructor returned invalid payload"))?
    };

    load_legacy_text_plugin_compat_v1(
        plugin,
        expected_name,
        expected_version,
        supports_pre_generate,
        supports_post_generate,
        supports_transform_logits,
        supports_post_sample,
        Some(library),
    )
}

unsafe fn load_plugin_constructor_v2(
    library: &Arc<Library>,
) -> Result<Option<Symbol<'_, LegacyPluginConstructorV2>>> {
    match library.get(b"loci_legacy_plugin_create_v2") {
        Ok(symbol) => Ok(Some(symbol)),
        Err(_) => match library.get(b"create_plugin_v2") {
            Ok(symbol) => Ok(Some(symbol)),
            Err(_) => Ok(None),
        },
    }
}

unsafe fn load_plugin_constructor_v1<'lib>(
    library: &'lib Arc<Library>,
    runtime_artifact_path: &Path,
) -> Result<Symbol<'lib, PluginConstructorV1>> {
    match library.get(b"create_plugin_v1") {
        Ok(symbol) => Ok(symbol),
        Err(_) => library.get(b"create_plugin").with_context(|| {
            format!(
                "failed to find legacy plugin constructor symbol in {}",
                runtime_artifact_path.display()
            )
        }),
    }
}

fn load_legacy_text_plugin_compat_v1(
    mut plugin: Box<dyn Plugin>,
    expected_name: &str,
    expected_version: &str,
    supports_pre_generate: bool,
    supports_post_generate: bool,
    supports_transform_logits: bool,
    supports_post_sample: bool,
    library: Option<Arc<Library>>,
) -> Result<Option<Arc<dyn LegacyTextCompat>>> {
    if plugin.name() != expected_name {
        bail!(
            "legacy plugin runtime name `{}` does not match contract name `{expected_name}`",
            plugin.name()
        );
    }
    if !expected_version.trim().is_empty() && plugin.version() != expected_version {
        bail!(
            "legacy plugin runtime version `{}` does not match contract version `{expected_version}`",
            plugin.version()
        );
    }

    plugin
        .init()
        .map_err(|error| anyhow::anyhow!("legacy plugin init failed: {}", error))?;

    Ok(Some(Arc::new(LoadedLegacyTextPlugin {
        runtime: Mutex::new(Some(LoadedLegacyPluginRuntime::V1(plugin))),
        _library: library,
        supports_pre_generate,
        supports_post_generate,
        supports_transform_logits,
        supports_post_sample,
    })))
}

fn load_legacy_text_plugin_compat_v2(
    api: LegacyPluginApiV2,
    expected_name: &str,
    expected_version: &str,
    supports_pre_generate: bool,
    supports_post_generate: bool,
    supports_transform_logits: bool,
    supports_post_sample: bool,
    library: Option<Arc<Library>>,
) -> Result<Option<Arc<dyn LegacyTextCompat>>> {
    if api.abi_version != LEGACY_PLUGIN_ABI_V2 {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!(
            "legacy plugin runtime ABI v{} does not match stable host ABI v{}",
            api.abi_version,
            LEGACY_PLUGIN_ABI_V2
        );
    }
    if api.instance.is_null() {
        bail!("legacy plugin constructor returned a null instance");
    }
    if supports_pre_generate && api.pre_generate.is_none() {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin declares `pre_generate`, but the v2 ABI hook is missing");
    }
    if supports_post_generate && api.post_generate.is_none() {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin declares `post_generate`, but the v2 ABI hook is missing");
    }
    if supports_transform_logits && api.transform_logits.is_none() {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin declares `transform_logits`, but the v2 ABI hook is missing");
    }
    if supports_post_sample && api.post_sample.is_none() {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin declares `post_sample`, but the v2 ABI hook is missing");
    }

    let name = unsafe { string_view_to_string((api.name)(api.instance.cast_const()))? };
    let version = unsafe { string_view_to_string((api.version)(api.instance.cast_const()))? };
    if name != expected_name {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin runtime name `{name}` does not match contract name `{expected_name}`");
    }
    if !expected_version.trim().is_empty() && version != expected_version {
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!(
            "legacy plugin runtime version `{version}` does not match contract version `{expected_version}`"
        );
    }

    let mut error = LegacyPluginBuffer::empty();
    let initialized = unsafe { (api.init)(api.instance, &mut error as *mut _) };
    if !initialized {
        let message =
            unsafe { decode_plugin_buffer(api.free_buffer, error, "legacy plugin init failed")? };
        unsafe {
            (api.destroy)(api.instance);
        }
        bail!("legacy plugin init failed: {message}");
    }
    unsafe {
        free_plugin_buffer(api.free_buffer, error);
    }

    Ok(Some(Arc::new(LoadedLegacyTextPlugin {
        runtime: Mutex::new(Some(LoadedLegacyPluginRuntime::V2(api))),
        _library: library,
        supports_pre_generate,
        supports_post_generate,
        supports_transform_logits,
        supports_post_sample,
    })))
}

unsafe fn call_v2_text_hook(
    api: LegacyPluginApiV2,
    hook: Option<
        unsafe extern "C" fn(
            instance: *const std::ffi::c_void,
            input: LegacyPluginStringView,
            output: *mut LegacyPluginBuffer,
            error: *mut LegacyPluginBuffer,
        ) -> bool,
    >,
    input: &str,
    hook_name: &str,
    default_output: String,
) -> Result<String> {
    let Some(hook) = hook else {
        return Ok(default_output);
    };

    let mut output = LegacyPluginBuffer::empty();
    let mut error = LegacyPluginBuffer::empty();
    let ok = hook(
        api.instance.cast_const(),
        LegacyPluginStringView::from_str(input),
        &mut output as *mut _,
        &mut error as *mut _,
    );

    if ok {
        free_plugin_buffer(api.free_buffer, error);
        return decode_plugin_buffer(api.free_buffer, output, hook_name);
    }

    free_plugin_buffer(api.free_buffer, output);
    let message = decode_plugin_buffer(
        api.free_buffer,
        error,
        &format!("legacy plugin {hook_name} failed"),
    )?;
    bail!("{message}");
}

unsafe fn call_v2_transform_logits(
    api: LegacyPluginApiV2,
    logits: &mut [f32],
    context_tokens: &[i32],
) -> Result<()> {
    let Some(hook) = api.transform_logits else {
        return Ok(());
    };

    let mut error = LegacyPluginBuffer::empty();
    let ok = hook(
        api.instance.cast_const(),
        logits.as_mut_ptr(),
        logits.len(),
        context_tokens.as_ptr(),
        context_tokens.len(),
        &mut error as *mut _,
    );
    if ok {
        free_plugin_buffer(api.free_buffer, error);
        return Ok(());
    }

    let message = decode_plugin_buffer(
        api.free_buffer,
        error,
        "legacy plugin transform_logits failed",
    )?;
    bail!("{message}");
}

unsafe fn call_v2_post_sample(api: LegacyPluginApiV2, token_id: i32) -> Result<i32> {
    let Some(hook) = api.post_sample else {
        return Ok(token_id);
    };

    let mut output = token_id;
    let mut error = LegacyPluginBuffer::empty();
    let ok = hook(
        api.instance.cast_const(),
        token_id,
        &mut output as *mut _,
        &mut error as *mut _,
    );
    if ok {
        free_plugin_buffer(api.free_buffer, error);
        return Ok(output);
    }

    let message = decode_plugin_buffer(api.free_buffer, error, "legacy plugin post_sample failed")?;
    bail!("{message}");
}

unsafe fn string_view_to_string(view: LegacyPluginStringView) -> Result<String> {
    if view.ptr.is_null() {
        if view.len == 0 {
            return Ok(String::new());
        }
        bail!("legacy plugin returned a null string view with non-zero length");
    }

    String::from_utf8(std::slice::from_raw_parts(view.ptr, view.len).to_vec())
        .map_err(|_| anyhow::anyhow!("legacy plugin string view must be valid UTF-8"))
}

unsafe fn decode_plugin_buffer(
    free_buffer: unsafe extern "C" fn(LegacyPluginBuffer),
    buffer: LegacyPluginBuffer,
    fallback: &str,
) -> Result<String> {
    let result = if buffer.ptr.is_null() {
        fallback.to_string()
    } else {
        String::from_utf8(std::slice::from_raw_parts(buffer.ptr, buffer.len).to_vec())
            .map_err(|_| anyhow::anyhow!("legacy plugin buffer must be valid UTF-8"))?
    };
    free_plugin_buffer(free_buffer, buffer);
    Ok(result)
}

unsafe fn free_plugin_buffer(
    free_buffer: unsafe extern "C" fn(LegacyPluginBuffer),
    buffer: LegacyPluginBuffer,
) {
    if !buffer.ptr.is_null() || buffer.len != 0 || buffer.cap != 0 {
        free_buffer(buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_legacy_plugin_api::abi::export_plugin_v2;

    struct LegacySamplerPlugin;

    impl Plugin for LegacySamplerPlugin {
        fn name(&self) -> &str {
            "legacy-sampler"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> loci_legacy_plugin_api::error::Result<String> {
            Ok(format!("[pre]{prompt}"))
        }

        fn post_generate(&self, response: &str) -> loci_legacy_plugin_api::error::Result<String> {
            Ok(format!("{response}[post]"))
        }

        fn transform_logits(
            &self,
            logits: &mut loci_legacy_plugin_api::sampler::LogitsView<'_>,
            _context_tokens: &[i32],
        ) -> loci_legacy_plugin_api::error::Result<()> {
            logits.set_usize(1, 77.0)?;
            Ok(())
        }

        fn post_sample(&self, _token_id: i32) -> loci_legacy_plugin_api::error::Result<i32> {
            Ok(9)
        }
    }

    #[test]
    fn loaded_legacy_sampler_compat_applies_hooks_from_v1_runtime() {
        let compat = LoadedLegacyTextPlugin {
            runtime: Mutex::new(Some(LoadedLegacyPluginRuntime::V1(Box::new(
                LegacySamplerPlugin,
            )))),
            _library: None,
            supports_pre_generate: true,
            supports_post_generate: true,
            supports_transform_logits: true,
            supports_post_sample: true,
        };

        assert_eq!(
            compat.pre_generate("hello").expect("pre generate"),
            "[pre]hello"
        );
        let mut logits = vec![1.0, 2.0, 3.0];
        compat
            .transform_logits(&mut logits, &[])
            .expect("transform logits");
        assert_eq!(logits[1], 77.0);
        assert_eq!(compat.post_sample(1).expect("post sample"), 9);
        assert_eq!(
            compat.post_generate("world").expect("post generate"),
            "world[post]"
        );
    }

    #[test]
    fn loaded_legacy_sampler_compat_applies_hooks_from_v2_runtime() {
        let api = export_plugin_v2(LegacySamplerPlugin);
        let compat = load_legacy_text_plugin_compat_v2(
            api,
            "legacy-sampler",
            "1.0.0",
            true,
            true,
            true,
            true,
            None,
        )
        .expect("load v2 runtime")
        .expect("compat runtime");

        assert_eq!(
            compat.pre_generate("hello").expect("pre generate"),
            "[pre]hello"
        );
        let mut logits = vec![1.0, 2.0, 3.0];
        compat
            .transform_logits(&mut logits, &[])
            .expect("transform logits");
        assert_eq!(logits[1], 77.0);
        assert_eq!(compat.post_sample(1).expect("post sample"), 9);
        assert_eq!(
            compat.post_generate("world").expect("post generate"),
            "world[post]"
        );
    }
}
