//! Dynamic tool plugin support.
//!
//! This module lets external shared libraries contribute executable tools
//! into the function-calling subsystem at runtime.

use crate::error::{LociError, Result};
use crate::function_calling::{
    FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionHandler,
};
use crate::plugin_contract::{
    load_and_validate_plugin_contract, validate_runtime_plugin_identity, PluginContractKind,
};
use libloading::{Library, Symbol};
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Tool plugin trait.
///
/// A tool plugin can expose one or more callable function definitions and
/// execute calls routed to those functions.
pub trait ToolPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn functions(&self) -> Vec<FunctionDefinition>;

    fn execute(&self, call: &FunctionCall) -> Result<Value>;

    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicToolPluginOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynToolPluginPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn ToolPlugin>` into opaque ABI payload.
pub fn dynamic_tool_plugin_into_opaque(plugin: Box<dyn ToolPlugin>) -> DynamicToolPluginOpaque {
    let raw: *mut dyn ToolPlugin = Box::into_raw(plugin);
    let parts: RawDynToolPluginPtr = unsafe { std::mem::transmute(raw) };
    DynamicToolPluginOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert opaque ABI payload back into plugin object.
///
/// # Safety
/// Payload must come from `dynamic_tool_plugin_into_opaque`.
pub unsafe fn dynamic_tool_plugin_from_opaque(
    opaque: DynamicToolPluginOpaque,
) -> Option<Box<dyn ToolPlugin>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynToolPluginPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn ToolPlugin = unsafe { std::mem::transmute(parts) };
    if raw.is_null() {
        None
    } else {
        Some(unsafe { Box::from_raw(raw) })
    }
}

type ToolPluginConstructor = unsafe extern "C" fn() -> DynamicToolPluginOpaque;

#[allow(improper_ctypes_definitions)]
type LegacyToolPluginConstructor = unsafe extern "C" fn() -> *mut dyn ToolPlugin;

struct ToolPluginRuntime {
    plugin: Mutex<Box<dyn ToolPlugin>>,
    #[allow(dead_code)]
    library: Option<Arc<Library>>,
}

impl Drop for ToolPluginRuntime {
    fn drop(&mut self) {
        let _ = self.plugin.lock().cleanup();
    }
}

struct ToolPluginHandler {
    plugin_name: String,
    runtime: Arc<ToolPluginRuntime>,
}

impl FunctionHandler for ToolPluginHandler {
    fn execute(&self, call: &FunctionCall) -> Result<Value> {
        self.runtime
            .plugin
            .lock()
            .execute(call)
            .map_err(|e| match e {
                LociError::Other(msg) => LociError::Other(format!(
                    "Tool plugin '{}' failed on '{}': {}",
                    self.plugin_name, call.name, msg
                )),
                other => other,
            })
    }
}

/// Loaded tool plugin metadata and runtime handle.
pub struct LoadedToolPlugin {
    pub name: String,
    pub version: String,
    pub function_names: Vec<String>,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
    runtime: Arc<ToolPluginRuntime>,
}

impl LoadedToolPlugin {
    pub fn is_alive(&self) -> bool {
        Arc::strong_count(&self.runtime) >= 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedToolPluginDescriptor {
    pub name: String,
    pub version: String,
    pub function_names: Vec<String>,
    pub dynamic: bool,
    pub source: Option<PathBuf>,
}

impl LoadedToolPlugin {
    pub fn descriptor(&self) -> LoadedToolPluginDescriptor {
        LoadedToolPluginDescriptor {
            name: self.name.clone(),
            version: self.version.clone(),
            function_names: self.function_names.clone(),
            dynamic: self.dynamic,
            source: self.source.clone(),
        }
    }
}

fn register_tool_plugin_runtime(
    manager: &mut FunctionCallingManager,
    runtime: Arc<ToolPluginRuntime>,
    dynamic: bool,
    source: Option<PathBuf>,
) -> Result<LoadedToolPlugin> {
    let (name, version, functions) = {
        let plugin = runtime.plugin.lock();
        (
            plugin.name().to_string(),
            plugin.version().to_string(),
            plugin.functions(),
        )
    };

    if name.trim().is_empty() {
        return Err(LociError::PluginError(
            "Tool plugin returned empty name".to_string(),
        ));
    }
    if functions.is_empty() {
        return Err(LociError::PluginError(format!(
            "Tool plugin '{}' exposes no functions",
            name
        )));
    }

    let mut seen = HashSet::new();
    for func in &functions {
        if func.name.trim().is_empty() {
            return Err(LociError::PluginError(format!(
                "Tool plugin '{}' contains function with empty name",
                name
            )));
        }
        if !seen.insert(func.name.clone()) {
            return Err(LociError::PluginError(format!(
                "Tool plugin '{}' contains duplicate function '{}'",
                name, func.name
            )));
        }
        if manager.get_function(&func.name).is_some() {
            return Err(LociError::PluginError(format!(
                "Function '{}' already registered; cannot load tool plugin '{}'",
                func.name, name
            )));
        }
    }

    let mut function_names = Vec::with_capacity(functions.len());
    for function in functions {
        let function_name = function.name.clone();
        manager.register_function_with_handler(
            function,
            ToolPluginHandler {
                plugin_name: name.clone(),
                runtime: Arc::clone(&runtime),
            },
        )?;
        function_names.push(function_name);
    }

    Ok(LoadedToolPlugin {
        name,
        version,
        function_names,
        dynamic,
        source,
        runtime,
    })
}

/// Register a tool plugin instance.
pub fn register_tool_plugin(
    manager: &mut FunctionCallingManager,
    mut plugin: Box<dyn ToolPlugin>,
) -> Result<LoadedToolPlugin> {
    plugin.init()?;
    let runtime = Arc::new(ToolPluginRuntime {
        plugin: Mutex::new(plugin),
        library: None,
    });
    register_tool_plugin_runtime(manager, runtime, false, None)
}

/// Load a dynamic tool plugin from shared library and register its tools.
pub fn load_dynamic_tool_plugin<P: AsRef<Path>>(
    library_path: P,
    manager: &mut FunctionCallingManager,
) -> Result<LoadedToolPlugin> {
    let path = library_path.as_ref();
    let manifest = load_and_validate_plugin_contract(path, PluginContractKind::ToolPlugin)?;
    if !path.exists() {
        return Err(LociError::PluginError(format!(
            "Tool plugin library not found: {}",
            path.display()
        )));
    }

    let library = unsafe {
        Library::new(path).map_err(|e| {
            LociError::PluginError(format!(
                "Failed to load tool plugin library '{}': {}",
                path.display(),
                e
            ))
        })?
    };
    let library = Arc::new(library);

    let mut plugin: Box<dyn ToolPlugin> = unsafe {
        if let Ok(constructor_v1) = library.get::<ToolPluginConstructor>(b"create_tool_plugin_v1") {
            let opaque = constructor_v1();
            dynamic_tool_plugin_from_opaque(opaque).ok_or_else(|| {
                LociError::PluginError(
                    "Tool plugin constructor returned invalid payload".to_string(),
                )
            })?
        } else {
            let constructor: Symbol<LegacyToolPluginConstructor> =
                library.get(b"create_tool_plugin").map_err(|e| {
                    LociError::PluginError(format!(
                        "Failed to find tool plugin constructor symbol ('create_tool_plugin_v1' or 'create_tool_plugin'): {}",
                        e
                    ))
                })?;
            let raw = constructor();
            if raw.is_null() {
                return Err(LociError::PluginError(
                    "Tool plugin constructor returned null".to_string(),
                ));
            }
            Box::from_raw(raw)
        }
    };

    plugin.init()?;
    validate_runtime_plugin_identity(manifest.as_ref(), plugin.name(), plugin.version())?;
    let runtime = Arc::new(ToolPluginRuntime {
        plugin: Mutex::new(plugin),
        library: Some(library),
    });
    register_tool_plugin_runtime(manager, runtime, true, Some(path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct MockToolPlugin;

    impl ToolPlugin for MockToolPlugin {
        fn name(&self) -> &str {
            "mock_tool_plugin"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn functions(&self) -> Vec<FunctionDefinition> {
            vec![FunctionDefinition::new("mock_add", "Add two numbers")
                .add_parameter("a", "number", "A", true)
                .add_parameter("b", "number", "B", true)]
        }

        fn execute(&self, call: &FunctionCall) -> Result<Value> {
            let a = call
                .get_number("a")
                .ok_or_else(|| LociError::InvalidArgument("missing a".to_string()))?;
            let b = call
                .get_number("b")
                .ok_or_else(|| LociError::InvalidArgument("missing b".to_string()))?;
            Ok(json!({ "result": a + b }))
        }
    }

    #[test]
    fn opaque_roundtrip_for_tool_plugin() {
        let plugin: Box<dyn ToolPlugin> = Box::new(MockToolPlugin);
        let opaque = dynamic_tool_plugin_into_opaque(plugin);
        let restored = unsafe { dynamic_tool_plugin_from_opaque(opaque) };
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().name(), "mock_tool_plugin");
    }

    #[test]
    fn register_tool_plugin_and_execute() {
        let mut manager = FunctionCallingManager::new();
        let loaded = register_tool_plugin(&mut manager, Box::new(MockToolPlugin)).unwrap();
        assert_eq!(loaded.name, "mock_tool_plugin");
        assert_eq!(loaded.function_names, vec!["mock_add".to_string()]);

        let call = FunctionCall::new("mock_add")
            .with_argument("a", json!(2))
            .with_argument("b", json!(5));
        let out = manager.execute_function_call(&call).unwrap();
        assert_eq!(out["result"].as_f64(), Some(7.0));
    }

    #[test]
    fn load_dynamic_tool_plugin_missing_library_fails() {
        let mut manager = FunctionCallingManager::new();
        let err = match load_dynamic_tool_plugin("missing_browser_tool_plugin.dll", &mut manager) {
            Ok(_) => panic!("missing library should fail"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("not found"));
    }
}
