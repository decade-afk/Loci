//! WASM Plugin System (Phase 3)
//!
//! This module provides sandboxed plugin execution using WebAssembly:
//! - **Security**: WASM sandbox isolates plugin code
//! - **Cross-platform**: WASM runs on any platform
//! - **Performance**: Near-native speed with AOT compilation
//! - **Resource limits**: CPU, memory, and time constraints
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  Loci Host (Rust)                                   │
//! │  ┌────────────────────────────────────────────┐    │
//! │  │  WASM Plugin Interface (ABI)               │    │
//! │  │  - transform_logits()                      │    │
//! │  │  - post_sample()                           │    │
//! │  │  - pre_generate()                          │    │
//! │  └────────────────────────────────────────────┘    │
//! │         ↕ (wasmtime FFI)                            │
//! │  ┌────────────────────────────────────────────┐    │
//! │  │  WASM Runtime (wasmtime)                   │    │
//! │  │  ┌──────────────────────────────────────┐  │    │
//! │  │  │  User Plugin (WASM)                  │  │    │
//! │  │  │  - Sandboxed                         │  │    │
//! │  │  │  - Resource limited                  │  │    │
//! │  │  └──────────────────────────────────────┘  │    │
//! │  └────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use loci::wasm_plugin::*;
//!
//! // Load WASM plugin
//! let config = WasmPluginConfig {
//!     name: "my_plugin".to_string(),
//!     wasm_path: "plugin.wasm".into(),
//!     max_memory: 10 * 1024 * 1024, // 10 MB
//!     max_fuel: 1_000_000,            // CPU limit
//!     ..Default::default()
//! };
//!
//! let plugin = WasmPlugin::load(config)?;
//!
//! // Use plugin (automatically sandboxed)
//! let result = plugin.transform_logits(&mut logits, &context)?;
//! ```

use crate::error::{LociError, Result};
use crate::plugin::Plugin;
use crate::sampler::LogitsView;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use wasmtime::*;
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

/// WASM plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPluginConfig {
    /// Plugin name (must be unique)
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Path to WASM binary
    pub wasm_path: PathBuf,
    /// Maximum memory in bytes (default: 16 MB)
    pub max_memory: usize,
    /// Maximum fuel (CPU instructions, 0 = unlimited)
    pub max_fuel: u64,
    /// Enable WASI (file I/O, env vars, etc.)
    pub enable_wasi: bool,
    /// Explicitly allow privileged WASI mode (disabled by default)
    pub allow_unsafe_wasi: bool,
    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,
}

impl Default for WasmPluginConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".to_string(),
            wasm_path: PathBuf::new(),
            max_memory: 16 * 1024 * 1024, // 16 MB
            max_fuel: 1_000_000,           // 1M instructions
            enable_wasi: false,            // Disabled by default for security
            allow_unsafe_wasi: false,      // Explicit opt-in for privileged mode
            timeout_ms: 5000,              // 5 seconds
        }
    }
}

/// WASM plugin runtime state
struct WasmRuntime {
    engine: Engine,
    module: Module,
    store: Store<WasiCtx>,
    instance: Instance,
}

/// WASM plugin wrapper
///
/// Implements the Plugin trait by delegating to sandboxed WASM code.
pub struct WasmPlugin {
    config: WasmPluginConfig,
    runtime: Arc<parking_lot::Mutex<WasmRuntime>>,
}

impl WasmPlugin {
    /// Get plugin configuration
    pub fn config(&self) -> &WasmPluginConfig {
        &self.config
    }

    /// Load a WASM plugin from file
    ///
    /// # Errors
    /// - File not found
    /// - Invalid WASM module
    /// - Missing required exports
    pub fn load(config: WasmPluginConfig) -> Result<Self> {
        Self::validate_sandbox_config(&config)?;

        // 1. Create WASM engine with configuration
        let mut engine_config = Config::new();
        engine_config.consume_fuel(config.max_fuel > 0);
        engine_config.epoch_interruption(config.timeout_ms > 0);
        engine_config.static_memory_maximum_size(config.max_memory as u64);
        engine_config.max_wasm_stack(1024 * 1024); // 1 MB stack

        let engine = Engine::new(&engine_config)
            .map_err(|e| LociError::PluginError(format!("Failed to create WASM engine: {}", e)))?;

        // 2. Load WASM module
        let module = Module::from_file(&engine, &config.wasm_path)
            .map_err(|e| LociError::PluginError(format!("Failed to load WASM module: {}", e)))?;

        // 3. Create WASI context
        let wasi = if config.enable_wasi {
            WasiCtxBuilder::new().build()
        } else {
            WasiCtxBuilder::new().build()
        };

        // 4. Create store with resource limits
        let mut store = Store::new(&engine, wasi);
        if config.max_fuel > 0 {
            store.set_fuel(config.max_fuel)
                .map_err(|e| LociError::PluginError(format!("Failed to set fuel: {}", e)))?;
        }

        // 5. Instantiate module
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| LociError::PluginError(format!("Failed to instantiate WASM module: {}", e)))?;

        // 6. Verify required exports
        Self::verify_exports(&instance, &mut store)?;

        let runtime = WasmRuntime {
            engine,
            module,
            store,
            instance,
        };

        Ok(Self {
            config,
            runtime: Arc::new(parking_lot::Mutex::new(runtime)),
        })
    }

    fn validate_sandbox_config(config: &WasmPluginConfig) -> Result<()> {
        if config.wasm_path.extension().and_then(|e| e.to_str()) != Some("wasm") {
            return Err(LociError::PluginError(
                "WASM plugin path must use .wasm extension".to_string(),
            ));
        }

        if config.max_memory < 1024 * 1024 {
            return Err(LociError::PluginError(
                "max_memory must be at least 1MB".to_string(),
            ));
        }

        if config.enable_wasi && !config.allow_unsafe_wasi {
            return Err(LociError::PluginError(
                "WASI is blocked by sandbox policy unless allow_unsafe_wasi is true".to_string(),
            ));
        }

        Ok(())
    }

    /// Verify that the WASM module exports required functions
    fn verify_exports(instance: &Instance, store: &mut Store<WasiCtx>) -> Result<()> {
        // Check for plugin metadata exports
        let _name = instance
            .get_typed_func::<(), i32>(&mut *store, "plugin_name")
            .map_err(|_| LociError::PluginError("Missing required export: plugin_name".to_string()))?;

        let _version = instance
            .get_typed_func::<(), i32>(&mut *store, "plugin_version")
            .map_err(|_| LociError::PluginError("Missing required export: plugin_version".to_string()))?;

        // Optional: Hook exports (at least one should exist)
        let has_transform = instance.get_typed_func::<(i32, i32, i32), i32>(&mut *store, "transform_logits").is_ok();
        let has_post_sample = instance.get_typed_func::<i32, i32>(&mut *store, "post_sample").is_ok();
        let has_pre_generate = instance.get_typed_func::<i32, i32>(&mut *store, "pre_generate").is_ok();

        if !has_transform && !has_post_sample && !has_pre_generate {
            return Err(LociError::PluginError(
                "Plugin must export at least one hook function".to_string()
            ));
        }

        Ok(())
    }

    /// Call a WASM function with timeout
    fn call_with_timeout<T, R>(
        &self,
        func_name: &str,
        args: T,
        timeout_ms: u64,
    ) -> Result<R>
    where
        T: WasmParams,
        R: WasmResults,
    {
        let mut runtime = self.runtime.lock();

        // Set fuel if configured
        if self.config.max_fuel > 0 {
            runtime.store.set_fuel(self.config.max_fuel)
                .map_err(|e| LociError::PluginError(format!("Fuel error: {}", e)))?;
        }

        if timeout_ms > 0 {
            runtime.store.set_epoch_deadline(1);
            let engine = runtime.engine.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(timeout_ms));
                engine.increment_epoch();
            });
        }

        // Split borrows
        let WasmRuntime { store, instance, .. } = &mut *runtime;

        // Get typed function
        let func = instance
            .get_typed_func::<T, R>(&mut *store, func_name)
            .map_err(|_| LociError::PluginError(format!("Function {} not found", func_name)))?;

        let result = func
            .call(&mut *store, args)
            .map_err(|e| {
                let message = e.to_string();
                if message.contains("interrupt")
                    || message.contains("epoch")
                    || message.contains("deadline")
                {
                    LociError::Timeout(format!("WASM call timed out in {func_name}"))
                } else {
                    LociError::PluginError(format!("WASM call failed: {}", message))
                }
            })?;

        Ok(result)
    }

    /// Read string from WASM memory
    fn read_string(&self, ptr: i32, len: i32) -> Result<String> {
        let mut runtime = self.runtime.lock();
        let runtime = &mut *runtime;

        let memory = runtime.instance
            .get_memory(&mut runtime.store, "memory")
            .ok_or_else(|| LociError::PluginError("WASM memory not found".to_string()))?;

        let data = memory.data(&runtime.store);
        let start = ptr as usize;
        let end = start + len as usize;

        if end > data.len() {
            return Err(LociError::PluginError("WASM memory access out of bounds".to_string()));
        }

        String::from_utf8(data[start..end].to_vec())
            .map_err(|e| LociError::PluginError(format!("Invalid UTF-8: {}", e)))
    }

    /// Write string to WASM memory
    fn write_string(&self, s: &str) -> Result<(i32, i32)> {
        let mut runtime = self.runtime.lock();
        let runtime = &mut *runtime;
        let len = s.len() as i32;

        // Allocate memory in WASM (call alloc function)
        let alloc = runtime.instance
            .get_typed_func::<i32, i32>(&mut runtime.store, "alloc")
            .map_err(|_| LociError::PluginError("WASM alloc function not found".to_string()))?;

        let ptr = alloc
            .call(&mut runtime.store, len)
            .map_err(|e| LociError::PluginError(format!("WASM alloc failed: {}", e)))?;

        // Get memory and write data to allocated memory
        let memory = runtime.instance
            .get_memory(&mut runtime.store, "memory")
            .ok_or_else(|| LociError::PluginError("WASM memory not found".to_string()))?;

        let data = memory.data_mut(&mut runtime.store);
        let start = ptr as usize;
        let end = start + s.len();

        if end > data.len() {
            return Err(LociError::PluginError("WASM memory write out of bounds".to_string()));
        }

        data[start..end].copy_from_slice(s.as_bytes());

        Ok((ptr, len))
    }
}

impl Plugin for WasmPlugin {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn version(&self) -> &str {
        &self.config.version
    }

    fn init(&mut self) -> Result<()> {
        // WASM module is already initialized in load()
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        // For now, pre_generate hook is not fully implemented for WASM
        // The WASM plugin would need to return a modified prompt string
        // This requires a more complex ABI with string return values

        // Check if the hook exists
        let mut runtime = self.runtime.lock();
        let runtime = &mut *runtime;

        if runtime.instance.get_typed_func::<i32, i32>(&mut runtime.store, "pre_generate").is_ok() {
            // Hook exists but we can't call it properly yet
            // Just return the original prompt for now
            Ok(prompt.to_string())
        } else {
            // No hook, return original prompt
            Ok(prompt.to_string())
        }
    }

    fn transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
        let n_vocab = logits.vocab_size() as i32;
        let context_len = context.len() as i32;

        // For now, just call the hook (full impl would pass actual data)
        let _result: i32 = self.call_with_timeout(
            "transform_logits",
            (0, n_vocab, context_len), // Placeholder
            self.config.timeout_ms,
        )?;

        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        self.call_with_timeout(
            "post_sample",
            token_id,
            self.config.timeout_ms,
        )
    }

    fn cleanup(&mut self) -> Result<()> {
        // WASM runtime cleanup is automatic via Drop
        Ok(())
    }
}

/// WASM plugin manager (extends PluginRegistry)
pub struct WasmPluginManager {
    plugins: Vec<WasmPlugin>,
}

impl WasmPluginManager {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Load a WASM plugin from configuration
    pub fn load(&mut self, config: WasmPluginConfig) -> Result<()> {
        let plugin = WasmPlugin::load(config)?;
        self.plugins.push(plugin);
        Ok(())
    }

    /// Get loaded plugins
    pub fn plugins(&self) -> &[WasmPlugin] {
        &self.plugins
    }
}

impl Default for WasmPluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_defaults() {
        let config = WasmPluginConfig::default();
        assert_eq!(config.max_memory, 16 * 1024 * 1024);
        assert_eq!(config.max_fuel, 1_000_000);
        assert!(!config.enable_wasi);
        assert!(!config.allow_unsafe_wasi);
    }

    #[test]
    fn test_wasm_plugin_manager() {
        let manager = WasmPluginManager::new();
        assert_eq!(manager.plugins().len(), 0);
    }

    #[test]
    fn test_wasi_policy_requires_explicit_override() {
        let config = WasmPluginConfig {
            enable_wasi: true,
            allow_unsafe_wasi: false,
            wasm_path: "plugin.wasm".into(),
            ..Default::default()
        };

        let result = WasmPlugin::validate_sandbox_config(&config);
        assert!(result.is_err());
    }
}
