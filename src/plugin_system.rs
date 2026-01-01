//! Plugin System Module
//!
//! This module provides core functionality for the Loci project.
//!























use anyhow::{bail, Context, Result};
use ed25519_dalek::{VerifyingKey, Signature, Verifier as Ed25519Verifier};
use libloading::Library;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use wasmtime::*;




pub type PluginId = String;


pub type PluginPriority = u32;




#[derive(Debug, Clone)]
    /// PluginControlFlow enumeration
pub enum PluginControlFlow {
    
    Continue,

    
    Suspend {
        reason: String,
        user_data: Option<String>,
    },

    
    Break,
}




    /// LogitsView structure
pub struct LogitsView<'a> {
    pub data: &'a mut [f32],
}


impl<'a> std::panic::UnwindSafe for LogitsView<'a> {}
impl<'a> std::panic::RefUnwindSafe for LogitsView<'a> {}

impl<'a> LogitsView<'a> {
    /// new function
    pub fn new(data: &'a mut [f32]) -> Self {
        Self { data }
    }

    /// len function
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// is_empty function
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}




#[derive(Clone)]
    /// PluginContext structure
pub struct PluginContext {
    
    pub session_id: String,

    
    pub generated_tokens: usize,

    
    pub temperature: f32,

    
    pub top_p: f32,
}

// Implementation for Default
impl Default for PluginContext {
    fn default() -> Self {
        Self {
            session_id: String::new(),
            generated_tokens: 0,
            temperature: 1.0,
            top_p: 1.0,
        }
    }
}




#[derive(Debug, Clone)]
    /// PluginMetadata structure
pub struct PluginMetadata {
    
    pub id: PluginId,

    
    pub name: String,

    
    pub version: String,

    
    pub description: String,

    
    pub author: String,

    
    pub plugin_type: PluginType,

    
    pub path: PathBuf,

    
    pub priority: PluginPriority,

    
    pub enabled: bool,

    
    pub signature_verified: bool,

    
    pub hash: String,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// PluginType enumeration
pub enum PluginType {
    
    Native,

    
    Wasm,
}




pub trait Plugin: Send + Sync + std::panic::RefUnwindSafe {
    
    fn metadata(&self) -> &PluginMetadata;

    
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }

    
    fn pre_process(&self, prompt: &mut String, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let _ = prompt;
        Ok(PluginControlFlow::Continue)
    }

    
    fn transform_logits(&self, logits: &mut LogitsView, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let _ = logits;
        Ok(PluginControlFlow::Continue)
    }

    
    fn post_process(&self, token: &mut String, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let _ = token;
        Ok(PluginControlFlow::Continue)
    }

    
    fn on_token_generated(&self, token_id: i32, token_text: &str, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let _ = (token_id, token_text);
        Ok(PluginControlFlow::Continue)
    }

    
    fn on_session_start(&self, session_id: &str) -> Result<()> {
        let _ = session_id;
        Ok(())
    }

    
    fn on_session_end(&self, session_id: &str) -> Result<()> {
        let _ = session_id;
        Ok(())
    }
}




type NativeTransformLogitsFn = unsafe extern "C" fn(*mut f32, usize) -> i32;
type NativeOnTokenGeneratedFn = unsafe extern "C" fn(i32, *const u8, usize) -> i32;
type NativeInitializeFn = unsafe extern "C" fn() -> i32;
type NativeCleanupFn = unsafe extern "C" fn() -> i32;


    /// NativePlugin structure
pub struct NativePlugin {
    metadata: PluginMetadata,
    #[allow(dead_code)]
    library: Arc<Library>,
    transform_logits_fn: Option<NativeTransformLogitsFn>,
    on_token_generated_fn: Option<NativeOnTokenGeneratedFn>,
    initialize_fn: Option<NativeInitializeFn>,
    cleanup_fn: Option<NativeCleanupFn>,
}

// Implementation for NativePlugin
impl NativePlugin {
    
    /// load function
    pub fn load(path: &Path, metadata: PluginMetadata) -> Result<Self> {
        unsafe {
            let library = Library::new(path)
                .with_context(|| format!("Failed to load Native plugin: {:?}", path))?;

            
            let transform_logits_fn: Option<NativeTransformLogitsFn> =
                library.get(b"loci_transform_logits").ok().map(|sym| *sym);

            let on_token_generated_fn: Option<NativeOnTokenGeneratedFn> =
                library.get(b"loci_on_token_generated").ok().map(|sym| *sym);

            let initialize_fn: Option<NativeInitializeFn> =
                library.get(b"loci_initialize").ok().map(|sym| *sym);

            let cleanup_fn: Option<NativeCleanupFn> =
                library.get(b"loci_cleanup").ok().map(|sym| *sym);

            Ok(Self {
                metadata,
                library: Arc::new(library),
                transform_logits_fn,
                on_token_generated_fn,
                initialize_fn,
                cleanup_fn,
            })
        }
    }
}


// Implementation for std
impl std::panic::RefUnwindSafe for NativePlugin {}

// Implementation for Plugin
impl Plugin for NativePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn initialize(&mut self) -> Result<()> {
        if let Some(init_fn) = self.initialize_fn {
            unsafe {
                let result = init_fn();
                if result != 0 {
                    bail!("Native plugin initialization failed: {}", result);
                }
            }
        }
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        if let Some(cleanup_fn) = self.cleanup_fn {
            unsafe {
                let result = cleanup_fn();
                if result != 0 {
                    log::warn!("Native plugin cleanup failed: {}", result);
                }
            }
        }
        Ok(())
    }

    fn transform_logits(&self, logits: &mut LogitsView, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        if let Some(transform_fn) = self.transform_logits_fn {
            unsafe {
                let result = transform_fn(logits.data.as_mut_ptr(), logits.len());
                if result == 0 {
                    Ok(PluginControlFlow::Continue)
                } else if result == 1 {
                    Ok(PluginControlFlow::Suspend {
                        reason: "Native plugin requested suspend".to_string(),
                        user_data: None,
                    })
                } else {
                    Ok(PluginControlFlow::Break)
                }
            }
        } else {
            Ok(PluginControlFlow::Continue)
        }
    }

    fn on_token_generated(&self, token_id: i32, token_text: &str, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        if let Some(callback_fn) = self.on_token_generated_fn {
            unsafe {
                let result = callback_fn(
                    token_id,
                    token_text.as_ptr(),
                    token_text.len(),
                );

                if result == 0 {
                    Ok(PluginControlFlow::Continue)
                } else if result == 1 {
                    Ok(PluginControlFlow::Suspend {
                        reason: "Native plugin requested suspend".to_string(),
                        user_data: None,
                    })
                } else {
                    Ok(PluginControlFlow::Break)
                }
            }
        } else {
            Ok(PluginControlFlow::Continue)
        }
    }
}




    /// WasmPlugin structure
pub struct WasmPlugin {
    metadata: PluginMetadata,
    #[allow(dead_code)]
    engine: Engine,
    module: Module,
    store: Mutex<Store<()>>,
}

// Implementation for WasmPlugin
impl WasmPlugin {
    
    /// load function
    pub fn load(path: &Path, metadata: PluginMetadata) -> Result<Self> {
        
        let mut config = Config::new();

        
        config.wasm_threads(false);  
        config.wasm_bulk_memory(true); 
        config.wasm_simd(true);  

        let engine = Engine::new(&config)?;

        
        let wasm_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read WASM plugin: {:?}", path))?;

        let module = Module::new(&engine, &wasm_bytes)
            .context("Failed to compile WASM module")?;

        let store = Store::new(&engine, ());

        Ok(Self {
            metadata,
            engine,
            module,
            store: Mutex::new(store),
        })
    }

    
    fn call_wasm_fn(&self, fn_name: &str, params: &[Val]) -> Result<Vec<Val>> {
        let mut store = self.store.lock();

        
        let instance = Instance::new(&mut *store, &self.module, &[])
            .context("Failed to instantiate WASM module")?;

        
        let func = instance
            .get_func(&mut *store, fn_name)
            .ok_or_else(|| anyhow::anyhow!("WASM function '{}' not found", fn_name))?;

        
        let mut results = vec![Val::I32(0)];
        func.call(&mut *store, params, &mut results)
            .with_context(|| format!("Failed to call WASM function '{}'", fn_name))?;

        Ok(results)
    }
}


// Implementation for std
impl std::panic::RefUnwindSafe for WasmPlugin {}

// Implementation for Plugin
impl Plugin for WasmPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn initialize(&mut self) -> Result<()> {
        
        match self.call_wasm_fn("loci_initialize", &[]) {
            Ok(results) => {
                if let Some(Val::I32(0)) = results.first() {
                    Ok(())
                } else {
                    bail!("WASM plugin initialization failed")
                }
            }
            Err(_) => Ok(()), 
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        
        match self.call_wasm_fn("loci_cleanup", &[]) {
            Ok(_) => Ok(()),
            Err(_) => Ok(()), 
        }
    }

    fn transform_logits(&self, logits: &mut LogitsView, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let mut store = self.store.lock();

        
        let instance = Instance::new(&mut *store, &self.module, &[])
            .context("Failed to instantiate WASM module")?;

        
        let memory = instance
            .get_memory(&mut *store, "memory")
            .ok_or_else(|| anyhow::anyhow!("WASM memory not found"))?;

        
        let logits_len = logits.len();
        let logits_size_bytes = logits_len * std::mem::size_of::<f32>();

        
        let logits_ptr = 0;
        unsafe {
            let data_slice = memory.data_mut(&mut *store);
            if data_slice.len() < logits_size_bytes {
                bail!("WASM memory too small for logits");
            }

            std::ptr::copy_nonoverlapping(
                logits.data.as_ptr() as *const u8,
                data_slice.as_mut_ptr(),
                logits_size_bytes,
            );
        }

        
        let func = instance
            .get_func(&mut *store, "loci_transform_logits")
            .ok_or_else(|| anyhow::anyhow!("WASM function 'loci_transform_logits' not found"))?;

        let mut results = vec![Val::I32(0)];
        func.call(
            &mut *store,
            &[Val::I32(logits_ptr), Val::I32(logits_len as i32)],
            &mut results,
        )?;

        
        let data_slice = memory.data(&mut *store);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_slice.as_ptr(),
                logits.data.as_mut_ptr() as *mut u8,
                logits_size_bytes,
            );
        }

        
        if let Some(Val::I32(ret)) = results.first() {
            match ret {
                0 => Ok(PluginControlFlow::Continue),
                1 => Ok(PluginControlFlow::Suspend {
                    reason: "WASM plugin requested suspend".to_string(),
                    user_data: None,
                }),
                _ => Ok(PluginControlFlow::Break),
            }
        } else {
            Ok(PluginControlFlow::Continue)
        }
    }

    fn on_token_generated(&self, token_id: i32, token_text: &str, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let mut store = self.store.lock();

        let instance = Instance::new(&mut *store, &self.module, &[])?;
        let memory = instance.get_memory(&mut *store, "memory").ok_or_else(|| anyhow::anyhow!("WASM memory not found"))?;

        
        let text_bytes = token_text.as_bytes();
        let text_ptr = 0;
        let data_slice = memory.data_mut(&mut *store);
        if data_slice.len() < text_bytes.len() {
            bail!("WASM memory too small for token text");
        }
        data_slice[..text_bytes.len()].copy_from_slice(text_bytes);

        
        let func = instance
            .get_func(&mut *store, "loci_on_token_generated")
            .ok_or_else(|| anyhow::anyhow!("WASM function 'loci_on_token_generated' not found"))?;

        let mut results = vec![Val::I32(0)];
        func.call(
            &mut *store,
            &[Val::I32(token_id), Val::I32(text_ptr), Val::I32(text_bytes.len() as i32)],
            &mut results,
        )?;

        if let Some(Val::I32(ret)) = results.first() {
            match ret {
                0 => Ok(PluginControlFlow::Continue),
                1 => Ok(PluginControlFlow::Suspend {
                    reason: "WASM plugin requested suspend".to_string(),
                    user_data: None,
                }),
                _ => Ok(PluginControlFlow::Break),
            }
        } else {
            Ok(PluginControlFlow::Continue)
        }
    }
}




    /// SignatureVerifier structure
pub struct SignatureVerifier {
    official_public_key: VerifyingKey,
}

// Implementation for SignatureVerifier
impl SignatureVerifier {
    
    /// new function
    pub fn new(public_key_bytes: &[u8; 32]) -> Result<Self> {
        let official_public_key = VerifyingKey::from_bytes(public_key_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid Ed25519 public key: {}", e))?;

        Ok(Self {
            official_public_key,
        })
    }

    
    /// verify_plugin function
    pub fn verify_plugin(&self, plugin_path: &Path) -> Result<()> {
        let plugin_data = std::fs::read(plugin_path)?;
        let sig_path = plugin_path.with_extension("sig");
        let sig_data = std::fs::read(&sig_path)?;

        
        if sig_data.len() != 64 {
            bail!("Invalid signature length: expected 64 bytes, got {}", sig_data.len());
        }

        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&sig_data);
        let signature = Signature::from_bytes(&sig_bytes);

        self.official_public_key
            .verify(&plugin_data, &signature)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {}", e))?;

        Ok(())
    }

    
    /// compute_hash function
    pub fn compute_hash(&self, plugin_path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};

        let data = std::fs::read(plugin_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();

        Ok(format!("{:x}", result))
    }
}



#[derive(Debug, Clone)]
    /// ResourceQuota structure
pub struct ResourceQuota {
    pub max_execution_time_ms: u64,
    #[allow(dead_code)]
    pub max_memory_bytes: usize,
    pub enabled: bool,
}

// Implementation for ResourceQuota
impl ResourceQuota {
    
    /// new function
    pub fn new(max_execution_time_ms: u64, max_memory_bytes: usize) -> Self {
        Self {
            max_execution_time_ms,
            max_memory_bytes,
            enabled: true,
        }
    }
}

// Implementation for Default
impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_execution_time_ms: 50,
            max_memory_bytes: 100 * 1024 * 1024,
            enabled: true,
        }
    }
}

    /// Watchdog structure
pub struct Watchdog {
    quota: ResourceQuota,
    timeout_count: Arc<Mutex<HashMap<PluginId, usize>>>,
}

// Implementation for Watchdog
impl Watchdog {
    /// new function
    pub fn new(quota: ResourceQuota) -> Self {
        Self {
            quota,
            timeout_count: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// execute function
    pub fn execute<F, T>(&self, plugin_id: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if !self.quota.enabled {
            return operation();
        }

        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        if elapsed.as_millis() > self.quota.max_execution_time_ms as u128 {
            let mut counts = self.timeout_count.lock();
            *counts.entry(plugin_id.to_string()).or_insert(0) += 1;

            log::warn!(
                "Plugin '{}' exceeded time limit: {}ms",
                plugin_id,
                elapsed.as_millis()
            );

            bail!("Plugin '{}' timeout", plugin_id);
        }

        result
    }
}




    /// PluginRegistry structure
pub struct PluginRegistry {
    native_plugins: Arc<RwLock<HashMap<PluginId, Arc<dyn Plugin>>>>,
    wasm_plugins: Arc<RwLock<HashMap<PluginId, Arc<dyn Plugin>>>>,
    metadata: Arc<RwLock<HashMap<PluginId, PluginMetadata>>>,
    priority_order: Arc<RwLock<Vec<PluginId>>>,
    verifier: Option<Arc<SignatureVerifier>>,
    watchdog: Arc<Watchdog>,
}

// Implementation for PluginRegistry
impl PluginRegistry {
    /// new function
    pub fn new(quota: ResourceQuota) -> Self {
        Self {
            native_plugins: Arc::new(RwLock::new(HashMap::new())),
            wasm_plugins: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            priority_order: Arc::new(RwLock::new(Vec::new())),
            verifier: None,
            watchdog: Arc::new(Watchdog::new(quota)),
        }
    }

    /// set_verifier function
    pub fn set_verifier(&mut self, verifier: SignatureVerifier) {
        self.verifier = Some(Arc::new(verifier));
    }

    /// register function
    pub fn register(&self, plugin: Arc<dyn Plugin>, metadata: PluginMetadata) -> Result<()> {
        let plugin_id = metadata.id.clone();

        
        if let Some(verifier) = &self.verifier {
            if !metadata.signature_verified {
                verifier.verify_plugin(&metadata.path)?;
            }
        }

        
        match metadata.plugin_type {
            PluginType::Native => {
                self.native_plugins.write().insert(plugin_id.clone(), plugin);
            }
            PluginType::Wasm => {
                self.wasm_plugins.write().insert(plugin_id.clone(), plugin);
            }
        }

        self.metadata.write().insert(plugin_id.clone(), metadata.clone());

        
        let mut priority_order = self.priority_order.write();
        priority_order.push(plugin_id.clone());
        priority_order.sort_by_key(|id| {
            self.metadata.read().get(id).map(|m| m.priority).unwrap_or(u32::MAX)
        });

        log::info!("Registered {} plugin '{}'",
            if metadata.plugin_type == PluginType::Native { "Native" } else { "WASM" },
            plugin_id
        );

        Ok(())
    }

    /// get_all_plugins function
    pub fn get_all_plugins(&self) -> Vec<Arc<dyn Plugin>> {
        let native = self.native_plugins.read();
        let wasm = self.wasm_plugins.read();
        let priority_order = self.priority_order.read();

        priority_order
            .iter()
            .filter_map(|id| {
                native.get(id).cloned().or_else(|| wasm.get(id).cloned())
            })
            .collect()
    }

    /// transform_logits function
    pub fn transform_logits(&self, logits: &mut LogitsView, ctx: &PluginContext) -> Result<PluginControlFlow> {
        let plugins = self.get_all_plugins();
        let metadata = self.metadata.read();

        for plugin in plugins {
            let plugin_id = plugin.metadata().id.clone();

            if let Some(meta) = metadata.get(&plugin_id) {
                if !meta.enabled {
                    continue;
                }
            }

            let control_flow = self.watchdog.execute(&plugin_id, || {
                plugin.transform_logits(logits, ctx)
            })?;

            match control_flow {
                PluginControlFlow::Continue => continue,
                _ => return Ok(control_flow),
            }
        }

        Ok(PluginControlFlow::Continue)
    }

    /// stats function
    pub fn stats(&self) -> PluginRegistryStats {
        let native_count = self.native_plugins.read().len();
        let wasm_count = self.wasm_plugins.read().len();
        let enabled_count = self.metadata.read().values().filter(|m| m.enabled).count();

        PluginRegistryStats {
            total_plugins: native_count + wasm_count,
            native_plugins: native_count,
            wasm_plugins: wasm_count,
            enabled_plugins: enabled_count,
        }
    }
}

#[derive(Debug, Clone)]
    /// PluginRegistryStats structure
pub struct PluginRegistryStats {
    pub total_plugins: usize,
    pub native_plugins: usize,
    pub wasm_plugins: usize,
    pub enabled_plugins: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_quota() {
        let quota = ResourceQuota::default();
        assert_eq!(quota.max_execution_time_ms, 50);
    }

    #[test]
    fn test_watchdog_timeout() {
        use std::time::Duration;

        let quota = ResourceQuota::new(10, 100 * 1024 * 1024);
        let watchdog = Watchdog::new(quota);

        let result = watchdog.execute("test", || {
            std::thread::sleep(Duration::from_millis(20));
            Ok(42)
        });

        assert!(result.is_err());
    }
}
