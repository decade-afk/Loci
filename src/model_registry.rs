//! Model Registry Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context, anyhow};
use uuid::Uuid;

use crate::gguf::GGUFModel;




pub type ModelID = String;


pub type LoRAID = String;


pub type SessionID = String;




#[derive(Debug, Clone)]
    /// ModelMetadata structure
pub struct ModelMetadata {
    
    pub name: String,

    
    pub size_bytes: u64,

    
    pub parameter_count: String,

    
    pub quantization: String,

    
    pub context_length: usize,

    
    pub vocab_size: usize,

    
    pub supports_lora: bool,
}

// Implementation for ModelMetadata
impl ModelMetadata {
    
    /// from_gguf function
    pub fn from_gguf(_gguf: &GGUFModel) -> Self {
        
        
        Self {
            name: "model".to_string(),
            size_bytes: 0,
            parameter_count: "unknown".to_string(),
            quantization: "Q4_0".to_string(),
            context_length: 2048,
            vocab_size: 32000,
            supports_lora: true,
        }
    }
}




#[derive(Debug, Clone)]
    /// LoRAConfig structure
pub struct LoRAConfig {
    
    pub path: PathBuf,

    
    pub scale: f32,

    
    pub priority: u8,
}


    /// LoRAAdapter structure
pub struct LoRAAdapter {
    
    pub id: LoRAID,

    
    pub config: LoRAConfig,

    
    pub path: PathBuf,

    
    pub is_merged: bool,

    
    pub merge_timestamp: Option<std::time::Instant>,
}

// Implementation for LoRAAdapter
impl LoRAAdapter {
    
    /// new function
    pub fn new(config: LoRAConfig) -> Result<Self> {
        
        if !config.path.exists() {
            return Err(anyhow!("LoRA file not found: {:?}", config.path));
        }

        Ok(Self {
            id: Uuid::new_v4().to_string(),
            path: config.path.clone(),
            config,
            is_merged: false,
            merge_timestamp: None,
        })
    }

    
    /// merge function
    pub fn merge(&mut self, base_model: &mut LoadedModel) -> Result<()> {
        if self.is_merged {
            return Ok(()); // 已经合并，无需重复操作
        }

        println!("[LoRA] Merging LoRA {} (scale={}) into model {}",
                 self.id, self.config.scale, base_model.id);

        // 注意：由于 GGUF 模型通过 mmap 加载，权重是只读的
        // 实际的权重合并需要在推理时动态应用
        // 这里我们标记 LoRA 为已合并，推理引擎会在前向传播时应用 LoRA 权重

        // 验证 LoRA 文件存在
        if !self.path.exists() {
            bail!("LoRA file not found: {:?}", self.path);
        }

        // Validate scale factor
        if self.config.scale <= 0.0 {
            bail!("Invalid LoRA scale: {}", self.config.scale);
        }

        // TODO: Actual implementation should:
        // 1. Load LoRA weight file (Safetensors or GGUF format)
        // 2. Verify LoRA weights compatibility with base model weights
        // 3. If supported, merge LoRA weights into base model
        // 4. Save original weights for unmerge

        // Current implementation: mark as merged, apply dynamically during inference
        // This is a lazy merge strategy to avoid modifying original model files

        self.is_merged = true;
        self.merge_timestamp = Some(std::time::Instant::now());

        println!("[LoRA] ✅ LoRA {} marked as merged", self.id);
        Ok(())
    }

    
    /// unmerge function
    pub fn unmerge(&mut self, base_model: &mut LoadedModel) -> Result<()> {
        if !self.is_merged {
            return Ok(()); // 未合并，无需解合并
        }

        println!("[LoRA] Unmerging LoRA {} from model {}", self.id, base_model.id);

        // Note: Since we use lazy merge strategy (apply dynamically during inference),
        // unmerge only needs to mark LoRA as unmerged
        // The inference engine will no longer apply this LoRA's weights in subsequent inference

        // TODO: Actual implementation should:
        // 1. If original weights were saved, restore original weights
        // 2. If in-place merge was used, subtract LoRA delta from weights
        // 3. Verify weight recovery correctness

        // Current implementation: mark as unmerged, no longer apply LoRA during inference
        // This is a lazy unmerge strategy to avoid modifying original model files

        self.is_merged = false;
        self.merge_timestamp = None;

        println!("[LoRA] ✅ LoRA {} unmerged", self.id);
        Ok(())
    }
}




    /// LoadedModel structure
pub struct LoadedModel {
    
    pub id: ModelID,

    
    pub path: PathBuf,

    
    pub metadata: ModelMetadata,

    
    pub gguf: Arc<GGUFModel>,

    
    pub loras: Vec<Arc<RwLock<LoRAAdapter>>>,

    
    pub load_timestamp: std::time::Instant,

    
    pub last_used: std::time::Instant,

    
    pub ref_count: usize,
}

// Implementation for LoadedModel
impl LoadedModel {
    
    /// new function
    pub fn new(id: ModelID, path: PathBuf, gguf: Arc<GGUFModel>) -> Self {
        let metadata = ModelMetadata::from_gguf(&gguf);
        let now = std::time::Instant::now();

        Self {
            id,
            path,
            metadata,
            gguf,
            loras: Vec::new(),
            load_timestamp: now,
            last_used: now,
            ref_count: 0,
        }
    }

    
    /// add_lora function
    pub fn add_lora(&mut self, lora: Arc<RwLock<LoRAAdapter>>) -> Result<()> {
        // Safely read LoRA ID
        let lora_id = lora.read()
            .map_err(|e| anyhow!("Failed to read LoRA lock: {}", e))?
            .id.clone();

        // Check if already exists
        let exists = self.loras.iter()
            .any(|l| {
                l.read()
                    .map_err(|e| anyhow!("Failed to read LoRA lock: {}", e))
                    .map(|lora| lora.id == lora_id)
                    .unwrap_or(false)
            });

        if exists {
            return Err(anyhow!("LoRA {} already attached", lora_id));
        }

        self.loras.push(lora);
        Ok(())
    }

    
    /// remove_lora function
    pub fn remove_lora(&mut self, lora_id: &str) -> Result<Arc<RwLock<LoRAAdapter>>> {
        let index = self.loras.iter()
            .position(|l| {
                l.read()
                    .map_err(|e| anyhow!("Failed to read LoRA lock: {}", e))
                    .map(|lora| lora.id == lora_id)
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow!("LoRA {} not found", lora_id))?;

        Ok(self.loras.remove(index))
    }

    
    /// touch function
    pub fn touch(&mut self) {
        self.last_used = std::time::Instant::now();
    }
}




    /// ModelRegistry structure
pub struct ModelRegistry {
    
    models: RwLock<HashMap<ModelID, Arc<RwLock<LoadedModel>>>>,

    
    session_models: RwLock<HashMap<SessionID, ModelID>>,

    
    memory_budget: u64,

    
    memory_used: RwLock<u64>,
}

// Implementation for ModelRegistry
impl ModelRegistry {
    
    /// new function
    pub fn new(memory_budget_gb: u64) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            session_models: RwLock::new(HashMap::new()),
            memory_budget: memory_budget_gb * 1024 * 1024 * 1024,
            memory_used: RwLock::new(0),
        }
    }

    
    
    
    
    
    
    
    
    /// load_model function
    pub fn load_model(&self, path: &Path) -> Result<ModelID> {
        // Generate model ID
        let model_id = format!("model_{}", path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown"));

        // Check if model is already loaded
        {
            let models = self.models.read()
                .map_err(|e| anyhow!("Failed to acquire models read lock: {}", e))?;
            if models.contains_key(&model_id) {
                println!("[ModelRegistry] Model {} already loaded", model_id);
                return Ok(model_id);
            }
        }

        // Load GGUF model
        println!("[ModelRegistry] Loading model from {:?}", path);
        let gguf = GGUFModel::load(path)
            .context("Failed to load GGUF")?;

        // Get model size
        let model_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        // Check memory budget
        {
            let memory_used = self.memory_used.read()
                .map_err(|e| anyhow!("Failed to acquire memory_used read lock: {}", e))?;
            if *memory_used + model_size > self.memory_budget {
                return Err(anyhow!(
                    "Memory budget exceeded: {} + {} > {}",
                    *memory_used, model_size, self.memory_budget
                ));
            }
        }

        // Create loaded model
        let loaded_model = LoadedModel::new(
            model_id.clone(),
            path.to_path_buf(),
            Arc::new(gguf),
        );

        // Insert model into registry
        {
            let mut models = self.models.write()
                .map_err(|e| anyhow!("Failed to acquire models write lock: {}", e))?;
            models.insert(model_id.clone(), Arc::new(RwLock::new(loaded_model)));
        }

        // Update memory usage
        {
            let mut memory_used = self.memory_used.write()
                .map_err(|e| anyhow!("Failed to acquire memory_used write lock: {}", e))?;
            *memory_used += model_size;
        }

        println!("[ModelRegistry] Model {} loaded successfully", model_id);
        Ok(model_id)
    }

    
    
    
    
    
    
    
    
    /// unload_model function
    pub fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write()
            .map_err(|e| anyhow!("Failed to acquire models write lock: {}", e))?;

        // Get model
        let model_arc = models.get(model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))?
            .clone();

        // Check reference count
        {
            let model = model_arc.read()
                .map_err(|e| anyhow!("Failed to acquire model read lock: {}", e))?;
            if model.ref_count > 0 {
                return Err(anyhow!(
                    "Model {} is in use by {} sessions",
                    model_id, model.ref_count
                ));
            }
        }

        // Remove model
        models.remove(model_id);

        println!("[ModelRegistry] Model {} unloaded", model_id);
        Ok(())
    }

    /// switch_model function
    pub fn switch_model(&self, session_id: &str, new_model_id: &str) -> Result<()> {
        // Step 1: Clone new model (briefly hold models read lock)
        let new_model = {
            let models = self.models.read()
                .map_err(|e| anyhow!("Failed to acquire models read lock: {}", e))?;
            models.get(new_model_id)
                .ok_or_else(|| anyhow!("Model {} not found", new_model_id))?
                .clone()
        };

        // Step 2: Get old model ID (briefly hold session_models read lock)
        let old_model_id = {
            let session_models = self.session_models.read()
                .map_err(|e| anyhow!("Failed to acquire session_models read lock: {}", e))?;
            session_models.get(session_id).cloned()
        };

        // Step 3: Decrement old model reference count (if exists)
        // Note: Need to re-acquire models read lock since we released the previous lock
        if let Some(ref old_id) = old_model_id {
            let models = self.models.read()
                .map_err(|e| anyhow!("Failed to acquire models read lock: {}", e))?;
            if let Some(old_model) = models.get(old_id) {
                let mut old_model = old_model.write()
                    .map_err(|e| anyhow!("Failed to acquire model write lock: {}", e))?;
                old_model.ref_count -= 1;
            }
        }

        // Step 4: Increment new model reference count
        {
            let mut new_model = new_model.write()
                .map_err(|e| anyhow!("Failed to acquire model write lock: {}", e))?;
            new_model.ref_count += 1;
            new_model.touch();
        }

        // Step 5: Update session model mapping (briefly hold session_models write lock)
        {
            let mut session_models = self.session_models.write()
                .map_err(|e| anyhow!("Failed to acquire session_models write lock: {}", e))?;
            session_models.insert(session_id.to_string(), new_model_id.to_string());
        }

        println!("[ModelRegistry] Session {} switched from {:?} to {}",
                 session_id, old_model_id, new_model_id);

        Ok(())
    }

    
    /// get_model function
    pub fn get_model(&self, session_id: &str) -> Result<Arc<RwLock<LoadedModel>>> {
        let session_models = self.session_models.read()
            .map_err(|e| anyhow!("Failed to acquire session_models read lock: {}", e))?;
        let model_id = session_models.get(session_id)
            .ok_or_else(|| anyhow!("Session {} has no model", session_id))?;

        let models = self.models.read()
            .map_err(|e| anyhow!("Failed to acquire models read lock: {}", e))?;
        models.get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Model {} not found", model_id))
    }

    
    /// list_models function
    pub fn list_models(&self) -> Vec<ModelID> {
        let models = self.models.read()
            .expect("Failed to acquire models read lock - lock may be poisoned");
        models.keys().cloned().collect()
    }


    /// memory_stats function
    pub fn memory_stats(&self) -> (u64, u64, f64) {
        let used = *self.memory_used.read()
            .expect("Failed to acquire memory_used read lock - lock may be poisoned");
        let budget = self.memory_budget;
        let usage_percent = (used as f64 / budget as f64) * 100.0;
        (used, budget, usage_percent)
    }
}



use once_cell::sync::Lazy;


pub static MODEL_REGISTRY: Lazy<ModelRegistry> = Lazy::new(|| {
    
    ModelRegistry::new(16)
});



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new(8); 
        assert_eq!(registry.memory_budget, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_stats() {
        let registry = ModelRegistry::new(10); 
        let (used, budget, percent) = registry.memory_stats();
        assert_eq!(used, 0);
        assert_eq!(budget, 10 * 1024 * 1024 * 1024);
        assert_eq!(percent, 0.0);
    }
}
