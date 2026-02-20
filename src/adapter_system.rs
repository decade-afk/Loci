//! Deep Programmable Adapter System
//!
//! This module provides a highly flexible and extensible adapter framework for model fine-tuning.

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Unique identifier for an adapter
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdapterId(pub u64);

impl AdapterId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Adapter trait - all adapters must implement this
pub trait Adapter: Send + Sync {
    fn adapter_type(&self) -> AdapterType;
    fn id(&self) -> AdapterId;
    fn apply(&self, input: &[f32]) -> Vec<f32>;
    fn merge(&self, base_weights: &[f32]) -> Vec<f32>;
    fn save(&self, path: &PathBuf) -> Result<()>;
    fn config_json(&self) -> String;
    fn memory_footprint(&self) -> usize;
}

/// Adapter type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdapterType {
    LoRA,
    QLoRA,
    AdapterFusion,
    PrefixTuning,
    PTuningV2,
    IA3,
    Custom(String),
}

/// Quantization type for QLoRA
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizationType {
    NF4,
    INT4,
    INT8,
}

/// Fusion strategy for combining multiple adapters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionStrategy {
    Weighted,
    Attention,
    Sequential,
    Gated,
}

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAAdapterConfig {
    pub path: String,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub use_bias: bool,
}

impl Default for LoRAAdapterConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            rank: 32,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            use_bias: false,
        }
    }
}

/// QLoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QLoRAAdapterConfig {
    pub path: String,
    pub rank: usize,
    pub alpha: f32,
    pub quantization: QuantizationType,
    pub double_quantization: bool,
    pub compute_dtype: String,
    pub target_modules: Vec<String>,
}

impl Default for QLoRAAdapterConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            rank: 16,
            alpha: 16.0,
            quantization: QuantizationType::NF4,
            double_quantization: true,
            compute_dtype: "float16".to_string(),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

/// Adapter fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterFusionConfig {
    pub adapter_ids: Vec<AdapterId>,
    pub fusion_weights: Vec<f32>,
    pub strategy: FusionStrategy,
    pub name: String,
}

/// Simple LoRA adapter implementation
pub struct SimpleLoRAAdapter {
    id: AdapterId,
    config: LoRAAdapterConfig,
}

impl SimpleLoRAAdapter {
    pub fn new(id: AdapterId, config: LoRAAdapterConfig) -> Self {
        Self { id, config }
    }
}

impl Adapter for SimpleLoRAAdapter {
    fn adapter_type(&self) -> AdapterType {
        AdapterType::LoRA
    }

    fn id(&self) -> AdapterId {
        self.id
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        // Simplified LoRA application
        let scaling = self.config.alpha / self.config.rank as f32;
        let mut output = input.to_vec();
        
        // Placeholder: add small delta
        for val in &mut output {
            *val += scaling * 0.01;
        }
        
        output
    }

    fn merge(&self, base_weights: &[f32]) -> Vec<f32> {
        // Simplified merge
        let scaling = self.config.alpha / self.config.rank as f32;
        let mut merged = base_weights.to_vec();
        
        for val in &mut merged {
            *val += scaling * 0.01;
        }
        
        merged
    }

    fn save(&self, path: &PathBuf) -> Result<()> {
        std::fs::write(path, serde_json::to_string(&self.config)?)?;
        Ok(())
    }

    fn config_json(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_default()
    }

    fn memory_footprint(&self) -> usize {
        self.config.rank * 4096 * 2 * std::mem::size_of::<f32>() // Simplified calculation
    }
}

/// Simple QLoRA adapter implementation
pub struct SimpleQLoRAAdapter {
    id: AdapterId,
    config: QLoRAAdapterConfig,
}

impl SimpleQLoRAAdapter {
    pub fn new(id: AdapterId, config: QLoRAAdapterConfig) -> Self {
        Self { id, config }
    }
}

impl Adapter for SimpleQLoRAAdapter {
    fn adapter_type(&self) -> AdapterType {
        AdapterType::QLoRA
    }

    fn id(&self) -> AdapterId {
        self.id
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        // Simplified QLoRA application
        let scaling = self.config.alpha / self.config.rank as f32;
        let mut output = input.to_vec();
        
        for val in &mut output {
            *val += scaling * 0.005; // Smaller delta for quantized
        }
        
        output
    }

    fn merge(&self, base_weights: &[f32]) -> Vec<f32> {
        let scaling = self.config.alpha / self.config.rank as f32;
        let mut merged = base_weights.to_vec();
        
        for val in &mut merged {
            *val += scaling * 0.005;
        }
        
        merged
    }

    fn save(&self, path: &PathBuf) -> Result<()> {
        std::fs::write(path, serde_json::to_string(&self.config)?)?;
        Ok(())
    }

    fn config_json(&self) -> String {
        serde_json::to_string(&self.config).unwrap_or_default()
    }

    fn memory_footprint(&self) -> usize {
        // Quantized weights use less memory
        self.config.rank * 4096 * std::mem::size_of::<u8>() / 2 // 4-bit quantization
    }
}

/// Adapter registry for managing multiple adapters
pub struct AdapterRegistry {
    adapters: HashMap<AdapterId, Box<dyn Adapter>>,
    next_id: u64,
}

impl AdapterRegistry {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            next_id: 1,
        }
    }

    fn next_id(&mut self) -> AdapterId {
        let id = AdapterId::new(self.next_id);
        self.next_id += 1;
        id
    }

    pub fn register_lora(&mut self, config: LoRAAdapterConfig) -> Result<AdapterId> {
        let id = self.next_id();
        let adapter = SimpleLoRAAdapter::new(id, config);
        self.adapters.insert(id, Box::new(adapter));
        Ok(id)
    }

    pub fn register_qlora(&mut self, config: QLoRAAdapterConfig) -> Result<AdapterId> {
        let id = self.next_id();
        let adapter = SimpleQLoRAAdapter::new(id, config);
        self.adapters.insert(id, Box::new(adapter));
        Ok(id)
    }

    pub fn get_adapter(&self, id: AdapterId) -> Option<&dyn Adapter> {
        self.adapters.get(&id).map(|a| a.as_ref())
    }

    pub fn remove_adapter(&mut self, id: AdapterId) -> Option<Box<dyn Adapter>> {
        self.adapters.remove(&id)
    }

    pub fn list_adapters(&self) -> Vec<AdapterId> {
        self.adapters.keys().copied().collect()
    }

    pub fn apply_adapter(&self, id: AdapterId, input: &[f32]) -> Result<Vec<f32>> {
        match self.adapters.get(&id) {
            Some(adapter) => Ok(adapter.apply(input)),
            None => Err(LociError::InvalidArgument(format!("Adapter {} not found", id.0))),
        }
    }

    pub fn merge_adapter(&self, id: AdapterId, base_weights: &[f32]) -> Result<Vec<f32>> {
        match self.adapters.get(&id) {
            Some(adapter) => Ok(adapter.merge(base_weights)),
            None => Err(LociError::InvalidArgument(format!("Adapter {} not found", id.0))),
        }
    }

    pub fn get_total_memory_footprint(&self) -> usize {
        self.adapters.values().map(|a| a.memory_footprint()).sum()
    }

    pub fn save_adapter(&self, id: AdapterId, path: &PathBuf) -> Result<()> {
        match self.adapters.get(&id) {
            Some(adapter) => adapter.save(path),
            None => Err(LociError::InvalidArgument(format!("Adapter {} not found", id.0))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_registry() {
        let mut registry = AdapterRegistry::new();
        
        let lora_config = LoRAAdapterConfig::default();
        let lora_id = registry.register_lora(lora_config).unwrap();
        
        assert!(registry.get_adapter(lora_id).is_some());
        assert_eq!(registry.list_adapters().len(), 1);
    }

    #[test]
    fn test_lora_adapter() {
        let config = LoRAAdapterConfig::default();
        let adapter = SimpleLoRAAdapter::new(AdapterId::new(1), config);
        
        let input = vec![1.0; 128];
        let output = adapter.apply(&input);
        
        assert_eq!(output.len(), 128);
        assert_eq!(adapter.adapter_type(), AdapterType::LoRA);
    }

    #[test]
    fn test_qlora_adapter() {
        let config = QLoRAAdapterConfig::default();
        let adapter = SimpleQLoRAAdapter::new(AdapterId::new(2), config);
        
        let input = vec![1.0; 128];
        let output = adapter.apply(&input);
        
        assert_eq!(output.len(), 128);
        assert_eq!(adapter.adapter_type(), AdapterType::QLoRA);
    }
}