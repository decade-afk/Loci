/**
 * Loci Phase 3 Week 2-4: 多模型管理系统
 *
 * 核心特性：
 * 1. ModelRegistry - 全局模型注册表
 * 2. 多模型热切换（运行时切换主模型）
 * 3. LoRA 动态合并（加载/卸载/stacking）
 * 4. KV Cache 管理（切换时的缓存策略）
 * 5. 内存预算控制（多模型共存）
 *
 * 设计目标：
 * - 支持同时加载多个模型（内存允许）
 * - 零停机切换（session 级别）
 * - LoRA 热插拔（无需重启）
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use anyhow::{Result, Context, anyhow};
use uuid::Uuid;

use crate::gguf::GGUFModel;

// ==================== 类型定义 ====================

/// 模型唯一标识符
pub type ModelID = String;

/// LoRA 适配器唯一标识符
pub type LoRAID = String;

/// Session 唯一标识符
pub type SessionID = String;

// ==================== 模型元数据 ====================

/// 模型元数据
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// 模型名称
    pub name: String,

    /// 模型大小（字节）
    pub size_bytes: u64,

    /// 参数量（如 7B、13B）
    pub parameter_count: String,

    /// 量化类型（如 Q4_0、Q8_0）
    pub quantization: String,

    /// 上下文长度
    pub context_length: usize,

    /// 词表大小
    pub vocab_size: usize,

    /// 是否支持 LoRA
    pub supports_lora: bool,
}

impl ModelMetadata {
    /// 从 GGUF 模型提取元数据（简化实现）
    pub fn from_gguf(_gguf: &GGUFModel) -> Self {
        // 由于 GGUFModel 字段是私有的，我们使用占位值
        // TODO: 添加 GGUF public accessors 或使用 builder pattern
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

// ==================== LoRA 适配器 ====================

/// LoRA 适配器配置
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// LoRA 权重文件路径
    pub path: PathBuf,

    /// 缩放因子（alpha）
    pub scale: f32,

    /// 优先级（用于 stacking 排序）
    pub priority: u8,
}

/// 已加载的 LoRA 适配器
pub struct LoRAAdapter {
    /// LoRA 唯一标识符
    pub id: LoRAID,

    /// 配置
    pub config: LoRAConfig,

    /// LoRA 文件路径（存储路径而非GGUF对象）
    pub path: PathBuf,

    /// 是否已合并到基础模型
    pub is_merged: bool,

    /// 合并时间戳（用于 LRU 管理）
    pub merge_timestamp: Option<std::time::Instant>,
}

impl LoRAAdapter {
    /// 创建新的 LoRA 适配器
    pub fn new(config: LoRAConfig) -> Result<Self> {
        // 验证文件存在
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

    /// 合并 LoRA 到基础模型权重
    pub fn merge(&mut self, base_model: &mut LoadedModel) -> Result<()> {
        if self.is_merged {
            return Ok(()); // 已合并，跳过
        }

        println!("[LoRA] Merging LoRA {} (scale={}) into model {}",
                 self.id, self.config.scale, base_model.id);

        // TODO: 实现实际的权重合并逻辑
        // 这需要：
        // 1. 遍历 LoRA 的 tensor
        // 2. 找到基础模型中对应的 tensor
        // 3. 执行 W' = W + scale * (A * B) 操作

        self.is_merged = true;
        self.merge_timestamp = Some(std::time::Instant::now());

        Ok(())
    }

    /// 从基础模型卸载 LoRA
    pub fn unmerge(&mut self, base_model: &mut LoadedModel) -> Result<()> {
        if !self.is_merged {
            return Ok(()); // 未合并，跳过
        }

        println!("[LoRA] Unmerging LoRA {} from model {}", self.id, base_model.id);

        // TODO: 实现实际的权重还原逻辑
        // W = W' - scale * (A * B)

        self.is_merged = false;
        self.merge_timestamp = None;

        Ok(())
    }
}

// ==================== 已加载模型 ====================

/// 已加载的模型
pub struct LoadedModel {
    /// 模型唯一标识符
    pub id: ModelID,

    /// 模型文件路径
    pub path: PathBuf,

    /// 模型元数据
    pub metadata: ModelMetadata,

    /// GGUF 数据
    pub gguf: Arc<GGUFModel>,

    /// 已加载的 LoRA 适配器
    pub loras: Vec<Arc<RwLock<LoRAAdapter>>>,

    /// 加载时间戳
    pub load_timestamp: std::time::Instant,

    /// 最后使用时间戳（用于 LRU 淘汰）
    pub last_used: std::time::Instant,

    /// 引用计数（有多少 session 在使用）
    pub ref_count: usize,
}

impl LoadedModel {
    /// 创建新的已加载模型
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

    /// 添加 LoRA 适配器
    pub fn add_lora(&mut self, lora: Arc<RwLock<LoRAAdapter>>) -> Result<()> {
        // 检查是否已存在
        let lora_id = lora.read().unwrap().id.clone();
        if self.loras.iter().any(|l| l.read().unwrap().id == lora_id) {
            return Err(anyhow!("LoRA {} already attached", lora_id));
        }

        self.loras.push(lora);
        Ok(())
    }

    /// 移除 LoRA 适配器
    pub fn remove_lora(&mut self, lora_id: &str) -> Result<Arc<RwLock<LoRAAdapter>>> {
        let index = self.loras.iter()
            .position(|l| l.read().unwrap().id == lora_id)
            .ok_or_else(|| anyhow!("LoRA {} not found", lora_id))?;

        Ok(self.loras.remove(index))
    }

    /// 更新最后使用时间
    pub fn touch(&mut self) {
        self.last_used = std::time::Instant::now();
    }
}

// ==================== 模型注册表 ====================

/// 全局模型注册表（单例）
pub struct ModelRegistry {
    /// 已加载的模型（ModelID -> LoadedModel）
    models: RwLock<HashMap<ModelID, Arc<RwLock<LoadedModel>>>>,

    /// Session 到模型的映射（SessionID -> ModelID）
    session_models: RwLock<HashMap<SessionID, ModelID>>,

    /// 内存预算（字节）
    memory_budget: u64,

    /// 当前已使用内存（字节）
    memory_used: RwLock<u64>,
}

impl ModelRegistry {
    /// 创建新的模型注册表
    pub fn new(memory_budget_gb: u64) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            session_models: RwLock::new(HashMap::new()),
            memory_budget: memory_budget_gb * 1024 * 1024 * 1024,
            memory_used: RwLock::new(0),
        }
    }

    /// 加载模型
    ///
    /// # 参数
    /// - path: GGUF 模型文件路径
    ///
    /// # 返回值
    /// - 成功：返回模型 ID
    /// - 失败：返回错误
    pub fn load_model(&self, path: &Path) -> Result<ModelID> {
        // 生成模型 ID（使用路径的哈希）
        let model_id = format!("model_{}", path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown"));

        // 检查是否已加载
        {
            let models = self.models.read().unwrap();
            if models.contains_key(&model_id) {
                println!("[ModelRegistry] Model {} already loaded", model_id);
                return Ok(model_id);
            }
        }

        // 加载 GGUF
        println!("[ModelRegistry] Loading model from {:?}", path);
        let gguf = GGUFModel::load(path)
            .context("Failed to load GGUF")?;

        // 获取文件大小作为模型大小估算
        let model_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        // 检查内存预算
        {
            let memory_used = self.memory_used.read().unwrap();
            if *memory_used + model_size > self.memory_budget {
                return Err(anyhow!(
                    "Memory budget exceeded: {} + {} > {}",
                    *memory_used, model_size, self.memory_budget
                ));
            }
        }

        // 创建 LoadedModel
        let loaded_model = LoadedModel::new(
            model_id.clone(),
            path.to_path_buf(),
            Arc::new(gguf),
        );

        // 注册模型
        {
            let mut models = self.models.write().unwrap();
            models.insert(model_id.clone(), Arc::new(RwLock::new(loaded_model)));
        }

        // 更新内存使用
        {
            let mut memory_used = self.memory_used.write().unwrap();
            *memory_used += model_size;
        }

        println!("[ModelRegistry] Model {} loaded successfully", model_id);
        Ok(model_id)
    }

    /// 卸载模型
    ///
    /// # 参数
    /// - model_id: 模型 ID
    ///
    /// # 返回值
    /// - 成功：返回 ()
    /// - 失败：返回错误
    pub fn unload_model(&self, model_id: &str) -> Result<()> {
        let mut models = self.models.write().unwrap();

        // 检查模型是否存在
        let model_arc = models.get(model_id)
            .ok_or_else(|| anyhow!("Model {} not found", model_id))?
            .clone();

        // 检查引用计数
        {
            let model = model_arc.read().unwrap();
            if model.ref_count > 0 {
                return Err(anyhow!(
                    "Model {} is in use by {} sessions",
                    model_id, model.ref_count
                ));
            }
        }

        // 移除模型
        models.remove(model_id);

        println!("[ModelRegistry] Model {} unloaded", model_id);
        Ok(())
    }

    /// 切换 Session 的模型
    ///
    /// # 参数
    /// - session_id: Session ID
    /// - new_model_id: 新模型 ID
    ///
    /// # 返回值
    /// - 成功：返回 ()
    /// - 失败：返回错误
    pub fn switch_model(&self, session_id: &str, new_model_id: &str) -> Result<()> {
        // 检查新模型是否存在
        let models = self.models.read().unwrap();
        let new_model = models.get(new_model_id)
            .ok_or_else(|| anyhow!("Model {} not found", new_model_id))?
            .clone();

        // 获取旧模型 ID
        let old_model_id = {
            let session_models = self.session_models.read().unwrap();
            session_models.get(session_id).cloned()
        };

        // 更新旧模型的引用计数
        if let Some(old_id) = old_model_id.as_ref() {
            if let Some(old_model) = models.get(old_id) {
                let mut old_model = old_model.write().unwrap();
                old_model.ref_count -= 1;
            }
        }

        // 更新新模型的引用计数
        {
            let mut new_model = new_model.write().unwrap();
            new_model.ref_count += 1;
            new_model.touch();
        }

        // 更新映射
        {
            let mut session_models = self.session_models.write().unwrap();
            session_models.insert(session_id.to_string(), new_model_id.to_string());
        }

        println!("[ModelRegistry] Session {} switched from {:?} to {}",
                 session_id, old_model_id, new_model_id);

        Ok(())
    }

    /// 获取 Session 的当前模型
    pub fn get_model(&self, session_id: &str) -> Result<Arc<RwLock<LoadedModel>>> {
        let session_models = self.session_models.read().unwrap();
        let model_id = session_models.get(session_id)
            .ok_or_else(|| anyhow!("Session {} has no model", session_id))?;

        let models = self.models.read().unwrap();
        models.get(model_id)
            .cloned()
            .ok_or_else(|| anyhow!("Model {} not found", model_id))
    }

    /// 获取已加载的模型列表
    pub fn list_models(&self) -> Vec<ModelID> {
        let models = self.models.read().unwrap();
        models.keys().cloned().collect()
    }

    /// 获取内存使用统计
    pub fn memory_stats(&self) -> (u64, u64, f64) {
        let used = *self.memory_used.read().unwrap();
        let budget = self.memory_budget;
        let usage_percent = (used as f64 / budget as f64) * 100.0;
        (used, budget, usage_percent)
    }
}

// ==================== 全局单例 ====================

use once_cell::sync::Lazy;

/// 全局模型注册表实例
pub static MODEL_REGISTRY: Lazy<ModelRegistry> = Lazy::new(|| {
    // 默认 16GB 内存预算
    ModelRegistry::new(16)
});

// ==================== 测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_creation() {
        let registry = ModelRegistry::new(8); // 8GB
        assert_eq!(registry.memory_budget, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_memory_stats() {
        let registry = ModelRegistry::new(10); // 10GB
        let (used, budget, percent) = registry.memory_stats();
        assert_eq!(used, 0);
        assert_eq!(budget, 10 * 1024 * 1024 * 1024);
        assert_eq!(percent, 0.0);
    }
}
