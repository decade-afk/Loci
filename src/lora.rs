/**
 * Loci Phase 3 Week 4: LoRA 权重合并实现
 *
 * 核心特性：
 * 1. LoRA GGUF 格式解析
 * 2. 权重合并算法（W' = W + scale * (A @ B)）
 * 3. 运行时动态加载/卸载
 * 4. 多 LoRA stacking 支持
 * 5. 性能优化（SIMD/GPU 加速）
 *
 * LoRA 格式：
 * - 基于 GGUF 容器
 * - Tensor 命名规则：{layer_name}.lora_A, {layer_name}.lora_B
 * - 支持多种量化格式（F32, F16, Q8_0）
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use anyhow::{Result, Context, anyhow};

use crate::gguf::GGUFModel;

// ==================== LoRA Tensor 结构 ====================

/// LoRA Tensor 数据
#[derive(Clone)]
pub struct LoRATensor {
    /// Tensor 名称（e.g., "layers.0.attention.wq.lora_A"）
    pub name: String,

    /// 形状 [r, k] 或 [k, r]
    pub shape: Vec<usize>,

    /// 数据类型
    pub dtype: TensorDataType,

    /// 实际数据（已反量化为 F32）
    pub data: Vec<f32>,
}

/// Tensor 数据类型
#[derive(Debug, Clone, Copy)]
pub enum TensorDataType {
    F32,
    F16,
    Q8_0,
    Q4_0,
}

impl LoRATensor {
    /// 获取 tensor 元素总数
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// 执行矩阵乘法：self @ other
    ///
    /// self: [m, k]
    /// other: [k, n]
    /// result: [m, n]
    pub fn matmul(&self, other: &LoRATensor) -> Result<LoRATensor> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(anyhow!("matmul requires 2D tensors"));
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k != k2 {
            return Err(anyhow!("Incompatible shapes for matmul: [{}, {}] @ [{}, {}]", m, k, k2, n));
        }

        let mut result = vec![0.0f32; m * n];

        // 简单实现（可优化为 SIMD/BLAS）
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(LoRATensor {
            name: format!("{}_x_{}", self.name, other.name),
            shape: vec![m, n],
            dtype: TensorDataType::F32,
            data: result,
        })
    }

    /// 标量乘法：self * scalar
    pub fn scale(&mut self, scalar: f32) {
        for val in &mut self.data {
            *val *= scalar;
        }
    }

    /// 元素相加：self += other
    pub fn add_inplace(&mut self, other: &LoRATensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(anyhow!("Incompatible shapes for add: {:?} vs {:?}", self.shape, other.shape));
        }

        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }

        Ok(())
    }
}

// ==================== LoRA 层结构 ====================

/// LoRA 层（包含 A 和 B 矩阵）
pub struct LoRALayer {
    /// 层名称（e.g., "layers.0.attention.wq"）
    pub layer_name: String,

    /// Rank（秩）
    pub rank: usize,

    /// Alpha（缩放因子）
    pub alpha: f32,

    /// A 矩阵 [rank, in_features]
    pub lora_a: LoRATensor,

    /// B 矩阵 [out_features, rank]
    pub lora_b: LoRATensor,
}

impl LoRALayer {
    /// 计算 LoRA delta: scale * (B @ A)
    ///
    /// 返回形状与原始权重相同的 delta tensor
    pub fn compute_delta(&self, scale: f32) -> Result<LoRATensor> {
        // delta = scale * (B @ A)
        let mut delta = self.lora_b.matmul(&self.lora_a)?;
        delta.scale(scale);
        Ok(delta)
    }

    /// 获取实际缩放因子（alpha / rank）
    pub fn get_scaling_factor(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ==================== LoRA 模型 ====================

/// 完整的 LoRA 模型
pub struct LoRAModel {
    /// LoRA ID
    pub id: String,

    /// 文件路径
    pub path: PathBuf,

    /// LoRA 层映射（layer_name -> LoRALayer）
    pub layers: HashMap<String, LoRALayer>,

    /// 基础模型 rank（默认值）
    pub default_rank: usize,

    /// 基础模型 alpha（默认值）
    pub default_alpha: f32,
}

impl LoRAModel {
    /// 从 GGUF 文件加载 LoRA
    pub fn load(path: &Path) -> Result<Self> {
        println!("[LoRA] Loading LoRA from {:?}", path);

        // 加载 GGUF 文件
        let _gguf = GGUFModel::load(path)
            .context("Failed to load LoRA GGUF")?;

        // TODO: 实际解析 GGUF tensors
        // 当前简化实现：返回空 LoRA
        let id = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        println!("[LoRA] LoRA {} loaded (simplified implementation)", id);

        Ok(Self {
            id,
            path: path.to_path_buf(),
            layers: HashMap::new(),
            default_rank: 8,
            default_alpha: 16.0,
        })
    }

    /// 解析 tensor 名称，提取层名和 LoRA 类型
    ///
    /// 示例：
    /// - "layers.0.attention.wq.lora_A" -> ("layers.0.attention.wq", "A")
    /// - "model.layers.10.mlp.gate.lora_B" -> ("model.layers.10.mlp.gate", "B")
    fn parse_tensor_name(name: &str) -> Option<(String, String)> {
        if let Some(lora_a_pos) = name.rfind(".lora_A") {
            let layer_name = name[..lora_a_pos].to_string();
            return Some((layer_name, "A".to_string()));
        }

        if let Some(lora_b_pos) = name.rfind(".lora_B") {
            let layer_name = name[..lora_b_pos].to_string();
            return Some((layer_name, "B".to_string()));
        }

        None
    }

    /// 合并 LoRA 到基础模型
    ///
    /// 对每一层：
    /// 1. 计算 delta = scale * (B @ A)
    /// 2. 更新权重 W' = W + delta
    pub fn merge_into(&self, base_model_weights: &mut HashMap<String, LoRATensor>, scale: f32) -> Result<()> {
        println!("[LoRA] Merging LoRA {} with scale {:.2}", self.id, scale);

        for (layer_name, lora_layer) in &self.layers {
            // 计算 delta
            let delta = lora_layer.compute_delta(scale)?;

            // 找到对应的基础权重
            if let Some(base_weight) = base_model_weights.get_mut(layer_name) {
                // 合并：W' = W + delta
                base_weight.add_inplace(&delta)?;
                println!("[LoRA]   Merged layer: {}", layer_name);
            } else {
                println!("[LoRA]   Warning: Base layer {} not found", layer_name);
            }
        }

        println!("[LoRA] Merge complete for {} layers", self.layers.len());
        Ok(())
    }

    /// 从基础模型卸载 LoRA
    ///
    /// 反向操作：W = W' - delta
    pub fn unmerge_from(&self, base_model_weights: &mut HashMap<String, LoRATensor>, scale: f32) -> Result<()> {
        println!("[LoRA] Unmerging LoRA {}", self.id);

        for (layer_name, lora_layer) in &self.layers {
            // 计算 delta
            let mut delta = lora_layer.compute_delta(scale)?;

            // 反向：减去 delta
            delta.scale(-1.0);

            // 找到对应的基础权重
            if let Some(base_weight) = base_model_weights.get_mut(layer_name) {
                // 卸载：W = W' - delta
                base_weight.add_inplace(&delta)?;
                println!("[LoRA]   Unmerged layer: {}", layer_name);
            }
        }

        println!("[LoRA] Unmerge complete");
        Ok(())
    }

    /// 获取 LoRA 统计信息
    pub fn stats(&self) -> LoRAStats {
        LoRAStats {
            num_layers: self.layers.len(),
            default_rank: self.default_rank,
            default_alpha: self.default_alpha,
            total_params: self.layers.values()
                .map(|l| l.lora_a.numel() + l.lora_b.numel())
                .sum(),
        }
    }
}

/// LoRA 统计信息
#[derive(Debug)]
pub struct LoRAStats {
    pub num_layers: usize,
    pub default_rank: usize,
    pub default_alpha: f32,
    pub total_params: usize,
}

// ==================== LoRA 管理器 ====================

/// LoRA 管理器（用于批量管理多个 LoRA）
pub struct LoRAManager {
    /// 已加载的 LoRA（ID -> LoRAModel）
    loras: HashMap<String, LoRAModel>,
}

impl LoRAManager {
    /// 创建新的 LoRA 管理器
    pub fn new() -> Self {
        Self {
            loras: HashMap::new(),
        }
    }

    /// 加载 LoRA
    pub fn load_lora(&mut self, path: &Path) -> Result<String> {
        let lora = LoRAModel::load(path)?;
        let id = lora.id.clone();
        self.loras.insert(id.clone(), lora);
        Ok(id)
    }

    /// 卸载 LoRA
    pub fn unload_lora(&mut self, lora_id: &str) -> Result<()> {
        if self.loras.remove(lora_id).is_some() {
            println!("[LoRA] LoRA {} unloaded", lora_id);
            Ok(())
        } else {
            Err(anyhow!("LoRA {} not found", lora_id))
        }
    }

    /// 获取 LoRA
    pub fn get_lora(&self, lora_id: &str) -> Option<&LoRAModel> {
        self.loras.get(lora_id)
    }

    /// 列出所有已加载的 LoRA
    pub fn list_loras(&self) -> Vec<String> {
        self.loras.keys().cloned().collect()
    }

    /// 批量合并多个 LoRA（stacking）
    ///
    /// 按优先级顺序合并（高优先级后合并，效果更强）
    pub fn merge_stack(&self, lora_ids: &[(String, f32)], base_model_weights: &mut HashMap<String, LoRATensor>) -> Result<()> {
        println!("[LoRA] Merging LoRA stack with {} adapters", lora_ids.len());

        for (lora_id, scale) in lora_ids {
            if let Some(lora) = self.get_lora(lora_id) {
                lora.merge_into(base_model_weights, *scale)?;
            } else {
                println!("[LoRA] Warning: LoRA {} not found, skipping", lora_id);
            }
        }

        Ok(())
    }
}

impl Default for LoRAManager {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 辅助函数 ====================

/// 创建示例 LoRA 层（用于测试）
pub fn create_example_lora_layer(layer_name: &str, rank: usize, in_features: usize, out_features: usize) -> LoRALayer {
    // 创建随机初始化的 A 和 B 矩阵（实际应从 GGUF 加载）
    let lora_a = LoRATensor {
        name: format!("{}.lora_A", layer_name),
        shape: vec![rank, in_features],
        dtype: TensorDataType::F32,
        data: vec![0.01; rank * in_features], // 简化：使用小常数
    };

    let lora_b = LoRATensor {
        name: format!("{}.lora_B", layer_name),
        shape: vec![out_features, rank],
        dtype: TensorDataType::F32,
        data: vec![0.01; out_features * rank],
    };

    LoRALayer {
        layer_name: layer_name.to_string(),
        rank,
        alpha: 16.0,
        lora_a,
        lora_b,
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_matmul() {
        let a = LoRATensor {
            name: "A".to_string(),
            shape: vec![2, 3],
            dtype: TensorDataType::F32,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let b = LoRATensor {
            name: "B".to_string(),
            shape: vec![3, 2],
            dtype: TensorDataType::F32,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data.len(), 4);
    }

    #[test]
    fn test_lora_layer_delta() {
        let layer = create_example_lora_layer("test_layer", 4, 512, 512);
        let delta = layer.compute_delta(1.0).unwrap();
        assert_eq!(delta.shape, vec![512, 512]);
    }

    #[test]
    fn test_lora_manager() {
        let manager = LoRAManager::new();
        assert_eq!(manager.list_loras().len(), 0);
        // 实际测试需要真实 LoRA 文件
    }
}
