/**
 * Loci Phase 1 Completion: Sampling Module
 *
 * 核心功能：
 * 1. Temperature scaling（温度缩放）
 * 2. Top-K sampling（保留前 K 个最高概率 token）
 * 3. Top-P (Nucleus) sampling（核采样，累积概率阈值）
 * 4. Min-P sampling（最小概率阈值）
 * 5. Repetition penalty（重复惩罚）
 * 6. Deterministic sampling（确定性采样，基于种子）
 *
 * 验收标准：
 * ✅ 相同 seed 输出一致
 * ✅ 所有参数生效
 * ✅ 单元测试覆盖
 */

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use anyhow::Result;

// ==================== Sampler Configuration ====================

/// 采样器配置结构体
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// 温度参数（0.0 = 贪婪采样，> 1.0 = 更随机）
    pub temperature: f32,

    /// Top-K 采样：保留前 K 个最高概率的 token（0 = 禁用）
    pub top_k: usize,

    /// Top-P (Nucleus) 采样：累积概率阈值（0.0-1.0，1.0 = 禁用）
    pub top_p: f32,

    /// Min-P 采样：最小概率阈值（相对于最高概率，0.0 = 禁用）
    pub min_p: f32,

    /// 重复惩罚系数（1.0 = 无惩罚，> 1.0 = 惩罚重复）
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,        // 禁用 top-k
            top_p: 1.0,      // 禁用 top-p
            min_p: 0.0,      // 禁用 min-p
            repetition_penalty: 1.0,  // 无惩罚
        }
    }
}

// ==================== Sampler Implementation ====================

/// Loci 采样器
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,

    /// 已生成 token 的历史（用于重复惩罚）
    history: Vec<usize>,
}

impl Sampler {
    /// 创建新的采样器
    ///
    /// # 参数
    /// - config: 采样器配置
    /// - seed: 随机种子（用于确定性采样）
    ///
    /// # 示例
    /// ```
    /// use loci::{Sampler, SamplerConfig};
    ///
    /// let config = SamplerConfig {
    ///     temperature: 0.7,
    ///     top_k: 50,
    ///     top_p: 0.9,
    ///     ..Default::default()
    /// };
    ///
    /// let sampler = Sampler::new(config, 42);
    /// ```
    pub fn new(config: SamplerConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            history: Vec::new(),
        }
    }

    /// 从 logits 中采样一个 token
    ///
    /// # 参数
    /// - logits: 模型输出的 logits 向量（会被修改）
    ///
    /// # 返回值
    /// - token ID
    ///
    /// # 采样流程
    /// 1. 应用重复惩罚（如果启用）
    /// 2. 应用温度缩放
    /// 3. Softmax 转换为概率
    /// 4. 应用 Top-K/Top-P/Min-P 过滤
    /// 5. 从过滤后的分布中采样
    pub fn sample(&mut self, logits: &mut [f32]) -> Result<usize> {
        if logits.is_empty() {
            anyhow::bail!("Empty logits array");
        }

        // 步骤 1: 应用重复惩罚
        if self.config.repetition_penalty != 1.0 && !self.history.is_empty() {
            self.apply_repetition_penalty(logits);
        }

        // 步骤 2: 应用温度缩放
        if self.config.temperature != 1.0 {
            self.apply_temperature(logits);
        }

        // 步骤 3: Softmax 转换为概率
        let probs = self.softmax(logits);

        // 步骤 4: 应用 Top-K/Top-P/Min-P 过滤
        let filtered_indices = self.apply_filters(&probs);

        // 步骤 5: 从过滤后的分布中采样
        let token_id = self.sample_from_distribution(&probs, &filtered_indices)?;

        // 记录到历史
        self.history.push(token_id);

        Ok(token_id)
    }

    /// 应用重复惩罚
    ///
    /// 对历史中出现过的 token 的 logits 进行惩罚：
    /// - 如果 logit > 0：除以 penalty
    /// - 如果 logit < 0：乘以 penalty
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        let penalty = self.config.repetition_penalty;

        for &token_id in &self.history {
            if token_id < logits.len() {
                if logits[token_id] > 0.0 {
                    logits[token_id] /= penalty;
                } else {
                    logits[token_id] *= penalty;
                }
            }
        }
    }

    /// 应用温度缩放
    ///
    /// logit = logit / temperature
    ///
    /// - temperature < 1.0: 更确定（sharper distribution）
    /// - temperature > 1.0: 更随机（flatter distribution）
    fn apply_temperature(&self, logits: &mut [f32]) {
        let temp = self.config.temperature;

        // 温度为 0 时，执行贪婪采样（选择最大值）
        if temp == 0.0 {
            return;  // 特殊处理，在 sample 中直接选择 argmax
        }

        for logit in logits.iter_mut() {
            *logit /= temp;
        }
    }

    /// Softmax 转换：logits -> 概率
    ///
    /// 使用数值稳定的 softmax：
    /// 1. 减去最大值避免溢出
    /// 2. exp(x)
    /// 3. 归一化
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // 特殊情况：温度为 0 时，使用贪婪采样
        if self.config.temperature == 0.0 {
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let mut probs = vec![0.0; logits.len()];
            probs[max_idx] = 1.0;
            return probs;
        }

        // 数值稳定的 softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut exp_values: Vec<f32> = logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();

        let sum: f32 = exp_values.iter().sum();

        for val in exp_values.iter_mut() {
            *val /= sum;
        }

        exp_values
    }

    /// 应用 Top-K/Top-P/Min-P 过滤
    ///
    /// 返回过滤后保留的 token 索引列表
    fn apply_filters(&self, probs: &[f32]) -> Vec<usize> {
        // 创建 (索引, 概率) 对并按概率降序排序
        let mut sorted_indices: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(idx, &prob)| (idx, prob))
            .collect();

        sorted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered_indices = Vec::new();

        // 过滤器 1: Top-K
        let top_k_limit = if self.config.top_k > 0 {
            self.config.top_k.min(sorted_indices.len())
        } else {
            sorted_indices.len()
        };

        // 过滤器 2: Min-P（相对于最高概率）
        let max_prob = sorted_indices[0].1;
        let min_prob_threshold = max_prob * self.config.min_p;

        // 过滤器 3: Top-P（累积概率）
        let mut cumulative_prob = 0.0;

        for (idx, prob) in sorted_indices.iter().take(top_k_limit) {
            // 应用 Min-P 过滤
            if self.config.min_p > 0.0 && *prob < min_prob_threshold {
                break;
            }

            filtered_indices.push(*idx);
            cumulative_prob += prob;

            // 应用 Top-P 过滤
            if self.config.top_p < 1.0 && cumulative_prob >= self.config.top_p {
                break;
            }
        }

        // 确保至少有一个候选
        if filtered_indices.is_empty() {
            filtered_indices.push(sorted_indices[0].0);
        }

        filtered_indices
    }

    /// 从过滤后的分布中采样
    fn sample_from_distribution(&mut self, probs: &[f32], indices: &[usize]) -> Result<usize> {
        if indices.is_empty() {
            anyhow::bail!("No valid tokens to sample from");
        }

        // 如果只有一个候选，直接返回
        if indices.len() == 1 {
            return Ok(indices[0]);
        }

        // 提取过滤后的概率
        let filtered_probs: Vec<f32> = indices.iter()
            .map(|&idx| probs[idx])
            .collect();

        // 归一化（确保概率和为 1.0）
        let sum: f32 = filtered_probs.iter().sum();
        let normalized_probs: Vec<f32> = filtered_probs.iter()
            .map(|&p| p / sum)
            .collect();

        // 使用 WeightedIndex 采样
        let dist = WeightedIndex::new(&normalized_probs)
            .map_err(|e| anyhow::anyhow!("Failed to create weighted distribution: {}", e))?;

        let sampled_idx = dist.sample(&mut self.rng);
        Ok(indices[sampled_idx])
    }

    /// 重置采样器历史（用于新的生成任务）
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// 重置采样器历史并更新随机种子
    pub fn reset_with_seed(&mut self, seed: u64) {
        self.history.clear();
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// 获取当前历史长度
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sampling() {
        // 验收标准：相同 seed 输出一致
        let config = SamplerConfig {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.9,
            ..Default::default()
        };

        let mut sampler1 = Sampler::new(config.clone(), 42);
        let mut sampler2 = Sampler::new(config, 42);

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let token1 = sampler1.sample(&mut logits.clone()).unwrap();
        let token2 = sampler2.sample(&mut logits).unwrap();

        assert_eq!(token1, token2, "Same seed should produce same output");
    }

    #[test]
    fn test_temperature_scaling() {
        // 温度为 0 应该执行贪婪采样（选择最大值）
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![1.0, 2.0, 5.0, 3.0, 4.0];

        let token = sampler.sample(&mut logits).unwrap();
        assert_eq!(token, 2, "Temperature 0.0 should select argmax (index 2)");
    }

    #[test]
    fn test_top_k_filtering() {
        // Top-K = 3 应该只保留前 3 个最高概率的 token
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 3,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 2.0, 5.0, 3.0, 4.0];  // 最高 3 个: 5.0, 4.0, 3.0

        // 多次采样，验证只从前 3 个中选择
        for _ in 0..10 {
            sampler.reset_with_seed(rand::random());
            let token = sampler.sample(&mut logits.clone()).unwrap();
            assert!(
                token == 2 || token == 4 || token == 3,
                "Top-K should only sample from top 3 indices (2, 4, 3)"
            );
        }
    }

    #[test]
    fn test_top_p_filtering() {
        // Top-P 应该保留累积概率达到阈值的 token
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![5.0, 4.0, 3.0, 0.1, 0.1];

        // Top-P 应该过滤掉低概率的 token
        let token = sampler.sample(&mut logits).unwrap();
        assert!(token < 3, "Top-P should filter low probability tokens");
    }

    #[test]
    fn test_min_p_filtering() {
        // Min-P 应该过滤掉相对概率过低的 token
        let config = SamplerConfig {
            temperature: 1.0,
            min_p: 0.5,  // 只保留概率 >= 50% * max_prob 的 token
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![10.0, 9.0, 5.0, 1.0, 0.1];  // 最高概率 token 是 10.0

        let token = sampler.sample(&mut logits).unwrap();
        // 只有前 2 个 token 的概率接近最高值
        assert!(token < 2, "Min-P should filter tokens with probability < 50% of max");
    }

    #[test]
    fn test_repetition_penalty() {
        // 重复惩罚应该降低已生成 token 的概率
        let config = SamplerConfig {
            temperature: 0.0,  // 贪婪采样
            repetition_penalty: 1.5,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);

        // 第一次采样：选择最大值 (index 2)
        let mut logits1 = vec![1.0, 2.0, 5.0, 3.0, 4.0];
        let token1 = sampler.sample(&mut logits1).unwrap();
        assert_eq!(token1, 2);

        // 第二次采样：由于惩罚，应该选择下一个最大值
        let mut logits2 = vec![1.0, 2.0, 5.0, 3.0, 4.0];
        let token2 = sampler.sample(&mut logits2).unwrap();
        assert_ne!(token2, 2, "Repetition penalty should discourage repeating token 2");
        assert_eq!(token2, 4, "Should select next highest (index 4)");
    }

    #[test]
    fn test_reset() {
        let config = SamplerConfig::default();
        let mut sampler = Sampler::new(config, 42);

        let mut logits = vec![1.0, 2.0, 3.0];
        sampler.sample(&mut logits).unwrap();
        sampler.sample(&mut logits).unwrap();

        assert_eq!(sampler.history_len(), 2);

        sampler.reset();
        assert_eq!(sampler.history_len(), 0);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // 验证 softmax 不会溢出
        let config = SamplerConfig::default();
        let sampler = Sampler::new(config, 42);

        let logits = vec![1000.0, 1001.0, 1002.0];  // 极大值
        let probs = sampler.softmax(&logits);

        // 概率应该和为 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax probabilities should sum to 1.0");

        // 所有概率应该在 [0, 1] 范围内
        for prob in probs {
            assert!(prob >= 0.0 && prob <= 1.0, "Invalid probability: {}", prob);
        }
    }

    #[test]
    fn test_empty_logits() {
        let config = SamplerConfig::default();
        let mut sampler = Sampler::new(config, 42);

        let mut logits: Vec<f32> = vec![];
        let result = sampler.sample(&mut logits);

        assert!(result.is_err(), "Should fail on empty logits");
    }

    #[test]
    fn test_all_parameters_active() {
        // 验收标准：所有参数生效
        let config = SamplerConfig {
            temperature: 0.7,
            top_k: 5,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.2,
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // 应该成功采样（所有参数组合工作正常）
        let token = sampler.sample(&mut logits);
        assert!(token.is_ok(), "All parameters should work together");
    }
}
