//! Sampling Module
//!
//! This module provides core functionality for the Loci project.
//!


use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use anyhow::Result;




#[derive(Debug, Clone)]
    /// SamplerConfig structure
pub struct SamplerConfig {
    
    pub temperature: f32,

    
    pub top_k: usize,

    
    pub top_p: f32,

    
    pub min_p: f32,

    
    pub repetition_penalty: f32,
}

// Implementation for Default
impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,        
            top_p: 1.0,      
            min_p: 0.0,      
            repetition_penalty: 1.0,  
        }
    }
}




    /// Sampler structure
pub struct Sampler {
    config: SamplerConfig,
    rng: StdRng,

    
    history: Vec<usize>,
}

// Implementation for Sampler
impl Sampler {
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /// new function
    pub fn new(config: SamplerConfig, seed: u64) -> Self {
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            history: Vec::new(),
        }
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /// sample function
    pub fn sample(&mut self, logits: &mut [f32]) -> Result<usize> {
        if logits.is_empty() {
            anyhow::bail!("Empty logits array");
        }

        
        if self.config.repetition_penalty != 1.0 && !self.history.is_empty() {
            self.apply_repetition_penalty(logits);
        }

        
        if self.config.temperature != 1.0 {
            self.apply_temperature(logits);
        }

        
        let probs = self.softmax(logits);

        
        let filtered_indices = self.apply_filters(&probs);

        
        let token_id = self.sample_from_distribution(&probs, &filtered_indices)?;

        
        self.history.push(token_id);

        Ok(token_id)
    }

    
    
    
    
    
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

    
    
    
    
    
    
    fn apply_temperature(&self, logits: &mut [f32]) {
        let temp = self.config.temperature;

        
        if temp == 0.0 {
            return;  
        }

        for logit in logits.iter_mut() {
            *logit /= temp;
        }
    }

    
    
    
    
    
    
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        
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

    
    
    
    fn apply_filters(&self, probs: &[f32]) -> Vec<usize> {
        
        let mut sorted_indices: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(idx, &prob)| (idx, prob))
            .collect();

        sorted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered_indices = Vec::new();

        
        let top_k_limit = if self.config.top_k > 0 {
            self.config.top_k.min(sorted_indices.len())
        } else {
            sorted_indices.len()
        };

        
        let max_prob = sorted_indices[0].1;
        let min_prob_threshold = max_prob * self.config.min_p;

        
        let mut cumulative_prob = 0.0;

        for (idx, prob) in sorted_indices.iter().take(top_k_limit) {
            
            if self.config.min_p > 0.0 && *prob < min_prob_threshold {
                break;
            }

            filtered_indices.push(*idx);
            cumulative_prob += prob;

            
            if self.config.top_p < 1.0 && cumulative_prob >= self.config.top_p {
                break;
            }
        }

        
        if filtered_indices.is_empty() {
            filtered_indices.push(sorted_indices[0].0);
        }

        filtered_indices
    }

    
    fn sample_from_distribution(&mut self, probs: &[f32], indices: &[usize]) -> Result<usize> {
        if indices.is_empty() {
            anyhow::bail!("No valid tokens to sample from");
        }

        
        if indices.len() == 1 {
            return Ok(indices[0]);
        }

        
        let filtered_probs: Vec<f32> = indices.iter()
            .map(|&idx| probs[idx])
            .collect();

        
        let sum: f32 = filtered_probs.iter().sum();
        let normalized_probs: Vec<f32> = filtered_probs.iter()
            .map(|&p| p / sum)
            .collect();

        
        let dist = WeightedIndex::new(&normalized_probs)
            .map_err(|e| anyhow::anyhow!("Failed to create weighted distribution: {}", e))?;

        let sampled_idx = dist.sample(&mut self.rng);
        Ok(indices[sampled_idx])
    }

    
    /// reset function
    pub fn reset(&mut self) {
        self.history.clear();
    }

    
    /// reset_with_seed function
    pub fn reset_with_seed(&mut self, seed: u64) {
        self.history.clear();
        self.rng = StdRng::seed_from_u64(seed);
    }

    
    /// history_len function
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sampling() {
        
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
        
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 3,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let logits = vec![1.0, 2.0, 5.0, 3.0, 4.0];  

        
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
        
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 0.9,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![5.0, 4.0, 3.0, 0.1, 0.1];

        
        let token = sampler.sample(&mut logits).unwrap();
        assert!(token < 3, "Top-P should filter low probability tokens");
    }

    #[test]
    fn test_min_p_filtering() {
        
        let config = SamplerConfig {
            temperature: 1.0,
            min_p: 0.5,  
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![10.0, 9.0, 5.0, 1.0, 0.1];  

        let token = sampler.sample(&mut logits).unwrap();
        
        assert!(token < 2, "Min-P should filter tokens with probability < 50% of max");
    }

    #[test]
    fn test_repetition_penalty() {
        
        let config = SamplerConfig {
            temperature: 0.0,  
            repetition_penalty: 1.5,
            ..Default::default()
        };

        let mut sampler = Sampler::new(config, 42);

        
        let mut logits1 = vec![1.0, 2.0, 5.0, 3.0, 4.0];
        let token1 = sampler.sample(&mut logits1).unwrap();
        assert_eq!(token1, 2);

        
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
        
        let config = SamplerConfig::default();
        let sampler = Sampler::new(config, 42);

        let logits = vec![1000.0, 1001.0, 1002.0];  
        let probs = sampler.softmax(&logits);

        
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax probabilities should sum to 1.0");

        
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
        
        let config = SamplerConfig {
            temperature: 0.7,
            top_k: 5,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.2,
        };

        let mut sampler = Sampler::new(config, 42);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        
        let token = sampler.sample(&mut logits);
        assert!(token.is_ok(), "All parameters should work together");
    }
}
