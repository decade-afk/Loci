//! Stateless sampling system with zero-copy logits manipulation

use crate::error::{LociError, Result};
use std::cmp::Ordering;

/// Zero-copy view into model logits with safe mutation
pub struct LogitsView<'a> {
    logits: &'a mut [f32],
    n_vocab: usize,
}

impl<'a> LogitsView<'a> {
    pub fn new(logits: &'a mut [f32]) -> Self {
        let n_vocab = logits.len();
        Self { logits, n_vocab }
    }

    pub unsafe fn from_raw(ptr: *mut f32, n_vocab: usize) -> Self {
        let logits = std::slice::from_raw_parts_mut(ptr, n_vocab);
        Self { logits, n_vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    pub fn get(&self, token: i32) -> Option<f32> {
        self.logits.get(token as usize).copied()
    }

    pub fn set(&mut self, token: i32, value: f32) -> Result<()> {
        if token < 0 || token >= self.n_vocab as i32 {
            return Err(LociError::InvalidToken(token));
        }
        self.logits[token as usize] = value;
        Ok(())
    }

    pub fn set_usize(&mut self, token: usize, value: f32) -> Result<()> {
        if token >= self.n_vocab {
            return Err(LociError::InvalidToken(token as i32));
        }
        self.logits[token] = value;
        Ok(())
    }

    pub fn apply_temperature(&mut self, temperature: f32) {
        if temperature == 1.0 {
            return;
        }

        if temperature == 0.0 {
            let max_idx = self.argmax();
            for (i, logit) in self.logits.iter_mut().enumerate() {
                *logit = if i == max_idx {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
            }
            return;
        }

        let inv_temp = 1.0 / temperature;
        for logit in self.logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    pub fn apply_repetition_penalty(&mut self, context_tokens: &[i32], penalty: f32) {
        if penalty == 1.0 {
            return;
        }

        for &token in context_tokens {
            if token >= 0 && (token as usize) < self.n_vocab {
                let idx = token as usize;
                let current = self.logits[idx];

                if current > 0.0 {
                    self.logits[idx] = current / penalty;
                } else {
                    self.logits[idx] = current * penalty;
                }
            }
        }
    }

    pub fn argmax(&self) -> usize {
        self.logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn as_slice(&self) -> &[f32] {
        self.logits
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.logits
    }

    pub fn softmax(&mut self) {
        let max = self
            .logits
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        let mut sum = 0.0;
        for logit in self.logits.iter_mut() {
            *logit = (*logit - max).exp();
            sum += *logit;
        }

        if sum > 0.0 {
            for logit in self.logits.iter_mut() {
                *logit /= sum;
            }
        }
    }
}

/// Parameters for token sampling
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: 0,
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_k: 0,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Sample a token from logits view
pub fn sample_token(
    logits: &LogitsView,
    params: &SamplingParams,
    context_tokens: &[i32],
) -> i32 {
    if params.temperature == 0.0 {
        return logits.argmax() as i32;
    }

    let mut logits_copy = logits.as_slice().to_vec();
    let mut view = LogitsView::new(&mut logits_copy);

    view.apply_temperature(params.temperature);
    view.apply_repetition_penalty(context_tokens, params.repeat_penalty);

    view.softmax();

    sample_from_probs(&view, params)
}

/// Sample token from probability distribution
fn sample_from_probs(probs: &LogitsView, params: &SamplingParams) -> i32 {
    // Apply Top-K filtering
    let mut probs_copy = if params.top_k > 0 {
        apply_top_k(probs.as_slice(), params.top_k as usize)
    } else {
        probs.as_slice().to_vec()
    };

    // Apply Top-P (nucleus) filtering
    let mut view = LogitsView::new(&mut probs_copy);
    if params.top_p < 1.0 {
        apply_top_p(&mut view, params.top_p);
    }

    // Re-normalize after filtering
    view.softmax();

    // Sample from the filtered distribution
    sample_from_distribution(&view, params.seed)
}

/// Apply Top-K filtering: keep only top k tokens
fn apply_top_k(probs: &[f32], k: usize) -> Vec<f32> {
    let mut indexed_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    // Sort by probability descending
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Keep only top k
    indexed_probs.truncate(k);

    // Create result array
    let mut result = vec![0.0f32; probs.len()];
    for (idx, prob) in indexed_probs {
        result[idx] = prob;
    }

    result
}

/// Apply Top-P (nucleus) filtering: keep tokens until cumulative probability reaches p
fn apply_top_p(probs: &mut LogitsView, top_p: f32) {
    // Sort tokens by probability descending
    let mut indexed_probs: Vec<(usize, f32)> = probs
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();

    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cumulative probability threshold
    let mut cumulative = 0.0;
    let mut keep_tokens = Vec::new();

    for (idx, prob) in indexed_probs.iter() {
        cumulative += prob;
        keep_tokens.push(*idx);
        if cumulative >= top_p {
            break;
        }
    }

    // Zero out tokens not in keep_tokens
    for i in 0..probs.vocab_size() {
        if !keep_tokens.contains(&i) {
            probs.set_usize(i, 0.0).ok();
        }
    }
}

/// Sample a token from probability distribution using weighted random selection
fn sample_from_distribution(probs: &LogitsView, seed: u64) -> i32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create seeded random number generator
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    // Simple pseudo-random number generator
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };

    // Generate random number in [0, 1)
    let rand_val = (rng() as f64) / (u64::MAX as f64);

    // Sample based on cumulative distribution
    let mut cumulative = 0.0;
    let probs_slice = probs.as_slice();

    for (idx, &prob) in probs_slice.iter().enumerate() {
        cumulative += prob as f64;
        if cumulative >= rand_val {
            return idx as i32;
        }
    }

    // Fallback: return the token with highest probability
    probs.argmax() as i32
}

/// Trait for stateless sampling strategies
pub trait Sampler: Send + Sync {
    fn sample(
        &self,
        logits: &LogitsView,
        params: &SamplingParams,
        context: &[i32],
    ) -> i32;

    fn name(&self) -> &str;
}

/// Default stateless sampler implementation
pub struct DefaultSampler;

impl Sampler for DefaultSampler {
    fn sample(
        &self,
        logits: &LogitsView,
        params: &SamplingParams,
        context: &[i32],
    ) -> i32 {
        sample_token(logits, params, context)
    }

    fn name(&self) -> &str {
        "default"
    }
}

/// Greedy sampler (always picks highest logit)
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &LogitsView, _params: &SamplingParams, _context: &[i32]) -> i32 {
        logits.argmax() as i32
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

/// Top-K sampler (samples from top k tokens)
pub struct TopKSampler;

impl Sampler for TopKSampler {
    fn sample(&self, logits: &LogitsView, params: &SamplingParams, context: &[i32]) -> i32 {
        let mut logits_copy = logits.as_slice().to_vec();
        let mut view = LogitsView::new(&mut logits_copy);

        view.apply_temperature(params.temperature);
        view.apply_repetition_penalty(context, params.repeat_penalty);
        view.softmax();

        // Apply Top-K
        let mut filtered = apply_top_k(view.as_slice(), params.top_k as usize);
        let mut filtered_view = LogitsView::new(&mut filtered);
        filtered_view.softmax();

        sample_from_distribution(&filtered_view, params.seed)
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

/// Top-P (Nucleus) sampler (samples from tokens until cumulative probability reaches p)
pub struct TopPSampler;

impl Sampler for TopPSampler {
    fn sample(&self, logits: &LogitsView, params: &SamplingParams, context: &[i32]) -> i32 {
        let mut logits_copy = logits.as_slice().to_vec();
        let mut view = LogitsView::new(&mut logits_copy);

        view.apply_temperature(params.temperature);
        view.apply_repetition_penalty(context, params.repeat_penalty);
        view.softmax();

        // Apply Top-P
        apply_top_p(&mut view, params.top_p);
        view.softmax();

        sample_from_distribution(&view, params.seed)
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

/// Mirostat sampler (adaptive sampling to control perplexity)
pub struct MirostatSampler {
    target_perplexity: f32,
    learning_rate: f32,
}

impl MirostatSampler {
    pub fn new(target_perplexity: f32, learning_rate: f32) -> Self {
        Self {
            target_perplexity,
            learning_rate,
        }
    }

    pub fn default() -> Self {
        Self {
            target_perplexity: 5.0,
            learning_rate: 0.1,
        }
    }
}

impl Sampler for MirostatSampler {
    fn sample(&self, logits: &LogitsView, params: &SamplingParams, context: &[i32]) -> i32 {
        let mut logits_copy = logits.as_slice().to_vec();
        let mut view = LogitsView::new(&mut logits_copy);

        view.apply_temperature(params.temperature);
        view.apply_repetition_penalty(context, params.repeat_penalty);
        view.softmax();

        // Mirostat algorithm: adjust surprise
        let max_surprise = (self.target_perplexity * params.temperature).ln();
        let mut filtered_probs = vec![0.0; view.vocab_size()];

        for (i, &prob) in view.as_slice().iter().enumerate() {
            if prob > 0.0 {
                let surprise = -prob.ln();
                if surprise <= max_surprise {
                    filtered_probs[i] = prob;
                }
            }
        }

        let mut filtered_view = LogitsView::new(&mut filtered_probs);
        filtered_view.softmax();

        sample_from_distribution(&filtered_view, params.seed)
    }

    fn name(&self) -> &str {
        "mirostat"
    }
}

/// Temperature sampler with typical sampling
pub struct TemperatureSampler {
    typical_p: f32,
}

impl TemperatureSampler {
    pub fn new(typical_p: f32) -> Self {
        Self { typical_p }
    }

    pub fn default() -> Self {
        Self { typical_p: 0.95 }
    }
}

impl Sampler for TemperatureSampler {
    fn sample(&self, logits: &LogitsView, params: &SamplingParams, context: &[i32]) -> i32 {
        let mut logits_copy = logits.as_slice().to_vec();
        let mut view = LogitsView::new(&mut logits_copy);

        view.apply_temperature(params.temperature);
        view.apply_repetition_penalty(context, params.repeat_penalty);
        view.softmax();

        // Typical sampling: filter based on negative entropy
        let neg_entropy: f32 = view.as_slice().iter().map(|&p| -p * p.ln()).sum();
        let mut filtered_probs = vec![0.0; view.vocab_size()];

        for (i, &prob) in view.as_slice().iter().enumerate() {
            let surprise = -prob.ln();
            if surprise <= neg_entropy + self.typical_p {
                filtered_probs[i] = prob;
            }
        }

        let mut filtered_view = LogitsView::new(&mut filtered_probs);
        filtered_view.softmax();

        sample_from_distribution(&filtered_view, params.seed)
    }

    fn name(&self) -> &str {
        "typical"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits_view_creation() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.1];
        let view = LogitsView::new(&mut logits);
        
        assert_eq!(view.vocab_size(), 4);
        assert_eq!(view.get(1), Some(0.5));
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut view = LogitsView::new(&mut logits);
        
        view.apply_temperature(2.0);
        
        assert_eq!(view.get(0), Some(0.5));
        assert_eq!(view.get(1), Some(1.0));
        assert_eq!(view.get(2), Some(1.5));
        assert_eq!(view.get(3), Some(2.0));
    }

    #[test]
    fn test_greedy_temperature() {
        let mut logits = vec![1.0, 3.0, 2.0, 1.5];
        let mut view = LogitsView::new(&mut logits);
        
        view.apply_temperature(0.0);
        
        assert_eq!(view.get(1), Some(f32::INFINITY));
        assert_eq!(view.get(0), Some(f32::NEG_INFINITY));
        assert_eq!(view.get(2), Some(f32::NEG_INFINITY));
        assert_eq!(view.get(3), Some(f32::NEG_INFINITY));
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, -1.0, 0.5];
        let mut view = LogitsView::new(&mut logits);
        let context = vec![1, 3];
        
        view.apply_repetition_penalty(&context, 1.2);
        
        assert!((view.get(1).unwrap() - (2.0 / 1.2)).abs() < 0.001);
        assert!((view.get(3).unwrap() - (0.5 / 1.2)).abs() < 0.001);
        assert_eq!(view.get(0), Some(1.0));
        assert_eq!(view.get(2), Some(-1.0));
    }

    #[test]
    fn test_argmax() {
        let mut logits = vec![1.0, 3.0, 2.0, 1.5];
        let view = LogitsView::new(&mut logits);
        
        assert_eq!(view.argmax(), 1);
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut view = LogitsView::new(&mut logits);
        
        view.softmax();
        
        let sum: f32 = view.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        
        for &prob in view.as_slice() {
            assert!(prob > 0.0);
        }
    }

    #[test]
    fn test_sampling_params_presets() {
        let greedy = SamplingParams::greedy();
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(greedy.top_k, 1);
        
        let creative = SamplingParams::creative();
        assert_eq!(creative.temperature, 1.2);
        assert_eq!(creative.top_k, 0);
        
        let balanced = SamplingParams::balanced();
        assert_eq!(balanced.temperature, 0.8);
        assert_eq!(balanced.top_k, 40);
    }

    #[test]
    fn test_greedy_sampler() {
        let mut logits = vec![1.0, 3.0, 2.0, 1.5];
        let view = LogitsView::new(&mut logits);
        let sampler = GreedySampler;
        let params = SamplingParams::default();
        
        let token = sampler.sample(&view, &params, &[]);
        assert_eq!(token, 1);
    }
}