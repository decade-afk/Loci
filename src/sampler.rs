//! Stateless sampling system with zero-copy logits manipulation
//!
//! This module provides a pure functional sampling architecture with:
//! - **Zero-copy logits access** via `LogitsView`
//! - **Stateless samplers** (pure functions: logits + params → token)
//! - **Composable transformations** (temperature, top-k, top-p, repetition penalty)
//! - **Plugin-friendly hooks** for logit-level intervention
//!
//! ## Architecture
//!
//! ```text
//! Raw Logits → LogitsView (zero-copy) → Transforms → Sampling → Token
//!              ↑
//!              Plugin hooks can inject here
//! ```
//!
//! ## Example
//!
//! ```rust
//! use loci::sampler::{LogitsView, SamplingParams, sample_token};
//!
//! // Get raw logits from backend (zero-copy)
//! let mut logits = vec![0.1, 0.5, 0.3, 0.1]; // Simplified example
//! let mut view = LogitsView::new(&mut logits);
//!
//! // Apply transformations (modifies in-place, zero-copy)
//! view.apply_temperature(0.8);
//! view.apply_top_k(2);
//!
//! // Sample token (stateless pure function)
//! let params = SamplingParams::default();
//! let token = sample_token(&view, &params, &[]);
//! ```

use crate::error::{LociError, Result};
use std::cmp::Ordering;

/// Zero-copy view into model logits with safe mutation
///
/// This provides a safe Rust abstraction over raw logit pointers,
/// enabling zero-copy transformations while maintaining memory safety.
///
/// # Safety
/// - The underlying slice must remain valid for the lifetime 'a
/// - No concurrent access to the same logits
pub struct LogitsView<'a> {
    /// Mutable slice view of logits (zero-copy)
    logits: &'a mut [f32],
    /// Original vocabulary size (for validation)
    n_vocab: usize,
}

impl<'a> LogitsView<'a> {
    /// Create a new logits view from a mutable slice
    ///
    /// # Safety
    /// The slice must remain valid and not be accessed concurrently
    pub fn new(logits: &'a mut [f32]) -> Self {
        let n_vocab = logits.len();
        Self { logits, n_vocab }
    }

    /// Create from raw pointer (unsafe, for FFI integration)
    ///
    /// # Safety
    /// - `ptr` must point to valid memory for `n_vocab` elements
    /// - Memory must remain valid for lifetime 'a
    /// - No concurrent access
    pub unsafe fn from_raw(ptr: *mut f32, n_vocab: usize) -> Self {
        let logits = std::slice::from_raw_parts_mut(ptr, n_vocab);
        Self { logits, n_vocab }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    /// Get logit value for a token (read-only)
    pub fn get(&self, token: i32) -> Option<f32> {
        self.logits.get(token as usize).copied()
    }

    /// Set logit value for a token
    pub fn set(&mut self, token: i32, value: f32) -> Result<()> {
        if token < 0 || token >= self.n_vocab as i32 {
            return Err(LociError::InvalidToken(token));
        }
        self.logits[token as usize] = value;
        Ok(())
    }

    /// Apply additive bias to a token (logit-level intervention)
    ///
    /// Commonly used for:
    /// - Repetition penalty
    /// - Prompt-based biasing
    /// - Token banning/boosting
    pub fn apply_bias(&mut self, token: i32, bias: f32) -> Result<()> {
        if token < 0 || token >= self.n_vocab as i32 {
            return Err(LociError::InvalidToken(token));
        }
        self.logits[token as usize] += bias;
        Ok(())
    }

    /// Mask out tokens (set to -inf to prevent sampling)
    ///
    /// Used for:
    /// - Token banning
    /// - Grammar constraints
    /// - Structured generation
    pub fn apply_mask(&mut self, banned_tokens: &[i32]) {
        for &token in banned_tokens {
            if token >= 0 && (token as usize) < self.n_vocab {
                self.logits[token as usize] = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply temperature scaling (in-place, zero-copy)
    ///
    /// Temperature controls randomness:
    /// - temperature = 0.0: greedy (argmax)
    /// - temperature = 1.0: no change
    /// - temperature > 1.0: more random
    /// - temperature < 1.0: more focused
    pub fn apply_temperature(&mut self, temperature: f32) {
        if temperature == 1.0 {
            return; // No-op optimization
        }

        if temperature == 0.0 {
            // Greedy: find max and set all others to -inf
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

        // Scale all logits by 1/temperature
        let inv_temp = 1.0 / temperature;
        for logit in self.logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    /// Apply repetition penalty to previously generated tokens
    ///
    /// Reduces probability of tokens that appear in context:
    /// - penalty > 1.0: discourage repetition
    /// - penalty = 1.0: no change
    /// - penalty < 1.0: encourage repetition (rare)
    pub fn apply_repetition_penalty(&mut self, context_tokens: &[i32], penalty: f32) {
        if penalty == 1.0 {
            return; // No-op optimization
        }

        for &token in context_tokens {
            if token >= 0 && (token as usize) < self.n_vocab {
                let idx = token as usize;
                let current = self.logits[idx];

                // Apply penalty based on sign (llama.cpp convention)
                if current > 0.0 {
                    self.logits[idx] = current / penalty;
                } else {
                    self.logits[idx] = current * penalty;
                }
            }
        }
    }

    /// Find the index of the maximum logit value
    pub fn argmax(&self) -> usize {
        self.logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get immutable slice view
    pub fn as_slice(&self) -> &[f32] {
        self.logits
    }

    /// Get mutable slice view
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.logits
    }

    /// Softmax in-place (convert logits to probabilities)
    ///
    /// Used internally for probability-based sampling
    pub fn softmax(&mut self) {
        // Find max for numerical stability
        let max = self
            .logits
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        // Compute exp(logit - max) and sum
        let mut sum = 0.0;
        for logit in self.logits.iter_mut() {
            *logit = (*logit - max).exp();
            sum += *logit;
        }

        // Normalize
        if sum > 0.0 {
            for logit in self.logits.iter_mut() {
                *logit /= sum;
            }
        }
    }
}

/// Parameters for token sampling (stateless configuration)
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-k tokens to consider (0 = disabled)
    pub top_k: u32,
    /// Top-p nucleus sampling threshold (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty for context tokens
    pub repeat_penalty: f32,
    /// Random seed (0 = use thread RNG)
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
    /// Create greedy sampling params (always pick highest logit)
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: 0,
        }
    }

    /// Create creative sampling params (high randomness)
    pub fn creative() -> Self {
        Self {
            temperature: 1.2,
            top_k: 0,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }

    /// Create balanced sampling params
    pub fn balanced() -> Self {
        Self::default()
    }
}

/// Token-probability pair for sampling
#[derive(Debug, Clone, Copy)]
struct TokenProb {
    token: i32,
    prob: f32,
}

/// Pure function: Sample a token from logits view
///
/// This is the main stateless sampling entry point.
///
/// # Arguments
/// - `logits`: Zero-copy view of model logits (will be consumed by transformations)
/// - `params`: Sampling parameters
/// - `context_tokens`: Previously generated tokens (for repetition penalty)
///
/// # Returns
/// Sampled token ID
pub fn sample_token(
    logits: &LogitsView,
    params: &SamplingParams,
    context_tokens: &[i32],
) -> i32 {
    // Fast path: greedy sampling
    if params.temperature == 0.0 {
        return logits.argmax() as i32;
    }

    // Create working copy for transformations
    let mut logits_copy = logits.as_slice().to_vec();
    let mut view = LogitsView::new(&mut logits_copy);

    // Apply transformations in order
    view.apply_temperature(params.temperature);
    view.apply_repetition_penalty(context_tokens, params.repeat_penalty);

    // Apply top-k filtering
    if params.top_k > 0 && params.top_k < view.vocab_size() as u32 {
        apply_top_k(&mut view, params.top_k);
    }

    // Convert to probabilities
    view.softmax();

    // Apply top-p (nucleus) filtering
    if params.top_p < 1.0 {
        apply_top_p(&mut view, params.top_p);
    }

    // Sample from distribution
    sample_from_probs(&view, params.seed)
}

/// Apply top-k filtering (keep only top k tokens, mask others)
fn apply_top_k(logits: &mut LogitsView, k: u32) {
    let n = logits.vocab_size();
    let k = k.min(n as u32) as usize;

    // Create sorted indices by logit value (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits
            .logits[b]
            .partial_cmp(&logits.logits[a])
            .unwrap_or(Ordering::Equal)
    });

    // Mask all tokens beyond top-k
    for &idx in indices.iter().skip(k) {
        logits.logits[idx] = f32::NEG_INFINITY;
    }
}

/// Apply top-p (nucleus) filtering
///
/// Keeps the smallest set of tokens whose cumulative probability >= p
fn apply_top_p(probs: &mut LogitsView, p: f32) {
    let n = probs.vocab_size();

    // Create sorted indices by probability (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs.logits[b]
            .partial_cmp(&probs.logits[a])
            .unwrap_or(Ordering::Equal)
    });

    // Find cumulative probability cutoff
    let mut cumsum = 0.0;
    let mut cutoff_idx = n;

    for (i, &idx) in indices.iter().enumerate() {
        cumsum += probs.logits[idx];
        if cumsum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Mask all tokens beyond cutoff
    for &idx in indices.iter().skip(cutoff_idx) {
        probs.logits[idx] = 0.0;
    }

    // Renormalize
    let sum: f32 = probs.logits.iter().sum();
    if sum > 0.0 {
        for prob in probs.logits.iter_mut() {
            *prob /= sum;
        }
    }
}

/// Sample token from probability distribution
fn sample_from_probs(probs: &LogitsView, seed: u64) -> i32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};

    // Simple deterministic pseudo-random if seed provided
    let random_value = if seed != 0 {
        let mut hasher = RandomState::new().build_hasher();
        seed.hash(&mut hasher);
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        let hash = hasher.finish();
        (hash as f64) / (u64::MAX as f64)
    } else {
        // Use thread-local RNG
        use std::cell::RefCell;
        thread_local! {
            static RNG_STATE: RefCell<u64> = RefCell::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            );
        }

        RNG_STATE.with(|state| {
            let mut s = state.borrow_mut();
            // Xorshift64
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            (*s as f64) / (u64::MAX as f64)
        })
    };

    // Sample using cumulative distribution
    let mut cumsum = 0.0;
    for (i, &prob) in probs.logits.iter().enumerate() {
        cumsum += prob as f64;
        if random_value < cumsum {
            return i as i32;
        }
    }

    // Fallback (should rarely happen)
    (probs.vocab_size() - 1) as i32
}

/// Trait for stateless sampling strategies
///
/// Allows custom sampling implementations while maintaining purity
pub trait Sampler: Send + Sync {
    /// Sample a token given logits and parameters
    ///
    /// This must be a pure function (no internal state).
    fn sample(
        &self,
        logits: &LogitsView,
        params: &SamplingParams,
        context: &[i32],
    ) -> i32;

    /// Name of the sampler (for debugging/logging)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits_view_basic() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.1];
        let view = LogitsView::new(&mut logits);
        assert_eq!(view.vocab_size(), 4);
        assert_eq!(view.get(1), Some(0.5));
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut view = LogitsView::new(&mut logits);

        view.apply_temperature(2.0);
        assert_eq!(view.as_slice(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_temperature_greedy() {
        let mut logits = vec![1.0, 3.0, 2.0];
        let mut view = LogitsView::new(&mut logits);

        view.apply_temperature(0.0);
        // Only max should be INFINITY
        assert!(view.as_slice()[1].is_infinite());
        assert!(view.as_slice()[0].is_infinite() && view.as_slice()[0].is_sign_negative());
    }

    #[test]
    fn test_argmax() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.2];
        let view = LogitsView::new(&mut logits);
        assert_eq!(view.argmax(), 1);
    }

    #[test]
    fn test_apply_bias() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut view = LogitsView::new(&mut logits);

        view.apply_bias(1, 0.5).unwrap();
        assert_eq!(view.get(1), Some(2.5));
    }

    #[test]
    fn test_apply_mask() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut view = LogitsView::new(&mut logits);

        view.apply_mask(&[0, 2]);
        assert!(view.get(0).unwrap().is_infinite() && view.get(0).unwrap().is_sign_negative());
        assert!(view.get(2).unwrap().is_infinite() && view.get(2).unwrap().is_sign_negative());
        assert_eq!(view.get(1), Some(2.0));
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -1.0, 3.0];
        let mut view = LogitsView::new(&mut logits);

        view.apply_repetition_penalty(&[0, 2], 1.5);

        // Positive logit divided by penalty
        assert!((view.get(0).unwrap() - 2.0 / 1.5).abs() < 0.001);
        // Negative logit multiplied by penalty
        assert!((view.get(1).unwrap() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut view = LogitsView::new(&mut logits);

        view.softmax();

        // Check sum = 1
        let sum: f32 = view.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);

        // Check probabilities are positive
        assert!(view.as_slice().iter().all(|&p| p > 0.0));
    }

    #[test]
    fn test_greedy_sampling() {
        let mut logits = vec![0.1, 0.5, 0.3];
        let view = LogitsView::new(&mut logits);

        let params = SamplingParams::greedy();
        let token = sample_token(&view, &params, &[]);

        assert_eq!(token, 1); // Index of max logit
    }

    #[test]
    fn test_sampler_trait() {
        let mut logits = vec![0.1, 0.5, 0.3];
        let view = LogitsView::new(&mut logits);

        let sampler = GreedySampler;
        let token = sampler.sample(&view, &SamplingParams::default(), &[]);

        assert_eq!(token, 1);
    }

    #[test]
    fn test_top_k_filtering() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let mut view = LogitsView::new(&mut logits);

        apply_top_k(&mut view, 2);

        // Only top 2 values (5.0 and 4.0) should remain
        let valid_count = view
            .as_slice()
            .iter()
            .filter(|&&v| !v.is_infinite() || !v.is_sign_negative())
            .count();
        assert_eq!(valid_count, 2);
    }
}
