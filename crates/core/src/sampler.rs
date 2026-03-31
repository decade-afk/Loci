use crate::error::{LociError, Result};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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
        let logits = unsafe { std::slice::from_raw_parts_mut(ptr, n_vocab) };
        Self { logits, n_vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    pub fn get(&self, token: i32) -> Option<f32> {
        if token < 0 {
            return None;
        }
        self.logits.get(token as usize).copied()
    }

    pub fn set_usize(&mut self, token: usize, value: f32) -> Result<()> {
        if token >= self.n_vocab {
            return Err(LociError::InvalidArgument(format!(
                "token index out of range: {token}"
            )));
        }

        self.logits[token] = value;
        Ok(())
    }

    pub fn argmax(&self) -> usize {
        self.logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    pub fn as_slice(&self) -> &[f32] {
        self.logits
    }

    pub fn apply_temperature(&mut self, temperature: f32) {
        if (temperature - 1.0).abs() < f32::EPSILON {
            return;
        }

        if temperature <= 0.0 {
            let max_index = self.argmax();
            for (index, logit) in self.logits.iter_mut().enumerate() {
                *logit = if index == max_index {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
            }
            return;
        }

        let inv_temperature = 1.0 / temperature;
        for logit in self.logits.iter_mut() {
            *logit *= inv_temperature;
        }
    }

    pub fn apply_repetition_penalty(&mut self, context_tokens: &[i32], penalty: f32) {
        if (penalty - 1.0).abs() < f32::EPSILON || penalty <= 0.0 {
            return;
        }

        for &token in context_tokens {
            if token < 0 {
                continue;
            }

            let token_index = token as usize;
            if token_index >= self.n_vocab {
                continue;
            }

            let current = self.logits[token_index];
            self.logits[token_index] = if current > 0.0 {
                current / penalty
            } else {
                current * penalty
            };
        }
    }

    pub fn softmax(&mut self) {
        let max_logit = self
            .logits
            .iter()
            .copied()
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
            .unwrap_or(0.0);

        let mut sum = 0.0f32;
        for logit in self.logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }

        if sum > 0.0 {
            for logit in self.logits.iter_mut() {
                *logit /= sum;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.1,
            seed: 0,
        }
    }
}

pub fn sample_token(
    logits: &LogitsView<'_>,
    params: &SamplingParams,
    context_tokens: &[i32],
) -> i32 {
    if params.temperature <= 0.0 {
        return logits.argmax() as i32;
    }

    let mut logits_copy = logits.as_slice().to_vec();
    let mut view = LogitsView::new(&mut logits_copy);
    view.apply_temperature(params.temperature);
    view.apply_repetition_penalty(context_tokens, params.repeat_penalty);
    view.softmax();

    sample_from_probs(&view, params)
}

fn sample_from_probs(probs: &LogitsView<'_>, params: &SamplingParams) -> i32 {
    let mut filtered = if params.top_k > 0 {
        apply_top_k(probs.as_slice(), params.top_k as usize)
    } else {
        probs.as_slice().to_vec()
    };

    let mut filtered_view = LogitsView::new(&mut filtered);
    if params.top_p < 1.0 {
        apply_top_p(&mut filtered_view, params.top_p);
    }
    if params.min_p > 0.0 {
        apply_min_p(&mut filtered_view, params.min_p);
    }
    filtered_view.softmax();

    sample_from_distribution(&filtered_view, params.seed)
}

fn apply_top_k(probs: &[f32], k: usize) -> Vec<f32> {
    let mut indexed_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    indexed_probs.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    indexed_probs.truncate(k);

    let mut result = vec![0.0f32; probs.len()];
    for (index, prob) in indexed_probs {
        result[index] = prob;
    }
    result
}

fn apply_top_p(probs: &mut LogitsView<'_>, top_p: f32) {
    let mut indexed_probs: Vec<(usize, f32)> = probs
        .as_slice()
        .iter()
        .enumerate()
        .map(|(index, &value)| (index, value))
        .collect();
    indexed_probs.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut keep = vec![false; probs.vocab_size()];
    for (index, prob) in indexed_probs {
        cumulative += prob;
        keep[index] = true;
        if cumulative >= top_p {
            break;
        }
    }

    for (index, keep_token) in keep.iter().enumerate() {
        if !keep_token {
            let _ = probs.set_usize(index, 0.0);
        }
    }
}

fn apply_min_p(probs: &mut LogitsView<'_>, min_p: f32) {
    let max_prob = probs.as_slice().iter().copied().fold(0.0f32, f32::max);
    if max_prob <= 0.0 {
        return;
    }

    let threshold = max_prob * min_p;
    for index in 0..probs.vocab_size() {
        if probs.as_slice()[index] < threshold {
            let _ = probs.set_usize(index, 0.0);
        }
    }
}

fn sample_from_distribution(probs: &LogitsView<'_>, seed: u64) -> i32 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    let mut next_u64 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let target = (next_u64() as f64) / (u64::MAX as f64);
    let mut cumulative = 0.0f64;
    for (index, &prob) in probs.as_slice().iter().enumerate() {
        cumulative += prob as f64;
        if cumulative >= target {
            return index as i32;
        }
    }

    probs.argmax() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_becomes_greedy() {
        let mut logits = vec![1.0, 3.0, 2.0];
        let token = sample_token(
            &LogitsView::new(&mut logits),
            &SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            &[],
        );

        assert_eq!(token, 1);
    }

    #[test]
    fn repetition_penalty_can_change_selected_token() {
        let mut logits = vec![1.0, 3.0, 2.9];
        let token = sample_token(
            &LogitsView::new(&mut logits),
            &SamplingParams {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 2.0,
                seed: 0,
            },
            &[1],
        );

        assert_eq!(token, 2);
    }

    #[test]
    fn top_k_limits_candidates() {
        let mut logits = vec![0.1, 5.0, 4.0, 3.0];
        let token = sample_token(
            &LogitsView::new(&mut logits),
            &SamplingParams {
                temperature: 1.0,
                top_k: 2,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                seed: 42,
            },
            &[],
        );

        assert!(token == 1 || token == 2);
    }

    #[test]
    fn top_p_filters_tail_tokens() {
        let mut probs = vec![0.60, 0.30, 0.09, 0.01];
        let mut view = LogitsView::new(&mut probs);
        apply_top_p(&mut view, 0.85);

        assert!(view.get(0).unwrap() > 0.0);
        assert!(view.get(1).unwrap() > 0.0);
        assert_eq!(view.get(2), Some(0.0));
        assert_eq!(view.get(3), Some(0.0));
    }

    #[test]
    fn min_p_filters_by_relative_threshold() {
        let mut probs = vec![0.60, 0.24, 0.12, 0.04];
        let mut view = LogitsView::new(&mut probs);
        apply_min_p(&mut view, 0.25);

        assert!(view.get(0).unwrap() > 0.0);
        assert!(view.get(1).unwrap() > 0.0);
        assert_eq!(view.get(2), Some(0.0));
        assert_eq!(view.get(3), Some(0.0));
    }
}
