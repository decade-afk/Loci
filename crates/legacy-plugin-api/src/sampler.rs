use crate::error::{LociError, Result};
use std::cmp::Ordering;

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
}
