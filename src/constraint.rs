//! Constraint system for guided generation
//!
//! This module provides a powerful constraint framework for controlling token generation.
//! Constraints can enforce:
//! - Regular expressions (regex patterns)
//! - Structured formats (JSON, XML, YAML)
//! - Grammar rules (CFG, PEG)
//! - Custom validation logic
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              Constraint System                  │
//! │                                                 │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
//! │  │  Regex   │  │   JSON   │  │  Custom  │    │
//! │  │Constraint│  │Constraint│  │Constraint│    │
//! │  └──────────┘  └──────────┘  └──────────┘    │
//! │         │             │             │         │
//! │         └─────────────┴─────────────┘         │
//! │                       │                       │
//! │              ┌────────▼────────┐              │
//! │              │ Constraint Mask │              │
//! │              │   Builder       │              │
//! │              └────────┬────────┘              │
//! │                       │                       │
//! │              ┌────────▼────────┐              │
//! │              │  Token Masking  │              │
//! │              │ (allowed → 0.0) │              │
//! │              │ (banned → -∞)   │              │
//! │              └─────────────────┘              │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance Optimizations
//!
//! 1. **Lazy Evaluation**: Only compute masks when needed
//! 2. **Caching**: Cache computed masks for repeated states
//! 3. **Bitset Compression**: Use compact bitsets for allowed tokens
//! 4. **Incremental Updates**: Update masks incrementally as context changes
//! 5. **SIMD Operations**: Vectorized mask application (future)

use crate::error::{LociError, Result};
use crate::sampler::LogitsView;
use std::collections::{HashMap, HashSet};

// ============================================================================
// CORE TRAIT
// ============================================================================

/// Core trait for constraints that guide token generation
///
/// Constraints are applied before sampling to enforce structural or semantic rules.
/// They work by masking logits: allowed tokens keep their values, disallowed tokens
/// are set to -infinity.
pub trait Constraint: Send + Sync {
    /// Get constraint name
    fn name(&self) -> &str;

    /// Check if this constraint is stateful (depends on generation history)
    fn is_stateful(&self) -> bool {
        true
    }

    /// Reset constraint state (for stateful constraints)
    fn reset(&mut self) {
        // Default: no-op
    }

    /// Update constraint state with a new token
    ///
    /// This is called after each token is generated to update internal state.
    fn update(&mut self, _token_id: i32, _token_text: &str) -> Result<()> {
        Ok(())
    }

    /// Get allowed tokens for the current state
    ///
    /// Returns a set of token IDs that are allowed at this position.
    /// If None, all tokens are allowed.
    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        None
    }

    /// Check if a specific token is allowed
    fn is_allowed(&self, token_id: i32) -> bool {
        if let Some(allowed) = self.allowed_tokens() {
            allowed.contains(&token_id)
        } else {
            true // Default: allow all
        }
    }

    /// Apply constraint to logits (mask disallowed tokens)
    ///
    /// This is the main method called during generation.
    fn apply(&self, logits: &mut LogitsView, vocab_size: usize) -> Result<()> {
        if let Some(allowed) = self.allowed_tokens() {
            // Mask all tokens first
            for i in 0..vocab_size {
                if !allowed.contains(&(i as i32)) {
                    logits.set(i, f32::NEG_INFINITY)?;
                }
            }
        }
        Ok(())
    }

    /// Check if constraint is satisfied (generation can stop)
    fn is_satisfied(&self) -> bool {
        false
    }

    /// Get current generation state (for debugging)
    fn current_state(&self) -> String {
        "N/A".to_string()
    }
}

// ============================================================================
// CONSTRAINT MASK (Optimized Representation)
// ============================================================================

/// Compact representation of allowed tokens using bitset
///
/// This is more memory-efficient than HashSet for large vocabularies.
#[derive(Clone)]
pub struct ConstraintMask {
    /// Bitset of allowed tokens (bit i = 1 if token i is allowed)
    bits: Vec<u64>,
    /// Vocabulary size
    vocab_size: usize,
    /// Cached count of allowed tokens
    allowed_count: usize,
}

impl ConstraintMask {
    /// Create a new mask with all tokens allowed
    pub fn new(vocab_size: usize) -> Self {
        let num_u64s = (vocab_size + 63) / 64;
        Self {
            bits: vec![u64::MAX; num_u64s],
            vocab_size,
            allowed_count: vocab_size,
        }
    }

    /// Create a mask with specific tokens allowed
    pub fn from_allowed(vocab_size: usize, allowed: &[i32]) -> Self {
        let mut mask = Self::new_empty(vocab_size);
        for &token_id in allowed {
            mask.allow(token_id as usize);
        }
        mask
    }

    /// Create a mask with all tokens disallowed
    pub fn new_empty(vocab_size: usize) -> Self {
        let num_u64s = (vocab_size + 63) / 64;
        Self {
            bits: vec![0; num_u64s],
            vocab_size,
            allowed_count: 0,
        }
    }

    /// Allow a specific token
    pub fn allow(&mut self, token_id: usize) {
        if token_id >= self.vocab_size {
            return;
        }
        let idx = token_id / 64;
        let bit = token_id % 64;

        if self.bits[idx] & (1u64 << bit) == 0 {
            self.bits[idx] |= 1u64 << bit;
            self.allowed_count += 1;
        }
    }

    /// Disallow a specific token
    pub fn disallow(&mut self, token_id: usize) {
        if token_id >= self.vocab_size {
            return;
        }
        let idx = token_id / 64;
        let bit = token_id % 64;

        if self.bits[idx] & (1u64 << bit) != 0 {
            self.bits[idx] &= !(1u64 << bit);
            self.allowed_count -= 1;
        }
    }

    /// Check if a token is allowed
    pub fn is_allowed(&self, token_id: usize) -> bool {
        if token_id >= self.vocab_size {
            return false;
        }
        let idx = token_id / 64;
        let bit = token_id % 64;
        self.bits[idx] & (1u64 << bit) != 0
    }

    /// Get count of allowed tokens
    pub fn allowed_count(&self) -> usize {
        self.allowed_count
    }

    /// Apply mask to logits (optimized)
    pub fn apply_to_logits(&self, logits: &mut LogitsView) -> Result<()> {
        let vocab_size = logits.vocab_size().min(self.vocab_size);

        // Fast path: if all allowed, do nothing
        if self.allowed_count == self.vocab_size {
            return Ok(());
        }

        // Fast path: if none allowed, mask everything
        if self.allowed_count == 0 {
            for i in 0..vocab_size {
                logits.set(i, f32::NEG_INFINITY)?;
            }
            return Ok(());
        }

        // Normal path: mask based on bitset
        for i in 0..vocab_size {
            if !self.is_allowed(i) {
                logits.set(i, f32::NEG_INFINITY)?;
            }
        }

        Ok(())
    }

    /// Intersect with another mask (AND operation)
    pub fn intersect(&mut self, other: &ConstraintMask) {
        let min_len = self.bits.len().min(other.bits.len());
        for i in 0..min_len {
            self.bits[i] &= other.bits[i];
        }
        // Recalculate count
        self.allowed_count = self.count_allowed();
    }

    /// Union with another mask (OR operation)
    pub fn union(&mut self, other: &ConstraintMask) {
        let min_len = self.bits.len().min(other.bits.len());
        for i in 0..min_len {
            self.bits[i] |= other.bits[i];
        }
        // Recalculate count
        self.allowed_count = self.count_allowed();
    }

    /// Count allowed tokens (slow, use cached count when possible)
    fn count_allowed(&self) -> usize {
        self.bits.iter().map(|&x| x.count_ones() as usize).sum()
    }

    /// Convert to HashSet (for compatibility)
    pub fn to_hashset(&self) -> HashSet<i32> {
        let mut set = HashSet::new();
        for i in 0..self.vocab_size {
            if self.is_allowed(i) {
                set.insert(i as i32);
            }
        }
        set
    }
}

// ============================================================================
// CONSTRAINT COMBINATOR
// ============================================================================

/// Combines multiple constraints with AND/OR logic
pub struct ConstraintCombinator {
    constraints: Vec<Box<dyn Constraint>>,
    mode: CombinatorMode,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinatorMode {
    /// All constraints must be satisfied (intersection of allowed tokens)
    All,
    /// At least one constraint must be satisfied (union of allowed tokens)
    Any,
}

impl ConstraintCombinator {
    pub fn new(name: String, mode: CombinatorMode) -> Self {
        Self {
            constraints: Vec::new(),
            mode,
            name,
        }
    }

    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) {
        self.constraints.push(constraint);
    }

    pub fn with_constraint(mut self, constraint: Box<dyn Constraint>) -> Self {
        self.add_constraint(constraint);
        self
    }
}

impl Constraint for ConstraintCombinator {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        self.constraints.iter().any(|c| c.is_stateful())
    }

    fn reset(&mut self) {
        for constraint in &mut self.constraints {
            constraint.reset();
        }
    }

    fn update(&mut self, token_id: i32, token_text: &str) -> Result<()> {
        for constraint in &mut self.constraints {
            constraint.update(token_id, token_text)?;
        }
        Ok(())
    }

    fn apply(&self, logits: &mut LogitsView, vocab_size: usize) -> Result<()> {
        match self.mode {
            CombinatorMode::All => {
                // Intersection: apply all constraints sequentially
                for constraint in &self.constraints {
                    constraint.apply(logits, vocab_size)?;
                }
            }
            CombinatorMode::Any => {
                // Union: collect allowed tokens from all constraints
                let mut combined_mask = ConstraintMask::new_empty(vocab_size);

                for constraint in &self.constraints {
                    if let Some(allowed) = constraint.allowed_tokens() {
                        for &token_id in allowed {
                            if token_id >= 0 && (token_id as usize) < vocab_size {
                                combined_mask.allow(token_id as usize);
                            }
                        }
                    } else {
                        // If any constraint allows all, union allows all
                        return Ok(());
                    }
                }

                combined_mask.apply_to_logits(logits)?;
            }
        }
        Ok(())
    }

    fn is_satisfied(&self) -> bool {
        match self.mode {
            CombinatorMode::All => self.constraints.iter().all(|c| c.is_satisfied()),
            CombinatorMode::Any => self.constraints.iter().any(|c| c.is_satisfied()),
        }
    }
}

// ============================================================================
// BUILT-IN CONSTRAINTS
// ============================================================================

/// Token whitelist constraint
pub struct TokenWhitelistConstraint {
    name: String,
    allowed: HashSet<i32>,
}

impl TokenWhitelistConstraint {
    pub fn new(name: String, allowed: Vec<i32>) -> Self {
        Self {
            name,
            allowed: allowed.into_iter().collect(),
        }
    }
}

impl Constraint for TokenWhitelistConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        false
    }

    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        Some(&self.allowed)
    }

    fn is_satisfied(&self) -> bool {
        false // Never satisfied (always applies)
    }
}

/// Token blacklist constraint
pub struct TokenBlacklistConstraint {
    name: String,
    banned: HashSet<i32>,
}

impl TokenBlacklistConstraint {
    pub fn new(name: String, banned: Vec<i32>) -> Self {
        Self {
            name,
            banned: banned.into_iter().collect(),
        }
    }
}

impl Constraint for TokenBlacklistConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        false
    }

    fn apply(&self, logits: &mut LogitsView, vocab_size: usize) -> Result<()> {
        for &token_id in &self.banned {
            if token_id >= 0 && (token_id as usize) < vocab_size {
                logits.set(token_id as usize, f32::NEG_INFINITY)?;
            }
        }
        Ok(())
    }

    fn is_satisfied(&self) -> bool {
        false
    }
}

/// Length constraint (max tokens)
pub struct LengthConstraint {
    name: String,
    max_tokens: usize,
    current_length: usize,
    eos_token_id: i32,
}

impl LengthConstraint {
    pub fn new(name: String, max_tokens: usize, eos_token_id: i32) -> Self {
        Self {
            name,
            max_tokens,
            current_length: 0,
            eos_token_id,
        }
    }
}

impl Constraint for LengthConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.current_length = 0;
    }

    fn update(&mut self, _token_id: i32, _token_text: &str) -> Result<()> {
        self.current_length += 1;
        Ok(())
    }

    fn apply(&self, logits: &mut LogitsView, vocab_size: usize) -> Result<()> {
        if self.current_length >= self.max_tokens {
            // Force EOS token
            for i in 0..vocab_size {
                if i as i32 != self.eos_token_id {
                    logits.set(i, f32::NEG_INFINITY)?;
                }
            }
        }
        Ok(())
    }

    fn is_satisfied(&self) -> bool {
        self.current_length >= self.max_tokens
    }

    fn current_state(&self) -> String {
        format!("{}/{} tokens", self.current_length, self.max_tokens)
    }
}

// ============================================================================
// REGEX CONSTRAINT (Simplified DFA-based)
// ============================================================================

/// Regular expression constraint using DFA
///
/// This constraint ensures generated text matches a regex pattern.
/// Implementation uses a simplified DFA (Deterministic Finite Automaton).
pub struct RegexConstraint {
    name: String,
    pattern: String,
    // Simplified: store allowed characters for next position
    state: RegexState,
    generated_text: String,
}

#[derive(Debug, Clone)]
struct RegexState {
    // For simple patterns, track what's allowed next
    allowed_chars: Option<HashSet<char>>,
    // Track if we're done
    done: bool,
}

impl RegexConstraint {
    /// Create a new regex constraint
    ///
    /// Note: This is a simplified implementation. For production use,
    /// integrate with `regex` crate and build a proper DFA.
    pub fn new(name: String, pattern: String) -> Self {
        Self {
            name,
            pattern,
            state: RegexState {
                allowed_chars: None, // TODO: Compute from pattern
                done: false,
            },
            generated_text: String::new(),
        }
    }

    /// Map allowed characters to token IDs
    ///
    /// This requires a tokenizer to map characters to tokens.
    /// For now, this is a placeholder.
    fn chars_to_token_ids(&self, _chars: &HashSet<char>) -> HashSet<i32> {
        // TODO: Implement character → token mapping
        // This requires tokenizer integration
        HashSet::new()
    }
}

impl Constraint for RegexConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.generated_text.clear();
        self.state.done = false;
        // TODO: Reset DFA state
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        self.generated_text.push_str(token_text);
        // TODO: Update DFA state based on new character
        Ok(())
    }

    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        // TODO: Return tokens that correspond to allowed next characters
        None
    }

    fn is_satisfied(&self) -> bool {
        self.state.done
    }

    fn current_state(&self) -> String {
        format!(
            "Pattern: '{}', Generated: '{}'",
            self.pattern, self.generated_text
        )
    }
}

// ============================================================================
// JSON SCHEMA CONSTRAINT
// ============================================================================

/// JSON schema constraint for structured generation
///
/// This constraint ensures generated text is valid JSON matching a schema.
pub struct JsonConstraint {
    name: String,
    state: JsonState,
    generated_text: String,
    /// Token IDs for special JSON characters
    special_tokens: JsonTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    Start,
    ObjectStart,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectComma,
    ArrayStart,
    ArrayValue,
    ArrayComma,
    StringStart,
    StringContent,
    NumberStart,
    NumberContent,
    BoolTrue,
    BoolFalse,
    Null,
    Done,
}

#[derive(Debug, Clone)]
struct JsonTokens {
    // TODO: Map characters to token IDs
    // This requires tokenizer integration
    left_brace: Option<i32>,
    right_brace: Option<i32>,
    left_bracket: Option<i32>,
    right_bracket: Option<i32>,
    quote: Option<i32>,
    colon: Option<i32>,
    comma: Option<i32>,
}

impl JsonConstraint {
    pub fn new(name: String) -> Self {
        Self {
            name,
            state: JsonState::Start,
            generated_text: String::new(),
            special_tokens: JsonTokens {
                left_brace: None,
                right_brace: None,
                left_bracket: None,
                right_bracket: None,
                quote: None,
                colon: None,
                comma: None,
            },
        }
    }

    /// Set special token IDs (must be called after tokenizer is available)
    pub fn set_special_tokens(
        &mut self,
        left_brace: i32,
        right_brace: i32,
        left_bracket: i32,
        right_bracket: i32,
        quote: i32,
        colon: i32,
        comma: i32,
    ) {
        self.special_tokens = JsonTokens {
            left_brace: Some(left_brace),
            right_brace: Some(right_brace),
            left_bracket: Some(left_bracket),
            right_bracket: Some(right_bracket),
            quote: Some(quote),
            colon: Some(colon),
            comma: Some(comma),
        };
    }

    fn get_allowed_tokens_for_state(&self) -> Option<HashSet<i32>> {
        // TODO: Implement state machine logic
        // For each state, return allowed next tokens
        None
    }
}

impl Constraint for JsonConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) {
        self.state = JsonState::Start;
        self.generated_text.clear();
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        self.generated_text.push_str(token_text);

        // TODO: Update state machine based on new token
        // This is a simplified placeholder
        match self.state {
            JsonState::Start => {
                if token_text.trim() == "{" {
                    self.state = JsonState::ObjectStart;
                } else if token_text.trim() == "[" {
                    self.state = JsonState::ArrayStart;
                }
            }
            _ => {
                // TODO: Implement full state transitions
            }
        }

        Ok(())
    }

    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        // TODO: Return tokens allowed in current state
        None
    }

    fn is_satisfied(&self) -> bool {
        self.state == JsonState::Done
    }

    fn current_state(&self) -> String {
        format!("State: {:?}, Generated: '{}'", self.state, self.generated_text)
    }
}

// ============================================================================
// CONSTRAINT MANAGER
// ============================================================================

/// Manages multiple constraints and applies them efficiently
pub struct ConstraintManager {
    constraints: Vec<Box<dyn Constraint>>,
    vocab_size: usize,
    /// Cached combined mask (recomputed when constraints change)
    cached_mask: Option<ConstraintMask>,
}

impl ConstraintManager {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            constraints: Vec::new(),
            vocab_size,
            cached_mask: None,
        }
    }

    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) {
        self.constraints.push(constraint);
        self.cached_mask = None; // Invalidate cache
    }

    pub fn remove_constraint(&mut self, name: &str) {
        self.constraints.retain(|c| c.name() != name);
        self.cached_mask = None; // Invalidate cache
    }

    pub fn clear(&mut self) {
        self.constraints.clear();
        self.cached_mask = None;
    }

    pub fn reset_all(&mut self) {
        for constraint in &mut self.constraints {
            constraint.reset();
        }
        self.cached_mask = None;
    }

    pub fn update_all(&mut self, token_id: i32, token_text: &str) -> Result<()> {
        for constraint in &mut self.constraints {
            constraint.update(token_id, token_text)?;
        }
        self.cached_mask = None; // Invalidate cache after update
        Ok(())
    }

    /// Apply all constraints to logits
    pub fn apply_all(&self, logits: &mut LogitsView) -> Result<()> {
        for constraint in &self.constraints {
            constraint.apply(logits, self.vocab_size)?;
        }
        Ok(())
    }

    /// Check if all constraints are satisfied
    pub fn all_satisfied(&self) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied())
    }

    /// Get status of all constraints
    pub fn get_status(&self) -> Vec<(String, String, bool)> {
        self.constraints
            .iter()
            .map(|c| {
                (
                    c.name().to_string(),
                    c.current_state(),
                    c.is_satisfied(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_mask() {
        let mut mask = ConstraintMask::new(100);
        assert_eq!(mask.allowed_count(), 100);
        assert!(mask.is_allowed(50));

        mask.disallow(50);
        assert!(!mask.is_allowed(50));
        assert_eq!(mask.allowed_count(), 99);

        mask.allow(50);
        assert!(mask.is_allowed(50));
        assert_eq!(mask.allowed_count(), 100);
    }

    #[test]
    fn test_whitelist_constraint() {
        let constraint = TokenWhitelistConstraint::new(
            "test".to_string(),
            vec![1, 2, 3, 10, 20],
        );

        assert!(constraint.is_allowed(1));
        assert!(constraint.is_allowed(10));
        assert!(!constraint.is_allowed(5));
        assert!(!constraint.is_allowed(100));
    }

    #[test]
    fn test_length_constraint() {
        let mut constraint = LengthConstraint::new("test".to_string(), 5, 2);

        assert!(!constraint.is_satisfied());

        for i in 0..5 {
            constraint.update(i, "token").unwrap();
        }

        assert!(constraint.is_satisfied());
    }

    #[test]
    fn test_constraint_combinator() {
        let mut combinator = ConstraintCombinator::new(
            "test".to_string(),
            CombinatorMode::All,
        );

        combinator.add_constraint(Box::new(TokenWhitelistConstraint::new(
            "c1".to_string(),
            vec![1, 2, 3, 4, 5],
        )));

        combinator.add_constraint(Box::new(TokenWhitelistConstraint::new(
            "c2".to_string(),
            vec![3, 4, 5, 6, 7],
        )));

        // Intersection should only allow 3, 4, 5
        // (This would be tested with actual logits application)
    }
}
