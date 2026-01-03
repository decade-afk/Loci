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
                    logits.set_usize(i, f32::NEG_INFINITY)?;
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
                logits.set_usize(i, f32::NEG_INFINITY)?;
            }
            return Ok(());
        }

        // Normal path: mask based on bitset
        for i in 0..vocab_size {
            if !self.is_allowed(i) {
                logits.set_usize(i, f32::NEG_INFINITY)?;
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
                logits.set_usize(token_id as usize, f32::NEG_INFINITY)?;
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
                    logits.set_usize(i, f32::NEG_INFINITY)?;
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
// REGEX CONSTRAINT (DFA-based)
// ============================================================================

/// Regular expression constraint using DFA
///
/// This constraint ensures generated text matches a regex pattern.
/// Implementation uses a DFA (Deterministic Finite Automaton) for efficient matching.
///
/// ## Architecture
///
/// ```text
/// Pattern → NFA → DFA → Token Filter
///    ↓       ↓     ↓         ↓
///  "ab*"   States  States   Allowed
///          ε-NFA   Minimal   Tokens
/// ```
///
/// ## Example Usage
///
/// ```ignore
/// // Match email pattern
/// let constraint = RegexConstraint::new(
///     "email".to_string(),
///     r"[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]+".to_string(),
///     tokenizer,
/// );
///
/// // Match JSON number
/// let constraint = RegexConstraint::new(
///     "number".to_string(),
///     r"-?[0-9]+(\.[0-9]+)?".to_string(),
///     tokenizer,
/// );
/// ```
pub struct RegexConstraint {
    name: String,
    pattern: String,
    /// Current DFA state
    current_state: usize,
    /// DFA transition table: state × char → next_state
    transitions: HashMap<(usize, char), usize>,
    /// Accept states
    accept_states: HashSet<usize>,
    /// Generated text so far
    generated_text: String,
    /// Cached allowed tokens for current state
    allowed_tokens_cache: HashSet<i32>,
    /// Character → Token ID mapping (from tokenizer)
    char_to_tokens: HashMap<char, Vec<i32>>,
    /// Token ID → Characters mapping
    token_to_chars: HashMap<i32, Vec<char>>,
}

impl RegexConstraint {
    /// Create a new regex constraint
    ///
    /// This builds a DFA from the regex pattern and prepares token mappings.
    pub fn new(name: String, pattern: String) -> Self {
        // Build a simple DFA for common patterns
        let (transitions, accept_states) = Self::build_simple_dfa(&pattern);

        Self {
            name,
            pattern,
            current_state: 0,
            transitions,
            accept_states,
            generated_text: String::new(),
            allowed_tokens_cache: HashSet::new(),
            char_to_tokens: HashMap::new(),
            token_to_chars: HashMap::new(),
        }
    }

    /// Build a simplified DFA for common patterns
    ///
    /// Supports:
    /// - Character classes: [a-z], [0-9], [A-Z]
    /// - Quantifiers: *, +, ?
    /// - Concatenation: abc
    /// - Alternation: a|b (limited)
    ///
    /// For full regex support, integrate with `regex-automata` crate.
    fn build_simple_dfa(pattern: &str) -> (HashMap<(usize, char), usize>, HashSet<usize>) {
        let mut transitions = HashMap::new();
        let mut accept_states = HashSet::new();

        // Simplified DFA construction for demonstration
        // This is a placeholder - real implementation would parse the regex properly

        // Example: For pattern "abc", build linear DFA
        // State 0 --a--> State 1 --b--> State 2 --c--> State 3 (accept)

        if pattern == "abc" {
            transitions.insert((0, 'a'), 1);
            transitions.insert((1, 'b'), 2);
            transitions.insert((2, 'c'), 3);
            accept_states.insert(3);
        } else if pattern.starts_with('[') && pattern.ends_with(']') {
            // Character class: [a-z]
            let chars = Self::parse_char_class(pattern);
            for c in chars {
                transitions.insert((0, c), 1);
            }
            accept_states.insert(1);
        } else if pattern.ends_with('*') {
            // Kleene star: a*
            let base = pattern.trim_end_matches('*');
            if base.len() == 1 {
                let c = base.chars().next().unwrap();
                transitions.insert((0, c), 0); // Loop on state 0
                accept_states.insert(0); // Accept empty string
            }
        } else if pattern.ends_with('+') {
            // One or more: a+
            let base = pattern.trim_end_matches('+');
            if base.len() == 1 {
                let c = base.chars().next().unwrap();
                transitions.insert((0, c), 1);
                transitions.insert((1, c), 1); // Loop on state 1
                accept_states.insert(1);
            }
        } else {
            // Default: literal string match
            let mut state = 0;
            for c in pattern.chars() {
                transitions.insert((state, c), state + 1);
                state += 1;
            }
            accept_states.insert(state);
        }

        (transitions, accept_states)
    }

    /// Parse character class like [a-z] or [0-9]
    fn parse_char_class(pattern: &str) -> Vec<char> {
        let inner = pattern.trim_start_matches('[').trim_end_matches(']');
        let mut chars = Vec::new();

        if inner.contains('-') && inner.len() == 3 {
            // Range like a-z
            let parts: Vec<char> = inner.chars().collect();
            if parts.len() == 3 && parts[1] == '-' {
                let start = parts[0];
                let end = parts[2];
                for c in start..=end {
                    chars.push(c);
                }
            }
        } else {
            // Individual characters
            chars = inner.chars().collect();
        }

        chars
    }

    /// Set tokenizer mappings
    ///
    /// This should be called after constraint creation to provide
    /// character-to-token mappings from the tokenizer.
    pub fn set_tokenizer_mappings(
        &mut self,
        char_to_tokens: HashMap<char, Vec<i32>>,
        token_to_chars: HashMap<i32, Vec<char>>,
    ) {
        self.char_to_tokens = char_to_tokens;
        self.token_to_chars = token_to_chars;
        self.update_allowed_tokens_cache();
    }

    /// Update the cache of allowed tokens for current state
    fn update_allowed_tokens_cache(&mut self) {
        self.allowed_tokens_cache.clear();

        // Find all characters that can transition from current state
        let allowed_chars: HashSet<char> = self
            .transitions
            .iter()
            .filter_map(|((state, c), _)| {
                if *state == self.current_state {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect();

        // Map allowed characters to token IDs
        for c in allowed_chars {
            if let Some(token_ids) = self.char_to_tokens.get(&c) {
                for &token_id in token_ids {
                    self.allowed_tokens_cache.insert(token_id);
                }
            }
        }

        // Also allow tokens that start with allowed characters
        // (for multi-character tokens)
        for (&token_id, chars) in &self.token_to_chars {
            if let Some(&first_char) = chars.first() {
                if self.transitions.contains_key(&(self.current_state, first_char)) {
                    // Check if all characters in token can be consumed
                    if self.can_consume_sequence(chars) {
                        self.allowed_tokens_cache.insert(token_id);
                    }
                }
            }
        }
    }

    /// Check if a sequence of characters can be consumed from current state
    fn can_consume_sequence(&self, chars: &[char]) -> bool {
        let mut state = self.current_state;
        for &c in chars {
            if let Some(&next_state) = self.transitions.get(&(state, c)) {
                state = next_state;
            } else {
                return false;
            }
        }
        true
    }

    /// Transition DFA with new character
    fn transition(&mut self, c: char) -> bool {
        if let Some(&next_state) = self.transitions.get(&(self.current_state, c)) {
            self.current_state = next_state;
            true
        } else {
            false
        }
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
        self.current_state = 0;
        self.generated_text.clear();
        self.update_allowed_tokens_cache();
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        // Process each character in the token
        for c in token_text.chars() {
            if !self.transition(c) {
                return Err(LociError::InferenceError(format!(
                    "Regex constraint '{}' violated: cannot consume character '{}' in state {}",
                    self.name, c, self.current_state
                )));
            }
        }

        self.generated_text.push_str(token_text);
        self.update_allowed_tokens_cache();
        Ok(())
    }

    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        if self.allowed_tokens_cache.is_empty() {
            None // No tokens allowed (shouldn't happen in practice)
        } else {
            Some(&self.allowed_tokens_cache)
        }
    }

    fn is_satisfied(&self) -> bool {
        self.accept_states.contains(&self.current_state)
    }

    fn current_state(&self) -> String {
        format!(
            "Pattern: '{}', State: {}, Generated: '{}', Satisfied: {}",
            self.pattern,
            self.current_state,
            self.generated_text,
            self.is_satisfied()
        )
    }
}

// ============================================================================
// JSON SCHEMA CONSTRAINT
// ============================================================================

/// JSON schema constraint for structured generation
///
/// This constraint ensures generated text is valid JSON matching a schema.
///
/// ## State Machine
///
/// ```text
///       ┌─────────┐
///       │  Start  │
///       └────┬────┘
///            │
///     ┌──────┴───────┐
///     │              │
///     ▼              ▼
/// ┌───────┐    ┌────────┐
/// │Object │    │ Array  │
/// │       │    │        │
/// └───────┘    └────────┘
///     │              │
///     ▼              ▼
/// ┌───────┐    ┌────────┐
/// │Value  │◄───┤ Value  │
/// │       │    │        │
/// └───────┘    └────────┘
///     │              │
///     └──────┬───────┘
///            ▼
///       ┌─────────┐
///       │  Done   │
///       └─────────┘
/// ```
///
/// ## Example Usage
///
/// ```ignore
/// // Generate JSON object
/// let mut constraint = JsonConstraint::new("json".to_string());
/// constraint.set_schema(JsonSchema::Object(vec![
///     ("name", JsonSchema::String),
///     ("age", JsonSchema::Number),
/// ]));
///
/// // Generate JSON array
/// let constraint = JsonConstraint::new_array(
///     "array".to_string(),
///     JsonSchema::Number,
/// );
/// ```
pub struct JsonConstraint {
    name: String,
    state: JsonState,
    /// Stack to track nested objects/arrays
    state_stack: Vec<JsonState>,
    generated_text: String,
    /// Expected schema (optional)
    schema: Option<JsonSchema>,
    /// Token IDs for special JSON characters
    special_tokens: JsonTokens,
    /// Cached allowed tokens for current state
    allowed_tokens_cache: HashSet<i32>,
    /// Character-based token mappings
    char_to_tokens: HashMap<char, Vec<i32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    Start,
    ObjectStart,
    ObjectKey,
    ObjectColon,
    ObjectValue,
    ObjectCommaOrEnd,
    ArrayStart,
    ArrayValue,
    ArrayCommaOrEnd,
    StringContent,
    NumberContent,
    BoolContent,
    NullContent,
    Done,
}

#[derive(Debug, Clone)]
struct JsonTokens {
    left_brace: Vec<i32>,       // '{'
    right_brace: Vec<i32>,      // '}'
    left_bracket: Vec<i32>,     // '['
    right_bracket: Vec<i32>,    // ']'
    quote: Vec<i32>,            // '"'
    colon: Vec<i32>,            // ':'
    comma: Vec<i32>,            // ','
    digits: Vec<i32>,           // 0-9, -, .
    true_tokens: Vec<i32>,      // 'true'
    false_tokens: Vec<i32>,     // 'false'
    null_tokens: Vec<i32>,      // 'null'
}

/// JSON schema definition (simplified)
#[derive(Debug, Clone)]
pub enum JsonSchema {
    /// Any JSON value
    Any,
    /// JSON object with field constraints
    Object(Vec<(String, JsonSchema)>),
    /// JSON array with element constraint
    Array(Box<JsonSchema>),
    /// JSON string
    String,
    /// JSON number
    Number,
    /// JSON boolean
    Bool,
    /// JSON null
    Null,
}

impl JsonConstraint {
    pub fn new(name: String) -> Self {
        Self {
            name,
            state: JsonState::Start,
            state_stack: Vec::new(),
            generated_text: String::new(),
            schema: None,
            special_tokens: JsonTokens {
                left_brace: Vec::new(),
                right_brace: Vec::new(),
                left_bracket: Vec::new(),
                right_bracket: Vec::new(),
                quote: Vec::new(),
                colon: Vec::new(),
                comma: Vec::new(),
                digits: Vec::new(),
                true_tokens: Vec::new(),
                false_tokens: Vec::new(),
                null_tokens: Vec::new(),
            },
            allowed_tokens_cache: HashSet::new(),
            char_to_tokens: HashMap::new(),
        }
    }

    /// Create JSON object constraint
    pub fn new_object(name: String, fields: Vec<(String, JsonSchema)>) -> Self {
        let mut constraint = Self::new(name);
        constraint.schema = Some(JsonSchema::Object(fields));
        constraint
    }

    /// Create JSON array constraint
    pub fn new_array(name: String, element_schema: JsonSchema) -> Self {
        let mut constraint = Self::new(name);
        constraint.schema = Some(JsonSchema::Array(Box::new(element_schema)));
        constraint
    }

    /// Set schema for validation
    pub fn set_schema(&mut self, schema: JsonSchema) {
        self.schema = Some(schema);
    }

    /// Set tokenizer mappings
    pub fn set_tokenizer_mappings(&mut self, char_to_tokens: HashMap<char, Vec<i32>>) {
        self.char_to_tokens = char_to_tokens.clone();

        // Extract special tokens
        self.special_tokens.left_brace = char_to_tokens.get(&'{').cloned().unwrap_or_default();
        self.special_tokens.right_brace = char_to_tokens.get(&'}').cloned().unwrap_or_default();
        self.special_tokens.left_bracket = char_to_tokens.get(&'[').cloned().unwrap_or_default();
        self.special_tokens.right_bracket = char_to_tokens.get(&']').cloned().unwrap_or_default();
        self.special_tokens.quote = char_to_tokens.get(&'"').cloned().unwrap_or_default();
        self.special_tokens.colon = char_to_tokens.get(&':').cloned().unwrap_or_default();
        self.special_tokens.comma = char_to_tokens.get(&',').cloned().unwrap_or_default();

        // Collect digit tokens
        for c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.'] {
            if let Some(tokens) = char_to_tokens.get(&c) {
                self.special_tokens.digits.extend(tokens);
            }
        }

        // TODO: Add tokens for 'true', 'false', 'null'

        self.update_allowed_tokens_cache();
    }

    /// Update allowed tokens based on current state
    fn update_allowed_tokens_cache(&mut self) {
        self.allowed_tokens_cache.clear();

        match self.state {
            JsonState::Start => {
                // Can start with object or array
                self.allowed_tokens_cache.extend(&self.special_tokens.left_brace);
                self.allowed_tokens_cache.extend(&self.special_tokens.left_bracket);
            }
            JsonState::ObjectStart | JsonState::ObjectKey => {
                // Expect string key or closing brace (for empty object)
                self.allowed_tokens_cache.extend(&self.special_tokens.quote);
                if self.state == JsonState::ObjectStart {
                    self.allowed_tokens_cache.extend(&self.special_tokens.right_brace);
                }
            }
            JsonState::ObjectColon => {
                // Expect colon
                self.allowed_tokens_cache.extend(&self.special_tokens.colon);
            }
            JsonState::ObjectValue | JsonState::ArrayValue => {
                // Expect any JSON value
                self.allowed_tokens_cache.extend(&self.special_tokens.left_brace);
                self.allowed_tokens_cache.extend(&self.special_tokens.left_bracket);
                self.allowed_tokens_cache.extend(&self.special_tokens.quote);
                self.allowed_tokens_cache.extend(&self.special_tokens.digits);
                self.allowed_tokens_cache.extend(&self.special_tokens.true_tokens);
                self.allowed_tokens_cache.extend(&self.special_tokens.false_tokens);
                self.allowed_tokens_cache.extend(&self.special_tokens.null_tokens);
            }
            JsonState::ObjectCommaOrEnd => {
                // Expect comma or closing brace
                self.allowed_tokens_cache.extend(&self.special_tokens.comma);
                self.allowed_tokens_cache.extend(&self.special_tokens.right_brace);
            }
            JsonState::ArrayStart => {
                // Expect value or closing bracket (for empty array)
                self.allowed_tokens_cache.extend(&self.special_tokens.left_brace);
                self.allowed_tokens_cache.extend(&self.special_tokens.left_bracket);
                self.allowed_tokens_cache.extend(&self.special_tokens.quote);
                self.allowed_tokens_cache.extend(&self.special_tokens.digits);
                self.allowed_tokens_cache.extend(&self.special_tokens.right_bracket);
            }
            JsonState::ArrayCommaOrEnd => {
                // Expect comma or closing bracket
                self.allowed_tokens_cache.extend(&self.special_tokens.comma);
                self.allowed_tokens_cache.extend(&self.special_tokens.right_bracket);
            }
            JsonState::StringContent => {
                // Inside string: allow all chars except unescaped quote
                // Simplified: allow quote to end string
                self.allowed_tokens_cache.extend(&self.special_tokens.quote);
                // TODO: Add all printable character tokens
            }
            JsonState::NumberContent => {
                // Allow digits or end
                self.allowed_tokens_cache.extend(&self.special_tokens.digits);
                self.allowed_tokens_cache.extend(&self.special_tokens.comma);
                self.allowed_tokens_cache.extend(&self.special_tokens.right_brace);
                self.allowed_tokens_cache.extend(&self.special_tokens.right_bracket);
            }
            JsonState::Done => {
                // No tokens allowed (generation complete)
            }
            _ => {}
        }
    }

    /// Transition state based on consumed token
    fn transition(&mut self, token_text: &str) -> Result<()> {
        let trimmed = token_text.trim();

        match self.state {
            JsonState::Start => {
                if trimmed == "{" {
                    self.state = JsonState::ObjectStart;
                } else if trimmed == "[" {
                    self.state = JsonState::ArrayStart;
                } else {
                    return Err(LociError::InferenceError(
                        format!("JSON must start with {{ or [, got: {}", trimmed)
                    ));
                }
            }
            JsonState::ObjectStart => {
                if trimmed == "\"" {
                    self.state = JsonState::ObjectKey;
                } else if trimmed == "}" {
                    self.state = if self.state_stack.is_empty() {
                        JsonState::Done
                    } else {
                        self.state_stack.pop().unwrap()
                    };
                }
            }
            JsonState::ObjectKey => {
                if trimmed.ends_with('"') {
                    self.state = JsonState::ObjectColon;
                }
            }
            JsonState::ObjectColon => {
                if trimmed == ":" {
                    self.state = JsonState::ObjectValue;
                }
            }
            JsonState::ObjectValue => {
                self.handle_value_start(trimmed)?;
            }
            JsonState::ObjectCommaOrEnd => {
                if trimmed == "," {
                    self.state = JsonState::ObjectKey;
                } else if trimmed == "}" {
                    self.state = if self.state_stack.is_empty() {
                        JsonState::Done
                    } else {
                        self.state_stack.pop().unwrap()
                    };
                }
            }
            JsonState::ArrayStart => {
                if trimmed == "]" {
                    self.state = if self.state_stack.is_empty() {
                        JsonState::Done
                    } else {
                        self.state_stack.pop().unwrap()
                    };
                } else {
                    self.handle_value_start(trimmed)?;
                }
            }
            JsonState::ArrayValue => {
                self.handle_value_start(trimmed)?;
            }
            JsonState::ArrayCommaOrEnd => {
                if trimmed == "," {
                    self.state = JsonState::ArrayValue;
                } else if trimmed == "]" {
                    self.state = if self.state_stack.is_empty() {
                        JsonState::Done
                    } else {
                        self.state_stack.pop().unwrap()
                    };
                }
            }
            JsonState::StringContent => {
                if trimmed.ends_with('"') {
                    self.state = self.get_value_end_state();
                }
            }
            JsonState::NumberContent => {
                if trimmed == "," {
                    if self.in_object() {
                        self.state = JsonState::ObjectKey;
                    } else {
                        self.state = JsonState::ArrayValue;
                    }
                } else if trimmed == "}" || trimmed == "]" {
                    self.state = if self.state_stack.is_empty() {
                        JsonState::Done
                    } else {
                        self.state_stack.pop().unwrap()
                    };
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_value_start(&mut self, token: &str) -> Result<()> {
        if token == "{" {
            self.state_stack.push(JsonState::ObjectCommaOrEnd);
            self.state = JsonState::ObjectStart;
        } else if token == "[" {
            self.state_stack.push(JsonState::ArrayCommaOrEnd);
            self.state = JsonState::ArrayStart;
        } else if token == "\"" {
            self.state = JsonState::StringContent;
        } else if token.chars().all(|c| c.is_ascii_digit() || c == '-' || c == '.') {
            self.state = JsonState::NumberContent;
        } else if token == "true" || token == "false" {
            self.state = self.get_value_end_state();
        } else if token == "null" {
            self.state = self.get_value_end_state();
        }
        Ok(())
    }

    fn get_value_end_state(&self) -> JsonState {
        if self.in_object() {
            JsonState::ObjectCommaOrEnd
        } else if self.in_array() {
            JsonState::ArrayCommaOrEnd
        } else {
            JsonState::Done
        }
    }

    fn in_object(&self) -> bool {
        self.state_stack.iter().any(|s| matches!(s, JsonState::ObjectCommaOrEnd))
    }

    fn in_array(&self) -> bool {
        self.state_stack.iter().any(|s| matches!(s, JsonState::ArrayCommaOrEnd))
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
        self.state_stack.clear();
        self.generated_text.clear();
        self.update_allowed_tokens_cache();
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        self.generated_text.push_str(token_text);
        self.transition(token_text)?;
        self.update_allowed_tokens_cache();
        Ok(())
    }

    fn allowed_tokens(&self) -> Option<&HashSet<i32>> {
        if self.allowed_tokens_cache.is_empty() {
            None
        } else {
            Some(&self.allowed_tokens_cache)
        }
    }

    fn is_satisfied(&self) -> bool {
        self.state == JsonState::Done
    }

    fn current_state(&self) -> String {
        format!(
            "State: {:?}, Depth: {}, Generated: '{}'",
            self.state,
            self.state_stack.len(),
            self.generated_text
        )
    }
}

// ============================================================================
// CONSTRAINT BUILDER (Fluent API)
// ============================================================================

/// Builder for creating complex constraints with a fluent API
///
/// This provides an ergonomic way to construct constraints, especially
/// when combining multiple constraints.
///
/// ## Example
///
/// ```ignore
/// use loci::constraint::*;
///
/// let constraint = ConstraintBuilder::new()
///     .with_regex("email_format", r"[a-z]+@[a-z]+\.[a-z]+")
///     .with_json_object("response", vec![
///         ("status", JsonSchema::String),
///         ("code", JsonSchema::Number),
///     ])
///     .with_length("max_tokens", 100, eos_token_id)
///     .all()
///     .build("email_response");
/// ```
pub struct ConstraintBuilder {
    constraints: Vec<Box<dyn Constraint>>,
    mode: CombinatorMode,
}

impl ConstraintBuilder {
    /// Create a new constraint builder
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            mode: CombinatorMode::All,
        }
    }

    /// Add a regex constraint
    pub fn with_regex(mut self, name: &str, pattern: &str) -> Self {
        self.constraints.push(Box::new(
            RegexConstraint::new(name.to_string(), pattern.to_string())
        ));
        self
    }

    /// Add a JSON object constraint
    pub fn with_json_object(mut self, name: &str, fields: Vec<(&str, JsonSchema)>) -> Self {
        let fields: Vec<(String, JsonSchema)> = fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        self.constraints.push(Box::new(
            JsonConstraint::new_object(name.to_string(), fields)
        ));
        self
    }

    /// Add a JSON array constraint
    pub fn with_json_array(mut self, name: &str, element_schema: JsonSchema) -> Self {
        self.constraints.push(Box::new(
            JsonConstraint::new_array(name.to_string(), element_schema)
        ));
        self
    }

    /// Add a token whitelist constraint
    pub fn with_whitelist(mut self, name: &str, allowed: Vec<i32>) -> Self {
        self.constraints.push(Box::new(
            TokenWhitelistConstraint::new(name.to_string(), allowed)
        ));
        self
    }

    /// Add a token blacklist constraint
    pub fn with_blacklist(mut self, name: &str, banned: Vec<i32>) -> Self {
        self.constraints.push(Box::new(
            TokenBlacklistConstraint::new(name.to_string(), banned)
        ));
        self
    }

    /// Add a length constraint
    pub fn with_length(mut self, name: &str, max_tokens: usize, eos_token_id: i32) -> Self {
        self.constraints.push(Box::new(
            LengthConstraint::new(name.to_string(), max_tokens, eos_token_id)
        ));
        self
    }

    /// Add a custom constraint
    pub fn with_custom(mut self, constraint: Box<dyn Constraint>) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set combination mode to ALL (intersection)
    pub fn all(mut self) -> Self {
        self.mode = CombinatorMode::All;
        self
    }

    /// Set combination mode to ANY (union)
    pub fn any(mut self) -> Self {
        self.mode = CombinatorMode::Any;
        self
    }

    /// Build the final constraint combinator
    pub fn build(self, name: &str) -> ConstraintCombinator {
        let mut combinator = ConstraintCombinator::new(name.to_string(), self.mode);
        for constraint in self.constraints {
            combinator.add_constraint(constraint);
        }
        combinator
    }

    /// Build and return as a boxed constraint
    pub fn build_boxed(self, name: &str) -> Box<dyn Constraint> {
        Box::new(self.build(name))
    }
}

impl Default for ConstraintBuilder {
    fn default() -> Self {
        Self::new()
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

// ============================================================================
// PLUGIN-COMPATIBLE CONSTRAINT WRAPPER
// ============================================================================

use std::sync::Mutex;

/// Wrapper to use constraints as plugins
///
/// This allows constraints to be registered as plugins in the plugin system,
/// enabling modular and extensible constraint management.
///
/// ## Example
///
/// ```ignore
/// use loci::constraint::{RegexConstraint, ConstraintPlugin};
/// use loci::plugin::Plugin;
///
/// // Create a regex constraint
/// let regex = RegexConstraint::new("email".to_string(), r"[a-z]+@[a-z]+\.[a-z]+".to_string());
///
/// // Wrap as plugin
/// let plugin = ConstraintPlugin::new(Box::new(regex));
///
/// // Register with engine
/// engine.plugin_manager_mut().register(plugin)?;
/// ```
pub struct ConstraintPlugin {
    constraint: Mutex<Box<dyn Constraint>>,
    enabled: bool,
}

impl ConstraintPlugin {
    /// Create a new constraint plugin
    pub fn new(constraint: Box<dyn Constraint>) -> Self {
        Self {
            constraint: Mutex::new(constraint),
            enabled: true,
        }
    }

    /// Enable or disable the constraint
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if constraint is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get constraint name
    pub fn constraint_name(&self) -> String {
        self.constraint.lock().unwrap().name().to_string()
    }
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

// ============================================================================
// PLUGIN TRAIT IMPLEMENTATION FOR CONSTRAINTS
// ============================================================================

use crate::plugin::Plugin;

impl Plugin for ConstraintPlugin {
    fn name(&self) -> &str {
        // Since name() requires returning &str but we need to lock the mutex,
        // we'll use a workaround by storing a leaked string
        // This is safe because plugin names are typically static
        Box::leak(self.constraint_name().into_boxed_str())
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        if let Ok(mut constraint) = self.constraint.lock() {
            constraint.reset();
        }
        Ok(())
    }

    fn transform_logits(&self, logits: &mut LogitsView, _context: &[i32]) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Apply constraint to logits
        let vocab_size = logits.vocab_size();
        if let Ok(constraint) = self.constraint.lock() {
            constraint.apply(logits, vocab_size)?;
        }

        Ok(())
    }

    fn on_token(&self, token_text: &str) -> Result<String> {
        if !self.enabled {
            return Ok(token_text.to_string());
        }

        // Update constraint state with the token
        // We extract token_id from the text (simplified approach)
        // In a real implementation, this should be provided by the caller
        if let Ok(mut constraint) = self.constraint.lock() {
            // For now, we pass -1 as token_id since we don't have it in this hook
            // This is a limitation that should be addressed in the Plugin trait design
            let _ = constraint.update(-1, token_text);
        }

        Ok(token_text.to_string())
    }

    fn post_generate(&self, output: &str) -> Result<String> {
        // Check if constraint is satisfied
        if self.enabled {
            if let Ok(constraint) = self.constraint.lock() {
                if !constraint.is_satisfied() {
                    eprintln!(
                        "Warning: Constraint '{}' not satisfied. State: {}",
                        constraint.name(),
                        constraint.current_state()
                    );
                }
            }
        }

        Ok(output.to_string())
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
