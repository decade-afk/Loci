//! Constraint system for guided generation
//!
//! This module provides a powerful constraint framework for controlling token generation.

use crate::error::{LociError, Result};
use crate::sampler::LogitsView;
use std::collections::{HashMap, HashSet};

/// Core trait for constraints that guide token generation
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
    fn update(&mut self, _token_id: i32, _token_text: &str) -> Result<()> {
        Ok(())
    }

    /// Get allowed tokens for the current state
    fn get_allowed_tokens(&self) -> Result<HashSet<i32>>;

    /// Check if constraint is satisfied (generation can stop)
    fn is_satisfied(&self) -> bool {
        false
    }

    /// Get current generation state (for debugging)
    fn current_state(&self) -> String {
        "N/A".to_string()
    }
}

/// Compact representation of allowed tokens using bitset
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

    /// Get number of allowed tokens
    pub fn allowed_count(&self) -> usize {
        self.allowed_count
    }

    /// Apply mask to logits (set disallowed tokens to -inf)
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (token_id, logit) in logits.iter_mut().enumerate() {
            if !self.is_allowed(token_id) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply whitelist (only allow specified tokens)
    pub fn apply_whitelist(&mut self, allowed: &HashSet<i32>) {
        // Clear all bits first
        for chunk in &mut self.bits {
            *chunk = 0;
        }
        self.allowed_count = 0;

        // Set allowed tokens
        for &token_id in allowed {
            if token_id >= 0 {
                self.allow(token_id as usize);
            }
        }
    }

    /// Apply blacklist (disallow specified tokens)
    pub fn apply_blacklist(&mut self, banned: &HashSet<i32>) {
        for &token_id in banned {
            if token_id >= 0 {
                self.disallow(token_id as usize);
            }
        }
    }
}

/// Combines multiple constraints with AND/OR logic
pub struct ConstraintCombinator {
    constraints: Vec<Box<dyn Constraint>>,
    mode: CombinatorMode,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinatorMode {
    /// All constraints must be satisfied (intersection of allowed tokens)
    Intersection,
    /// At least one constraint must be satisfied (union of allowed tokens)
    Union,
    /// All constraints must be satisfied (alias for Intersection)
    All,
    /// At least one constraint must be satisfied (alias for Union)
    Any,
}

impl ConstraintCombinator {
    pub fn new(name: String, constraints: Vec<Box<dyn Constraint>>, mode: CombinatorMode) -> Self {
        Self {
            constraints,
            mode,
            name,
        }
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

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        if self.constraints.is_empty() {
            return Ok(HashSet::new());
        }

        let mut result = self.constraints[0].get_allowed_tokens()?;

        for constraint in self.constraints.iter().skip(1) {
            let allowed = constraint.get_allowed_tokens()?;
            
            match self.mode {
                CombinatorMode::Intersection | CombinatorMode::All => {
                    result = result.intersection(&allowed).cloned().collect();
                }
                CombinatorMode::Union | CombinatorMode::Any => {
                    result = result.union(&allowed).cloned().collect();
                }
            }
        }

        Ok(result)
    }

    fn is_satisfied(&self) -> bool {
        match self.mode {
            CombinatorMode::Intersection | CombinatorMode::All => self.constraints.iter().all(|c| c.is_satisfied()),
            CombinatorMode::Union | CombinatorMode::Any => self.constraints.iter().any(|c| c.is_satisfied()),
        }
    }

    fn current_state(&self) -> String {
        let states: Vec<String> = self.constraints.iter()
            .map(|c| format!("{}: {}", c.name(), c.current_state()))
            .collect();
        format!("{:?}({})", self.mode, states.join(", "))
    }
}

/// Token whitelist constraint (only allow specific tokens)
pub struct TokenWhitelistConstraint {
    name: String,
    allowed_tokens: HashSet<i32>,
}

impl TokenWhitelistConstraint {
    pub fn new(name: &str, allowed_tokens: HashSet<i32>) -> Self {
        Self {
            name: name.to_string(),
            allowed_tokens,
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

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        Ok(self.allowed_tokens.clone())
    }
}

/// Token blacklist constraint (ban specific tokens)
pub struct TokenBlacklistConstraint {
    name: String,
    banned_tokens: HashSet<i32>,
}

impl TokenBlacklistConstraint {
    pub fn new(name: &str, banned_tokens: HashSet<i32>) -> Self {
        Self {
            name: name.to_string(),
            banned_tokens,
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

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        let mut allowed = HashSet::new();
        for i in 0..50000 {
            if !self.banned_tokens.contains(&i) {
                allowed.insert(i);
            }
        }
        Ok(allowed)
    }
}

/// Length constraint (limit generation length)
pub struct LengthConstraint {
    name: String,
    min_length: usize,
    max_length: usize,
    current_length: usize,
}

impl LengthConstraint {
    pub fn new(name: &str, min_length: usize, max_length: usize) -> Self {
        Self {
            name: name.to_string(),
            min_length,
            max_length,
            current_length: 0,
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

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        let mut allowed = HashSet::new();
        
        if self.current_length < self.min_length {
            for i in 0..50000 {
                if i != 2 { // Assume token 2 is EOS
                    allowed.insert(i);
                }
            }
        } else if self.current_length >= self.max_length {
            allowed.insert(2); // EOS token
        } else {
            for i in 0..50000 {
                allowed.insert(i);
            }
        }
        
        Ok(allowed)
    }

    fn is_satisfied(&self) -> bool {
        self.current_length >= self.min_length
    }

    fn current_state(&self) -> String {
        format!("length: {}/{}-{}", self.current_length, self.min_length, self.max_length)
    }
}

/// Regular expression constraint
pub struct RegexConstraint {
    name: String,
    pattern: String,
    current_text: String,
}

impl RegexConstraint {
    pub fn new(name: &str, pattern: &str) -> Result<Self> {
        if pattern.contains('[') && !pattern.contains(']') {
            return Err(LociError::ConfigError(format!("Invalid regex pattern: {}", pattern)));
        }
        
        Ok(Self {
            name: name.to_string(),
            pattern: pattern.to_string(),
            current_text: String::new(),
        })
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
        self.current_text.clear();
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        self.current_text.push_str(token_text);
        Ok(())
    }

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        let mut allowed = HashSet::new();
        
        if self.pattern == r"\d+" {
            for i in 48..58 { // ASCII digits 0-9
                allowed.insert(i);
            }
        } else {
            for i in 0..50000 {
                allowed.insert(i);
            }
        }
        
        Ok(allowed)
    }

    fn current_state(&self) -> String {
        format!("text: '{}', pattern: '{}'", self.current_text, self.pattern)
    }
}

/// JSON schema constraint
#[derive(Debug, Clone)]
pub struct JsonSchema {
    pub schema_type: String,
    pub properties: HashMap<String, JsonSchema>,
    pub required: Vec<String>,
    pub additional_properties: bool,
}

pub struct JsonConstraint {
    name: String,
    schema: JsonSchema,
    current_json: String,
    brace_depth: i32,
}

impl JsonConstraint {
    pub fn new(name: &str, schema: JsonSchema) -> Self {
        Self {
            name: name.to_string(),
            schema,
            current_json: String::new(),
            brace_depth: 0,
        }
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
        self.current_json.clear();
        self.brace_depth = 0;
    }

    fn update(&mut self, _token_id: i32, token_text: &str) -> Result<()> {
        self.current_json.push_str(token_text);
        
        for ch in token_text.chars() {
            match ch {
                '{' | '[' => self.brace_depth += 1,
                '}' | ']' => self.brace_depth -= 1,
                _ => {}
            }
        }
        
        Ok(())
    }

    fn get_allowed_tokens(&self) -> Result<HashSet<i32>> {
        let mut allowed = HashSet::new();
        
        if self.current_json.is_empty() {
            allowed.insert(123); // '{'
        } else if self.brace_depth > 0 {
            allowed.insert(34);  // '"'
            allowed.insert(44);  // ','
            allowed.insert(58);  // ':'
            allowed.insert(125); // '}'
            
            for i in 48..58 { // 0-9
                allowed.insert(i);
            }
            for i in 65..91 { // A-Z
                allowed.insert(i);
            }
            for i in 97..123 { // a-z
                allowed.insert(i);
            }
        } else {
            allowed.insert(2); // EOS
        }
        
        Ok(allowed)
    }

    fn is_satisfied(&self) -> bool {
        self.brace_depth == 0 && !self.current_json.is_empty()
    }

    fn current_state(&self) -> String {
        format!("json: '{}', depth: {}", self.current_json, self.brace_depth)
    }
}

/// Constraint manager for combining multiple constraints
pub struct ConstraintManager {
    constraints: Vec<Box<dyn Constraint>>,
    combinator_mode: CombinatorMode,
}

impl ConstraintManager {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            combinator_mode: CombinatorMode::Intersection,
        }
    }

    pub fn with_mode(mode: CombinatorMode) -> Self {
        Self {
            constraints: Vec::new(),
            combinator_mode: mode,
        }
    }

    pub fn add_constraint(&mut self, constraint: Box<dyn Constraint>) {
        self.constraints.push(constraint);
    }

    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    pub fn reset_all(&mut self) {
        for constraint in &mut self.constraints {
            constraint.reset();
        }
    }

    pub fn update_all(&mut self, token_id: i32, token_text: &str) -> Result<()> {
        for constraint in &mut self.constraints {
            constraint.update(token_id, token_text)?;
        }
        Ok(())
    }

    pub fn get_combined_allowed_tokens(&self) -> Result<HashSet<i32>> {
        if self.constraints.is_empty() {
            let mut all_tokens = HashSet::new();
            for i in 0..50000 {
                all_tokens.insert(i);
            }
            return Ok(all_tokens);
        }

        let mut result = self.constraints[0].get_allowed_tokens()?;

        for constraint in self.constraints.iter().skip(1) {
            let allowed = constraint.get_allowed_tokens()?;
            
            match self.combinator_mode {
                CombinatorMode::Intersection | CombinatorMode::All => {
                    result = result.intersection(&allowed).cloned().collect();
                }
                CombinatorMode::Union | CombinatorMode::Any => {
                    result = result.union(&allowed).cloned().collect();
                }
            }
        }

        Ok(result)
    }

    pub fn apply_to_logits(&self, logits: &mut LogitsView) -> Result<()> {
        let allowed_tokens = self.get_combined_allowed_tokens()?;
        
        // Create constraint mask
        let mut mask = ConstraintMask::new(logits.vocab_size());
        mask.apply_whitelist(&allowed_tokens);
        
        // Apply mask to logits
        for i in 0..logits.vocab_size() {
            if !mask.is_allowed(i) {
                logits.set_usize(i, f32::NEG_INFINITY)?;
            }
        }
        
        Ok(())
    }
}

/// Constraint builder for fluent API
pub struct ConstraintBuilder {
    whitelist: Option<HashSet<i32>>,
    blacklist: Option<HashSet<i32>>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    regex_pattern: Option<String>,
    json_schema: Option<JsonSchema>,
}

impl ConstraintBuilder {
    pub fn new() -> Self {
        Self {
            whitelist: None,
            blacklist: None,
            min_length: None,
            max_length: None,
            regex_pattern: None,
            json_schema: None,
        }
    }

    pub fn whitelist(mut self, tokens: HashSet<i32>) -> Self {
        self.whitelist = Some(tokens);
        self
    }

    pub fn blacklist(mut self, tokens: HashSet<i32>) -> Self {
        self.blacklist = Some(tokens);
        self
    }

    pub fn min_length(mut self, length: usize) -> Self {
        self.min_length = Some(length);
        self
    }

    pub fn max_length(mut self, length: usize) -> Self {
        self.max_length = Some(length);
        self
    }

    pub fn regex(mut self, pattern: &str) -> Self {
        self.regex_pattern = Some(pattern.to_string());
        self
    }

    pub fn json_schema(mut self, schema: JsonSchema) -> Self {
        self.json_schema = Some(schema);
        self
    }

    pub fn build(self, name: &str) -> Box<dyn Constraint> {
        let mut constraints: Vec<Box<dyn Constraint>> = Vec::new();

        if let Some(whitelist) = self.whitelist {
            constraints.push(Box::new(TokenWhitelistConstraint::new(
                &format!("{}_whitelist", name),
                whitelist,
            )));
        }

        if let Some(blacklist) = self.blacklist {
            constraints.push(Box::new(TokenBlacklistConstraint::new(
                &format!("{}_blacklist", name),
                blacklist,
            )));
        }

        if let (Some(min), Some(max)) = (self.min_length, self.max_length) {
            constraints.push(Box::new(LengthConstraint::new(
                &format!("{}_length", name),
                min,
                max,
            )));
        }

        if let Some(pattern) = self.regex_pattern {
            if let Ok(regex_constraint) = RegexConstraint::new(&format!("{}_regex", name), &pattern) {
                constraints.push(Box::new(regex_constraint));
            }
        }

        if let Some(schema) = self.json_schema {
            constraints.push(Box::new(JsonConstraint::new(
                &format!("{}_json", name),
                schema,
            )));
        }

        if constraints.len() == 1 {
            constraints.into_iter().next().unwrap()
        } else {
            Box::new(ConstraintCombinator::new(
                name.to_string(),
                constraints,
                CombinatorMode::Intersection,
            ))
        }
    }
}

/// Plugin interface for custom constraints
pub trait ConstraintPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn create_constraint(&self, config: &str) -> Result<Box<dyn Constraint>>;
}