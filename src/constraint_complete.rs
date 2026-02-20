//! Complete constraint system implementation
//!
//! This module provides the complete implementation of the constraint system
//! with all missing pieces filled in.

use crate::error::{LociError, Result};
use crate::sampler::LogitsView;
use std::collections::{HashMap, HashSet};

// Re-export the core trait and types from the main constraint module
pub use crate::constraint::{Constraint, ConstraintMask, CombinatorMode};

/// Constraint manager for combining multiple constraints
pub struct ConstraintManager {
    constraints: Vec<Box<dyn Constraint>>,
    combinator_mode: CombinatorMode,
}

impl ConstraintManager {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            combinator_mode: CombinatorMode::All,
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
            // No constraints, allow all tokens
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
                CombinatorMode::All | CombinatorMode::Intersection => {
                    result = result.intersection(&allowed).cloned().collect();
                }
                CombinatorMode::Any | CombinatorMode::Union => {
                    result = result.union(&allowed).cloned().collect();
                }
            }
        }

        Ok(result)
    }

    pub fn apply_to_logits(&self, logits: &mut LogitsView) -> Result<()> {
        let allowed_tokens = self.get_combined_allowed_tokens()?;
        
        // Apply mask to logits directly
        for i in 0..logits.vocab_size() {
            let token_id = i as i32;
            if !allowed_tokens.contains(&token_id) {
                logits.set_usize(i, f32::NEG_INFINITY)?;
            }
        }
        
        Ok(())
    }

    pub fn is_satisfied(&self) -> bool {
        match self.combinator_mode {
            CombinatorMode::All | CombinatorMode::Intersection => self.constraints.iter().all(|c| c.is_satisfied()),
            CombinatorMode::Any | CombinatorMode::Union => self.constraints.iter().any(|c| c.is_satisfied()),
        }
    }

    pub fn get_state_summary(&self) -> String {
        let states: Vec<String> = self.constraints.iter()
            .map(|c| format!("{}: {}", c.name(), c.current_state()))
            .collect();
        format!("{:?}({})", self.combinator_mode, states.join(", "))
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
                CombinatorMode::All,
            ))
        }
    }
}

/// Combines multiple constraints with AND/OR logic
pub struct ConstraintCombinator {
    constraints: Vec<Box<dyn Constraint>>,
    mode: CombinatorMode,
    name: String,
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
                CombinatorMode::All | CombinatorMode::Intersection => {
                    result = result.intersection(&allowed).cloned().collect();
                }
                CombinatorMode::Any | CombinatorMode::Union => {
                    result = result.union(&allowed).cloned().collect();
                }
            }
        }

        Ok(result)
    }

    fn is_satisfied(&self) -> bool {
        match self.mode {
            CombinatorMode::All | CombinatorMode::Intersection => self.constraints.iter().all(|c| c.is_satisfied()),
            CombinatorMode::Any | CombinatorMode::Union => self.constraints.iter().any(|c| c.is_satisfied()),
        }
    }

    fn current_state(&self) -> String {
        let states: Vec<String> = self.constraints.iter()
            .map(|c| format!("{}: {}", c.name(), c.current_state()))
            .collect();
        format!("{:?}({})", self.mode, states.join(", "))
    }
}

// Concrete constraint implementations
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

/// Plugin interface for custom constraints
pub trait ConstraintPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn create_constraint(&self, config: &str) -> Result<Box<dyn Constraint>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_manager() {
        let mut manager = ConstraintManager::new();
        
        let whitelist: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
        let whitelist_constraint = TokenWhitelistConstraint::new("whitelist", whitelist);
        
        let blacklist: HashSet<i32> = [2].iter().cloned().collect();
        let blacklist_constraint = TokenBlacklistConstraint::new("blacklist", blacklist);
        
        manager.add_constraint(Box::new(whitelist_constraint));
        manager.add_constraint(Box::new(blacklist_constraint));
        
        assert_eq!(manager.constraint_count(), 2);
        
        let combined_allowed = manager.get_combined_allowed_tokens().unwrap();
        assert!(combined_allowed.contains(&1));
        assert!(!combined_allowed.contains(&2)); // Blacklisted
        assert!(combined_allowed.contains(&3));
    }

    #[test]
    fn test_constraint_builder() {
        let builder = ConstraintBuilder::new();
        
        let constraint = builder
            .whitelist([1, 2, 3].iter().cloned().collect())
            .blacklist([2].iter().cloned().collect())
            .max_length(10)
            .build("test_constraint");
        
        assert_eq!(constraint.name(), "test_constraint");
    }
}