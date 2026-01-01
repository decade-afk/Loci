//! Constraints Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::collections::HashSet;
use anyhow::Result;




#[derive(Debug)]
    /// ConstraintContext structure
pub struct ConstraintContext<'a> {
    
    pub generated_tokens: &'a [i32],

    
    pub generated_text: Option<&'a str>,

    
    pub candidate_token: i32,

    
    pub candidate_text: Option<&'a str>,

    
    pub vocab_size: usize,
}







pub trait Constraint: Send + Sync {
    
    
    
    
    
    
    
    
    
    fn is_allowed(&self, token_id: i32, context: &ConstraintContext) -> bool;

    
    
    
    fn update(&mut self, _token_id: i32, _context: &ConstraintContext) {
        
    }

    
    fn reset(&mut self) {
        
    }

    
    fn description(&self) -> &str {
        "Generic Constraint"
    }

    
    
    
    fn allowed_tokens(&self, _context: &ConstraintContext) -> Option<HashSet<i32>> {
        None 
    }
}






#[derive(Debug, Clone)]
    /// TokenMask structure
pub struct TokenMask {
    
    pub allowed: Vec<bool>,

    
    pub allowed_count: usize,
}

// Implementation for TokenMask
impl TokenMask {
    
    /// new_allow_all function
    pub fn new_allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![true; vocab_size],
            allowed_count: vocab_size,
        }
    }

    
    /// new_deny_all function
    pub fn new_deny_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![false; vocab_size],
            allowed_count: 0,
        }
    }

    
    /// from_constraint function
    pub fn from_constraint(
        constraint: &dyn Constraint,
        context: &ConstraintContext,
    ) -> Self {
        let vocab_size = context.vocab_size;

        
        if let Some(allowed_set) = constraint.allowed_tokens(context) {
            let mut allowed = vec![false; vocab_size];
            let mut count = 0;

            for &token_id in &allowed_set {
                if (token_id as usize) < vocab_size {
                    allowed[token_id as usize] = true;
                    count += 1;
                }
            }

            return Self {
                allowed,
                allowed_count: count,
            };
        }

        
        let mut allowed = Vec::with_capacity(vocab_size);
        let mut count = 0;

        for token_id in 0..vocab_size as i32 {
            let is_allowed = constraint.is_allowed(token_id, context);
            allowed.push(is_allowed);
            if is_allowed {
                count += 1;
            }
        }

        Self {
            allowed,
            allowed_count: count,
        }
    }

    
    #[inline]
    /// is_allowed function
    pub fn is_allowed(&self, token_id: i32) -> bool {
        self.allowed.get(token_id as usize).copied().unwrap_or(false)
    }

    
    /// allowed_count function
    pub fn allowed_count(&self) -> usize {
        self.allowed_count
    }

    
    /// apply_to_logits function
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, &allowed) in self.allowed.iter().enumerate() {
            if !allowed && i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}













    /// RegexConstraint structure
pub struct RegexConstraint {
    
    pattern: regex::Regex,

    
    pub current_text: String,

    
    
    
    prefix_mode: bool,
}

// Implementation for RegexConstraint
impl RegexConstraint {
    
    
    
    
    
    /// new function
    pub fn new(pattern: &str, prefix_mode: bool) -> Result<Self> {
        let pattern = regex::Regex::new(pattern)?;

        Ok(Self {
            pattern,
            current_text: String::new(),
            prefix_mode,
        })
    }

    
    fn is_partial_match(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        
        if let Some(mat) = self.pattern.find(text) {
            
            return mat.start() == 0;
        }

        
        
        
        true
    }
}

// Implementation for Constraint
impl Constraint for RegexConstraint {
    fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
        
        let candidate_text = match context.candidate_text {
            Some(text) => text,
            None => return true, 
        };

        
        let simulated_text = format!("{}{}", self.current_text, candidate_text);

        if self.prefix_mode {
            
            self.is_partial_match(&simulated_text)
        } else {
            
            self.pattern.is_match(&simulated_text)
        }
    }

    fn update(&mut self, _token_id: i32, context: &ConstraintContext) {
        
        if let Some(text) = context.candidate_text {
            self.current_text.push_str(text);
        }
    }

    fn reset(&mut self) {
        self.current_text.clear();
    }

    fn description(&self) -> &str {
        "Regex Constraint"
    }
}















#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// JsonState enumeration
pub enum JsonState {
    Idle,           
    InObject,       
    InArray,        
    ExpectKey,      
    ExpectColon,    
    ExpectValue,    
    InString,       
    Complete,       
}

    /// JsonSchemaConstraint structure
pub struct JsonSchemaConstraint {
    
    target_type: JsonType,

    
    pub state: JsonState,

    
    pub current_text: String,

    
    pub brace_depth: i32,
    pub bracket_depth: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// JsonType enumeration
pub enum JsonType {
    Object,
    Array,
    String,
    Number,
    Boolean,
    Null,
}

// Implementation for JsonSchemaConstraint
impl JsonSchemaConstraint {
    
    /// new function
    pub fn new(target_type: JsonType) -> Self {
        Self {
            target_type,
            state: JsonState::Idle,
            current_text: String::new(),
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    
    fn is_valid_next_char(&self, ch: char) -> bool {
        match self.state {
            JsonState::Idle => {
                match self.target_type {
                    JsonType::Object => ch == '{',
                    JsonType::Array => ch == '[',
                    JsonType::String => ch == '"',
                    JsonType::Number => ch.is_ascii_digit() || ch == '-',
                    JsonType::Boolean => ch == 't' || ch == 'f',
                    JsonType::Null => ch == 'n',
                }
            }
            JsonState::InObject => {
                ch == '"' || ch == '}' || ch.is_whitespace()
            }
            JsonState::InArray => {
                ch == '"' || ch == '{' || ch == '[' || ch == ']' ||
                ch.is_ascii_digit() || ch == '-' || ch.is_whitespace()
            }
            JsonState::InString => {
                true 
            }
            JsonState::ExpectColon => {
                ch == ':' || ch.is_whitespace()
            }
            JsonState::ExpectValue => {
                ch == '"' || ch == '{' || ch == '[' ||
                ch.is_ascii_digit() || ch == '-' || ch == 't' || ch == 'f' || ch == 'n' ||
                ch.is_whitespace()
            }
            JsonState::Complete => false,
            _ => true,
        }
    }

    
    fn update_state(&mut self, ch: char) {
        match ch {
            '{' => {
                self.brace_depth += 1;
                self.state = JsonState::InObject;
            }
            '}' => {
                self.brace_depth -= 1;
                if self.brace_depth == 0 && self.target_type == JsonType::Object {
                    self.state = JsonState::Complete;
                }
            }
            '[' => {
                self.bracket_depth += 1;
                self.state = JsonState::InArray;
            }
            ']' => {
                self.bracket_depth -= 1;
                if self.bracket_depth == 0 && self.target_type == JsonType::Array {
                    self.state = JsonState::Complete;
                }
            }
            '"' => {
                if self.state == JsonState::InString {
                    self.state = if self.brace_depth > 0 {
                        JsonState::InObject
                    } else {
                        JsonState::Complete
                    };
                } else {
                    self.state = JsonState::InString;
                }
            }
            ':' if self.state == JsonState::ExpectColon => {
                self.state = JsonState::ExpectValue;
            }
            _ => {}
        }
    }
}

// Implementation for Constraint
impl Constraint for JsonSchemaConstraint {
    fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
        let candidate_text = match context.candidate_text {
            Some(text) => text,
            None => return true,
        };

        
        for ch in candidate_text.chars() {
            if !self.is_valid_next_char(ch) {
                return false;
            }
        }

        true
    }

    fn update(&mut self, _token_id: i32, context: &ConstraintContext) {
        if let Some(text) = context.candidate_text {
            self.current_text.push_str(text);

            
            for ch in text.chars() {
                self.update_state(ch);
            }
        }
    }

    fn reset(&mut self) {
        self.state = JsonState::Idle;
        self.current_text.clear();
        self.brace_depth = 0;
        self.bracket_depth = 0;
    }

    fn description(&self) -> &str {
        "JSON Schema Constraint"
    }
}




    /// AndConstraint structure
pub struct AndConstraint {
    constraints: Vec<Box<dyn Constraint>>,
}

// Implementation for AndConstraint
impl AndConstraint {
    /// new function
    pub fn new(constraints: Vec<Box<dyn Constraint>>) -> Self {
        Self { constraints }
    }
}

// Implementation for Constraint
impl Constraint for AndConstraint {
    fn is_allowed(&self, token_id: i32, context: &ConstraintContext) -> bool {
        self.constraints.iter().all(|c| c.is_allowed(token_id, context))
    }

    fn update(&mut self, token_id: i32, context: &ConstraintContext) {
        for c in &mut self.constraints {
            c.update(token_id, context);
        }
    }

    fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset();
        }
    }

    fn description(&self) -> &str {
        "AND Combinator"
    }
}


    /// OrConstraint structure
pub struct OrConstraint {
    constraints: Vec<Box<dyn Constraint>>,
}

// Implementation for OrConstraint
impl OrConstraint {
    /// new function
    pub fn new(constraints: Vec<Box<dyn Constraint>>) -> Self {
        Self { constraints }
    }
}

// Implementation for Constraint
impl Constraint for OrConstraint {
    fn is_allowed(&self, token_id: i32, context: &ConstraintContext) -> bool {
        self.constraints.iter().any(|c| c.is_allowed(token_id, context))
    }

    fn update(&mut self, token_id: i32, context: &ConstraintContext) {
        for c in &mut self.constraints {
            c.update(token_id, context);
        }
    }

    fn reset(&mut self) {
        for c in &mut self.constraints {
            c.reset();
        }
    }

    fn description(&self) -> &str {
        "OR Combinator"
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_mask_basic() {
        let mask = TokenMask::new_allow_all(100);
        assert_eq!(mask.allowed_count(), 100);
        assert!(mask.is_allowed(50));

        let mask = TokenMask::new_deny_all(100);
        assert_eq!(mask.allowed_count(), 0);
        assert!(!mask.is_allowed(50));
    }

    #[test]
    fn test_regex_constraint_email() {
        let mut constraint = RegexConstraint::new(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            true,
        ).unwrap();

        let context = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some(""),
            candidate_token: 0,
            candidate_text: Some("user"),
            vocab_size: 1000,
        };

        
        assert!(constraint.is_allowed(0, &context));

        constraint.update(0, &context);

        let context2 = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some("user"),
            candidate_token: 1,
            candidate_text: Some("@"),
            vocab_size: 1000,
        };

        
        assert!(constraint.is_allowed(1, &context2));
    }

    #[test]
    fn test_json_constraint_object() {
        let mut constraint = JsonSchemaConstraint::new(JsonType::Object);

        let context = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some(""),
            candidate_token: 0,
            candidate_text: Some("{"),
            vocab_size: 1000,
        };

        
        assert!(constraint.is_allowed(0, &context));

        constraint.update(0, &context);
        assert_eq!(constraint.state, JsonState::InObject);
        assert_eq!(constraint.brace_depth, 1);
    }
}
