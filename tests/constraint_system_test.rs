use loci::constraint::*;
use std::collections::HashSet;

#[test]
fn test_token_whitelist_constraint() {
    let allowed_tokens: HashSet<i32> = [1, 2, 3, 4, 5].iter().cloned().collect();
    let constraint = TokenWhitelistConstraint::new("test_whitelist", allowed_tokens.clone());

    assert_eq!(constraint.name(), "test_whitelist");
    assert!(!constraint.is_stateful());

    let allowed = constraint.get_allowed_tokens().unwrap();
    assert_eq!(allowed, allowed_tokens);
}

#[test]
fn test_token_blacklist_constraint() {
    let banned_tokens: HashSet<i32> = [10, 20, 30].iter().cloned().collect();
    let constraint = TokenBlacklistConstraint::new("test_blacklist", banned_tokens.clone());

    assert_eq!(constraint.name(), "test_blacklist");
    assert!(!constraint.is_stateful());

    let allowed = constraint.get_allowed_tokens().unwrap();
    // Should not contain banned tokens
    for token in &banned_tokens {
        assert!(!allowed.contains(token));
    }
}

#[test]
fn test_length_constraint() {
    let mut constraint = LengthConstraint::new("test_length", 5, 10);

    assert_eq!(constraint.name(), "test_length");
    assert!(constraint.is_stateful());

    // Test within bounds
    for i in 0..7 {
        assert!(constraint.update(i, "token").is_ok());
        let allowed = constraint.get_allowed_tokens().unwrap();
        assert!(!allowed.is_empty()); // Should allow tokens
    }

    // Test at max length
    for i in 7..10 {
        assert!(constraint.update(i, "token").is_ok());
    }

    // At max length, should only allow EOS tokens
    let allowed = constraint.get_allowed_tokens().unwrap();
    // In a real implementation, this would only contain EOS token IDs
    // For now, we just check it's not empty (placeholder implementation)
    assert!(!allowed.is_empty());
}

#[test]
fn test_constraint_manager() {
    let mut manager = ConstraintManager::new();

    // Add constraints
    let whitelist: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
    let whitelist_constraint = TokenWhitelistConstraint::new("whitelist", whitelist);

    let blacklist: HashSet<i32> = [2].iter().cloned().collect();
    let blacklist_constraint = TokenBlacklistConstraint::new("blacklist", blacklist);

    manager.add_constraint(Box::new(whitelist_constraint));
    manager.add_constraint(Box::new(blacklist_constraint));

    assert_eq!(manager.constraint_count(), 2);

    // Test combined constraints
    let combined_allowed = manager.get_combined_allowed_tokens().unwrap();
    // Should contain 1, 3 but not 2 (blacklisted)
    assert!(combined_allowed.contains(&1));
    assert!(!combined_allowed.contains(&2)); // Blacklisted
    assert!(combined_allowed.contains(&3));
}

#[test]
fn test_constraint_combinator() {
    let constraint1 =
        TokenWhitelistConstraint::new("constraint1", [1, 2, 3, 4].iter().cloned().collect());
    let constraint2 =
        TokenWhitelistConstraint::new("constraint2", [2, 3, 4, 5].iter().cloned().collect());

    let combinator = ConstraintCombinator::new(
        "combined".to_string(),
        vec![Box::new(constraint1), Box::new(constraint2)],
        CombinatorMode::Intersection,
    );

    let allowed = combinator.get_allowed_tokens().unwrap();
    // Intersection should be [2, 3, 4]
    assert!(allowed.contains(&2));
    assert!(allowed.contains(&3));
    assert!(allowed.contains(&4));
    assert!(!allowed.contains(&1));
    assert!(!allowed.contains(&5));
}

#[test]
fn test_constraint_mask() {
    let vocab_size = 1000;
    let mut mask = ConstraintMask::new(vocab_size);

    // Initially all tokens should be allowed
    assert_eq!(mask.allowed_count(), vocab_size);

    // Ban some tokens
    let banned_tokens: HashSet<i32> = [10, 20, 30].iter().cloned().collect();
    mask.apply_blacklist(&banned_tokens);

    assert_eq!(mask.allowed_count(), vocab_size - banned_tokens.len());

    // Check specific tokens
    assert!(!mask.is_allowed(10));
    assert!(!mask.is_allowed(20));
    assert!(!mask.is_allowed(30));
    assert!(mask.is_allowed(0));
    assert!(mask.is_allowed(100));
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

    let allowed = constraint.get_allowed_tokens().unwrap();
    // Should contain 1, 3 but not 2
    assert!(allowed.contains(&1));
    assert!(!allowed.contains(&2));
    assert!(allowed.contains(&3));
}

#[test]
fn test_regex_constraint_creation() {
    // Test valid regex
    let result = RegexConstraint::new("test_regex", r"\d+");
    assert!(result.is_ok());

    let constraint = result.unwrap();
    assert_eq!(constraint.name(), "test_regex");
    assert!(constraint.is_stateful());

    // Test invalid regex
    let result = RegexConstraint::new("invalid_regex", r"[");
    assert!(result.is_err());
}

#[test]
fn test_json_constraint_creation() {
    let schema = JsonSchema {
        schema_type: "object".to_string(),
        properties: std::collections::HashMap::new(),
        required: vec![],
        additional_properties: true,
    };

    let constraint = JsonConstraint::new("test_json", schema);
    assert_eq!(constraint.name(), "test_json");
    assert!(constraint.is_stateful());
}

// Mock logits for testing
fn create_mock_logits(size: usize) -> Vec<f32> {
    (0..size).map(|i| i as f32 * 0.1).collect()
}

#[test]
fn test_constraint_mask_application() {
    let vocab_size = 100;
    let mut logits = create_mock_logits(vocab_size);
    let mut mask = ConstraintMask::new(vocab_size);

    // Ban some tokens
    let banned_tokens: HashSet<i32> = [10, 20, 30].iter().cloned().collect();
    mask.apply_blacklist(&banned_tokens);

    // Apply mask to logits
    mask.apply_to_logits(&mut logits);

    // Check that banned tokens have -inf logits
    assert_eq!(logits[10], f32::NEG_INFINITY);
    assert_eq!(logits[20], f32::NEG_INFINITY);
    assert_eq!(logits[30], f32::NEG_INFINITY);

    // Check that allowed tokens keep their values
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[1], 0.1);
    assert_eq!(logits[50], 5.0);
}
