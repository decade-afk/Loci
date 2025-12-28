/**
 * Phase 2 Week 2: Constraint Sampling 综合测试
 *
 * 测试目标：
 * 1. Token Mask 功能验证
 * 2. RegexConstraint 各种模式测试
 * 3. JsonSchemaConstraint 状态机正确性
 * 4. 约束组合器（AND/OR）
 * 5. 实际场景：强制生成 JSON
 */

use loci::constraints::*;

// ==================== Token Mask 测试 ====================

#[test]
fn test_token_mask_creation() {
    // 测试全允许 mask
    let mask_all = TokenMask::new_allow_all(1000);
    assert_eq!(mask_all.allowed_count(), 1000);
    assert!(mask_all.is_allowed(0));
    assert!(mask_all.is_allowed(999));

    // 测试全拒绝 mask
    let mask_none = TokenMask::new_deny_all(1000);
    assert_eq!(mask_none.allowed_count(), 0);
    assert!(!mask_none.is_allowed(0));
    assert!(!mask_none.is_allowed(999));

    println!("✅ Token mask creation test passed");
}

#[test]
fn test_token_mask_apply_to_logits() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 创建只允许 token 1 和 3 的 mask
    let mut allowed = vec![false; 5];
    allowed[1] = true;
    allowed[3] = true;

    let mask = TokenMask {
        allowed,
        allowed_count: 2,
    };

    mask.apply_to_logits(&mut logits);

    // 检查不允许的 token 被设置为 -inf
    assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
    assert_eq!(logits[1], 2.0);
    assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
    assert_eq!(logits[3], 4.0);
    assert!(logits[4].is_infinite() && logits[4].is_sign_negative());

    println!("✅ Token mask logits application test passed");
}

// ==================== RegexConstraint 测试 ====================

#[test]
fn test_regex_constraint_email() {
    let mut constraint = RegexConstraint::new(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        true, // 前缀模式
    ).unwrap();

    println!("🧪 Testing email regex constraint...");

    // 测试 "user" 前缀
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("user"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(0, &ctx1), "Should allow 'user'");
    constraint.update(0, &ctx1);

    // 测试 "@" 符号
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("user"),
        candidate_token: 1,
        candidate_text: Some("@"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(1, &ctx2), "Should allow '@'");
    constraint.update(1, &ctx2);

    // 测试域名
    let ctx3 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("user@"),
        candidate_token: 2,
        candidate_text: Some("example"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(2, &ctx3), "Should allow 'example'");

    println!("✅ Email regex constraint test passed");
}

#[test]
fn test_regex_constraint_phone_number() {
    let mut constraint = RegexConstraint::new(
        r"^\+?[1-9]\d{1,14}$", // 国际电话号码格式
        true,
    ).unwrap();

    println!("🧪 Testing phone number regex constraint...");

    // 测试 "+"
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("+"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(0, &ctx1), "Should allow '+'");
    constraint.update(0, &ctx1);

    // 测试数字 "1"
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("+"),
        candidate_token: 1,
        candidate_text: Some("1"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(1, &ctx2), "Should allow '1'");

    println!("✅ Phone number regex constraint test passed");
}

#[test]
fn test_regex_constraint_reset() {
    let mut constraint = RegexConstraint::new(r"^test", true).unwrap();

    let ctx = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("te"),
        vocab_size: 1000,
    };

    constraint.update(0, &ctx);
    assert_eq!(constraint.current_text, "te");

    // 重置后应该清空
    constraint.reset();
    assert_eq!(constraint.current_text, "");

    println!("✅ Regex constraint reset test passed");
}

// ==================== JsonSchemaConstraint 测试 ====================

#[test]
fn test_json_constraint_object_basic() {
    let mut constraint = JsonSchemaConstraint::new(JsonType::Object);

    println!("🧪 Testing JSON object constraint...");

    // 测试 "{"
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("{"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(0, &ctx1), "Should allow '{{'");
    constraint.update(0, &ctx1);
    assert_eq!(constraint.brace_depth, 1);
    assert_eq!(constraint.state, JsonState::InObject);

    // 测试 """（键的开始）
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("{{"),
        candidate_token: 1,
        candidate_text: Some("\""),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(1, &ctx2), "Should allow '\"'");
    constraint.update(1, &ctx2);
    assert_eq!(constraint.state, JsonState::InString);

    println!("✅ JSON object constraint test passed");
}

#[test]
fn test_json_constraint_array_basic() {
    let mut constraint = JsonSchemaConstraint::new(JsonType::Array);

    println!("🧪 Testing JSON array constraint...");

    // 测试 "["
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("["),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(0, &ctx1), "Should allow '['");
    constraint.update(0, &ctx1);
    assert_eq!(constraint.bracket_depth, 1);
    assert_eq!(constraint.state, JsonState::InArray);

    // 测试数字
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("["),
        candidate_token: 1,
        candidate_text: Some("1"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(1, &ctx2), "Should allow '1'");

    println!("✅ JSON array constraint test passed");
}

#[test]
fn test_json_constraint_nested_objects() {
    let mut constraint = JsonSchemaConstraint::new(JsonType::Object);

    println!("🧪 Testing nested JSON objects...");

    // 外层对象
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("{"),
        vocab_size: 1000,
    };

    constraint.update(0, &ctx1);
    assert_eq!(constraint.brace_depth, 1);

    // 内层对象
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("{\"nested\":"),
        candidate_token: 1,
        candidate_text: Some("{"),
        vocab_size: 1000,
    };

    constraint.update(1, &ctx2);
    assert_eq!(constraint.brace_depth, 2);

    // 关闭内层对象
    let ctx3 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("{\"nested\":{"),
        candidate_token: 2,
        candidate_text: Some("}"),
        vocab_size: 1000,
    };

    constraint.update(2, &ctx3);
    assert_eq!(constraint.brace_depth, 1);
    assert_ne!(constraint.state, JsonState::Complete); // 外层还未关闭

    // 关闭外层对象
    let ctx4 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("{\"nested\":{}}"),
        candidate_token: 3,
        candidate_text: Some("}"),
        vocab_size: 1000,
    };

    constraint.update(3, &ctx4);
    assert_eq!(constraint.brace_depth, 0);
    assert_eq!(constraint.state, JsonState::Complete);

    println!("✅ Nested JSON objects test passed");
}

#[test]
fn test_json_constraint_string_type() {
    let mut constraint = JsonSchemaConstraint::new(JsonType::String);

    println!("🧪 Testing JSON string constraint...");

    // 测试开始引号
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("\""),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(0, &ctx1), "Should allow opening quote");
    constraint.update(0, &ctx1);
    assert_eq!(constraint.state, JsonState::InString);

    // 测试字符串内容
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("\""),
        candidate_token: 1,
        candidate_text: Some("hello"),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(1, &ctx2), "Should allow string content");
    constraint.update(1, &ctx2);

    // 测试结束引号
    let ctx3 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some("\"hello"),
        candidate_token: 2,
        candidate_text: Some("\""),
        vocab_size: 1000,
    };

    assert!(constraint.is_allowed(2, &ctx3), "Should allow closing quote");
    constraint.update(2, &ctx3);
    assert_eq!(constraint.state, JsonState::Complete);

    println!("✅ JSON string constraint test passed");
}

// ==================== 约束组合器测试 ====================

#[test]
fn test_and_constraint() {
    // 创建两个简单约束（模拟：必须是字母 AND 必须是小写）
    struct LetterConstraint;
    impl Constraint for LetterConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_alphabetic())).unwrap_or(true)
        }
    }

    struct LowercaseConstraint;
    impl Constraint for LowercaseConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_lowercase())).unwrap_or(true)
        }
    }

    let and = AndConstraint::new(vec![
        Box::new(LetterConstraint),
        Box::new(LowercaseConstraint),
    ]);

    println!("🧪 Testing AND constraint combinator...");

    // 测试通过：小写字母
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("abc"),
        vocab_size: 1000,
    };

    assert!(and.is_allowed(0, &ctx1), "Should allow lowercase letters");

    // 测试失败：大写字母（不满足 lowercase 约束）
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 1,
        candidate_text: Some("ABC"),
        vocab_size: 1000,
    };

    assert!(!and.is_allowed(1, &ctx2), "Should reject uppercase letters");

    // 测试失败：数字（不满足 letter 约束）
    let ctx3 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 2,
        candidate_text: Some("123"),
        vocab_size: 1000,
    };

    assert!(!and.is_allowed(2, &ctx3), "Should reject numbers");

    println!("✅ AND constraint combinator test passed");
}

#[test]
fn test_or_constraint() {
    struct NumberConstraint;
    impl Constraint for NumberConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_numeric())).unwrap_or(true)
        }
    }

    struct LetterConstraint;
    impl Constraint for LetterConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_alphabetic())).unwrap_or(true)
        }
    }

    let or = OrConstraint::new(vec![
        Box::new(NumberConstraint),
        Box::new(LetterConstraint),
    ]);

    println!("🧪 Testing OR constraint combinator...");

    // 测试通过：数字
    let ctx1 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("123"),
        vocab_size: 1000,
    };

    assert!(or.is_allowed(0, &ctx1), "Should allow numbers");

    // 测试通过：字母
    let ctx2 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 1,
        candidate_text: Some("abc"),
        vocab_size: 1000,
    };

    assert!(or.is_allowed(1, &ctx2), "Should allow letters");

    // 测试失败：符号
    let ctx3 = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 2,
        candidate_text: Some("!@#"),
        vocab_size: 1000,
    };

    assert!(!or.is_allowed(2, &ctx3), "Should reject symbols");

    println!("✅ OR constraint combinator test passed");
}

// ==================== 实际场景测试 ====================

#[test]
fn test_force_json_generation_simulation() {
    let mut constraint = JsonSchemaConstraint::new(JsonType::Object);

    println!("🧪 Simulating forced JSON generation...");

    // 模拟生成 {"name": "test"}
    let steps = vec![
        "{", "\"", "name", "\"", ":", " ", "\"", "test", "\"", "}",
    ];

    for (i, step) in steps.iter().enumerate() {
        let ctx = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some(""),
            candidate_token: i as i32,
            candidate_text: Some(step),
            vocab_size: 1000,
        };

        assert!(
            constraint.is_allowed(i as i32, &ctx),
            "Step {} ('{}') should be allowed",
            i,
            step
        );

        constraint.update(i as i32, &ctx);
    }

    // 验证最终状态
    assert_eq!(constraint.state, JsonState::Complete);
    assert_eq!(constraint.brace_depth, 0);

    println!("   Generated: {}", constraint.current_text);
    println!("✅ Forced JSON generation simulation passed");
}

#[test]
fn test_invalid_json_rejected() {
    let constraint = JsonSchemaConstraint::new(JsonType::Object);

    println!("🧪 Testing invalid JSON rejection...");

    // 尝试生成 "[" (应该拒绝，因为期待对象不是数组)
    let ctx = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: Some("["),
        vocab_size: 1000,
    };

    assert!(!constraint.is_allowed(0, &ctx), "Should reject array start for object type");

    println!("✅ Invalid JSON rejection test passed");
}
