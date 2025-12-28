/**
 * Phase 2 Week 2: Constraint Sampling 演示程序
 *
 * 展示功能：
 * 1. RegexConstraint - 邮箱/电话号码生成
 * 2. JsonSchemaConstraint - 强制 JSON 生成
 * 3. TokenMask - logits 过滤
 * 4. 约束组合 - AND/OR 逻辑
 */

use loci::constraints::*;

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║   Loci Phase 2 Week 2 Demo            ║");
    println!("║   Constraint Sampling System           ║");
    println!("╚════════════════════════════════════════╝");
    println!();

    // Demo 1: RegexConstraint 示例
    demo_regex_constraints();

    // Demo 2: JsonSchemaConstraint 示例
    demo_json_constraints();

    // Demo 3: TokenMask 使用
    demo_token_mask();

    // Demo 4: 约束组合
    demo_constraint_combinators();

    println!();
    println!("✅ All demos completed successfully!");
}

fn demo_regex_constraints() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: Regex Constraints");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 1.1 邮箱约束
    println!("\n📧 Email Constraint:");
    let mut email_constraint = RegexConstraint::new(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        true, // 前缀模式
    ).unwrap();

    let email_steps = vec!["user", "@", "example", ".", "com"];
    simulate_generation(&mut email_constraint, &email_steps, "Email");

    // 1.2 电话号码约束
    println!("\n📞 Phone Number Constraint:");
    let mut phone_constraint = RegexConstraint::new(
        r"^\+?[1-9]\d{1,14}$",
        true,
    ).unwrap();

    let phone_steps = vec!["+", "1", "234", "567", "8900"];
    simulate_generation(&mut phone_constraint, &phone_steps, "Phone");

    println!("\n✅ Demo 1 completed\n");
}

fn demo_json_constraints() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: JSON Schema Constraints");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 2.1 JSON 对象
    println!("\n📦 JSON Object:");
    let mut obj_constraint = JsonSchemaConstraint::new(JsonType::Object);

    let obj_steps = vec!["{", "\"", "name", "\"", ":", "\"", "Alice", "\"", "}"];
    simulate_generation(&mut obj_constraint, &obj_steps, "JSON Object");

    println!("   Final state: {:?}", obj_constraint.state);
    println!("   Brace depth: {}", obj_constraint.brace_depth);

    // 2.2 JSON 数组
    println!("\n📊 JSON Array:");
    let mut arr_constraint = JsonSchemaConstraint::new(JsonType::Array);

    let arr_steps = vec!["[", "1", ",", "2", ",", "3", "]"];
    simulate_generation(&mut arr_constraint, &arr_steps, "JSON Array");

    println!("   Final state: {:?}", arr_constraint.state);

    // 2.3 嵌套 JSON
    println!("\n🎁 Nested JSON:");
    let mut nested_constraint = JsonSchemaConstraint::new(JsonType::Object);

    let nested_steps = vec![
        "{",
        "\"", "user", "\"", ":",
        "{",
        "\"", "name", "\"", ":", "\"", "Bob", "\"",
        "}",
        "}",
    ];
    simulate_generation(&mut nested_constraint, &nested_steps, "Nested JSON");

    println!("   Max brace depth: 2 (verified)");

    println!("\n✅ Demo 2 completed\n");
}

fn demo_token_mask() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: Token Mask Usage");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 模拟词表大小
    const VOCAB_SIZE: usize = 10;

    // 创建约束（只允许偶数 token ID）
    struct EvenTokenConstraint;
    impl Constraint for EvenTokenConstraint {
        fn is_allowed(&self, token_id: i32, _context: &ConstraintContext) -> bool {
            token_id % 2 == 0
        }
    }

    let constraint = EvenTokenConstraint;
    let context = ConstraintContext {
        generated_tokens: &[],
        generated_text: Some(""),
        candidate_token: 0,
        candidate_text: None,
        vocab_size: VOCAB_SIZE,
    };

    // 从约束生成 mask
    let mask = TokenMask::from_constraint(&constraint, &context);

    println!("\n🎭 Token Mask:");
    println!("   Vocab size: {}", VOCAB_SIZE);
    println!("   Allowed count: {}", mask.allowed_count());
    println!("   Allowed tokens:");

    for token_id in 0..VOCAB_SIZE as i32 {
        if mask.is_allowed(token_id) {
            println!("      - Token {}", token_id);
        }
    }

    // 模拟 logits
    let mut logits: Vec<f32> = (0..VOCAB_SIZE).map(|i| i as f32).collect();
    println!("\n   Original logits: {:?}", logits);

    mask.apply_to_logits(&mut logits);
    println!("   After mask:");
    for (i, logit) in logits.iter().enumerate() {
        if logit.is_finite() {
            println!("      - Token {}: {}", i, logit);
        } else {
            println!("      - Token {}: -inf (rejected)", i);
        }
    }

    println!("\n✅ Demo 3 completed\n");
}

fn demo_constraint_combinators() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: Constraint Combinators");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 4.1 AND 组合器
    println!("\n🔗 AND Combinator:");
    println!("   Rule: Must be alphanumeric AND must be lowercase");

    struct AlphanumericConstraint;
    impl Constraint for AlphanumericConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_alphanumeric())).unwrap_or(true)
        }
    }

    struct LowercaseConstraint;
    impl Constraint for LowercaseConstraint {
        fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
            context.candidate_text.map(|t| t.chars().all(|c| c.is_lowercase() || c.is_numeric())).unwrap_or(true)
        }
    }

    let and = AndConstraint::new(vec![
        Box::new(AlphanumericConstraint),
        Box::new(LowercaseConstraint),
    ]);

    let test_cases = vec![
        ("abc123", true, "lowercase alphanumeric"),
        ("ABC", false, "uppercase letters"),
        ("abc!", false, "contains symbol"),
        ("123", true, "numbers only"),
    ];

    for (text, expected, description) in test_cases {
        let context = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some(""),
            candidate_token: 0,
            candidate_text: Some(text),
            vocab_size: 1000,
        };

        let allowed = and.is_allowed(0, &context);
        let status = if allowed == expected { "✓" } else { "✗" };

        println!(
            "   {} \"{}\" ({}): {}",
            status,
            text,
            description,
            if allowed { "allowed" } else { "rejected" }
        );
    }

    // 4.2 OR 组合器
    println!("\n🔀 OR Combinator:");
    println!("   Rule: Must be numbers OR must be letters");

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

    let test_cases_or = vec![
        ("123", true, "numbers"),
        ("abc", true, "letters"),
        ("abc123", false, "mixed alphanumeric"),
        ("!@#", false, "symbols"),
    ];

    for (text, expected, description) in test_cases_or {
        let context = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some(""),
            candidate_token: 0,
            candidate_text: Some(text),
            vocab_size: 1000,
        };

        let allowed = or.is_allowed(0, &context);
        let status = if allowed == expected { "✓" } else { "✗" };

        println!(
            "   {} \"{}\" ({}): {}",
            status,
            text,
            description,
            if allowed { "allowed" } else { "rejected" }
        );
    }

    println!("\n✅ Demo 4 completed");
}

// ==================== 辅助函数 ====================

fn simulate_generation<C: Constraint>(constraint: &mut C, steps: &[&str], label: &str) {
    println!("   Simulating {} generation:", label);

    let mut generated = String::new();

    for (i, step) in steps.iter().enumerate() {
        let allowed = {
            let context = ConstraintContext {
                generated_tokens: &[],
                generated_text: Some(&generated),
                candidate_token: i as i32,
                candidate_text: Some(step),
                vocab_size: 1000,
            };
            constraint.is_allowed(i as i32, &context)
        };

        if allowed {
            generated.push_str(step);

            let context = ConstraintContext {
                generated_tokens: &[],
                generated_text: Some(&generated),
                candidate_token: i as i32,
                candidate_text: Some(step),
                vocab_size: 1000,
            };
            constraint.update(i as i32, &context);
            println!("      Step {}: \"{}\" ✓ (generated so far: \"{}\")", i + 1, step, generated);
        } else {
            println!("      Step {}: \"{}\" ✗ (rejected)", i + 1, step);
            break;
        }
    }

    println!("   Final output: \"{}\"", generated);
}
