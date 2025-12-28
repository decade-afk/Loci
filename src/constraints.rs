/**
 * Phase 2 Week 2: Constraint Sampling 核心实现
 *
 * 特性：
 * 1. Constraint Trait：统一的约束接口
 * 2. RegexConstraint：正则表达式约束
 * 3. JsonSchemaConstraint：简单 JSON 模式约束
 * 4. TokenMask：高效的 logits 过滤
 * 5. 状态追踪：支持有状态约束（如 JSON 嵌套深度）
 *
 * 设计原则：
 * - 零分配热路径：约束检查避免堆分配
 * - 可组合性：支持多个约束组合（AND/OR）
 * - 性能优先：使用 bitmap 加速 token 过滤
 */

use std::collections::HashSet;
use anyhow::Result;

// ==================== 核心 Trait 定义 ====================

/// 约束上下文（传递给约束的完整信息）
#[derive(Debug)]
pub struct ConstraintContext<'a> {
    /// 已生成的 token 序列
    pub generated_tokens: &'a [i32],

    /// 已生成的文本（如果可用）
    pub generated_text: Option<&'a str>,

    /// 当前候选 token ID
    pub candidate_token: i32,

    /// 当前候选 token 文本（如果可用）
    pub candidate_text: Option<&'a str>,

    /// 词表大小
    pub vocab_size: usize,
}

/// 约束 Trait：判断 token 是否符合约束
///
/// # 设计说明
/// - `is_allowed`: 纯函数检查，不修改状态
/// - `update`: 可选的状态更新（用于有状态约束）
/// - `reset`: 重置约束状态（新会话开始时调用）
pub trait Constraint: Send + Sync {
    /// 检查候选 token 是否被允许
    ///
    /// # 参数
    /// - `token_id`: 候选 token ID
    /// - `context`: 约束上下文
    ///
    /// # 返回
    /// - `true`: 允许该 token
    /// - `false`: 拒绝该 token
    fn is_allowed(&self, token_id: i32, context: &ConstraintContext) -> bool;

    /// 更新约束内部状态（可选）
    ///
    /// 在采样器确认选择某个 token 后调用
    fn update(&mut self, _token_id: i32, _context: &ConstraintContext) {
        // 默认无状态，无需更新
    }

    /// 重置约束状态（新会话开始）
    fn reset(&mut self) {
        // 默认无状态，无需重置
    }

    /// 获取约束描述（用于调试）
    fn description(&self) -> &str {
        "Generic Constraint"
    }

    /// 批量过滤：生成允许的 token 集合（可选优化）
    ///
    /// 默认实现逐个调用 `is_allowed`，子类可优化
    fn allowed_tokens(&self, _context: &ConstraintContext) -> Option<HashSet<i32>> {
        None // 默认不实现批量过滤
    }
}

// ==================== Token Mask（高效过滤） ====================

/// Token Mask：使用 bitmap 加速 logits 过滤
///
/// 在采样前预先计算允许的 token 集合，避免每次都调用约束
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// 允许的 token 位图（vocab_size 位）
    pub allowed: Vec<bool>,

    /// 允许的 token 数量
    pub allowed_count: usize,
}

impl TokenMask {
    /// 创建全允许的 mask
    pub fn new_allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![true; vocab_size],
            allowed_count: vocab_size,
        }
    }

    /// 创建全拒绝的 mask
    pub fn new_deny_all(vocab_size: usize) -> Self {
        Self {
            allowed: vec![false; vocab_size],
            allowed_count: 0,
        }
    }

    /// 从约束生成 mask
    pub fn from_constraint(
        constraint: &dyn Constraint,
        context: &ConstraintContext,
    ) -> Self {
        let vocab_size = context.vocab_size;

        // 尝试批量优化
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

        // 回退到逐个检查
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

    /// 检查 token 是否被允许
    #[inline]
    pub fn is_allowed(&self, token_id: i32) -> bool {
        self.allowed.get(token_id as usize).copied().unwrap_or(false)
    }

    /// 获取允许的 token 数量
    pub fn allowed_count(&self) -> usize {
        self.allowed_count
    }

    /// 应用 mask 到 logits（将不允许的 token 设置为 -inf）
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, &allowed) in self.allowed.iter().enumerate() {
            if !allowed && i < logits.len() {
                logits[i] = f32::NEG_INFINITY;
            }
        }
    }
}

// ==================== 内置约束：Regex ====================

/// 正则表达式约束
///
/// # 工作原理
/// 1. 维护当前已生成文本
/// 2. 对每个候选 token，模拟追加后检查是否匹配正则
/// 3. 拒绝所有不可能匹配的候选
///
/// # 限制
/// - 性能开销较大（每个 token 都需要正则匹配）
/// - 适合短文本约束（如邮箱、电话号码）
pub struct RegexConstraint {
    /// 正则表达式
    pattern: regex::Regex,

    /// 当前已生成的文本
    pub current_text: String,

    /// 是否启用前缀匹配模式
    /// - true: 允许部分匹配（生成过程中）
    /// - false: 要求完全匹配
    prefix_mode: bool,
}

impl RegexConstraint {
    /// 创建新的正则约束
    ///
    /// # 参数
    /// - `pattern`: 正则表达式字符串
    /// - `prefix_mode`: 是否启用前缀匹配
    pub fn new(pattern: &str, prefix_mode: bool) -> Result<Self> {
        let pattern = regex::Regex::new(pattern)?;

        Ok(Self {
            pattern,
            current_text: String::new(),
            prefix_mode,
        })
    }

    /// 检查文本是否部分匹配正则（前缀）
    fn is_partial_match(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        // 尝试从字符串开头匹配
        if let Some(mat) = self.pattern.find(text) {
            // 如果匹配起始位置是 0，说明是前缀匹配
            return mat.start() == 0;
        }

        // 如果完全匹配失败，检查是否可能是部分匹配
        // （简化版：只要不完全不匹配就允许）
        // TODO: 更精确的部分匹配算法
        true
    }
}

impl Constraint for RegexConstraint {
    fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
        // 获取候选 token 的文本
        let candidate_text = match context.candidate_text {
            Some(text) => text,
            None => return true, // 没有文本信息，默认允许
        };

        // 模拟追加候选文本
        let simulated_text = format!("{}{}", self.current_text, candidate_text);

        if self.prefix_mode {
            // 前缀模式：允许部分匹配
            self.is_partial_match(&simulated_text)
        } else {
            // 完全匹配模式：必须完整匹配
            self.pattern.is_match(&simulated_text)
        }
    }

    fn update(&mut self, _token_id: i32, context: &ConstraintContext) {
        // 更新当前文本
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

// ==================== 内置约束：JSON Schema ====================

/// 简单 JSON 模式约束
///
/// # 支持的模式
/// 1. `{"type": "object"}` - 强制生成对象 `{...}`
/// 2. `{"type": "array"}` - 强制生成数组 `[...]`
/// 3. `{"type": "string"}` - 强制生成字符串 `"..."`
/// 4. `{"type": "number"}` - 强制生成数字
///
/// # 状态机
/// - `Idle` -> `InObject` -> `ExpectKey` -> `ExpectValue` -> ...
/// - 追踪括号嵌套深度
/// - 验证 JSON 格式正确性
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonState {
    Idle,           // 初始状态
    InObject,       // 在对象内 `{`
    InArray,        // 在数组内 `[`
    ExpectKey,      // 期待对象键
    ExpectColon,    // 期待冒号
    ExpectValue,    // 期待值
    InString,       // 在字符串内
    Complete,       // 完成
}

pub struct JsonSchemaConstraint {
    /// 目标 JSON 类型
    target_type: JsonType,

    /// 当前状态机状态
    pub state: JsonState,

    /// 当前已生成的文本
    pub current_text: String,

    /// 括号嵌套深度
    pub brace_depth: i32,
    pub bracket_depth: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonType {
    Object,
    Array,
    String,
    Number,
    Boolean,
    Null,
}

impl JsonSchemaConstraint {
    /// 创建新的 JSON 约束
    pub fn new(target_type: JsonType) -> Self {
        Self {
            target_type,
            state: JsonState::Idle,
            current_text: String::new(),
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    /// 检查字符是否合法
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
                true // 字符串内任意字符（简化版，实际需要转义处理）
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

    /// 更新状态机
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

impl Constraint for JsonSchemaConstraint {
    fn is_allowed(&self, _token_id: i32, context: &ConstraintContext) -> bool {
        let candidate_text = match context.candidate_text {
            Some(text) => text,
            None => return true,
        };

        // 检查每个字符是否合法
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

            // 更新状态机
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

// ==================== 约束组合器 ====================

/// AND 组合器：所有约束都必须满足
pub struct AndConstraint {
    constraints: Vec<Box<dyn Constraint>>,
}

impl AndConstraint {
    pub fn new(constraints: Vec<Box<dyn Constraint>>) -> Self {
        Self { constraints }
    }
}

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

/// OR 组合器：至少一个约束满足
pub struct OrConstraint {
    constraints: Vec<Box<dyn Constraint>>,
}

impl OrConstraint {
    pub fn new(constraints: Vec<Box<dyn Constraint>>) -> Self {
        Self { constraints }
    }
}

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

// ==================== 测试模块 ====================

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

        // "user" 应该被允许（是 email 的前缀）
        assert!(constraint.is_allowed(0, &context));

        constraint.update(0, &context);

        let context2 = ConstraintContext {
            generated_tokens: &[],
            generated_text: Some("user"),
            candidate_token: 1,
            candidate_text: Some("@"),
            vocab_size: 1000,
        };

        // "@" 应该被允许
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

        // "{" 应该被允许
        assert!(constraint.is_allowed(0, &context));

        constraint.update(0, &context);
        assert_eq!(constraint.state, JsonState::InObject);
        assert_eq!(constraint.brace_depth, 1);
    }
}
