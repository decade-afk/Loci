# Loci 插件开发指南

本指南介绍如何为 Loci 开发插件，让第三方应用可以轻松集成和扩展功能。

**版本**: MVP v0.2.0
**最后更新**: 2026-01-02

---

## 目录

- [插件系统概述](#插件系统概述)
- [快速开始](#快速开始)
- [Plugin Trait](#plugin-trait)
- [内置插件示例](#内置插件示例)
- [高级用法](#高级用法)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 插件系统概述

Loci 的插件系统允许您在推理流程的关键点插入自定义逻辑：

```
                    ┌───────────────┐
                    │  User Prompt  │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ pre_generate  │  ← 插件钩子 1
                    │   (plugins)   │
                    └───────┬───────┘
                            │
                  ┌─────────▼─────────┐
                  │  Inference Engine │
                  └─────────┬─────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
    ┌───────▼────────┐            ┌────────▼────────┐
    │   on_token     │  ← 钩子 2  │  Final Response │
    │   (streaming)  │            └────────┬────────┘
    └────────────────┘                     │
                                   ┌───────▼────────┐
                                   │ post_generate  │  ← 钩子 3
                                   │   (plugins)    │
                                   └───────┬────────┘
                                           │
                                   ┌───────▼────────┐
                                   │ Final Output   │
                                   └────────────────┘
```

### 钩子说明

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `pre_generate` | 推理之前 | 修改提示词、添加模板、上下文注入 |
| `on_token` | 生成每个 token 时 | 实时过滤、转换、统计 |
| `post_generate` | 推理完成后 | 后处理、格式化、过滤敏感内容 |

---

## 快速开始

### 1. 创建简单插件

```rust
use loci::plugin::Plugin;
use loci::error::Result;

struct MyPlugin {
    name: String,
}

impl Plugin for MyPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        // 在提示词前添加系统指令
        Ok(format!("System: Be helpful.\n\nUser: {}", prompt))
    }
}
```

### 2. 注册插件

```rust
use loci::prelude::*;

fn main() -> Result<()> {
    let config = ModelConfig::new("model.gguf");
    let mut engine = InferenceEngine::new(config)?;

    // 注册插件
    let plugin = MyPlugin {
        name: "my_plugin".to_string(),
    };
    engine.plugin_manager_mut().register(plugin)?;

    // 使用引擎（插件自动生效）
    let params = loci::inference::GenerationParams::default();
    let response = engine.generate("Hello", params)?;

    Ok(())
}
```

---

## Plugin Trait

### 完整接口

```rust
pub trait Plugin: Send + Sync {
    /// 获取插件名称（必须实现）
    fn name(&self) -> &str;

    /// 获取插件版本（必须实现）
    fn version(&self) -> &str;

    /// 初始化插件（可选）
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// 推理前处理（可选）
    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(prompt.to_string())
    }

    /// 推理后处理（可选）
    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(response.to_string())
    }

    /// Token 流式处理（可选）
    fn on_token(&self, token: &str) -> Result<String> {
        Ok(token.to_string())
    }

    /// 清理资源（可选）
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
```

### 方法详解

#### `init()`

在插件注册时调用，用于初始化资源。

```rust
fn init(&mut self) -> Result<()> {
    println!("Plugin {} initialized", self.name());
    // 加载配置、连接数据库等
    Ok(())
}
```

#### `pre_generate()`

在推理前修改提示词。

**典型用途**:
- 添加聊天模板
- 注入系统提示
- 提示词优化

```rust
fn pre_generate(&self, prompt: &str) -> Result<String> {
    let template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
    Ok(template.replace("{}", prompt))
}
```

#### `post_generate()`

在推理后处理响应。

**典型用途**:
- 格式化输出
- 过滤敏感内容
- 添加元数据

```rust
fn post_generate(&self, response: &str) -> Result<String> {
    // 移除特殊标记
    let cleaned = response
        .replace("<|im_end|>", "")
        .trim()
        .to_string();
    Ok(cleaned)
}
```

#### `on_token()`

处理每个生成的 token（仅在流式模式）。

**典型用途**:
- 实时过滤
- 统计分析
- 流式转换

```rust
fn on_token(&self, token: &str) -> Result<String> {
    // 过滤特定词汇
    if token.contains("badword") {
        Ok("[filtered]".to_string())
    } else {
        Ok(token.to_string())
    }
}
```

#### `cleanup()`

在插件卸载时调用。

```rust
fn cleanup(&mut self) -> Result<()> {
    println!("Plugin {} cleaned up", self.name());
    // 关闭文件、断开连接等
    Ok(())
}
```

---

## 内置插件示例

### 1. 提示词模板插件

```rust
use loci::plugin::Plugin;
use loci::error::Result;

pub struct PromptTemplatePlugin {
    template: String,
}

impl PromptTemplatePlugin {
    pub fn new(template: String) -> Self {
        Self { template }
    }

    // 常用模板
    pub fn chatml() -> Self {
        Self::new(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n".to_string()
        )
    }

    pub fn alpaca() -> Self {
        Self::new(
            "### Instruction:\n{prompt}\n\n### Response:\n".to_string()
        )
    }
}

impl Plugin for PromptTemplatePlugin {
    fn name(&self) -> &str {
        "prompt_template"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(self.template.replace("{prompt}", prompt))
    }
}
```

**使用方法**:

```rust
// ChatML 模板
let plugin = PromptTemplatePlugin::chatml();
engine.plugin_manager_mut().register(plugin)?;

// 自定义模板
let custom = PromptTemplatePlugin::new(
    "Q: {prompt}\nA: ".to_string()
);
engine.plugin_manager_mut().register(custom)?;
```

### 2. 内容过滤插件

```rust
pub struct ContentFilterPlugin {
    blocked_words: Vec<String>,
}

impl ContentFilterPlugin {
    pub fn new(blocked_words: Vec<String>) -> Self {
        Self { blocked_words }
    }
}

impl Plugin for ContentFilterPlugin {
    fn name(&self) -> &str {
        "content_filter"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        let mut filtered = response.to_string();
        for word in &self.blocked_words {
            filtered = filtered.replace(word, "[FILTERED]");
        }
        Ok(filtered)
    }

    fn on_token(&self, token: &str) -> Result<String> {
        for word in &self.blocked_words {
            if token.contains(word) {
                return Ok("[*]".to_string());
            }
        }
        Ok(token.to_string())
    }
}
```

### 3. 日志记录插件

```rust
use std::fs::OpenOptions;
use std::io::Write;

pub struct LoggingPlugin {
    log_file: Option<std::fs::File>,
}

impl LoggingPlugin {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            log_file: Some(file),
        })
    }
}

impl Plugin for LoggingPlugin {
    fn name(&self) -> &str {
        "logging"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if let Some(ref mut file) = self.log_file.as_ref() {
            let mut f = file.try_clone().unwrap();
            writeln!(f, "[PROMPT] {}", prompt).ok();
        }
        Ok(prompt.to_string())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        if let Some(ref mut file) = self.log_file.as_ref() {
            let mut f = file.try_clone().unwrap();
            writeln!(f, "[RESPONSE] {}", response).ok();
            writeln!(f, "---").ok();
        }
        Ok(response.to_string())
    }
}
```

### 4. 统计分析插件

```rust
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct Stats {
    pub total_prompts: usize,
    pub total_tokens: usize,
    pub total_responses: usize,
}

pub struct StatsPlugin {
    stats: Arc<Mutex<Stats>>,
}

impl StatsPlugin {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(Stats::default())),
        }
    }

    pub fn get_stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }
}

impl Plugin for StatsPlugin {
    fn name(&self) -> &str {
        "stats"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_prompts += 1;
        Ok(prompt.to_string())
    }

    fn on_token(&self, token: &str) -> Result<String> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_tokens += 1;
        Ok(token.to_string())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        let mut stats = self.stats.lock().unwrap();
        stats.total_responses += 1;
        Ok(response.to_string())
    }
}
```

---

## 高级用法

### 插件链

多个插件会按注册顺序依次执行：

```rust
// 插件链：模板 -> 过滤 -> 日志
engine.plugin_manager_mut().register(
    PromptTemplatePlugin::chatml()
)?;

engine.plugin_manager_mut().register(
    ContentFilterPlugin::new(vec!["badword".to_string()])
)?;

engine.plugin_manager_mut().register(
    LoggingPlugin::new("inference.log")?
)?;

// 执行顺序：
// 1. 模板插件处理提示词
// 2. 过滤插件检查内容
// 3. 日志插件记录
```

### 条件插件

```rust
struct ConditionalPlugin {
    condition: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl Plugin for ConditionalPlugin {
    fn name(&self) -> &str {
        "conditional"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if (self.condition)(prompt) {
            Ok(format!("[SPECIAL] {}", prompt))
        } else {
            Ok(prompt.to_string())
        }
    }
}
```

### 状态保持插件

```rust
struct CachingPlugin {
    cache: Arc<Mutex<HashMap<String, String>>>,
}

impl Plugin for CachingPlugin {
    fn name(&self) -> &str {
        "cache"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        let cache = self.cache.lock().unwrap();
        if let Some(cached) = cache.get(prompt) {
            println!("Cache hit!");
            // 可以在这里实现缓存逻辑
        }
        Ok(prompt.to_string())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        // 保存到缓存
        Ok(response.to_string())
    }
}
```

---

## 最佳实践

### 1. 错误处理

```rust
fn pre_generate(&self, prompt: &str) -> Result<String> {
    // 使用 ? 传播错误
    let processed = self.process(prompt)?;

    // 或者包装错误
    self.validate(&processed)
        .map_err(|e| LociError::Other(format!("Validation failed: {}", e)))?;

    Ok(processed)
}
```

### 2. 性能优化

```rust
// ✅ 好：最小化分配
fn on_token(&self, token: &str) -> Result<String> {
    if self.should_filter(token) {
        Ok("[*]".to_string())
    } else {
        Ok(token.to_string())  // 重用字符串
    }
}

// ❌ 差：不必要的处理
fn on_token(&self, token: &str) -> Result<String> {
    let upper = token.to_uppercase();
    let lower = upper.to_lowercase();
    Ok(lower)  // 无意义的转换
}
```

### 3. 线程安全

```rust
// 使用 Arc<Mutex<>> 共享状态
pub struct SharedStatePlugin {
    state: Arc<Mutex<State>>,
}

// 确保 Send + Sync
unsafe impl Send for SharedStatePlugin {}
unsafe impl Sync for SharedStatePlugin {}
```

### 4. 资源管理

```rust
impl Plugin for ResourcePlugin {
    fn init(&mut self) -> Result<()> {
        self.connection = Some(open_connection()?);
        Ok(())
    }

    fn cleanup(&mut self) -> Result<()> {
        if let Some(conn) = self.connection.take() {
            conn.close()?;
        }
        Ok(())
    }
}
```

---

## 常见问题

### Q: 插件执行顺序如何确定？

A: 按注册顺序执行。先注册的先执行。

### Q: 一个插件可以停止执行链吗？

A: 不能直接停止，但可以返回错误来中断：

```rust
fn pre_generate(&self, prompt: &str) -> Result<String> {
    if prompt.contains("stop") {
        return Err(LociError::Other("Blocked by plugin".into()));
    }
    Ok(prompt.to_string())
}
```

### Q: 如何卸载插件？

```rust
engine.plugin_manager_mut().unregister("plugin_name")?;
```

### Q: 插件可以访问模型吗？

A: 当前版本不能直接访问。插件只能处理文本流。

### Q: 如何调试插件？

```rust
fn pre_generate(&self, prompt: &str) -> Result<String> {
    eprintln!("[DEBUG] Plugin input: {}", prompt);
    let output = self.process(prompt)?;
    eprintln!("[DEBUG] Plugin output: {}", output);
    Ok(output)
}
```

---

## 完整示例

```rust
use loci::prelude::*;
use loci::plugin::Plugin;
use std::sync::{Arc, Mutex};

// 自定义插件
struct MyCustomPlugin {
    counter: Arc<Mutex<usize>>,
}

impl Plugin for MyCustomPlugin {
    fn name(&self) -> &str {
        "custom"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!("Initializing custom plugin");
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("Enhanced: {}", prompt))
    }

    fn on_token(&self, token: &str) -> Result<String> {
        let mut count = self.counter.lock().unwrap();
        *count += 1;
        Ok(token.to_string())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        let count = self.counter.lock().unwrap();
        println!("Generated {} tokens", *count);
        Ok(response.to_string())
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("Cleaning up custom plugin");
        Ok(())
    }
}

fn main() -> Result<()> {
    // 创建引擎
    let config = ModelConfig::new("model.gguf");
    let mut engine = InferenceEngine::new(config)?;

    // 注册插件
    let plugin = MyCustomPlugin {
        counter: Arc::new(Mutex::new(0)),
    };
    engine.plugin_manager_mut().register(plugin)?;

    // 使用引擎
    let params = loci::inference::GenerationParams::default();
    let response = engine.generate("Hello", params)?;

    println!("Response: {}", response);

    Ok(())
}
```

---

## 更多资源

- [API Reference](API_REFERENCE.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [GitHub Issues](https://github.com/decade-afk/loci/issues)

---

**维护者**: decade-afk
**许可证**: MIT OR Apache-2.0
