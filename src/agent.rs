/**
 * Agent 系统模块 - 支持多模型池和智能 Agent
 *
 * 本模块提供：
 * - 模型池管理：同时加载和管理多个 AI 模型
 * - Agent 系统：为不同任务配置专用 Agent（模型 + 提示词）
 * - 会话管理：支持多轮对话的上下文保持
 * - 智能路由：根据任务类型自动选择最合适的 Agent
 *
 * 架构设计：
 * ```
 * AgentSystem
 * ├── ModelPool (模型池)
 * │   ├── Model A (通用对话)
 * │   ├── Model B (代码生成)
 * │   └── Model C (创意写作)
 * ├── Agents (Agent 集合)
 * │   ├── Agent "assistant" -> Model A + 通用提示词
 * │   ├── Agent "coder" -> Model B + 代码提示词
 * │   └── Agent "writer" -> Model C + 创意提示词
 * └── Sessions (会话管理)
 *     ├── Session 1 -> Agent "assistant"
 *     └── Session 2 -> Agent "writer"
 * ```
 */

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, AddBos, Special},
    model::params::LlamaModelParams,
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
    token::LlamaToken,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::num::NonZeroU32;

// ==================== 配置结构 ====================

/**
 * 模型配置
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 模型唯一标识符
    pub model_id: String,

    /// 模型文件路径
    pub model_path: String,

    /// 上下文大小
    pub context_size: u32,

    /// GPU 层数
    pub gpu_layers: u32,

    /// CPU 线程数
    pub threads: u32,
}

/**
 * Agent 配置
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent 唯一标识符
    pub agent_id: String,

    /// 使用的模型 ID
    pub model_id: String,

    /// 系统提示词（定义 Agent 的角色和行为）
    pub system_prompt: String,

    /// 默认采样参数
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,

    /// Agent 描述（用于 UI 显示）
    pub description: String,
}

/**
 * 生成请求
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenerateRequest {
    /// 使用的 Agent ID
    pub agent_id: String,

    /// 用户输入
    pub prompt: String,

    /// 会话 ID（可选，用于多轮对话）
    pub session_id: Option<String>,

    /// 最大生成 token 数
    pub max_tokens: Option<u32>,

    /// 临时覆盖温度
    pub temperature: Option<f32>,

    /// 停止词
    pub stop_words: Option<Vec<String>>,
}

/**
 * 生成响应
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenerateResponse {
    /// Agent ID
    pub agent_id: String,

    /// 生成的内容
    pub content: String,

    /// Token 数量
    pub tokens_generated: u32,

    /// 是否提前停止
    pub stopped_early: bool,

    /// 会话 ID
    pub session_id: Option<String>,
}

// ==================== 内部状态 ====================

/**
 * 加载的模型实例
 */
struct LoadedModel {
    /// 模型配置
    config: ModelConfig,

    /// 模型实例
    model: Arc<LlamaModel>,

    /// 加载时间戳
    loaded_at: std::time::SystemTime,
}

/**
 * 会话状态
 */
struct Session {
    /// 会话 ID
    session_id: String,

    /// 使用的 Agent ID
    agent_id: String,

    /// 历史 token（用于上下文保持）
    history_tokens: Vec<LlamaToken>,

    /// 最后活跃时间
    last_active: std::time::SystemTime,
}

// ==================== Agent 系统主结构 ====================

/**
 * Agent 系统
 *
 * 管理多个模型和 Agent，支持：
 * - 模型池：按需加载和卸载模型
 * - Agent 管理：创建、删除、查询 Agent
 * - 会话管理：多轮对话上下文保持
 * - 并发推理：多个 Agent 可以同时工作
 */
pub struct AgentSystem {
    /// Llama 后端（全局唯一）
    backend: LlamaBackend,

    /// 模型池：model_id -> LoadedModel
    models: RwLock<HashMap<String, LoadedModel>>,

    /// Agent 配置：agent_id -> AgentConfig
    agents: RwLock<HashMap<String, AgentConfig>>,

    /// 活跃会话：session_id -> Session
    sessions: RwLock<HashMap<String, Session>>,
}

impl AgentSystem {
    /**
     * 创建新的 Agent 系统
     */
    pub fn new() -> Result<Self, String> {
        let backend = LlamaBackend::init()
            .map_err(|e| format!("初始化后端失败: {}", e))?;

        Ok(Self {
            backend,
            models: RwLock::new(HashMap::new()),
            agents: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
        })
    }

    // ==================== 模型池管理 ====================

    /**
     * 加载模型到模型池
     *
     * 如果模型已存在，会先卸载旧模型再加载新的。
     */
    pub fn load_model(&self, config: ModelConfig) -> Result<(), String> {
        let model_path = PathBuf::from(&config.model_path);
        if !model_path.exists() {
            return Err(format!("模型文件不存在: {}", config.model_path));
        }

        // 配置模型参数
        let mut model_params = LlamaModelParams::default();
        if config.gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(config.gpu_layers);
        }

        // 加载模型
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)
            .map_err(|e| format!("加载模型失败: {}", e))?;

        // 添加到模型池
        let mut models = self.models.write();
        models.insert(config.model_id.clone(), LoadedModel {
            config,
            model: Arc::new(model),
            loaded_at: std::time::SystemTime::now(),
        });

        Ok(())
    }

    /**
     * 从模型池卸载模型
     */
    pub fn unload_model(&self, model_id: &str) -> Result<(), String> {
        // 检查是否有 Agent 正在使用这个模型
        let agents = self.agents.read();
        let using_agents: Vec<_> = agents.values()
            .filter(|a| a.model_id == model_id)
            .map(|a| a.agent_id.clone())
            .collect();

        if !using_agents.is_empty() {
            return Err(format!(
                "模型 {} 正在被以下 Agent 使用，无法卸载: {:?}",
                model_id, using_agents
            ));
        }

        // 卸载模型
        let mut models = self.models.write();
        if models.remove(model_id).is_none() {
            return Err(format!("模型 {} 不存在", model_id));
        }

        Ok(())
    }

    /**
     * 列出所有已加载的模型
     */
    pub fn list_models(&self) -> Vec<ModelConfig> {
        self.models.read()
            .values()
            .map(|m| m.config.clone())
            .collect()
    }

    /**
     * 检查模型是否已加载
     */
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.models.read().contains_key(model_id)
    }

    // ==================== Agent 管理 ====================

    /**
     * 创建新的 Agent
     */
    pub fn create_agent(&self, config: AgentConfig) -> Result<(), String> {
        // 验证模型是否已加载
        if !self.is_model_loaded(&config.model_id) {
            return Err(format!("模型 {} 未加载", config.model_id));
        }

        // 添加 Agent
        let mut agents = self.agents.write();
        if agents.contains_key(&config.agent_id) {
            return Err(format!("Agent {} 已存在", config.agent_id));
        }

        agents.insert(config.agent_id.clone(), config);
        Ok(())
    }

    /**
     * 删除 Agent
     */
    pub fn delete_agent(&self, agent_id: &str) -> Result<(), String> {
        let mut agents = self.agents.write();
        if agents.remove(agent_id).is_none() {
            return Err(format!("Agent {} 不存在", agent_id));
        }

        // 清理相关会话
        let mut sessions = self.sessions.write();
        sessions.retain(|_, s| s.agent_id != agent_id);

        Ok(())
    }

    /**
     * 列出所有 Agent
     */
    pub fn list_agents(&self) -> Vec<AgentConfig> {
        self.agents.read().values().cloned().collect()
    }

    /**
     * 获取 Agent 配置
     */
    pub fn get_agent(&self, agent_id: &str) -> Option<AgentConfig> {
        self.agents.read().get(agent_id).cloned()
    }

    // ==================== 文本生成 ====================

    /**
     * 使用 Agent 生成文本（阻塞式，返回完整结果）
     *
     * 支持多轮对话（通过 session_id）：
     * - 如果提供 session_id，会加载历史上下文
     * - 生成后会自动更新会话的历史记录
     * - 自动管理上下文窗口，避免超出限制
     *
     * **注意**: 此方法会阻塞直到生成完成。如需流式输出，请使用 `generate_stream`。
     */
    pub fn generate(&self, request: AgentGenerateRequest) -> Result<AgentGenerateResponse, String> {
        // 获取 Agent 配置
        let agent_config = self.agents.read()
            .get(&request.agent_id)
            .cloned()
            .ok_or(format!("Agent {} 不存在", request.agent_id))?;

        // 获取模型
        let models = self.models.read();
        let loaded_model = models.get(&agent_config.model_id)
            .ok_or(format!("模型 {} 未加载", agent_config.model_id))?;

        // 加载历史上下文（如果提供了 session_id）
        let mut history_tokens = Vec::new();
        if let Some(ref session_id) = request.session_id {
            let sessions = self.sessions.read();
            if let Some(session) = sessions.get(session_id) {
                // 验证会话使用的是正确的 Agent
                if session.agent_id != request.agent_id {
                    return Err(format!(
                        "会话 {} 属于 Agent {}，不能用于 Agent {}",
                        session_id, session.agent_id, request.agent_id
                    ));
                }
                history_tokens = session.history_tokens.clone();
            } else {
                return Err(format!("会话 {} 不存在", session_id));
            }
        }

        // 构建当前轮次的提示词
        // 首轮对话：包含系统提示词
        // 后续对话：只包含用户输入（系统提示词在历史中）
        let current_prompt = if history_tokens.is_empty() {
            format!("{}\n\n{}", agent_config.system_prompt, request.prompt)
        } else {
            request.prompt.clone()
        };

        // 配置上下文
        let ctx_size = NonZeroU32::new(loaded_model.config.context_size)
            .ok_or("Context size must be greater than 0")?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));

        if loaded_model.config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(loaded_model.config.threads as i32)
                .with_n_threads_batch(loaded_model.config.threads as i32);
        }

        // 创建上下文
        let mut ctx = loaded_model.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Tokenize current prompt
        // 修复：只在首次对话时添加 BOS token，避免多轮对话中重复插入
        let add_bos = if history_tokens.is_empty() {
            AddBos::Always  // 首次对话需要 BOS
        } else {
            AddBos::Never   // 后续轮次不需要 BOS（历史中已有）
        };
        let current_tokens = loaded_model.model
            .str_to_token(&current_prompt, add_bos)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        if current_tokens.is_empty() {
            return Err("Tokenization result is empty".to_string());
        }

        // Merge history and current tokens
        let mut all_tokens = Vec::new();

        // 计算可用的上下文窗口大小（预留生成空间）
        let max_tokens = request.max_tokens.unwrap_or(512);
        let available_context = (loaded_model.config.context_size as usize)
            .saturating_sub(max_tokens as usize);

        // Intelligently truncate history：如果历史 + 当前超出窗口，从历史开头截断
        let total_needed = history_tokens.len() + current_tokens.len();
        if total_needed > available_context {
            let history_budget = available_context.saturating_sub(current_tokens.len());
            if history_budget > 0 && history_tokens.len() > history_budget {
                // 保留最近的历史（从末尾取）
                let skip = history_tokens.len() - history_budget;
                all_tokens.extend_from_slice(&history_tokens[skip..]);
            } else if history_budget > 0 {
                all_tokens.extend_from_slice(&history_tokens);
            }
            // 如果 history_budget 为 0，就完全不使用历史
        } else {
            // 空间足够，使用全部历史
            all_tokens.extend_from_slice(&history_tokens);
        }

        all_tokens.extend_from_slice(&current_tokens);

        // 创建批处理
        let n_ctx = loaded_model.config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // 添加所有 tokens 到批处理
        let last_index = all_tokens.len() - 1;
        for (i, token) in all_tokens.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("Failed to add token: {}", e))?;
        }

        // 解码提示词
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // 创建采样器
        let temperature = request.temperature.unwrap_or(agent_config.temperature);
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(agent_config.top_k as i32),
                LlamaSampler::top_p(agent_config.top_p, 1),
                LlamaSampler::dist(1234),
            ])
        };

        // 生成文本
        let mut result = String::new();
        let mut generated_tokens: Vec<LlamaToken> = Vec::new();
        let mut token_count = 0u32;
        let n_prompt = all_tokens.len();

        for i in 0..max_tokens {
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            // Save generated token
            generated_tokens.push(new_token_id);

            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("Token conversion failed: {}", e))?;

            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            batch.clear();
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode generated token: {}", e))?;
        }

        // Update session history if session_id provided
        if let Some(ref session_id) = request.session_id {
            let mut sessions = self.sessions.write();
            if let Some(session) = sessions.get_mut(session_id) {
                // 将当前对话的 tokens 添加到历史
                // 历史格式：[...旧历史] + [当前用户输入] + [AI回复]
                session.history_tokens.extend_from_slice(&current_tokens);
                session.history_tokens.extend_from_slice(&generated_tokens);

                // 更新活跃时间
                session.last_active = std::time::SystemTime::now();

                // 可选：限制历史长度，防止无限增长
                // 保留最近的 N 个 tokens（这里设为上下文窗口的 80%）
                let max_history = (loaded_model.config.context_size as usize * 80) / 100;
                if session.history_tokens.len() > max_history {
                    let remove_count = session.history_tokens.len() - max_history;
                    session.history_tokens.drain(0..remove_count);
                }
            }
        }

        Ok(AgentGenerateResponse {
            agent_id: request.agent_id,
            content: result.trim().to_string(),
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,
            session_id: request.session_id,
        })
    }

    /// Generate text using an Agent with streaming callbacks.
    ///
    /// Similar to `generate`, but with streaming support:
    /// - Invokes callback immediately for each generated token
    /// - Supports user interruption (callback returns Stop)
    /// - Automatically handles multi-turn dialogue and history
    ///
    /// # Parameters
    /// - `request`: Generation request
    /// - `callback`: Streaming callback (implements `StreamCallback` trait)
    ///
    /// # Returns
    /// - `AgentGenerateResponse`: Complete result with statistics
    ///
    /// # Example
    /// ```no_run
    /// use loci::{AgentSystem, AgentGenerateRequest, ConsoleCallback};
    ///
    /// let agent_system = AgentSystem::new()?;
    /// let request = AgentGenerateRequest {
    ///     agent_id: "assistant".to_string(),
    ///     prompt: "Hello".to_string(),
    ///     session_id: None,
    ///     max_tokens: Some(100),
    ///     temperature: None,
    ///     stop_words: None,
    /// };
    ///
    /// let response = agent_system.generate_stream(
    ///     request,
    ///     &mut ConsoleCallback::new(true)
    /// )?;
    /// # Ok::<(), String>(())
    /// ```
    pub fn generate_stream<C>(
        &self,
        request: AgentGenerateRequest,
        callback: &mut C,
    ) -> Result<AgentGenerateResponse, String>
    where
        C: crate::streaming::StreamCallback,
    {
        use crate::streaming::StreamControlFlow;

        let agent_config = self.agents.read()
            .get(&request.agent_id)
            .cloned()
            .ok_or(format!("Agent {} not found", request.agent_id))?;

        let models = self.models.read();
        let loaded_model = models.get(&agent_config.model_id)
            .ok_or(format!("Model {} not loaded", agent_config.model_id))?;

        // Load historical context if session_id provided
        let mut history_tokens = Vec::new();
        if let Some(ref session_id) = request.session_id {
            let sessions = self.sessions.read();
            if let Some(session) = sessions.get(session_id) {
                if session.agent_id != request.agent_id {
                    return Err(format!(
                        "Session {} belongs to Agent {}, cannot be used with Agent {}",
                        session_id, session.agent_id, request.agent_id
                    ));
                }
                history_tokens = session.history_tokens.clone();
            } else {
                return Err(format!("Session {} not found", session_id));
            }
        }

        // Build prompt for current turn
        let current_prompt = if history_tokens.is_empty() {
            format!("{}\n\n{}", agent_config.system_prompt, request.prompt)
        } else {
            request.prompt.clone()
        };

        // 配置上下文
        let ctx_size = NonZeroU32::new(loaded_model.config.context_size)
            .ok_or("Context size must be greater than 0")?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));

        if loaded_model.config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(loaded_model.config.threads as i32)
                .with_n_threads_batch(loaded_model.config.threads as i32);
        }

        // 创建上下文
        let mut ctx = loaded_model.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Tokenize current prompt
        let add_bos = if history_tokens.is_empty() {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let current_tokens = loaded_model.model
            .str_to_token(&current_prompt, add_bos)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        if current_tokens.is_empty() {
            return Err("Tokenization result is empty".to_string());
        }

        // Merge history and current tokens
        let mut all_tokens = Vec::new();
        let max_tokens = request.max_tokens.unwrap_or(512);
        let available_context = (loaded_model.config.context_size as usize)
            .saturating_sub(max_tokens as usize);

        // Intelligently truncate history
        let total_needed = history_tokens.len() + current_tokens.len();
        if total_needed > available_context {
            let history_budget = available_context.saturating_sub(current_tokens.len());
            if history_budget > 0 && history_tokens.len() > history_budget {
                let skip = history_tokens.len() - history_budget;
                all_tokens.extend_from_slice(&history_tokens[skip..]);
            } else if history_budget > 0 {
                all_tokens.extend_from_slice(&history_tokens);
            }
        } else {
            all_tokens.extend_from_slice(&history_tokens);
        }

        all_tokens.extend_from_slice(&current_tokens);

        // 创建批处理
        let n_ctx = loaded_model.config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // 添加所有 tokens 到批处理
        let last_index = all_tokens.len() - 1;
        for (i, token) in all_tokens.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("Failed to add token: {}", e))?;
        }

        // 解码提示词
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // 创建采样器
        let temperature = request.temperature.unwrap_or(agent_config.temperature);
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(agent_config.top_k as i32),
                LlamaSampler::top_p(agent_config.top_p, 1),
                LlamaSampler::dist(1234),
            ])
        };

        // Stream text generation
        let mut result = String::new();
        let mut generated_tokens: Vec<LlamaToken> = Vec::new();
        let mut token_count = 0u32;
        let n_prompt = all_tokens.len();

        for i in 0..max_tokens {
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            // Save generated token
            generated_tokens.push(new_token_id);

            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("Token conversion failed: {}", e))?;

            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            match callback.on_token(&token_str, new_token_id.0, i as usize) {
                StreamControlFlow::Continue => {}
                StreamControlFlow::Stop => {
                    break;
                }
            }

            // Check stop words
            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            batch.clear();
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode generated token: {}", e))?;
        }

        // Notify completion with temporary stats
        let stats = crate::streaming::StreamStats {
            generated_tokens: token_count as usize,
            total_tokens: all_tokens.len() + token_count as usize,
            ..Default::default()
        };
        callback.on_complete(&stats);

        // Update session history if session_id provided
        if let Some(ref session_id) = request.session_id {
            let mut sessions = self.sessions.write();
            if let Some(session) = sessions.get_mut(session_id) {
                session.history_tokens.extend_from_slice(&current_tokens);
                session.history_tokens.extend_from_slice(&generated_tokens);
                session.last_active = std::time::SystemTime::now();

                // Limit history length
                let max_history = (loaded_model.config.context_size as usize * 80) / 100;
                if session.history_tokens.len() > max_history {
                    let remove_count = session.history_tokens.len() - max_history;
                    session.history_tokens.drain(0..remove_count);
                }
            }
        }

        Ok(AgentGenerateResponse {
            agent_id: request.agent_id,
            content: result.trim().to_string(),
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,
            session_id: request.session_id,
        })
    }

    // ==================== 会话管理 ====================

    /**
     * 创建新会话
     */
    pub fn create_session(&self, agent_id: String) -> Result<String, String> {
        // 验证 Agent 存在
        if !self.agents.read().contains_key(&agent_id) {
            return Err(format!("Agent {} 不存在", agent_id));
        }

        // 生成会话 ID
        let session_id = uuid::Uuid::new_v4().to_string();

        // 创建会话
        let session = Session {
            session_id: session_id.clone(),
            agent_id,
            history_tokens: Vec::new(),
            last_active: std::time::SystemTime::now(),
        };

        self.sessions.write().insert(session_id.clone(), session);
        Ok(session_id)
    }

    /**
     * 删除会话
     */
    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write();
        if sessions.remove(session_id).is_none() {
            return Err(format!("会话 {} 不存在", session_id));
        }
        Ok(())
    }

    /**
     * 清理过期会话（超过 1 小时未活跃）
     */
    pub fn cleanup_expired_sessions(&self) -> usize {
        let now = std::time::SystemTime::now();
        let mut sessions = self.sessions.write();

        let before_count = sessions.len();
        sessions.retain(|_, s| {
            now.duration_since(s.last_active)
                .map(|d| d.as_secs() < 3600)
                .unwrap_or(false)
        });

        before_count - sessions.len()
    }
}

// 线程安全标记
unsafe impl Send for AgentSystem {}
unsafe impl Sync for AgentSystem {}

// ==================== 预定义 Agent 模板 ====================

/**
 * 预定义的 Agent 配置模板
 */
pub struct AgentTemplates;

impl AgentTemplates {
    /// 通用助手
    pub fn general_assistant(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "assistant".to_string(),
            model_id,
            system_prompt: "你是一个有帮助的AI助手。请礼貌、准确、详细地回答用户的问题。".to_string(),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            description: "通用对话助手，适合日常问答".to_string(),
        }
    }

    /// 代码助手
    pub fn code_assistant(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "coder".to_string(),
            model_id,
            system_prompt: "你是一个专业的编程助手。请提供清晰、高效、符合最佳实践的代码。包含必要的注释和解释。".to_string(),
            temperature: 0.2,
            top_p: 0.95,
            top_k: 50,
            repeat_penalty: 1.0,
            description: "代码生成和调试助手".to_string(),
        }
    }

    /// 创意写作助手
    pub fn creative_writer(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "writer".to_string(),
            model_id,
            system_prompt: "你是一个富有创造力的作家助手。擅长编写短剧剧本、小说故事。请发挥想象力，创作引人入胜的内容。".to_string(),
            temperature: 0.9,
            top_p: 0.95,
            top_k: 100,
            repeat_penalty: 1.15,
            description: "创意写作助手，适合剧本和故事创作".to_string(),
        }
    }

    /// 分镜师
    pub fn storyboard_artist(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "storyboard".to_string(),
            model_id,
            system_prompt: "你是一个专业的分镜师。根据剧本内容，生成详细的分镜描述，包括场景、角色动作、镜头角度、情绪氛围等。".to_string(),
            temperature: 0.6,
            top_p: 0.9,
            top_k: 60,
            repeat_penalty: 1.1,
            description: "分镜脚本生成助手".to_string(),
        }
    }
}
