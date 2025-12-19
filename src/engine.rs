/**
 * AI 推理服务模块（使用 llama-cpp-2）
 *
 * 本模块提供完整的本地 AI 模型推理功能，支持：
 * - 加载和管理 GGUF 格式的大语言模型
 * - 线程安全的模型实例管理
 * - 灵活的文本生成配置
 * - 高级采样策略（温度、top-k、top-p等）
 * - 资源自动管理和错误处理
 *
 * 技术栈：
 * - llama-cpp-2: 提供 llama.cpp 的 Rust 绑定
 * - parking_lot: 高性能的读写锁实现
 * - serde: 序列化/反序列化支持
 *
 * 使用示例：
 * ```rust
 * let service = AIService::new();
 *
 * // 配置模型
 * service.update_config(AIConfig {
 *     model_path: "/path/to/model.gguf".to_string(),
 *     context_size: 4096,
 *     gpu_layers: 32,  // GPU 加速
 *     ..Default::default()
 * });
 *
 * // 加载模型
 * service.load_model()?;
 *
 * // 生成文本
 * let response = service.generate(GenerateRequest {
 *     prompt: "你好，请介绍一下你自己".to_string(),
 *     max_tokens: Some(512),
 *     ..Default::default()
 * })?;
 * ```
 */

use llama_cpp_2::{
    llama_backend::LlamaBackend,        // Llama 后端管理，负责初始化和设备管理
    llama_batch::LlamaBatch,            // 批处理 token，提升推理效率
    model::{LlamaModel, AddBos, Special}, // 模型加载和 token 转换
    model::params::LlamaModelParams,     // 模型加载参数（GPU层数等）
    context::params::LlamaContextParams, // 上下文参数（线程数、上下文大小等）
    sampling::LlamaSampler,              // 采样器，控制生成策略
};
use parking_lot::RwLock;  // 高性能读写锁，支持多读单写
use serde::{Deserialize, Serialize};  // 用于 JSON 序列化
use std::path::PathBuf;   // 文件路径处理
use std::sync::Arc;       // 原子引用计数，用于线程安全的共享
use std::num::NonZeroU32; // 非零 u32 类型，用于上下文大小
use tokio::sync::watch;   // 用于取消信号传递

/**
 * AI 配置参数
 *
 * 这些参数控制模型的加载和推理行为。
 * 可以通过前端 Tauri 命令进行动态更新。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIConfig {
    /// 模型文件路径
    /// 支持 GGUF 格式的模型文件
    /// 示例："/home/user/models/llama-2-7b.Q4_K_M.gguf"
    pub model_path: String,

    /// 上下文窗口大小（token 数量）
    /// 决定了模型能"记住"多少上下文
    /// - 较大的值（如 4096、8192）能处理更长的对话
    /// - 但会消耗更多内存和计算资源
    /// 常见值：2048, 4096, 8192
    pub context_size: u32,

    /// GPU 层数（0 = 仅使用 CPU）
    /// 决定有多少模型层在 GPU 上运行
    /// - 0: 完全使用 CPU（慢但兼容性好）
    /// - 32-40: 中等 GPU（如 GTX 1060 6GB）
    /// - 全部层数: 完全 GPU 加速（需要足够显存）
    pub gpu_layers: u32,

    /// CPU 线程数
    /// 当使用 CPU 推理时，决定并行计算的线程数
    /// 建议值：物理 CPU 核心数
    pub threads: u32,

    /// 温度参数（控制生成的随机性）
    /// - 0.0: 完全确定性，总是选择概率最高的 token（适合事实性任务）
    /// - 0.7: 平衡的创造性（推荐默认值）
    /// - 1.0: 标准随机性
    /// - 1.5+: 高创造性，但可能产生不连贯的输出
    pub temperature: f32,

    /// Top-P 采样（核采样）
    /// 从累积概率达到 p 的最小 token 集合中采样
    /// - 0.9: 仅考虑累积概率前 90% 的 token（推荐）
    /// - 1.0: 考虑所有 token
    /// 用于过滤低概率的不合理词汇
    pub top_p: f32,

    /// Top-K 采样
    /// 仅从概率最高的 k 个 token 中采样
    /// - 40: 考虑前 40 个最可能的 token（推荐）
    /// - 0: 不限制
    /// 用于避免选择极低概率的词汇
    pub top_k: u32,

    /// 重复惩罚系数
    /// 降低已生成 token 再次出现的概率
    /// - 1.0: 不惩罚
    /// - 1.1: 轻微惩罚（推荐）
    /// - 1.5+: 强力惩罚（可能导致不自然的输出）
    pub repeat_penalty: f32,
}

impl Default for AIConfig {
    /**
     * 提供合理的默认配置
     *
     * 这些默认值适用于大多数场景：
     * - 4096 token 上下文（足够处理中等长度对话）
     * - 仅 CPU 推理（无需 GPU）
     * - 4 个线程（适合大多数现代 CPU）
     * - 平衡的采样参数（既有创造性又不失连贯性）
     */
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_size: 4096,
            gpu_layers: 0,      // 默认 CPU only
            threads: 4,
            temperature: 0.7,   // 平衡的创造性
            top_p: 0.9,         // 核采样
            top_k: 40,          // 限制候选 token
            repeat_penalty: 1.1, // 轻微惩罚重复
        }
    }
}

/**
 * AI 生成请求
 *
 * 封装单次文本生成所需的所有参数。
 * 这个结构体会从前端通过 Tauri 命令传递过来。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// 用户输入的提示词
    /// 这是模型生成文本的起点
    pub prompt: String,

    /// 系统提示词（可选）
    /// 用于设定模型的角色和行为
    /// 示例："你是一个专业的编剧助手，擅长创作短剧剧本。"
    pub system_prompt: Option<String>,

    /// 最大生成 token 数量
    /// 限制生成文本的长度
    /// - None: 使用默认值 512
    /// - Some(256): 生成较短的回复
    /// - Some(2048): 生成较长的内容
    pub max_tokens: Option<u32>,

    /// 临时覆盖全局温度设置
    /// 允许在不修改配置的情况下，为单次请求调整随机性
    pub temperature: Option<f32>,

    /// 停止词列表
    /// 当生成的文本包含任何停止词时，立即停止生成
    /// 示例：vec!["</s>", "[DONE]", "用户:"]
    pub stop_words: Option<Vec<String>>,
}

/**
 * AI 生成响应
 *
 * 包含生成结果和相关元数据。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// 生成的文本内容
    /// 已去除首尾空白字符
    pub content: String,

    /// 实际生成的 token 数量
    /// 可用于统计和性能监控
    pub tokens_generated: u32,

    /// 是否提前停止
    /// - true: 遇到停止词或模型自然结束
    /// - false: 达到 max_tokens 限制
    pub stopped_early: bool,
}

/**
 * 流式生成事件
 *
 * 通过 Tauri Event 发送给前端的流式数据。
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StreamEvent {
    /// Token 事件：每生成一个 token 发送一次
    Token {
        /// 生成的文本片段
        content: String,
        /// 当前已生成的 token 总数
        token_count: u32,
    },
    /// 完成事件：生成结束时发送
    Done {
        /// 完整的生成内容
        content: String,
        /// 总共生成的 token 数
        tokens_generated: u32,
        /// 是否提前停止
        stopped_early: bool,
    },
    /// 错误事件：发生错误时发送
    Error {
        /// 错误信息
        message: String,
    },
}

/**
 * 模型状态（内部使用）
 *
 * 封装已加载的模型实例和后端。
 * 使用 Option 包装以支持"未加载"状态。
 */
struct ModelState {
    /// Llama 后端实例
    /// 负责管理底层的 llama.cpp 运行时
    backend: LlamaBackend,

    /// 模型实例
    /// 使用 Arc 允许多个上下文共享同一个模型
    model: Arc<LlamaModel>,
}

/**
 * AI 服务主结构（线程安全）
 *
 * 这是对外暴露的主要接口，支持：
 * - 多线程并发访问（通过 RwLock）
 * - 配置热更新
 * - 模型动态加载/卸载
 * - 文本生成
 *
 * 线程安全保证：
 * - config: RwLock 允许多读单写
 * - state: RwLock 保护模型实例，确保加载/卸载操作的原子性
 */
pub struct AIService {
    /// 当前配置
    /// 使用 RwLock 允许多个线程同时读取配置
    config: RwLock<AIConfig>,

    /// 模型状态
    /// None 表示模型未加载，Some 表示已加载
    state: RwLock<Option<ModelState>>,
}

impl AIService {
    /**
     * 创建新的 AI 服务实例
     *
     * 初始状态：
     * - 使用默认配置
     * - 模型未加载
     *
     * 使用示例：
     * ```rust
     * let service = Arc::new(AIService::new());
     * ```
     */
    pub fn new() -> Self {
        Self {
            config: RwLock::new(AIConfig::default()),
            state: RwLock::new(None),
        }
    }

    /**
     * 更新配置
     *
     * 注意：
     * - 配置更新不会自动重新加载模型
     * - 如果需要使新配置生效，需要手动调用 unload_model() 然后 load_model()
     *
     * 参数：
     * - config: 新的配置对象
     */
    pub fn update_config(&self, config: AIConfig) {
        let mut current_config = self.config.write();
        *current_config = config;
    }

    /**
     * 获取当前配置的克隆
     *
     * 返回配置的深拷贝，不会持有锁。
     */
    pub fn get_config(&self) -> AIConfig {
        self.config.read().clone()
    }

    /**
     * 加载模型到内存
     *
     * 执行流程：
     * 1. 验证模型文件是否存在
     * 2. 初始化 Llama 后端
     * 3. 配置模型参数（GPU 层数等）
     * 4. 加载模型文件
     * 5. 更新内部状态
     *
     * 注意事项：
     * - 模型加载可能需要几秒到几分钟（取决于模型大小）
     * - 会占用大量内存（7B 模型约 4-8GB）
     * - 如果模型已加载，会被新模型替换
     *
     * 返回：
     * - Ok(()): 加载成功
     * - Err(String): 包含错误描述
     */
    pub fn load_model(&self) -> Result<(), String> {
        // 获取当前配置的克隆，避免长时间持有锁
        let config = self.config.read().clone();

        // 验证模型文件路径
        let model_path = PathBuf::from(&config.model_path);
        if !model_path.exists() {
            return Err(format!("模型文件不存在: {}", config.model_path));
        }

        // 初始化 Llama 后端
        // 这会设置全局的 llama.cpp 运行时环境
        let backend = LlamaBackend::init()
            .map_err(|e| format!("初始化 Llama 后端失败: {}", e))?;

        // 配置模型加载参数
        let mut model_params = LlamaModelParams::default();
        if config.gpu_layers > 0 {
            // 设置 GPU 加速层数
            // 0 = 完全 CPU，正数 = 在 GPU 上运行指定数量的层
            model_params = model_params.with_n_gpu_layers(config.gpu_layers);
        }

        // 从文件加载模型
        // 这是最耗时的操作，可能需要几秒到几分钟
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| format!("加载模型失败: {}", e))?;

        // 更新内部状态
        // 使用 write() 获取独占锁，确保状态更新的原子性
        let mut state = self.state.write();
        *state = Some(ModelState {
            backend,
            model: Arc::new(model),
        });

        Ok(())
    }

    /**
     * 卸载模型，释放内存
     *
     * 将状态设置为 None，Rust 的 RAII 机制会自动：
     * - 释放模型占用的内存
     * - 清理 GPU 资源（如果使用了 GPU）
     * - 销毁后端实例
     */
    pub fn unload_model(&self) {
        let mut state = self.state.write();
        *state = None;
    }

    /**
     * 检查模型是否已加载
     *
     * 返回：
     * - true: 模型已加载，可以进行推理
     * - false: 模型未加载，需要先调用 load_model()
     */
    pub fn is_model_loaded(&self) -> bool {
        self.state.read().is_some()
    }

    /**
     * 生成文本
     *
     * 这是核心的推理功能，执行流程：
     * 1. 验证模型已加载
     * 2. 构建完整提示词（系统提示 + 用户提示）
     * 3. 创建推理上下文
     * 4. 分词（文本 → token 序列）
     * 5. 处理提示词
     * 6. 循环生成新 token
     * 7. 解码并拼接结果
     *
     * 采样策略：
     * - 温度 = 0: 使用贪婪采样（完全确定性）
     * - 温度 > 0: 使用采样器链（温度 → top_k → top_p → 分布采样）
     *
     * 停止条件：
     * 1. 达到 max_tokens 限制
     * 2. 模型生成结束标记（EOS token）
     * 3. 生成内容包含停止词
     *
     * 参数：
     * - request: 生成请求，包含提示词和参数
     *
     * 返回：
     * - Ok(GenerateResponse): 生成成功
     * - Err(String): 包含错误描述
     */
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, String> {
        // ========== 第一步：准备工作 ==========

        // 获取模型状态的读锁，并验证模型已加载
        let state_guard = self.state.read();
        let state = state_guard.as_ref()
            .ok_or("模型未加载，请先加载模型")?;

        // 获取配置的读锁
        let config = self.config.read();

        // 构建完整的提示词
        // 如果提供了系统提示词，则将其放在用户提示词之前
        let full_prompt = if let Some(system) = &request.system_prompt {
            format!("{}\n\n{}", system, request.prompt)
        } else {
            request.prompt.clone()
        };

        // ========== 第二步：配置推理上下文 ==========

        // 上下文大小必须是非零值
        let ctx_size = NonZeroU32::new(config.context_size)
            .ok_or("上下文大小必须大于0")?;

        // 创建上下文参数
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));  // 设置上下文窗口大小

        // 配置线程数（用于 CPU 推理）
        if config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(config.threads as i32)        // 推理线程数
                .with_n_threads_batch(config.threads as i32); // 批处理线程数
        }

        // 创建推理上下文
        // 上下文是独立的推理会话，包含 KV 缓存等状态
        let mut ctx = state.model
            .new_context(&state.backend, ctx_params)
            .map_err(|e| format!("创建上下文失败: {}", e))?;

        // ========== 第三步：分词 ==========

        // 将文本转换为 token 序列
        // AddBos::Always 表示总是在开头添加 BOS (Beginning of Sequence) token
        let tokens_list = state.model
            .str_to_token(&full_prompt, AddBos::Always)
            .map_err(|e| format!("分词失败: {}", e))?;

        // 验证分词结果不为空
        if tokens_list.is_empty() {
            return Err("分词结果为空".to_string());
        }

        // ========== 第四步：创建批处理并处理提示词 ==========

        // 创建批处理对象
        // LlamaBatch 允许一次性处理多个 token，提升效率
        let n_ctx = config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);  // (最大序列长度, 序列数量)

        // 将所有提示词 token 添加到批处理
        let last_index = tokens_list.len() - 1;
        for (i, token) in tokens_list.iter().enumerate() {
            let is_last = i == last_index;
            // add(token, 位置, 序列ID, 是否需要输出)
            // 只有最后一个 token 需要输出 logits（用于生成下一个 token）
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("添加 token 失败: {}", e))?;
        }

        // 清除 KV 缓存，确保从干净状态开始
        ctx.clear_kv_cache();

        // 解码提示词
        // 这会计算整个提示词序列的表示，并缓存到 KV cache 中
        ctx.decode(&mut batch)
            .map_err(|e| format!("解码提示词失败: {}", e))?;

        // ========== 第五步：配置采样器 ==========

        // 获取生成文本的最大长度
        let max_tokens = request.max_tokens.unwrap_or(512);

        // 初始化结果变量
        let mut result = String::new();
        let mut token_count = 0u32;
        let n_prompt = tokens_list.len();

        // 根据温度参数选择采样策略
        let temperature = request.temperature.unwrap_or(config.temperature);
        let mut sampler = if temperature <= 0.0 {
            // 温度为 0 或负数：使用贪婪采样
            // 总是选择概率最高的 token，输出完全确定
            LlamaSampler::greedy()
        } else {
            // 温度 > 0：使用采样器链
            // 采样器按顺序应用，每个采样器过滤或调整 token 概率
            LlamaSampler::chain_simple([
                // 1. 温度缩放：调整 logits 的分布
                //    温度越高，分布越平滑（更随机）
                LlamaSampler::temp(temperature),

                // 2. Top-K 过滤：只保留概率最高的 k 个 token
                //    减少低质量候选
                LlamaSampler::top_k(config.top_k as i32),

                // 3. Top-P (核采样)：保留累积概率达到 p 的最小集合
                //    动态调整候选集大小
                LlamaSampler::top_p(config.top_p, 1),

                // 4. 分布采样：从过滤后的分布中随机采样
                //    使用时间戳作为随机种子，确保每次生成结果不同
                //    如需可重现性，可在请求中传入固定 seed
                LlamaSampler::dist({
                    use std::time::{SystemTime, UNIX_EPOCH};
                    (SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs() & 0xFFFFFFFF) as u32  // 截取低32位转为u32
                }),
            ])
        };

        // ========== 第六步：主生成循环 ==========

        for i in 0..max_tokens {
            // 从上下文中采样新 token
            // -1 表示从最后一个位置采样
            let new_token_id = sampler.sample(&ctx, -1);

            // 告知采样器已接受该 token
            // 这对于某些有状态的采样器（如重复惩罚）很重要
            sampler.accept(new_token_id);

            // 检查是否为结束 token (End of Generation)
            // 模型自然结束生成
            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            // 将 token 转换为文本
            // Special::Tokenize 表示使用标准的 token 解码方式
            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("token 转换失败: {}", e))?;

            // 将字节转换为字符串（使用有损转换处理非 UTF-8 字节）
            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            // 检查是否触发停止词
            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            // 准备下一次迭代的批处理
            batch.clear();  // 清空批处理

            // 添加新生成的 token
            // 位置 = 提示词长度 + 当前生成的 token 数量
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("添加生成的 token 失败: {}", e))?;

            // 解码新 token，更新 KV cache
            ctx.decode(&mut batch)
                .map_err(|e| format!("解码生成的 token 失败: {}", e))?;
        }

        // ========== 第七步：返回结果 ==========

        Ok(GenerateResponse {
            content: result.trim().to_string(),  // 去除首尾空白
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,  // 是否在达到限制前停止
        })
    }

    /**
     * 流式生成文本
     *
     * 与 generate() 类似，但通过 Tauri Event 实时发送生成的 token。
     * 支持通过 watch channel 取消生成。
     *
     * 事件流：
     * 1. Token 事件：每生成一个 token 发送一次
     * 2. Done 事件：生成完成时发送
     * 3. Error 事件：发生错误时发送
     *
     * 取消机制：
     * - 通过 cancel_rx 监听取消信号
     * - 在每次生成 token 后检查是否需要取消
     *
     * 参数：
     * - request: 生成请求
     * - callback: 事件回调函数，接收 StreamEvent
     * - cancel_rx: 取消信号接收器
     *
     * 返回：
     * - Ok(()): 生成成功完成或被取消
     * - Err(String): 发生错误
     */
    pub async fn generate_stream<F>(
        &self,
        request: GenerateRequest,
        mut callback: F,
        mut cancel_rx: watch::Receiver<bool>,
    ) -> Result<(), String>
    where
        F: FnMut(StreamEvent) + Send,
    {
        // ========== 第一步：准备工作 ==========

        let state_guard = self.state.read();
        let state = state_guard.as_ref()
            .ok_or("模型未加载，请先加载模型")?;

        let config = self.config.read();

        let full_prompt = if let Some(system) = &request.system_prompt {
            format!("{}\n\n{}", system, request.prompt)
        } else {
            request.prompt.clone()
        };

        // ========== 第二步：配置推理上下文 ==========

        let ctx_size = NonZeroU32::new(config.context_size)
            .ok_or("上下文大小必须大于0")?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));

        if config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(config.threads as i32)
                .with_n_threads_batch(config.threads as i32);
        }

        let mut ctx = state.model
            .new_context(&state.backend, ctx_params)
            .map_err(|e| format!("创建上下文失败: {}", e))?;

        // ========== 第三步：分词 ==========

        let tokens_list = state.model
            .str_to_token(&full_prompt, AddBos::Always)
            .map_err(|e| format!("分词失败: {}", e))?;

        if tokens_list.is_empty() {
            return Err("分词结果为空".to_string());
        }

        // ========== 第四步：处理提示词 ==========

        let n_ctx = config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        let last_index = tokens_list.len() - 1;
        for (i, token) in tokens_list.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("添加 token 失败: {}", e))?;
        }

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("解码提示词失败: {}", e))?;

        // ========== 第五步：配置采样器 ==========

        let max_tokens = request.max_tokens.unwrap_or(512);
        let mut result = String::new();
        let mut token_count = 0u32;
        let n_prompt = tokens_list.len();

        let temperature = request.temperature.unwrap_or(config.temperature);
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(config.top_k as i32),
                LlamaSampler::top_p(config.top_p, 1),
                LlamaSampler::dist(1234),
            ])
        };

        // ========== 第六步：主生成循环（带流式输出） ==========

        for i in 0..max_tokens {
            // 检查取消信号
            if *cancel_rx.borrow_and_update() {
                // 发送取消事件
                callback(StreamEvent::Error {
                    message: "生成已取消".to_string(),
                });
                return Err("生成已取消".to_string());
            }

            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("token 转换失败: {}", e))?;

            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            // 发送 Token 事件
            callback(StreamEvent::Token {
                content: token_str.to_string(),
                token_count,
            });

            // 检查停止词
            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            // 准备下一次迭代
            batch.clear();
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("添加生成的 token 失败: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("解码生成的 token 失败: {}", e))?;
        }

        // ========== 第七步：发送完成事件 ==========

        callback(StreamEvent::Done {
            content: result.trim().to_string(),
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,
        });

        Ok(())
    }
}

// ========== 线程安全标记 ==========

// 显式标记 AIService 可以安全地在线程间传递
// 这是必需的，因为 llama-cpp-2 的类型没有自动实现 Send/Sync
unsafe impl Send for AIService {}
unsafe impl Sync for AIService {}

// ========== 单元测试 ==========

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试默认配置是否符合预期
    #[test]
    fn test_default_config() {
        let config = AIConfig::default();
        assert_eq!(config.context_size, 4096);
        assert_eq!(config.temperature, 0.7);
    }

    /// 测试服务创建和初始状态
    #[test]
    fn test_service_creation() {
        let service = AIService::new();
        // 新创建的服务应该没有加载模型
        assert!(!service.is_model_loaded());
    }
}
