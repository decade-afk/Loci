/**
 * Phase 1: 推理引擎核心
 *
 * 直接FFI调用llama.cpp，实现：
 * 1. Model加载 (使用GGUF零拷贝)
 * 2. Context管理 (KV Cache)
 * 3. 推理循环: Tokenize -> Forward -> Sample -> Detokenize
 * 4. 流式生成支持
 * 5. 性能监控
 *
 * 目标：Eval Speed > 10 tokens/s (CPU), > 20 tokens/s (GPU)
 */

use std::sync::{Arc, RwLock};
use anyhow::{Result, Context, bail};

use crate::backend::{ComputeBackend, detect_backend};
use crate::gguf::GGUFModel;

// ==================== FFI声明（待bindgen生成） ====================
// 这些将由build.rs生成，这里先手动声明核心接口

#[repr(C)]
struct LlamaModel {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaContext {
    _private: [u8; 0],
}

#[repr(C)]
struct LlamaSampler {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LlamaModelParams {
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub main_gpu: i32,
    pub vocab_only: u8,    // C的bool是uint8_t
    pub use_mmap: u8,
    pub use_mlock: u8,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
}

// FFI函数声明（将由bindgen自动生成）
#[link(name = "llama", kind = "static")]
extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();

    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_load_model_from_file(path: *const i8, params: LlamaModelParams) -> *mut LlamaModel;
    fn llama_free_model(model: *mut LlamaModel);

    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_new_context_with_model(model: *mut LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);

    fn llama_tokenize(
        ctx: *mut LlamaContext,
        text: *const i8,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    fn llama_decode(
        ctx: *mut LlamaContext,
        batch: *const LlamaBatch,
    ) -> i32;

    #[allow(dead_code)]
    fn llama_get_logits(ctx: *mut LlamaContext) -> *const f32;

    fn llama_token_to_piece(
        ctx: *mut LlamaContext,
        token: i32,
        buf: *mut i8,
        length: i32,
        special: bool,
    ) -> i32;

    fn llama_sampler_chain_default_params() -> LlamaSamplerParams;
    fn llama_sampler_chain_init(params: LlamaSamplerParams) -> *mut LlamaSampler;
    fn llama_sampler_free(sampler: *mut LlamaSampler);
    fn llama_sampler_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext, idx: i32) -> i32;

    // 修复P1-3：添加EOS token查询
    fn llama_token_eos(model: *const LlamaModel) -> i32;
    #[allow(dead_code)]
    fn llama_token_bos(model: *const LlamaModel) -> i32;
}

#[repr(C)]
struct LlamaBatch {
    n_tokens: i32,
    tokens: *mut i32,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

#[repr(C)]
struct LlamaSamplerParams {
    top_k: i32,
    top_p: f32,
    min_p: f32,
    temp: f32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
}

// ==================== Rust安全封装 ====================

/// Phase 1推理引擎
pub struct LociEngine {
    /// 内部状态（线程安全）
    inner: Arc<RwLock<EngineInner>>,

    /// 计算后端
    backend: Arc<dyn ComputeBackend>,
}

struct EngineInner {
    /// llama.cpp模型指针
    model: Option<*mut LlamaModel>,

    /// llama.cpp上下文指针
    context: Option<*mut LlamaContext>,

    /// 采样器
    sampler: Option<*mut LlamaSampler>,

    /// GGUF模型（用于元数据访问）
    #[allow(dead_code)]
    gguf: Option<Arc<GGUFModel>>,

    /// 配置参数
    #[allow(dead_code)]
    config: EngineConfig,

    /// 性能统计
    stats: PerformanceStats,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: String,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub n_batch: u32,
    pub n_threads: u32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: 2048,
            n_gpu_layers: 0,
            n_batch: 512,
            n_threads: num_cpus::get() as u32,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub load_time_ms: u64,
    pub prompt_eval_count: u64,
    pub prompt_eval_time_ms: u64,
    pub eval_count: u64,
    pub eval_time_ms: u64,
}

impl PerformanceStats {
    pub fn prompt_tokens_per_second(&self) -> f64 {
        if self.prompt_eval_time_ms == 0 {
            return 0.0;
        }
        (self.prompt_eval_count as f64 / self.prompt_eval_time_ms as f64) * 1000.0
    }

    pub fn eval_tokens_per_second(&self) -> f64 {
        if self.eval_time_ms == 0 {
            return 0.0;
        }
        (self.eval_count as f64 / self.eval_time_ms as f64) * 1000.0
    }
}

// ==================== Engine实现 ====================

impl LociEngine {
    /// Phase 1核心方法：加载模型
    ///
    /// 目标：Load Time < 500ms
    pub fn new(config: EngineConfig) -> Result<Self> {
        let load_start = std::time::Instant::now();

        eprintln!("🚀 Initializing Loci Engine (Phase 1)");
        eprintln!("   Model: {}", config.model_path);

        // 步骤1：自动探测Backend
        let backend = detect_backend();
        eprintln!("   Backend: {}", backend.name());

        // 步骤2：根据Backend推荐GPU层数
        let recommended_layers = backend.recommended_gpu_layers();
        let n_gpu_layers = if config.n_gpu_layers < 0 {
            recommended_layers as i32
        } else {
            config.n_gpu_layers
        };

        eprintln!("   GPU Layers: {}", n_gpu_layers);

        // 步骤3：初始化llama.cpp backend
        unsafe {
            llama_backend_init();
        }

        // 步骤4：加载GGUF元数据（零拷贝）
        let gguf = GGUFModel::load(&config.model_path)
            .context("Failed to load GGUF model")?;
        eprintln!("   ✅ GGUF loaded: {:.2} GB", gguf.total_size_gb());

        // 步骤5：加载llama.cpp模型
        let mut model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = n_gpu_layers;
        model_params.use_mmap = 1;  // Phase 1要求：零拷贝 (true)
        model_params.use_mlock = 0;  // false

        let model_path_c = std::ffi::CString::new(config.model_path.as_str())
            .context("Invalid model path")?;

        let model = unsafe {
            llama_load_model_from_file(model_path_c.as_ptr(), model_params)
        };

        if model.is_null() {
            bail!("Failed to load llama.cpp model");
        }

        eprintln!("   ✅ Model loaded");

        // 步骤6：创建Context
        let mut ctx_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = config.n_ctx;
        ctx_params.n_batch = config.n_batch;
        ctx_params.n_threads = config.n_threads;
        ctx_params.n_threads_batch = config.n_threads;

        let context = unsafe {
            llama_new_context_with_model(model, ctx_params)
        };

        if context.is_null() {
            unsafe { llama_free_model(model); }
            bail!("Failed to create llama.cpp context");
        }

        eprintln!("   ✅ Context created (n_ctx={})", config.n_ctx);

        // 步骤7：初始化采样器
        let mut sampler_params = unsafe { llama_sampler_chain_default_params() };
        sampler_params.temp = config.temperature;
        sampler_params.top_k = config.top_k;
        sampler_params.top_p = config.top_p;
        sampler_params.penalty_repeat = config.repeat_penalty;

        let sampler = unsafe {
            llama_sampler_chain_init(sampler_params)
        };

        if sampler.is_null() {
            unsafe {
                llama_free(context);
                llama_free_model(model);
            }
            bail!("Failed to create sampler");
        }

        let load_time = load_start.elapsed();
        eprintln!("   ✅ Engine initialized in {:.2}ms {}",
                 load_time.as_millis(),
                 if load_time.as_millis() < 500 { "🎯" } else { "⚠️" });

        let inner = EngineInner {
            model: Some(model),
            context: Some(context),
            sampler: Some(sampler),
            gguf: Some(Arc::new(gguf)),
            config,
            stats: PerformanceStats {
                load_time_ms: load_time.as_millis() as u64,
                ..Default::default()
            },
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            backend,
        })
    }

    /// Phase 1核心方法：文本生成（阻塞式，返回完整结果）
    ///
    /// 实现完整的推理循环：Tokenize -> Forward -> Sample -> Detokenize
    ///
    /// **注意**: 此方法会阻塞直到生成完成。如需流式输出，请使用 `generate_stream`。
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut inner = self.inner.write().unwrap();

        let context = inner.context.ok_or_else(|| anyhow::anyhow!("Context not initialized"))?;
        let model = inner.model.ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
        let sampler = inner.sampler.ok_or_else(|| anyhow::anyhow!("Sampler not initialized"))?;

        eprintln!("🔮 Generating (max_tokens={})...", max_tokens);

        // 修复P1-3：动态查询EOS token
        let eos_token = unsafe { llama_token_eos(model) };
        eprintln!("   EOS token: {}", eos_token);

        // ========== 步骤1：Tokenize ==========
        let prompt_c = std::ffi::CString::new(prompt)?;
        let mut tokens = vec![0i32; 4096];

        let n_tokens = unsafe {
            llama_tokenize(
                context,
                prompt_c.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,  // add_special (BOS)
                true,  // parse_special
            )
        };

        if n_tokens < 0 {
            bail!("Tokenization failed");
        }

        tokens.truncate(n_tokens as usize);
        eprintln!("   Prompt tokens: {}", n_tokens);

        // ========== 步骤2：Prompt Evaluation ==========
        let prompt_eval_start = std::time::Instant::now();

        // 构建batch并执行prompt evaluation
        let mut pos_data: Vec<i32> = (0..tokens.len() as i32).collect();
        let mut seq_id_data = vec![1i32; tokens.len()];
        let mut logits_data = vec![0i8; tokens.len()];
        logits_data[tokens.len() - 1] = 1;  // 只需要最后一个token的logits

        let batch = LlamaBatch {
            n_tokens: tokens.len() as i32,
            tokens: tokens.as_mut_ptr(),
            embd: std::ptr::null_mut(),
            pos: pos_data.as_mut_ptr(),
            n_seq_id: seq_id_data.as_mut_ptr(),
            seq_id: std::ptr::null_mut(),
            logits: logits_data.as_mut_ptr(),
        };

        let ret = unsafe { llama_decode(context, &batch) };
        if ret != 0 {
            bail!("Prompt evaluation failed: llama_decode returned {}", ret);
        }

        let prompt_eval_time = prompt_eval_start.elapsed();
        inner.stats.prompt_eval_count = n_tokens as u64;
        inner.stats.prompt_eval_time_ms = prompt_eval_time.as_millis() as u64;

        eprintln!("   ✅ Prompt evaluated: {:.2} tokens/s",
                 inner.stats.prompt_tokens_per_second());

        // ========== 步骤3：生成循环 ==========
        let mut generated_text = String::new();
        let eval_start = std::time::Instant::now();
        let mut current_pos = tokens.len() as i32;

        // 预分配buffer，避免每次循环创建新Vec（修复P0-2内存安全问题）
        let mut next_tokens_buf = vec![0i32];
        let mut next_pos_buf = vec![0i32];
        let mut next_seq_id_buf = vec![1i32];
        let mut next_logits_buf = vec![1i8];

        for i in 0..max_tokens {
            // 3.1 采样下一个token
            let next_token = unsafe {
                llama_sampler_sample(sampler, context, -1)
            };

            // 修复P1-3：使用动态查询的EOS token
            if next_token == eos_token {
                eprintln!("   Reached EOS (token={})", eos_token);
                break;
            }

            // 3.2 Detokenize
            let mut piece_buf = vec![0i8; 256];
            let piece_len = unsafe {
                llama_token_to_piece(
                    context,
                    next_token,
                    piece_buf.as_mut_ptr(),
                    piece_buf.len() as i32,
                    false,
                )
            };

            if piece_len > 0 {
                let piece_bytes = unsafe {
                    std::slice::from_raw_parts(piece_buf.as_ptr() as *const u8, piece_len as usize)
                };
                // 修复P1-5：添加UTF-8验证
                match std::str::from_utf8(piece_bytes) {
                    Ok(piece) => {
                        generated_text.push_str(piece);
                        print!("{}", piece);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                    Err(_) => {
                        eprintln!("⚠️  Invalid UTF-8 in token {}", next_token);
                    }
                }
            }

            // 3.3 Forward pass (下一轮)
            // 复用预分配的buffer（修复P0-2）
            next_tokens_buf[0] = next_token;
            next_pos_buf[0] = current_pos;

            let next_batch = LlamaBatch {
                n_tokens: 1,
                tokens: next_tokens_buf.as_mut_ptr(),
                embd: std::ptr::null_mut(),
                pos: next_pos_buf.as_mut_ptr(),
                n_seq_id: next_seq_id_buf.as_mut_ptr(),
                seq_id: std::ptr::null_mut(),
                logits: next_logits_buf.as_mut_ptr(),
            };

            let ret = unsafe { llama_decode(context, &next_batch) };
            if ret != 0 {
                eprintln!("   ⚠️  Decode failed for token {}, stopping generation", i);
                break;
            }

            current_pos += 1;
            inner.stats.eval_count += 1;
        }

        println!();

        let eval_time = eval_start.elapsed();
        inner.stats.eval_time_ms = eval_time.as_millis() as u64;

        eprintln!("   ✅ Generated {} tokens: {:.2} tokens/s {}",
                 inner.stats.eval_count,
                 inner.stats.eval_tokens_per_second(),
                 if inner.stats.eval_tokens_per_second() > 10.0 { "🎯" } else { "⚠️" });

        Ok(generated_text)
    }

    /// Generate text with streaming token-level callbacks.
    ///
    /// Implements the complete inference loop, invoking the callback
    /// immediately for each generated token.
    ///
    /// # Parameters
    /// - `prompt`: Input prompt text
    /// - `max_tokens`: Maximum number of tokens to generate
    /// - `callback`: Streaming callback (implements `StreamCallback` trait)
    ///
    /// # Returns
    /// - `StreamStats`: Generation statistics
    ///
    /// # Example
    /// ```no_run
    /// use loci::{LociEngine, EngineConfig, ConsoleCallback};
    ///
    /// let engine = LociEngine::new(EngineConfig::default())?;
    /// let stats = engine.generate_stream(
    ///     "Hello",
    ///     50,
    ///     &mut ConsoleCallback::new(true)
    /// )?;
    /// println!("Generated {} tokens", stats.generated_tokens);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn generate_stream<C>(
        &self,
        prompt: &str,
        max_tokens: usize,
        callback: &mut C,
    ) -> Result<crate::streaming::StreamStats>
    where
        C: crate::streaming::StreamCallback,
    {
        use crate::streaming::{StreamStats, StreamControlFlow, safe_callback_invoke};

        let mut inner = self.inner.write().unwrap();

        let context = inner.context.ok_or_else(|| anyhow::anyhow!("Context not initialized"))?;
        let model = inner.model.ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
        let sampler = inner.sampler.ok_or_else(|| anyhow::anyhow!("Sampler not initialized"))?;

        eprintln!("🔮 Generating (stream, max_tokens={})...", max_tokens);

        let eos_token = unsafe { llama_token_eos(model) };

        let mut stats = StreamStats::new();
        let total_start = std::time::Instant::now();

        // Step 1: Tokenize prompt
        let prompt_c = std::ffi::CString::new(prompt)?;
        let mut tokens = vec![0i32; 4096];

        let n_tokens = unsafe {
            llama_tokenize(
                context,
                prompt_c.as_ptr(),
                prompt.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            )
        };

        if n_tokens < 0 {
            bail!("Tokenization failed");
        }

        tokens.truncate(n_tokens as usize);
        stats.prompt_tokens = n_tokens as usize;
        eprintln!("   Prompt tokens: {}", n_tokens);

        // Step 2: Evaluate prompt
        let prompt_eval_start = std::time::Instant::now();

        let mut pos_data: Vec<i32> = (0..tokens.len() as i32).collect();
        let mut seq_id_data = vec![1i32; tokens.len()];
        let mut logits_data = vec![0i8; tokens.len()];
        logits_data[tokens.len() - 1] = 1;

        let batch = LlamaBatch {
            n_tokens: tokens.len() as i32,
            tokens: tokens.as_mut_ptr(),
            embd: std::ptr::null_mut(),
            pos: pos_data.as_mut_ptr(),
            n_seq_id: seq_id_data.as_mut_ptr(),
            seq_id: std::ptr::null_mut(),
            logits: logits_data.as_mut_ptr(),
        };

        let ret = unsafe { llama_decode(context, &batch) };
        if ret != 0 {
            let error = anyhow::anyhow!("Prompt evaluation failed: llama_decode returned {}", ret);
            callback.on_error(&error);
            return Err(error);
        }

        let prompt_eval_time = prompt_eval_start.elapsed();
        stats.prompt_time_ms = prompt_eval_time.as_millis() as u64;

        eprintln!("   ✅ Prompt evaluated: {:.2} tokens/s",
                 (stats.prompt_tokens as f64 / stats.prompt_time_ms as f64) * 1000.0);

        // Step 3: Streaming generation loop
        let eval_start = std::time::Instant::now();
        let mut current_pos = tokens.len() as i32;

        // Pre-allocate buffers to avoid allocations in the hot loop
        let mut next_tokens_buf = vec![0i32];
        let mut next_pos_buf = vec![0i32];
        let mut next_seq_id_buf = vec![1i32];
        let mut next_logits_buf = vec![1i8];

        for i in 0..max_tokens {
            let next_token = unsafe {
                llama_sampler_sample(sampler, context, -1)
            };

            if next_token == eos_token {
                eprintln!("   Reached EOS (token={})", eos_token);
                break;
            }

            // Detokenize the sampled token
            let mut piece_buf = vec![0i8; 256];
            let piece_len = unsafe {
                llama_token_to_piece(
                    context,
                    next_token,
                    piece_buf.as_mut_ptr(),
                    piece_buf.len() as i32,
                    false,
                )
            };

            if piece_len > 0 {
                let piece_bytes = unsafe {
                    std::slice::from_raw_parts(piece_buf.as_ptr() as *const u8, piece_len as usize)
                };

                match std::str::from_utf8(piece_bytes) {
                    Ok(piece) => {
                        match safe_callback_invoke(callback, piece, next_token, i) {
                            Ok(StreamControlFlow::Continue) => {}
                            Ok(StreamControlFlow::Stop) => {
                                eprintln!("   ⏹️  User requested stop at token {}", i);
                                break;
                            }
                            Err(e) => {
                                eprintln!("   ⚠️  Callback error: {}", e);
                                callback.on_error(&e);
                                return Err(e);
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("⚠️  Invalid UTF-8 in token {}", next_token);
                    }
                }
            }

            // Forward pass for next iteration
            next_tokens_buf[0] = next_token;
            next_pos_buf[0] = current_pos;

            let next_batch = LlamaBatch {
                n_tokens: 1,
                tokens: next_tokens_buf.as_mut_ptr(),
                embd: std::ptr::null_mut(),
                pos: next_pos_buf.as_mut_ptr(),
                n_seq_id: next_seq_id_buf.as_mut_ptr(),
                seq_id: std::ptr::null_mut(),
                logits: next_logits_buf.as_mut_ptr(),
            };

            let ret = unsafe { llama_decode(context, &next_batch) };
            if ret != 0 {
                eprintln!("   ⚠️  Decode failed for token {}, stopping generation", i);
                break;
            }

            current_pos += 1;
            stats.generated_tokens += 1;
        }

        let eval_time = eval_start.elapsed();
        stats.generation_time_ms = eval_time.as_millis() as u64;
        stats.total_tokens = stats.prompt_tokens + stats.generated_tokens;
        stats.total_time_ms = total_start.elapsed().as_millis() as u64;
        stats.calculate_tps();

        callback.on_complete(&stats);

        eprintln!("   ✅ Generated {} tokens: {:.2} tokens/s {}",
                 stats.generated_tokens,
                 stats.tokens_per_second,
                 if stats.tokens_per_second > 10.0 { "🎯" } else { "⚠️" });

        inner.stats.eval_count = stats.generated_tokens as u64;
        inner.stats.eval_time_ms = stats.generation_time_ms;

        Ok(stats)
    }

    /// 获取性能统计
    pub fn stats(&self) -> PerformanceStats {
        self.inner.read().unwrap().stats.clone()
    }

    /// 获取Backend信息
    pub fn backend_name(&self) -> String {
        self.backend.name().to_string()
    }
}

// ==================== 资源清理 ====================

impl Drop for EngineInner {
    fn drop(&mut self) {
        eprintln!("🧹 Cleaning up Loci Engine...");

        // 使用panic::catch_unwind确保即使FFI调用panic也不会泄漏资源
        unsafe {
            if let Some(sampler) = self.sampler.take() {
                let _ = std::panic::catch_unwind(|| {
                    llama_sampler_free(sampler);
                });
            }

            if let Some(context) = self.context.take() {
                let _ = std::panic::catch_unwind(|| {
                    llama_free(context);
                });
            }

            if let Some(model) = self.model.take() {
                let _ = std::panic::catch_unwind(|| {
                    llama_free_model(model);
                });
            }

            let _ = std::panic::catch_unwind(|| {
                llama_backend_free();
            });
        }

        eprintln!("   ✅ Cleanup completed");
    }
}

// ==================== 线程安全性 ====================

unsafe impl Send for LociEngine {}
unsafe impl Sync for LociEngine {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.n_ctx, 2048);
        assert!(config.n_threads > 0);
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats {
            eval_count: 100,
            eval_time_ms: 5000,
            ..Default::default()
        };

        assert_eq!(stats.eval_tokens_per_second(), 20.0);
    }
}
