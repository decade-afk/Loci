/**
 * Loci - Phase 1+2 核心库
 *
 * 高性能本地AI推理引擎，特性：
 * 1. 零拷贝GGUF加载（memmap2）
 * 2. 多Backend支持（CUDA/Metal/ROCm/CPU）
 * 3. 直接FFI绑定llama.cpp
 * 4. 工业级性能目标（Load < 500ms, Eval > 10 t/s）
 * 5. [Phase 2] Paged Attention + 内存预算器（128k+ 上下文）
 *
 * Phase 1目标（已完成）：
 * - 可靠的基础架构
 * - 完整的Backend抽象
 * - 高效的模型加载
 * - 稳定的推理循环
 *
 * Phase 2 Week 1 目标（进行中）：
 * - Paged Attention Block Table
 * - 物理块分配器 + LRU 置换
 * - 内存预算器（System Floor + Swap Threshold）
 * - 支持 8x 128k 并发会话
 */

// ==================== 公开模块 ====================

/// Backend抽象层
///
/// 提供统一的ComputeBackend trait，支持：
/// - CUDA (NVIDIA GPU)
/// - Metal (Apple Silicon)
/// - ROCm (AMD GPU)
/// - CPU (AVX512/AVX2 fallback)
pub mod backend;

/// GGUF零拷贝加载器
///
/// 使用memmap2实现零拷贝模型加载：
/// - 虚拟内存映射
/// - 懒加载权重数据
/// - 32字节张量对齐
/// - 元数据快速解析
pub mod gguf;

/// 推理引擎核心
///
/// 实现完整的推理循环：
/// - Tokenize
/// - Forward Pass
/// - Sampling
/// - Detokenize
pub mod engine;

/// [Phase 1] 采样器（Sampler）
///
/// 核心特性：
/// - Temperature scaling（温度缩放）
/// - Top-K sampling（前 K 个最高概率）
/// - Top-P (Nucleus) sampling（核采样）
/// - Min-P sampling（最小概率阈值）
/// - Repetition penalty（重复惩罚）
/// - Deterministic sampling（确定性采样）
pub mod sampling;

/// [Phase 2 Week 1] Paged Attention 系统
///
/// 核心特性：
/// - Block Table 映射（逻辑块 → 物理块）
/// - Physical Block Allocator（LRU 驱动的块分配器）
/// - Memory Budgeter（智能显存管理）
/// - Swap 机制（VRAM ↔ RAM 自动置换）
/// - 支持 128k+ 上下文 + 多会话并发
pub mod paged_attention;

/// [Phase 2 Week 2] Constraint Sampling 系统
///
/// 核心特性：
/// - Constraint Trait（统一约束接口）
/// - RegexConstraint（正则表达式约束）
/// - JsonSchemaConstraint（JSON 模式约束）
/// - TokenMask（高效 logits 过滤）
/// - 约束组合器（AND/OR）
pub mod constraints;

/// [Phase 2 Week 3] 推理挂起/恢复机制
///
/// 核心特性：
/// - ControlFlow（挂起信号与控制流）
/// - SessionState（状态机）
/// - ResumeContext（恢复上下文）
/// - SuspendableSession（可挂起会话）
/// - Agent 工作流支持（ReAct 循环）
pub mod suspend;

/// [Phase 2 Week 4] Radix Tree 前缀缓存
///
/// 核心特性：
/// - RadixNode（Radix Tree 节点）
/// - RadixTree（前缀共享数据结构）
/// - KVCacheManager（KV Cache 管理器）
/// - 内存节省 50%+（通过前缀共享）
/// - 与 Paged Attention 集成
pub mod radix_tree;

/// [Phase 2 Week 5] 双轨制插件系统（Native + WASM）
///
/// 核心特性：
/// - Plugin（统一插件接口）
/// - NativePlugin（高性能 Native 插件）
/// - WasmPlugin（安全沙箱 WASM 插件）
/// - PluginRegistry（双轨插件注册表）
/// - SignatureVerifier（Ed25519 签名验证）
/// - Watchdog（超时监控与 Panic 隔离）
pub mod plugin_system;

/// [Phase 3 Week 1] 移动端 C FFI 接口
///
/// 核心特性：
/// - C FFI 导出函数（loci_init, loci_generate, loci_destroy）
/// - 流式生成回调支持（loci_generate_stream）
/// - Android JNI 接口（Java_com_loci_LociEngine_*）
/// - iOS Objective-C 兼容接口
/// - 统一的跨平台 API
#[cfg(any(target_os = "android", target_os = "ios", feature = "mobile-ffi"))]
pub mod mobile_ffi;

/// [Phase 3 Week 2-4] 多模型管理系统
///
/// 核心特性：
/// - ModelRegistry（全局模型注册表）
/// - 多模型热切换（运行时切换主模型）
/// - LoRA 动态合并（加载/卸载/stacking）
/// - KV Cache 管理（切换时的缓存策略）
/// - 内存预算控制（多模型共存）
pub mod model_registry;

/// [Phase 3 Week 4] LoRA 权重合并实现
///
/// 核心特性：
/// - LoRA GGUF 格式解析
/// - 权重合并算法（W' = W + scale * (A @ B)）
/// - 运行时动态加载/卸载
/// - 多 LoRA stacking 支持
/// - Tensor 操作（矩阵乘法、标量乘法、元素相加）
pub mod lora;

/// [Phase 3 Week 7] 插件市场系统
///
/// 核心特性：
/// - PluginRegistry（本地 + 远程注册表）
/// - 插件发现与搜索
/// - 插件下载与安装
/// - 版本管理与更新
/// - 签名验证与安全检查
/// - 依赖解析
pub mod plugin_marketplace;

/// [Phase 4 Week 1-2] 模型加密系统
///
/// 核心特性：
/// - AES-256-GCM 加密 GGUF 模型
/// - 运行时内存解密（零拷贝）
/// - 密钥自动擦除（zeroize）
/// - 多种密钥源支持（环境变量/文件/KMS/硬件）
pub mod model_encryption;

/// [Phase 4 Week 1-2] 多租户隔离系统
///
/// 核心特性：
/// - 租户级别资源隔离（Session/KV Cache/Plugin）
/// - 细粒度资源配额控制
/// - 租户生命周期管理
/// - 资源使用统计与监控
pub mod multi_tenancy;

/// [Phase 4 Week 3-4] 多模态支持（Vision）
///
/// 核心特性：
/// - Vision Encoder 接口（CLIP/SigLIP）
/// - CLIP ViT-L/14@336 实现
/// - 图像预处理 pipeline
/// - 多模态 KV Cache
pub mod multimodal;

/// [Phase 4 Week 5-6] 插件市场客户端
///
/// 核心特性：
/// - 插件搜索与发现
/// - 插件下载与安装
/// - 版本管理与更新
/// - 签名验证
pub mod marketplace_client;

/// [Phase 4 Week 7-8] 高级量化格式
///
/// 核心特性：
/// - IQ2_XXS（2-bit 重要性加权量化）
/// - BitNet b1.58（三元量化 {-1, 0, +1}）
/// - 统一量化接口
/// - 高性能反量化
pub mod quantization;

/// [Phase 4 Week 7-8] Kernel 融合优化
///
/// 核心特性：
/// - RMSNorm + RoPE 融合
/// - MatMul + Add 融合（GEMM + Bias）
/// - LayerNorm + Linear 融合
/// - SIMD 优化（AVX2/AVX512）
pub mod kernel_fusion;

/// 配置管理系统
///
/// 核心特性：
/// - 多格式支持（TOML/JSON）
/// - 环境变量覆盖
/// - 配置验证
/// - 配置热重载
pub mod config;

// ==================== 重导出 ====================

pub use backend::{
    ComputeBackend,
    DeviceInfo,
    BackendType,
    detect_backend,
};

pub use gguf::{
    GGUFModel,
    GGUFMetadata,
    TensorInfo,
};

pub use engine::{
    LociEngine,
    EngineConfig,
    PerformanceStats,
};

pub use sampling::{
    Sampler,
    SamplerConfig,
};

pub use paged_attention::{
    SessionManager,
    SessionId,
    PhysicalBlockId,
    LogicalBlockId,
    BlockTable,
    PhysicalBlock,
    BlockLocation,
    BLOCK_SIZE,
};

pub use constraints::{
    Constraint,
    ConstraintContext,
    TokenMask,
    RegexConstraint,
    JsonSchemaConstraint,
    JsonType,
    JsonState,
    AndConstraint,
    OrConstraint,
};

pub use suspend::{
    ControlFlow,
    SuspendReason,
    StopReason,
    SessionState,
    ResumeContext,
    InjectionType,
    SuspendableSession,
    SuspendableSessionManager,
    SessionInfo,
};

pub use radix_tree::{
    RadixNode,
    RadixTree,
    RadixTreeStats,
    KVCacheManager,
    PrefixCacheStats,
    TokenId,
    NodeId,
    CacheBlockId,
};

pub use plugin_system::{
    Plugin,
    PluginType,
    PluginMetadata,
    PluginControlFlow,
    PluginContext,
    LogitsView,
    NativePlugin,
    WasmPlugin,
    PluginRegistry,
    PluginRegistryStats,
    SignatureVerifier,
    ResourceQuota,
    Watchdog,
};

pub use model_registry::{
    ModelID,
    LoRAID,
    SessionID,
    ModelMetadata,
    LoRAConfig,
    LoRAAdapter,
    LoadedModel,
    ModelRegistry,
    MODEL_REGISTRY,
};

pub use lora::{
    LoRATensor,
    TensorDataType,
    LoRALayer,
    LoRAModel,
    LoRAStats,
    LoRAManager,
    create_example_lora_layer,
};

pub use plugin_marketplace::{
    PluginManifest,
    PluginAuthor,
    PluginKind,
    PluginDependency,
    PluginHooks,
    PluginLimits,
    PluginDownloadInfo,
    PluginRegistry as MarketplaceRegistry,
    InstalledPlugin,
};

pub use model_encryption::{
    EncryptedModelConfig,
    EncryptedModelLoader,
    KeySource,
    generate_key,
};

pub use multi_tenancy::{
    TenantID,
    TenantQuota,
    TenantResourceUsage,
    TenantContext,
    TenantManager,
    TenantSessionID,
};

pub use multimodal::{
    ImageBuffer,
    Tensor,
    VisionEncoder,
    CLIPVisionEncoder,
    TokenType,
    TypedToken,
    MultimodalKVCache,
};

pub use marketplace_client::{
    MarketplaceClient,
    MarketplaceClientConfig,
    PluginSearchResult,
    PluginSummary,
    PluginDetails,
    PluginUpdate,
};

pub use quantization::{
    QuantizationType,
    QuantizationScheme,
    QuantizedTensor,
    QuantizationMetadata,
    Iq2Xxs,
    BitNet158,
    QuantizationManager,
};

pub use kernel_fusion::{
    RMSNormParams,
    RoPEParams,
    RMSNormRoPEFusion,
    MatMulAddFusion,
    LayerNormParams,
    LayerNormLinearFusion,
    KernelFusionManager,
};

pub use config::{
    LociConfig,
    EngineSettings,
    BackendSettings,
    MemorySettings,
    PluginSettings,
    LoggingSettings,
    ServerSettings,
    ConfigLoader,
};

// ==================== 版本信息 ====================

/// Phase 2 版本
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Phase 2 构建信息
pub const BUILD_INFO: &str = concat!(
    "Loci Phase 2 Week 1 - ",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("CARGO_PKG_NAME"),
    ")"
);

/// 打印 Phase 2 横幅
pub fn print_banner() {
    eprintln!("╔════════════════════════════════════════╗");
    eprintln!("║         Loci Phase 2 Engine            ║");
    eprintln!("║  Paged Attention + Memory Budgeter     ║");
    eprintln!("╚════════════════════════════════════════╝");
    eprintln!("  Version: {}", VERSION);
    eprintln!("  Features: 128k+ Context, Multi-Session");
    eprintln!();
}

// ==================== 快速启动API ====================

/// Phase 1快速启动：自动配置并加载模型
///
/// 示例：
/// ```no_run
/// use loci::quick_load;
///
/// let engine = quick_load("path/to/model.gguf").expect("Failed to load model");
/// let output = engine.generate("Hello, world!", 50).expect("Generation failed");
/// println!("{}", output);
/// ```
pub fn quick_load(model_path: &str) -> anyhow::Result<LociEngine> {
    print_banner();

    let config = EngineConfig {
        model_path: model_path.to_string(),
        n_gpu_layers: -1,  // 自动探测
        ..Default::default()
    };

    LociEngine::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        println!("Loci version: {}", VERSION);
    }

    #[test]
    fn test_backend_detection() {
        let backend = detect_backend();
        assert!(backend.is_available());
        println!("Detected backend: {}", backend.name());
    }
}
