/**
 * Loci - Local AI Inference Core
 *
 * 端侧 AI 推理核心库，为 Creative Studio 提供本地大语言模型推理能力。
 *
 * ## 架构设计
 *
 * ```
 * loci (Core Library)
 * ├── engine     - AI 推理引擎（基于 llama.cpp）
 * ├── agent      - Agent 系统（多模型池 + 智能路由）
 * ├── sysinfo    - 系统信息检测（硬件分析 + 模型推荐）
 * ├── ffi        - FFI 接口层（C ABI + Flutter Bridge）
 * └── errors     - 统一错误处理
 * ```
 *
 * ## 使用场景
 *
 * ### Tauri Desktop（静态链接）
 *
 * ```rust
 * use loci::engine::{AIService, AIConfig};
 *
 * let service = AIService::new();
 * service.update_config(AIConfig {
 *     model_path: "/path/to/model.gguf".to_string(),
 *     context_size: 4096,
 *     gpu_layers: 32,
 *     ..Default::default()
 * });
 * service.load_model()?;
 *
 * let response = service.generate(GenerateRequest {
 *     prompt: "你好".to_string(),
 *     max_tokens: Some(100),
 *     ..Default::default()
 * })?;
 * ```
 *
 * ### Tauri Desktop（动态链接）
 *
 * ```rust
 * // 通过 FFI 调用
 * use loci::ffi::*;
 *
 * let service = loci_ai_service_new();
 * let config_json = serde_json::to_string(&config)?;
 * loci_ai_service_load_model(service, config_json.as_ptr());
 * ```
 *
 * ### Flutter Mobile（FFI Bridge）
 *
 * ```dart
 * import 'package:loci/loci.dart';
 *
 * final service = await LociAIService.create();
 * await service.loadModel(config);
 * final response = await service.generate(request);
 * ```
 *
 * ## 版本约定
 *
 * - 主版本号（Major）：不兼容的 API 变更
 * - 次版本号（Minor）：向后兼容的功能新增
 * - 修订号（Patch）：向后兼容的 Bug 修复
 *
 * 当前版本：v0.1.0（Alpha）
 */

// ==================== 版本信息 ====================

/// Loci Core 版本号
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Loci Core 版本号（用于 FFI）
///
/// C 函数签名：`const char* loci_version(void)`
#[no_mangle]
pub extern "C" fn loci_version() -> *const std::os::raw::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}

// ==================== 模块声明 ====================

/// AI 推理引擎模块
///
/// 提供基于 llama.cpp 的本地 LLM 推理能力。
///
/// 主要组件：
/// - `AIService`: AI 服务主体
/// - `AIConfig`: 配置参数
/// - `GenerateRequest` / `GenerateResponse`: 请求/响应
pub mod engine;

/// Agent 系统模块
///
/// 提供多模型池管理和智能 Agent 系统。
///
/// 主要组件：
/// - `AgentSystem`: Agent 系统主体
/// - `ModelConfig`: 模型配置
/// - `AgentConfig`: Agent 配置
pub mod agent;

/// 系统信息检测模块
///
/// 提供硬件检测和模型推荐功能。
///
/// 主要组件：
/// - `SystemInfo`: 系统信息
/// - `ModelRecommendation`: 模型推荐
pub mod sysinfo;

/// API 统一入口模块
///
/// 提供 Flutter Rust Bridge 和 C FFI 接口。
///
/// 主要组件：
/// - `LociAPI`: 高级 API 封装
/// - C 风格的 FFI 函数
pub mod api;

/// FFI 接口层
///
/// 提供 C ABI 兼容的接口，用于跨语言调用。
///
/// 用途：
/// - Tauri Desktop 动态链接
/// - Flutter FFI Bridge
pub mod ffi;

/// 错误类型定义
///
/// 统一的错误处理机制。
pub mod errors;

// ==================== 重导出（Re-exports）====================

// 重导出核心类型，方便外部使用
pub use engine::{AIService, AIConfig, GenerateRequest, GenerateResponse};
pub use agent::{AgentSystem, ModelConfig, AgentConfig, AgentGenerateRequest, AgentGenerateResponse};
pub use sysinfo::{SystemInfo, ModelRecommendation};
pub use api::{LociAPI, LociBuilder, create_loci};
pub use errors::{AIError, AgentError, SystemInfoError, ExportError};

// ==================== 初始化函数 ====================

/// 初始化 Loci Core
///
/// 这个函数应该在使用任何 Loci 功能之前调用一次。
///
/// 功能：
/// - 初始化 llama.cpp 后端
/// - 设置日志系统
/// - 检测硬件环境
///
/// # 示例
///
/// ```rust
/// use loci;
///
/// fn main() {
///     loci::init().expect("Failed to initialize Loci");
///
///     // 使用 Loci 功能...
/// }
/// ```
pub fn init() -> Result<(), String> {
    // TODO: 实现初始化逻辑
    // 1. 初始化 llama.cpp 后端
    // 2. 设置日志系统
    // 3. 检测硬件环境
    Ok(())
}

/// 关闭 Loci Core
///
/// 清理所有资源，应该在程序退出前调用。
///
/// # 示例
///
/// ```rust
/// use loci;
///
/// fn main() {
///     loci::init().unwrap();
///
///     // 使用 Loci 功能...
///
///     loci::shutdown();
/// }
/// ```
pub fn shutdown() {
    // TODO: 实现清理逻辑
    // 1. 释放所有模型
    // 2. 清理 llama.cpp 后端
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_init_shutdown() {
        assert!(init().is_ok());
        shutdown();
    }
}
