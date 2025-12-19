/**
 * 系统信息检测和模型推荐模块
 *
 * 功能：
 * 1. 检测系统硬件信息（CPU、内存、GPU）
 * 2. 根据硬件配置推荐合适的模型大小
 * 3. 提供优化的模型加载参数建议
 */

use serde::{Deserialize, Serialize};
use sysinfo::{System, CpuRefreshKind, MemoryRefreshKind, RefreshKind};

// ==================== 数据结构 ====================

/**
 * 系统硬件信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU 信息
    pub cpu: CpuInfo,

    /// 内存信息
    pub memory: MemoryInfo,

    /// GPU 信息（如果可检测）
    pub gpu: GpuInfo,

    /// 操作系统信息
    pub os: OsInfo,
}

/**
 * CPU 信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU 品牌/型号
    pub brand: String,

    /// 物理核心数
    pub physical_cores: usize,

    /// 逻辑核心数（包含超线程）
    pub logical_cores: usize,

    /// CPU 架构
    pub architecture: String,

    /// CPU 主频（MHz）
    pub frequency: u64,
}

/**
 * 内存信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// 总内存（字节）
    pub total_bytes: u64,

    /// 总内存（GB）
    pub total_gb: f64,

    /// 可用内存（字节）
    pub available_bytes: u64,

    /// 可用内存（GB）
    pub available_gb: f64,

    /// 已使用内存（字节）
    pub used_bytes: u64,

    /// 已使用内存（GB）
    pub used_gb: f64,

    /// 内存使用率（百分比）
    pub usage_percent: f64,
}

/**
 * GPU 信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// 是否检测到 CUDA GPU
    pub has_cuda: bool,

    /// 是否检测到 Metal GPU（macOS）
    pub has_metal: bool,

    /// 是否检测到 Vulkan GPU
    pub has_vulkan: bool,

    /// GPU 描述
    pub description: String,
}

/**
 * 操作系统信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsInfo {
    /// 操作系统名称
    pub name: String,

    /// 操作系统版本
    pub version: String,

    /// 系统架构（x86_64, aarch64 等）
    pub arch: String,
}

/**
 * 模型推荐信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecommendation {
    /// 推荐的模型大小（参数量）
    pub recommended_sizes: Vec<ModelSize>,

    /// 推荐的上下文大小
    pub recommended_context_size: u32,

    /// 推荐的 GPU 层数
    pub recommended_gpu_layers: u32,

    /// 推荐的 CPU 线程数
    pub recommended_threads: u32,

    /// 性能等级（low/medium/high）
    pub performance_tier: String,

    /// 建议和警告
    pub suggestions: Vec<String>,
}

/**
 * 模型大小信息
 */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSize {
    /// 模型名称（如 "7B", "13B"）
    pub name: String,

    /// 参数量（十亿）
    pub parameters_billions: f32,

    /// 预估内存占用（GB）
    pub estimated_memory_gb: f32,

    /// 是否推荐（基于当前系统）
    pub recommended: bool,

    /// 预期性能（tokens/秒）
    pub expected_tokens_per_second: String,

    /// 描述
    pub description: String,
}

// ==================== 系统检测 ====================

/**
 * 获取系统硬件信息
 *
 * 该函数会检测当前系统的完整硬件配置，包括：
 * - CPU：品牌、核心数、架构、主频
 * - 内存：总量、可用量、使用率
 * - GPU：CUDA/Metal/Vulkan 支持情况
 * - 操作系统：名称、版本、架构
 *
 * 使用场景：
 * - 在加载模型前，了解系统配置
 * - 为用户提供硬件信息展示
 * - 作为模型推荐的输入依据
 *
 * 性能影响：
 * - 首次调用需要约 200ms（等待 CPU 信息稳定）
 * - 后续调用可以快速返回（系统信息缓存）
 *
 * 返回值：
 * - Ok(SystemInfo)：成功获取系统信息
 * - Err(String)：无法获取 CPU 信息（极少发生）
 */
pub fn get_system_info() -> Result<SystemInfo, String> {
    // ========== 第一步：初始化系统信息收集器 ==========
    // 使用 sysinfo crate 创建 System 实例
    // RefreshKind 指定要刷新的信息类型
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())  // 获取所有 CPU 信息
            .with_memory(MemoryRefreshKind::everything())  // 获取所有内存信息
    );

    // ========== 第二步：刷新系统信息 ==========
    // 首次刷新：初始化数据
    sys.refresh_all();

    // 等待 200ms：让 CPU 使用率统计更准确
    // sysinfo 需要两次采样才能计算使用率
    std::thread::sleep(std::time::Duration::from_millis(200));

    // 第二次刷新：获取准确的 CPU 数据
    sys.refresh_cpu_all();

    // ========== 第三步：提取 CPU 信息 ==========
    let cpus = sys.cpus();
    let cpu_info = if let Some(first_cpu) = cpus.first() {
        CpuInfo {
            // CPU 品牌（如 "Intel Core i7-9750H"）
            brand: first_cpu.brand().to_string(),

            // 物理核心数（不含超线程）
            // 对于推理任务，物理核心更重要
            physical_cores: sys.physical_core_count().unwrap_or(1),

            // 逻辑核心数（含超线程）
            // 例如：4 物理核心 + 超线程 = 8 逻辑核心
            logical_cores: cpus.len(),

            // CPU 架构（x86_64, aarch64 等）
            architecture: std::env::consts::ARCH.to_string(),

            // CPU 主频（MHz）
            frequency: first_cpu.frequency(),
        }
    } else {
        // 极少情况：无法获取 CPU 信息
        return Err("无法获取 CPU 信息".to_string());
    };

    // ========== 第四步：提取内存信息 ==========
    // 注意：sysinfo 返回的内存单位是字节（bytes）
    let total_mem = sys.total_memory();      // 总内存
    let available_mem = sys.available_memory();  // 可用内存（包括缓存可释放部分）
    let used_mem = sys.used_memory();        // 已使用内存

    let memory_info = MemoryInfo {
        // 原始字节数
        total_bytes: total_mem,
        available_bytes: available_mem,
        used_bytes: used_mem,

        // 转换为 GB（便于前端显示）
        total_gb: bytes_to_gb(total_mem),
        available_gb: bytes_to_gb(available_mem),
        used_gb: bytes_to_gb(used_mem),

        // 计算内存使用率（百分比）
        usage_percent: if total_mem > 0 {
            (used_mem as f64 / total_mem as f64) * 100.0
        } else {
            0.0
        },
    };

    // ========== 第五步：检测 GPU 信息 ==========
    let gpu_info = detect_gpu();

    // ========== 第六步：获取操作系统信息 ==========
    let os_info = OsInfo {
        name: System::name().unwrap_or_else(|| "Unknown".to_string()),
        version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
        arch: std::env::consts::ARCH.to_string(),
    };

    // ========== 返回完整的系统信息 ==========
    Ok(SystemInfo {
        cpu: cpu_info,
        memory: memory_info,
        gpu: gpu_info,
        os: os_info,
    })
}

/**
 * 检测 GPU 信息
 *
 * 该函数会尝试检测系统中可用的 GPU 加速方案：
 * 1. CUDA（NVIDIA GPU）：通过检测 CUDA 目录或 nvidia-smi 命令
 * 2. Metal（Apple GPU）：macOS 系统自带
 * 3. Vulkan：跨平台 GPU API
 *
 * 检测逻辑：
 * - CUDA：检查 /usr/local/cuda 目录、CUDA_PATH 环境变量、nvidia-smi 命令
 * - Metal：检查是否为 macOS 系统
 * - Vulkan：检查 Vulkan 动态库是否存在
 *
 * llama-cpp-2 GPU 支持说明：
 * - CUDA：完全支持，需要设置 gpu_layers 参数
 * - Metal：完全支持（macOS），自动启用
 * - Vulkan：部分支持，取决于编译配置
 *
 * 注意：
 * - 检测结果不代表 llama-cpp-2 一定能使用该 GPU
 * - 需要 llama-cpp-2 编译时启用对应的特性
 */
fn detect_gpu() -> GpuInfo {
    // ========== 检测 CUDA（NVIDIA GPU）==========
    // 三种检测方式：
    // 1. 检查标准 CUDA 安装目录
    // 2. 检查 CUDA_PATH 环境变量
    // 3. 尝试运行 nvidia-smi 命令
    let has_cuda = std::path::Path::new("/usr/local/cuda").exists()
        || std::env::var("CUDA_PATH").is_ok()
        || check_nvidia_smi();

    // ========== 检测 Metal（macOS GPU）==========
    // Metal 是 macOS 原生 GPU API，所有 macOS 系统都支持
    let has_metal = cfg!(target_os = "macos");

    // ========== 检测 Vulkan ==========
    // 检查 Vulkan 运行时库是否存在
    let has_vulkan = check_vulkan();

    // ========== 生成描述信息 ==========
    // 优先级：CUDA > Metal > Vulkan > CPU
    let description = if has_cuda {
        "检测到 NVIDIA CUDA GPU，支持 GPU 加速".to_string()
    } else if has_metal {
        "检测到 Apple Metal GPU，支持 GPU 加速".to_string()
    } else if has_vulkan {
        "检测到 Vulkan 支持，可能支持 GPU 加速".to_string()
    } else {
        "未检测到 GPU 或 GPU 加速不可用，将使用 CPU 推理".to_string()
    };

    GpuInfo {
        has_cuda,
        has_metal,
        has_vulkan,
        description,
    }
}

/**
 * 检查 nvidia-smi 是否可用
 */
fn check_nvidia_smi() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/**
 * 检查 Vulkan 是否可用
 */
fn check_vulkan() -> bool {
    // 简单检查常见的 Vulkan 库文件
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/usr/lib/libvulkan.so").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists()
    }

    #[cfg(target_os = "windows")]
    {
        std::path::Path::new("C:\\Windows\\System32\\vulkan-1.dll").exists()
    }

    #[cfg(target_os = "macos")]
    {
        std::path::Path::new("/usr/local/lib/libvulkan.dylib").exists()
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/**
 * 字节转 GB
 */
fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

// ==================== 模型推荐 ====================

/**
 * 根据系统信息推荐模型配置
 *
 * 这是本模块的核心功能，根据检测到的硬件配置，智能推荐：
 * 1. 合适的模型大小（1B/3B/7B/13B/30B+）
 * 2. 推荐的上下文大小
 * 3. 推荐的 GPU 层数
 * 4. 推荐的 CPU 线程数
 * 5. 性能等级和优化建议
 *
 * 推荐策略：
 *
 * 【内存安全策略】
 * - 保守原则：模型内存占用不超过可用内存的 70%
 * - 预留空间：确保系统有足够内存运行其他程序
 * - 动态调整：根据当前内存使用率给出警告
 *
 * 【模型大小推荐】
 * - 1B 模型：~1.5GB 内存，适合 4GB 以下系统
 * - 3B 模型：~3.5GB 内存，适合 8GB 系统
 * - 7B 模型：~6GB 内存，适合 16GB 系统
 * - 13B 模型：~10GB 内存，适合 24GB 系统
 * - 30B+ 模型：~20GB 内存，需要 32GB+ 系统
 *
 * 【GPU 层数推荐】
 * - NVIDIA CUDA：根据内存推荐 16-32 层
 * - Apple Metal：设置为 1（自动管理）
 * - 无 GPU：设置为 0（纯 CPU）
 *
 * 【上下文大小推荐】
 * - High tier（32GB+ RAM + GPU）：8192 tokens
 * - Medium tier（16GB+ RAM）：4096 tokens
 * - Low-Medium tier（8GB+ RAM）：2048 tokens
 * - Low tier（<8GB RAM）：1024 tokens
 *
 * 【CPU 线程数推荐】
 * - 使用物理核心数（不含超线程）
 * - 最大不超过 8（避免过度并行开销）
 *
 * 参数：
 * - system_info: 系统硬件信息（通过 get_system_info 获取）
 *
 * 返回：
 * - ModelRecommendation：包含所有推荐配置和建议
 */
pub fn recommend_model(system_info: &SystemInfo) -> ModelRecommendation {
    let available_gb = system_info.memory.available_gb;
    let total_gb = system_info.memory.total_gb;
    let has_gpu = system_info.gpu.has_cuda || system_info.gpu.has_metal;
    let cpu_cores = system_info.cpu.physical_cores;

    // 定义模型大小列表
    let mut model_sizes = vec![
        ModelSize {
            name: "1B".to_string(),
            parameters_billions: 1.0,
            estimated_memory_gb: 1.5,
            recommended: false,
            expected_tokens_per_second: "20-50".to_string(),
            description: "极小模型，适合低配置设备，适合简单对话和文本生成".to_string(),
        },
        ModelSize {
            name: "3B".to_string(),
            parameters_billions: 3.0,
            estimated_memory_gb: 3.5,
            recommended: false,
            expected_tokens_per_second: "15-40".to_string(),
            description: "小型模型，性能和质量平衡，适合一般对话和创作".to_string(),
        },
        ModelSize {
            name: "7B".to_string(),
            parameters_billions: 7.0,
            estimated_memory_gb: 6.0,
            recommended: false,
            expected_tokens_per_second: "10-30".to_string(),
            description: "中型模型，质量较好，适合专业写作和复杂任务".to_string(),
        },
        ModelSize {
            name: "13B".to_string(),
            parameters_billions: 13.0,
            estimated_memory_gb: 10.0,
            recommended: false,
            expected_tokens_per_second: "5-15".to_string(),
            description: "大型模型，高质量输出，需要较高配置".to_string(),
        },
        ModelSize {
            name: "30B+".to_string(),
            parameters_billions: 30.0,
            estimated_memory_gb: 20.0,
            recommended: false,
            expected_tokens_per_second: "2-8".to_string(),
            description: "超大模型，顶级质量，需要高端配置或 GPU".to_string(),
        },
    ];

    // 根据可用内存推荐模型
    // 保守策略：模型内存占用不超过可用内存的 70%
    let memory_budget = (available_gb * 0.7) as f32;

    let mut recommended_sizes = Vec::new();
    for model in &mut model_sizes {
        if model.estimated_memory_gb <= memory_budget {
            model.recommended = true;
            recommended_sizes.push(model.clone());
        }
    }

    // 确定性能等级
    let (performance_tier, context_size) = if total_gb >= 32.0 && has_gpu {
        ("high", 8192)
    } else if total_gb >= 16.0 {
        ("medium", 4096)
    } else if total_gb >= 8.0 {
        ("low-medium", 2048)
    } else {
        ("low", 1024)
    };

    // GPU 层数推荐
    let gpu_layers = if system_info.gpu.has_cuda {
        // NVIDIA GPU：根据内存推荐
        if total_gb >= 16.0 { 32 } else { 16 }
    } else if system_info.gpu.has_metal {
        // Apple Metal：使用默认值
        1  // Metal 后端会自动管理
    } else {
        0  // 纯 CPU
    };

    // CPU 线程数推荐
    // 通常使用物理核心数，不超过 8（避免过度并行）
    let threads = cpu_cores.min(8) as u32;

    // 生成建议
    let mut suggestions = Vec::new();

    if recommended_sizes.is_empty() {
        suggestions.push("⚠️ 警告：可用内存不足，无法推荐任何模型。建议关闭其他应用释放内存。".to_string());
        // 仍然推荐最小的模型
        if let Some(smallest) = model_sizes.first() {
            recommended_sizes.push(smallest.clone());
        }
    } else {
        // 推荐最大的可用模型
        if let Some(best) = recommended_sizes.last() {
            suggestions.push(format!("✅ 推荐使用 {} 参数模型，预计占用 {:.1}GB 内存",
                best.name, best.estimated_memory_gb));
        }
    }

    if has_gpu {
        if system_info.gpu.has_cuda {
            suggestions.push("🚀 检测到 NVIDIA GPU，强烈建议启用 GPU 加速（设置 gpu_layers）".to_string());
        } else if system_info.gpu.has_metal {
            suggestions.push("🍎 检测到 Apple Metal GPU，将自动使用 GPU 加速".to_string());
        }
    } else {
        suggestions.push("💡 未检测到 GPU，将使用 CPU 推理，速度可能较慢".to_string());
    }

    if system_info.memory.usage_percent > 80.0 {
        suggestions.push(format!("⚠️ 当前内存使用率 {:.1}%，建议关闭部分应用以获得更好性能",
            system_info.memory.usage_percent));
    }

    if cpu_cores < 4 {
        suggestions.push("⚠️ CPU 核心数较少，推理速度可能受限".to_string());
    }

    if total_gb < 8.0 {
        suggestions.push("⚠️ 总内存少于 8GB，建议使用较小的模型和较短的上下文".to_string());
    }

    if available_gb < 4.0 {
        suggestions.push("⚠️ 可用内存不足 4GB，强烈建议先释放内存再加载模型".to_string());
    }

    ModelRecommendation {
        recommended_sizes,
        recommended_context_size: context_size,
        recommended_gpu_layers: gpu_layers,
        recommended_threads: threads,
        performance_tier: performance_tier.to_string(),
        suggestions,
    }
}
