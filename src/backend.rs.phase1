/**
 * Phase 1: Backend抽象层
 *
 * 实现ComputeBackend trait统一接口，支持：
 * - CUDA (NVIDIA GPU)
 * - Metal (Apple Silicon)
 * - ROCm (AMD GPU)
 * - CPU (Fallback + AVX512)
 *
 * 运行时自动探测并选择最优Backend
 */

use std::sync::Arc;
use anyhow::{Result, Context};

// ==================== Backend Trait定义 ====================

/// 统一的计算后端接口
pub trait ComputeBackend: Send + Sync {
    /// Backend名称
    fn name(&self) -> &str;

    /// 是否可用
    fn is_available(&self) -> bool;

    /// 初始化Backend
    fn initialize(&mut self) -> Result<()>;

    /// 获取设备信息
    fn device_info(&self) -> DeviceInfo;

    /// 获取推荐的GPU层数（0表示纯CPU）
    fn recommended_gpu_layers(&self) -> u32;

    /// 预分配显存暂存区（Scratch Buffer）
    fn allocate_scratch_buffer(&mut self, size_mb: usize) -> Result<()>;
}

/// 设备信息
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_available_mb: u64,
    pub compute_capability: String,
    pub backend_type: BackendType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cuda,
    Metal,
    Rocm,
    Vulkan,
    Cpu,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Rocm => write!(f, "ROCm"),
            BackendType::Vulkan => write!(f, "Vulkan"),
            BackendType::Cpu => write!(f, "CPU"),
        }
    }
}

// ==================== CUDA Backend ====================

pub struct CudaBackend {
    device_id: i32,
    initialized: bool,
    device_name: String,
    total_memory_mb: u64,
}

impl CudaBackend {
    pub fn new() -> Self {
        Self {
            device_id: 0,
            initialized: false,
            device_name: String::new(),
            total_memory_mb: 0,
        }
    }

    fn check_cuda_available() -> bool {
        // 检查nvidia-smi
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,memory.total")
            .arg("--format=csv,noheader")
            .output()
            .map(|output| {
                if output.status.success() {
                    let s = String::from_utf8_lossy(&output.stdout);
                    !s.is_empty()
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        Self::check_cuda_available()
    }

    fn initialize(&mut self) -> Result<()> {
        if !self.is_available() {
            anyhow::bail!("CUDA not available");
        }

        // 查询设备信息
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
            .context("Failed to query CUDA device")?;

        if output.status.success() {
            let line = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = line.trim().split(',').collect();
            if parts.len() >= 2 {
                self.device_name = parts[0].trim().to_string();
                self.total_memory_mb = parts[1].trim().parse().unwrap_or(0);
            }
        }

        self.initialized = true;
        eprintln!("✅ CUDA Backend initialized: {} ({} MB)",
                 self.device_name, self.total_memory_mb);
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        // 查询可用显存
        let available_mb = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.free")
            .arg("--format=csv,noheader,nounits")
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    String::from_utf8_lossy(&output.stdout)
                        .trim()
                        .parse::<u64>()
                        .ok()
                } else {
                    None
                }
            })
            .unwrap_or(self.total_memory_mb);  // 回退到总内存

        DeviceInfo {
            name: self.device_name.clone(),
            memory_total_mb: self.total_memory_mb,
            memory_available_mb: available_mb,
            compute_capability: "Unknown".to_string(),
            backend_type: BackendType::Cuda,
        }
    }

    fn recommended_gpu_layers(&self) -> u32 {
        // 根据显存推荐GPU层数
        // 7B模型约6GB，每层约200MB
        if self.total_memory_mb > 16000 {
            35  // 大部分层在GPU
        } else if self.total_memory_mb > 8000 {
            24
        } else if self.total_memory_mb > 4000 {
            16
        } else {
            0
        }
    }

    fn allocate_scratch_buffer(&mut self, size_mb: usize) -> Result<()> {
        // TODO: 调用CUDA API预分配显存
        eprintln!("📦 Allocating CUDA scratch buffer: {} MB", size_mb);
        Ok(())
    }
}

// ==================== Metal Backend ====================

pub struct MetalBackend {
    initialized: bool,
}

impl MetalBackend {
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos")
    }

    fn initialize(&mut self) -> Result<()> {
        if !self.is_available() {
            anyhow::bail!("Metal only available on macOS");
        }

        self.initialized = true;
        eprintln!("✅ Metal Backend initialized");
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "Apple Silicon GPU".to_string(),
            memory_total_mb: 0,  // Metal共享系统内存
            memory_available_mb: 0,
            compute_capability: "Metal 3.0".to_string(),
            backend_type: BackendType::Metal,
        }
    }

    fn recommended_gpu_layers(&self) -> u32 {
        // Metal自动管理，设置为1即可
        1
    }

    fn allocate_scratch_buffer(&mut self, _size_mb: usize) -> Result<()> {
        // Metal自动管理内存
        Ok(())
    }
}

// ==================== CPU Backend ====================

pub struct CpuBackend {
    avx512_available: bool,
    avx2_available: bool,
    cpu_cores: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            avx512_available: false,
            avx2_available: false,
            cpu_cores: num_cpus::get(),
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        if self.avx512_available {
            "CPU (AVX512)"
        } else if self.avx2_available {
            "CPU (AVX2)"
        } else {
            "CPU (Basic)"
        }
    }

    fn is_available(&self) -> bool {
        true  // CPU总是可用
    }

    fn initialize(&mut self) -> Result<()> {
        // 检测CPU特性
        #[cfg(target_arch = "x86_64")]
        {
            self.avx512_available = is_x86_feature_detected!("avx512f");
            self.avx2_available = is_x86_feature_detected!("avx2");
        }

        eprintln!("✅ CPU Backend initialized: {} cores, AVX512={}, AVX2={}",
                 self.cpu_cores, self.avx512_available, self.avx2_available);
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: format!("CPU ({} cores)", self.cpu_cores),
            memory_total_mb: 0,
            memory_available_mb: 0,
            compute_capability: self.name().to_string(),
            backend_type: BackendType::Cpu,
        }
    }

    fn recommended_gpu_layers(&self) -> u32 {
        0  // CPU模式
    }

    fn allocate_scratch_buffer(&mut self, _size_mb: usize) -> Result<()> {
        // CPU使用系统内存，无需预分配
        Ok(())
    }
}

// ==================== Backend自动探测 ====================

/// Phase 1核心功能：自动探测并返回最优Backend
pub fn detect_backend() -> Arc<dyn ComputeBackend> {
    eprintln!("🔍 Detecting optimal compute backend...");

    // 优先级：CUDA > Metal > CPU
    let mut cuda = CudaBackend::new();
    if cuda.is_available() {
        if cuda.initialize().is_ok() {
            eprintln!("🚀 Selected Backend: CUDA");
            return Arc::new(cuda);
        }
    }

    let mut metal = MetalBackend::new();
    if metal.is_available() {
        if metal.initialize().is_ok() {
            eprintln!("🚀 Selected Backend: Metal");
            return Arc::new(metal);
        }
    }

    // 回退到CPU
    let mut cpu = CpuBackend::new();
    cpu.initialize().expect("CPU backend should always work");
    eprintln!("🚀 Selected Backend: CPU (Fallback)");
    Arc::new(cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = detect_backend();
        assert!(backend.is_available());
        println!("Detected: {}", backend.name());
        println!("Device Info: {:?}", backend.device_info());
    }
}
