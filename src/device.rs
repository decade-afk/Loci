//! Device detection and automatic backend selection
//!
//! This module provides runtime device detection and automatic selection
//! of the optimal inference backend based on available hardware.

use crate::error::Result;
use std::fmt;

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum DeviceType {
    /// CPU only (fallback)
    CPU = 0,
    /// NVIDIA CUDA
    CUDA = 1,
    /// Apple Metal (Apple Silicon)
    Metal = 2,
    /// Vulkan (cross-platform)
    Vulkan = 3,
    /// AMD ROCm
    ROCm = 4,
    /// OpenCL (fallback GPU)
    OpenCL = 5,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::CUDA => write!(f, "CUDA"),
            DeviceType::Metal => write!(f, "Metal"),
            DeviceType::Vulkan => write!(f, "Vulkan"),
            DeviceType::ROCm => write!(f, "ROCm"),
            DeviceType::OpenCL => write!(f, "OpenCL"),
        }
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device ID (0-based)
    pub id: i32,
    /// Device name/model
    pub name: String,
    /// Total memory in bytes
    pub memory_bytes: u64,
    /// Device type
    pub device_type: DeviceType,
    /// Compute capability (CUDA) or equivalent
    pub compute_capability: f32,
    /// Is this device available and usable
    pub available: bool,
}

impl DeviceInfo {
    /// Get memory in GB
    pub fn memory_gb(&self) -> f32 {
        self.memory_bytes as f32 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if device has enough memory for model
    pub fn has_memory_for(&self, required_bytes: u64) -> bool {
        self.memory_bytes >= required_bytes
    }
}

/// Device configuration for inference
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Device type to use
    pub device_type: DeviceType,
    /// Device ID (for multi-GPU)
    pub device_id: i32,
    /// Number of GPU layers (-1 = all)
    pub n_gpu_layers: i32,
}

impl DeviceConfig {
    /// Create CPU-only configuration
    pub fn cpu_only() -> Self {
        Self {
            device_type: DeviceType::CPU,
            device_id: 0,
            n_gpu_layers: 0,
        }
    }

    /// Create GPU configuration with all layers
    pub fn gpu_all(device_type: DeviceType, device_id: i32) -> Self {
        Self {
            device_type,
            device_id,
            n_gpu_layers: -1,
        }
    }

    /// Create GPU configuration with specified layers
    pub fn gpu_layers(device_type: DeviceType, device_id: i32, n_layers: i32) -> Self {
        Self {
            device_type,
            device_id,
            n_gpu_layers: n_layers,
        }
    }
}

/// Device selector for automatic backend selection
pub struct DeviceSelector {
    available_devices: Vec<DeviceInfo>,
}

impl DeviceSelector {
    /// Create new device selector and detect all devices
    pub fn new() -> Self {
        let available_devices = Self::detect_all_devices();
        Self { available_devices }
    }

    /// Get all detected devices
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.available_devices
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        self.available_devices.len()
    }

    /// Get device by index
    pub fn device(&self, index: usize) -> Option<&DeviceInfo> {
        self.available_devices.get(index)
    }

    /// Automatically select the best device
    ///
    /// Priority: CUDA > Metal > Vulkan > ROCm > OpenCL > CPU
    pub fn auto_select(&self) -> DeviceConfig {
        // Try CUDA first (best performance)
        if let Some(cuda) = self.find_best_device(DeviceType::CUDA) {
            return DeviceConfig::gpu_all(DeviceType::CUDA, cuda.id);
        }

        // Try Metal (Apple Silicon)
        if let Some(metal) = self.find_best_device(DeviceType::Metal) {
            return DeviceConfig::gpu_all(DeviceType::Metal, metal.id);
        }

        // Try Vulkan (cross-platform)
        if let Some(vulkan) = self.find_best_device(DeviceType::Vulkan) {
            return DeviceConfig::gpu_all(DeviceType::Vulkan, vulkan.id);
        }

        // Try ROCm (AMD)
        if let Some(rocm) = self.find_best_device(DeviceType::ROCm) {
            return DeviceConfig::gpu_all(DeviceType::ROCm, rocm.id);
        }

        // Try OpenCL (fallback)
        if let Some(opencl) = self.find_best_device(DeviceType::OpenCL) {
            return DeviceConfig::gpu_all(DeviceType::OpenCL, opencl.id);
        }

        // Fallback to CPU
        DeviceConfig::cpu_only()
    }

    /// Find best device of specific type (most memory)
    fn find_best_device(&self, device_type: DeviceType) -> Option<&DeviceInfo> {
        self.available_devices
            .iter()
            .filter(|d| d.device_type == device_type && d.available)
            .max_by_key(|d| d.memory_bytes)
    }

    /// Detect all available devices
    fn detect_all_devices() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        // Detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda_devices) = Self::detect_cuda_devices() {
                devices.extend(cuda_devices);
            }
        }

        // Detect Metal devices
        #[cfg(feature = "metal")]
        {
            if let Ok(metal_devices) = Self::detect_metal_devices() {
                devices.extend(metal_devices);
            }
        }

        // Detect Vulkan devices
        #[cfg(feature = "vulkan")]
        {
            if let Ok(vulkan_devices) = Self::detect_vulkan_devices() {
                devices.extend(vulkan_devices);
            }
        }

        // Detect ROCm devices
        #[cfg(feature = "rocm")]
        {
            if let Ok(rocm_devices) = Self::detect_rocm_devices() {
                devices.extend(rocm_devices);
            }
        }

        // Always add CPU as fallback
        devices.push(DeviceInfo {
            id: 0,
            name: "CPU".to_string(),
            memory_bytes: Self::get_system_memory(),
            device_type: DeviceType::CPU,
            compute_capability: 0.0,
            available: true,
        });

        devices
    }

    /// Detect NVIDIA CUDA devices
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>> {
        use std::process::Command;

        // Try nvidia-smi first (most reliable)
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let devices: Vec<DeviceInfo> = stdout
                    .lines()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 3 {
                            Some(DeviceInfo {
                                id: parts[0].parse().unwrap_or(0),
                                name: format!("NVIDIA {}", parts[1]),
                                memory_bytes: parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024,
                                device_type: DeviceType::CUDA,
                                compute_capability: 0.0, // Would need CUDA runtime API
                                available: true,
                            })
                        } else {
                            None
                        }
                    })
                    .collect();

                if !devices.is_empty() {
                    return Ok(devices);
                }
            }
        }

        Ok(Vec::new())
    }

    /// Detect NVIDIA CUDA devices (no feature)
    #[cfg(not(feature = "cuda"))]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Detect Apple Metal devices
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn detect_metal_devices() -> Result<Vec<DeviceInfo>> {
        use std::process::Command;

        // Check if running on Apple Silicon
        if let Ok(output) = Command::new("sysctl")
            .args(&["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let cpu_brand = String::from_utf8_lossy(&output.stdout);
            if cpu_brand.contains("Apple") {
                // Apple Silicon detected
                let memory = Self::get_system_memory();

                return Ok(vec![DeviceInfo {
                    id: 0,
                    name: "Apple Silicon GPU".to_string(),
                    memory_bytes: memory / 2, // Unified memory, assume half for GPU
                    device_type: DeviceType::Metal,
                    compute_capability: 0.0,
                    available: true,
                }]);
            }
        }

        Ok(Vec::new())
    }

    /// Detect Metal devices (not macOS)
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn detect_metal_devices() -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Detect Vulkan devices
    #[cfg(feature = "vulkan")]
    fn detect_vulkan_devices() -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Detect Vulkan devices (no feature)
    #[cfg(not(feature = "vulkan"))]
    fn detect_vulkan_devices() -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Detect AMD ROCm devices
    #[cfg(feature = "rocm")]
    fn detect_rocm_devices() -> Result<Vec<DeviceInfo>> {
        use std::process::Command;

        // Try rocm-smi
        if let Ok(output) = Command::new("rocm-smi").arg("--showproductname").output() {
            if output.status.success() {
                return Ok(vec![DeviceInfo {
                    id: 0,
                    name: "AMD ROCm GPU".to_string(),
                    memory_bytes: 8 * 1024 * 1024 * 1024,
                    device_type: DeviceType::ROCm,
                    compute_capability: 0.0,
                    available: true,
                }]);
            }
        }

        Ok(Vec::new())
    }

    /// Detect ROCm devices (no feature)
    #[cfg(not(feature = "rocm"))]
    fn detect_rocm_devices() -> Result<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Get system memory size
    fn get_system_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(content) = fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb) = line.split_whitespace().nth(1) {
                            if let Ok(kb_val) = kb.parse::<u64>() {
                                return kb_val * 1024;
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use std::mem;

            #[repr(C)]
            struct MemoryStatusEx {
                dw_length: u32,
                dw_memory_load: u32,
                ull_total_phys: u64,
                ull_avail_phys: u64,
                ull_total_page_file: u64,
                ull_avail_page_file: u64,
                ull_total_virtual: u64,
                ull_avail_virtual: u64,
                ull_avail_extended_virtual: u64,
            }

            extern "system" {
                fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
            }

            unsafe {
                let mut status: MemoryStatusEx = mem::zeroed();
                status.dw_length = mem::size_of::<MemoryStatusEx>() as u32;

                if GlobalMemoryStatusEx(&mut status) != 0 {
                    return status.ull_total_phys;
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "hw.memsize"])
                .output()
            {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(mem) = mem_str.trim().parse::<u64>() {
                        return mem;
                    }
                }
            }
        }

        // Default: 16GB
        16 * 1024 * 1024 * 1024
    }

    /// Check if specific backend is available
    pub fn has_backend(&self, device_type: DeviceType) -> bool {
        self.available_devices
            .iter()
            .any(|d| d.device_type == device_type && d.available)
    }

    /// Get recommended configuration for model size
    pub fn recommend_for_model(&self, model_size_gb: f32) -> DeviceConfig {
        let required_memory = (model_size_gb * 1.5 * 1024.0 * 1024.0 * 1024.0) as u64;

        // Find devices with enough memory
        let suitable: Vec<_> = self
            .available_devices
            .iter()
            .filter(|d| {
                d.available
                    && d.device_type != DeviceType::CPU
                    && d.memory_bytes >= required_memory
            })
            .collect();

        if let Some(best) = suitable.iter().max_by_key(|d| d.memory_bytes) {
            return DeviceConfig::gpu_all(best.device_type, best.id);
        }

        // Check if CPU has enough memory
        if let Some(cpu) = self.find_best_device(DeviceType::CPU) {
            if cpu.memory_bytes >= required_memory {
                return DeviceConfig::cpu_only();
            }
        }

        // Fallback: use GPU with partial offloading
        if let Some(best_gpu) = self
            .available_devices
            .iter()
            .filter(|d| d.available && d.device_type != DeviceType::CPU)
            .max_by_key(|d| d.memory_bytes)
        {
            // Estimate layers that fit in GPU memory
            let estimated_layers =
                ((best_gpu.memory_bytes as f32 / required_memory as f32) * 32.0) as i32;
            return DeviceConfig::gpu_layers(
                best_gpu.device_type,
                best_gpu.id,
                estimated_layers.max(1),
            );
        }

        // Last resort: CPU only
        DeviceConfig::cpu_only()
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selector_creation() {
        let selector = DeviceSelector::new();
        assert!(selector.device_count() >= 1); // At least CPU
    }

    #[test]
    fn test_auto_select() {
        let selector = DeviceSelector::new();
        let config = selector.auto_select();
        assert!(config.device_id >= 0);
    }

    #[test]
    fn test_cpu_fallback() {
        let selector = DeviceSelector::new();
        assert!(selector.has_backend(DeviceType::CPU));
    }

    #[test]
    fn test_recommend_for_model() {
        let selector = DeviceSelector::new();
        let config = selector.recommend_for_model(7.0); // 7GB model
        assert!(config.device_id >= 0);
    }
}
