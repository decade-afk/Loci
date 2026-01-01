//! Sysinfo Module
//!
//! This module provides core functionality for the Loci project.
//!


use serde::{Deserialize, Serialize};
use sysinfo::{System, CpuRefreshKind, MemoryRefreshKind, RefreshKind};




#[derive(Debug, Clone, Serialize, Deserialize)]
    /// SystemInfo structure
pub struct SystemInfo {
    
    pub cpu: CpuInfo,

    
    pub memory: MemoryInfo,

    
    pub gpu: GpuInfo,

    
    pub os: OsInfo,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// CpuInfo structure
pub struct CpuInfo {
    
    pub brand: String,

    
    pub physical_cores: usize,

    
    pub logical_cores: usize,

    
    pub architecture: String,

    
    pub frequency: u64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// MemoryInfo structure
pub struct MemoryInfo {
    
    pub total_bytes: u64,

    
    pub total_gb: f64,

    
    pub available_bytes: u64,

    
    pub available_gb: f64,

    
    pub used_bytes: u64,

    
    pub used_gb: f64,

    
    pub usage_percent: f64,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// GpuInfo structure
pub struct GpuInfo {
    
    pub has_cuda: bool,

    
    pub has_metal: bool,

    
    pub has_vulkan: bool,

    
    pub description: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// OsInfo structure
pub struct OsInfo {
    
    pub name: String,

    
    pub version: String,

    
    pub arch: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// ModelRecommendation structure
pub struct ModelRecommendation {
    
    pub recommended_sizes: Vec<ModelSize>,

    
    pub recommended_context_size: u32,

    
    pub recommended_gpu_layers: u32,

    
    pub recommended_threads: u32,

    
    pub performance_tier: String,

    
    pub suggestions: Vec<String>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// ModelSize structure
pub struct ModelSize {
    
    pub name: String,

    
    pub parameters_billions: f32,

    
    pub estimated_memory_gb: f32,

    
    pub recommended: bool,

    
    pub expected_tokens_per_second: String,

    
    pub description: String,
}




    /// get_system_info function
pub fn get_system_info() -> Result<SystemInfo, String> {
    
    
    
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())  
            .with_memory(MemoryRefreshKind::everything())  
    );

    
    
    sys.refresh_all();

    
    
    std::thread::sleep(std::time::Duration::from_millis(200));

    
    sys.refresh_cpu_all();

    
    let cpus = sys.cpus();
    let cpu_info = if let Some(first_cpu) = cpus.first() {
        CpuInfo {
            
            brand: first_cpu.brand().to_string(),

            
            
            physical_cores: sys.physical_core_count().unwrap_or(1),

            
            
            logical_cores: cpus.len(),

            
            architecture: std::env::consts::ARCH.to_string(),

            
            frequency: first_cpu.frequency(),
        }
    } else {
        
        return Err("Unable to obtain CPU information".to_string());
    };

    
    
    let total_mem = sys.total_memory();      
    let available_mem = sys.available_memory();  
    let used_mem = sys.used_memory();        

    let memory_info = MemoryInfo {
        
        total_bytes: total_mem,
        available_bytes: available_mem,
        used_bytes: used_mem,

        
        total_gb: bytes_to_gb(total_mem),
        available_gb: bytes_to_gb(available_mem),
        used_gb: bytes_to_gb(used_mem),

        
        usage_percent: if total_mem > 0 {
            (used_mem as f64 / total_mem as f64) * 100.0
        } else {
            0.0
        },
    };

    
    let gpu_info = detect_gpu();

    
    let os_info = OsInfo {
        name: System::name().unwrap_or_else(|| "Unknown".to_string()),
        version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
        arch: std::env::consts::ARCH.to_string(),
    };

    
    Ok(SystemInfo {
        cpu: cpu_info,
        memory: memory_info,
        gpu: gpu_info,
        os: os_info,
    })
}


fn detect_gpu() -> GpuInfo {
    
    
    
    
    
    let has_cuda = std::path::Path::new("/usr/local/cuda").exists()
        || std::env::var("CUDA_PATH").is_ok()
        || check_nvidia_smi();

    
    
    let has_metal = cfg!(target_os = "macos");

    
    
    let has_vulkan = check_vulkan();

    
    
    let description = if has_cuda {
        "NVIDIA CUDA GPU detected, GPU acceleration supported".to_string()
    } else if has_metal {
        "Apple Metal GPU detected, GPU acceleration supported".to_string()
    } else if has_vulkan {
        "Vulkan support detected, GPU acceleration may be supported".to_string()
    } else {
        "No GPU detected or GPU acceleration unavailable, will use CPU inference".to_string()
    };

    GpuInfo {
        has_cuda,
        has_metal,
        has_vulkan,
        description,
    }
}


fn check_nvidia_smi() -> bool {
    std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}


fn check_vulkan() -> bool {
    
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


fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}




    /// recommend_model function
pub fn recommend_model(system_info: &SystemInfo) -> ModelRecommendation {
    let available_gb = system_info.memory.available_gb;
    let total_gb = system_info.memory.total_gb;
    let has_gpu = system_info.gpu.has_cuda || system_info.gpu.has_metal;
    let cpu_cores = system_info.cpu.physical_cores;

    
    let mut model_sizes = vec![
        ModelSize {
            name: "1B".to_string(),
            parameters_billions: 1.0,
            estimated_memory_gb: 1.5,
            recommended: false,
            expected_tokens_per_second: "20-50".to_string(),
            description: "Extra small model, suitable for low-end devices, simple conversations and text generation".to_string(),
        },
        ModelSize {
            name: "3B".to_string(),
            parameters_billions: 3.0,
            estimated_memory_gb: 3.5,
            recommended: false,
            expected_tokens_per_second: "15-40".to_string(),
            description: "Small model, balanced performance and quality, suitable for general conversations and creative writing".to_string(),
        },
        ModelSize {
            name: "7B".to_string(),
            parameters_billions: 7.0,
            estimated_memory_gb: 6.0,
            recommended: false,
            expected_tokens_per_second: "10-30".to_string(),
            description: "Medium model, good quality, suitable for professional writing and complex tasks".to_string(),
        },
        ModelSize {
            name: "13B".to_string(),
            parameters_billions: 13.0,
            estimated_memory_gb: 10.0,
            recommended: false,
            expected_tokens_per_second: "5-15".to_string(),
            description: "Large model, high quality output, requires higher configuration".to_string(),
        },
        ModelSize {
            name: "30B+".to_string(),
            parameters_billions: 30.0,
            estimated_memory_gb: 20.0,
            recommended: false,
            expected_tokens_per_second: "2-8".to_string(),
            description: "Extra large model, top quality, requires high-end configuration or GPU".to_string(),
        },
    ];

    
    
    let memory_budget = (available_gb * 0.7) as f32;

    let mut recommended_sizes = Vec::new();
    for model in &mut model_sizes {
        if model.estimated_memory_gb <= memory_budget {
            model.recommended = true;
            recommended_sizes.push(model.clone());
        }
    }

    
    let (performance_tier, context_size) = if total_gb >= 32.0 && has_gpu {
        ("high", 8192)
    } else if total_gb >= 16.0 {
        ("medium", 4096)
    } else if total_gb >= 8.0 {
        ("low-medium", 2048)
    } else {
        ("low", 1024)
    };

    
    let gpu_layers = if system_info.gpu.has_cuda {
        
        if total_gb >= 16.0 { 32 } else { 16 }
    } else if system_info.gpu.has_metal {
        
        1  
    } else {
        0  
    };

    
    
    let threads = cpu_cores.min(8) as u32;

    
    let mut suggestions = Vec::new();

    if recommended_sizes.is_empty() {
        suggestions.push("⚠️ Warning: Insufficient available memory, cannot recommend any model. Suggest closing other applications to free memory.".to_string());
        
        if let Some(smallest) = model_sizes.first() {
            recommended_sizes.push(smallest.clone());
        }
    } else {
        
        if let Some(best) = recommended_sizes.last() {
            suggestions.push(format!("✅ Recommended to use {} parameter model, estimated to use {:.1}GB memory",
                best.name, best.estimated_memory_gb));
        }
    }

    if has_gpu {
        if system_info.gpu.has_cuda {
            suggestions.push("🚀 NVIDIA GPU detected, strongly recommend enabling GPU acceleration (set gpu_layers)".to_string());
        } else if system_info.gpu.has_metal {
            suggestions.push("🍎 Apple Metal GPU detected, will automatically use GPU acceleration".to_string());
        }
    } else {
        suggestions.push("💡 No GPU detected, will use CPU inference, speed may be slow".to_string());
    }

    if system_info.memory.usage_percent > 80.0 {
        suggestions.push(format!("⚠️ Current memory usage {:.1}%, suggest closing some applications for better performance",
            system_info.memory.usage_percent));
    }

    if cpu_cores < 4 {
        suggestions.push("⚠️ Low CPU core count, inference speed may be limited".to_string());
    }

    if total_gb < 8.0 {
        suggestions.push("⚠️ Total memory less than 8GB, suggest using smaller models and shorter context".to_string());
    }

    if available_gb < 4.0 {
        suggestions.push("⚠️ Available memory less than 4GB, strongly recommend freeing memory before loading model".to_string());
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
