//! 设备检测功能深度验证测试
//!
//! 验证以下核心功能：
//! 1. 检测当前可用环境
//! 2. 设置推理环境
//! 3. 一键选择最优环境

use loci::prelude::*;

#[test]
fn test_capability_1_detect_available_devices() {
    println!("\n=== 测试1: 检测当前可用环境 ===");

    // 创建设备选择器
    let selector = DeviceSelector::new();

    // 验证：至少应该检测到CPU
    assert!(selector.device_count() >= 1, "应该至少检测到CPU设备");

    // 验证：CPU设备应该可用
    assert!(
        selector.has_backend(DeviceType::CPU),
        "CPU后端应该始终可用"
    );

    // 打印所有检测到的设备
    println!("检测到 {} 个设备:", selector.device_count());
    for (i, device) in selector.devices().iter().enumerate() {
        println!(
            "  设备 {}: {} ({}) - {:.2} GB - 可用: {}",
            i,
            device.name,
            device.device_type,
            device.memory_gb(),
            device.available
        );
    }

    // 验证：设备信息完整
    if let Some(device) = selector.device(0) {
        assert!(!device.name.is_empty(), "设备名称不应为空");
        assert!(device.memory_bytes > 0, "设备内存应大于0");
        assert!(device.id >= 0, "设备ID应为非负数");
    }

    println!("✅ 测试1通过：可以成功检测当前可用环境");
}

#[test]
fn test_capability_2_set_inference_environment() {
    println!("\n=== 测试2: 设置推理环境 ===");

    let selector = DeviceSelector::new();

    // 方式1: 手动创建配置
    println!("方式1: 手动创建CPU配置");
    let cpu_config = DeviceConfig::cpu_only();
    assert_eq!(cpu_config.device_type, DeviceType::CPU);
    assert_eq!(cpu_config.n_gpu_layers, 0);
    println!("  ✓ CPU配置: {:?}", cpu_config.device_type);

    // 方式2: 根据设备类型创建配置
    if selector.has_backend(DeviceType::CUDA) {
        println!("方式2: 创建CUDA配置");
        let cuda_config = DeviceConfig::gpu_all(DeviceType::CUDA, 0);
        assert_eq!(cuda_config.device_type, DeviceType::CUDA);
        assert_eq!(cuda_config.n_gpu_layers, -1);
        println!("  ✓ CUDA配置: 所有层GPU加速");
    }

    // 方式3: 指定GPU层数
    println!("方式3: 创建部分GPU offload配置");
    let partial_config = DeviceConfig::gpu_layers(DeviceType::CPU, 0, 20);
    assert_eq!(partial_config.n_gpu_layers, 20);
    println!("  ✓ 部分GPU配置: {} 层", partial_config.n_gpu_layers);

    // 方式4: 根据模型大小推荐
    println!("方式4: 根据模型大小自动推荐");
    let recommended = selector.recommend_for_model(7.0); // 7B模型
    println!(
        "  ✓ 7B模型推荐: {} (Device {}, {} GPU layers)",
        recommended.device_type, recommended.device_id, recommended.n_gpu_layers
    );

    println!("✅ 测试2通过：可以灵活设置推理环境");
}

#[test]
fn test_capability_3_auto_select_optimal() {
    println!("\n=== 测试3: 一键选择最优环境 ===");

    let selector = DeviceSelector::new();

    // 一键自动选择
    let auto_config = selector.auto_select();

    println!("自动选择结果:");
    println!("  设备类型: {}", auto_config.device_type);
    println!("  设备ID: {}", auto_config.device_id);
    println!("  GPU层数: {}", auto_config.n_gpu_layers);

    // 验证：选择的设备应该存在
    let selected_device = selector.device(auto_config.device_id as usize);
    assert!(selected_device.is_some(), "选中的设备应该存在");

    if let Some(device) = selected_device {
        assert!(device.available, "选中的设备应该可用");
        println!("  设备名称: {}", device.name);
        println!("  设备内存: {:.2} GB", device.memory_gb());
    }

    // 验证：选择逻辑正确
    // 优先级: CUDA > Metal > Vulkan > ROCm > OpenCL > CPU
    if selector.has_backend(DeviceType::CUDA) {
        assert_eq!(
            auto_config.device_type,
            DeviceType::CUDA,
            "有CUDA时应该优先选择CUDA"
        );
    } else if selector.has_backend(DeviceType::Metal) {
        assert_eq!(
            auto_config.device_type,
            DeviceType::Metal,
            "无CUDA但有Metal时应该选择Metal"
        );
    } else {
        assert_eq!(
            auto_config.device_type,
            DeviceType::CPU,
            "无GPU时应该降级到CPU"
        );
    }

    println!("✅ 测试3通过：可以一键选择最优环境");
}

#[test]
fn test_backend_availability_check() {
    println!("\n=== 额外测试: 后端可用性检查 ===");

    let selector = DeviceSelector::new();

    println!("后端可用性:");
    println!("  CUDA:   {}", selector.has_backend(DeviceType::CUDA));
    println!("  Metal:  {}", selector.has_backend(DeviceType::Metal));
    println!("  Vulkan: {}", selector.has_backend(DeviceType::Vulkan));
    println!("  ROCm:   {}", selector.has_backend(DeviceType::ROCm));
    println!("  OpenCL: {}", selector.has_backend(DeviceType::OpenCL));
    println!("  CPU:    {}", selector.has_backend(DeviceType::CPU));

    // CPU应该始终可用
    assert!(selector.has_backend(DeviceType::CPU));

    println!("✅ 后端可用性检查通过");
}

#[test]
fn test_model_size_recommendation() {
    println!("\n=== 额外测试: 模型大小匹配 ===");

    let selector = DeviceSelector::new();

    let test_cases = vec![
        ("Qwen-0.5B", 0.6),
        ("Llama-7B", 7.0),
        ("Llama-13B", 13.0),
        ("Llama-70B", 70.0),
    ];

    println!("模型大小推荐:");
    for (name, size) in test_cases {
        let config = selector.recommend_for_model(size);
        println!(
            "  {} ({:.1}GB) -> {} (Device {}, {} layers)",
            name, size, config.device_type, config.device_id, config.n_gpu_layers
        );

        // 验证：推荐的设备应该存在
        assert!(selector.device(config.device_id as usize).is_some());
    }

    println!("✅ 模型大小推荐测试通过");
}
