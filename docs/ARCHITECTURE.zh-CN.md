# Loci 架构说明

## 定位

Loci 是 `PetCompanion` 之下的推理基础设施层，不承担产品层 UI 或桌宠逻辑。

## 三层设计

### 1. 核心层

`crates/core` 负责：

- 后端注册表
- 模型加载配置
- 推理参数归并
- 插件清单加载与激活
- 运行时快照

### 2. 插件层

当前稳定主线只识别两类插件：

- `model_loader`
- `hardware_backend`

当前激活流程也只做狭义的运行时物化：

- `native` 运行时按动态库加载
- `wasm` 运行时只做模块校验并保留为运行时工件

这保证了宿主契约是实用且可实现的，而不是提前承诺稳定的插件符号 ABI。`kv_cache`、`distributed`、`multimodal`、`agent` 仍然只是路线图方向，暂时不进入当前稳定宿主契约。当前主实现仍以 `llama.cpp` 为主要后端，插件机制负责扩展格式和硬件能力。

### 3. 接口层

Loci 对外提供：

- Rust API
- C ABI
- 本地 HTTP sidecar 接口

## 运行流程

1. 扫描插件目录中的 `manifest.toml`
2. 注册或激活插件
3. 通过后端加载模型
4. 合并默认推理参数与请求参数
5. 执行推理并返回运行时信息

本地 HTTP sidecar 也保持最小化，只暴露运行时/模型控制以及单活动模型下的 OpenAI 兼容 `models`、`completions`、`chat/completions` 推理接口。

## 边界约束

- 不在 `loci-core` 中引入桌面窗口或 UI host 逻辑
- 不把 `PetCompanion` 的交互流程抽象塞进插件 API
- 硬件选择与 offload 策略归属于后端或硬件插件，而不是应用层
- 不在 server 层引入 tools、assistants、workflow engine 或 agent loop
