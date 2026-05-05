**Loci 项目设计文档（v1.5）—— 纯端侧异构分层推理基础设施（含可选动态模型路由）**

**License**：MIT（鼓励社区贡献新 backend feature 或路由策略）  

**项目名称**：Loci  
**全称**：Loci — Lightweight End-side Heterogeneous & Tiered LLM Inference Infrastructure in Rust  

**定位**：Rust 生态中**专注端侧异构算力 + 分层 offload 的纯底层推理基础设施**。  
Loci 提供稳定、高性能的 Inference API，支持 CPU + GPU + NPU 同时使用，实现**精细化参数放置**（部分层/子图在 NPU、部分在 CPU/GPU）和**磁盘分层 offload**（内存不足时部分权重 spill 到磁盘，懒加载 + prefetch）。  
**新增**：动态模型路由作为**可选 Cargo feature**，允许根据 prompt 复杂度、功耗预算、当前负载在多个模型间智能路由，但**绝不污染核心单模型异构路径**。

**核心目标**：在端侧设备上实现低功耗、高灵活性的 LLM 推理，成为 Rust 端侧 Infra 的标杆项目，同时为多 Agent 场景提供平滑扩展能力。

### 1. 设计原则（严格遵守）
- **极简核心**：loci-core 只负责 Pipeline + Advanced Hetero Offload Planner，动态路由必须是可选 feature。
- **无动态插件**：全部通过 Cargo features 静态注入 backend（性能零开销）。
- **端侧优先**：NPU-first + 功率/热量感知 + 精细化混合 + 磁盘 tier + 可选动态路由。
- **硬件无关抽象**：支持 CPU/GPU/NPU 同时运行，未来芯片通过新增 feature 扩展。
- **性能与可维护性优先**：动态路由仅作为增强特性，不影响单模型异构主路径。
- **纯 Rust 主路径 + 必要 FFI**：执行 backend 可以使用必要 FFI，但核心资产层、分层加载层、调度层必须保持 Rust 主导与 backend-agnostic。
- **模型资产抽象优先于 backend 格式**：Loci 不应把 `OpenVINO IR / GGUF / SafeTensors / ONNX` 视为架构边界，核心应围绕统一的 `asset inventory / shard / residency / spill` 抽象组织。

**参考开源项目整合说明**（2026 年 4 月最新状态）：
- **OpenVINO 2026.1 Heterogeneous Execution + GenAI**：子图自动拆分 + affinities 设置（精细化 NPU/GPU/CPU 放置）。
- **FlexLLMGen / FlexGen**（con3code/FlexGen fork）：磁盘 tier 权重管理、mmap + 异步 prefetch。
- **llm.npu（ASPLOS ’25，UbiquitousLearning/mllm）**：out-of-order subgraph 执行 + block-level 调度算法。
- **Fox（ferrumox/fox）**：PagedAttention + prefix caching + multi-model lazy loading + 硬件自动检测（动态路由重要参考）。
- **Candle（Hugging Face 0.10+）**：Device 抽象 + per-tensor 放置 + GGUF 支持。
- **Burn（tracel-ai/burn）**：Rust-native tensor / backend / kernel 生态，适合作为纯 Rust backend、kernel/runtime 组织与未来训练-推理一体化能力的重要参考。
- **TAPAS / PowerInfer**：功率/热量感知调度思路。
- **vLLM**：生产级动态路由思路（model pool + 策略），但 Loci 只取端侧轻量版。

这些项目被精准融入，既保留 Loci“端侧异构分层”特定，又通过可选 feature 控制复杂度。

### 1.1 为什么必须自己实现统一层（2026-04-27 新增结论）

当前生态里没有任何一个现成 Rust 项目能同时满足 Loci 的完整目标：

- OpenVINO NPU 最优路径（Heterogeneous Execution、GenAI pipeline、NPU-first、子图拆分）
- Candle 纯 Rust fallback（CPU / CUDA / Metal / WASM）
- 两者之上的统一异构调度（部分层 NPU、部分层 CPU / GPU、磁盘 spill、动态 re-offload）
- 端侧 Infra 定位（轻量 Pipeline + Advanced Hetero Offload Planner + tiered-offload）

因此 Loci 的核心价值不在于重新实现 backend 底层，而在于补齐当前生态缺失的“统一异构调度层”。

#### 生态判断

- `openvino-rs` 与 `openvino-genai` 已足够成熟，适合直接作为 OpenVINO 主路径的官方 Rust 封装。
- Candle 已是最强纯 Rust fallback 路径，但并不覆盖 OpenVINO NPU。
- Burn 提供了更系统的 Rust-native tensor/backend 设计范式，但当前并不直接覆盖 Intel NPU 最优执行路径。
- 现有 Rust 引擎大多只覆盖单一后端，缺少跨 backend 的统一异构调度与磁盘分层能力。

#### Loci 需要自己实现的部分

你不需要：

- 从零编写 OpenVINO FFI
- 从零实现 Candle
- 从零重写 tensor kernel

你需要自己实现：

- `loci-backend-openvino`：对 `openvino-rs` / `openvino-genai` 的薄封装
- `loci-backend-candle`：对 Candle `Device` / 模型执行路径的薄封装
- 可进一步评估 `loci-backend-burn`：对 Burn backend / tensor runtime / kernel 生态的薄封装
- `Advanced Hetero Offload Planner`：统一决定 NPU / GPU / CPU / Disk 的放置策略
- `tiered-offload`：围绕统一 asset inventory 的权重 / shard / KV mmap、prefetch、spill 策略

#### 工程判断

这不是重复造轮子，而是在成熟官方 crate 之上实现端侧 Rust Infra 缺失的统一调度层。

因此，Loci v1.5 的正确实现原则是：

- OpenVINO 作为主路径，追求 NPU 最优性能
- Candle 作为纯 Rust fallback
- Burn 作为 Rust-native backend / kernel 设计参考与潜在执行后端补充
- 自己实现的重点始终是统一调度与分层 offload，而不是重写 backend 底层

### 1.2 多 backend / 多芯片架构原则（2026-04-30 新增）

既然 Loci 的目标不是单一 Intel 设备适配，而是**端侧异构 + 未来多芯片扩展**，那么必须明确下面这条总原则：

> **Loci 在编排层必须 backend-agnostic，在执行层必须 backend-specialized。**

也就是说：

- `loci-core` 不能被 OpenVINO 绑定
- `loci-core` 负责统一：
  - device discovery 抽象
  - placement plan
  - model asset inventory
  - tiered offload
  - paged KV
  - power / thermal policy
  - model routing
- 每个芯片家族都应有各自最合适的 backend 执行层

#### 这意味着什么

- **OpenVINO 不是 Loci 本体**
  - 它只是 Intel 路线当前最成熟、最务实的 backend
  - 在 Intel `CPU/GPU/NPU` 上，Loci 当前需要依赖它获得真实执行能力

- **Loci 不能被设计成“只要去掉 OpenVINO 也能保留全部能力”**
  - 对 Intel NPU 路线来说这是不现实的
  - 因为 Loci 自己不会重写厂商 runtime / kernel / driver 接口

- **Loci 也不能被设计成“永远只服务 OpenVINO”**
  - 否则未来无法自然扩展到 Qualcomm / Rockchip / 其他端侧芯片

#### 正确的分层方式

1. **统一编排层**
   - `loci-core`
   - 不绑定单一厂商
   - 只依赖稳定 trait / protocol / execution plan

2. **厂商 / runtime 专用执行层**
   - `loci-backend-openvino`
     - Intel `CPU/GPU/NPU`
   - `loci-backend-candle`
     - 纯 Rust fallback / 通用 CPU-GPU 路线
   - 可进一步考虑：
     - `loci-backend-burn`
       - 纯 Rust tensor / backend 路线
       - 可作为 Candle 之外的 Rust-native backend 补充
       - 更适合承接未来 backend-local artifact generation、kernel dispatch 与多后端收敛实验
   - 未来可新增：
     - `loci-backend-qnn`
     - `loci-backend-rknn`
     - `loci-backend-onnxruntime`
     - `loci-backend-tract`
     - 其他芯片专用 backend

3. **Planner 只依赖 capability，不依赖厂商名**
   - Planner 不应该写死 “只有 OpenVINO 才能做异构”
   - Planner 应只关注：
     - backend 支持哪些 accelerator
     - backend 是否支持 multimodal
     - backend 是否支持 disk tier
     - backend 是否支持 paged KV
     - backend 是否支持真实 hetero execution

#### 对当前实现的直接要求

- backend descriptor 需要逐步具备更明确的 capability 描述
- backend 选择逻辑要考虑 multimodal / accelerator / capability，而不是只按 model format
- protocol / planner / snapshot 层要继续避免写死单一 backend 家族语义

#### 阶段性判断

当前阶段最合理的路线仍然是：

- **OpenVINO = Intel 主路径**
- **Candle = 纯 Rust fallback**
- **Loci = 跨 backend 的统一异构基础设施**

这条路线既保留当前可落地性，也不会把未来 Qualcomm / Rockchip / 其他芯片接入路径堵死。

### 2. 核心架构（三层设计）

1. **核心层 (loci-core)**  
   - Pipeline：Model Load → Tiered Hetero Context → Sampling → Output。  
   - **统一模型资产层（本轮方向修正）**：
     - `asset layout`
     - `asset inventory`
     - `shard descriptor`
     - `residency / spill / prefetch` 的统一抽象
     - 这层必须尽量不绑定单一 backend 格式
   - **Advanced Hetero Offload Planner**（Loci 特定创新，参考 llm.npu + OpenVINO Hetero）：  
     - 启动时自动探测 CPU/GPU/NPU + 内存/功耗/电池/磁盘 IO。  
     - 运行时精细化决策：部分层/子图分配到 NPU（低功耗 decode）、部分到 GPU/CPU（高吞吐 prefill）、内存不足时部分权重 spill 到磁盘。  
     - 支持动态 re-offload（推理中根据实时算力调整）。  
   - **ModelRouter**（新增，可选 feature = "dynamic-routing"）：  
     - 轻量 ModelPool + 路由决策（power-aware / latency-aware / prompt-complexity）。  
     - 路由后仍调用 Advanced Hetero Offload Planner（无缝融合）。  
     - 支持 prefix caching 跨模型共享（参考 Fox）。  
   - KV Cache 抽象 + PagedAttention-like 管理（参考 Fox）。  
   - 异步 Inference API（streaming + structured output + tool calling 友好）。  
   - 配置、tracing、错误处理。

2. **Backend 层（Cargo features 静态注入）**  
   - **主路径**（默认 `feature = "openvino"`）：loci-backend-openvino  
     - openvino-rs + OpenVINO 2026.1 GenAI + Heterogeneous Execution（子图拆分 + affinities）。  
   - **可选 fallback**（`feature = "candle"`）：loci-backend-candle  
     - Candle 0.10+（CPU + CUDA + Metal + WASM），per-tensor Device 放置。  
   - **Rust-native backend 参考 / 候选**：Burn  
     - Burn 更适合作为 Rust tensor/runtime/backend 设计参考，帮助 Loci 在不依赖单一执行格式的前提下推进 backend-local artifact preparation、kernel dispatch 与潜在纯 Rust 执行路径。  
   - **未来芯片**：`feature = "hexagon"` / `"rockchip"` / `"tpu"`（Planner 统一调度）。

3. **扩展层（可选 features）**  
   - `tiered-offload`：磁盘分层管理（FlexLLMGen 思路）。  
   - `paged-kv`：PagedAttention + prefix caching（Fox）。  
   - `power-aware`：实时功率/热量感知（TAPAS + PowerInfer）。  
   - `dynamic-routing`：ModelRouter + 路由策略（Fox + vLLM 轻量版）。  
   - `server`：OpenAI 兼容 sidecar。

### 3. 项目结构（Cargo Workspace）

```bash
loci/
├── loci-core/                  # Pipeline + Advanced Hetero Offload Planner + ModelRouter（可选）
├── loci-backend-openvino/      # 主路径（NPU/GPU/CPU 精细化 Hetero）
├── loci-backend-candle/        # fallback
├── loci-tiered-offload/        # 磁盘分层 + 懒加载
├── loci-paged-kv/              # Paged KV + prefix caching
├── loci-cli/                   # 测试 + benchmark
└── loci-protocol/              # 稳定 InferenceRequest/Response 定义
```

Cargo.toml features 示例：
```toml
[features]
default = ["openvino", "power-aware", "tiered-offload", "paged-kv"]
openvino = ["dep:loci-backend-openvino"]
candle = ["dep:loci-backend-candle"]
tiered-offload = ["loci-core/tiered-offload"]
power-aware = ["loci-core/power-aware"]
paged-kv = ["loci-core/paged-kv"]
dynamic-routing = ["loci-core/dynamic-routing"]   # 新增，可选
```

### 4. 关键特性（Loci 特定 + 竞品融合）

- **CPU + GPU + NPU 全支持 + 精细化参数放置**  
  - OpenVINO Hetero 子图拆分 + llm.npu block-level 调度。  

- **磁盘分层 offload + 懒加载**  
  - FlexLLMGen 风格的 mmap + 异步 prefetch + 动态 spill。  
  - 实现边界应建立在 shard / tensor 级别，而不是建立在某一种 backend 目录布局假设上。  

- **功率/热量感知调度**  
  - TAPAS + PowerInfer 思路，NPU-first + 自动降频。  

- **动态模型路由（可选）**  
  - 根据 prompt 复杂度、功耗、负载智能路由到不同模型。  
  - 路由后仍走 Advanced Hetero Offload Planner + 磁盘 tier。  
  - 参考 Fox multi-model lazy loading + vLLM 路由策略（端侧轻量实现）。  

- **融入优质特性**  
  - Fox：PagedAttention + prefix caching + 硬件自动检测。  
  - Candle：干净 Device 抽象 + GGUF 优化。  

- **端侧优化**：动态模型切换（不重启）、异构 KV 共享、低延迟 streaming。

### 5. 开发路线图（务实聚焦）

**Phase 1（4 周 MVP）**：  
- loci-core + loci-backend-openvino + Advanced Hetero Offload Planner + 基础磁盘 tier（**不包含 dynamic-routing**）。  
- **必须交付**：Intel Core Ultra 机器上真实 benchmark（部分层 NPU + 部分 CPU + 磁盘 spill 场景）。  

**Phase 2（4 周）**：  
- loci-backend-candle + 完整磁盘 prefetch + loci-paged-kv + power-aware + **可选 dynamic-routing**（ModelRouter 最小实现）。  

**Phase 3**：Hexagon/Rockchip/TPU 等新芯片 backend + 高级路由策略（社区 PR）。

### 5.1 当前代码审计结果（2026-04-28，基于本仓库当前实现）

以下内容不是愿景，而是**对当前代码状态的审计结论**。

#### 已完成的部分

- **Workspace 重构已经完成**  
  当前仓库已经按 Loci v1.5 主体结构拆分为：
  - `protocol`
  - `core`
  - `backend-openvino`
  - `backend-candle`
  - `tiered-offload`
  - `paged-kv`
  - `cli`
  - `server`

- **统一异构控制面已基本成型**  
  `loci-core` 里已经具备：
  - 统一 `ExecutionPlan`
  - backend 选择
  - CPU/GPU/NPU/Disk 的放置决策
  - 模型注册、alias、resident/prepared 状态管理
  - keep-alive eviction
  - resident memory budget enforcement

- **功耗/热量感知的 Planner 已有第一版真实逻辑**  
  当前代码已实现：
  - prefill 偏 GPU、decode 偏 NPU 的基础异构策略
  - thermal / battery / power budget 驱动的 KV / weights 回退
  - `dynamic_reoffload` 标记
  - `weights` stage 明确化，不再允许 plan 中权重放置缺失

- **paged-kv / tiered-offload 的“策略层”已经存在**  
  当前代码已实现：
  - KV block/page 规划
  - prefix cache 是否共享的策略
  - spill bytes / prefetch window / offload profile 推导
  - `auto / gpu_resident / balanced / disk_heavy` profile

- **动态路由已经从“只在内部存在”提升到“可端到端配置”**  
  当前代码已新增：
  - runtime 内部 routing config setter
  - `POST /v1/config/routing`
  - CLI 动态路由参数透传
  - `cli/server` 对 `dynamic-routing` feature 的转发

- **服务层已可用**  
  当前 server 已支持：
  - runtime/config/models/plan/inference
  - alias 注册与删除
  - planner config 更新
  - routing config 更新
  - OpenAI 风格 `models / completions / chat/completions`
  - `Content-Length`
  - chunked body
  - `Expect: 100-continue`

#### 当前实现已经进入“半真实 runtime”阶段，但仍未完全达到设计目标

- **OpenVINO backend 已经接入真实 `openvino-rs` / `openvino-genai`**
  - 已有：
    - 真实 OpenVINO runtime topology 发现
    - 真实 `LlmPipeline` / `VlmPipeline` 接入
    - 基于模型 architecture 的 text / multimodal 分流
    - `path` / `file://` / `data:` / base64 图像输入解码
    - 真实 runtime 不可用时的 fallback session 缓存与错误回传
  - 仍缺：
    - planner 到 OpenVINO 子图 affinities 的真正细粒度映射
    - 基于层/子图的真实 NPU/GPU/CPU 精细放置执行
    - 对未导出模型目录的自动转换能力
    - 更细的真实 perf telemetry 与 benchmark 管线

- **Candle backend 仍是“控制面已接通、执行面未落地”的部分实现**
  - 已有：
    - format compatibility
    - NPU rejection
    - residency 估算
    - 明确的 multimodal reject
    - 已纳入统一 model readiness / asset layout 诊断
  - 未有：
    - 真实 Candle tensor/device 执行路径
    - 真实 GGUF / SafeTensors 加载执行
    - 真实 CPU/GPU tensor placement runtime

- **tiered-offload 已不再只是纯 policy**
  - 已有：
    - spill / policy / profile / prefetch window 推导
    - mmap-backed spill artifact
    - 后台 prefetch runtime
    - 与 `loci-core` prepare / eviction 的接线
    - 已开始接入统一 `asset inventory / shard` 视图，而不是只依赖单一源文件猜测
  - 仍缺：
    - 更完整的动态 reload / spill state machine
    - 更强的 IO-aware 调度与 runtime 观测
    - 更细的 tensor-level pager

- **paged-kv 仍主要停留在 planner 层**
  - 已有：page/block/cache size 策略
  - 未有：真实 paged attention
  - 未有：真实 KV page allocation / eviction
  - 未有：跨模型 prefix cache 真正共享的数据层

#### 当前与设计文档仍不一致的关键缺口

- **“异步 Inference API”只完成了传输层，不等于完成了完整语义层**
  - 当前已经有 SSE / chunk streaming：
    - `/v1/inference/stream`
    - OpenAI 风格 `completions` / `chat.completions` streaming
  - 但 `structured_output` / `tool_calling` 目前仍主要停留在 request / routing / protocol 层
  - 尚未驱动真实 grammar-constrained generation、tool execution orchestration、schema-aware decoding

- **“动态 re-offload（推理中实时调整）”尚未真正落地**
  - 当前只有 planner/profile 上的信号与偏置
  - 还没有推理进行中的实时迁移执行层

- **“prefix caching 跨模型共享”尚未真正落地**
  - 当前只有 `shared_across_models` 策略字段
  - 没有真实跨模型 cache store / reuse 机制

- **“真实 benchmark on Intel Core Ultra”尚未完成**
  - 当前测试已经不再只是 unit/integration，已有真实 OpenVINO runtime 实机探测与模型加载尝试
  - 但当前这台机器实测只有 `CPU + GPU`，没有可用 NPU
  - 因此还没有真实 `NPU/GPU/CPU` 协同 benchmark 数据

- **“任意模型可支持”目前仍然依赖模型资产工作流，而不是只靠 runtime trait**
  - 当前已经补入统一 `model readiness` 诊断层
  - Loci 现在可以明确区分：
    - 已可直接执行的导出模型
    - 需要 OpenVINO 导出/IR 转换的原始模型
    - 当前 backend 尚未实现真实执行的模型
  - 这一步非常关键，因为它把“格式兼容”与“真实可运行”正式分开了

- **“统一模型资产层”之前还不够明确，现在必须提升为正式架构层**
  - 正确方向不是把所有模型都先转换成某个 backend 专用格式再组织内存层
  - 正确方向应当是：
    - 先抽象 `asset inventory`
    - 再抽象 `shard`
    - 再抽象 `residency / spill / prefetch`
    - 最后由 backend 消费这些资产并负责执行 lowering
  - 这也是后续摆脱单一 OpenVINO 依赖的关键前提

- **“低层芯片算子 / 子图 ABI”目前仍未落地**
  - 当前 planner 能输出 `CPU/GPU/NPU/Disk` placement
  - 但还没有：
    - 面向芯片 backend 的 layer / subgraph affinity ABI
    - 自定义低层 kernel/operator registry
    - backend-specific operator lowering / partition callback
  - 这意味着当前 Loci 还是“统一编排层 + 部分真实 backend 接线”，还不是“可插拔低层算子基础设施”

- **但这一缺口已经开始进入“结构化落地阶段”**
  - 本轮已补：
    - `ExecutionPlan.lowering_plan`
    - `BackendLoweringPlan`
    - `LoweringSubgraphPlan`
    - `LoweringAffinityMode`
  - 当前 planner 已能把 pipeline-stage placement 进一步展开为 backend-facing 的 subgraph guidance：
    - `embedding_lookup`
    - `prefill_attention_block`
    - `prefill_mlp_block`
    - `decode_attention_block`
    - `decode_mlp_block`
    - `kv_state_region`
    - `weights_residency_region`
    - `sampling_head`
    - multimodal 下额外生成 `vision_encoder`
  - 这一步的意义不是“已经实现真实 layer affinity 执行”，而是：
    - **Loci 终于有了稳定的 lowering 协议骨架**
    - 后续 OpenVINO / QNN / RKNN / 其他 backend 都可以围绕同一份 lowering plan 接执行层
  - 当前状态应准确描述为：
    - **subgraph planning ABI 已实现**
    - **real backend lowering 仍未完全实现**

- **OpenVINO lowering 已进入“半真实执行消费”阶段**
  - 本轮新增：
    - backend prepare 阶段会尝试基于 `lowering_plan` 进行一次真实的 OpenVINO IR shadow compile
    - 会从 lowering subgraph 中提取 device priority 顺序
    - 会把该顺序传入 `HETERO` compile intent / runtime properties
  - 这意味着：
    - lowering plan 不再只是 JSON / 调试信息
    - OpenVINO backend 已经开始真实消费 planner 产出的 lowering 指导
  - 但当前仍然没有做到：
    - `query_model`
    - `rt_info["affinity"]` 的逐节点写回
    - 基于真实 op graph 的 per-layer affinity 修正
  - 因此当前最准确的说法应是：
    - **OpenVINO lowering 已实现真实 priorities 级消费**
    - **逐节点 affinity lowering 仍待补完**

#### 当前最务实的后续优先级（按实现顺序）

1. **继续补强统一模型资产层**
   - 现在 backend 已经是真实 OpenVINO 路径
   - 当前最大阻塞不再只是“没接 OpenVINO”，而是“资产层仍未完全独立于 backend 格式”
   - 已补：
     - asset layout 检测
     - asset inventory / shard 级别清单
     - backend readiness 报告
     - 模型是否需要导出/转换的明确诊断
   - 下一步仍需补：
     - 更稳定的 shard 分类
     - tensor-level pager
     - residency target 抽象
     - 文档化导出流程
     - 自动校验更多必需文件

2. **把 planner 输出进一步映射到真实 hetero 执行**
   - 当前已经能生成异构 plan
   - 但 OpenVINO backend 还没有把 plan 精确映射为真实子图 / affinities / per-stage placement

3. **为未来多芯片 backend 设计低层能力边界**
   - 需要新增或明确：
     - subgraph placement ABI
     - backend capability matrix
     - layer/operator 级别的下沉接口
   - 这是后续接入 `QNN / RKNN / 其他芯片 backend` 的前置条件

4. **继续增强 `tiered-offload` 执行层**
   - 现有 mmap / prefetch 已有基础
   - 且已经开始从“模型级”向“shard 级”迁移
   - 下一步是补动态 reload / spill state machine、IO-aware 行为与 tensor-level pager

5. **把 `paged-kv` 从 planning helper 变成真实缓存层**
   - page allocator
   - prefix cache index
   - eviction policy

6. **最后再做 Candle 真实 fallback**
   - 因为 OpenVINO 主路径仍然是 Loci 当前差异化核心

#### 审计结论

当前 Loci **已经不是旧的错误实现**，并且已经从“只有控制面”推进到“控制面 + 部分真实 runtime + 模型资产诊断”的阶段。  
它现在具备：

- 真实 OpenVINO runtime 接线
- 真实 multimodal request plumbing
- 真实 CPU/GPU topology 探测
- 真实 tiered-offload 基础 runtime
- 真实 model readiness / asset layout 诊断
- 初步成型的统一 `asset inventory / shard` 视图

但它**仍不能被视为完全可用的端侧异构推理 runtime**，因为当前最关键的剩余问题是：

- 还缺少“原始模型目录 → OpenVINO GenAI 导出目录”的稳定工作流
- 统一资产层还没有下沉到 tensor-level pager
- planner 还没有真正驱动细粒度子图异构执行
- 还没有面向未来多芯片 backend 的低层子图 / 算子 ABI
- Candle 仍未落地为真实 fallback runtime
- paged KV 仍未落地为真实缓存执行层
- structured output / tool calling 仍未进入真实生成语义层

也就是说：  
**当前最强的是 Rust 控制面 + OpenVINO 主路径接线；当前最弱的是导出工作流、细粒度异构执行和完整 memory runtime。**

### 5.2 2026-04-30 实机测试补充结论（本轮新增）

以下内容基于本机真实测试，不是静态代码推断。

#### 实机 OpenVINO runtime 状态

- 已在本地解压并接入 `OpenVINO GenAI 2026.1` runtime
- 真实可见设备：
  - `CPU`
  - `GPU`
- 本机**没有可用 NPU**
- 因此本轮异构测试的真实落点是：
  - `CPU + GPU + Disk`
  - 而不是 `NPU-first`

#### 已验证通过的真实能力

- **真实 OpenVINO runtime 发现**
  - Loci 已能在正确环境变量下发现真实 `CPU/GPU` 设备，而不是 synthetic topology

- **真实异构 Planner 输出**
  - 对 multimodal 请求，当前机器上生成的真实 plan 为：
    - prefill → `gpu:0`
    - decode → `gpu:0`
    - kv → `gpu:0`
    - sampling → `CPU`
  - backend profile 为：
    - `execution_mode = hetero`
    - `hetero_devices = ["CPU", "GPU"]`

- **真实 MiniCPM-V-4_5 目录测试**
  - 已使用 `MiniCPM-V-4_5` 仓库目录进行真实请求测试
  - Loci 已能正确：
    - 注册模型目录
    - 生成真实异构 plan
    - 尝试进入真实 OpenVINO VLM 路径
    - 在失败时返回明确 fallback reason

#### 本轮发现并已修复的问题

- **目录模型格式识别 bug 已修复**
  - 之前已有文件名的目录路径会被错误识别为 `unknown`
  - 现已改为：只要路径真实存在且是目录，即识别为 `ModelFormat::Directory`

- **OpenVINO 模型目录校验已增强**
  - 现在会在 backend 初始化阶段直接检查必需 OpenVINO GenAI 文件
  - 对 raw Transformers checkpoint 会明确报错，而不再只给出笼统的 `unknown exception`

- **模型 readiness / asset layout 诊断已补入**
  - 现在 Loci 能直接报告：
    - `asset_layout`
    - `ready_for_inference`
    - `recommended_backend`
    - 每个 backend 是否真实 ready
    - 是否需要导出/转换
  - 这让“任意模型支持”第一次具备了可观察、可调试的基础

- **MiniCPM-V 架构识别已修复**
  - 之前 `minicpm-v` 没有被归类为 multimodal architecture
  - 现已纳入统一 multimodal 检测逻辑
  - 否则会误伤 backend 选择与 readiness 诊断

#### 当前 MiniCPM-V-4_5 的真实阻塞

- 本次测试使用的 `MiniCPM-V-4_5` 仓库目录是**原始 Transformers checkpoint**
- 它不包含 OpenVINO GenAI VLM 所需的导出文件：
  - `openvino_language_model.xml`
- 因此当前真实报错为：
  - 模型目录是 raw Transformers checkpoint
  - 期望的是 OpenVINO GenAI 导出目录

#### 这意味着什么

- **Loci 当前并不是“跑不起来 OpenVINO”**
  - OpenVINO runtime 已经真实接通
  - 异构 planner 也已经真实工作

- **当前不能直接跑的是“未导出的 MiniCPM-V-4_5 原始仓库”**
  - 问题在模型资产格式
  - 不在 Loci 的 request / planner / backend 基础接线

#### 当前阶段的准确判断

- 对 `MiniCPM-V-4_5`：
  - 已完成真实接线验证
  - 未完成真实生成验证
  - 根因是**缺少 OpenVINO 导出模型目录**

- 对 Loci 主路径：
  - 控制面：可用
  - OpenVINO runtime 接线：可用
  - multimodal plumbing：可用
  - 真实 VLM 推理结果：仍依赖正确的 OpenVINO 导出模型资产

#### 本轮测试后的最直接后续任务

1. 为 MiniCPM 系列补齐 `Transformers → OpenVINO GenAI` 导出流程
2. 让 Loci 文档明确区分：
   - raw Transformers 模型目录
   - OpenVINO GenAI 导出目录
3. 在拿到真实导出目录后，继续补做：
   - 真实 multimodal 生成验证
   - 真实 CPU/GPU hetero benchmark

### 5.3 基于 `tmp` 参考项目与论文的进一步审查结论（2026-04-30 本轮新增）

这一轮不是泛泛“看了参考资料”，而是把 `tmp/references` 里的项目和论文重新对照到当前代码上，得到以下更具体的工程结论。

#### 从参考项目得到的直接结论

- **OpenVINO**
  - Loci 当前对它的利用仍偏 runtime pipeline 层
  - 还没有像其异构执行体系那样，把 planner 输出进一步下沉到更细粒度的 subgraph affinity / partition 行为

- **Candle**
  - Loci 当前只借用了“纯 Rust fallback”的边界定义
  - 还没有真正进入 tensor/device runtime 层

- **FlexLLMGen / FlexGen**
  - Loci 已经吸收了 mmap / spill / prefetch 的核心方向
  - 但还没有做到更成熟的 IO-aware 调度与更完整的 reload state machine

- **Fox / llm-d / CAKE**
  - Loci 已经有 paged-kv 的 planning 形态
  - 但还没有真实 prefix cache store、page allocator、跨模型复用的数据层

- **llm.npu / HeteroLLM / PowerInfer-2**
  - 当前 Loci 还没有真正的 block-level / layer-level / subgraph-level 低层执行映射
  - 也还没有面向芯片 backend 的 operator / partition ABI

- **zenlm/engine**
  - 它提醒了一个现实问题：高质量推理基础设施必须把“模型是否可执行”作为一等能力，而不是等到 runtime 报错
  - Loci 本轮新增的 readiness 诊断正是在补这个基础设施缺口

#### 由此得到的结论

Loci 下一阶段最重要的不是再扩很多概念，而是继续把下面三条补实：

1. **模型资产工作流**
   - 导出、校验、诊断、后续自动转换 helper

2. **真实 hetero 执行映射**
   - planner → subgraph / layer / affinity

3. **低层 backend ABI**
   - 为未来 `QNN / RKNN / 其他芯片 backend` 提前定义好 operator / partition / capability 边界

#### 当前关于“低层芯片算子”的准确判断

如果严格按工程标准说，**当前 Loci 还不具备真正的低层芯片算子支持**。  
现在拥有的是：

- 统一异构 placement plan
- backend capability 描述
- OpenVINO 真实 runtime 路径

现在还没有的是：

- 自定义 chip operator registry
- backend-specific kernel ABI
- layer/subgraph 级 dispatch callback
- 跨 backend 通用的 operator lowering 接口

因此当前最准确的说法应当是：

> **Loci 已经具备端侧异构推理 Infra 的统一编排骨架，但低层芯片算子层仍未实现。**

### 5.4 宿主机算力检测实现状态（2026-04-30 本轮新增）

这一项现在已经不是“计划中能力”，而是**Loci 当前代码里已经真实实现并接入 runtime snapshot 的能力**。

#### 已完成内容

- `loci-core` 已新增 backend-agnostic 的宿主机能力采集模块
- `InferenceEngine::build()` 在启动时会生成一次宿主机能力快照
- `runtime_snapshot()` 已正式暴露 `host` 字段
- CLI 默认 runtime 输出现在已经能看到真实宿主机信息
- 这些快照现在已经开始参与真实规划：
  - resident budget 估算
  - tiered-offload spill budget 修正
  - cold KV / cold weights 的 disk 倾向决策

#### 当前 `host` 快照已覆盖的信息

- 主机与系统：
  - `host_name`
  - `os_name`
  - `os_version`
  - `kernel_version`
- CPU：
  - `cpu_brand`
  - `cpu_vendor`
  - `cpu_frequency_mhz`
  - `physical_cores`
  - `logical_cores`
- 内存与交换区：
  - `total_memory_bytes`
  - `available_memory_bytes`
  - `total_swap_bytes`
  - `free_swap_bytes`
- 运行时环境：
  - `uptime_secs`
  - `load_average_one / five / fifteen`
- 磁盘：
  - mount point
  - file system
  - total / available bytes
  - removable 标记
- 轻量级本地探针：
  - `cpu_scalar_gops`
  - `memory_bandwidth_gbps`
  - `disk_read_mbps`
  - `disk_write_mbps`
  - `probe_bytes`
  - `probe_duration_ms`

#### 这项能力在架构上的意义

这一步非常关键，因为它把下面两类信息正式区分开了：

- **`host`**
  - 代表宿主机从 OS 视角可观测到的整体资源与轻量级本地能力
  - 不依赖某个 backend 是否愿意暴露更多硬件信息

- **`topology`**
  - 代表 backend 当前能够用于执行的 device 视图
  - 例如 OpenVINO 当前实际暴露的 `CPU / GPU / NPU / Disk` 执行目标

这意味着后续 Planner 可以逐步形成两层决策：

1. 先看宿主机整体资源是否允许某种 residency / spill / prefetch 策略
2. 再看具体 backend topology 是否真的具备对应执行设备

这意味着当 Loci 后续继续做更细的图拆分 / 子图下沉时，Planner 不必再完全依赖 backend 的乐观设备视图，而可以提前结合：

- 宿主机真实可用内存
- 磁盘真实剩余空间
- 启动时磁盘吞吐探针

更早剪掉明显不合理的放置组合，减少“先规划、后回退”的重复开销。

#### 当前边界与不足

- 还没有更底层的芯片计数器采集：
  - PCIe / NUMA 带宽
  - GPU / NPU 实时占用率
  - backend-specific perf counter
- 轻量级本地探针目前只适合做启动时估计与调度启发
  - 还不能替代正式 benchmark
- 当前 `host` 快照是 build 时采样
  - 不是高频实时刷新 telemetry
- 不同平台上 `load average` 的可用性与语义会有差异

#### 对后续实现的直接要求

- Planner 后续应逐步消费 `host` 快照，而不是只依赖 backend topology
- `tiered-offload` 后续应把：
  - `available_memory_bytes`
  - 磁盘可用空间
  - 磁盘吞吐探针
  纳入更真实的 spill / prefetch 决策
- 多芯片 backend 接入后，`host` 应继续保持 backend-agnostic，不与任何单一厂商 runtime 绑定

### 5.5 本轮推进后的架构收敛结论（2026-04-30 本轮新增）

这一轮继续实现后，Loci 在“去 OpenVINO 中心化、保留 OpenVINO 强执行路径”这件事上已经更接近正确架构。

#### 本轮已落地的关键收敛

- **backend 资产边界已经正式协议化**
  - `protocol` 中已新增：
    - `ExecutionArtifactKind`
    - `BackendAssetCapabilities`
  - backend 现在可以明确声明：
    - 哪些 layout 可直接执行
    - 哪些 layout 仅可 ingest、仍需 lowering / conversion
    - 自己偏好的执行 artifact 是什么

- **runtime snapshot 已开始暴露 backend 资产能力**
  - 当前 snapshot 不再只暴露 topology / lowering
  - 也会暴露每个 backend 的 `backend_assets`
  - 这让“后端到底吃什么模型资产”第一次变成可观察能力

- **backend 选择逻辑已从 `supports_model()` 向 readiness 诊断收敛**
  - `choose_backend()` 现在优先依据：
    - readiness 是否 `ready`
    - readiness 是否 `format_supported`
    - backend 是否支持 multimodal / NPU
  - 而不是继续依赖 backend 内部零散的 `supports_model()` 启发式
  - `supports_model()` 现在更适合作为 backend 自己的保守安全检查，而不是 Loci 核心真相来源

- **backend profile 分发已不再依赖字面名字**
  - planner 现在按 `runtime_family` 分发 `OpenVino` / `Candle` backend profile
  - 这意味着未来即使 backend 名称不是字面 `openvino` / `candle`，只要 runtime family 正确，也能进入正确 profile 路径
  - 这是后续接入：
    - Intel 多实现变体
    - Qualcomm / RKNN / 其他芯片 backend
    的必要前置收敛

- **显式 backend 资产声明已开始压缩 core 里的 OpenVINO 语义外溢**
  - `model_inspector` 现在会优先尊重 backend 自己声明的 asset boundary
  - 只在必要场景（如路径缺失、仍需保守回退）才使用更粗的 format fallback
  - 这一步很重要，因为它避免 core 再次把“runtime family 的刻板印象”当作真实资产边界

#### 这一步的真实意义

这并不意味着 Loci 已经摆脱 OpenVINO。

真正的意义是：

- **OpenVINO 仍然是当前最强执行路径**
- **但 Loci 核心现在开始围绕“backend 声明的能力边界”组织，而不是围绕 OpenVINO 专用格式组织**

这是后续继续接入：

- `QNN`
- `RKNN`
- `ONNX Runtime`
- `tract`
- 其他芯片 backend

时最重要的架构前提之一。

#### 本轮审计后仍明确存在的缺口

- **低层芯片算子支持仍未实现**
  - 目前仍只有：
    - operator class 枚举
    - lowering capability 描述
    - planner 产出的 subgraph guidance
  - 还没有：
    - chip operator registry
    - backend-specific operator callback
    - per-layer / per-op lowering ABI
    - 跨 backend 通用 kernel dispatch 边界

- **OpenVINO 仍主要消费“planner 指导 + runtime priorities”，不是完整图级 affinity 写回**
  - 当前已经进入“真实消费 lowering 指导”的阶段
  - 但还没进入“逐节点 / 逐子图真实重写执行图”的阶段

- **任意模型支持还停留在“资产诊断正确、执行路径未全打通”**
  - 现在 Loci 已经更清楚地知道：
    - 模型是否 ready
    - 需要什么 artifact
    - 哪个 backend 更适合
  - 但自动转换 / 导出 workflow 仍未完成

#### 当前可准确对外描述的能力状态

如果现在要对外开源描述，最准确的说法应当是：

> **Loci 已具备 backend-agnostic 的统一编排骨架、真实 OpenVINO 执行路径、宿主机能力探测、资产级 readiness 诊断、基础 tiered-offload runtime 与 backend-lowering 协议骨架；但低层芯片算子 ABI、真实 Candle 执行路径、paged KV 执行层、自动导出工作流仍未完成。**

### 5.6 本轮 lowering ABI 补强结论（2026-04-30 本轮新增）

这一轮继续推进后，Loci 的 lowering 协议不再只有 `subgraph` 级标签，而是开始形成更可执行的中间层。

#### 本轮新增内容

- `BackendLoweringPlan` 现在除 `subgraphs` 外，还新增：
  - `partitions`
  - `operators`

- `partitions` 的作用：
  - 把多个 lowering region 按 target / device / affinity 归并
  - 形成 backend 更容易消费的执行分区视图

- `operators` 的作用：
  - 把现有 subgraph guidance 正规化成 operator-facing 记录
  - 为后续：
    - per-layer affinity
    - chip operator callback
    - kernel dispatch
    提前提供协议骨架

#### 当前真实状态

- **这仍不是“低层芯片算子已经实现”**
  - 当前实现的是：
    - operator / partition 级 lowering 协议骨架
    - backend 基本校验
    - OpenVINO priorities 路径优先消费 partition affinity

- **但这已经明显比只有 `subgraphs` 更接近真实 chip backend 需求**
  - 因为未来 backend 不必直接从松散的 subgraph 列表猜测执行分区
  - 而可以直接消费：
    - partition
    - operator
    两层中间表示

#### 对当前阶段的准确判断

现在最准确的说法应当改成：

> **Loci 已实现 subgraph + partition + operator 三层 lowering 协议骨架，但仍未实现真实 per-layer / per-op 执行 runtime。**


### 6. 参考项目与论文（按能力模块整理）

下面将参考资料拆成“项目”与“论文”两类，并按 Loci 目标能力归档，方便后续下载、对标与做实现拆分。

#### 6.1 异构执行 / NPU-GPU-CPU 协同

- 项目：[OpenVINO](https://github.com/openvinotoolkit/openvino)
  - 参考点：Heterogeneous Execution、NPU/GPU/CPU 统一 runtime、GenAI runtime。
- 项目：[mllm](https://github.com/UbiquitousLearning/mllm)
  - 参考点：移动端多后端推理框架，可作为 llm.npu 相关工程入口。
- 论文：llm.npu, arXiv:2407.05858
  - 链接：https://arxiv.org/abs/2407.05858
  - PDF：https://arxiv.org/pdf/2407.05858.pdf
  - 参考点：prompt chunking、outlier tensor split、block-level out-of-order NPU 调度。
- 论文：PowerInfer-2, arXiv:2406.06282
  - 链接：https://arxiv.org/abs/2406.06282
  - PDF：https://arxiv.org/pdf/2406.06282.pdf
  - 参考点：手机端异构资源利用、fine-grained cluster scheduling。

#### 6.2 磁盘分层 Offload / 冷启动 / 懒加载

- 项目：[FlexLLMGen](https://github.com/FMInference/FlexLLMGen)
  - 参考点：mmap、CPU/GPU/磁盘三级分层、异步 prefetch。
- 项目：[llama.cpp](https://github.com/ggml-org/llama.cpp)
  - 参考点：本地模型加载、量化、端侧部署工程化。
- 论文：FlexGen, arXiv:2303.06865
  - 链接：https://arxiv.org/abs/2303.06865
  - PDF：https://arxiv.org/pdf/2303.06865.pdf
  - 参考点：throughput-oriented offload、I/O-aware scheduling。
- 论文：EdgeFlow, arXiv:2604.09083
  - 链接：https://arxiv.org/abs/2604.09083
  - PDF：https://arxiv.org/pdf/2604.09083.pdf
  - 参考点：移动端冷启动、flash bandwidth 优化、CPU/NPU 粒度流水。

#### 6.3 KV Cache / Prefix Cache / 长上下文

- 项目：[Fox](https://github.com/ferrumox/fox)
  - 参考点：PagedAttention、continuous batching、prefix caching、OpenAI-compatible API。
- 项目：[CAKE / cakekv](https://github.com/antgroup/cakekv)
  - 参考点：KV cache eviction、layer-aware cache policy。
- 项目：[llm-d](https://github.com/llm-d/llm-d)
  - 参考点：prefix-cache-aware routing、P/D-aware inference scheduling。
- 论文：CAKE（ICLR 2025）
  - 链接：https://openreview.net/forum?id=EQgEMAD4kv
  - 参考点：KV cache eviction 与分层缓存策略。

#### 6.4 Rust 推理主路径 / 端侧 Runtime 设计

- 项目：[Candle](https://github.com/huggingface/candle)
  - 参考点：Rust tensor/runtime 抽象、GGUF 支持、跨 CPU/CUDA/Metal/WASM。
- 项目：[Burn](https://github.com/tracel-ai/burn)
  - 参考点：Rust-native tensor abstraction、backend trait、kernel/runtime 组织、ONNX 与多后端边界设计，可用于补强 Loci 的纯 Rust backend 设计与未来 `loci-backend-burn` 方向。
- 项目：[rvLLM](https://github.com/m0at/rvllm)
  - 参考点：Rust 高性能推理引擎设计、吞吐导向 runtime。
- 项目：[atoma-infer](https://github.com/atoma-network/atoma-infer)
  - 参考点：Rust 推理服务基础设施、OpenAI-compatible serving、工程化拆分方式。
- 项目：[rustformers/llm](https://github.com/rustformers/llm)
  - 参考点：早期 Rust LLM 推理实现，可用于理解社区历史设计路径。
- 项目：[llama-gguf](https://github.com/Lexmata/llama-gguf)
  - 参考点：纯 Rust GGUF 推理路径、轻量本地执行。
- 项目：[cake](https://github.com/evilsocket/cake)
  - 参考点：Rust 分布式/异构系统设计风格，可参考工程组织但不直接照搬架构。
- 项目：[neo-ai-dlr](https://github.com/neo-ai/neo-ai-dlr)
  - 参考点：轻量 runtime / compiler integration 思路。

#### 6.5 动态模型路由 / 多模型调度

- 项目：[LLMRouter](https://github.com/ulab-uiuc/LLMRouter)
  - 参考点：复杂度感知路由、统一 router framework、多策略可插拔。
- 项目：[vLLM](https://github.com/vllm-project/vllm)
  - 参考点：高性能 serving、模型池化思路、生产级调度接口。
- 论文：LLMRouterBench, arXiv:2601.07206
  - 链接：https://arxiv.org/abs/2601.07206
  - PDF：https://arxiv.org/pdf/2601.07206.pdf
  - 参考点：路由 benchmark、router 评估框架。

#### 6.6 端侧功耗 / 热量 / 稀疏激活

- 项目：[PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)
  - 参考点：hot/cold neuron 拆分、GPU-CPU hybrid inference。
- 论文：PowerInfer, arXiv:2312.12456
  - 链接：https://arxiv.org/abs/2312.12456
  - PDF：https://arxiv.org/pdf/2312.12456.pdf
  - 参考点：activation locality、consumer GPU + CPU 混合推理。

#### 6.7 综述 / 导航型仓库

- 项目：[Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference)
- 项目：[Awesome-On-Device-AI-Systems](https://github.com/jeho-lee/Awesome-On-Device-AI-Systems)
- 项目：[Awesome-Efficient-LLM](https://github.com/horseee/Awesome-Efficient-LLM)
- 项目：[Awesome-LLM-Inference-Serving](https://github.com/zenrran4nlp/Awesome-LLM-Inference-Serving)
- 项目：[Awesome-LLM-Inference-Engine](https://github.com/sihyeong/Awesome-LLM-Inference-Engine)

#### 6.8 歧义项补全与取舍

- `heteroinfer` / `heterollm`
  - 当前可落到同一条移动端异构推理论文线上，优先引用：**HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs**, arXiv:2501.14794
  - 链接：https://arxiv.org/abs/2501.14794
  - PDF：https://arxiv.org/pdf/2501.14794.pdf
  - 说明：部分二级索引或摘要描述会把系统写作 `HeteroInfer`，但公开预印本更适合统一写作 `HeteroLLM`。

- `dynamicinfer`
  - 论文：**DynamicInfer: Runtime-Aware Sparse Offloading for LLMs Inference on a Consumer-Grade GPU**
  - 链接：https://openreview.net/forum?id=CvjmvjlczZ
  - PDF：https://openreview.net/pdf?id=CvjmvjlczZ
  - 说明：截至 2026-04-27，未找到稳定公开的官方 GitHub 仓库，更适合作为论文参考而不是工程依赖。

- `tapas`
  - 论文：**TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms**, arXiv:2501.02600
  - 链接：https://arxiv.org/abs/2501.02600
  - PDF：https://arxiv.org/pdf/2501.02600.pdf
  - 说明：更适合作为功耗/热量调度策略参考。

- `rust-llm-inference`
  - 不建议继续保留为单一模糊名字，建议拆成具体 Rust 项目：`Candle`、`Burn`、`rvLLM`、`atoma-infer`、`rustformers/llm`、`llama-gguf`。

- `dovetail`
  - 论文：**Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM Inference**, arXiv:2412.18934
  - 链接：https://arxiv.org/abs/2412.18934
  - PDF：https://arxiv.org/pdf/2412.18934.pdf
  - 说明：重点在 CPU/GPU 协同 speculative decoding。

- `specoffload`
  - 论文：**SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices**, arXiv:2505.10259
  - 链接：https://arxiv.org/abs/2505.10259
  - PDF：https://arxiv.org/pdf/2505.10259.pdf
  - 说明：重点在资源受限设备上的 offload 与 latent GPU capacity 挖掘。

- `zenlm/engine`
  - 仓库链接：https://github.com/zenlm/engine
  - 说明：该仓库页当前可访问；仓库 README 中仍保留 `zen-engine` 的旧命名与 clone 示例，后续引用时建议统一使用仓库主页 `zenlm/engine`，并将 `zen-engine` 视作历史名称。

**本地归档位置**：`tmp/references/`
- `papers/`：论文 PDF
- `repos/`：成功拉取的 Git 仓库
- `repo-archives/`：通过 GitHub zipball 下载的源码归档
- `repo-sources/`：从压缩包解压后的源码目录
- `manifests/`：下载清单与索引
