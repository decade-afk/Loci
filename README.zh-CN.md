# Loci

[English](README.md)

Loci 是一个通用化、可嵌入到其他软件中的 AI 推理运行时，面向希望把模型能力真正集成进产品、而不是单独再造一套运行时基础设施的团队。

它的目标形态不是一个面向最终用户的聊天界面，而是桌面应用、IDE 助手、本地自动化工具、服务进程和自定义 Agent Shell 背后的“推理引擎层”。Loci 负责承接模型执行、插件升级、工具接入、会话管理和宿主集成。

## 定位

Loci 希望处在这样一个层级：

- 模型执行层：提供本地推理能力（文本/图像路径均可扩展），并预留动态内核扩展点。
- 运行时层：统一处理插件生命周期、工具注册、会话、策略和宿主可控的生成流程。
- 集成层：通过 Rust 嵌入、C ABI、REST 服务和多语言模板，接入其他软件。

如果你需要的是一个可以被你的软件嵌入、升级、控制和扩展的 AI 推理引擎，Loci 的定位就是这里。这个定位是领域中立的：游戏工具链、IDE 助手、桌面自动化、内部 Copilot、服务包装层都可复用同一运行时核心。
如果你希望把它继续打造成可演进的 Agent Runtime，可参考 [`docs/AGENT_RUNTIME_BLUEPRINT.md`](docs/AGENT_RUNTIME_BLUEPRINT.md)。

## Loci 目前已经具备的能力

- 基于 Rust 的本地优先文本推理运行时。
- 具有确定性顺序的运行时插件链路。
- 支持静态、动态和 WASM 三类运行时插件。
- 支持动态工具插件和 MCP stdio 连接。
- 支持会话管理和可持久化的 session-store 后端。
- 支持 serve 控制面的分发策略、执行策略和管理鉴权策略。
- 支持模型拉取策略注册表，可对导入源、远程下载和校验要求进行插件化治理。
- 支持模型拉取 verifier 注册表，可在下载与校验完成后继续基于 sidecar、签名或其他宿主信任规则拒绝导入。
- 在原生运行时接口之上提供 OpenAI 兼容 REST 桥接层，覆盖 `/v1/models`、`/v1/chat/completions`，并提供部分 `tool_calls` 兼容能力，底层仍复用 Loci 原生工具注册表。
- 提供 `/v1/embeddings` 的 OpenAI 兼容 embeddings 桥接接口。
- 提供原生 REST 生成面与模型资产、session、tool、策略控制面，方便宿主侧编排。
- 提供 `/events` 与 `/events/stream` 运行时审计事件流，方便宿主做活动面板、监督、日志桥接与控制面追踪。
- 提供 `serve` 运行时控制参数，可配置 worker、队列、backpressure 和治理插件。
- 提供多种集成方式：
  - Rust crate
  - C ABI，入口见 [`include/loci.h`](include/loci.h)
  - REST 服务 `loci serve`
  - 集成模板，见 [`examples/integration/templates/README.md`](examples/integration/templates/README.md)

## 插件与扩展体系

Loci 的核心理念之一，是运行时应该可以在不重写核心的情况下持续升级。

运行时插件类型：

- 静态插件：编译进二进制。
- 动态插件：运行时热加载 `.dll` / `.so` / `.dylib`。
- WASM 插件：以沙箱方式进行运行时升级。

运行时插件钩子：

- `pre_generate`
- `transform_logits`
- `post_sample`
- `on_token`
- `post_generate`

仓库内还提供了其他扩展点：

- 工具插件
- 动态推理后端内核
- 动态图像内核
- Serve 分发策略插件
- 执行策略插件
- 管理鉴权策略插件

## 快速开始

### 1. 环境要求

- Rust 1.70+
- CMake 3.15+
- C/C++ 编译器
- 用于真实推理的 GGUF 模型文件

更完整的平台构建说明见 [`BUILD.md`](BUILD.md)。

### 2. 准备 `llama.cpp`

Loci 默认要求 `llama.cpp` 源码位于 `deps/llama.cpp`。

常见做法：

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
```

如果 `deps/llama.cpp` 缺失，当前构建脚本会自动回退到 stub backend，用于测试和非 `llama.cpp` 工作流；但如果你要做真实模型推理，仍然需要把真正的 `llama.cpp` 源码放到 `deps/llama.cpp`。

### 3. 编译

```bash
cargo build --release
```

主要产物：

- CLI：`target/release/loci`
- Rust 库：`target/release`
- C ABI 头文件：[`include/loci.h`](include/loci.h)

### 4. 下载一个小模型

例如：

```bash
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

端侧资源受限实测（真实模型导入 + 推理，远程下载失败时自动回退本地文件）：

```powershell
pwsh ./scripts/edge_resource_smoke.ps1 `
  -LociExe ./target/debug/loci.exe `
  -ModelStore models `
  -ModelId edge-qwen05b-q4km `
  -LocalSource D:/Code/Reptile/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

脚本会验证：
- 托管模型导入（`model pull`）
- 端侧受限推理（`cpu-only`、`threads=1`、`context=256`、`max_tokens=16`）
- `--mmap` 与 `--no-mmap` 都可稳定执行，并输出耗时

### 5. 开始推理

CLI：

```bash
cargo run --release -- generate \
  --model path/to/model.gguf \
  --prompt "用一段话解释 Loci 的定位。"
```

流式输出：

```bash
cargo run --release -- generate \
  --model path/to/model.gguf \
  --prompt "请一步一步思考。" \
  --stream
```

启动 REST 服务：

```bash
cargo run --release -- serve \
  --model path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-prompt-bytes 65536 \
  --workers 4 \
  --queue-size 128
```

通过 REST 发起生成：

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello from Loci\",\"max_tokens\":64}"
```

通过 REST 获取运行时信息：

```bash
curl http://127.0.0.1:8080/info
```

通过 OpenAI 兼容接口发现模型并发起聊天：

```bash
curl http://127.0.0.1:8080/v1/models

curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"loci\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话解释 Loci。\"}]}"
```

通过 OpenAI 兼容 embeddings 接口获取向量：

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"loci\",\"input\":[\"Loci\",\"runtime\"]}"
```

通过控制面进行模型资源规划：

```bash
curl http://127.0.0.1:8080/models/plan ^
  -H "Content-Type: application/json" ^
  -d "{\"context_size\":8192}"
```

向托管模型清单注册一个现有模型文件：

```bash
curl http://127.0.0.1:8080/models/assets ^
  -H "Content-Type: application/json" ^
  -d "{\"path\":\"D:/models/qwen.gguf\",\"id\":\"qwen-local\",\"tags\":[\"reasoning\"]}"
```

以 NDJSON 流方式导入模型并回传进度：

```bash
curl http://127.0.0.1:8080/models/assets/pull?stream=true ^
  -H "Content-Type: application/json" ^
  -d "{\"source\":\"D:/downloads/qwen.gguf\",\"id\":\"qwen-managed\"}"
```

该 NDJSON 流会发出 `progress`、`complete` 和 `error` 事件，方便宿主实时展示模型导入状态。

把同样的导入流程提交为后台控制面任务：

```bash
curl http://127.0.0.1:8080/models/assets/pulls ^
  -H "Content-Type: application/json" ^
  -d "{\"source\":\"https://example.com/qwen.gguf\",\"id\":\"qwen-managed\"}"
```

订阅单个后台模型拉取任务：

```bash
curl http://127.0.0.1:8080/models/assets/pulls/pull-1730937600000-1/events
```

启用更严格的远程模型拉取策略：

```bash
curl http://127.0.0.1:8080/model-pull-policies/checksum-required-remote.model.pull/activate -X POST
```

启用模型拉取后的 verifier：

```bash
curl http://127.0.0.1:8080/model-pull-verifiers/sidecar-sha256.model.verify/activate -X POST
```

获取运行中的 OpenAPI 规范：

```bash
curl http://127.0.0.1:8080/openapi.yaml
```

```bash
curl http://127.0.0.1:8080/openapi.json
```

读取最近的运行时审计事件：

```bash
curl http://127.0.0.1:8080/events?limit=20
```

持续跟随运行时审计流：

```bash
curl http://127.0.0.1:8080/events/stream?replay=20
```

## 将 Loci 集成进你的软件

### Rust 嵌入

```rust
use loci::inference::{GenerationParams, InferenceEngine};
use loci::model::ModelConfig;

fn main() -> loci::Result<()> {
    let config = ModelConfig::new("path/to/model.gguf")
        .with_context_size(4096)
        .with_gpu_layers(0);

    let mut engine = InferenceEngine::new(config)?;
    let output = engine.generate(
        "总结一下 Loci 的作用。",
        GenerationParams::default(),
    )?;

    println!("{output}");
    Ok(())
}
```

### C ABI

Loci 提供了适合宿主软件和其他语言绑定使用的 C ABI，覆盖：

- engine 生命周期
- 同步和流式生成
- 设备探测辅助
- 运行时插件生命周期
- 插件注册表管理

建议从这些入口开始：

- [`include/loci.h`](include/loci.h)
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)
- [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md)

### REST

`loci serve` 提供了适合集成方使用的生成面和控制面，包括：

- `/health`
- `/info`
- `/metrics`
- `/generate`
- `/v1/models`
- `/v1/chat/completions`
- `/v1/embeddings`
- `/api/generate`
- `/models/plan`
- `/models/assets`
- `/sessions`
- `/tools`
- `/dispatch-policies`
- `/execution-policies`
- `/auth-policies`
- `/model-pull-policies`
- `/model-pull-verifiers`

其中 OpenAI 兼容路由的目标，是帮助现有客户端和 SDK 更快接入；Loci 的原生 REST、C ABI 和插件体系仍然是主要的集成契约。
当启用 `tool_calls` 兼容时，真正可调用的工具仍然来自 Loci 已注册的内置工具、tool plugin 或 MCP 工具，而不是兼容层单独维护的一套工具目录。
embeddings 兼容接口也是同样思路：它只是对当前 Loci 后端能力的桥接，而不是单独再维护一套独立的向量子系统。

适合嵌入式宿主使用的 `loci serve` 运行时控制参数：

- `--workers`
- `--queue-size`
- `--backpressure`
- `--management-auth-policy-name`
- `--model-pull-policy-name`
- `--model-pull-verifier-name`

OpenAPI 规范见：

- [`docs/openapi/loci-rest-v1.yaml`](docs/openapi/loci-rest-v1.yaml)

## 项目结构

```
loci/
|-- src/                # 核心运行时、控制面、插件契约
|-- include/            # C 头文件
|-- docs/               # 架构、API、ADR、路线文档
|-- tests/              # 集成与 E2E 测试
|-- benches/            # 基准测试
|-- examples/           # 示例、插件示例、集成模板
|-- scripts/            # 本地工具与冒烟脚本
|-- web/                # 内置助手控制台静态资源
|-- android-sdk/        # Android SDK 与 sample app
|-- wasm-plugin-sdk/    # WASM 插件 SDK crate
|-- models/             # 本地模型目录（已忽略）
|-- deps/
|   `-- llama.cpp/      # llama.cpp 子模块
`-- Cargo.toml
```

## 当前项目状态

Loci 目前已经可以作为“可嵌入推理层”和“运行时控制层”使用，但并不是所有高级路径都同样成熟。

当前状态大致如下：

- 本地文本推理：稳定
- 运行时插件系统：稳定
- C ABI 与 REST 集成：可用
- 工具 / 插件 / 策略控制面：已实现
- 多模态与编排路径：部分完成

更细的分阶段说明见 [`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md)。

## 仓库文档入口

- 架构设计：[`docs/ARCHITECTURE.zh-CN.md`](docs/ARCHITECTURE.zh-CN.md)
- 构建说明：[`BUILD.md`](BUILD.md)
- API 参考：[`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)
- 集成说明：[`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md)
- 产品路线：[`docs/PRODUCT_STRATEGY_2026.md`](docs/PRODUCT_STRATEGY_2026.md)
- REST OpenAPI：[`docs/openapi/loci-rest-v1.yaml`](docs/openapi/loci-rest-v1.yaml)
- 插件指南：[`PLUGIN_GUIDE.md`](PLUGIN_GUIDE.md)
- 集成模板：[`examples/integration/templates/README.md`](examples/integration/templates/README.md)

## 适合的使用方向

Loci 适合这类产品形态：

- 需要本地推理运行时的 Tauri / Electron 应用。
- 需要嵌入式推理层的 IDE 助手。
- 需要工具调用与插件升级能力的桌面 Copilot。
- 需要本地自动化和推理能力的宿主运行时。
- 想做类似 Ollama 本地执行核心、但更强调嵌入式集成和运行时可控性的产品。

## 许可证

双许可证：

- Apache-2.0
- MIT
