# Loci

Loci 是一个基于 Rust 工作区构建的、本地优先、插件治理的 AI 运行时与管理平面。

重构后的仓库以新工作区架构为唯一主线：

- `crates/core`：运行时内核、模型加载、治理接口、插件库存、管理服务
- `crates/cli`：`loci` 命令行程序，负责插件发现与管理 HTTP 服务
- `crates/plugin-api`：插件与宿主共享的 manifest / 能力类型
- `crates/ffi`：稳定公开的 C ABI 与原生集成入口
- `crates/legacy-plugin-api`、`crates/legacy-plugin-compat`：旧文本插件兼容隔离层

Loci 的定位不是聊天壳，而是给桌面软件、IDE 助手、本地 agent 系统、企业自动化产品提供可嵌入的运行时底座。

## 架构方向

新架构按“可治理边界”组织，而不是把能力硬编码进核心流程。核心行为可以在组件边界上被插件重写。

当前核心重写缝隙包括：

- `inference`
- `model`
- `hardware`
- `workflow`
- `event_bus`
- `plugin_manager`
- `ui_host`

插件以 manifest bundle 为主路径；原生 manifest 是主线，旧插件仅通过显式兼容桥接保留。

## 工作区结构

```text
loci/
|-- crates/
|   |-- cli/
|   |-- core/
|   |-- ffi/
|   |-- plugin-api/
|   |-- legacy-plugin-api/
|   `-- legacy-plugin-compat/
|-- plugins/                 # 新架构示例插件 manifest
|-- include/                 # 稳定 ABI 的公共 C 头文件
|-- deps/llama.cpp/          # 可选 `llama` feature 使用的子模块
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MANAGEMENT_API.md
|   |-- PRODUCT_STRATEGY_2026.md
|   `-- architecture/
|-- scripts/
|   `-- full_test.ps1
`-- wasm-plugin-sdk/
```

## 快速开始

先拉取仓库与 `llama.cpp` 子模块：

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
```

构建带 `llama.cpp` 后端的 CLI：

```bash
cargo build -p loci-cli --release --features llama
```

使用仓库内示例 manifest 启动管理服务：

```bash
cargo run -p loci-cli --features llama -- \
  --plugin-dir plugins \
  --management-bind 127.0.0.1:8080
```

基础检查：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/runtime
curl http://127.0.0.1:8080/v1/core/rewriters/inventory
```

通过控制平面加载模型：

```bash
curl http://127.0.0.1:8080/v1/model/load \
  -H "Content-Type: application/json" \
  -d "{\"backend_name\":\"llama.cpp\",\"config\":{\"model_path\":\"D:/models/qwen.gguf\"}}"
```

执行文本生成：

```bash
curl http://127.0.0.1:8080/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"hello from loci\"}"
```

## 插件模型

Loci 现在优先使用 manifest bundle，而不是旧式的随意运行时契约。

- 用 `--plugin-dir <path>` 装载整个插件目录
- 用 `POST /v1/plugins/load` 装载 bundle 文件或目录
- 用 `POST /v1/core/rewriters/activate` 激活组件重写权
- 只有在确实需要兼容旧文本插件时，才使用 `POST /v1/legacy-text/activate`

示例 manifest 位于：

- `plugins/example-inference`
- `plugins/example-infra`
- `plugins/example-agent`

细节见 [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md)。

## 真实构建与测试命令

工作区验证：

```bash
cargo test -q
```

`llama.cpp` 集成验证：

```bash
cargo test -q -p loci-core --features llama
cargo test -q -p loci-cli --features llama
```

Windows 辅助脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/full_test.ps1
```

## 文档

- [架构说明](docs/ARCHITECTURE.md)
- [管理 API](docs/MANAGEMENT_API.md)
- [FFI 说明](docs/FFI.md)
- [插件指南](PLUGIN_GUIDE.md)
- [架构 ADR](docs/architecture/README.md)
- [2026 产品策略](docs/PRODUCT_STRATEGY_2026.md)
- [构建指南](BUILD.md)

## 当前状态

现在以工作区主线为准。旧的根层单体源码、旧示例程序、旧 `serve/generate` 时代文档不再是真相来源。

当前已经做实的能力：

- 插件治理的运行时核心
- 管理 HTTP 控制平面
- 运行时快照与插件库存
- inference、model、hardware、workflow、event bus、plugin manager、ui host 七类重写激活
- 真实模型加载治理与可选 `llama.cpp` 集成
- `crates/ffi` + `include/loci.h` 组成的稳定公开 C ABI
- 有边界的旧文本插件兼容

刻意收敛的部分：

- 旧插件桥接只保留在 compat crate 中

## 许可证

本项目可按以下任一许可证使用：

- Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE)）
- MIT License（[LICENSE-MIT](LICENSE-MIT)）
