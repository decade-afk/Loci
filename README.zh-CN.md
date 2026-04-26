# Loci

Loci 是一个轻量、插件化、面向本地 LLM 推理的 Rust Infra 运行时。

当前仓库只承担基础设施职责：

- 模型加载
- 推理运行时
- 硬件后端选择
- 插件发现与激活
- Rust / C / 本地 HTTP 接口

明确不承担：

- 桌面 UI
- 桌宠业务逻辑
- 终端产品工作流外壳

## 当前主线

新的主构建路径已经按 Infra 边界收口：

- `crates/core`: 推理引擎、插件注册表、运行时快照、`llama.cpp` 绑定
- `crates/plugin-api`: 稳定插件清单类型
- `crates/ffi`: C ABI
- `crates/server`: 本地 sidecar HTTP 接口
- `crates/cli`: 本地调试与启动入口

`Loci-refactor` 中的 UI host、workflow 治理、legacy 文本插件兼容等历史实验代码，不再属于新的 workspace 主线。
