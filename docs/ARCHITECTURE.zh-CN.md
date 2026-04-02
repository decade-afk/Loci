# Loci 架构说明

最后更新：2026-04-02

## 定位

Loci 是一个面向宿主产品的、本地优先、插件治理的 AI 运行时与管理平面。

它处在“底层推理绑定”和“最终用户 UX 外壳”之间：

- 低于具体产品的聊天或 agent UI
- 高于原始推理后端接入
- 核心关注点是可治理运行时缝隙、插件化升级、宿主可控的运维与管控

## 工作区结构

仓库现在是 Rust 工作区，而不是根层单体。

| Crate | 职责 |
|---|---|
| `crates/core` | 运行时内核、模型加载、插件库存、治理缝隙、管理服务 |
| `crates/cli` | `loci` 二进制与管理 HTTP 服务 |
| `crates/plugin-api` | 插件 manifest 与共享能力类型 |
| `crates/ffi` | 预留给原生集成稳定化的 crate |
| `crates/legacy-plugin-api` | 旧插件契约类型 |
| `crates/legacy-plugin-compat` | 旧文本插件兼容桥 |

## 架构主张

主规则是：凡是需要治理的行为，都应该放在显式缝隙之后，而不是硬编码在主流程里。

当前由 `CoreComponent` 定义的重写缝隙：

- `Inference`
- `Model`
- `Hardware`
- `Workflow`
- `EventBus`
- `PluginManager`
- `UiHost`

每个缝隙都可以被：

- 插件 manifest 声明
- 运行时清单查询
- 由宿主通过管理平面激活

## 分层模型

```text
宿主产品 / 自动化系统 / IDE 助手
    |
    v
CLI 管理入口（`crates/cli`）
    |
    v
管理服务（`crates/core::management`）
    |
    +--> 运行时快照 / 插件库存 / 激活控制
    +--> 模型加载治理
    +--> workflow、command、event、ui 路由
    |
    v
推理引擎（`crates/core::engine`）
    |
    +--> backend registry
    +--> core registry traits
    +--> plugin manager
    +--> legacy compatibility materialization
    |
    v
可选 `llama.cpp` 后端（`crates/core --features llama`）
```

## Core Registry

核心运行时被刻意拆分在 trait 后面：

- `ModelRepository`
- `WorkflowEngine`
- `EventBus`
- `HardwareAbstraction`
- `UiHost`
- `PluginManager`
- `CoreRegistry`

这样宿主或插件就可以替换这些能力，而不会把整个运行时重新耦合成一个大而乱的服务对象。

## 插件模型

插件以 manifest 为中心。

关键概念：

- track：`ai_infra`、`ai_agent`
- contribution points：模型提供方、加速器、推理 hook、事件、工作流、自定义节点、命令、UI 贡献
- core rewriters：缝隙级治理声明
- runtime artifacts：动态库路径、wasm 路径、sampling profile
- bootstrap activation：可选自动激活
- compatibility metadata：旧插件桥接信息

示例 manifest 位于 `plugins/`。

## 管理平面

管理平面保持克制，只暴露运维和治理所需能力：

- 运行时发现
- 后端发现
- 插件状态与详情查询
- core rewriter 清单与激活
- 模型加载
- 文本生成
- workflow 执行
- UI surface 呈现
- command 路由
- event 发布
- legacy text plugin 激活

路由详情见 `docs/MANAGEMENT_API.md`。

## 兼容策略

兼容能力是有边界的。

- 新架构是唯一主线。
- 旧文本插件仅通过 `legacy-plugin-api` 与 `legacy-plugin-compat` 保留。
- 旧根层单体模块、旧 CLI 路由、旧示例程序都不再属于维护中的架构。

这样迁移债务被限制在兼容隔离层里，而不会再次反向污染主线运行时。

## 运行时后端

`loci-core` 可以在不启用原生后端绑定的情况下构建，用于架构演进和治理验证。

如果要启用真实本地模型执行，请打开：

```bash
cargo build -p loci-core --features llama
```

`llama.cpp` 集成通过 `deps/llama.cpp` 子模块与 `crates/core/build.rs` 完成。

## 维护规则

- 以工作区 crate 为准
- 新运行时能力如果需要治理，就必须进入 seam 或 registry
- 文档只描述仍被维护的入口
- 旧兼容能力必须保持隔离并显式激活
