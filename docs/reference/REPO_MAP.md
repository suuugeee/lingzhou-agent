# 仓库地图与依赖边界

定义「目录职责 + 允许依赖方向」。新增代码前先对照 [变更落点决策树](#变更落点决策树)。  
工程分阶段计划见 [ENGINEERING_OPTIMIZATION_ROADMAP.md](../design/ENGINEERING_OPTIMIZATION_ROADMAP.md)；产品/蓝图差距见 [ARCHITECTURE.md](../design/ARCHITECTURE.md)。

## 目录职责

| 目录 | 职责 | 允许依赖 | 不应承载 |
|------|------|----------|----------|
| `cli/` | CLI 入口、参数、用户交互 | `core/*`、`provider/*`、`tools/*`、`memory/*`、`store/*` | 核心状态机、工具实现、模型协议 |
| `core/` | 认知循环（感知/判断/执行/反思）与编排；横切见 `log_fields`、`smoke_tests` | `provider/*`、`memory/*`、`store/*`、`tools/registry` | `cli/*`、渠道适配细节 |
| `core/config/` | `lingzhou.json` → `Config` | `config_models` | 业务逻辑 |
| `core/config_models/` | Pydantic 子配置模型 | 仅 stdlib/pydantic | 运行时编排 |
| `core/persona/` | Soul 工作区文件、`SelfModel`、`PersonaEngine` | `config`、`workspace` | 判断/执行 |
| `core/skill/`、`core/subagent/` | 技能注册、子灵 Runner | `tools`、judgment/execution | 主循环细节 |
| `core/paths/`、`core/plugin/` | 路径解析、插件加载 | — | 领域逻辑 |
| `core/contracts/` | 跨层稳定类型契约（探针、执行签名、工具 metadata） | 仅 stdlib / 彼此 | 运行时编排、IO |
| `core/execution/` | 工具派发、Run 收尾、Worker 池 | `contracts`、`tools/registry`、memory/store | LLM、judgment 决策 |
| `core/loop` | 认知循环包；公开入口 `from core.loop import CognitionLoop` | 同上 | 业务策略、工具实现细节 |
| `core/loop/runtime/` | `main` / `chain` / `startup` / `reload` / `memory_hooks` | `cycle`、`tick`、`runs`、`drive` | 工具实现 |
| `core/loop/tick/` | 单轮 tick（`prep` / `exec` / `memory` / `types`；`__init__` 编排） | `shared`、`cycle`、`task`、`judgment`、`execution` | 工具实现 |
| `core/loop/task/` | 任务 hint（`runtime`）、并行委派（`parallel`） | `loop.runtime`、`judgment`、`execution` | tick 编排 |
| `core/loop/cycle/` | 事件驱动（`driver` / `dispatcher` / `chat` / `focus`） | `runtime`、`shared` | Run 业务 |
| `core/loop/runs/` | Run 路由（`driver`）与刷新（`refresh`） | `cycle`、`execution` | tick 细节 |
| `core/loop/shared/` | tick 共享 helper（`common` / `logging` / `continue_phase` / …） | `config`、`contracts` | LLM、store 实现 |
| `core/loop/drive/` | `behavior`、`self_drive` | `store`、WM | 判断/执行 |
| `core/judgment` | 决策、路由、编排入口 | 同上 | 直接外部 IO |
| `core/judgment/boundary` | 输出归一化、解析后流水线 | `core/judgment/output` | LLM 调用 |
| `core/judgment/decision` | 模型路由、重试、LLM 调用、`rounds` 编排、`executor` mixins | `provider/*`、`output` | 上下文组装 |
| `core/judgment/frame.py` | `CognitionFrame` 认知基底 | `perception`、`store`、`memory` | LLM、编排 |
| `core/judgment/policy` | 可配置阈值与窗口策略（continue、routing 预算） | `core/config` | LLM、工具执行 |
| `core/judgment/context/` | 判断上下文子模块（`utils` / `sections` / `budget` / …）；**直引子模块**，`__init__` 不重导出 | `perception`、`store` | LLM 调用 |
| `tools/` | 工具实现与 `ToolResult` | `core` 稳定接口、`memory/*`、`store/*`、SDK | 主循环编排 |
| `tools/paths.py` | workspace/skills 路径解析（Config 与测试 fixture 兼容） | `tools/registry` | 业务逻辑 |
| `provider/` | 模型协议、认证、用量 | `core/config` 等 | 任务编排、工具策略 |
| `memory/` | WM / 情节 / 语义 / TaskStore 门面 | `store/*` | LLM 路由、UI |
| `store/` | SQLite / 文件持久化 | 无上层业务 | — |
| `channels/` | wechat / webhook / local | `core` 稳定入口 | 核心决策、工具实现 |
| `docs/` | 架构、参考、治理文档 | — | — |

## 依赖方向

自上而下（下层不得反向依赖上层）：

```text
cli / channels  →  core  →  provider | memory | tools  →  store
```

- `store` 不依赖 `core/*`。
- `tools` 由 `core` 调度，不反向驱动 `core/loop`。
- `provider` 不接管任务调度。

## 变更落点决策树

1. **编排 vs 执行** → 编排：`core/loop`、`core/judgment`；执行：`tools/*`
2. **模型访问 vs 业务策略** → 模型：`provider/*`；策略：`core/*`
3. **持久化 vs 运行时逻辑** → 持久化：`store/*`、`memory/*`；运行时：`core/*`

## ADR 与文档同步

以下变更须先写 ADR（流程见 [docs/adr/README.md](../adr/README.md)）：

- 主循环语义（tick 顺序、并发、热重载）
- provider 协议或认证链路
- memory / store schema 或持久化语义
- `ToolResult` 核心字段契约

同步更新：

| 变更类型 | 更新文档 |
|----------|----------|
| 目录/依赖边界 | 本页 |
| 认知循环语义 | [ARCHITECTURE.md](../design/ARCHITECTURE.md) |
| 配置项 | [CONFIG.md](CONFIG.md) |
| 工具能力 | [TOOLS.md](TOOLS.md) |
