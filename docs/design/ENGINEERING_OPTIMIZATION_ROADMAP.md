# 工程优化路线图

本页是 Lingzhou **编码、架构、目录与文档演进** 的单一执行入口。  
与下列文档分工明确，避免重复维护：

| 文档 | 职责 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 认知循环、模块说明、**产品/蓝图差距** |
| [REPO_MAP.md](../reference/REPO_MAP.md) | 目录职责、依赖方向、变更落点决策树 |
| [adr/README.md](../adr/README.md) | 关键架构决策记录（ADR）流程与模板 |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | 开发环境、提交规范、测试命令 |

## 优化原则

- **边界优先**：先收敛模块边界，再优化实现细节。
- **机制先于策略**：优先修复输入校验、状态机、回退链，再做 prompt 或阈值调优。
- **不丢信息**：默认不截断业务信息；控制日志体量时用 `metadata.log_summary` 等结构化摘要，原文保留在 summary / evidence / artifact。
- **LLM 感知优先**：信号以叙事注入 WM，不机械阻塞或改写模型输出；禁止对模型可见正文做头尾切片（详见 [ADR 0015](../adr/0015-llm-perception-integrity.md)）。
- **可回归验证**：每个架构动作至少配一条行为测试或回归测试。

## 当前主要问题（工程视角）

- 模块边界存在跨层直连风险：`core/loop`、`core/judgment`、`tools` 职责仍有交叠。
- 部分关键路径可观测性不足，问题定位依赖长日志人工检索。
- 目录已形成事实标准，边界约束需通过 REPO_MAP + 静态扫描持续落地。
- 功能说明文档较全，变更决策与迁移策略依赖 ADR 补齐。

## 工作节奏

### 文档阶段（已完成）

- 统一 README / CONTRIBUTING 文档入口。
- 落地 [REPO_MAP.md](../reference/REPO_MAP.md) 与 [docs/adr/](../adr/README.md)。
- 架构总览与差距仍以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准（按源码核对，不重复罗列 gap 表）。

### 代码阶段（首批边界收敛已落地）

**准入条件**（须全部满足）：

1. 本路线图与 REPO_MAP 评审结论为「可执行」。
2. 已选定第一批边界修复点（见下文），每项对应 ADR 编号与验证计划。
3. 不引入机械截断/改写 LLM 输出的补丁；优先机制与边界。

**首批 3 个落点**（静态 import 扫描，2026-06）：

| 优先级 | 状态 | ADR | 说明 |
|--------|------|-----|------|
| 1 | 已完成 | [0001](../adr/0001-smoke-and-registry-public-api.md) | `check_command_risk`、`lookup_registered_tool` |
| 2 | 已完成 | [0002](../adr/0002-probe-contracts-layer.md) | `core/contracts/probe` |
| 3 | 已完成 | [0003](../adr/0003-provider-capabilities-facade.md) | `provider/capabilities.resolve_model_ref_for_input` |

涉及主循环语义、provider 认证、memory/store schema、`ToolResult` 核心字段变更时，须先写 ADR（见 [adr/README.md](../adr/README.md)）。

## 分阶段计划

### P0：稳定性与边界收敛

| 项 | 状态 | 说明 |
|----|------|------|
| Provider 空 token 防线 | 已落地 | chat/embed/refresh 路径拒绝空 `Authorization`；回归见 `tests/test_provider.py` |
| Hot-reload 原子语义 | 已落地 | candidate 只创建一次 provider 栈；见 `core/loop/runtime/reload.py` |
| 执行层日志摘要 | 已落地 | `metadata.log_summary` + `_run_progress_text` 偏好摘要，不裁工具正文 |
| 跨层依赖首批收敛 | 已完成 | ADR 0001–0003；回归 `tests/test_boundary_contracts.py` |

### P1：架构分层重构

| 项 | 状态 | ADR / 说明 |
|----|------|------------|
| `core/judgment` 分层 | 已完成 | [0004](../adr/0004-judgment-layer-packages.md) |
| `core/loop` 轻量化 | 首期已完成 | `policy.continue_history` |
| 路由上下文策略 | 已完成 | [0006](../adr/0006-routing-context-policy.md) — `policy/routing_context.py` |
| Tools metadata 契约 | 已完成 | [0005](../adr/0005-tool-result-metadata.md) — `tools/` 成功路径统一 `tool_metadata()`（含 `browser`/`probe`/`config`/`plan`/`skill`/`subagent`/`tts`/`ask` 等） |

跳过的错误/空参返回可不写 metadata；新增工具成功路径应默认使用 `tool_metadata()`。

### P2：目录与文档体系

| 项 | 状态 |
|----|------|
| Repo Map（目录 + 依赖方向） | 已完成 → [REPO_MAP.md](../reference/REPO_MAP.md) |
| ADR 机制与模板 | 已完成 → [docs/adr/](../adr/README.md) |
| README 文档入口（设计 / 参考 / 治理） | 已完成 |

文档目录目标形态：

```text
docs/
├── design/     # 架构、路线图、蓝图
├── reference/  # 配置、工具、REPO_MAP
├── guide/      # 操作与开发指南
└── adr/        # 架构决策记录
```

### P3（持续）：质量与可观测性

- 质量门禁：**已落地** — `scripts/run_governance_checks.sh`、CI（import 边界 + Ruff 治理面 + pytest）；演化 smoke 见 `core/smoke_tests.py` / ADR 0001。
- 日志字段统一：**已落地** → [ADR 0007](../adr/0007-structured-log-fields.md) — `core/log_fields.py`；执行/判断/主循环 tick；loop 反馈行优先 `log_summary`；`model_routing` 缓存 `catalog_models` 与工具能力映射。
- 执行层包化：**已落地** → [ADR 0008](../adr/0008-execution-package.md) — `core/execution/`；contracts 承载 `action_key_param`；shim 退役见 [ADR 0009](../adr/0009-compat-shim-retirement.md)。
- Import 边界门禁：**已落地** — `tests/test_import_boundaries.py`、`scripts/check_import_boundaries.py`。
- 每两周技术债盘点：新增 / 已清 / 风险变化。

### P1+（下一批）

| 项 | 状态 | 说明 |
|----|------|------|
| `judgment/runtime` 编排拆分 | 已完成 | [ADR 0010](../adr/0010-judgment-runtime-orchestration.md) — `frame.py` + `decision/rounds.py` |
| 共享 `tools/paths.py` | 已完成 | `workspace_dir_from_ctx` / `skills_dir_from_ctx` |
| `judgment/executor` mixin 拆分 | 已完成 | [ADR 0011](../adr/0011-executor-mixin-split.md) |
| 兼容 shim 删除 | 已完成 | [ADR 0009](../adr/0009-compat-shim-retirement.md) |
| GitHub Actions CI | 已完成 | `.github/workflows/ci.yml` |
| `judgment/context` 预算/信号拆分 | 已完成 | [ADR 0012](../adr/0012-judgment-context-budget-signals.md) |
| CI Ruff 全量 | 已完成 | `ruff check core tools` |
| CI 覆盖率 ≥80%（治理包） | 已完成 | judgment / execution / contracts |
| `context` chat/memory 拆分 | 已完成 | [ADR 0013](../adr/0013-context-sections-and-ci-gates.md) |
| `context` utils 拆分 | 已完成 | [ADR 0014](../adr/0014-context-utils-split.md) — `cache` / `tokens` / `template` / `schema` |
| LLM 感知完整性 | 已完成 | [ADR 0015](../adr/0015-llm-perception-integrity.md) — 整消息/整节省略，禁止 prompt 头尾截断 |
| `context` 包归并 | 已完成 | [ADR 0016](../adr/0016-context-package-consolidation.md) — `utils` + `sections`，去掉多层 re-export |
| `loop/tick_*` 扁平文件归并 | 已完成 | [ADR 0017](../adr/0017-loop-tick-package-consolidation.md) — 仅保留 `core/loop/tick/` 包 |
| `core` 根目录模块归并 | 已完成 | [ADR 0018](../adr/0018-core-package-layout.md) — config/skill/subagent/persona 等 |
| `core/loop` 子包分层 | 已完成 | [ADR 0020](../adr/0020-loop-subpackage-layout.md) — runtime/tick/task/cycle/runs/shared/drive |
| `judgment.context` 直引 | 已完成 | [ADR 0021](../adr/0021-judgment-context-direct-imports.md) — 去掉 context `__init__` 聚合 |
| startup schema 校验精简 | 已完成 | 删除源码注入 patch，仅 `config_models` 字段存在性检查 |

## 编码规范（增量）

- 新增/重构函数：单一职责、明确输入输出、错误可分类。
- 避免在 runtime 热路径引入静默 fallback。
- 修日志问题优先修机制，不用日志掩盖根因。
- 贡献流程与测试命令见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## 验收标准

- **P0**：新日志中无 `Bearer ` 空头；热重载无重复 `create_provider`；工具结果不因日志路径被截断。
- **P1**：`core/judgment` 与 `core/loop` 跨层 import 热点下降（可用静态扫描对比）。
- **P2**：架构变更有 ADR；从 README 可一跳到 ARCHITECTURE / 本路线图 / REPO_MAP / ADR。

## 与产品差距的关系

[ARCHITECTURE.md](ARCHITECTURE.md) 中的「当前实现差距」表描述**蓝图 vs 现状**（多模态感知、task-level routing、Run 语义等）。  
本路线图描述**如何改代码与仓库**；二者互补，不合并为同一表格，避免双处维护。
