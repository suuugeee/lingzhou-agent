# ADR 0016: judgment context 包归并

- Status: Accepted
- Date: 2026-06-01
- Supersedes: [0013](0013-context-sections-and-ci-gates.md)、[0014](0014-context-utils-split.md) 的「多文件 + 聚合 re-export」形态（行为不变）

## Context

`context/` 曾拆为 `cache`、`template`、`schema`、`tokens`、`chat_sections`、`memory_sections` 及 `runtime`/`utils` 双层转发，单文件过小、跳转成本高，与「按职责分包」初衷相悖。

## Decision

`core/judgment/context/` 收敛为按**稳定职责**的少量模块：

| 文件 | 职责 |
|------|------|
| `utils.py` | LRU 缓存、模板填充、schema 校验、token 估算（含仅测试用的 `_compress_*`） |
| `sections.py` | 全部 `_fmt_*` section 格式化（原 chat + memory） |
| `budget.py` | token/字符预算（整 section 清空） |
| `signals.py` | 判断信号叙事 |
| `tasks.py` / `skills.py` / `facts.py` | 任务、技能、事实快照 |

删除：`runtime.py` 及已并入 `utils`/`sections` 的子文件。对外仍通过 `core.judgment.context` 导入，无破坏性变更。

## Consequences

- 新增 section 格式化逻辑进 `sections.py`；共享工具进 `utils.py`。
- ADR 0015（LLM 感知完整性）约束不变。

## Validation

- `tests/test_judgment_layers.py`
- 全量 `pytest tests/`
