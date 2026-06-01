# ADR 0004: judgment 分层包（boundary / decision / policy）

- Status: Accepted
- Date: 2026-06-01

## Context

`core/judgment` 将解析归一化、LLM 调用与阈值策略混在同一扁平目录，不利于边界治理与测试。

## Decision

- `boundary/`：输出归一化（`normalize.py`）与解析后流水线（`pipeline.py`）。
- `decision/`：LLM 路由、重试实现（自 `executor_helpers` 迁入 `helpers.py`）。
- `policy/`：可配置阈值策略（首期：`continue_history.tool_history_compact_limits`）。
- `parser.py`、`executor_helpers.py` 保留为兼容 re-export。

## Validation

- `tests/test_judgment_layers.py`；既有 judgment 测试套件通过。
