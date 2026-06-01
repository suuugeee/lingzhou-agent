# ADR 0007: 结构化日志字段统一

- Status: Accepted
- Date: 2026-06-01

## Context

执行层、run 收尾与 LLM 调用日志字段命名不一致（有的用 `run=%s` 位置参数，有的缺 `task`/`run`），检索成本高。

## Decision

- 新增 `core/log_fields.py`：`format_log_fields`、`execution_scope_fields`、`llm_call_fields`。
- 约定键名：`run`、`task`、`tool`、`tier`、`worker`、`status`、`model_ref`、`usage_source`（值为空则省略）；布尔值输出 `true`/`false`。
- `tool_tier_mapping` 与 capability 映射按 `registry_manifest_signature` 缓存（`core/judgment/output.py`）。
- `core/execution/`（`layer`、`helpers.finalize_run`）、`core/judgment/decision/helpers`（含 LLM 失败/重试）、`core/judgment/runtime` 结果日志改用上述 helper。
- 不截断 `summary` / `evidence`；短摘要仍优先 `metadata.log_summary`（ADR 0005）。

## Validation

- `tests/test_log_fields.py`；既有 `test_tool_result_log_fields_*` / `test_run_progress_text_*` 不变。
