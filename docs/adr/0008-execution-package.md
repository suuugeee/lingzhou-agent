# ADR 0008: core/execution 包化

- Status: Accepted
- Date: 2026-06-01

## Context

执行层逻辑分散在 `execution.py`、`execution_helpers.py`、`worker.py`，与 judgment 分包后的结构不一致，且 `policy/routing_context` 为使用 `action_key_param` 反向依赖整包 execution。

## Decision

- 新建 `core/execution/`：`layer.py`（ExecutionLayer）、`helpers.py`、`workers.py`（WorkerLayer）。
- 包根 `core/execution/__init__.py` 为稳定导出；删除顶层 `core/execution.py` 模块文件（仅保留包）。
- ~~`core/execution_helpers.py`、`core/worker.py` shim~~ 已删除，见 [ADR 0009](0009-compat-shim-retirement.md)。
- `action_key_param` 迁至 `core/contracts/execution.py`；judgment policy 与 loop 从 contracts 引用。

## Validation

- `tests/test_boundary_contracts.py` 扩展 import 边界用例
- `tests/test_core.py` / `tests/test_tools.py` 中 execution 相关用例
