# ADR 0009: 兼容 shim 退役计划

- Status: Accepted（shim 已于 2026-06-01 删除）
- Date: 2026-06-01

## Context

分层重构后曾保留旧 import 路径；仓库内测试与生产代码已全部迁至 canonical 路径。

## 已删除 shim（勿再引用）

| 原路径 | 改用 |
|--------|------|
| `core/judgment/parser.py` | `core.judgment.boundary` |
| `core/judgment/executor_helpers.py` | `core.judgment.decision.helpers` |
| `core/probe/types.py` | `core.contracts.probe` |
| `core/execution_helpers.py` | `core.execution.helpers` 或 `core.execution` |
| `core/worker.py` | `core.execution.workers` 或 `core.execution.WorkerLayer` |

## 规则

- 新代码只使用上表「改用」列路径。
- `tests/test_import_boundaries.py::test_compat_shim_files_removed` 防止 shim 文件回潮。

## Validation

- 全量 `pytest tests/` + `core.import_boundary_check`
