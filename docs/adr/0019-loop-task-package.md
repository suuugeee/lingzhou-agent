# ADR 0019: loop 任务子包归并

- Status: Accepted
- Date: 2026-06-01
- Related: [0017](0017-loop-tick-package-consolidation.md)、[0018](0018-core-package-layout.md)

## Context

`core/loop/` 根目录同时存在 `task_runtime.py` 与 `task_parallel.py`，与已归并的 `tick/` 子包并列，命名冗长且职责边界不清晰。

## Decision

- **唯一实现**：`core/loop/task/{runtime,parallel}.py`
- **删除** 根目录 `task_runtime.py`、`task_parallel.py`
- `task/__init__.py` 仅文档说明，**不做** 符号重导出
- 调用方直引子模块，例如：
  - `from core.loop.task.runtime import _consume_task_runtime_hints, _sync_task_progress_state`
  - `from core.loop.task.parallel import run_tasks_parallel`

## Validation

- 全量 `pytest tests/`
- `tests/test_import_boundaries.py::test_compat_shim_files_removed` 含旧路径
