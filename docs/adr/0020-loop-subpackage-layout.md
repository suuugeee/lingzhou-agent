# ADR 0020: core.loop 子包分层

- Status: Accepted
- Date: 2026-06-01
- Related: [0017](0017-loop-tick-package-consolidation.md)、[0019](0019-loop-task-package.md)、[0018](0018-core-package-layout.md)

## Context

`core/loop/` 在 `tick/`、`task/` 归并后仍残留十余个扁平 `.py`（driver、startup、common 等），职责混杂，与「按子域直引、无聚合 shim」不一致。

## Decision

`core/loop/` 仅保留 `__init__.py`（导出 `CognitionLoop`）与子包：

| 子包 | 模块 | 职责 |
|------|------|------|
| `runtime/` | `main`, `chain`, `memory_hooks`, `startup`, `reload` | 循环实例、链状态、启动/热重载 |
| `tick/` | `prep`, `exec`, `memory`, `types`, `__init__` | 单轮 tick 编排 |
| `task/` | `runtime`, `parallel` | 任务 hint 与并行委派 |
| `cycle/` | `driver`, `dispatcher`, `chat`, `focus` | 事件驱动调度与焦点 |
| `runs/` | `driver`, `refresh` | Run 路由与运行中刷新 |
| `shared/` | `common`, `logging`, `continue_phase`, `postprocess`, `progress` | tick 共享 helper（`continue` 为关键字，模块名用 `continue_phase`） |
| `drive/` | `behavior`, `self_drive` | 行为统计与自驱力 |

- 各子包 `__init__.py` **仅文档**，不重导出符号。
- 删除 `loop/` 根目录下上述旧扁平文件。
- 对外仍可用 `from core.loop import CognitionLoop`；其余一律直引，例如 `from core.loop.cycle.focus import resolve_focus_task`。

## Validation

- 全量 `pytest tests/`
- `tests/test_import_boundaries.py::test_compat_shim_files_removed`
