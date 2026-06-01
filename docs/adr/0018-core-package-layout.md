# ADR 0018: core 根目录模块归并

- Status: Accepted
- Date: 2026-06-01

## Context

`core/` 根目录堆积 `tick_*`、`config_models_*`、`skill_helpers`、`subagent_task_store_view` 等与已有子包重复或命名冗长，和 [ADR 0017](0017-loop-tick-package-consolidation.md) 同类问题。

## Decision

### 归并到子包（删除根目录同名 `.py`）

| 原路径 | 新路径 |
|--------|--------|
| `config.py` | `config/loader.py` + `config/__init__.py` 导出 `Config` |
| `config_models_*.py` | `config_models/{base,runtime,advanced}.py` |
| `skill.py` + `skill_helpers.py` | `skill/__init__.py` + `skill/helpers.py` |
| `subagent.py` + `subagent_task_store_view.py` | `subagent/__init__.py` + `subagent/task_store_view.py` |
| `soul.py` / `self_model.py` | `persona/soul.py` / `persona/self_model.py` |
| `behavior_tracker.py` 等 | `loop/behavior.py`、`loop/run_refresh.py`、`loop/self_drive.py` |
| `task_runtime.py` / `task_parallel.py` | `loop/task/{runtime,parallel}.py`（见 [ADR 0019](0019-loop-task-package.md)） |
| `paths.py` / `plugin.py` | `paths/`、`plugin/` 包 |
| `loop/runtime/helpers_*.py` | `loop/runtime/chain.py`、`loop/runtime/memory_hooks.py` |

### 对外 import（canonical，无聚合 re-export）

- `from core.config import Config`（仅此与 `config_reference_defaults`；**不**从 `core.config` 导入子模型）
- `from core.config_models import ThresholdsConfig` 等子模型
- `from core.skill import SkillRegistry`
- `from core.subagent import make_subagent_runner`
- `from core.loop import CognitionLoop`（实现位于 `core.loop.runtime.main`）
- `from core.persona.soul import SoulManager`；`from core.loop.drive.behavior import BehaviorTracker`
- `from core.loop.task.runtime import …`；`from core.loop.task.parallel import run_tasks_parallel`
- 仓库根路径：`from core.paths import project_root`（禁止在业务代码用 `Path(__file__).parents[n]` 猜项目根）

### 仍留在 `core/` 根目录

`__init__.py`、`log_fields.py`、`import_boundary_check.py`、`smoke_tests.py`、`version.py` 等横切/门禁文件。

## Validation

- 全量 `pytest tests/`
- `ruff check core tools`
