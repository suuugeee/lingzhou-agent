# ADR 0017: loop tick 归并为 `core/loop/tick/` 包

- Status: Accepted
- Date: 2026-06-01

## Context

`core/loop/` 同时存在 `tick.py` + `tick_prep.py` / `tick_exec.py` / `tick_memory.py` / `tick_types.py` 与目录 `tick/{prep,exec,memory,types}.py` **两套完整副本**（约 1800 行重复），命名 `tick_xxxx` 散落根目录，难以维护。

## Decision

- **唯一实现**：`core/loop/tick/` 包
  - `__init__.py` — 编排入口、口腔回复、`_tick_impl`、测试用 re-export
  - `prep.py` — 感知 + 判断准备
  - `exec.py` — 执行 + continue + 收尾
  - `memory.py` — post-tick 记忆结晶
  - `types.py` — 共享类型与 `_log`
- **删除** 根目录 `tick.py`、`tick_prep.py`、`tick_exec.py`、`tick_memory.py`、`tick_types.py`
- 对外路径不变：`from core.loop.tick import _tick_impl`、测试 patch `core.loop.tick.*`

## Validation

- `tests/test_cognition.py`、`tests/test_oral_and_brain.py`、全量 pytest
