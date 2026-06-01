# ADR 0012: judgment context 预算与信号模块拆分

- Status: Accepted
- Date: 2026-06-01

## Context

`core/judgment/context/runtime.py` 超过 500 行，混合聊天/记忆 section 格式化、token 预算裁剪与判断信号格式化，不利于定位与单测。

## Decision

- `context/budget.py`：`apply_context_budget`。
- `context/signals.py`：`_fmt_judgment_signals`、`_fmt_hard_boundaries`、`_fmt_perception_replay`。
- `context/__init__.py` 保持 `core.judgment.context` 稳定导出不变。

## Validation

- `tests/test_judgment_ctx.py`、`tests/test_core.py` 中 `apply_context_budget` 用例
- `tests/test_judgment_layers.py` 增加子模块 import 断言
