# ADR 0013: context 分区拆分与 CI 全量门禁

- Status: Accepted
- Date: 2026-06-01

## Context

`context/runtime.py` 仍超 400 行；Ruff 仅覆盖治理面子集；缺少覆盖率回归。

## Decision

- `context/chat_sections.py`：对话/连续性格式化。
- `context/memory_sections.py`：WM、记忆、工具、配置快照、感知/ethos。
- `context/runtime.py`：聚合 re-export，保持 `from .runtime import _fmt_*` 路径稳定。
- CI：`ruff check core tools`；全量 pytest 后对 `core/judgment`、`core/execution`、`core/contracts` 要求 `--cov-fail-under=80`（全量测试下已验证）。
- `core/loop/tick.py` 保留兼容 re-export；Ruff `F401` 对该文件忽略。

## Validation

- `tests/test_judgment_layers.py`
- 全量 `pytest tests/`
