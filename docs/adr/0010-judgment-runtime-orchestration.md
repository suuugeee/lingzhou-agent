# ADR 0010: judgment 编排迁出 runtime

- Status: Accepted
- Date: 2026-06-01

## Context

`JudgmentLayer.decide` / `decide_continue` 与 `CognitionFrame` 同处 `runtime.py`，assembler 在 TYPE_CHECKING 与运行时反向依赖 `runtime`，不利于继续拆分 executor/assembler。

## Decision

- `core/judgment/frame.py`：`CognitionFrame` 独立模块。
- `core/judgment/decision/rounds.py`：`JudgmentRoundDeps`、`decide_initial`、`decide_continue`、`finalize_continue_output`。
- `core/judgment/runtime.py`：仅保留 `JudgmentLayer` 薄 façade，委托 `rounds`。
- 公开 API 不变：`from core.judgment import CognitionFrame, JudgmentLayer`。

## Validation

- `tests/test_judgment_layers.py`
- `tests/test_judgment_ctx.py` 与 `tests/test_core.py` 中 judgment 相关用例
