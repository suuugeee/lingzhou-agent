# ADR 0014: context 工具模块拆分

- Status: Accepted
- Date: 2026-06-01

## Context

`context/utils.py` 混合 LRU 缓存、token 压缩、模板填充与 schema 校验（~220 行），覆盖率偏低且职责不清。

## Decision

| 模块 | 职责 |
|------|------|
| `cache.py` | `_context_fmt_cache`、`_cache_put`、`_clear_context_cache` |
| `tokens.py` | `_estimate_tokens`、段落拆分与 `_compress_*` |
| `template.py` | `_fill_template`、`_format_fact_value`、`_run_summary`、`_clip_text` |
| `schema.py` | `_CONTEXT_SCHEMA_KEYS`、`_validate_context_schema` |
| `utils.py` | 聚合 re-export，保持 `from core.judgment.context import _fill_template` 等路径稳定 |

子模块（`budget`、`chat_sections` 等）可继续 `from .utils import …` 或直连子包；对外 `context.__init__` 不变。

## Validation

- `tests/test_judgment_ctx.py`（含 `_fill_template`）
- `tests/test_judgment_layers.py` 子模块 import 断言
- 全量 pytest + 治理包覆盖率 ≥80%
