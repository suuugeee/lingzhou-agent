# ADR 0021: judgment.context 取消聚合导出

- Status: Accepted
- Date: 2026-06-01
- Related: [0016](0016-context-package-consolidation.md)、[0020](0020-loop-subpackage-layout.md)

## Context

`core/judgment/context/__init__.py` 曾 re-export 全部 `_fmt_*` / `apply_context_budget`，与 loop 子包「直引、无 barrel」原则不一致。

## Decision

- `context/__init__.py` 仅保留包说明，`__all__ = []`。
- 生产代码从子模块导入，例如 `core.judgment.context.sections`、`core.judgment.context.utils`。
- 包级公开 API 若需 `apply_context_budget`，由 `core.judgment` 从 `context.budget` 导出（judgment 包门面，非 context 聚合）。

## Validation

- `tests/test_import_boundaries.py::test_judgment_context_package_does_not_aggregate_exports`
- 全量 `pytest tests/`
