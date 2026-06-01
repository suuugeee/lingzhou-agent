# ADR 0001: Smoke 与工具注册公开 API

- Status: Accepted
- Date: 2026-06-01

## Context

`core/smoke_tests.py` 直接 import `tools.shell._check_risky` 与 `tools.registry._registry`，演化 smoke 与工具重构易碎。

## Decision

- `tools.shell.check_command_risk` 作为公开契约；`_check_risky` 保留为内部别名。
- `tools.registry.lookup_registered_tool` 作为查询已注册 `@tool` 的公开入口。

## Validation

- `core/smoke_tests.py` 片段改用上述 API。
- `tests/test_boundary_contracts.py` 覆盖行为不变。
