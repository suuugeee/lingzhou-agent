# ADR 0003: Provider 能力查询门面

- Status: Accepted
- Date: 2026-06-01

## Context

`tools/image.py` 直接依赖 `provider.catalog` 的 `model_supports` / `find_model_ref_for_capability`，工具与目录实现耦合。

## Decision

- 新增 `provider/capabilities.resolve_model_ref_for_input`，封装能力解析与回退。
- `tools/image.py` 仅调用该门面；catalog 细节留在 provider 包内。

## Validation

- `image.analyze` 路由行为不变；`tests/test_boundary_contracts.py` 覆盖门面逻辑。
