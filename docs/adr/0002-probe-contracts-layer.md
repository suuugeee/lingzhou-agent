# ADR 0002: 探针类型契约层

- Status: Accepted
- Date: 2026-06-01

## Context

`tools/probe.py` 依赖 `core.probe.types`，工具层耦合探针运行时子包内部模块。

## Decision

- 新增 `core/contracts/probe.py` 承载 `ProbeConfig` 等稳定类型。
- `core/probe/types.py` 仅 re-export，保持 core 内部兼容。
- `tools/probe.py` 与 judgment context 改从 `core.contracts.probe` 导入。

## Validation

- 探针相关测试通过；`core/probe/*` 与 `tools/probe.py` 均从 `core.contracts.probe` 导入类型。
