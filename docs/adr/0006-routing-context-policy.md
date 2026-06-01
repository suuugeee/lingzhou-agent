# ADR 0006: 模型路由上下文策略下沉

- Status: Accepted
- Date: 2026-06-01

## Context

`assembler/model_routing.py` 内联工具历史统计、cost posture 与 continue 压缩阈值，策略与 JSON 组装耦合，难以单测与复用。

## Decision

- 新增 `core/judgment/policy/routing_context.py`：
  - `analyze_tool_history_budget`
  - `routing_posture`
  - `continue_phase_policy_payload`（复用 `tool_history_compact_limits`）
- `model_routing.py` 仅负责 catalog/executor 数据收集与 JSON 序列化。

## Validation

- `tests/test_judgment_layers.py` 中 routing 策略用例；judgment 测试套件通过。
