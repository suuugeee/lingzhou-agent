# ADR 0011: JudgmentExecutor mixin 拆分

- Status: Accepted
- Date: 2026-06-01

## Context

`core/judgment/executor.py` 约 545 行，混合 tier 路由、模型健康冷却、prompt 超限处理与 LLM 调用入口，不利于单测与阅读。

## Decision

- `decision/routing_mixin.py`：`ExecutorRoutingMixin`（tier 候选、provider 解析、成本/延迟标签）。
- `decision/health_mixin.py`：`ExecutorHealthMixin`（错误分类、冷却、可用性）。
- `decision/prompt_mixin.py`：`ExecutorPromptMixin`（prompt 正则、压缩、trim 委托 helpers）。
- `executor.py`：仅保留 `JudgmentExecutor` 构造、`set_routing_providers`、`_select_provider` / `_chat_with_retry` / `_repair_output` 委托。

## Validation

- `tests/test_judgment_layers.py`（mixin 组合断言）
- `tests/test_judgment_ctx.py` 与 judgment 相关 `test_core` 用例
