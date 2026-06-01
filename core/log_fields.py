"""core/log_fields.py — 结构化日志字段格式化（P3 可观测性）。

统一 run / task / tool / model_ref / tier / usage_source 等键名，
便于 grep 与日志聚合；不截断业务正文（正文仍走 summary / evidence）。
"""

from __future__ import annotations

from typing import Any


def _stringify_log_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.replace("\n", "\\n").strip()
        return text or None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return ",".join(str(item) for item in value)
    text = str(value).replace("\n", "\\n").strip()
    return text or None


def format_log_fields(**fields: Any) -> str:
    """按固定顺序输出 key=value 片段（跳过空值）。"""
    parts: list[str] = []
    for key, value in fields.items():
        text = _stringify_log_value(value)
        if text is None:
            continue
        parts.append(f"{key}={text}")
    return " ".join(parts)


def execution_scope_fields(
    *,
    run_id: int | None = None,
    task_id: int | None = None,
    tool: str | None = None,
    tier: str | None = None,
    worker: str | None = None,
    status: str | None = None,
    **extra: Any,
) -> str:
    """工具执行 / run 生命周期日志共用字段。"""
    return format_log_fields(
        run=run_id,
        task=task_id,
        tool=tool,
        tier=tier,
        worker=worker,
        status=status,
        **extra,
    )


def llm_call_fields(
    *,
    model_ref: str,
    tier: str,
    phase: str,
    usage_source: str | None = None,
    thinking: Any = None,
    attempt: int | None = None,
    **extra: Any,
) -> str:
    """LLM 调用成功/失败/重试日志共用字段。"""
    return format_log_fields(
        model_ref=model_ref,
        tier=tier,
        phase=phase,
        usage_source=usage_source,
        thinking=thinking,
        attempt=attempt,
        **extra,
    )


def tick_scope_fields(
    *,
    tick: int,
    task_id: int | None = None,
    decision: str | None = None,
    tool: str | None = None,
    **extra: Any,
) -> str:
    """主循环 tick 日志字段（判断结果 / 执行轨迹）。"""
    return format_log_fields(
        tick=tick,
        task=task_id,
        decision=decision,
        tool=tool,
        **extra,
    )


def judgment_outcome_fields(
    *,
    phase: str,
    tier: str,
    model_ref: str,
    thinking: Any = None,
    applied_skills: str | None = None,
    **extra: Any,
) -> str:
    """判断层单次 decide 结果日志字段。"""
    return format_log_fields(
        phase=phase,
        tier=tier,
        model_ref=model_ref,
        thinking=thinking,
        applied_skills=applied_skills,
        **extra,
    )
