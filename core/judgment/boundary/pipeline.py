"""判断输出边界流水线：解析修复 + 形态归一化。"""
from __future__ import annotations

from typing import Any

from core.judgment.boundary.normalize import normalize_action_shape, normalize_reply_pseudo_tool
from core.judgment.output import JudgmentOutput


async def normalize_judgment_output(
    executor: Any,
    output: JudgmentOutput,
    *,
    context_text: str,
    raw: str,
    record_parse_failure: Any | None = None,
    registry: Any | None = None,
    allow_delegate_tasks: bool = False,
) -> JudgmentOutput:
    """在输出进入执行层前完成边界校验与归一化。"""
    if output.rationale.startswith("LLM 输出解析失败"):
        repaired = await executor._repair_output(context_text, raw)
        if repaired is not None:
            output = repaired
        elif record_parse_failure is not None:
            await record_parse_failure("judgment_parse", output.rationale)

    output = normalize_reply_pseudo_tool(output)
    return normalize_action_shape(
        output,
        registry=registry,
        allow_delegate_tasks=allow_delegate_tasks,
    )
