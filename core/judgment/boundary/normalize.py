"""core/judgment/boundary/normalize.py — 判断输出边界归一化（纯函数）。

职责：
- 对 JudgmentOutput 做纯函数级别的修正（不依赖任何 LLM 调用）
- reply_only 模式强制、动作形态归一化等

与 JudgmentLayer 解耦：不知道上下文如何组装，不持有任何 provider 引用。
需要调用 LLM 进行修复（_repair_output）的逻辑保留在 executor.py。
记忆表述由 LLM 结合 context 中的 recall_mode 自行判断，不在此层做正则改写。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.judgment.output import JudgmentOutput

if TYPE_CHECKING:
    from core.perception import JudgmentSignals
    from tools.registry import ToolRegistry

_REPLY_PSEUDO_TOOLS = {"chat_reply"}


def simulate_safe_output(
    failure_count: int,
    signals: JudgmentSignals | None,
    hard_boundaries: list[str],
    reason: str = "",
) -> JudgmentOutput:
    """LLM 不可用时的确定性回退。行为原则：posture > wait。"""
    if signals and signals.posture in ("pause", "narrow"):
        return JudgmentOutput.wait(reason=f"[fallback] posture={signals.posture}, LLM 不可用: {reason}")
    return JudgmentOutput.wait(reason=f"[fallback] LLM 不可用: {reason}")


def coerce_reply_only_output(output: JudgmentOutput) -> JudgmentOutput:
    """将 continue 续判结果强制修正为 reply_only 模式（禁止 act，必须有 reply_to_user）。"""
    if not output.reply_to_user.strip():
        fallback = str(output.rationale or output.speech_intent or output.next_step or "").strip()
        if not fallback:
            fallback = "本轮执行已完成，下一步按既定任务目标继续推进。"
        else:
            fallback = f"已完成本轮执行：{fallback}"
        return JudgmentOutput(
            decision=output.decision if output.decision in {"pause", "wait"} else "wait",
            chosen_action_id="",
            params={},
            rationale=output.rationale,
            reply_to_user=fallback,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
        )
    return JudgmentOutput(
        decision=output.decision if output.decision in {"pause", "wait"} else "wait",
        chosen_action_id="",
        params={},
        rationale=output.rationale,
        reflection=output.reflection,
        reply_to_user=output.reply_to_user,
        next_step=output.next_step,
        model_strategy=dict(output.model_strategy or {}),
    )


def normalize_reply_pseudo_tool(output: JudgmentOutput) -> JudgmentOutput:
    """将误写成工具的直接回复动作归一化回 reply 链路。"""
    tool_name = str(output.chosen_action_id or "").strip().lower()
    if output.decision != "act" or tool_name not in _REPLY_PSEUDO_TOOLS:
        return output

    reply = str(output.reply_to_user or output.speech_intent or "").strip()
    if not reply:
        return JudgmentOutput.wait(reason=f"伪工具 {tool_name!r} 缺少 reply_to_user")

    return JudgmentOutput(
        decision="wait",
        chosen_action_id="",
        params={},
        rationale=output.rationale,
        reflection=output.reflection,
        speech_intent=output.speech_intent,
        reply_to_user=reply,
        next_step=output.next_step,
        model_strategy=dict(output.model_strategy or {}),
        applied_skills=list(output.applied_skills or []),
    )


def normalize_action_shape(
    output: JudgmentOutput,
    *,
    registry: ToolRegistry | None = None,
    allow_delegate_tasks: bool = False,
) -> JudgmentOutput:
    """Normalize model action shape before it reaches execution.

    The executor should only see actionable tool ids. Whitespace-only tool
    names, invalid decisions, and stale/unregistered tool names are judgment
    boundary failures, not executable actions.
    """
    output.decision = str(output.decision or "wait").strip().lower()
    output.chosen_action_id = str(output.chosen_action_id or "").strip()
    if not isinstance(output.params, dict):
        output.params = {}

    if output.decision not in {"act", "pause", "wait"}:
        return JudgmentOutput(
            decision="wait",
            rationale=f"无效 decision: {output.decision!r}",
            reflection=output.reflection,
            reply_to_user=output.reply_to_user,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            params={},
        )

    if output.decision != "act":
        return JudgmentOutput(
            decision=output.decision,
            chosen_action_id="",
            params={},
            rationale=output.rationale,
            reflection=output.reflection,
            speech_intent=output.speech_intent,
            reply_to_user=output.reply_to_user,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            applied_skills=list(output.applied_skills or []),
        )

    if output.parallel_actions:
        normalized_parallel: list[dict[str, Any]] = []
        unknown_parallel: list[str] = []
        for item in output.parallel_actions:
            action_id = str(item.get("action_id") or "").strip()
            if not action_id:
                continue
            if registry is not None and registry.get(action_id) is None:
                unknown_parallel.append(action_id)
                continue
            normalized = dict(item)
            normalized["action_id"] = action_id
            if not isinstance(normalized.get("params"), dict):
                normalized["params"] = {}
            normalized_parallel.append(normalized)
        output.parallel_actions = normalized_parallel
        if unknown_parallel and not normalized_parallel and not output.chosen_action_id:
            return JudgmentOutput.wait(reason=f"未知并行动作: {', '.join(repr(name) for name in unknown_parallel)}")

    has_parallel_actions = bool(output.parallel_actions)
    has_delegate_tasks = bool(output.delegate_tasks)
    if not output.chosen_action_id:
        if has_parallel_actions:
            return output
        if has_delegate_tasks and allow_delegate_tasks:
            return output
        return JudgmentOutput(
            decision="wait",
            rationale="act 决策缺少 chosen_action_id",
            speech_intent=output.speech_intent,
            reply_to_user=output.reply_to_user,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            params={},
        )

    if registry is not None and registry.get(output.chosen_action_id) is None:
        return JudgmentOutput(
            decision="wait",
            rationale=f"未知工具: {output.chosen_action_id!r}",
            speech_intent=output.speech_intent,
            reply_to_user=output.reply_to_user,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            params={},
        )

    return output
