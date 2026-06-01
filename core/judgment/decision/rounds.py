"""core/judgment/decision/rounds.py — 首轮与续判 LLM 编排（从 runtime 迁出）。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.judgment.boundary import (
    coerce_reply_only_output as _coerce_reply_only_output_fn,
)
from core.judgment.boundary import (
    normalize_judgment_output as _normalize_judgment_output_fn,
)
from core.judgment.boundary import (
    simulate_safe_output as _simulate_safe_output_fn,
)
from core.judgment.frame import CognitionFrame
from core.judgment.output import JudgmentOutput, ModelSelection
from core.log_fields import judgment_outcome_fields

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment.assembler import JudgmentContextAssembler
    from core.judgment.executor import JudgmentExecutor
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore

_log = logging.getLogger("lingzhou.judgment")


@dataclass(slots=True)
class JudgmentRoundDeps:
    assembler: JudgmentContextAssembler
    executor: JudgmentExecutor
    cfg: Config


def finalize_continue_output(
    deps: JudgmentRoundDeps,
    output: JudgmentOutput,
    *,
    reply_only: bool,
    tool_history: list[dict[str, Any]],
    selection: ModelSelection,
) -> JudgmentOutput:
    if reply_only:
        output = _coerce_reply_only_output_fn(output)
    applied = ",".join(output.applied_skills) if output.applied_skills else "none"
    if output.applied_skills:
        deps.assembler._last_applied_skill_names = list(output.applied_skills)

    _log.info(
        "[judgment.continue] round=%d phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s",
        len(tool_history),
        selection.phase,
        selection.tier,
        selection.model_ref,
        deps.executor._last_call_meta["thinking"],
        applied,
        output.decision,
        output.action_label(),
    )
    return output


async def decide_initial(
    deps: JudgmentRoundDeps,
    frame_or_percept: CognitionFrame | Percept,
    wm: WorkingMemory | None = None,
    task_store: TaskStore | None = None,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    emotion: EmotionState | None = None,
    *,
    active_task: Any | None = None,
    user_message: str = "",
    chat_id: str | None = None,
    ethos_state: EthosState | None = None,
    judgment_signals: JudgmentSignals | None = None,
    hard_boundaries: list[str] | None = None,
    perception_replay: PerceptionReplaySummary | None = None,
    cognitive_signals: CognitiveSignals | None = None,
    thinking_override: str | None = None,
    prefer_tier: str | None = None,
    routing_overrides: dict[str, str] | None = None,
    phase: str = "initial",
    registry_override: Any | None = None,
) -> JudgmentOutput:
    from core.judgment.context.utils import _clear_context_cache

    percept, wm, task_store, episodic, semantic, emotion = deps.assembler._coerce_frame_args(
        frame_or_percept,
        wm,
        task_store,
        episodic,
        semantic,
        emotion,
    )
    try:
        deps.assembler._context_cache.clear()
        _clear_context_cache()
        context_text = await deps.assembler._assemble_context(
            percept,
            wm,
            task_store,
            episodic,
            semantic,
            emotion,
            active_task=active_task,
            user_message=user_message,
            chat_id=chat_id,
            ethos_state=ethos_state,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
            phase=phase,
            current_action="",
            tool_history=None,
            effective_thinking=thinking_override or deps.cfg.thinking,
            routing_overrides=routing_overrides,
            registry_override=registry_override,
        )
    except Exception as ctx_exc:
        _log.exception("[judgment] _assemble_context() 异常，返回 wait 兜底: %s", ctx_exc)
        return _simulate_safe_output_fn(
            failure_count=0,
            signals=judgment_signals,
            hard_boundaries=hard_boundaries or [],
            reason=f"上下文组装异常: {ctx_exc}",
        )

    deps.assembler._last_context_text = context_text
    messages = deps.assembler._build_messages(context_text)

    selected_provider, selection = deps.executor._select_provider(
        phase=phase,
        user_message=user_message,
        prefer_tier=prefer_tier,
        thinking_override=thinking_override,
        routing_overrides=routing_overrides,
    )
    primary = deps.assembler._last_selected_skills[0] if deps.assembler._last_selected_skills else None
    raw, selection, llm_error = await deps.executor._chat_with_retry(
        selected_provider=selected_provider,
        selection=selection,
        messages=messages,
        phase=phase,
        user_message=user_message,
        thinking_override=thinking_override,
        routing_overrides=routing_overrides,
        log_prefix="[judgment]",
        skills=(
            ",".join(skill.name for skill in deps.assembler._last_selected_skills[:3])
            if deps.assembler._last_selected_skills
            else "none"
        ),
        primary_skill_name=primary.name if primary else None,
        primary_skill_guidance=bool(primary and getattr(primary, "guidance", None)),
    )
    if raw is None:
        err = str(llm_error) or repr(llm_error) if llm_error is not None else "unknown error"
        return _simulate_safe_output_fn(
            failure_count=0,
            signals=judgment_signals,
            hard_boundaries=hard_boundaries or [],
            reason=err,
        )

    output = JudgmentOutput.from_llm(raw)
    output = await _normalize_judgment_output_fn(
        deps.executor,
        output,
        context_text=context_text,
        raw=raw,
        record_parse_failure=task_store.record_failure,
        registry=registry_override or deps.assembler._registry,
        allow_delegate_tasks=True,
    )
    applied = ",".join(output.applied_skills) if output.applied_skills else "none"
    if output.applied_skills:
        deps.assembler._last_applied_skill_names = list(output.applied_skills)
    _log.info(
        "[judgment] %s decision=%s action=%s rationale=%s",
        judgment_outcome_fields(
            phase=selection.phase,
            tier=selection.tier,
            model_ref=selection.model_ref,
            thinking=selection.thinking,
            applied_skills=applied,
        ),
        output.decision,
        output.action_label(),
        output.rationale or "",
    )
    return output


async def decide_continue(
    deps: JudgmentRoundDeps,
    tool_history: list[dict],
    *,
    user_message: str = "",
    active_task: Any | None = None,
    prefer_tier: str | None = None,
    thinking_override: str | None = None,
    routing_overrides: dict[str, str] | None = None,
    reply_only: bool = False,
    wm_delta: list[dict[str, Any]] | None = None,
    speech_intent: str = "",
    action_result: Any | None = None,
    emotion_state: dict[str, Any] | None = None,
) -> JudgmentOutput:
    if not deps.assembler._last_context_text:
        return JudgmentOutput.wait(reason="[inner-loop] no cached context for continuation")

    continuation_context = deps.assembler._build_continue_context(
        tool_history,
        user_message=user_message,
        reply_only=reply_only,
        wm_delta=wm_delta,
        speech_intent=speech_intent,
        action_result=action_result,
        emotion_state=emotion_state,
    )
    messages = deps.assembler._build_messages(continuation_context)

    current_action = "" if reply_only else str(tool_history[-1].get("tool", "")) if tool_history else ""
    phase = "reply" if reply_only else "continue"
    forced_prefer_tier = "reasoner" if reply_only else prefer_tier
    selected_provider, selection = deps.executor._select_provider(
        phase=phase,
        user_message=user_message,
        current_action=current_action,
        tool_history=tool_history,
        prefer_tier=forced_prefer_tier,
        thinking_override=thinking_override,
        routing_overrides=routing_overrides,
    )
    resolved_thinking = thinking_override
    if resolved_thinking is None and selection.tier == "reasoner" and user_message:
        resolved_thinking = "low"
    raw, selection, llm_error = await deps.executor._chat_with_retry(
        selected_provider=selected_provider,
        selection=selection,
        messages=messages,
        phase=phase,
        user_message=user_message,
        current_action=current_action,
        tool_history=tool_history,
        thinking_override=resolved_thinking,
        routing_overrides=routing_overrides,
        fallback_prefer_tier="reasoner" if reply_only else None,
        log_prefix="[judgment.continue]",
        skills=deps.executor._last_call_meta.get("skills") or "none",
    )
    if raw is None:
        if llm_error is not None:
            return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {llm_error!r}")
        return JudgmentOutput.wait(reason="[inner-loop] LLM returned None")

    output = JudgmentOutput.from_llm(raw)
    output = await _normalize_judgment_output_fn(
        deps.executor,
        output,
        context_text=continuation_context,
        raw=raw,
        registry=deps.assembler._registry,
    )
    return finalize_continue_output(
        deps,
        output,
        reply_only=reply_only,
        tool_history=tool_history,
        selection=selection,
    )
