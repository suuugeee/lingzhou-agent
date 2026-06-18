"""core/judgment/decision/rounds.py — 首轮与续判 LLM 编排（从 runtime 迁出）。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.cortex.actions import build_workbench_action
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
from core.judgment.tiers import (
    CONTINUE_PHASE,
    INITIAL_PHASE,
    REASONER_TIER,
    REPLY_ONLY_FALLBACK_TIER,
    REPLY_PHASE,
)
from core.log_fields import judgment_outcome_fields
from tools.registry import registry_has_tool

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
_ASSEMBLE_CONTEXT_ERROR_STATE: dict[str, Any] = {}
_THINKING_ORDER = {"off": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4}


def _thinking_floor(value: str | None, floor: str | None) -> str | None:
    if floor is None:
        return value
    if value is None:
        return floor
    return value if _THINKING_ORDER.get(value, -1) >= _THINKING_ORDER.get(floor, -1) else floor


def _track_assemble_context_failure(exc: BaseException) -> int:
    signature = f"{type(exc).__name__}:{exc}"
    previous = str(_ASSEMBLE_CONTEXT_ERROR_STATE.get("signature") or "")
    if previous == signature:
        count = int(_ASSEMBLE_CONTEXT_ERROR_STATE.get("count") or 0) + 1
    else:
        count = 1
    _ASSEMBLE_CONTEXT_ERROR_STATE["signature"] = signature
    _ASSEMBLE_CONTEXT_ERROR_STATE["count"] = count
    return count


def _assemble_context_failure_backoff_ms(repeat_count: int) -> int:
    if repeat_count <= 1:
        return 0
    return min(60_000, 2_000 * (repeat_count - 1))


def _can_build_workbench_recovery(active_task: Any | None, registry: Any | None) -> bool:
    return active_task is not None and registry is not None and registry_has_tool(registry, "task.workbench")


def _task_id_text(active_task: Any | None) -> str:
    return str(getattr(active_task, "id", "") or "-")


def _assemble_context_failure_output(
    *,
    exc: BaseException,
    repeat_count: int,
    judgment_signals: JudgmentSignals | None,
    hard_boundaries: list[str] | None,
    active_task: Any | None,
    registry: Any | None,
) -> JudgmentOutput:
    reason = f"上下文组装异常: {exc}"
    backoff_ms = _assemble_context_failure_backoff_ms(repeat_count)
    if _can_build_workbench_recovery(active_task, registry):
        task_id = _task_id_text(active_task)
        workbench = {
            "domain": "runtime-context",
            "intent": "恢复上下文组装异常并继续任务闭环",
            "evidence": [
                f"任务 {task_id} 的判断上下文组装失败: {type(exc).__name__}: {exc}",
                f"同一异常连续出现 {repeat_count} 次。",
            ],
            "hypothesis": "当前不行动不是模型放弃，而是运行时上下文材料构建失败，需要先定位异常来源。",
            "recovery_state": "recovering_from_context_assembly_failure",
            "next_verification": "读取最新异常栈或相关上下文组装实现，修复后重新触发同一任务验证判断链路恢复。",
            "completion_checks": [
                "已定位上下文组装异常的具体实现位置。",
                "已完成修复或规避，并确认下一轮不再因同一异常直接 wait。",
            ],
        }
        model_strategy: dict[str, Any] = {}
        if backoff_ms > 0:
            model_strategy["next_idle_gap_ms"] = backoff_ms
        return build_workbench_action(
            workbench=workbench,
            rationale=(
                "上下文组装异常不能只 wait；当前有活跃任务，先把异常转成任务皮层恢复态，"
                "保留下轮可执行验证入口。"
            ),
            model_strategy=model_strategy,
        )

    safe_output = _simulate_safe_output_fn(
        failure_count=0,
        signals=judgment_signals,
        hard_boundaries=hard_boundaries or [],
        reason=reason,
    )
    if backoff_ms > 0:
        safe_output.model_strategy["next_idle_gap_ms"] = backoff_ms
    return safe_output


def _llm_unavailable_output(
    *,
    err: str,
    active_task: Any | None,
    registry: Any | None,
    judgment_signals: JudgmentSignals | None = None,
    hard_boundaries: list[str] | None = None,
    reply_only: bool = False,
) -> JudgmentOutput:
    if reply_only:
        return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {err}")

    if _can_build_workbench_recovery(active_task, registry):
        task_id = _task_id_text(active_task)
        workbench = {
            "domain": "runtime-provider",
            "intent": "恢复 LLM/provider 不可用导致的任务中断",
            "evidence": [
                f"任务 {task_id} 判断阶段未拿到 LLM 输出。",
                f"provider/模型调用错误: {err}",
            ],
            "hypothesis": "当前任务不是自然完成或主动放弃，而是被模型/provider 可用性阻断。",
            "recovery_state": "recovering_from_llm_unavailable",
            "next_verification": "检查 provider 健康状态、路由模型与最近 LLM 错误；若 provider 恢复，继续执行原任务的下一步验证。",
            "completion_checks": [
                "已确认 LLM/provider 是否仍不可用。",
                "已记录或修复 provider/路由问题。",
                "provider 恢复后已回到原任务的下一步验证，而不是直接 wait/complete。",
            ],
        }
        return build_workbench_action(
            workbench=workbench,
            rationale=(
                "LLM 不可用不能被解释成任务无事可做；当前有活跃任务，"
                "先把 provider 故障写入任务皮层恢复态，保留下轮验证入口。"
            ),
            model_strategy={"next_idle_gap_ms": 2000},
        )

    return _simulate_safe_output_fn(
        failure_count=0,
        signals=judgment_signals,
        hard_boundaries=hard_boundaries or [],
        reason=err,
    )


def _log_assemble_context_failure(exc: BaseException, repeat_count: int) -> None:
    if repeat_count == 1:
        _log.error(
            "[judgment] _assemble_context() 异常，返回 wait 兜底: %s",
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    if repeat_count & (repeat_count - 1) == 0:
        _log.warning(
            "[judgment] _assemble_context() 异常重复 x%d，继续 wait 兜底: %s",
            repeat_count,
            exc,
        )


@dataclass(slots=True)
class JudgmentRoundDeps:
    assembler: JudgmentContextAssembler
    executor: JudgmentExecutor
    cfg: Config


def _sync_prompt_capsule(deps: JudgmentRoundDeps) -> None:
    capsule = str(getattr(deps.executor, "_last_prompt_capsule", "") or "").strip()
    deps.assembler._last_context_compression_capsule = capsule


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
    phase: str = INITIAL_PHASE,
    registry_override: Any | None = None,
    runtime_life_snapshot: dict[str, Any] | None = None,
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
            runtime_life_snapshot=runtime_life_snapshot,
        )
    except Exception as ctx_exc:
        repeat_count = _track_assemble_context_failure(ctx_exc)
        _log_assemble_context_failure(ctx_exc, repeat_count)
        return _assemble_context_failure_output(
            exc=ctx_exc,
            repeat_count=repeat_count,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries,
            active_task=active_task,
            registry=registry_override or deps.assembler._registry,
        )

    # 上下文组装成功：清零聚合状态，确保下一波相同异常能重新触发 error 日志
    if _ASSEMBLE_CONTEXT_ERROR_STATE:
        _ASSEMBLE_CONTEXT_ERROR_STATE.clear()
    deps.assembler._last_context_text = context_text
    deps.assembler._last_context_compression_capsule = ""
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
    _sync_prompt_capsule(deps)
    if raw is None:
        err = str(llm_error) or repr(llm_error) if llm_error is not None else "unknown error"
        return _llm_unavailable_output(
            err=err,
            active_task=active_task,
            registry=registry_override or deps.assembler._registry,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries or [],
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
    phase = REPLY_PHASE if reply_only else CONTINUE_PHASE
    forced_prefer_tier = REPLY_ONLY_FALLBACK_TIER if reply_only else prefer_tier
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
    if selection.tier == REASONER_TIER:
        if active_task is not None:
            resolved_thinking = _thinking_floor(resolved_thinking, "medium")
        elif user_message:
            resolved_thinking = _thinking_floor(resolved_thinking, "low")
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
        fallback_prefer_tier=REPLY_ONLY_FALLBACK_TIER if reply_only else None,
        log_prefix="[judgment.continue]",
        skills=deps.executor._last_call_meta.get("skills") or "none",
    )
    _sync_prompt_capsule(deps)
    if raw is None:
        if llm_error is not None:
            err = str(llm_error) or repr(llm_error)
        else:
            err = "LLM returned None"
        return _llm_unavailable_output(
            err=err,
            active_task=active_task,
            registry=deps.assembler._registry,
            reply_only=reply_only,
        )

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
