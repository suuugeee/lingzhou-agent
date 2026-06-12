"""core.loop.tick — tick 编排（prep / exec / memory / types）与对外稳定导入。"""

from __future__ import annotations

import json
import re
from typing import Any

from core.judgment import JudgmentOutput
from tools.registry import ToolResult

from ..cycle.chat import _resolve_reply_chat_id
from ..shared.common import _resolve_thinking_override, _thinking_floor
from ..shared.logging import (
    _clip_reply_for_log,
    _fallback_reply_for_user,
    _strip_memory_context,
)
from .exec import (
    _apply_tick_model_strategy,
    _execute_tick_action,
    _maybe_record_success_stall_reflection,
    _maybe_run_tick_continue_phase,
    _maybe_run_tick_evolution,
    _persist_tick_post_state,
    _run_tick_maintenance,
    _sync_tick_action_state,
    _tick_finalize_impl,
)
from .memory import (
    _crystallize_chat_to_semantic,
    _crystallize_reflection_to_semantic,
    _crystallize_task_done_to_semantic,
    _post_tick_memory,
)
from .prep import (
    _TickJudgmentPhase,
    _TickPerceptionPhase,
    _consume_active_task_inbox,
    _decide_initial_action,
    _inject_tick_side_signals,
    _maybe_steer_active_task_from_user_message,
    _prepare_active_task_for_tick,
    _prepare_tick_judgment_state,
    _review_delegate_tasks,
    _should_steer_active_task_from_user_message,
)
from .types import (
    _TASK_REPLY_STATS_EVERY,
    _ActionResultSummary,
    _TickJudgmentPrep,
    _build_action_result_summary,
    _build_tool_context,
    _log,
    _loop_metabolic,
)


async def _finalize_tick_user_reply(
    loop: Any,
    action: JudgmentOutput,
    result: ToolResult,
    tool_history: list[dict[str, Any]],
    user_message: str,
    active_task: Any,
    chat_id: str | None,
) -> None:
    """口腔器官：基于执行结果生成真正的对外回复。"""
    if user_message:
        reply_draft = str(action.reply_to_user or "").strip()
        if (
            reply_draft
            and action.decision in {"wait", "pause"}
            and not tool_history
            and not result.error
            and not result.skipped
        ):
            await _persist_tick_user_reply(loop, action, active_task, chat_id, user_message)
            return
        action.reply_to_user = ""
        reply_only = await _maybe_fill_tick_user_reply(loop, action, tool_history, user_message, active_task, result)
        reply_only_rationale = str(getattr(reply_only, "rationale", "") or "").strip()
        if reply_only and getattr(reply_only, "decision", "") in {"wait", "pause"}:
            # reply-only 语义上必须是可见响应态，不应在最终状态里保留 act。
            action.decision = reply_only.decision
            action.chosen_action_id = ""
            action.params = {}
            action.parallel_actions = []
            action.delegate_tasks = []
            if not action.rationale and reply_only.rationale:
                action.rationale = reply_only.rationale
            if not action.next_step and reply_only.next_step:
                action.next_step = reply_only.next_step
            if not action.reflection and reply_only.reflection:
                action.reflection = reply_only.reflection
            if not action.speech_intent and reply_only.speech_intent:
                action.speech_intent = reply_only.speech_intent
        if not action.reply_to_user:
            fallback = _fallback_reply_for_user(action, result, active_task)
            can_reuse_draft = bool(reply_draft) and action.decision in {"wait", "pause"} and not result.error
            action.reply_to_user = reply_draft if can_reuse_draft else fallback
            if not action.reply_to_user:
                action.reply_to_user = (
                    "已完成本轮处理，接下来我会基于证据继续执行闭环验证。"
                    if not user_message else "我先整理本轮结果，随后继续推进。"
                )
                _log.warning(
                    "[oral-bypass] reply_only与fallback均未生成可见回复（rationale=%s）",
                    (reply_only_rationale[:80] if reply_only_rationale else "empty"),
                )
    elif action.speech_intent and not action.reply_to_user:
        if action.decision != "act":
            action.reply_to_user = action.speech_intent

    await _persist_tick_user_reply(loop, action, active_task, chat_id, user_message)


async def _maybe_fill_tick_user_reply(
    loop: Any,
    action: JudgmentOutput,
    tool_history: list[dict[str, Any]],
    user_message: str,
    active_task: Any,
    result: ToolResult | None = None,
) -> JudgmentOutput | None:
    cfg = loop._cfg
    if not user_message:
        return None

    _ar = _build_action_result_summary(action, result or ToolResult(summary=""), tool_history)
    emotion = getattr(loop, "_emotion", None)
    _emotion_state = None
    if emotion is not None:
        regulation = getattr(getattr(emotion, "regulation", None), "strategy", "")
        _emotion_state = {
            "dominant": getattr(emotion, "dominant", ""),
            "valence": round(float(getattr(emotion, "valence", 0.0) or 0.0), 3),
            "arousal": round(float(getattr(emotion, "arousal", 0.0) or 0.0), 3),
            "regulation_strategy": str(regulation or ""),
        }

    reply_only = await loop._judgment.decide_continue(
        tool_history,
        user_message=user_message,
        active_task=active_task,
        speech_intent=action.speech_intent,
        prefer_tier="reasoner",
        thinking_override=_thinking_floor(
            _resolve_thinking_override(
                cfg,
                user_message=user_message,
                model_strategy=action.model_strategy,
            ),
            "medium" if active_task is not None else "low",
        ),
        routing_overrides=loop._pending_routing_overrides,
        reply_only=True,
        action_result=_ar,
        emotion_state=_emotion_state,
    )
    if not reply_only.reply_to_user:
        return reply_only

    reply_text = str(reply_only.reply_to_user or "").strip()
    if _reply_looks_like_internal_payload(reply_text):
        _log.warning(
            "[mouth-check] reply_only produced internal tool payload; falling back to natural reply preview=%s",
            _clip_reply_for_log(reply_text, 120),
        )
        return JudgmentOutput.wait(reason="[reply-only] reply_to_user 包含内部工具载荷")

    action.reply_to_user = reply_text
    if reply_only.rationale:
        action.rationale = reply_only.rationale
    if reply_only.reflection and not action.reflection:
        action.reflection = reply_only.reflection
    if reply_only.next_step and not action.next_step:
        action.next_step = reply_only.next_step
    _check_mouth_consistency(action.reply_to_user, _ar)
    return reply_only


def _json_like_internal_payload(value: Any) -> bool:
    if isinstance(value, dict):
        internal_keys = {
            "command",
            "tool",
            "params",
            "chosen_action_id",
            "parallel_actions",
            "stdout",
            "stderr",
            "exit_code",
            "workdir",
        }
        if any(key in value for key in internal_keys):
            return True
        return any(_json_like_internal_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_json_like_internal_payload(item) for item in value)
    return False


def _reply_looks_like_internal_payload(reply: str) -> bool:
    text = str(reply or "").strip()
    if not text:
        return False
    fenced = re.fullmatch(r"```(?:json|bash|sh)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if _json_like_internal_payload(parsed):
        return True

    json_lines = [line.strip() for line in text.splitlines() if line.strip().startswith(("{", "["))]
    for line in json_lines[:4]:
        try:
            parsed_line = json.loads(line)
        except Exception:
            continue
        if _json_like_internal_payload(parsed_line):
            return True

    return bool(re.search(r'^\s*\{[^}\n]*"(?:command|tool|chosen_action_id|params|workdir)"\s*:', text, flags=re.DOTALL))


def _check_mouth_consistency(reply: str, action_result: _ActionResultSummary) -> None:
    """内部语音环路检测。"""
    if not (action_result.action_ran and action_result.action_succeeded is True):
        return

    done_markers = (
        "已经",
        "完成了",
        "刚才",
        "刚刚",
        "已完成",
        "已修改",
        "已创建",
        "已删除",
        "已读",
        "已写",
        "成功",
        "完毕",
        "好了",
        "做好",
    )
    if any(m in reply for m in done_markers):
        return
    premature = ("马上", "立即", "现在去", "我将", "接下来我会", "稍后", "我去")
    for p in premature:
        if p in reply:
            _log.warning(
                "[mouth-check] 承诺措辞矛盾: tool=%s succeeded=True, but reply contains '%s'. "
                "Check action_result injection in Patch A.",
                action_result.tool_name,
                p,
            )
            break


async def _should_skip_duplicate_autonomous_reply(
    loop: Any,
    *,
    outbound_chat_id: str,
    reply: str,
) -> bool:
    task_store = getattr(loop, "_task_store", None)
    getter = getattr(task_store, "get_recent_chat_messages", None)
    if getter is None:
        return False
    try:
        recent = await getter(limit=6, chat_id=outbound_chat_id)
    except TypeError:
        recent = await getter(6, outbound_chat_id)
    except Exception:
        return False
    if not isinstance(recent, list):
        return False

    def _jaccard(a: str, b: str) -> float:
        wa = set(a.split())
        wb = set(b.split())
        if not wa and not wb:
            return 1.0
        union = len(wa | wb)
        return len(wa & wb) / union if union > 0 else 0.0

    for row in reversed(recent):
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "")
        if role == "assistant":
            prev = str(row.get("content") or "")
            if prev == reply:
                return True
            # 近似重复：词级 Jaccard 相似度 ≥ 0.72（自驱状态报告常用微小变异发出相同内容）
            if len(reply) > 20 and _jaccard(reply, prev) >= 0.72:
                _log.info(
                    "[task-reply] suppress near-duplicate autonomous reply jaccard=%.2f chat=%s",
                    _jaccard(reply, prev),
                    outbound_chat_id,
                )
                return True
            return False
        if role == "user":
            return False
    return False


async def _persist_tick_user_reply(
    loop: Any,
    action: JudgmentOutput,
    active_task: Any,
    chat_id: str | None,
    user_message: str = "",
) -> None:
    if not action.reply_to_user:
        return

    action.reply_to_user = _strip_memory_context(action.reply_to_user)
    _log.info(
        "[task-reply] task=%s decision=%s reply=%s",
        active_task.id if active_task else 0,
        action.decision,
        _clip_reply_for_log(action.reply_to_user),
    )
    _periodic_log_task_reply_stats(loop, action.decision, bool(user_message))

    outbound_chat_id = await _resolve_reply_chat_id(loop, active_task, chat_id)
    if outbound_chat_id is not None:
        if not user_message and await _should_skip_duplicate_autonomous_reply(
            loop,
            outbound_chat_id=outbound_chat_id,
            reply=action.reply_to_user,
        ):
            _log.info(
                "[task-reply] suppress duplicate autonomous reply task=%s chat=%s",
                active_task.id if active_task else 0,
                outbound_chat_id,
            )
            return
        await loop._task_store.add_chat_message(
            "assistant",
            action.reply_to_user,
            chat_id=outbound_chat_id,
        )
        if not user_message:
            _episodic = getattr(loop, "_episodic", None)
            if _episodic is not None:
                interlocutor_id = ""
                for key in (
                    f"chat:{outbound_chat_id}:interlocutor_profile_id" if outbound_chat_id else "",
                    f"task:{active_task.id}:interlocutor_profile_id" if active_task is not None else "",
                ):
                    if not key:
                        continue
                    value, exists = await loop._task_store.get_fact(key)
                    normalized = str(value or "").strip()
                    if exists and normalized:
                        interlocutor_id = normalized
                        break
                _affect = {
                    "valence": getattr(getattr(loop, "_emotion", None), "valence", 0.0),
                    "arousal": getattr(getattr(loop, "_emotion", None), "arousal", 0.0),
                }
                _episodic.record(
                    role="assistant_reply",
                    content=action.reply_to_user,
                    task_id=str(active_task.id) if active_task else None,
                    affect=_affect,
                    chat_id=outbound_chat_id,
                    interlocutor_id=interlocutor_id or None,
                )


def _periodic_log_task_reply_stats(loop: Any, decision: str, has_user_message: bool) -> None:
    """每固定窗口打印一次对外回复聚合统计。"""
    stats = getattr(loop, "_task_reply_window_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "count": 0,
            "user": 0,
            "auto": 0,
            "decisions": {},
        }
        loop._task_reply_window_stats = stats

    stats["count"] = int(stats.get("count", 0)) + 1
    if has_user_message:
        stats["user"] = int(stats.get("user", 0)) + 1
    else:
        stats["auto"] = int(stats.get("auto", 0)) + 1

    decision_key = str(decision or "") or "unknown"
    decisions = stats.get("decisions")
    if not isinstance(decisions, dict):
        decisions = {}
        stats["decisions"] = decisions
    decisions[decision_key] = int(decisions.get(decision_key, 0)) + 1

    if stats["count"] % _TASK_REPLY_STATS_EVERY != 0:
        return

    breakdown = ",".join(
        f"{name}:{count}" for name, count in sorted(decisions.items(), key=lambda item: item[0])
    )
    _log.info(
        "[task-reply-stats] window=%d user=%d auto=%d decisions=%s",
        stats["count"],
        stats["user"],
        stats["auto"],
        breakdown or "-",
    )

    stats["count"] = 0
    stats["user"] = 0
    stats["auto"] = 0
    stats["decisions"] = {}


class _TickExecutionPhase:
    """执行阶段编排器：工具执行 -> continue phase。"""

    @staticmethod
    async def run(
        loop: Any,
        ctx: Any,
        user_message: str,
        active_task: Any,
        cognitive_signals: Any,
        action: JudgmentOutput,
    ) -> tuple[JudgmentOutput, ToolResult, list[dict[str, Any]]]:
        result, tool_history = await _execute_tick_action(loop, ctx, active_task, action)
        from ..shared.common import _maybe_reconcile_bootstrap

        await _maybe_reconcile_bootstrap(loop)
        action, result = await _maybe_run_tick_continue_phase(
            loop,
            ctx,
            user_message,
            active_task,
            cognitive_signals,
            action,
            result,
            tool_history,
        )
        return action, result, tool_history


class _TickMemoryPhase:
    """记忆阶段编排器：用户回复最终化 -> tick 收尾。"""

    @staticmethod
    async def run(
        loop: Any,
        cycle: int,
        user_message: str,
        chat_id: str | None,
        active_task: Any,
        action: JudgmentOutput,
        result: ToolResult,
        tool_history: list[dict[str, Any]],
        perception_replay: Any,
        ethos_state: Any,
    ) -> str:
        await _finalize_tick_user_reply(loop, action, result, tool_history, user_message, active_task, chat_id)
        return await _tick_finalize_impl(
            loop,
            action,
            result,
            active_task,
            cycle,
            user_message,
            chat_id,
            perception_replay,
            ethos_state,
        )


async def _tick_impl(loop: Any, cycle: int, user_message: str = "", chat_id: str | None = None) -> str:
    """执行一轮完整认知 tick。"""
    if user_message:
        from memory.working import WMItem

        loop._wm.add(WMItem(
            kind="user_message",
            content=f"[用户消息] {user_message}",
            priority=loop._cfg.thresholds.wm_pri_user_msg,
        ))

    prep, active_task = await _TickPerceptionPhase.run(loop, user_message, chat_id)
    ctx = _build_tool_context(loop, active_task=active_task)

    action = await _TickJudgmentPhase.run(
        loop, ctx, cycle, user_message, active_task, chat_id, prep
    )

    action, result, tool_history = await _TickExecutionPhase.run(
        loop, ctx, user_message, active_task, prep.cognitive_signals, action
    )

    return await _TickMemoryPhase.run(
        loop,
        cycle,
        user_message,
        chat_id,
        active_task,
        action,
        result,
        tool_history,
        prep.perception_replay,
        prep.ethos_state,
    )


_post_tick_memory_impl = _post_tick_memory
_maybe_record_success_stall_reflection_impl = _maybe_record_success_stall_reflection
