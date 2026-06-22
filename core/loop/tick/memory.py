"""core.loop.tick.memory - 记忆结晶与 post-tick 处理。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from core.metabolic import add_semantic_memory, submit_fact
from memory.consolidation import is_low_value_semantic_text
from memory.working import WMItem

from ..cycle.chat import _resolve_reply_chat_id
from ..shared.common import _SEM_TAG_TASK_CHARS, _infer_valence_from_text
from ..shared.logging import _strip_memory_context
from ..shared.progress import action_key_param


async def _crystallize_task_done_to_semantic(loop: Any, active_task: Any) -> None:
    """任务首次完成/失败时，将 episodic 叙事结晶到 semantic 长期记忆（幂等）。"""
    if not (active_task and active_task.status not in ("done", "failed")):
        return
    refreshed = await loop._task_store.get_task_by_id(active_task.id)
    if not (refreshed and refreshed.status in ("done", "failed")):
        return
    marker = f"crystallized:{refreshed.id}"
    _, already = await loop._task_store.get_fact(marker)
    if already:
        return
    narrative = loop._episodic.load_for_context(str(refreshed.id))
    if narrative.strip():
        await add_semantic_memory(
            loop,
            node_id=f"task_summary_{refreshed.id}",
            kind="task_summary",
            title=f"[{refreshed.status}] task#{refreshed.id} {refreshed.title}",
            body=narrative,
            activation=0.9 if refreshed.status == "done" else 0.7,
            valence=loop._emotion.valence,
            tags=["task_summary", refreshed.status, f"task_{refreshed.id}"],
            source="loop/tick/run_done",
            decision_basis="task_done_crystallization",
        )
    await submit_fact(loop, key=marker, value="1", scope="system", source="loop/tick/run_done")


async def _crystallize_reflection_to_semantic(
    loop: Any,
    action: Any,
    active_task: Any,
    resolved_chat_id: str | None,
    clean_reflection: str,
) -> None:
    """将本轮反思写入 semantic insight 节点，调节情感，并按阈值结晶任务事件摘要。"""
    if not clean_reflection:
        return
    semantic_worthy = not is_low_value_semantic_text("learned_insight", "", clean_reflection)
    if semantic_worthy:
        node_id = f"insight_{hashlib.md5(clean_reflection.encode()).hexdigest()[:10]}"
        insight_suffix = f" [{node_id.split('_', 1)[-1][:6]}]"
        await add_semantic_memory(
            loop,
            node_id=node_id,
            kind="learned_insight",
            title=f"{clean_reflection}{insight_suffix}",
            body=clean_reflection,
            activation=0.9,
            valence=loop._emotion.valence,
            tags=["reflection", active_task.title[:_SEM_TAG_TASK_CHARS] if active_task else "free"],
            source="loop/tick/reflection",
            decision_basis="reflection_crystallization",
        )
    ref_valence = _infer_valence_from_text(clean_reflection, loop._emotion.valence, loop._cfg.emotion)
    delta = ref_valence - loop._emotion.valence
    if abs(delta) > 0.01:
        loop._emotion.valence = round(
            loop._emotion.valence + min(max(delta, -0.05), 0.05),
            4,
        )
    if not active_task:
        return
    turns_key = f"task:{active_task.id}:reflection_turns"
    turns_val, _ = await loop._task_store.get_fact(turns_key)
    turns = int(turns_val or "0") + 1
    await submit_fact(
        loop,
        key=turns_key,
        value=str(turns),
        scope="system",
        source="loop/tick/reflection_turns",
    )
    crystallize_every = loop._cfg.memory.chat_crystallize_every
    if semantic_worthy and turns % crystallize_every == 0:
        ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
        evt_id = f"event-task{active_task.id}-{ts_label}"
        existing = loop._semantic.get(evt_id)
        if existing:
            existing.body = existing.body + f"\n- {clean_reflection}"
            existing.activation = min(1.0, existing.activation + 0.05)
            await add_semantic_memory(
                loop,
                node_id=evt_id,
                kind="event",
                title=f"[{ts_label}] task#{active_task.id} {active_task.title}",
                body=existing.body,
                activation=existing.activation,
                valence=existing.valence,
                importance=getattr(existing, "importance", 0.0) or 0.5,
                tags=list(getattr(existing, "tags", [])),
                created_at=str(getattr(existing, "created_at", "")),
                source="loop/tick/reflection_event",
                decision_basis="reflection_event_append",
            )
        else:
            tags = ["event", ts_label]
            if resolved_chat_id:
                tags.append(f"chat:{resolved_chat_id}")
            await add_semantic_memory(
                loop,
                node_id=evt_id,
                kind="event",
                title=f"[{ts_label}] task#{active_task.id} {active_task.title}",
                body=clean_reflection,
                activation=0.85,
                valence=loop._emotion.valence,
                tags=tags,
                source="loop/tick/reflection_event",
                decision_basis="reflection_event_new",
            )


async def _crystallize_chat_to_semantic(
    loop: Any,
    action: Any,
    active_task: Any,
    user_message: str,
    resolved_chat_id: str | None,
    clean_reflection: str,
) -> None:
    """按轮次阈值，将本轮对话摘要结晶到 semantic chat_summary 节点。"""
    if not (resolved_chat_id and (user_message or action.reply_to_user or clean_reflection)):
        return
    turns_key = f"chat:{resolved_chat_id}:turns"
    turns_val, _ = await loop._task_store.get_fact(turns_key)
    turns = int(turns_val or "0") + 1
    await submit_fact(loop, key=turns_key, value=str(turns), scope="system", source="loop/tick/chat_turns")
    crystallize_every = loop._cfg.memory.chat_crystallize_every
    if turns % crystallize_every != 0:
        return
    ts_label = datetime.now(UTC).strftime("%Y-%m-%d")
    summary_id = f"chat-summary-{hashlib.md5(resolved_chat_id.encode('utf-8')).hexdigest()[:12]}-{ts_label}"
    digest = hashlib.md5(resolved_chat_id.encode("utf-8")).hexdigest()[:6]
    summary_parts: list[str] = []
    user_text = str(user_message or "").strip()
    reply_text = _strip_memory_context(action.reply_to_user or "").strip()
    reflection_text = str(clean_reflection or "").strip()
    if is_low_value_semantic_text("learned_insight", "", reflection_text):
        reflection_text = ""
    if user_text:
        summary_parts.append(f"用户: {user_text}")
    if reply_text:
        summary_parts.append(f"我: {reply_text}")
    if reflection_text:
        summary_parts.append(f"洞察: {reflection_text}")
    summary_entry = " | ".join(summary_parts)
    if not summary_entry and active_task is not None:
        summary_entry = f"任务: {active_task.title}"
    existing = loop._semantic.get(summary_id)
    if existing is not None:
        if summary_entry:
            existing.body = existing.body + f"\n- {summary_entry}"
            existing.activation = min(1.0, existing.activation + 0.05)
            existing.importance = max(float(getattr(existing, "importance", 0.0) or 0.0), 0.5)
            await add_semantic_memory(
                loop,
                node_id=summary_id,
                kind="chat_summary",
                title=f"[{ts_label}] chat[{digest}] {active_task.title if active_task is not None else resolved_chat_id}",
                body=existing.body,
                activation=existing.activation,
                valence=loop._emotion.valence,
                importance=max(float(getattr(existing, "importance", 0.0) or 0.0), 0.5),
                tags=list(getattr(existing, "tags", [])),
                created_at=str(getattr(existing, "created_at", "")),
                source="loop/tick/chat_summary",
                decision_basis="chat_summary_append",
            )
    else:
        title_seed = active_task.title if active_task is not None else resolved_chat_id
        tags = ["chat_summary", ts_label, f"chat:{resolved_chat_id}"]
        if active_task is not None:
            tags.append(f"task:{active_task.id}")
        await add_semantic_memory(
            loop,
            node_id=summary_id,
            kind="chat_summary",
            title=f"[{ts_label}] chat[{digest}] {title_seed}",
            body=summary_entry or "对话结晶",
            activation=0.85,
            valence=loop._emotion.valence,
            importance=0.5,
            tags=tags,
            source="chat_summary",
            decision_basis="chat_summary_new",
        )


async def _resolve_interlocutor_id(loop: Any, *, resolved_chat_id: str | None, active_task: Any) -> str:
    for key in (
        f"chat:{resolved_chat_id}:interlocutor_profile_id" if resolved_chat_id else "",
        f"task:{active_task.id}:interlocutor_profile_id" if active_task is not None else "",
    ):
        if not key:
            continue
        value, exists = await loop._task_store.get_fact(key)
        normalized = str(value or "").strip()
        if exists and normalized:
            return normalized
    return ""


async def _post_tick_memory(
    loop: Any,
    action: Any,
    result: Any,
    active_task: Any,
    cycle: int,
    user_message: str,
    chat_id: str | None = None,
) -> None:
    await _crystallize_task_done_to_semantic(loop, active_task)

    if result.summary and (not result.skipped or result.error):
        tool_id = action.chosen_action_id or ""
        key_param = action_key_param(action.params)
        wm_prefix = f"[{tool_id}{'  ' + key_param if key_param else ''}] "
        loop._wm.add(WMItem(
            kind=tool_id or result.kind,
            content=wm_prefix + result.summary,
            priority=result.priority,
        ))

    clean_reflection = _strip_memory_context(action.reflection) if action.reflection else ""
    if clean_reflection:
        loop._wm.add(WMItem(
            kind="synthesis",
            content=f"[合成] {clean_reflection}",
            priority=loop._cfg.thresholds.wm_pri_insight,
        ))

    affect = {"valence": loop._emotion.valence, "arousal": loop._emotion.arousal}
    resolved_chat_id = await _resolve_reply_chat_id(loop, active_task, chat_id)
    resolved_interlocutor_id = await _resolve_interlocutor_id(
        loop,
        resolved_chat_id=resolved_chat_id,
        active_task=active_task,
    )
    if action.rationale:
        loop._episodic.record(
            role="assistant",
            content=f"[cycle={cycle}] {action.rationale}",
            task_id=str(active_task.id) if active_task else None,
            affect=affect,
            chat_id=resolved_chat_id,
            interlocutor_id=resolved_interlocutor_id or None,
        )

    await _crystallize_reflection_to_semantic(loop, action, active_task, resolved_chat_id, clean_reflection)
    await _crystallize_chat_to_semantic(loop, action, active_task, user_message, resolved_chat_id, clean_reflection)

    if user_message:
        loop._episodic.record(
            role="user",
            content=user_message,
            task_id=str(active_task.id) if active_task else None,
            source_type="human",
            chat_id=resolved_chat_id,
            interlocutor_id=resolved_interlocutor_id or None,
        )
        if action.reply_to_user:
            loop._episodic.record(
                role="assistant_reply",
                content=_strip_memory_context(action.reply_to_user),
                task_id=str(active_task.id) if active_task else None,
                affect=affect,
                chat_id=resolved_chat_id,
                interlocutor_id=resolved_interlocutor_id or None,
            )
