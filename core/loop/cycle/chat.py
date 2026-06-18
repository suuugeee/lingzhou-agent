"""core/loop/cycle/chat.py - loop 的 chat 绑定、回复落库与交互入口。"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from core.metabolic import submit_fact

from ..shared.logging import _strip_memory_context
from .focus import resolve_task_chat_id, try_dispatch_tick_job
from .focus import _tick_dispatcher_enabled, _tick_dispatcher_is_full

if TYPE_CHECKING:
    from store.task import Task

_log = logging.getLogger("lingzhou.loop")


async def _bind_chat_id(
    loop: Any,
    active_task: Task | None,
    chat_id: str | None,
) -> None:
    resolved_chat_id = (chat_id or "").strip()
    if not resolved_chat_id:
        return
    await submit_fact(
        loop,
        key="chat:last_chat_id",
        value=resolved_chat_id,
        scope="system",
        source="loop/chat/bind",
    )
    if active_task is not None:
        await submit_fact(
            loop,
            key=f"task:{active_task.id}:chat_id",
            value=resolved_chat_id,
            scope="task",
            source="loop/chat/bind",
        )


async def _resolve_reply_chat_id(
    loop: Any,
    active_task: Task | None,
    chat_id: str | None,
) -> str | None:
    if chat_id is not None:
        return str(chat_id or "").strip()

    if active_task is not None:
        resolved_chat_id = await resolve_task_chat_id(loop, active_task)
        if resolved_chat_id:
            return resolved_chat_id

    last_chat_id, last_chat_found = await loop._task_store.get_fact("chat:last_chat_id")
    if last_chat_found and last_chat_id.strip():
        return last_chat_id.strip()
    return None


async def _merge_chat_followups(
    loop: Any,
    *,
    chat_id: str,
    msg_id: int,
    user_message: str,
    reserved_message_ids: list[int],
) -> str:
    # wechat 通道：用户经常紧接文字发图片，各为独立 iLink 消息。
    # 图片消息需要下载解密才能写入 DB，可能晚于文字消息被 asyncio 消费。
    # 短暂等待让同批次图片消息有机会写入 DB 再 drain。
    if chat_id.startswith("wechat:"):
        delay = loop._cfg.loop.wechat_coalesce_delay
        if delay > 0:
            await asyncio.sleep(delay)
    drain_t0 = asyncio.get_running_loop().time()
    follow_ups = await loop._task_store.drain_pending_for_chat(chat_id, after_id=msg_id)
    drain_dt = asyncio.get_running_loop().time() - drain_t0
    if drain_dt >= 1.0:
        _log.warning(
            "[chat] drain_pending_for_chat slow dt=%.3fs chat_id=%s after_id=%s",
            drain_dt,
            chat_id,
            msg_id,
        )
    if not follow_ups:
        return user_message
    reserved_message_ids.extend(int(m.get("id") or 0) for m in follow_ups if int(m.get("id") or 0) > 0)
    extra = "\n".join(m["content"] for m in follow_ups)
    merged_message = f"{user_message}\n{extra}".strip()
    _log.debug("[chat] merged %d follow-up message(s) into turn (ids=%s)",
               len(follow_ups), [m["id"] for m in follow_ups])
    return merged_message


async def _process_pending_chat_turn(loop: Any, cycle: int) -> tuple[int, bool]:
    original_cycle = cycle
    if _tick_dispatcher_is_full(loop):
        _log.debug("[chat] tick queue full, defer pending chat pickup")
        return cycle, False

    pop_t0 = asyncio.get_running_loop().time()
    chat_message = await loop._task_store.pop_pending_chat_message()
    pop_dt = asyncio.get_running_loop().time() - pop_t0
    if pop_dt >= 1.0:
        _log.warning("[chat] pop_pending_chat_message slow dt=%.3fs", pop_dt)
    if not chat_message:
        return cycle, False

    user_message = str(chat_message.get("content") or "")
    chat_id = str(chat_message.get("chat_id") or "")
    msg_id: int = chat_message.get("id") or 0
    reserved_message_ids: list[int] = [msg_id] if msg_id else []

    # drain 同一会话紧随而来的消息，合并为同一 LLM 上下文轮次
    if chat_id:
        user_message = await _merge_chat_followups(
            loop,
            chat_id=chat_id,
            msg_id=msg_id,
            user_message=user_message,
            reserved_message_ids=reserved_message_ids,
        )

    if _tick_dispatcher_enabled(loop):
        dispatch_result = await try_dispatch_tick_job(
            loop,
            cycle,
            source="chat",
            chat_id=chat_id,
            user_message=user_message,
            chat_message_ids=tuple(reserved_message_ids),
            include_waiting=True,
        )
        if dispatch_result.accepted:
            _log.info("[chat] user › %s", user_message)
            return dispatch_result.context.dispatch_cycle, True
        if dispatch_result.can_retry:
            await loop._task_store.release_chat_messages(reserved_message_ids)
            _log.debug("[chat] tick queue full, defer chat retry chat_id=%s", chat_id)
            return original_cycle, True

    cycle += 1
    _log.info("[chat] user › %s", user_message)
    try:
        reply = await loop._tick(
            cycle,
            user_message=user_message,
            chat_id=chat_id,
        )
    except Exception:
        await loop._task_store.release_chat_messages(reserved_message_ids)
        raise

    await loop._task_store.mark_chat_messages_processed(reserved_message_ids)
    if reply:
        reply = _strip_memory_context(reply)
    _log.info("[chat] assistant › %s", reply or "")
    if not reply:
        await loop._task_store.add_chat_message(
            "assistant",
            "(请求已处理,任务正在后台继续)",
            chat_id=chat_id,
        )
    return cycle, True
