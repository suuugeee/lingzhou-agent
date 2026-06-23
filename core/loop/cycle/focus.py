"""core/loop/cycle/focus.py - 中央焦点协调层。

职责：
1. 统一解析当前真正应当过脑的任务焦点
2. 让 chat 消息优先回到其所属 task，而不是并发起独立脑链
3. 把等待用户/外部输入的任务显式切到 waiting，并在同会话消息到来时恢复
4. 让 task.add/task.resume/task.wait 等 task 工具的结果能在本轮内被焦点主路径吸收
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.cortex import intent as cortex_intent
from core.metabolic import delete_fact, mark_task_waiting, resume_task, submit_fact

from .dispatcher import TickJob

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from store.task import Task
    from tools.registry import ToolResult

_log = logging.getLogger("lingzhou.loop")

_FOCUS_CURRENT_TASK_KEY = "focus:current_task_id"
_FOCUS_CHAT_PREFIX = "focus:chat:"
_RUNNABLE_STATUSES = frozenset({"pending", "ready", "in_progress", "resumed"})
_OPEN_STATUSES = frozenset({*tuple(_RUNNABLE_STATUSES), "waiting"})
_RUNNABLE_STATUS_ORDER = ("pending", "ready", "in_progress", "resumed")
_OPEN_STATUS_ORDER = ("pending", "ready", "in_progress", "resumed", "waiting")
_EXTERNAL_WAIT_TEXT_HINTS = (
    "等你",
    "等您",
    "等用户",
    "用户提供",
    "用户补充",
    "用户反馈",
    "收到用户",
    "收到新消息",
    "收到 url",
    "收到URL",
    "明天",
)


@dataclass(frozen=True)
class TickDispatcherState:
    enabled: bool
    can_accept: bool
    running_count: int
    pending_count: int

    @property
    def has_work(self) -> bool:
        return self.running_count > 0 or self.pending_count > 0

    @property
    def is_full(self) -> bool:
        return not self.can_accept


def _tick_dispatcher(loop: Any):
    return getattr(loop, "_tick_dispatcher", None)


def _tick_dispatcher_state(loop: Any) -> TickDispatcherState:
    dispatcher = _tick_dispatcher(loop)
    enabled = bool(dispatcher is not None and getattr(dispatcher, "enabled", False))
    if not enabled:
        return TickDispatcherState(enabled=False, can_accept=True, running_count=0, pending_count=0)

    can_accept = getattr(dispatcher, "can_accept", None)
    can_accept_value = True
    if callable(can_accept):
        with contextlib.suppress(Exception):
            can_accept_value = bool(can_accept())

    return TickDispatcherState(
        enabled=enabled,
        can_accept=can_accept_value,
        running_count=_tick_dispatcher_count(dispatcher, field="running_count"),
        pending_count=_tick_dispatcher_count(dispatcher, field="pending_count"),
    )


def _tick_dispatcher_enabled(loop: Any) -> bool:
    return _tick_dispatcher_state(loop).enabled


def _tick_dispatcher_has_capacity(loop: Any) -> bool:
    return _tick_dispatcher_state(loop).can_accept


def _tick_dispatcher_is_full(loop: Any) -> bool:
    return not _tick_dispatcher_has_capacity(loop)


def _tick_dispatcher_has_work(loop: Any) -> bool:
    return _tick_dispatcher_state(loop).has_work


def _tick_dispatcher_count(dispatcher: Any, *, field: str) -> int:
    return int(getattr(dispatcher, field, 0) or 0)


@dataclass(frozen=True)
class TickDispatchContext:
    dispatch_cycle: int
    chain_key: str
    active_task: Any | None


@dataclass(frozen=True)
class TickDispatchResult:
    context: TickDispatchContext
    accepted: bool
    can_retry: bool


def _normalize_chat_id(chat_id: str | None) -> str:
    return str(chat_id or "").strip()


def _task_status(task: Task | None) -> str:
    return str(getattr(task, "status", "") or "").strip()


def _task_is_runnable(task: Task | None) -> bool:
    return task is not None and _task_status(task) in _RUNNABLE_STATUSES


def _task_is_waiting(task: Task | None) -> bool:
    return task is not None and _task_status(task) == "waiting"


def _text_requests_external_wait(*values: Any) -> bool:
    text = " ".join(str(value or "").strip() for value in values if str(value or "").strip())
    if not text:
        return False
    lowered = text.lower()
    return cortex_intent.contains_wait_dependency(text) or any(hint.lower() in lowered for hint in _EXTERNAL_WAIT_TEXT_HINTS)


def _pause_requests_waiting(action: JudgmentOutput, active_task: Task, planned_next_step: str) -> bool:
    if action.decision != "pause":
        return False
    current_action_requests_wait = _text_requests_external_wait(
        planned_next_step,
        action.reply_to_user,
        action.rationale,
    )
    if current_action_requests_wait:
        return True
    if planned_next_step or str(action.reply_to_user or "").strip() or str(action.rationale or "").strip():
        return False
    return _text_requests_external_wait(
        getattr(active_task, "next_step", ""),
        getattr(active_task, "goal", ""),
    )


def _task_is_open(task: Task | None) -> bool:
    return task is not None and _task_status(task) in _OPEN_STATUSES


def _clear_terminal_task_attention(loop: Any, task: Task | None) -> None:
    if task is None:
        return
    wm = getattr(loop, "_wm", None)
    clearer = getattr(wm, "clear", None)
    if not callable(clearer):
        return
    source = str(getattr(task, "source", "") or "")
    kinds = {"task_anchor"}
    if source == "self_drive":
        kinds.add("self_drive")
    try:
        clearer(kinds=kinds)
    except Exception:
        _log.debug("[focus] terminal attention cleanup failed task=%s", getattr(task, "id", "-"), exc_info=True)


async def _safe_get_fact(task_store: Any, key: str) -> tuple[str, bool]:
    getter = getattr(task_store, "get_fact", None)
    if getter is not None:
        with contextlib.suppress(Exception):
            value, exists = await getter(key)
            return str(value or ""), bool(exists)
    return "", False


async def _safe_set_fact(task_store: Any, key: str, value: str, *, scope: str = "system") -> None:
    setter = getattr(task_store, "set_fact", None)
    if setter is None:
        return
    with contextlib.suppress(TypeError):
        await setter(key, value, scope=scope)
        return
    with contextlib.suppress(Exception):
        await setter(key, value)


async def _safe_delete_fact(task_store: Any, key: str) -> None:
    deleter = getattr(task_store, "delete_fact", None)
    if deleter is not None:
        with contextlib.suppress(Exception):
            await deleter(key)
            return
    # 没有 delete_fact 能力时，退化为写空字符串；读取方会做有效性校验。
    await _safe_set_fact(task_store, key, "", scope="system")


async def _submit_focus_fact(loop: Any, key: str, value: str, *, scope: str = "system") -> None:
    if not await submit_fact(
        loop,
        key=key,
        value=value,
        scope=scope,
        source="loop/focus",
    ):
        await _safe_set_fact(getattr(loop, "_task_store", None), key, value, scope=scope)


async def _delete_focus_fact(loop: Any, key: str) -> None:
    if not await delete_fact(
        loop,
        key=key,
        scope="system",
        source="loop/focus",
    ):
        await _safe_delete_fact(getattr(loop, "_task_store", None), key)


async def _safe_get_task_by_id(task_store: Any, task_id: int) -> Task | None:
    getter = getattr(task_store, "get_task_by_id", None)
    if getter is not None:
        with contextlib.suppress(Exception):
            return await getter(int(task_id))
    return None


async def _safe_get_active(task_store: Any) -> Task | None:
    getter = getattr(task_store, "get_active", None)
    if getter is not None:
        with contextlib.suppress(Exception):
            return await getter()
    return None


async def _safe_list_open_tasks(task_store: Any, *, include_waiting: bool, limit: int = 50) -> list[Task]:
    lister = getattr(task_store, "list_open_tasks", None)
    if lister is not None:
        statuses = _OPEN_STATUS_ORDER if include_waiting else _RUNNABLE_STATUS_ORDER
        with contextlib.suppress(Exception):
            rows = await lister(limit=limit, statuses=statuses)
            return list(rows or [])

    rows: list[Task] = []
    runnable_lister = getattr(task_store, "list_runnable_tasks", None)
    if runnable_lister is not None:
        with contextlib.suppress(Exception):
            rows.extend(list(await runnable_lister(limit=limit) or []))
    if include_waiting:
        task_lister = getattr(task_store, "list_tasks", None)
        if task_lister is not None:
            with contextlib.suppress(Exception):
                rows.extend(list(await task_lister(status="waiting", limit=limit) or []))

    deduped: list[Task] = []
    seen: set[int] = set()
    for task in rows:
        task_id = int(getattr(task, "id", 0) or 0)
        if task_id <= 0 or task_id in seen:
            continue
        seen.add(task_id)
        deduped.append(task)
    return deduped


async def _safe_resume_task(
    task_store: Any,
    task_id: int,
    *,
    status: str,
    current_step: str | None,
    next_step: str | None,
    result_json: dict[str, Any],
) -> None:
    with contextlib.suppress(Exception):
        await resume_task(
            task_store,
            task_id,
            source="loop/focus",
            status=status,
            current_step=current_step,
            next_step=next_step,
            result_json=result_json,
            decision_basis="focus resume via chat or user signal",
        )
        return

    resumer = getattr(task_store, "resume_task", None)
    if resumer is None:
        return
    with contextlib.suppress(Exception):
        await resumer(
            int(task_id),
            status=status,
            current_step=current_step,
            next_step=next_step,
            result_json=result_json,
        )


async def _safe_mark_waiting(
    task_store: Any,
    task_id: int,
    *,
    wait_kind: str,
    wait_key: str,
    wait_json: dict[str, Any],
    current_step: str | None,
    next_step: str | None,
    result_json: dict[str, Any],
) -> None:
    with contextlib.suppress(Exception):
        await mark_task_waiting(
            task_store,
            task_id,
            wait_kind=wait_kind,
            wait_key=wait_key,
            wait_json=wait_json,
            source="loop/focus",
            current_step=current_step,
            next_step=next_step,
            result_json=result_json,
            decision_basis="focus wait parking for user/chat signal",
        )
        return

    marker = getattr(task_store, "mark_waiting", None)
    if marker is None:
        return
    with contextlib.suppress(Exception):
        await marker(
            int(task_id),
            wait_kind=wait_kind,
            wait_key=wait_key,
            wait_json=wait_json,
            current_step=current_step,
            next_step=next_step,
            result_json=result_json,
        )


async def resolve_task_chat_id(loop: Any, task: Task | None) -> str:
    if task is None:
        return ""

    source = str(getattr(task, "source", "") or "").strip()
    if source.startswith(("wechat:", "chat:")):
        return source

    task_store = getattr(loop, "_task_store", None)
    value, exists = await _safe_get_fact(task_store, f"task:{task.id}:chat_id")
    if exists and value.strip():
        return value.strip()

    wait_kind = str(getattr(task, "wait_kind", "") or "").strip()
    wait_key = str(getattr(task, "wait_key", "") or "").strip()
    if wait_kind == "external" and wait_key:
        return wait_key

    wait_json = getattr(task, "wait_json", None) or {}
    if isinstance(wait_json, dict):
        for key in ("chat_id", "wait_key"):
            value = str(wait_json.get(key) or "").strip()
            if value:
                return value
    return ""


async def task_matches_chat(loop: Any, task: Task | None, chat_id: str | None) -> bool:
    normalized_chat_id = _normalize_chat_id(chat_id)
    if not normalized_chat_id or task is None:
        return False
    return await resolve_task_chat_id(loop, task) == normalized_chat_id


async def _load_focus_task_from_fact(loop: Any, key: str, *, include_waiting: bool) -> Task | None:
    task_store = getattr(loop, "_task_store", None)
    raw_task_id, exists = await _safe_get_fact(task_store, key)
    if not exists:
        return None
    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        return None
    task = await _safe_get_task_by_id(task_store, task_id)
    if task is None:
        return None
    if _task_is_runnable(task):
        return task
    if include_waiting and _task_is_waiting(task):
        return task
    return None


async def resolve_focus_task(
    loop: Any,
    *,
    chat_id: str | None = None,
    include_waiting: bool = False,
    fallback_active: bool = False,
) -> Task | None:
    task_store = getattr(loop, "_task_store", None)
    normalized_chat_id = _normalize_chat_id(chat_id)

    if normalized_chat_id:
        focused_task = await _load_focus_task_from_fact(
            loop,
            f"{_FOCUS_CHAT_PREFIX}{normalized_chat_id}",
            include_waiting=include_waiting,
        )
        if focused_task is not None and await task_matches_chat(loop, focused_task, normalized_chat_id):
            return focused_task

        for task in await _safe_list_open_tasks(task_store, include_waiting=include_waiting):
            if await task_matches_chat(loop, task, normalized_chat_id):
                return task

        if not fallback_active:
            return None

    current_focus = await _load_focus_task_from_fact(
        loop,
        _FOCUS_CURRENT_TASK_KEY,
        include_waiting=include_waiting,
    )
    if current_focus is not None:
        if not normalized_chat_id or await task_matches_chat(loop, current_focus, normalized_chat_id):
            return current_focus
        if fallback_active:
            return current_focus
        return None

    active_task = await _safe_get_active(task_store)
    if not normalized_chat_id:
        return active_task
    if active_task is not None and await task_matches_chat(loop, active_task, normalized_chat_id):
        return active_task
    return None


async def resolve_tick_dispatch_context(
    loop: Any,
    cycle: int,
    *,
    source: str,
    chat_id: str | None = None,
    include_waiting: bool = False,
    fallback_active: bool = True,
) -> TickDispatchContext:
    """返回可复用的 tick 分发上下文（分发周期、链路键、当前活跃任务）。"""
    normalized_chat_id = _normalize_chat_id(chat_id)
    dispatch_cycle = cycle
    with contextlib.suppress(Exception):
        dispatch_cycle = await loop._next_dispatch_cycle()
    active_task = None
    with contextlib.suppress(Exception):
        active_task = await resolve_focus_task(
            loop,
            chat_id=normalized_chat_id or None,
            include_waiting=include_waiting,
            fallback_active=fallback_active,
        )
    chain_chat_id = normalized_chat_id if active_task is None else None
    chain_key = "default"
    try:
        chain_key = loop._resolve_tick_chain_key(
            active_task=active_task,
            chat_id=chain_chat_id,
            source=source,
        )
    except TypeError:
        with contextlib.suppress(Exception):
            chain_key = loop._resolve_tick_chain_key(
                active_task=active_task,
                source=source,
            )
    except Exception:
        pass
    return TickDispatchContext(
        dispatch_cycle=dispatch_cycle,
        chain_key=chain_key,
        active_task=active_task,
    )


async def try_dispatch_tick_job(
    loop: Any,
    cycle: int,
    *,
    source: str,
    user_message: str = "",
    chat_id: str | None = None,
    chat_message_ids: tuple[int, ...] = (),
    include_waiting: bool = False,
    fallback_active: bool = False,
) -> TickDispatchResult:
    """尝试把 tick 任务放入 dispatcher。

    - accepted=False 且 can_retry=False：dispatcher 不可用，交给上层走 direct 路径
    - accepted=False 且 can_retry=True：队列已满，适合回退重试
    """
    dispatch_context = await resolve_tick_dispatch_context(
        loop,
        cycle,
        source=source,
        chat_id=chat_id,
        include_waiting=include_waiting,
        fallback_active=fallback_active,
    )

    dispatcher = getattr(loop, "_tick_dispatcher", None)
    if dispatcher is None or not getattr(dispatcher, "enabled", False):
        return TickDispatchResult(context=dispatch_context, accepted=False, can_retry=False)

    enqueue = getattr(dispatcher, "enqueue", None)
    if not callable(enqueue):
        return TickDispatchResult(context=dispatch_context, accepted=False, can_retry=False)

    accepted = await enqueue(
        build_tick_job(
            dispatch_context,
            source=source,
            user_message=user_message,
            chat_id=chat_id,
            chat_message_ids=tuple(chat_message_ids),
        )
    )
    accepted_bool = bool(accepted)
    return TickDispatchResult(
        context=dispatch_context,
        accepted=accepted_bool,
        can_retry=(not accepted_bool),
    )


def build_tick_job(
    dispatch_context: TickDispatchContext,
    *,
    source: str,
    user_message: str = "",
    chat_id: str | None = None,
    chat_message_ids: tuple[int, ...] = (),
) -> TickJob:
    return TickJob(
        cycle=dispatch_context.dispatch_cycle,
        chain_key=dispatch_context.chain_key,
        user_message=user_message,
        chat_id=chat_id,
        chat_message_ids=tuple(chat_message_ids),
        source=source,
    )


async def claim_focus_task(
    loop: Any,
    task: Task | None,
    *,
    chat_id: str | None = None,
    clear_current: bool = True,
) -> None:
    normalized_chat_id = _normalize_chat_id(chat_id)
    if not normalized_chat_id and task is not None:
        normalized_chat_id = await resolve_task_chat_id(loop, task)

    # waiting 状态的任务仍应保留焦点事实，防止 UI 瞬时显示“无活跃任务”。
    if _task_is_open(task):
        await _submit_focus_fact(loop, _FOCUS_CURRENT_TASK_KEY, str(task.id), scope="system")
    elif clear_current:
        await _delete_focus_fact(loop, _FOCUS_CURRENT_TASK_KEY)

    if normalized_chat_id:
        if _task_is_open(task):
            await _submit_focus_fact(
                loop,
                f"{_FOCUS_CHAT_PREFIX}{normalized_chat_id}",
                str(task.id),
                scope="system",
            )
        else:
            await _delete_focus_fact(loop, f"{_FOCUS_CHAT_PREFIX}{normalized_chat_id}")


async def prepare_focus_task(
    loop: Any,
    *,
    user_message: str,
    chat_id: str | None,
) -> Task | None:
    normalized_chat_id = _normalize_chat_id(chat_id)
    focus_task = await resolve_focus_task(
        loop,
        chat_id=normalized_chat_id or None,
        include_waiting=bool(str(user_message or "").strip()),
        fallback_active=not bool(normalized_chat_id),
    )
    if focus_task is None:
        return None

    if not _task_is_waiting(focus_task):
        return focus_task
    if not str(user_message or "").strip():
        return focus_task

    task_store = getattr(loop, "_task_store", None)
    await _safe_resume_task(
        task_store,
        focus_task.id,
        status="resumed",
        current_step=str(getattr(focus_task, "current_step", "") or "").strip() or None,
        next_step=str(getattr(focus_task, "next_step", "") or "").strip() or None,
        result_json={
            "resumed_via": "focus.chat" if normalized_chat_id else "focus.user",
            "chat_id": normalized_chat_id,
        },
    )
    resumed = await _safe_get_task_by_id(task_store, focus_task.id)
    if resumed is not None:
        _log.info("[focus] resumed waiting task=%s chat_id=%s", resumed.id, normalized_chat_id or "-")
        return resumed
    return focus_task


async def adopt_result_task(
    loop: Any,
    active_task: Task | None,
    action: JudgmentOutput,
    result: ToolResult,
) -> Task | None:
    if action.decision != "act":
        return active_task

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    state_delta = result.state_delta if isinstance(result.state_delta, dict) else {}
    tool_name = str(action.chosen_action_id or "")
    raw_task_id: Any = metadata.get("task_id") or state_delta.get("task_id")
    if raw_task_id in (None, "") and tool_name.startswith("task."):
        raw_task_id = result.resource_key or ""
    if raw_task_id in (None, ""):
        return active_task

    try:
        task_id = int(raw_task_id)
    except (TypeError, ValueError):
        return active_task
    task = await _safe_get_task_by_id(getattr(loop, "_task_store", None), task_id)
    return task or active_task


async def finalize_focus_task(
    loop: Any,
    *,
    action: JudgmentOutput,
    active_task: Task | None,
    chat_id: str | None,
    user_message: str,
) -> Task | None:
    if active_task is None:
        if not _normalize_chat_id(chat_id):
            await claim_focus_task(loop, None, clear_current=True)
        return None

    resolved_chat_id = _normalize_chat_id(chat_id) or await resolve_task_chat_id(loop, active_task)
    planned_next_step = str(action.next_step or "").strip()
    should_wait_for_user = (
        _task_is_runnable(active_task)
        and bool(resolved_chat_id or str(user_message or "").strip() or str(action.reply_to_user or "").strip())
        and (
            _pause_requests_waiting(action, active_task, planned_next_step)
            or (action.decision == "wait" and not planned_next_step)
        )
    )
    if should_wait_for_user:
        next_step = str(action.next_step or getattr(active_task, "next_step", "") or "").strip() or None
        current_step = str(getattr(active_task, "current_step", "") or "").strip() or None
        wait_json = {
            "wait_kind": "external",
            "wait_key": resolved_chat_id,
            "terminal_decision": action.decision,
            "chat_id": resolved_chat_id,
        }
        await _safe_mark_waiting(
            getattr(loop, "_task_store", None),
            active_task.id,
            wait_kind="external",
            wait_key=resolved_chat_id,
            wait_json=wait_json,
            current_step=current_step,
            next_step=next_step,
            result_json={
                "terminal_decision": action.decision,
                "paused_for_user": True,
            },
        )
        refreshed = await _safe_get_task_by_id(getattr(loop, "_task_store", None), active_task.id)
        if refreshed is not None:
            active_task = refreshed
        _log.info(
            "[focus] task=%s parked waiting decision=%s wait_key=%s",
            active_task.id,
            action.decision,
            resolved_chat_id or "-",
        )

    if _task_is_open(active_task):
        await claim_focus_task(loop, active_task, chat_id=resolved_chat_id or None, clear_current=True)
    else:
        _clear_terminal_task_attention(loop, active_task)
        await claim_focus_task(loop, None, chat_id=resolved_chat_id or None, clear_current=True)
    return active_task
