from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

_LOCAL_FACT_PREFIXES: tuple[str, ...] = (
    "durable_failure:",
)

_LOCAL_FACT_KEYS: frozenset[str] = frozenset({
    "control:durable_failure_policy",
})


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_locally_absorbable_fact(key: str, scope: str) -> bool:
    if scope == "system":
        return True
    if key in _LOCAL_FACT_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in _LOCAL_FACT_PREFIXES)


class _SubagentReadonlyViolation(RuntimeError):
    pass


@dataclass
class _SubagentLocalState:
    """子灵 TaskStoreView 的本地运行期状态（原四个独立 dict，现整合为单一 dataclass）。"""

    facts: dict[str, tuple[str, str]] = field(default_factory=dict)
    failures: list[Any] = field(default_factory=list)
    task_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    runs: dict[int, Any] = field(default_factory=dict)
    meta_reflections: list[Any] = field(default_factory=list)
    next_run_id: int = -1


class _SubagentTaskStoreView:  # noqa: D101 - internal view for subagent tick context
    """父灵 TaskStore 的子灵隔离视图：读透传，运行期 bookkeeping 本地吸收。"""

    def __init__(self, parent: Any, active_task: Any | None = None) -> None:
        self._parent = parent
        self._active_task = active_task
        self._local = _SubagentLocalState()

    def _reject(self, action: str) -> _SubagentReadonlyViolation:
        return _SubagentReadonlyViolation(f"子灵只读模式禁止修改父灵状态: {action}")

    def _overlay_task(self, task: Any | None) -> Any | None:
        if task is None:
            return None
        task_id = int(getattr(task, "id", 0) or 0)
        local_result = self._local.task_results.get(task_id)
        if not local_result:
            return task
        merged_result = dict(getattr(task, "result_json", {}) or {})
        merged_result.update(local_result)
        return replace(task, result_json=merged_result)

    def _task_matches_status(self, task: Any, status: Any | None) -> bool:
        if status is None:
            return True
        return str(getattr(task, "status", "") or "") == str(status)

    def __getattr__(self, name: str) -> Any:
        if name == "get_active":
            async def _get_active() -> Any:
                if self._active_task is not None:
                    return self._overlay_task(self._active_task)
                return None

            return _get_active
        raise AttributeError(name)

    async def get_task_by_id(self, task_id: int) -> Any:
        if self._active_task is not None:
            active_id = int(getattr(self._active_task, "id", 0) or 0)
            if active_id == int(task_id):
                return self._overlay_task(self._active_task)
            return None
        return self._overlay_task(await self._parent.get_task_by_id(task_id))

    async def list_tasks(self, *args: Any, **kwargs: Any) -> list[Any]:
        status = kwargs.get("status")
        if status is None and args:
            status = args[0]
        limit = kwargs.get("limit")
        if limit is None and len(args) > 1:
            limit = args[1]
        limit = int(limit or 50)
        if self._active_task is not None:
            if str(status or "") == "waiting":
                return []
            if self._task_matches_status(self._active_task, status) and limit > 0:
                return [self._overlay_task(self._active_task)]
            return []
        if str(status or "") == "waiting":
            return []
        merged: list[Any] = []
        if self._active_task is not None and self._task_matches_status(self._active_task, status):
            merged.append(self._overlay_task(self._active_task))
        seen = {
            int(getattr(task, "id", 0) or 0)
            for task in merged
        }
        for task in await self._parent.list_tasks(*args, **kwargs):
            task_id = int(getattr(task, "id", 0) or 0)
            if task_id in seen:
                continue
            merged.append(self._overlay_task(task))
            if len(merged) >= limit:
                break
        return merged[:limit]

    async def list_runnable_tasks(self, limit: int = 20) -> list[Any]:
        runnable_statuses = {"pending", "ready", "in_progress", "resumed"}
        if self._active_task is not None:
            status = str(getattr(self._active_task, "status", "") or "")
            if status not in runnable_statuses or limit <= 0:
                return []
            return [self._overlay_task(self._active_task)]
        rows = await self._parent.list_runnable_tasks(limit=limit)
        return [self._overlay_task(task) for task in rows[:limit]]

    async def list_runs(
        self,
        task_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Any]:
        if self._active_task is not None:
            active_id = int(getattr(self._active_task, "id", 0) or 0)
            if task_id is not None and int(task_id) != active_id:
                return []
            return [
                run
                for run in sorted(self._local.runs.values(), key=lambda item: getattr(item, "id", 0), reverse=True)
                if (task_id is None or int(getattr(run, "task_id", 0) or 0) == int(task_id))
                and (not status or str(getattr(run, "status", "") or "") == status)
            ][:limit]
        local = [
            run
            for run in sorted(self._local.runs.values(), key=lambda item: getattr(item, "id", 0), reverse=True)
            if (task_id is None or int(getattr(run, "task_id", 0) or 0) == int(task_id))
            and (not status or str(getattr(run, "status", "") or "") == status)
        ][:limit]
        parent = await self._parent.list_runs(task_id=task_id, status=status, limit=limit)
        merged = list(local)
        seen = {getattr(run, "id", None) for run in merged}
        for item in parent:
            if getattr(item, "id", None) in seen:
                continue
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    async def add_run(self, **kwargs: Any) -> int:
        from store.task import Run

        now = _utc_now_iso()
        run_id = self._local.next_run_id
        self._local.next_run_id -= 1
        self._local.runs[run_id] = Run(
            id=run_id,
            task_id=int(kwargs.get("task_id", 0) or 0),
            run_type=str(kwargs.get("run_type", "tool_chain") or "tool_chain"),
            worker_type=str(kwargs.get("worker_type", "tool-chain-worker") or "tool-chain-worker"),
            status=str(kwargs.get("status", "running") or "running"),
            created_at=now,
            started_at=now,
            completed_at="",
            input_json=dict(kwargs.get("input_json") or {}),
            output_json=dict(kwargs.get("output_json") or {}),
            log_text=str(kwargs.get("log_text") or ""),
            error_text=str(kwargs.get("error_text") or ""),
            tool_name=str(kwargs.get("tool_name") or ""),
            session_id=str(kwargs.get("session_id") or ""),
            model_tier=str(kwargs.get("model_tier") or ""),
            progress=str(kwargs.get("progress") or ""),
            extras=dict(kwargs.get("extras") or {}),
        )
        return run_id

    async def get_run_by_id(self, run_id: int) -> Any | None:
        local = self._local.runs.get(run_id)
        if local is not None:
            return local
        if self._active_task is not None:
            return None
        return await self._parent.get_run_by_id(run_id)

    async def update_run(self, run_id: int, **kwargs: Any) -> None:
        run = self._local.runs.get(run_id)
        if run is None:
            return
        if "status" in kwargs and kwargs["status"] is not None:
            run.status = str(kwargs["status"])
        if "output_json" in kwargs and kwargs["output_json"] is not None:
            run.output_json = dict(kwargs["output_json"] or {})
        if "log_text" in kwargs and kwargs["log_text"] is not None:
            run.log_text = str(kwargs["log_text"] or "")
        if "error_text" in kwargs and kwargs["error_text"] is not None:
            run.error_text = str(kwargs["error_text"] or "")
        if "session_id" in kwargs and kwargs["session_id"] is not None:
            run.session_id = str(kwargs["session_id"] or "")
        if "model_tier" in kwargs and kwargs["model_tier"] is not None:
            run.model_tier = str(kwargs["model_tier"] or "")
        if "progress" in kwargs and kwargs["progress"] is not None:
            run.progress = str(kwargs["progress"] or "")
        extras = kwargs.get("extras")
        if isinstance(extras, dict) and extras:
            run.extras.update(extras)
        status = str(kwargs.get("status") or getattr(run, "status", "") or "")
        if status in {"succeeded", "failed", "cancelled"}:
            run.completed_at = _utc_now_iso()

    async def record_failure(
        self,
        kind: str,
        summary: str,
        context: str = "",
        task_id: str = "",
    ) -> None:
        from store.task import Failure

        self._local.failures.append(Failure(
            id=-(len(self._local.failures) + 1),
            kind=kind,
            dismissed=False,
            created_at=_utc_now_iso(),
            summary=summary,
            context=context,
            task_id=task_id,
        ))

    async def list_failures(self, limit: int = 20) -> list[Any]:
        if self._active_task is not None:
            return list(self._local.failures[-limit:])
        local = list(self._local.failures[-limit:])
        parent = await self._parent.list_failures(limit=limit)
        return local + list(parent[: max(0, limit - len(local))])

    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Any]:
        if self._active_task is not None:
            active_id = str(getattr(self._active_task, "id", 0) or 0)
            if str(task_id) != active_id:
                return []
            return [item for item in self._local.failures if str(getattr(item, "task_id", "")) == active_id][-limit:]
        local = [item for item in self._local.failures if str(getattr(item, "task_id", "")) == str(task_id)][-limit:]
        parent = await self._parent.list_failures_for_task(task_id, limit=limit)
        return local + list(parent[: max(0, limit - len(local))])

    async def count_failures_by_kind(self, kind: str) -> int:
        if self._active_task is not None:
            return sum(
                1
                for item in self._local.failures
                if str(getattr(item, "kind", "") or "") == kind and not bool(getattr(item, "dismissed", False))
            )
        local = sum(
            1
            for item in self._local.failures
            if str(getattr(item, "kind", "") or "") == kind and not bool(getattr(item, "dismissed", False))
        )
        return local + await self._parent.count_failures_by_kind(kind)

    async def dismiss_failure(self, failure_id: int) -> None:
        for item in self._local.failures:
            if int(getattr(item, "id", 0) or 0) == int(failure_id):
                item.dismissed = True
                return
        raise self._reject("dismiss_failure")

    async def set_fact(self, key: str, value: str, scope: str = "general") -> None:
        if not _is_locally_absorbable_fact(key, scope):
            raise self._reject(f"set_fact:{key}")
        self._local.facts[key] = (value, scope)

    async def get_fact(self, key: str) -> tuple[str, bool]:
        local = self._local.facts.get(key)
        if local is not None:
            return local[0], True
        return await self._parent.get_fact(key)

    async def list_facts(self, prefix: str = "", limit: int = 100) -> list[tuple[str, str]]:
        local = [
            (key, value)
            for key, (value, _) in self._local.facts.items()
            if not prefix or key.startswith(prefix)
        ]
        parent = await self._parent.list_facts(prefix=prefix, limit=limit)
        merged = list(local[-limit:])
        seen = {key for key, _ in merged}
        for item in parent:
            if item[0] in seen:
                continue
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    async def delete_fact(self, key: str) -> None:
        if key in self._local.facts:
            self._local.facts.pop(key, None)
            return
        raise self._reject(f"delete_fact:{key}")

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        existing = dict(self._local.task_results.get(task_id) or {})
        existing.update(dict(result_json or {}))
        self._local.task_results[task_id] = existing

    async def pop_task_inbox(self, task_id: int) -> list[str]:
        return []

    async def sync_task_progress(
        self,
        task_id: int,
        *,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        raise self._reject("sync_task_progress")

    async def add_meta_reflection(self, **kwargs: Any) -> None:
        from store.task import MetaReflection

        self._local.meta_reflections.append(MetaReflection(
            id=str(kwargs.get("reflection_id") or ""),
            target_kind=str(kwargs.get("target_kind") or ""),
            trigger=str(kwargs.get("trigger") or ""),
            loop_level=str(kwargs.get("loop_level") or ""),
            diagnosis=str(kwargs.get("diagnosis") or ""),
            proposal=str(kwargs.get("proposal") or ""),
            verification_plan=str(kwargs.get("verification_plan") or ""),
            decision=str(kwargs.get("decision") or "defer"),
            created_at=_utc_now_iso(),
            task_id=int(kwargs.get("task_id") or 0),
            run_id=int(kwargs.get("run_id") or 0),
            tool_name=str(kwargs.get("tool_name") or ""),
            extras=dict(kwargs.get("extras") or {}),
        ))

    async def list_meta_reflections(self, limit: int = 20, loop_level: str | None = None) -> list[Any]:
        if self._active_task is not None:
            return [
                item
                for item in self._local.meta_reflections
                if loop_level is None or str(getattr(item, "loop_level", "") or "") == loop_level
            ][-limit:]
        local = [
            item
            for item in self._local.meta_reflections
            if loop_level is None or str(getattr(item, "loop_level", "") or "") == loop_level
        ][-limit:]
        parent = await self._parent.list_meta_reflections(limit=limit, loop_level=loop_level)
        merged = list(local)
        seen = {getattr(item, "id", None) for item in merged}
        for item in parent:
            if getattr(item, "id", None) in seen:
                continue
            merged.append(item)
            if len(merged) >= limit:
                break
        return merged

    async def due_signals(self) -> list[dict[str, Any]]:
        return await self._parent.due_signals()
    async def list_signals(self, limit: int = 30, include_done: bool = False) -> list[dict[str, Any]]:
        return await self._parent.list_signals(limit=limit, include_done=include_done)
    async def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        return await self._parent.get_signal(signal_id)
    async def enqueue_if_absent(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "internal",
    ) -> bool:
        raise self._reject("enqueue_if_absent")
    async def add_chat_message(self, role: str, content: str, chat_id: str = "") -> int:
        raise self._reject("add_chat_message")
    async def has_pending_chat_message(self) -> bool:
        return False
    async def pop_pending_chat_message(self) -> dict[str, Any] | None:
        return None
    async def drain_pending_for_chat(self, chat_id: str, after_id: int) -> list[dict[str, Any]]:
        return []
    async def mark_chat_messages_processed(self, message_ids: list[int] | tuple[int, ...]) -> None:
        return None
    async def release_chat_messages(self, message_ids: list[int] | tuple[int, ...]) -> None:
        return None
    async def get_chat_messages_since(
        self,
        since_id: int = 0,
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        return []
    async def get_recent_chat_messages(
        self,
        limit: int = 6,
        chat_id: str = "",
    ) -> list[dict[str, Any]]:
        return []
    async def reset_in_progress_tasks(self) -> int:
        raise self._reject("reset_in_progress_tasks")
    async def add_task(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("add_task")
    async def update_status(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("update_status")
    async def update_task_data(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("update_task_data")
    async def add_signal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("add_signal")
    async def ack_signal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("ack_signal")
    async def cancel_signal(self, *args: Any, **kwargs: Any) -> Any:
        raise self._reject("cancel_signal")
    async def mark_waiting(self, *args: Any, **kwargs: Any) -> None:
        raise self._reject("mark_waiting")
    async def resume_task(self, *args: Any, **kwargs: Any) -> None:
        raise self._reject("resume_task")
    async def ledger_append(self, op: str, key: str, value: str, **kwargs: Any) -> None:
        return None  # 子灵不写父灵账本
    async def ledger_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._parent.ledger_recent(limit=limit)
    async def ledger_since(self, after_id: int, limit: int = 100) -> list[dict[str, Any]]:
        return await self._parent.ledger_since(after_id, limit=limit)
