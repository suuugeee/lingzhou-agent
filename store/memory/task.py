from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import aiosqlite

if TYPE_CHECKING:
    from memory.task_store import Task

logger = logging.getLogger(__name__)


class TaskStateStore:
    def __init__(self, db_getter: Callable[[], aiosqlite.Connection]) -> None:
        self._db_getter = db_getter

    @property
    def _db(self) -> aiosqlite.Connection:
        return self._db_getter()

    async def add_task(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "external",
        *,
        status: str = "pending",
        next_step: str = "",
        chain_id: str = "",
        parent_task_id: str = "",
        current_step: str = "",
        wait_kind: str = "",
        wait_key: str = "",
        state_json: dict[str, Any] | None = None,
        wait_json: dict[str, Any] | None = None,
        result_json: dict[str, Any] | None = None,
        async_job_id: str = "",
        model_tier: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        data = {
            "goal": goal,
            "source": source,
            "next_step": next_step,
            "chain_id": chain_id,
            "parent_task_id": parent_task_id,
            "current_step": current_step,
            "wait_kind": wait_kind,
            "wait_key": wait_key,
            "state_json": state_json or {},
            "wait_json": wait_json or {},
            "result_json": result_json or {},
            "async_job_id": async_job_id,
            "model_tier": model_tier,
        }
        if extras:
            data.update(extras)
        async with self._db.execute(
            "INSERT INTO tasks (title, status, priority, data) VALUES (?,?,?,?)",
            (title.strip(), status, priority, json.dumps(data, ensure_ascii=False)),
        ) as cur:
            task_id: int = cur.lastrowid or 0
        await self._db.commit()
        return task_id

    async def get_task_by_id(self, task_id: int) -> Optional["Task"]:
        from memory.task_store import Task

        async with self._db.execute(
            "SELECT id, title, status, priority, created_at, data FROM tasks WHERE id=?",
            (task_id,),
        ) as cur:
            row = await cur.fetchone()
        return Task.from_row(row) if row else None

    async def list_runnable_tasks(self, limit: int = 20) -> list["Task"]:
        from memory.task_store import Task

        async with self._db.execute(
            """SELECT id, title, status, priority, created_at, data
               FROM tasks
               WHERE status IN ('pending','ready','in_progress','resumed')
               ORDER BY
                 CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                               WHEN 'normal' THEN 2 ELSE 3 END,
                 id
               LIMIT ?""",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(row) for row in rows]

    async def get_active(self) -> Optional["Task"]:
        runnable = await self.list_runnable_tasks(limit=1)
        return runnable[0] if runnable else None

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list["Task"]:
        from memory.task_store import Task

        if status:
            sql = (
                "SELECT id, title, status, priority, created_at, data "
                "FROM tasks WHERE status=? ORDER BY id LIMIT ?"
            )
            args: tuple[Any, ...] = (status, limit)
        else:
            sql = (
                "SELECT id, title, status, priority, created_at, data "
                "FROM tasks ORDER BY id LIMIT ?"
            )
            args = (limit,)
        async with self._db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [Task.from_row(row) for row in rows]

    async def update_status(
        self,
        task_id: int,
        status: str,
        next_step: str | None = None,
    ) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        task.status = status
        if next_step is not None:
            task.next_step = next_step
        await self._db.execute(
            "UPDATE tasks SET status=?, data=? WHERE id=?",
            (status, task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def mark_waiting(
        self,
        task_id: int,
        *,
        wait_kind: str,
        wait_key: str = "",
        wait_json: dict[str, Any] | None = None,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        task.status = "waiting"
        task.wait_kind = wait_kind
        task.wait_key = wait_key
        task.wait_json = wait_json or {}
        if current_step is not None:
            task.current_step = current_step
        if next_step is not None:
            task.next_step = next_step
        await self._db.execute(
            "UPDATE tasks SET status=?, data=? WHERE id=?",
            (task.status, task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def resume_task(
        self,
        task_id: int,
        *,
        status: str = "resumed",
        current_step: str | None = None,
        next_step: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        task.status = status
        task.wait_kind = ""
        task.wait_key = ""
        task.wait_json = {}
        if current_step is not None:
            task.current_step = current_step
        if next_step is not None:
            task.next_step = next_step
        if result_json is not None:
            merged = dict(task.result_json or {})
            merged.update(result_json)
            task.result_json = merged
        await self._db.execute(
            "UPDATE tasks SET status=?, data=? WHERE id=?",
            (task.status, task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        protected_keys = {"goal", "source", "next_step", "result_json"}
        ignored = [key for key in extra_dict.keys() if key in protected_keys]
        if ignored:
            logger.debug("update_task_data ignored protected task fields: %s", ",".join(sorted(ignored)))

        if "current_step" in extra_dict:
            task.current_step = str(extra_dict.get("current_step") or "")
        if "model_tier" in extra_dict:
            task.model_tier = str(extra_dict.get("model_tier") or "")

        task.extras.update(
            {
                key: value
                for key, value in extra_dict.items()
                if key not in protected_keys and key not in {"current_step", "model_tier"}
            }
        )
        await self._db.execute(
            "UPDATE tasks SET data=? WHERE id=?",
            (task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def pop_task_inbox(self, task_id: int) -> list[str]:
        task = await self.get_task_by_id(task_id)
        if not task:
            return []

        raw_messages = task.extras.get("inbox_messages")
        if not isinstance(raw_messages, list):
            return []

        messages = [str(item).strip() for item in raw_messages if str(item).strip()]
        if not messages:
            if raw_messages != []:
                task.extras["inbox_messages"] = []
                await self._db.execute(
                    "UPDATE tasks SET data=? WHERE id=?",
                    (task.to_data_json(), task_id),
                )
                await self._db.commit()
            return []

        task.extras["inbox_messages"] = []
        await self._db.execute(
            "UPDATE tasks SET data=? WHERE id=?",
            (task.to_data_json(), task_id),
        )
        await self._db.commit()
        return messages

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        merged = dict(task.result_json or {})
        merged.update(result_json or {})
        task.result_json = merged
        await self._db.execute(
            "UPDATE tasks SET data=? WHERE id=?",
            (task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def sync_task_progress(
        self,
        task_id: int,
        *,
        current_step: str | None = None,
        next_step: str | None = None,
    ) -> None:
        task = await self.get_task_by_id(task_id)
        if not task:
            return
        if current_step is not None:
            task.current_step = current_step
        if next_step is not None:
            task.next_step = next_step
        await self._db.execute(
            "UPDATE tasks SET data=? WHERE id=?",
            (task.to_data_json(), task_id),
        )
        await self._db.commit()

    async def enqueue_if_absent(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "internal",
    ) -> bool:
        title = title.strip()
        async with self._db.execute(
            "SELECT id FROM tasks WHERE title=? AND status NOT IN ('done','failed') LIMIT 1",
            (title,),
        ) as cur:
            row = await cur.fetchone()
        if row:
            return False
        await self.add_task(title, goal=goal, priority=priority, source=source)
        return True
