from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from .base import BaseAsyncStore
from .compact import compact_runtime_mapping, compact_runtime_text
from .models import Run


class RunStore(BaseAsyncStore):

    async def add_run(
        self,
        *,
        task_id: int = 0,
        run_type: str = "tool_chain",
        worker_type: str = "tool-chain-worker",
        status: str = "running",
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str = "",
        error_text: str = "",
        tool_name: str = "",
        session_id: str = "",
        model_tier: str = "",
        progress: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int:
        data = {
            "input_json": compact_runtime_mapping(input_json),
            "output_json": compact_runtime_mapping(output_json),
            "log_text": compact_runtime_text(log_text, marker_label="run log"),
            "error_text": compact_runtime_text(error_text, marker_label="run error"),
            "tool_name": tool_name,
            "session_id": session_id,
            "model_tier": model_tier,
            "progress": compact_runtime_text(progress, marker_label="run progress"),
        }
        if extras:
            data.update(compact_runtime_mapping(extras))
        now = datetime.now(UTC).isoformat()
        async with self._db.execute(
            "INSERT INTO runs (task_id, run_type, worker_type, status, created_at, started_at, data) VALUES (?,?,?,?,?,?,?)",
            (task_id, run_type, worker_type, status, now, now, json.dumps(data, ensure_ascii=False)),
        ) as cur:
            run_id: int = cur.lastrowid or 0
        await self._db.commit()
        return run_id

    async def get_run_by_id(self, run_id: int) -> Run | None:
        async with self._db.execute(
            "SELECT id, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data FROM runs WHERE id=?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        return Run.from_row(row) if row else None

    async def list_runs(
        self,
        *,
        task_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Run]:
        clauses: list[str] = []
        args: list[Any] = []
        if task_id is not None:
            clauses.append("task_id=?")
            args.append(task_id)
        if status:
            clauses.append("status=?")
            args.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)
        async with self._db.execute(
            f"SELECT id, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data FROM runs {where} ORDER BY id DESC LIMIT ?",
            tuple(args),
        ) as cur:
            rows = await cur.fetchall()
        return [Run.from_row(row) for row in rows]

    async def update_run(
        self,
        run_id: int,
        *,
        task_id: int | None = None,
        status: str | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str | None = None,
        error_text: str | None = None,
        session_id: str | None = None,
        model_tier: str | None = None,
        progress: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        run = await self.get_run_by_id(run_id)
        if not run:
            return
        if task_id is not None:
            run.task_id = task_id
        if status:
            run.status = status
        if output_json is not None:
            run.output_json = compact_runtime_mapping(output_json)
        if log_text is not None:
            run.log_text = compact_runtime_text(log_text, marker_label="run log")
        if error_text is not None:
            run.error_text = compact_runtime_text(error_text, marker_label="run error")
        if session_id is not None:
            run.session_id = session_id
        if model_tier is not None:
            run.model_tier = model_tier
        if progress is not None:
            run.progress = compact_runtime_text(progress, marker_label="run progress")
        if extras:
            run.extras.update(compact_runtime_mapping(extras))
        if run.status in {"succeeded", "failed", "cancelled"} and not run.completed_at:
            run.completed_at = datetime.now(UTC).isoformat()
        await self._db.execute(
            "UPDATE runs SET task_id=?, status=?, completed_at=?, data=? WHERE id=?",
            (run.task_id, run.status, run.completed_at, run.to_data_json(), run_id),
        )
        await self._db.commit()

    async def get_pending_runs(self, *, limit: int = 10) -> list[Run]:
        """返回 status='pending' 的 Run，按 created_at 升序（最早优先，Phase 3d 调度器轮询用）。"""
        async with self._db.execute(
            "SELECT id, task_id, run_type, worker_type, status, created_at, started_at, completed_at, data"
            " FROM runs WHERE status='pending' ORDER BY created_at LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [Run.from_row(r) for r in rows]

    async def cancel_stale_runs(self, stale_after_seconds: int = 600) -> int:
        """将超时的 running/pending Run 标为 cancelled（Phase 3d 崩溃恢复）。

        用于进程重启时清理上次崩溃遗留的非终态 Run，
        保持 DB 状态一致性，避免 runs 表中出现永久 'running' 记录。

        Returns:
            取消的 Run 数量。
        """
        cutoff = (datetime.now(UTC) - timedelta(seconds=stale_after_seconds)).isoformat()
        async with self._db.execute(
            "SELECT id FROM runs WHERE status IN ('running', 'pending') AND started_at < ?",
            (cutoff,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return 0
        count = 0
        for (run_id,) in rows:
            try:
                await self.update_run(
                    run_id,
                    status="cancelled",
                    error_text="[startup] stale run cancelled — process restarted",
                )
                count += 1
            except Exception:
                pass
        return count
