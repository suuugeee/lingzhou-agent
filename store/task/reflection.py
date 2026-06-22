from __future__ import annotations

import json
from typing import Any

from .base import BaseAsyncStore
from .compact import compact_runtime_mapping, compact_runtime_text
from .models import MetaReflection


class MetaReflectionStore(BaseAsyncStore):

    async def add_meta_reflection(
        self,
        *,
        reflection_id: str,
        target_kind: str,
        trigger: str,
        loop_level: str,
        diagnosis: str,
        proposal: str,
        verification_plan: str = "",
        decision: str = "defer",
        task_id: int = 0,
        run_id: int = 0,
        tool_name: str = "",
        extras: dict[str, Any] | None = None,
    ) -> None:
        data = {
            "task_id": task_id,
            "run_id": run_id,
            "tool_name": tool_name,
        }
        if extras:
            data.update(compact_runtime_mapping(extras))
        await self._db.execute(
            "INSERT OR REPLACE INTO meta_reflections (id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, data) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                reflection_id,
                target_kind,
                trigger,
                loop_level,
                compact_runtime_text(diagnosis, marker_label="meta_reflection diagnosis"),
                compact_runtime_text(proposal, marker_label="meta_reflection proposal"),
                compact_runtime_text(verification_plan, marker_label="meta_reflection verification_plan"),
                compact_runtime_text(decision, marker_label="meta_reflection decision"),
                json.dumps(compact_runtime_mapping(data), ensure_ascii=False),
            ),
        )
        await self._db.commit()

    async def list_meta_reflections(
        self,
        limit: int = 20,
        loop_level: str | None = None,
    ) -> list[MetaReflection]:
        if loop_level:
            async with self._db.execute(
                "SELECT id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data FROM meta_reflections WHERE loop_level=? ORDER BY created_at ASC, id ASC LIMIT ?",
                (loop_level, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self._db.execute(
                "SELECT id, target_kind, trigger, loop_level, diagnosis, proposal, verification_plan, decision, created_at, data FROM meta_reflections ORDER BY created_at ASC, id ASC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [MetaReflection.from_row(row) for row in rows]
