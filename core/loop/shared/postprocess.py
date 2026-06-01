"""core/loop/shared/postprocess.py - loop 停滞反思等后处理 helper。"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from core.metabolic import StateProposal
from tools.registry import ToolRegistry, ToolResult, tool_has_capability

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from store.task import Task, TaskStore

_log = logging.getLogger("lingzhou.loop")

_SUCCESS_STALL_TRACK_TOOLS = frozenset(("file.read", "file.list", "memory.search", "shell.run", "file.edit", "file.write"))
_SUCCESS_STALL_TRACK_CAPABILITIES = (
    "completion_info_only",
    "completion_verify",
    "completion_mutation",
)


def _should_track_success_stall_tool(tool_name: str, registry: ToolRegistry | None = None) -> bool:
    return any(tool_has_capability(registry, tool_name, capability) for capability in _SUCCESS_STALL_TRACK_CAPABILITIES) \
        or tool_name in _SUCCESS_STALL_TRACK_TOOLS


async def _write_success_stall_meta_reflection(
    task_store: TaskStore,
    task: Task,
    action: JudgmentOutput,
    result: ToolResult,
    *,
    streak: int,
    cycle: int,
    metabolic: Any = None,
) -> None:
    tool_name = action.chosen_action_id or "unknown"
    summary = " ".join((result.summary or "").split())
    if len(summary) > 160:
        summary = summary
    payload = {
        "reflection_id": f"stall-{task.id}-{cycle}",
        "decision": "apply",
        "target_kind": "stall_recovery",
        "proposal": (
            f"连续 {streak} 次成功动作均未推进 next_step,先停止重复 {tool_name},"
            "基于当前已知事实收敛,再决定是否换路径、换工具或转写入。"
        ),
        "verification_plan": (
            "下一轮应先总结当前事实并给出更窄的下一步,"
            "而不是继续同类探索。"
        ),
        "tool_name": tool_name,
        "recent_summary": summary,
    }
    if metabolic is None:
        from core.metabolic import MetabolicEngine
        metabolic = MetabolicEngine(task_store)
    await metabolic.submit(StateProposal(
        op="set_fact",
        key=f"task:{task.id}:meta_reflection",
        value=json.dumps(payload, ensure_ascii=False),
        scope="task", source="loop/postprocess/stall",
    ))
    _log.info("[stall-reflection] task=%s tool=%s streak=%d", task.id, tool_name, streak)
