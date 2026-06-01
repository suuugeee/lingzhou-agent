"""core.loop.tick.types - tick 共享类型与基础工具。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tools.registry import ToolContext, ToolResult

from ..shared.logging import _clip_signal_text

_log = logging.getLogger("lingzhou.loop")
_TASK_REPLY_STATS_EVERY = 20


@dataclass(slots=True)
class _ActionResultSummary:
    """口腔器官所需的结构化执行状态。"""

    action_ran: bool
    action_succeeded: bool | None
    tool_name: str
    summary: str
    error: str


@dataclass(slots=True)
class _TickJudgmentPrep:
    percept: Any
    perception_replay: Any
    cognitive_signals: Any
    ethos_state: Any
    signals: Any
    hard_boundaries: list[str]


def _build_action_result_summary(
    action: Any,
    result: ToolResult,
    tool_history: list[dict[str, Any]],
) -> _ActionResultSummary:
    """从当前 action/result 构建口腔器官所需的结构化执行状态。"""
    if action.decision != "act":
        return _ActionResultSummary(
            action_ran=False,
            action_succeeded=None,
            tool_name="",
            summary="",
            error="",
        )

    last_entry = tool_history[-1] if tool_history else {}
    tool_name = str(last_entry.get("tool", "") or action.chosen_action_id or "")
    entry_status = str(last_entry.get("status", "") or "")
    if entry_status == "ok":
        succeeded: bool | None = True
    elif entry_status == "error":
        succeeded = False
    elif entry_status in ("skipped", "compacted"):
        succeeded = None
    else:
        if result.skipped:
            succeeded = None
        elif result.error:
            succeeded = False
        else:
            succeeded = True

    return _ActionResultSummary(
        action_ran=True,
        action_succeeded=succeeded,
        tool_name=tool_name,
        summary=_clip_signal_text(result.summary or "", 300),
        error=(result.error or ""),
    )


def _loop_metabolic(loop: Any) -> Any:
    """获取 loop 的 metabolic 实例；若不存在则创建临时实例（兼容测试 mock）。"""
    metabolic = getattr(loop, "_metabolic", None)
    if metabolic is None:
        from core.metabolic import MetabolicEngine

        metabolic = MetabolicEngine(loop._task_store)
    return metabolic


def _build_tool_context(loop: Any, *, active_task: Any = None) -> ToolContext:
    return ToolContext(
        config=loop._cfg,
        wm=loop._wm,
        task_store=loop._task_store,
        episodic=loop._episodic,
        semantic=loop._semantic,
        emotion=loop._emotion,
        active_task=active_task,
        probe_manager=getattr(loop, "_probe_manager", None),
        judgment=loop._judgment,
        execution=loop._execution,
        registry=loop._registry,
        metabolic=_loop_metabolic(loop),
    )


_LLM_WAKE_WM_KINDS = {
    "heartbeat",
    "scheduler",
    "bootstrap",
    "crash_recovery",
    "curiosity",
    "self_drive",
    "self_awareness",
    "behavior_sense",
}
