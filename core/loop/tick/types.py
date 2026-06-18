"""core.loop.tick.types - tick 共享类型与基础工具。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tools.registry import ToolContext, ToolResult

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


def _action_succeeded_from_status(status: str, result: ToolResult) -> bool | None:
    if status == "ok":
        return True
    if status == "error":
        return False
    if status in ("skipped", "compacted"):
        return None
    if result.skipped:
        return None
    return False if result.error else True


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

    return _ActionResultSummary(
        action_ran=True,
        action_succeeded=_action_succeeded_from_status(entry_status, result),
        tool_name=tool_name,
        summary=result.summary or "",
        error=(result.error or ""),
    )


def _loop_metabolic(loop: Any) -> Any:
    """获取 loop 的代谢器官；测试 mock 缺省时也走统一解析规则。"""
    from core.metabolic import resolve_metabolic

    metabolic = resolve_metabolic(loop)
    if metabolic is None:
        raise RuntimeError("loop 缺少可用的代谢器官或 task_store")
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
