"""模型路由上下文策略：工具历史预算、姿态与 continue 压缩快照。"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.contracts.execution import action_key_param
from core.judgment.policy.continue_history import tool_history_compact_limits
from tools.registry import tool_has_capability

if TYPE_CHECKING:
    from core.config import Config

_TASK_EXPLORATION_CAPABILITIES = ("ask_evidence", "completion_info_only", "completion_verify")


@dataclass(frozen=True, slots=True)
class ToolHistoryBudget:
    task_explore_count: int
    repeat_action_count: int
    repeat_read_count: int
    ask_evidence_hits: int
    global_cost_posture: str


def trailing_repeat_count(
    tool_history: list[dict[str, Any]],
    matcher: Callable[[dict[str, Any]], bool],
) -> int:
    count = 0
    for item in reversed(tool_history):
        if not matcher(item):
            break
        count += 1
    return count


def routing_posture(
    *,
    user_message: str,
    task_explore_count: int,
    task_explore_converge_after: int,
) -> str:
    if user_message:
        return "respond"
    if task_explore_count >= task_explore_converge_after:
        return "converge"
    return "conserve"


def _tool_name(item: dict[str, Any]) -> str:
    return str(item.get("tool") or "")


def _action_signature(item: dict[str, Any]) -> str:
    return f"{_tool_name(item)}|{action_key_param(item.get('params') or {})}"


def _is_successful_ask_evidence(registry: Any, item: dict[str, Any]) -> bool:
    result = str(item.get("result") or "").strip()
    return (
        tool_has_capability(registry, _tool_name(item), "ask_evidence")
        and bool(result)
        and not result.startswith("ERROR[")
    )


def analyze_tool_history_budget(
    registry: Any,
    cfg: Config,
    tool_history: list[dict[str, Any]] | None,
    *,
    user_message: str,
) -> ToolHistoryBudget:
    """从工具历史推导探索/重复/取证计数与全局 cost posture。"""
    history = tool_history or []
    task_explore_count = 0
    repeat_action_count = 0
    repeat_read_count = 0

    if history:
        task_explore_count = sum(
            1
            for item in history
            if any(
                tool_has_capability(registry, _tool_name(item), capability)
                for capability in _TASK_EXPLORATION_CAPABILITIES
            )
        )
        if len(history) >= 2:
            last_tool = _tool_name(history[-1])
            last_action_sig = _action_signature(history[-1])
            repeat_action_count = trailing_repeat_count(
                history,
                lambda item: _action_signature(item) == last_action_sig,
            )
            if last_tool == "file.read":
                last_path = json.dumps(history[-1].get("params", {}), ensure_ascii=False)
                repeat_read_count = trailing_repeat_count(
                    history,
                    lambda item: (
                        _tool_name(item) == "file.read"
                        and json.dumps(item.get("params", {}), ensure_ascii=False) == last_path
                    ),
                )

    ask_evidence_hits = sum(1 for item in history if _is_successful_ask_evidence(registry, item))
    posture = routing_posture(
        user_message=user_message,
        task_explore_count=task_explore_count,
        task_explore_converge_after=int(cfg.thresholds.task_explore_converge_after),
    )
    return ToolHistoryBudget(
        task_explore_count=task_explore_count,
        repeat_action_count=repeat_action_count,
        repeat_read_count=repeat_read_count,
        ask_evidence_hits=ask_evidence_hits,
        global_cost_posture=posture,
    )


def continue_phase_policy_payload(cfg: Config, tool_history_count: int) -> dict[str, Any]:
    compact_threshold, keep_last = tool_history_compact_limits(cfg)
    max_inner_rounds = max(1, int(cfg.thresholds.continue_max_inner_rounds))
    return {
        "tool_history_count": tool_history_count,
        "tool_history_compact_threshold": compact_threshold,
        "tool_history_keep_last": keep_last,
        "max_inner_rounds": max_inner_rounds,
        "will_hit_inner_round_limit_next": tool_history_count >= max_inner_rounds,
        "tool_history_will_compact_next": (
            tool_history_count >= compact_threshold and tool_history_count > keep_last
        ),
    }
