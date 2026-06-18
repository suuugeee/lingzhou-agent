"""生命体运行状态快照。

该模块只收集 runtime 已知事实，不替 LLM 下判断。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeLifeSnapshot:
    memory: dict[str, Any] = field(default_factory=dict)
    startup: dict[str, Any] = field(default_factory=dict)
    pressure: dict[str, Any] = field(default_factory=dict)
    drive: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_runtime_life_snapshot(loop: Any) -> RuntimeLifeSnapshot:
    """从 loop 收集 LLM 可感知的生命状态。"""
    dispatcher = getattr(loop, "_tick_dispatcher", None)
    wm = getattr(loop, "_wm", None)
    cfg = getattr(loop, "_cfg", None)
    behavior = getattr(loop, "_behavior", None)
    self_drive = getattr(loop, "_self_drive", None)
    semantic = getattr(loop, "_semantic", None)
    self_model = getattr(getattr(loop, "_judgment", None), "self_model", None)

    semantic_stats = _safe_semantic_stats(semantic)
    wm_token_budget = int(getattr(wm, "_token_budget", 0) or 0)
    wm_total_tokens = int(getattr(wm, "total_tokens", 0) or 0)
    wm_pressure = float(getattr(wm, "pressure", 0.0) or 0.0)
    pending_count = int(getattr(dispatcher, "pending_count", 0) or 0)
    running_count = int(getattr(dispatcher, "running_count", 0) or 0)
    max_queue = int(getattr(getattr(cfg, "loop", None), "max_tick_queue", 0) or 0)

    return RuntimeLifeSnapshot(
        memory={
            "wm_pressure": round(wm_pressure, 4),
            "wm_tokens": wm_total_tokens,
            "wm_token_budget": wm_token_budget,
            "semantic_nodes": int(semantic_stats.get("nodes") or 0),
            "semantic_maintenance_state": semantic_stats.get("maintenance_state") or "unknown",
            "semantic_maintenance_deferred": bool(semantic_stats.get("maintenance_deferred")),
            "semantic_maintenance_last_error": semantic_stats.get("maintenance_last_error") or "",
        },
        startup={
            "bootstrap_mode": getattr(loop, "_bootstrap_mode", "none"),
            "runtime_ready_callback_pending": bool(getattr(loop, "_runtime_ready_callback", None)),
            "tick_count": int(getattr(self_model, "tick_count", 0) or 0),
        },
        pressure={
            "dispatcher_enabled": bool(getattr(dispatcher, "enabled", False)),
            "dispatch_running": running_count,
            "dispatch_pending": pending_count,
            "dispatch_queue_pressure": round(pending_count / max(max_queue, 1), 4),
            "idle_cycles": int(getattr(loop, "_idle_cycles", 0) or 0),
            "pending_idle_gap": getattr(loop, "_pending_idle_gap", None),
            "wait_streak": int(getattr(behavior, "wait_streak", 0) or 0),
        },
        drive=_safe_drive_snapshot(self_drive),
        action={
            "last_decision": getattr(loop, "_last_decision", ""),
            "last_tool": getattr(loop, "_last_action_tool", ""),
            "last_status": getattr(loop, "_last_action_status", ""),
            "last_progressful": bool(getattr(loop, "_last_act_progressful", False)),
            "last_progress_reason": getattr(loop, "_last_act_progress_reason", ""),
        },
    )


def _safe_semantic_stats(semantic: Any) -> Mapping[str, Any]:
    return _safe_mapping_method(semantic, "stats")


def _safe_drive_snapshot(self_drive: Any) -> dict[str, Any]:
    return dict(_safe_mapping_method(self_drive, "snapshot"))


def _safe_mapping_method(target: Any, method_name: str) -> Mapping[str, Any]:
    if target is None:
        return {}
    method = getattr(target, method_name, None)
    if not callable(method):
        return {}
    try:
        value = method()
        return value if isinstance(value, Mapping) else {}
    except Exception:
        return {}
