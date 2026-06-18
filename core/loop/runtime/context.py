"""运行时上下文。

RuntimeContext 是 CognitionLoop 的器官容器。CognitionLoop 作为 façade 保持旧调用点，
具体器官由 builder 构造后挂载到 context，后续 lifecycle/driver 可逐步改为依赖 context。
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _recent_action_feedback() -> deque[str]:
    return deque(maxlen=3)


def _conversation_history() -> deque[tuple[str, str]]:
    return deque(maxlen=6)


@dataclass
class RuntimeContext:
    _cfg: Any
    _registry: Any
    _plugin_manager: Any
    _wm: Any
    _episodic: Any
    _task_store: Any
    _emotion: Any
    _provider: Any
    _embedding_provider: Any
    _perception: Any
    _judgment: Any
    _execution: Any
    _run_driver: Any
    _evolution: Any
    _routing_providers: dict[str, Any]
    _semantic: Any
    _metabolic: Any
    _soul: Any
    _behavior: Any
    _self_drive: Any
    _probe_manager: Any
    _cfg_file: Path
    _cfg_mtime: float
    _auth_profiles_path: Path
    _auth_profiles_mtime: float
    _tick_dispatcher: Any
    _dispatch_cycle: int = 0
    _dispatch_cycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _dispatch_state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _chain_runtime_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    _last_next_step: str = ""
    _last_decision: str = "wait"
    _last_act_progressful: bool = False
    _last_act_progress_reason: str = ""
    _last_action_tool: str = ""
    _last_action_key: str = ""
    _last_action_status: str = ""
    _last_action_summary: str = ""
    _last_action_error: str = ""
    _last_action_state_delta: str = ""
    _success_stall_task_id: str | None = None
    _success_stall_streak: int = 0
    _recent_action_feedback: deque[str] = field(default_factory=_recent_action_feedback)
    _last_action_sig: str = ""
    _last_result_fp: str = ""
    _idle_cycles: int = 0
    _last_curiosity_signal_idle_cycle: int = 0
    _conv_history: deque[tuple[str, str]] = field(default_factory=_conversation_history)
    _last_heartbeat_at: float = 0.0
    _bootstrap_mode: str = "none"
    _ticks_since_judge: int = 0
    _current_chain_key: str = ""
    _pending_tier: str | None = None
    _pending_idle_gap: float | None = None
    _pending_routing_overrides: dict[str, str] | None = None
    _pending_thinking_override: str | None = None
    _runtime_ready_callback: Any = None

    def install_on(self, target: Any) -> None:
        """把 context 字段桥接回 CognitionLoop 旧私有属性。

        这是迁移期桥接层：外部模块还在访问 loop._xxx，后续逐步改为 RuntimeContext。
        """
        for name, value in vars(self).items():
            setattr(target, name, value)
