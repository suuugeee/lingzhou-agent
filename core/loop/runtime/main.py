"""core/loop/runtime/main.py - 认知主循环(CognitionLoop)。

一个 tick 的流程:
  感知 → 情绪更新 → 伦理评估 → 判断信号生成 → LLM 判断 → 工具执行 → 记忆整合
  每 consolidate_every 轮:WM 内容写入情节记忆
  每 evolve_every 轮:触发自进化检查

解耦原则:loop 只编排,不包含业务逻辑;各层职责内聚。
"""

from __future__ import annotations

import dataclasses
from collections import deque
from typing import TYPE_CHECKING, Any

from core.judgment import JudgmentOutput
from core.metabolic import MetabolicEngine
from core.probe import ProbeManager
from store.episodic import EpisodicMemory
from store.semantic import SemanticMemory
from store.task import Task, TaskStore

from ..cycle.dispatcher import TickJob
from ..cycle.focus import resolve_focus_task
from ..tick import _post_tick_memory_impl, _tick_impl
from .builder import build_runtime_context
from .chain import (
    mount_chain_view,
    new_chain_runtime_state,
    run_dispatched_tick,
    sync_chain_state_from_view,
)
from .lifecycle import run_runtime_forever
from .memory_hooks import consolidate, emit_curiosity_signal, emit_self_drive_signal
from .startup import _open_runtime_impl

if TYPE_CHECKING:
    from core.config import Config


def _chain_recent_action_feedback() -> deque:
    return deque(maxlen=3)


def _chain_conversation_history() -> deque:
    return deque(maxlen=6)


@dataclasses.dataclass
class ChainState:
    """tick 链运行状态快照（取代硬编码字符串元组 _CHAIN_STATE_FIELDS）。

    字段变更由编译器/静态分析检测，不再依赖运行时反射字符串。
    _conv_history 在新建链时总是从空 deque 开始（不继承父链历史）。
    """

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
    _recent_action_feedback: deque = dataclasses.field(default_factory=_chain_recent_action_feedback)
    _last_action_sig: str = ""
    _last_result_fp: str = ""
    _idle_cycles: int = 0
    _last_curiosity_signal_idle_cycle: int = 0
    _ticks_since_judge: int = 0
    _pending_tier: str | None = None
    _pending_idle_gap: float | None = None
    _pending_routing_overrides: dict | None = None
    _pending_thinking_override: str | None = None
    _conv_history: deque = dataclasses.field(default_factory=_chain_conversation_history)


class CognitionLoop:
    def __init__(self, cfg: Config) -> None:
        self._runtime = build_runtime_context(cfg, owner=self)
        self._runtime.install_on(self)

    @property
    def metabolic(self) -> MetabolicEngine:
        return self._metabolic

    @property
    def probe_manager(self) -> ProbeManager:
        return self._probe_manager

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    async def open(self) -> None:
        """打开数据库连接、执行启动引导和状态恢复。interact 模式下替代 run() 前两步。"""
        await _open_runtime_impl(self)

    async def run(self) -> None:
        await run_runtime_forever(self)

    async def _next_dispatch_cycle(self) -> int:
        async with self._dispatch_cycle_lock:
            self._dispatch_cycle += 1
            return self._dispatch_cycle

    def _resolve_tick_chain_key(
        self,
        *,
        active_task: Task | None = None,
        chat_id: str | None = None,
        source: str = "auto",
    ) -> str:
        # chat 在无任务焦点时使用独立 per-session 链；
        # 一旦上游已解析出明确的 focus task，则复用 task 链，避免同一任务被 chat/auto 并发推进。
        cid = str(chat_id or "").strip()
        if cid:
            return f"chat:{cid}"
        if active_task is not None:
            chain_id = str(getattr(active_task, "chain_id", "") or "").strip()
            if chain_id:
                return f"task-chain:{chain_id}"
            return f"task:{active_task.id}"
        return f"global:{source}"

    def _new_chain_runtime_state(self) -> dict[str, Any]:
        return new_chain_runtime_state(self, ChainState)

    def _mount_chain_view(self, view: Any, state: dict[str, Any]) -> None:
        mount_chain_view(view, state, ChainState)

    def _sync_chain_state_from_view(self, state: dict[str, Any], view: Any) -> None:
        sync_chain_state_from_view(self, state, view, ChainState)

    async def _run_dispatched_tick(self, job: TickJob) -> None:
        await run_dispatched_tick(self, job, ChainState)

    async def _tick(
        self,
        cycle: int,
        user_message: str = "",
        chat_id: str | None = None,
    ) -> str:
        return await _tick_impl(self, cycle, user_message=user_message, chat_id=chat_id)

    async def _emit_self_drive_signal(self) -> None:
        await emit_self_drive_signal(self)

    async def _maybe_inject_self_drive(self) -> None:
        """兼容 tick 调用点：按当前策略尝试注入自驱信号。"""
        await self._emit_self_drive_signal()

    async def _post_tick_memory(
        self,
        action: JudgmentOutput,
        result: Any,
        active_task: Any,
        cycle: int,
        user_message: str,
        chat_id: str | None = None,
    ) -> None:
        await _post_tick_memory_impl(self, action, result, active_task, cycle, user_message, chat_id)

    @property
    def task_store(self) -> TaskStore:
        return self._task_store

    @property
    def provider(self):
        return self._provider

    async def state_snapshot(self) -> dict[str, Any]:
        """返回当前可见状态快照,供 interact REPL 渲染(Clark & Schaefer 1989 基础共识)。

        P2-A: 扩展字段,包含行为循环探针、空闲计数、WM 压力等诊断信息。
        """
        active_task = await resolve_focus_task(self)
        running_runs = await self._task_store.list_runs(status="running", limit=5)
        wm_items = self._wm.get_top(3)
        behavior_snapshot = self._behavior.snapshot()
        return {
            "valence": round(self._emotion.valence, 4),
            "arousal": round(self._emotion.arousal, 4),
            "dominance": round(self._emotion.dominance, 4),
            "dominant_emotion": self._emotion.dominant,
            "task_title": active_task.title if active_task else None,
            "task_id": str(active_task.id) if active_task else None,
            "task_status": active_task.status if active_task else None,
            "wm_size": len(self._wm.get_top(100)),
            "wm_pressure": round(self._wm.pressure, 4),
            "wm_top": [item.get("content", "") for item in wm_items],
            "idle_cycles": self._idle_cycles,
            "running_runs": [
                {
                    "id": run.id,
                    "task_id": run.task_id,
                    "tool": run.tool_name,
                    "worker": run.worker_type,
                    "session_id": run.session_id,
                }
                for run in running_runs
            ],
            "action_streak": behavior_snapshot["action_streak"],
            "read_streak": behavior_snapshot["read_streak"],
            "loop_probe_version": behavior_snapshot["loop_probe_version"],
            "conv_history_len": len(self._conv_history),
            "fts5_ok": self._semantic.fts5_ok,
        }

    async def _emit_curiosity_signal(self, ethos_state: Any) -> None:
        await emit_curiosity_signal(self, ethos_state)

    async def _maybe_curiosity_task(self, ethos_state: Any) -> None:
        """兼容 tick 调用点：按阈值注入好奇心信号。"""
        await self._emit_curiosity_signal(ethos_state)

    async def _consolidate(self, active_task: Task | None) -> None:
        await consolidate(self, active_task)
