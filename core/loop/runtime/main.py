"""core/loop/runtime/main.py - 认知主循环(CognitionLoop)。

一个 tick 的流程:
  感知 → 情绪更新 → 伦理评估 → 判断信号生成 → LLM 判断 → 工具执行 → 记忆整合
  每 consolidate_every 轮:WM 内容写入情节记忆
  每 evolve_every 轮:触发自进化检查

解耦原则:loop 只编排,不包含业务逻辑;各层职责内聚。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel

from core.evolution import EvolutionEngine
from core.execution import ExecutionLayer
from core.judgment import JudgmentLayer, JudgmentOutput
from core.loop.drive.behavior import BehaviorTracker
from core.metabolic import MetabolicEngine
from core.perception import EmotionState
from core.persona.soul import SoulManager
from core.probe import ProbeManager
from memory.working import WorkingMemory
from provider import create_provider
from provider.base import EmbeddingProvider
from store.episodic import EpisodicMemory
from store.semantic import SemanticMemory
from store.task import Task, TaskStore
from tools.registry import ToolRegistry

from ..cycle.dispatcher import ConcurrentTickDispatcher, TickJob
from ..cycle.driver import _run_cycle_impl, _wait_after_cycle_impl
from ..cycle.focus import resolve_focus_task
from ..runs.driver import RunDriver
from ..tick import _post_tick_memory_impl, _tick_impl
from .chain import (
    mount_chain_view,
    new_chain_runtime_state,
    run_dispatched_tick,
    sync_chain_state_from_view,
)
from .memory_hooks import consolidate, emit_curiosity_signal, emit_self_drive_signal
from .startup import _open_runtime_impl, _prepare_runtime_run_impl

if TYPE_CHECKING:
    from core.config import Config

console = Console()
_log = logging.getLogger("lingzhou.loop")


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
    _recent_action_feedback: deque = dataclasses.field(default_factory=lambda: deque(maxlen=3))
    _last_action_sig: str = ""
    _last_result_fp: str = ""
    _idle_cycles: int = 0
    _last_curiosity_signal_idle_cycle: int = 0
    _ticks_since_judge: int = 0
    _pending_tier: str | None = None
    _pending_idle_gap: float | None = None
    _pending_routing_overrides: dict | None = None
    _pending_thinking_override: str | None = None
    _conv_history: deque = dataclasses.field(default_factory=lambda: deque(maxlen=6))


class CognitionLoop:
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

        # 工具注册
        self._registry = ToolRegistry()
        repo_root = Path(__file__).resolve().parents[3]
        tools_dir = repo_root / "tools"
        self._registry.discover(tools_dir)

        # 插件系统：发现并加载插件
        from core.plugin import PluginManager

        plugins_dir = repo_root / "plugins"
        self._plugin_manager = PluginManager(plugins_dir)
        self._plugin_manager.discover()
        self._plugin_manager.load_all()
        self._plugin_manager.register_all(tool_registry=self._registry)
        self._plugin_manager.start_all()
        _log.info("[plugin] 已加载 %d 个插件", len(self._plugin_manager.list_plugins()))

        # 记忆层
        self._wm = WorkingMemory(
            capacity=cfg.memory.working_capacity,
            token_budget=cfg.effective_wm_token_budget(),
            item_max_tokens=cfg.memory.wm_item_max_tokens,
        )
        self._episodic = EpisodicMemory(cfg.memory_dir, max_events=cfg.memory.max_events)
        self._task_store = TaskStore(Path(cfg.db_path))
        self._metabolic = MetabolicEngine(self._task_store)  # 代谢器官（公理 A5）

        # 情绪状态(初始值来自 config)
        self._emotion = EmotionState.from_config(cfg)

        # 认知组件
        self._provider = create_provider(cfg)
        from core.perception import PerceptionLayer

        self._perception = PerceptionLayer(cfg)
        self._judgment = JudgmentLayer(self._provider, self._registry, cfg)
        self._execution = ExecutionLayer(self._registry, cfg)
        self._run_driver = RunDriver(self._execution)  # Phase 3b: Run 路由层
        self._evolution = EvolutionEngine(cfg, self._provider, self._registry)

        # 分层路由 providers({"simple": p1, "complex": p2},由 open() 注入 JudgmentLayer)
        self._routing_providers: dict[str, Any] = {}

        # embedding 混合检索(embed_fn=None 则纯关键词模式)
        embed_fn: Callable[..., Any] | None = None
        if cfg.memory.local_embed_model:
            try:
                import importlib
                import os as _os

                _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _os.environ.setdefault("HF_HUB_OFFLINE", "1")
                st_kwargs: dict[str, Any] = {}
                if cfg.memory.local_embed_cache_dir:
                    st_kwargs["cache_folder"] = cfg.memory.local_embed_cache_dir
                st_module = importlib.import_module("sentence_transformers")
                sentence_transformer = st_module.SentenceTransformer
                local_st = sentence_transformer(cfg.memory.local_embed_model, **st_kwargs)

                def _local_embed(texts: list[str]) -> list[list[float]]:
                    return local_st.encode(texts, normalize_embeddings=True).tolist()

                embed_fn = _local_embed
            except Exception as exc:
                _log.warning("[loop] 本地 embedding 模型加载失败，回退到 API: %s", exc)
                embed_fn = (
                    self._provider.embed
                    if cfg.memory.embedding_model and isinstance(self._provider, EmbeddingProvider)
                    else None
                )
        elif cfg.memory.embedding_model:
            embed_fn = self._provider.embed if isinstance(self._provider, EmbeddingProvider) else None
        self._semantic = SemanticMemory(
            cfg.memory_dir,
            decay_lambda=cfg.memory.semantic_decay_lambda,
            embed_fn=embed_fn,
            embedding_weight=cfg.memory.embedding_weight,
            source_weight=cfg.memory.semantic_source_weight,
            temporal_weight=cfg.memory.semantic_temporal_weight,
            temporal_window_days=cfg.memory.semantic_temporal_window_days,
        )

        # 子系统:Soul 文件管理 + 行为模式追踪
        self._soul = SoulManager(self._cfg, self._task_store, self._wm)
        self._behavior = BehaviorTracker(
            wait_streak_notify=list(cfg.loop.wait_streak_notify),
            streak_threshold=cfg.loop.behavior_streak_threshold,
            wm_priorities={
                "behavior_loop": cfg.thresholds.wm_pri_user_msg,
                "edit_caution": cfg.thresholds.wm_pri_self_aware,
                "belief_stale": cfg.thresholds.wm_pri_critical,
            },
            registry=self._registry,
            seq_window_warn_at=cfg.thresholds.behavior_seq_window_warn_at,
            seq_window_gap_ratio=cfg.thresholds.behavior_seq_window_gap_ratio,
            belief_stale_threshold=cfg.thresholds.behavior_belief_stale_threshold,
            belief_window=cfg.thresholds.behavior_belief_window,
        )

        # 自驱力引擎 (Active Inference + Intrinsic Motivation)
        from core.loop.drive.self_drive import SelfDriveEngine

        self._self_drive = SelfDriveEngine(str(cfg.db_path))

        # tick 间连续性追踪(预测误差 + 认知信号计算用)
        self._last_next_step: str = ""
        self._last_decision: str = "wait"
        self._last_act_progressful: bool = False
        self._last_act_progress_reason: str = ""  # LLM 可见的进展判断原因
        self._last_action_tool: str = ""
        self._last_action_key: str = ""
        self._last_action_status: str = ""
        self._last_action_summary: str = ""
        self._last_action_error: str = ""
        self._last_action_state_delta: str = ""
        self._success_stall_task_id: str | None = None
        self._success_stall_streak: int = 0
        self._recent_action_feedback: deque[str] = deque(maxlen=3)
        self._last_action_sig: str = ""
        self._last_result_fp: str = ""
        self._idle_cycles: int = 0
        self._last_curiosity_signal_idle_cycle: int = 0

        # 多轮对话历史(最多保留 6 轮 user/assistant 对)
        self._conv_history: deque[tuple[str, str]] = deque(maxlen=6)
        # 心跳计时(monotonic,独立于用户 cron,不存 DB)
        self._last_heartbeat_at: float = 0.0
        # bootstrap 模式（由 soul.bootstrap() 在 open/run 时写入）
        # "full" = 首次运行；"none" = 正常运行（BOOTSTRAP.md 已删除）
        self._bootstrap_mode: str = "none"
        # 探针系统：配置来自工作区 probes.json（与主 DB 完全解耦）
        probe_file = cfg.workspace_dir / "probes.json"
        self._probe_manager: ProbeManager = ProbeManager(probe_file)
        self._judgment._assembler._probe_manager = self._probe_manager
        # 按请求计费聚合:追踪距上次真正调用 LLM 已经过了几轮
        self._ticks_since_judge: int = 0
        # 当前执行链标识(由 _run_chain_job 临时注入)
        self._current_chain_key: str = ""
        # LLM 通过 model_strategy.next_phase_tier 跨 tick 传递的 tier 偏好
        self._pending_tier: str | None = None
        self._pending_idle_gap: float | None = None
        # LLM 通过 routing_overrides 临时覆盖 tier→model
        self._pending_routing_overrides: dict[str, str] | None = None
        # LLM 通过 thinking_override 覆盖下轮 thinking 等级
        self._pending_thinking_override: str | None = None

        cfg_file = cfg._base_dir / "lingzhou.json"
        self._cfg_file: Path = cfg_file
        self._cfg_mtime: float = cfg_file.stat().st_mtime if cfg_file.exists() else 0.0

        # 同时监听 auth-profiles.json(token 更新时重建 provider)
        from store.auth import AUTH_PROFILES_PATH

        self._auth_profiles_path: Path = AUTH_PROFILES_PATH
        self._auth_profiles_mtime: float = (
            AUTH_PROFILES_PATH.stat().st_mtime if AUTH_PROFILES_PATH.exists() else 0.0
        )

        # 并发 tick 调度：由 cfg.loop.max_concurrent_ticks 控制；默认配置为 4。
        # 同一 chain 内仍严格 FIFO，不同 chain 才会并行。
        self._tick_dispatcher = ConcurrentTickDispatcher(
            self,
            max_concurrent=cfg.loop.max_concurrent_ticks,
            max_queue=cfg.loop.max_tick_queue,
        )
        self._dispatch_cycle: int = 0
        self._dispatch_cycle_lock = asyncio.Lock()
        self._dispatch_state_lock = asyncio.Lock()
        self._chain_runtime_state: dict[str, dict[str, Any]] = {}

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
        cfg, routing_summary = await _prepare_runtime_run_impl(self)

        console.print(
            Panel(
                f"[bold green]lingzhou[/bold green] 启动\n"
                f"provider={cfg.model}  idle_gap={cfg.loop.max_idle_gap}ms  "
                f"act={'yes' if cfg.loop.act else 'dry-run'}\n"
                f"routing:\n{routing_summary}",
                title="🌱 认知循环",
            )
        )

        cycle = 0
        consecutive_errors = 0

        try:
            while True:
                try:
                    cycle = await _run_cycle_impl(self, cycle)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    console.print_exception(max_frames=5)
                    if consecutive_errors >= cfg.loop.max_consecutive_errors:
                        console.print(f"[red]连续错误 {consecutive_errors} 次,暂停循环[/red]")
                        break

                try:
                    await _wait_after_cycle_impl(self)
                except Exception:
                    _log.exception("[loop] _wait_after_cycle_impl 异常，跳过本次等待")
                    await asyncio.sleep(1.0)  # 防止异常紧循环消耗 CPU
                cfg = self._cfg  # 可能已更新
        finally:
            if self._tick_dispatcher.enabled:
                await self._tick_dispatcher.shutdown()
            self._probe_manager.stop()
            await self._task_store.close()
            await self._provider.close()
            for routing_provider in self._routing_providers.values():
                try:
                    await routing_provider.close()
                except Exception:
                    _log.exception("[loop] 关闭 routing provider 失败")
            # 干净退出：更新 survival.json 的 exit_type，下次启动不触发崩溃注入
            try:
                import json

                snapshot_path = self._cfg.state_dir / "survival.json"
                if snapshot_path.exists():
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    snapshot["exit_type"] = "clean"
                    snapshot_path.write_text(
                        json.dumps(snapshot, ensure_ascii=False),
                        encoding="utf-8",
                    )
            except Exception:
                pass

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
