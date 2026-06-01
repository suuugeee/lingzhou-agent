"""core/loop/runs/driver.py — Run 类型路由层（Phase 3b/3c）。

职责：按 run_type 将执行请求分发到对应执行器，成为 tick → executor 的单一入口。

Phase 3b（已完成）：thin wrapper，代理 ExecutionLayer；引入 run_type → 默认档位映射表。
Phase 3c（当前）：从 provider/models.json 的 run_type_routing 段加载档位映射，替换硬编码。
Phase 3d：在此处从 pending Run queue 驱动执行，不再由 tick 直接调用。
"""
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolContext, ToolResult

from ..cycle.dispatcher import TickJob
from ..cycle.focus import resolve_focus_task

_log = logging.getLogger("lingzhou.run_driver")

# ── 内置兜底映射（当 models.json 无 run_type_routing 时使用）────────────────
# "task_default" 表示继承活跃任务的 model_tier；其他值直接选路由档位。
_RUN_TYPE_DEFAULT_TIER: dict[str, str] = {
    "judge":      "reader",
    "tool_chain": "task_default",
    "chat_reply": "reader",
    "evolve":     "reasoner",
    "subagent":   "task_default",
    "probe":      "reader",
    "llm":        "reader",
    "exec":       "task_default",
    "multimodal": "task_default",
}


def _load_catalog_routing() -> dict[str, str]:
    """加载 catalog 的 run_type → tier 映射，失败时返回空映射。"""
    try:
        from provider.catalog import get_run_type_routing

        routing = get_run_type_routing()
        if isinstance(routing, dict):
            return {
                str(k): str(v)
                for k, v in routing.items()
                if isinstance(k, str) and isinstance(v, str)
            }
    except Exception:
        pass
    return {}


class RunDriver:
    """按 run_type 路由执行的单一入口。

    Phase 3b：thin wrapper over ExecutionLayer。
    Phase 3c：从 models.json 加载 run_type → tier 映射，替换硬编码表。
    Phase 3d：将由 pending Run queue 驱动，不再由 tick 直接调用。
    """

    def __init__(self, execution: ExecutionLayer) -> None:
        self._execution = execution
        # Phase 3c：合并 catalog 路由（catalog 优先）与内置兜底，任何异常均退回内置
        try:
            _catalog = _load_catalog_routing()
        except Exception:
            _catalog = {}
        # 三层合并：内置兜底 → catalog → Config（Config 最高优先级）
        _config_routing: dict[str, str] = {}
        try:
            _cfg = getattr(execution, "_cfg", None)
            _rt = getattr(_cfg, "run_type_routing", None) if _cfg is not None else None
            if isinstance(_rt, dict):
                _config_routing = {k: v for k, v in _rt.items() if isinstance(k, str) and isinstance(v, str)}
        except Exception:
            pass
        self._tier_routing: dict[str, str] = {**_RUN_TYPE_DEFAULT_TIER, **_catalog, **_config_routing}

    def default_tier_for(self, run_type: str) -> str:
        """返回 run_type 对应默认档位，不存在时回退 task_default。"""
        key = str(run_type or "").strip()
        if not key:
            return "task_default"
        return self._tier_routing.get(key, "task_default")

    async def dispatch(
        self,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        """分发执行请求。

        Phase 3b/3c：直接委托 ExecutionLayer.dispatch()，保持现有行为。
        Phase 3d：将由 pending Run 驱动，不再由 tick 直接调用。
        """
        return await self._execution.dispatch(action, ctx)

    async def poll_pending_runs(self, loop: Any, cycle: int) -> int | None:
        """轮询 DB pending Runs，认领 judge 类型并注入 TickJob（Phase 3d）。

        返回 新的 cycle 值（已认领并入队），若无待处理 pending Run 则返回 None。
        """
        try:
            task_store = getattr(loop, "_task_store", None)
            if task_store is None:
                return None
            pending_runs = await task_store.get_pending_runs(limit=1)
            if not pending_runs:
                return None
            run = pending_runs[0]
            if run.run_type != "judge":
                # 非 judge 类型的 pending Run 暂不处理（留给未来扩展）
                return None
            # 认领：pending → running（started_at 由 update_run 通过写入 completed_at 时机判断）
            # 注意：update_run 只在终态时设置 completed_at，running 不设置。这里用 log_text 记录认领时间
            await task_store.update_run(
                run.id,
                status="running",
                log_text="[poll] claimed by run_driver",
            )
            # 注入 TickJob 到 dispatcher
            dispatcher = getattr(loop, "_tick_dispatcher", None)
            if dispatcher is not None and dispatcher.enabled:
                active_task = None
                with contextlib.suppress(Exception):
                    active_task = await resolve_focus_task(loop)
                dispatch_cycle = cycle
                with contextlib.suppress(Exception):
                    dispatch_cycle = await loop._next_dispatch_cycle()
                chain_key = "default"
                with contextlib.suppress(Exception):
                    chain_key = loop._resolve_tick_chain_key(active_task=active_task, source="poll")
                accepted = await dispatcher.enqueue(
                    TickJob(cycle=dispatch_cycle, chain_key=chain_key, source="poll")
                )
                if not accepted:
                    _log.debug("[poll] dispatcher queue full, pending Run #%d 回退到 pending", run.id)
                    await task_store.update_run(run.id, status="pending", log_text="[poll] queue full, requeue")
                    return None
                # TickJob 已入队，bootstrap Run 使命完成 → succeeded
                with contextlib.suppress(Exception):
                    await task_store.update_run(
                        run.id,
                        status="succeeded",
                        log_text="[poll] TickJob enqueued, bootstrap Run completed",
                    )
                _log.debug("[poll] pending Run #%d → succeeded，已注入 TickJob cycle=%d", run.id, dispatch_cycle)
                return dispatch_cycle
            else:
                # 无 dispatcher 时直接 tick
                new_cycle = cycle + 1
                with contextlib.suppress(Exception):
                    await loop._tick(new_cycle)
                # 直接 tick 也算完成
                with contextlib.suppress(Exception):
                    await task_store.update_run(
                        run.id,
                        status="succeeded",
                        log_text="[poll] direct tick completed, bootstrap Run completed",
                    )
                _log.debug("[poll] pending Run #%d → succeeded，直接 tick cycle=%d", run.id, new_cycle)
                return new_cycle
        except Exception:
            _log.exception("[poll_pending_runs] 失败，跳过")
            return None
