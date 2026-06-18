"""core/loop/runs/driver.py — run_type 分发入口."""
from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolContext, ToolResult

from core.execution.run_profile import (
    RUN_TYPE_JUDGE,
    resolve_default_tier_for_run_type,
)
from core.execution.routing import resolve_run_type_routing
from core.metabolic import update_run
from ..cycle.focus import try_dispatch_tick_job

_log = logging.getLogger("lingzhou.run_driver")

_RUN_BOOTSTRAP_SOURCES: dict[str, str] = {
    "claim": "loop/runs/driver/poll_pending_runs",
    "requeue": "loop/runs/driver/poll_pending_runs_requeue",
    "enqueued": "loop/runs/driver/poll_pending_runs_enqueued",
    "tick": "loop/runs/driver/poll_pending_runs_tick",
}


async def _update_polled_run(loop: Any, run_id: int, *, status: str, log_text: str, source: str) -> None:
    await update_run(
        loop,
        run_id,
        status=status,
        log_text=log_text,
        source=source,
        proposal_run_id=run_id,
    )


async def _update_bootstrap_run(
    loop: Any,
    run_id: int,
    status: str,
    source_key: str,
    message: str,
) -> None:
    await _update_polled_run(
        loop,
        run_id,
        status=status,
        log_text=message,
        source=_RUN_BOOTSTRAP_SOURCES.get(source_key, source_key),
    )


class RunDriver:
    """按 run_type 路由执行的单一入口。"""

    def __init__(self, execution: ExecutionLayer) -> None:
        self._execution = execution
        shared_routing = getattr(execution, "_run_type_routing", None)
        if isinstance(shared_routing, dict):
            self._run_type_routing = dict(shared_routing)
        else:
            _cfg = getattr(execution, "_cfg", None)
            self._run_type_routing = resolve_run_type_routing(_cfg)

    def default_tier_for(self, run_type: str) -> str:
        """返回 run_type 对应默认档位，不存在时回退 task_default。"""
        return resolve_default_tier_for_run_type(run_type, self._run_type_routing)

    async def dispatch(
        self,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        """分发执行请求。
        """
        return await self._execution.dispatch(action, ctx)

    async def poll_pending_runs(self, loop: Any, cycle: int) -> int | None:
        """轮询 DB pending Runs，认领 judge 类型并注入 TickJob。

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
            if run.run_type != RUN_TYPE_JUDGE:
                # 非 judge 类型的 pending Run 暂不处理（留给未来扩展）
                return None
            # 认领：pending → running（started_at 由 update_run 通过写入 completed_at 时机判断）
            # 注意：update_run 只在终态时设置 completed_at，running 不设置。这里用 log_text 记录认领时间
            await _update_bootstrap_run(
                loop,
                run.id,
                status="running",
                source_key="claim",
                message="[poll] claimed by run_driver",
            )
            # 注入 TickJob 到 dispatcher
            dispatch_result = await try_dispatch_tick_job(
                loop,
                cycle,
                source="poll",
            )
            if dispatch_result.accepted:
                # TickJob 已入队，bootstrap Run 使命完成 → succeeded
                with contextlib.suppress(Exception):
                    await _update_bootstrap_run(
                        loop,
                        run.id,
                        status="succeeded",
                        source_key="enqueued",
                        message="[poll] TickJob enqueued, bootstrap Run completed",
                    )
                _log.debug(
                    "[poll] pending Run #%d → succeeded，已注入 TickJob cycle=%d",
                    run.id,
                    dispatch_result.context.dispatch_cycle,
                )
                return dispatch_result.context.dispatch_cycle

            if dispatch_result.can_retry:
                _log.debug("[poll] dispatcher queue full, pending Run #%d 回退到 pending", run.id)
                with contextlib.suppress(Exception):
                    await _update_bootstrap_run(
                        loop,
                        run.id,
                        status="pending",
                        source_key="requeue",
                        message="[poll] queue full, requeue",
                    )
                return None

            # 无 dispatcher 时直接 tick
            new_cycle = cycle + 1
            with contextlib.suppress(Exception):
                await loop._tick(new_cycle)
            # 直接 tick 也算完成
            with contextlib.suppress(Exception):
                await _update_bootstrap_run(
                    loop,
                    run.id,
                    status="succeeded",
                    source_key="tick",
                    message="[poll] direct tick completed, bootstrap Run completed",
                )
            _log.debug("[poll] pending Run #%d → succeeded，直接 tick cycle=%d", run.id, new_cycle)
            return new_cycle
        except Exception:
            _log.exception("[poll_pending_runs] 失败，跳过")
            return None
