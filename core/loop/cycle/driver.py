"""core.loop.cycle.driver — loop 生命周期调度与事件驱动等待。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from core.log_fields import format_log_fields

from ..runtime.reload import _maybe_hot_reload_provider_impl
from .chat import _process_pending_chat_turn
from .dispatcher import TickJob
from .focus import resolve_focus_task

_log = logging.getLogger("lingzhou.loop")


async def _run_cycle_impl(loop: Any, cycle: int) -> int:
    cycle, handled_chat = await _process_pending_chat_turn(loop, cycle)
    if not handled_chat:
        # Phase 3d：优先认领 DB pending Runs（bootstrap/崩溃恢复路径）
        run_driver = getattr(loop, "_run_driver", None)
        if run_driver is not None:
            polled_cycle = await run_driver.poll_pending_runs(loop, cycle)
            if polled_cycle is not None:
                return polled_cycle

        if getattr(loop, "_tick_dispatcher", None) is not None and loop._tick_dispatcher.enabled:
            dispatcher = loop._tick_dispatcher
            if dispatcher.has_running() or dispatcher.has_pending():
                _log.debug(
                    "[tick-dispatch] active work running=%d pending=%d, skip auto tick",
                    dispatcher.running_count,
                    dispatcher.pending_count,
                )
                return cycle
            active_task = await resolve_focus_task(loop)
            if active_task is None and not getattr(loop, "_auto_tick_due", True):
                _log.debug("[tick-dispatch] auto tick not due, wait for idle gap")
                return cycle
            dispatch_cycle = await loop._next_dispatch_cycle()
            chain_key = loop._resolve_tick_chain_key(active_task=active_task, source="auto")
            accepted = await dispatcher.enqueue(
                TickJob(cycle=dispatch_cycle, chain_key=chain_key, source="auto")
            )
            if accepted:
                if active_task is None:
                    loop._auto_tick_due = False
                cycle = dispatch_cycle
            else:
                _log.debug("[tick-dispatch] queue full, skip auto tick")
        else:
            cycle += 1
            await loop._tick(cycle)
    return cycle


async def _wait_after_cycle_impl(loop: Any) -> None:
    cfg = loop._cfg
    # arousal 调制系数：高唤醒→更短间隔（最多缩 20%），低唤醒→更长间隔（最多扩 20%）
    # 不干预 LLM 主动设置的 _pending_idle_gap，只影响配置兜底值
    _arousal = float(getattr(getattr(loop, "_emotion", None), "arousal", 0.5))
    _arousal_factor = max(cfg.loop.arousal_min_factor, 1.0 - cfg.loop.arousal_sensitivity * (_arousal - cfg.loop.arousal_neutral))

    dispatcher = getattr(loop, "_tick_dispatcher", None)
    if dispatcher is not None and dispatcher.enabled:
        has_work = dispatcher.has_running() or dispatcher.has_pending()
        after_task = await resolve_focus_task(loop)
        if has_work:
            gap = max(float(cfg.loop.min_act_gap) / 1000.0, 0.2)
        elif after_task is not None:
            gap = cfg.loop.active_idle_gap / 1000.0 * _arousal_factor
        else:
            gap = cfg.loop.max_idle_gap / 1000.0 * _arousal_factor
        await _wait_for_event_impl(loop, gap, after_task)
        if not has_work and after_task is None:
            loop._auto_tick_due = True
        await _maybe_hot_reload_provider_impl(loop)
        return

    after_task = await resolve_focus_task(loop)
    if loop._last_decision == "act" and after_task is not None:
        min_wait = cfg.loop.idle_with_task_bounds[0] / 1000.0 if cfg.loop.idle_with_task_bounds else cfg.loop.min_act_gap / 1000.0
        act_gap = max(float(min_wait), float(cfg.loop.min_act_gap) / 1000.0)
        await _wait_for_event_impl(loop, act_gap, after_task)
    else:
        # 等待间隔决策树：
        #   ① LLM 上轮主动要求 next_idle_gap_secs → 优先尊重 LLM 意图（不加 arousal 调制）
        #   ② 有活跃 task（上轮 decision≠act，或 act 后 task 仍在）→ active_idle_gap（较短，保持响应）
        #   ③ bootstrap 未完成 → 同 ② 缩短间隔（避免引导阶段空等 max_idle_gap）
        #   ④ 真正空闲（无 task、无 bootstrap）→ max_idle_gap（节省 CPU/计费）
        if loop._pending_idle_gap is not None:
            gap = loop._pending_idle_gap                                       # ① LLM 主动调度
        elif after_task is not None:
            gap = cfg.loop.active_idle_gap / 1000.0 * _arousal_factor         # ② 有活跃任务
        elif getattr(loop, '_bootstrap_mode', 'none') == "full":
            # bootstrap 未完成时，等同于有隐式未完成工作：缩短轮询间隔提升响应度
            gap = cfg.loop.active_idle_gap / 1000.0 * _arousal_factor         # ③ bootstrap 阶段
        else:
            gap = cfg.loop.max_idle_gap / 1000.0 * _arousal_factor            # ④ 完全空闲
        await _wait_for_event_impl(loop, gap, after_task)
    await _maybe_hot_reload_provider_impl(loop)


async def _wait_for_event_impl(loop: Any, max_wait: float, before_task: Any) -> None:
    """事件驱动等待: chat 消息、task 状态变化、探针告警、超时任一发生即唤醒。"""
    cfg = loop._cfg
    poll = max(cfg.loop.wake_poll_interval / 1000.0, 0.05)  # 最小 50ms，防止 wake_poll_interval=0 导致紧密轮询
    before_sig = (
        before_task.id if before_task else None,
        before_task.status if before_task else None,
    )
    event_loop = asyncio.get_running_loop()
    deadline = event_loop.time() + max_wait
    # 获取探针告警事件（ProbeManager 在 attach 时创建）
    _pm = getattr(loop, "_probe_manager", None)
    alert_event = getattr(_pm, "alert_event", None) if _pm is not None else None
    while True:
        remaining = deadline - event_loop.time()
        if remaining <= 0:
            break
        sleep_secs = min(poll, remaining)
        if alert_event is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(alert_event.wait(), timeout=sleep_secs)
        else:
            await asyncio.sleep(sleep_secs)
        if alert_event is not None and alert_event.is_set():
            alert_event.clear()
            _log.info("[wake] %s", format_log_fields(reason="probe_alert"))
            break
        pending_check_t0 = event_loop.time()
        has_pending_chat = await loop._task_store.has_pending_chat_message()
        pending_check_dt = event_loop.time() - pending_check_t0
        if pending_check_dt >= 1.0:
            _log.warning("[wake] has_pending_chat_message slow dt=%.3fs", pending_check_dt)
        if has_pending_chat:
            dispatcher = getattr(loop, "_tick_dispatcher", None)
            can_accept = getattr(dispatcher, "can_accept", None) if dispatcher is not None else None
            if dispatcher is None or not dispatcher.enabled or not callable(can_accept) or can_accept():
                _log.info("[wake] %s", format_log_fields(reason="chat_pending"))
                break
        if cfg.loop.wake_on_task_change:
            now = await resolve_focus_task(loop)
            now_sig = (now.id if now else None, now.status if now else None)
            if now_sig != before_sig:
                _log.info(
                    "[wake] %s",
                    format_log_fields(
                        reason="task_change",
                        task_before=before_sig[0],
                        status_before=before_sig[1],
                        task=now_sig[0],
                        status=now_sig[1],
                    ),
                )
                break
