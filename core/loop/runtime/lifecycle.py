"""运行时生命周期。

生命周期负责启动准备、主循环和关闭清理；CognitionLoop 只作为 façade 暴露入口。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ..cycle.driver import _run_cycle_impl, _wait_after_cycle_impl
from .startup import _prepare_runtime_run_impl

_console = Console()
_log = logging.getLogger("lingzhou.loop")


async def run_runtime_forever(loop: Any) -> None:
    """运行完整认知生命周期，直到错误阈值或外部取消。"""
    cfg, routing_summary = await _prepare_runtime_run_impl(loop)
    _print_startup_panel(cfg, routing_summary)
    _invoke_runtime_ready_callback(loop)

    cycle = 0
    consecutive_errors = 0

    try:
        while True:
            try:
                cycle = await _run_cycle_impl(loop, cycle)
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                _console.print_exception(max_frames=5)
                if consecutive_errors >= cfg.loop.max_consecutive_errors:
                    _console.print(f"[red]连续错误 {consecutive_errors} 次,暂停循环[/red]")
                    break

            try:
                await _wait_after_cycle_impl(loop)
            except Exception:
                _log.exception("[loop] _wait_after_cycle_impl 异常，跳过本次等待")
                await asyncio.sleep(1.0)
            cfg = loop._cfg
    finally:
        await shutdown_runtime(loop)


async def shutdown_runtime(loop: Any) -> None:
    """关闭运行时器官并记录干净退出。"""
    dispatcher = getattr(loop, "_tick_dispatcher", None)
    if getattr(dispatcher, "enabled", False):
        await dispatcher.shutdown()
    loop._probe_manager.stop()
    await loop._task_store.close()
    embedding_provider = getattr(loop, "_embedding_provider", None)
    if embedding_provider is not None and embedding_provider is not loop._provider:
        await _close_auxiliary_provider(embedding_provider, label="embedding provider")
    await loop._provider.close()
    for routing_provider in loop._routing_providers.values():
        await _close_auxiliary_provider(routing_provider, label="routing provider")
    _mark_clean_exit(loop)


async def _close_auxiliary_provider(provider: Any, *, label: str) -> None:
    try:
        await provider.close()
    except Exception:
        _log.exception("[loop] 关闭 %s 失败", label)


def _print_startup_panel(cfg: Any, routing_summary: str) -> None:
    _console.print(
        Panel(
            f"[bold green]lingzhou[/bold green] 启动\n"
            f"provider={cfg.model}  idle_gap={cfg.loop.max_idle_gap}ms  "
            f"act={'yes' if cfg.loop.act else 'dry-run'}\n"
            f"routing:\n{routing_summary}",
            title="🌱 认知循环",
        )
    )


def _invoke_runtime_ready_callback(loop: Any) -> None:
    ready_callback = loop._runtime_ready_callback
    loop._runtime_ready_callback = None
    if ready_callback is None:
        return

    callback_started = time.monotonic()
    _log.info("[startup] runtime ready; invoking ready callback")
    try:
        ready_callback()
    except Exception:
        _log.exception("[startup] runtime ready callback failed")
        raise
    _log.info("[startup] ready callback done dt=%.3fs", time.monotonic() - callback_started)


def _mark_clean_exit(loop: Any) -> None:
    try:
        snapshot_path = loop._cfg.state_dir / "survival.json"
        if not snapshot_path.exists():
            return
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["exit_type"] = "clean"
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        _log.debug("[survival] 标记干净退出失败", exc_info=True)
