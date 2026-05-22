"""core/loop/dispatcher.py - 有界 tick dispatcher。

语义：
1. 同一 chain 内严格 FIFO
2. 不同 chain 在全局并发上限内并行
3. 等待中的 job 总数受 max_queue 限制
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger("lingzhou.loop")


@dataclass(slots=True)
class TickJob:
    cycle: int
    chain_key: str
    user_message: str = ""
    chat_id: str | None = None
    source: str = "auto"


class ConcurrentTickDispatcher:
    def __init__(self, loop: Any, *, max_concurrent: int, max_queue: int) -> None:
        self._loop = loop
        self._max_concurrent = max(1, int(max_concurrent or 1))
        self._max_queue = max(1, int(max_queue or 1))
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._queues: dict[str, asyncio.Queue[TickJob]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._pending_count: int = 0
        self._running_count: int = 0

    @property
    def enabled(self) -> bool:
        return self._max_concurrent > 1

    @property
    def pending_count(self) -> int:
        return self._pending_count

    @property
    def running_count(self) -> int:
        return self._running_count

    def has_pending(self) -> bool:
        return self._pending_count > 0

    def has_running(self) -> bool:
        return self._running_count > 0

    async def shutdown(self) -> None:
        workers = list(self._workers.values())
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
        self._pending_count = 0
        self._running_count = 0

    async def enqueue(self, job: TickJob) -> bool:
        if self._pending_count >= self._max_queue:
            return False
        queue = self._queues.setdefault(job.chain_key, asyncio.Queue())
        queue.put_nowait(job)
        self._pending_count += 1
        worker = self._workers.get(job.chain_key)
        if worker is None or worker.done():
            self._workers[job.chain_key] = asyncio.create_task(self._run_chain(job.chain_key))
        return True

    async def _run_chain(self, chain_key: str) -> None:
        queue = self._queues.setdefault(chain_key, asyncio.Queue())
        try:
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                self._pending_count = max(0, self._pending_count - 1)
                async with self._semaphore:
                    self._running_count += 1
                    try:
                        await self._loop._run_dispatched_tick(job)
                    except Exception:
                        _log.exception(
                            "[tick-dispatch] chain=%s cycle=%s failed",
                            chain_key,
                            getattr(job, "cycle", 0),
                        )
                    finally:
                        self._running_count = max(0, self._running_count - 1)
        finally:
            self._workers.pop(chain_key, None)
