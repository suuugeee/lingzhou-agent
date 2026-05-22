from __future__ import annotations

import asyncio

import pytest

from core.loop.dispatcher import ConcurrentTickDispatcher, TickJob


class _FakeLoop:
    def __init__(self) -> None:
        self.started: list[tuple[str, int]] = []
        self.finished: list[tuple[str, int]] = []
        self._events: dict[tuple[str, int], asyncio.Event] = {}

    def event_for(self, chain_key: str, cycle: int) -> asyncio.Event:
        event = asyncio.Event()
        self._events[(chain_key, cycle)] = event
        return event

    async def _run_dispatched_tick(self, job: TickJob) -> None:
        key = (job.chain_key, job.cycle)
        self.started.append(key)
        await self._events[key].wait()
        self.finished.append(key)


@pytest.mark.asyncio
async def test_dispatcher_preserves_fifo_within_chain_and_allows_cross_chain_parallelism():
    loop = _FakeLoop()
    dispatcher = ConcurrentTickDispatcher(loop, max_concurrent=2, max_queue=4)

    a1 = loop.event_for("chain:a", 1)
    a2 = loop.event_for("chain:a", 2)
    b1 = loop.event_for("chain:b", 3)

    assert await dispatcher.enqueue(TickJob(cycle=1, chain_key="chain:a")) is True
    assert await dispatcher.enqueue(TickJob(cycle=2, chain_key="chain:a")) is True
    assert await dispatcher.enqueue(TickJob(cycle=3, chain_key="chain:b")) is True

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert ("chain:a", 1) in loop.started
    assert ("chain:b", 3) in loop.started
    assert ("chain:a", 2) not in loop.started

    a1.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert ("chain:a", 2) in loop.started

    b1.set()
    a2.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert loop.finished == [("chain:a", 1), ("chain:b", 3), ("chain:a", 2)] or loop.finished == [("chain:b", 3), ("chain:a", 1), ("chain:a", 2)]


@pytest.mark.asyncio
async def test_dispatcher_rejects_when_queue_is_full():
    loop = _FakeLoop()
    dispatcher = ConcurrentTickDispatcher(loop, max_concurrent=1, max_queue=1)

    loop.event_for("chain:a", 1)
    loop.event_for("chain:b", 2)

    assert await dispatcher.enqueue(TickJob(cycle=1, chain_key="chain:a")) is True
    assert await dispatcher.enqueue(TickJob(cycle=2, chain_key="chain:b")) is False