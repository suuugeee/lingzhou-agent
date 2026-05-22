"""并发安全测试：_ScopedTaskStore / parallel dispatch / aiosqlite 行隔离"""
import asyncio
import builtins
import io
import json
import logging
import math
import os
import tempfile
import time
from functools import lru_cache
from datetime import datetime, UTC, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import aiosqlite
import pytest

from conftest import (
    _proj_root,
    _test_config,
    _tool_ctx,
    _execution_layer,
    _tool_registry,
    _judgment_output,
)
# ══════════════════════════════════════════════════════════════════════════════
# 并发安全测试
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. _ScopedTaskStore 单元测试 ──────────────────────────────────────────────

def test_compact_tool_history_keeps_same_list_reference():
    """tool_history 压缩必须原地修改，不能切断外层列表引用。"""
    from core.loop.continue_phase import _compact_tool_history

    history = [
        {"tool": f"tool-{idx}", "params": {}, "result": f"result-{idx}", "status": "ok", "error": ""}
        for idx in range(6)
    ]
    original_id = id(history)

    compacted = _compact_tool_history(history)

    assert compacted is history
    assert id(history) == original_id
    assert len(history) == 4
    assert history[0]["tool"] == "[compacted]"


def test_process_pending_chat_turn_defers_when_dispatch_queue_full_without_blocking(monkeypatch):
    asyncio.run(_process_pending_chat_turn_defers_when_dispatch_queue_full_without_blocking(monkeypatch))


async def _process_pending_chat_turn_defers_when_dispatch_queue_full_without_blocking(monkeypatch):
    from core.loop.chat import _process_pending_chat_turn

    released_ids: list[tuple[int, ...]] = []

    async def _unexpected_sleep(_delay: float):
        raise AssertionError("queue full 时不应在 chat 主循环里 sleep 阻塞")

    monkeypatch.setattr("core.loop.chat.asyncio.sleep", _unexpected_sleep)

    class _FakeStore:
        def __init__(self) -> None:
            self._popped = False

        async def pop_pending_chat_message(self):
            if self._popped:
                return None
            self._popped = True
            return {"id": 11, "content": "hello", "chat_id": "chat:test"}

        async def drain_pending_for_chat(self, chat_id: str, after_id: int):
            return []

        async def release_chat_messages(self, message_ids):
            released_ids.append(tuple(int(mid) for mid in message_ids))

        async def get_active(self):
            return None

    class _FakeDispatcher:
        enabled = True

        async def enqueue(self, job):
            return False

    async def _next_dispatch_cycle() -> int:
        return 8

    loop = SimpleNamespace(
        _task_store=_FakeStore(),
        _tick_dispatcher=_FakeDispatcher(),
        _cfg=SimpleNamespace(loop=SimpleNamespace(wechat_coalesce_delay=0, wake_poll_interval=200)),
        _next_dispatch_cycle=_next_dispatch_cycle,
        _resolve_tick_chain_key=lambda **kwargs: "chat:test",
    )

    cycle, handled = await _process_pending_chat_turn(loop, 7)

    assert handled is True
    assert cycle == 7
    assert released_ids == [(11,)]

def test_scoped_task_store_get_active_returns_pinned():
    """_ScopedTaskStore.get_active() 必须始终返回构造时传入的 pinned task。"""
    asyncio.run(_scoped_task_store_get_active_returns_pinned())


async def _scoped_task_store_get_active_returns_pinned():
    from core.loop.task_parallel import _ScopedTaskStore

    # inner store 返回 task_A
    task_a = SimpleNamespace(id=1, goal="goal-A")
    task_b = SimpleNamespace(id=2, goal="goal-B")

    class _InnerStore:
        async def get_active(self):
            return task_a  # inner 返回 A

    scoped = _ScopedTaskStore(_InnerStore(), task_b)  # pin 为 B

    result = await scoped.get_active()
    assert result is task_b, "scoped store 应返回 pinned task_b，而非 inner 的 task_a"


def test_scoped_task_store_delegates_other_methods():
    """除 get_active() 外的所有方法必须透传给 inner store。"""
    asyncio.run(_scoped_task_store_delegates_other_methods())


async def _scoped_task_store_delegates_other_methods():
    from core.loop.task_parallel import _ScopedTaskStore

    calls: list[str] = []

    class _InnerStore:
        async def update_status(self, task_id, status, next_step=None):
            calls.append(f"update_status:{task_id}:{status}")

        async def update_task_result(self, task_id, result_json):
            calls.append(f"update_task_result:{task_id}")

        async def add_run(self, **kwargs):
            calls.append("add_run")
            return 42

        async def get_active(self):
            calls.append("inner_get_active")
            return None

    task_pin = SimpleNamespace(id=99, goal="pin")
    scoped = _ScopedTaskStore(_InnerStore(), task_pin)

    await scoped.update_status(7, "done")
    await scoped.update_task_result(7, {"k": "v"})
    run_id = await scoped.add_run(task_id=7)
    _ = await scoped.get_active()  # 不应调用 inner

    assert "update_status:7:done" in calls
    assert "update_task_result:7" in calls
    assert "add_run" in calls
    assert run_id == 42
    # get_active 被 scoped 覆盖，inner 的 get_active 不应被调用
    assert "inner_get_active" not in calls


# ── 2. _run_one_task ctx 隔离测试 ─────────────────────────────────────────────

def test_run_one_task_dispatch_ctx_pins_own_task():
    """_run_one_task 传给 dispatch 的 ctx 必须把 get_active() 固定为本子任务。"""
    asyncio.run(_run_one_task_dispatch_ctx_pins_own_task())


async def _run_one_task_dispatch_ctx_pins_own_task():
    from core.loop.task_parallel import _run_one_task
    from core.judgment import JudgmentOutput
    from tools.registry import ToolResult

    # inner store 的 get_active() 返回"错误"任务（id=999）
    wrong_task = SimpleNamespace(id=999, goal="wrong")

    class _FakeStore:
        async def get_active(self):
            return wrong_task  # 若不 scoped，dispatch 会看到此任务

        async def update_task_result(self, *a, **kw): pass
        async def update_status(self, *a, **kw): pass

    seen_active_ids: list[int] = []

    class _MockJudgment:
        async def decide_continue(self, *, tool_history, user_message="",
                                  active_task=None, prefer_tier=None):
            await asyncio.sleep(0)  # 让出控制权，模拟真实 await
            if len(tool_history) == 1:
                return JudgmentOutput(decision="act",
                                      chosen_action_id="mock_tool", params={})
            return JudgmentOutput(decision="wait", rationale="done")

    class _MockExecution:
        async def dispatch(self, output, ctx):
            active = await ctx.task_store.get_active()
            seen_active_ids.append(active.id if active else -1)
            return ToolResult(summary="ok")

    target_task = SimpleNamespace(id=42, goal="target-goal")
    store = _FakeStore()
    loop = SimpleNamespace(
        _judgment=_MockJudgment(),
        _execution=_MockExecution(),
        _task_store=store,
    )
    ctx = _tool_ctx(task_store=store)
    spec = {"id": "T", "goal": "target-goal", "tools": [], "max_rounds": 3}

    await _run_one_task(target_task, spec, ctx, loop)

    assert seen_active_ids, "dispatch 应被调用至少一次"
    assert all(aid == 42 for aid in seen_active_ids), (
        f"dispatch 看到了错误的 active_task id: {seen_active_ids}（期望全为 42）"
    )


# ── 3. run_tasks_parallel 并发不污染测试 ──────────────────────────────────────

def test_run_tasks_parallel_each_task_sees_own_active():
    """asyncio.gather 并发两个子任务时，每个 dispatch 只能看到自己的 task_id。"""
    asyncio.run(_run_tasks_parallel_each_task_sees_own_active())


async def _run_tasks_parallel_each_task_sees_own_active():
    from memory.task_store import TaskStore
    from core.loop.task_parallel import run_tasks_parallel
    from core.judgment import JudgmentOutput
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "parallel.db")
        await store.open()

        # dispatch 记录：{ task_id_pinned: [seen_active_id, ...] }
        dispatch_log: list[tuple[int, int]] = []  # (spec_task_id, seen_active_id)

        class _MockJudgment:
            async def decide_continue(self, *, tool_history, user_message="",
                                      active_task=None, prefer_tier=None):
                await asyncio.sleep(0)  # 强制 interleave
                if len(tool_history) == 1:
                    return JudgmentOutput(
                        decision="act", chosen_action_id="mock", params={}
                    )
                return JudgmentOutput(decision="wait", rationale="done")

        class _MockExecution:
            async def dispatch(self, output, ctx):
                active = await ctx.task_store.get_active()
                dispatch_log.append(active.id if active else -1)
                return ToolResult(summary=f"result-for-{active.id if active else 'none'}")

        loop = SimpleNamespace(
            _judgment=_MockJudgment(),
            _execution=_MockExecution(),
            _task_store=store,
        )
        ctx = _tool_ctx(task_store=store)
        specs = [
            {"id": "alpha", "goal": "do alpha", "tools": [], "max_rounds": 3},
            {"id": "beta",  "goal": "do beta",  "tools": [], "max_rounds": 3},
        ]

        entries = await run_tasks_parallel(specs, ctx, loop)

        # 应创建 2 个子任务并各有 1 次 dispatch
        assert len(entries) == 2
        assert len(dispatch_log) == 2, f"期望 2 次 dispatch，实际 {dispatch_log}"

        # 两次 dispatch 看到的 task_id 必须不同（各自隔离）
        assert dispatch_log[0] != dispatch_log[1], (
            f"两个子任务 dispatch 看到了相同的 active_task id={dispatch_log[0]}，未隔离"
        )

        # 每个 task 的状态应写为 done
        tasks = await store.list_tasks(limit=10)
        subtasks = [t for t in tasks if t.source == "internal"]
        statuses = {t.id: t.status for t in subtasks}
        assert all(s == "done" for s in statuses.values()), f"部分子任务未完成: {statuses}"

        await store.close()


# ── 4. aiosqlite 并发写不同行测试 ────────────────────────────────────────────

def test_aiosqlite_concurrent_writes_different_rows():
    """两个协程并发对不同行做 N 次更新，最终各行数据正确，无跨行污染。"""
    asyncio.run(_aiosqlite_concurrent_writes())


async def _aiosqlite_concurrent_writes():
    from memory.task_store import TaskStore

    N = 10  # 每个协程的更新次数

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "concurrent.db")
        await store.open()

        # 创建两个任务行
        id_a = await store.add_task(title="task-A", goal="task-A", source="test")
        id_b = await store.add_task(title="task-B", goal="task-B", source="test")

        async def write_row(task_id: int):
            for i in range(N):
                await store.update_task_result(task_id, {"seq": i, "owner": task_id})
                await asyncio.sleep(0)  # 强制 interleave

        await asyncio.gather(write_row(id_a), write_row(id_b))

        row_a = await store.get_task_by_id(id_a)
        row_b = await store.get_task_by_id(id_b)

        # 各行的最终写入 seq 为 N-1（最后一次写）
        assert row_a.result_json.get("seq") == N - 1, f"row_a seq={row_a.result_json}"
        assert row_b.result_json.get("seq") == N - 1, f"row_b seq={row_b.result_json}"

        # 无跨行污染：owner 字段与 task_id 匹配
        assert row_a.result_json.get("owner") == id_a, f"row_a owner 污染: {row_a.result_json}"
        assert row_b.result_json.get("owner") == id_b, f"row_b owner 污染: {row_b.result_json}"

        await store.close()


# ── 5. parallel_actions 并发 dispatch 合并测试 ────────────────────────────────

def test_dispatch_parallel_merges_results():
    """_dispatch_parallel 并发执行两个工具，merged_summary 包含双方输出。"""
    asyncio.run(_dispatch_parallel_merges_results())


async def _dispatch_parallel_merges_results():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolRegistry, ToolResult

    # 构造 mock registry，get() 始终返回 None（走 ToolNotFound 路径）
    # 我们 patch _dispatch_act，让它直接返回 ToolResult（避免真实 DB/工具调用）
    registry = ToolRegistry()

    class _FakeCfg:
        class loop:
            debug = False

    layer = ExecutionLayer(registry, _FakeCfg())

    # 记录每次 _dispatch_act 收到的 action_id 和调用顺序
    call_log: list[str] = []

    async def _mock_dispatch_act(action, ctx):
        await asyncio.sleep(0)  # 强制 interleave
        call_log.append(action.chosen_action_id)
        return ToolResult(summary=f"ok-{action.chosen_action_id}")

    layer._dispatch_act = _mock_dispatch_act  # type: ignore[method-assign]

    action = JudgmentOutput(
        decision="act",
        chosen_action_id="__parallel__",
        parallel_actions=[
            {"action_id": "tool_x", "params": {}},
            {"action_id": "tool_y", "params": {}},
        ],
    )

    ctx = _tool_ctx()
    result = await layer._dispatch_parallel(action, ctx)

    # 两个工具都被调用
    assert "tool_x" in call_log, f"tool_x 未被 dispatch: {call_log}"
    assert "tool_y" in call_log, f"tool_y 未被 dispatch: {call_log}"

    # merged_summary 包含双方输出
    assert "ok-tool_x" in result.summary, f"summary 缺少 tool_x 输出: {result.summary!r}"
    assert "ok-tool_y" in result.summary, f"summary 缺少 tool_y 输出: {result.summary!r}"

    # kind 正确
    assert result.kind == "execute_result"

