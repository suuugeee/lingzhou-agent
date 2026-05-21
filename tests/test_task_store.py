"""TaskStore 持久化测试"""
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
# TaskStore — JSON-first
# ══════════════════════════════════════════════════════════════════════════════

def test_task_store_basic():
    asyncio.run(_task_store_basic())

async def _task_store_basic():
    from memory.task_store import TaskStore
    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "test.db")
        await store.open()

        tid = await store.add_task("任务A", goal="目标", priority="high", source="external")
        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.goal == "目标"
        assert t.source == "external"
        assert t.next_step == ""

        await store.update_status(tid, "in_progress", "步骤1")
        t2 = await store.get_task_by_id(tid)
        assert t2 is not None
        assert t2.status == "in_progress"
        assert t2.next_step == "步骤1"

        # 扩展字段（无需 ALTER TABLE）
        await store.update_task_data(
            tid,
            {"tags": ["ai"], "score": 99, "model_tier": "reader", "current_step": "检查任务状态", "next_step": "不应覆盖"},
        )
        t3 = await store.get_task_by_id(tid)
        assert t3 is not None
        assert t3.extras["score"] == 99
        assert t3.model_tier == "reader"
        assert t3.current_step == "检查任务状态"
        assert t3.next_step == "步骤1"  # 原有字段未被覆盖

        await store.sync_task_progress(tid, current_step="步骤1", next_step="步骤2")
        t4 = await store.get_task_by_id(tid)
        assert t4 is not None
        assert t4.current_step == "步骤1"
        assert t4.next_step == "步骤2"

        await store.sync_task_progress(tid, next_step="")
        t5 = await store.get_task_by_id(tid)
        assert t5 is not None
        assert t5.next_step == ""

        await store.update_task_result(tid, {"summary": "first", "score": 1})
        await store.update_task_result(tid, {"last_run_status": "succeeded"})
        t6 = await store.get_task_by_id(tid)
        assert t6 is not None
        assert t6.result_json["summary"] == "first"
        assert t6.result_json["score"] == 1
        assert t6.result_json["last_run_status"] == "succeeded"

        await store.mark_waiting(tid, wait_kind="process", wait_key="exec-1")
        await store.resume_task(tid, result_json={"resumed_via": "task.resume"})
        t7 = await store.get_task_by_id(tid)
        assert t7 is not None
        assert t7.result_json["summary"] == "first"
        assert t7.result_json["last_run_status"] == "succeeded"
        assert t7.result_json["resumed_via"] == "task.resume"

        # 失败记录
        await store.record_failure("tool_error", "报错", context="ctx", task_id=str(tid))
        await store.record_failure("provider_error", "网络", task_id="")
        failures = await store.list_failures_for_task(str(tid))
        assert len(failures) == 2
        assert failures[0].summary == "网络"  # DESC: 最新在前

        # count_failures_by_kind
        assert await store.count_failures_by_kind("tool_error") == 1

        # facts
        import json
        await store.set_fact("soul:ethos_baseline", json.dumps({"truth": 0.85}))
        v, found = await store.get_fact("soul:ethos_baseline")
        assert found and json.loads(v)["truth"] == 0.85

        # enqueue_if_absent 去重
        a1 = await store.enqueue_if_absent("dup task")
        a2 = await store.enqueue_if_absent("dup task")
        assert a1 and not a2

        await store.close()


def test_task_store_run_lifecycle():
    asyncio.run(_task_store_run_lifecycle())


async def _task_store_run_lifecycle():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "runs.db")
        await store.open()
        task_id = await store.add_task("任务A", goal="目标")
        run_id = await store.add_run(
            task_id=task_id,
            run_type="tool_chain",
            worker_type="tool-chain-worker",
            input_json={"tool": "file.read"},
            tool_name="file.read",
            model_tier="reader",
            progress="queued",
        )

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"
        assert run.tool_name == "file.read"
        assert run.model_tier == "reader"
        assert run.progress == "queued"

        await store.update_run(
            run_id,
            status="succeeded",
            output_json={"summary": "ok"},
            log_text="ok",
            progress="done",
        )
        finished = await store.get_run_by_id(run_id)
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.output_json["summary"] == "ok"
        assert finished.progress == "done"
        assert finished.completed_at

        runs = await store.list_runs(task_id=task_id)
        assert len(runs) == 1

        await store.close()


def test_task_update_can_clear_runtime_fields():
    asyncio.run(_task_update_can_clear_runtime_fields())


async def _task_update_can_clear_runtime_fields():
    from memory.task_store import TaskStore
    from tools.task_ops import task_update

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "tasks.db")
        await store.open()
        task_id = await store.add_task(
            "清理任务状态",
            goal="验证 task.update 可以清空运行提示字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
            model_tier="reader",
        )

        ctx = _tool_ctx(task_store=store)
        res = await task_update(
            {
                "task_id": task_id,
                "status": "in_progress",
                "next_step": "",
                "current_step": "",
                "model_tier": "",
            },
            ctx,
        )

        assert res.error is None
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "in_progress"
        assert task.next_step == ""
        assert task.current_step == ""
        assert task.model_tier == ""

        await store.close()


def test_task_wait_resume_can_clear_runtime_fields():
    asyncio.run(_task_wait_resume_can_clear_runtime_fields())


async def _task_wait_resume_can_clear_runtime_fields():
    from memory.task_store import TaskStore
    from tools.task_ops import task_resume, task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-resume.db")
        await store.open()
        task_id = await store.add_task(
            "等待恢复任务",
            goal="验证 waiting/resume 可以显式清空步骤字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
        )

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "process",
                "wait_key": "exec-1",
                "current_step": "",
                "next_step": "",
            },
            ctx,
        )
        assert wait_res.error is None

        waited = await store.get_task_by_id(task_id)
        assert waited is not None
        assert waited.status == "waiting"
        assert waited.current_step == ""
        assert waited.next_step == ""

        resume_res = await task_resume(
            {
                "task_id": task_id,
                "status": "ready",
                "current_step": "",
                "next_step": "",
            },
            ctx,
        )
        assert resume_res.error is None

        resumed = await store.get_task_by_id(task_id)
        assert resumed is not None
        assert resumed.status == "ready"
        assert resumed.current_step == ""
        assert resumed.next_step == ""

        await store.close()


def test_chat_messages_are_sanitized_on_write():
    asyncio.run(_chat_messages_are_sanitized_on_write())


def test_task_wait_allows_external_wait_without_wait_key():
    asyncio.run(_task_wait_allows_external_wait_without_wait_key())


async def _task_wait_allows_external_wait_without_wait_key():
    from memory.task_store import TaskStore
    from tools.task_ops import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-guard.db")
        await store.open()
        task_id = await store.add_task("等待外部路径", goal="验证 task.wait 不会强制要求 wait_key")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "external",
            },
            ctx,
        )

        assert wait_res.skipped is False
        assert wait_res.error is None

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "waiting"
        assert task.wait_kind == "external"
        assert task.wait_key == ""

        await store.close()


def test_task_wait_rejects_unknown_wait_kind():
    asyncio.run(_task_wait_rejects_unknown_wait_kind())


async def _task_wait_rejects_unknown_wait_kind():
    from memory.task_store import TaskStore
    from tools.task_ops import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-kind.db")
        await store.open()
        task_id = await store.add_task("非法等待类型", goal="验证 task.wait 只接受受支持的等待类型")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "missing-evidence",
                "wait_key": "source-path",
            },
            ctx,
        )

        assert wait_res.skipped is True
        assert "不支持的 wait_kind" in wait_res.summary

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "pending"

        await store.close()


def test_task_steer_inbox_is_consumed_once():
    asyncio.run(_task_steer_inbox_is_consumed_once())


async def _task_steer_inbox_is_consumed_once():
    from memory.task_store import TaskStore
    from tools.task_ops import task_steer

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer.db")
        await store.open()
        task_id = await store.add_task("转向任务", goal="验证 steering inbox 只消费一次")

        ctx = _tool_ctx(task_store=store)
        steer_res = await task_steer(
            {
                "task_id": task_id,
                "message": "改成先读取配置文件，再决定是否继续旧计划",
            },
            ctx,
        )

        assert steer_res.error is None

        steered = await store.get_task_by_id(task_id)
        assert steered is not None
        assert steered.extras["inbox_messages"] == ["改成先读取配置文件，再决定是否继续旧计划"]

        inbox = await store.pop_task_inbox(task_id)
        assert inbox == ["改成先读取配置文件，再决定是否继续旧计划"]

        consumed = await store.get_task_by_id(task_id)
        assert consumed is not None
        assert consumed.extras.get("inbox_messages") == []

        inbox_again = await store.pop_task_inbox(task_id)
        assert inbox_again == []

        await store.close()


async def _chat_messages_are_sanitized_on_write():
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "chat.db")
        await store.open()
        await store.add_chat_message("user", "\x1b[31mhi\x1b[0m\ufeff\u200b\ufffd\x07there\r\n")
        await store.add_chat_message("user", "删 掉中文 后 就会 多 出 空格")
        msgs = await store.get_chat_messages_since(0)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hithere"
        assert msgs[1]["content"] == "删掉中文后就会多出空格"
        await store.close()


def test_task_store_migration():
    asyncio.run(_task_store_migration())

async def _task_store_migration():
    """旧列式 schema → JSON-first 自动迁移。"""
    from memory.task_store import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "old.db"
        async with aiosqlite.connect(str(db_path)) as db:
            await db.executescript("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    goal TEXT DEFAULT '',
                    priority TEXT DEFAULT 'normal',
                    status TEXT DEFAULT 'pending',
                    source TEXT DEFAULT 'external',
                    next_step TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    task_id TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE facts (
                    key TEXT PRIMARY KEY,
                    value TEXT DEFAULT '',
                    scope TEXT DEFAULT 'general',
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO tasks (title, goal, source, next_step)
                    VALUES ('旧任务', '旧目标', 'external', '旧步骤');
                INSERT INTO failures (kind, summary, context, task_id)
                    VALUES ('old_error', '旧摘要', '旧上下文', '1');
            """)
            await db.commit()

        store = TaskStore(db_path)
        await store.open()

        tasks = await store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "旧任务"
        assert tasks[0].goal == "旧目标"
        assert tasks[0].source == "external"
        assert tasks[0].next_step == "旧步骤"

        failures = await store.list_failures()
        assert len(failures) == 1
        assert failures[0].summary == "旧摘要"
        assert failures[0].context == "旧上下文"
        assert failures[0].task_id == "1"

        await store.close()


# ══════════════════════════════════════════════════════════════════════════════
