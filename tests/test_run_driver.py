"""tests/test_run_driver.py — RunDriver 路由层单元测试（Phase 3b/3c）。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.execution import (
    RUN_TYPE_CHAT_REPLY,
    RUN_TYPE_DEFAULT_TIER,
    RUN_TYPE_EVOLUTION,
    RUN_TYPE_JUDGE,
    RUN_TYPE_PROBE,
    KNOWN_RUN_TYPES,
    RUN_TYPE_TOOL_CHAIN,
    TASK_DEFAULT_TIER,
)

# ─────────────────────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────────────────────

def _make_execution_mock():
    m = AsyncMock()
    m.dispatch = AsyncMock(return_value=MagicMock(error=None, summary="ok"))
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3b：RunDriver 路由层存在性与委托
# ─────────────────────────────────────────────────────────────────────────────

def test_run_driver_importable():
    from core.loop.runs.driver import RunDriver
    assert RunDriver is not None


def test_run_driver_has_dispatch_and_default_tier():
    from core.loop.runs.driver import RunDriver
    execution = _make_execution_mock()
    driver = RunDriver(execution)
    assert callable(driver.dispatch)
    assert callable(driver.default_tier_for)


@pytest.mark.asyncio
async def test_run_driver_dispatch_delegates_to_execution():
    from core.loop.runs.driver import RunDriver
    execution = _make_execution_mock()
    driver = RunDriver(execution)

    action = MagicMock()
    ctx = MagicMock()
    await driver.dispatch(action, ctx)

    execution.dispatch.assert_awaited_once_with(action, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3b：内置档位映射表完整性
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("run_type,expected_tier", tuple(RUN_TYPE_DEFAULT_TIER.items()))
def test_default_tier_for_known_run_types(run_type, expected_tier):
    from core.loop.runs.driver import RunDriver
    execution = _make_execution_mock()
    driver = RunDriver(execution)
    assert driver.default_tier_for(run_type) == expected_tier


def test_default_tier_for_unknown_run_type_uses_catalog_and_config_override():
    """未知 run_type 仍可通过 catalog/config 映射到自定义档位。"""
    from core.loop.runs.driver import RunDriver

    execution = _make_execution_mock()
    execution._cfg = MagicMock()
    execution._cfg.run_type_routing = {"custom_type": "reasoner", RUN_TYPE_JUDGE: "reader"}

    with patch(
        "core.loop.runs.driver.resolve_run_type_routing",
        return_value={RUN_TYPE_JUDGE: "reader", "custom_type": "reasoner"},
    ):
        driver = RunDriver(execution)

    assert driver.default_tier_for("custom_type") == "reasoner"
    assert driver.default_tier_for(RUN_TYPE_JUDGE) == "reader"
    assert driver.default_tier_for("") == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_TOOL_CHAIN]


def test_run_driver_prefers_execution_layer_routing_snapshot():
    """当 ExecutionLayer 已有路由快照时，RunDriver 复用快照，避免重复解析。"""
    from core.loop.runs.driver import RunDriver

    execution = _make_execution_mock()
    execution._run_type_routing = {RUN_TYPE_JUDGE: "reasoner", RUN_TYPE_TOOL_CHAIN: "task_default"}

    with patch("core.loop.runs.driver.resolve_run_type_routing") as resolve_mock:
        driver = RunDriver(execution)

    resolve_mock.assert_not_called()
    assert driver.default_tier_for(RUN_TYPE_JUDGE) == "reasoner"
    assert driver.default_tier_for(RUN_TYPE_PROBE) == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_PROBE]


def test_default_tier_for_unknown_run_type_returns_task_default():
    from core.loop.runs.driver import RunDriver
    execution = _make_execution_mock()
    driver = RunDriver(execution)
    assert driver.default_tier_for("nonexistent_type") == TASK_DEFAULT_TIER


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3c：从 catalog 加载路由表（mock catalog）
# ─────────────────────────────────────────────────────────────────────────────

def test_run_driver_merges_catalog_routing_over_defaults():
    """catalog 路由应覆盖内置默认值。"""
    from core.loop.runs.driver import RunDriver

    # 模拟 catalog 返回自定义映射
    custom_routing = {RUN_TYPE_JUDGE: "reasoner", "custom_type": "reader"}
    with patch(
        "core.loop.runs.driver.resolve_run_type_routing",
        return_value=custom_routing,
    ):
        # 直接注入路由快照
        execution = _make_execution_mock()
        driver = RunDriver(execution)

    assert driver.default_tier_for(RUN_TYPE_JUDGE) == "reasoner"      # catalog 覆盖
    assert driver.default_tier_for("custom_type") == "reader"  # catalog 新增
    assert driver.default_tier_for(RUN_TYPE_EVOLUTION) == "reasoner"     # 内置兜底


def test_run_driver_falls_back_to_defaults_when_catalog_empty():
    """catalog 为空时完全使用内置默认值。"""
    from core.loop.runs.driver import RunDriver

    with patch("core.loop.runs.driver.resolve_run_type_routing", return_value=dict(RUN_TYPE_DEFAULT_TIER)):
        execution = _make_execution_mock()
        driver = RunDriver(execution)

    # 应等于内置默认值
    for run_type, tier in RUN_TYPE_DEFAULT_TIER.items():
        assert driver.default_tier_for(run_type) == tier


def test_run_driver_tolerates_catalog_load_failure():
    """路由快照为空时，不应抛异常并回退到内置默认。"""
    from core.loop.runs.driver import RunDriver

    with patch("core.loop.runs.driver.resolve_run_type_routing", return_value={}):
        execution = _make_execution_mock()
        driver = RunDriver(execution)
        assert driver.default_tier_for(RUN_TYPE_JUDGE) == RUN_TYPE_DEFAULT_TIER[RUN_TYPE_JUDGE]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3c：catalog.get_run_type_routing 函数
# ─────────────────────────────────────────────────────────────────────────────

def test_get_run_type_routing_returns_dict():
    from provider.catalog import get_run_type_routing
    routing = get_run_type_routing()
    assert isinstance(routing, dict)
    # _doc 键应被过滤
    assert "_doc" not in routing


def test_get_run_type_routing_has_expected_keys():
    from provider.catalog import get_run_type_routing
    routing = get_run_type_routing()
    # models.json 中声明的 run_type 应全部存在
    for key in KNOWN_RUN_TYPES:
        assert key in routing, f"expected '{key}' in run_type_routing"


def test_get_run_type_routing_values_are_strings():
    from provider.catalog import get_run_type_routing
    routing = get_run_type_routing()
    for k, v in routing.items():
        assert isinstance(v, str), f"run_type_routing[{k!r}] should be str, got {type(v)}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3d：cancel_stale_runs 崩溃恢复
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_stale_runs_marks_old_running_as_cancelled():
    """超过 stale 阈值的 running/pending Run 应被标为 cancelled。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        # 插入一个 running run（started_at 设为足够早）
        run_id = await store.add_run(run_type=RUN_TYPE_TOOL_CHAIN, status="running")
        # 直接修改 DB 里的 started_at 为很早的时间
        import datetime
        old_ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=700)).isoformat()
        async with store._db.execute(
            "UPDATE runs SET started_at=? WHERE id=?",
            (old_ts, run_id),
        ):
            pass
        await store._db.commit()

        # 执行清理（阈值 600s）
        count = await store.cancel_stale_runs(stale_after_seconds=600)
        assert count == 1

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "cancelled"
        assert "stale run cancelled" in (run.error_text or "")
        await store.close()


@pytest.mark.asyncio
async def test_cancel_stale_runs_does_not_cancel_recent_runs():
    """最近的 running Run 不应被 cancel_stale_runs 影响。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        # 插入一个最近的 running run（started_at = 现在）
        run_id = await store.add_run(run_type=RUN_TYPE_TOOL_CHAIN, status="running")

        count = await store.cancel_stale_runs(stale_after_seconds=600)
        assert count == 0

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "running"  # 未被取消
        await store.close()


@pytest.mark.asyncio
async def test_cancel_stale_runs_ignores_terminal_runs():
    """已完成（succeeded/failed/cancelled）的 Run 不受影响。"""
    import datetime
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_TOOL_CHAIN, status="succeeded")
        old_ts = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=700)).isoformat()
        async with store._db.execute(
            "UPDATE runs SET started_at=? WHERE id=?",
            (old_ts, run_id),
        ):
            pass
        await store._db.commit()

        count = await store.cancel_stale_runs(stale_after_seconds=600)
        assert count == 0
        await store.close()


# ─────────────────────────────────────────────────────────────────────────────
# Config.run_type_routing 覆盖支持（Task 2）
# ─────────────────────────────────────────────────────────────────────────────

def test_run_driver_merges_config_run_type_routing_over_catalog():
    """Config.run_type_routing 的值应覆盖 catalog 内置映射。"""
    from core.loop.runs.driver import RunDriver

    execution = _make_execution_mock()
    # 模拟 cfg.run_type_routing = {RUN_TYPE_JUDGE: "reasoner"}
    cfg_mock = MagicMock()
    cfg_mock.run_type_routing = {RUN_TYPE_JUDGE: "reasoner"}
    execution._cfg = cfg_mock

    with patch("core.loop.runs.driver.resolve_run_type_routing") as resolve_mock:
        resolve_mock.return_value = {RUN_TYPE_JUDGE: "reasoner"}
        driver = RunDriver(execution)

    resolve_mock.assert_called_once_with(cfg_mock)

    # Config 覆盖优先于 catalog，judge → reasoner
    assert driver.default_tier_for(RUN_TYPE_JUDGE) == "reasoner"


def test_run_driver_config_routing_empty_falls_back_to_catalog():
    """Config.run_type_routing 为空时不影响 catalog 结果。"""
    from core.loop.runs.driver import RunDriver

    execution = _make_execution_mock()
    cfg_mock = MagicMock()
    cfg_mock.run_type_routing = {}
    execution._cfg = cfg_mock

    with patch("core.loop.runs.driver.resolve_run_type_routing") as resolve_mock:
        resolve_mock.return_value = {RUN_TYPE_PROBE: RUN_TYPE_CHAT_REPLY}
        driver = RunDriver(execution)

    resolve_mock.assert_called_once_with(cfg_mock)
    assert driver.default_tier_for(RUN_TYPE_PROBE) == RUN_TYPE_CHAT_REPLY


def test_config_run_type_routing_field_exists():
    """Config 应有 run_type_routing 字段，默认空 dict。"""
    from core.config import Config
    cfg = Config.model_validate({
        "providers": {"cp": {"type": "openai_compat", "base_url": "https://x", "api_key_env": "A"}},
        "model": "cp/m",
    })
    assert hasattr(cfg, "run_type_routing")
    assert isinstance(cfg.run_type_routing, dict)
    assert cfg.run_type_routing == {}


def test_config_run_type_routing_accepts_overrides():
    """lingzhou.json 可通过 run_type_routing 字段覆盖档位映射。"""
    from core.config import Config
    cfg = Config.model_validate({
        "providers": {"cp": {"type": "openai_compat", "base_url": "https://x", "api_key_env": "A"}},
        "model": "cp/m",
        "run_type_routing": {RUN_TYPE_JUDGE: "reasoner", RUN_TYPE_CHAT_REPLY: "reader"},
    })
    assert cfg.run_type_routing == {RUN_TYPE_JUDGE: "reasoner", RUN_TYPE_CHAT_REPLY: "reader"}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3d：get_pending_runs + poll_pending_runs + bootstrap
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_pending_runs_returns_pending_only():
    """get_pending_runs 只返回 status='pending' 的 Run。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        await store.add_run(run_type=RUN_TYPE_TOOL_CHAIN, status="running")
        await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")
        await store.add_run(run_type=RUN_TYPE_PROBE, status="pending")

        pending = await store.get_pending_runs(limit=10)
        assert len(pending) == 2
        assert all(r.status == "pending" for r in pending)
        run_types = {r.run_type for r in pending}
        assert run_types == {RUN_TYPE_JUDGE, RUN_TYPE_PROBE}
        await store.close()


@pytest.mark.asyncio
async def test_get_pending_runs_ordered_by_created_at():
    """get_pending_runs 按 created_at 升序返回（最早优先）。"""
    import asyncio
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        id1 = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")
        await asyncio.sleep(0.01)
        id2 = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")

        pending = await store.get_pending_runs(limit=10)
        assert len(pending) == 2
        assert pending[0].id == id1
        assert pending[1].id == id2
        await store.close()


@pytest.mark.asyncio
async def test_add_run_pending_has_started_at():
    """pending Run 的 started_at 应已写入（NOT NULL 约束，默认为创建时间）。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")
        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.started_at  # NOT NULL 应有字符串値
        await store.close()


@pytest.mark.asyncio
async def test_cancel_stale_runs_does_not_cancel_fresh_pending_run():
    """cancel_stale_runs 不应取消刚创建的 pending Run（started_at=当前时间，未过超时）。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")
        # stale_after_seconds=600：刚建的 Run 不会被取消
        count = await store.cancel_stale_runs(stale_after_seconds=600)
        assert count == 0

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "pending"
        await store.close()


@pytest.mark.asyncio
async def test_poll_pending_runs_claims_judge_run_and_enqueues_tick():
    """poll_pending_runs 找到 pending judge Run → 认领并注入 TickJob，返回新 cycle。"""
    import tempfile
    from pathlib import Path

    from core.loop.cycle.dispatcher import TickJob
    from core.loop.runs.driver import RunDriver
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")

        enqueued: list[TickJob] = []

        class _FakeDispatcher:
            enabled = True
            async def enqueue(self, job):
                enqueued.append(job)
                return True

        class _FakeLoop:
            _task_store = store
            _tick_dispatcher = _FakeDispatcher()
            async def _next_dispatch_cycle(self):
                return 42
            def _resolve_tick_chain_key(self, *, active_task, source):
                return "default"

        execution = _make_execution_mock()
        driver = RunDriver(execution)
        result = await driver.poll_pending_runs(_FakeLoop(), cycle=1)

        assert result == 42
        assert len(enqueued) == 1
        assert enqueued[0].source == "poll"

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "succeeded"  # TickJob 已入队，bootstrap Run 使命完成
        await store.close()


@pytest.mark.asyncio
async def test_poll_pending_runs_returns_none_when_no_pending():
    """无 pending Run 时 poll_pending_runs 返回 None。"""
    import tempfile
    from pathlib import Path

    from core.loop.runs.driver import RunDriver
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        class _FakeLoop:
            _task_store = store
            _tick_dispatcher = None

        execution = _make_execution_mock()
        driver = RunDriver(execution)
        result = await driver.poll_pending_runs(_FakeLoop(), cycle=5)
        assert result is None
        await store.close()


@pytest.mark.asyncio
async def test_poll_pending_runs_skips_non_judge_run():
    """poll_pending_runs 跳过非 judge 类型的 pending Run。"""
    import tempfile
    from pathlib import Path

    from core.loop.runs.driver import RunDriver
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_PROBE, status="pending")

        class _FakeLoop:
            _task_store = store
            _tick_dispatcher = None

        execution = _make_execution_mock()
        driver = RunDriver(execution)
        result = await driver.poll_pending_runs(_FakeLoop(), cycle=1)
        assert result is None

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "pending"
        await store.close()


@pytest.mark.asyncio
async def test_poll_pending_runs_restores_pending_when_queue_full():
    """dispatcher 队列满时，认领失败应将 Run 回退到 pending。"""
    import tempfile
    from pathlib import Path

    from core.loop.runs.driver import RunDriver
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        run_id = await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")

        class _FullDispatcher:
            enabled = True
            async def enqueue(self, job):
                return False

        class _FakeLoop:
            _task_store = store
            _tick_dispatcher = _FullDispatcher()
            async def _next_dispatch_cycle(self):
                return 10
            def _resolve_tick_chain_key(self, *, active_task, source):
                return "default"

        execution = _make_execution_mock()
        driver = RunDriver(execution)
        result = await driver.poll_pending_runs(_FakeLoop(), cycle=1)
        assert result is None

        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert run.status == "pending"
        await store.close()


@pytest.mark.asyncio
async def test_poll_pending_runs_uses_focus_task_chain_over_global_active():
    import tempfile
    from pathlib import Path

    from core.loop.cycle.focus import claim_focus_task
    from core.loop.runs.driver import RunDriver
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        await store.add_run(run_type=RUN_TYPE_JUDGE, status="pending")
        await store.add_task("全局活跃任务", goal="旧 get_active 会误命中这里", status="in_progress")
        focus_id = await store.add_task(
            "当前焦点任务",
            goal="poll 应沿 focus task chain 入队",
            status="pending",
            chain_id="focus-chain",
        )
        focus_task = await store.get_task_by_id(focus_id)
        assert focus_task is not None

        loop_ref = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop_ref, focus_task, clear_current=True)

        seen_chain_keys: list[str] = []

        class _FakeDispatcher:
            enabled = True

            async def enqueue(self, job):
                seen_chain_keys.append(job.chain_key)
                return True

        class _FakeLoop:
            _task_store = store
            _tick_dispatcher = _FakeDispatcher()

            async def _next_dispatch_cycle(self):
                return 42

            def _resolve_tick_chain_key(self, *, active_task, source):
                return f"task:{active_task.id}" if active_task is not None else "default"

        execution = _make_execution_mock()
        driver = RunDriver(execution)
        result = await driver.poll_pending_runs(_FakeLoop(), cycle=1)

        assert result == 42
        assert seen_chain_keys == [f"task:{focus_id}"]
        await store.close()


@pytest.mark.asyncio
async def test_startup_bootstrap_creates_pending_run():
    """startup 应在无 pending Run 时写入 bootstrap pending Run。"""
    import tempfile
    from pathlib import Path

    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(Path(tmpdir) / "test.db")
        await store.open()

        pending_before = await store.get_pending_runs(limit=10)
        assert len(pending_before) == 0

        # 模拟 startup bootstrap 逻辑
        existing = await store.get_pending_runs(limit=1)
        if not existing:
            await store.add_run(
                run_type=RUN_TYPE_JUDGE,
                status="pending",
                log_text="[startup] bootstrap pending Run",
            )

        pending_after = await store.get_pending_runs(limit=10)
        assert len(pending_after) == 1
        assert pending_after[0].run_type == RUN_TYPE_JUDGE
        assert pending_after[0].started_at  # started_at 应已设置（NOT NULL 字段）
        await store.close()
