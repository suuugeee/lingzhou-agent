import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace


def test_resolve_tick_dispatch_context_prefers_task_chain_over_chat_id():
    asyncio.run(_resolve_tick_dispatch_context_prefers_task_chain_over_chat_id())


async def _resolve_tick_dispatch_context_prefers_task_chain_over_chat_id():
    from core.loop.cycle.focus import resolve_tick_dispatch_context
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "dispatch-context-task.db")
        await store.open()
        task_id = await store.add_task(
            "任务链任务",
            goal="确保 chat 在同任务链上续跑",
            status="in_progress",
            chain_id="chain-chat",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        async def _next_dispatch_cycle() -> int:
            return 77

        loop = SimpleNamespace(
            _task_store=store,
            _next_dispatch_cycle=_next_dispatch_cycle,
            _resolve_tick_chain_key=lambda **kwargs: (
                f"task-chain:{kwargs['active_task'].chain_id}"
                if kwargs.get("active_task") is not None
                else f"chat:{kwargs.get('chat_id')}"
            ),
        )
        await store.set_fact("focus:current_task_id", str(task_id), scope="system")
        context = await resolve_tick_dispatch_context(loop, 5, source="chat", chat_id="chat:test")

        assert context.dispatch_cycle == 77
        assert context.active_task is not None
        assert context.active_task.id == task_id
        assert context.chain_key == "task-chain:chain-chat"
        await store.close()


def test_resolve_tick_dispatch_context_falls_back_to_chat_chain_for_unbound_task():
    asyncio.run(_resolve_tick_dispatch_context_falls_back_to_chat_chain_for_unbound_task())


async def _resolve_tick_dispatch_context_falls_back_to_chat_chain_for_unbound_task():
    from core.loop.cycle.focus import resolve_tick_dispatch_context
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "dispatch-context-chat.db")
        await store.open()

        async def _next_dispatch_cycle() -> int:
            return 88

        loop = SimpleNamespace(
            _task_store=store,
            _next_dispatch_cycle=_next_dispatch_cycle,
            _resolve_tick_chain_key=lambda **kwargs: (
                f"task:{kwargs['active_task'].id}"
                if kwargs.get("active_task") is not None
                else f"chat:{kwargs.get('chat_id')}"
            ),
        )
        context = await resolve_tick_dispatch_context(
            loop,
            5,
            source="chat",
            chat_id="chat:test",
            include_waiting=True,
        )

        assert context.dispatch_cycle == 88
        assert context.active_task is None
        assert context.chain_key == "chat:chat:test"
        await store.close()


def test_try_dispatch_tick_job_enqueues_when_dispatcher_ready():
    asyncio.run(_try_dispatch_tick_job_enqueues_when_dispatcher_ready())


def test_run_cycle_skips_auto_tick_when_waiting_focus_exists():
    asyncio.run(_run_cycle_skips_auto_tick_when_waiting_focus_exists())


def test_run_cycle_skips_direct_tick_when_waiting_focus_exists():
    asyncio.run(_run_cycle_skips_direct_tick_when_waiting_focus_exists())


async def _try_dispatch_tick_job_enqueues_when_dispatcher_ready():
    from core.loop.cycle.focus import try_dispatch_tick_job

    seen_jobs: list[object] = []

    class _Dispatcher:
        enabled = True

        async def enqueue(self, job):
            seen_jobs.append(job)
            return True

    async def _next_dispatch_cycle() -> int:
        return 88

    loop = SimpleNamespace(
        _task_store=SimpleNamespace(),
        _next_dispatch_cycle=_next_dispatch_cycle,
        _resolve_tick_chain_key=lambda **kwargs: f"chat:{kwargs.get('chat_id')}",
        _tick_dispatcher=_Dispatcher(),
    )

    result = await try_dispatch_tick_job(
        loop,
        7,
        source="chat",
        chat_id="chat:test",
        user_message="hello",
    )

    assert result.accepted is True
    assert result.can_retry is False
    assert result.context.dispatch_cycle == 88
    assert result.context.chain_key == "chat:chat:test"
    assert len(seen_jobs) == 1


async def _run_cycle_skips_auto_tick_when_waiting_focus_exists():
    from core.loop.cycle.driver import _run_cycle_impl
    from core.loop.cycle.focus import claim_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "run-cycle-waiting-dispatcher.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "等待用户消息的任务",
                goal="waiting focus 不应触发 auto tick",
                status="waiting",
                wait_kind="external",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            loop_ref = SimpleNamespace(_task_store=store)
            await claim_focus_task(loop_ref, task, clear_current=True)

            seen_jobs: list[object] = []

            class _Dispatcher:
                enabled = True
                running_count = 0
                pending_count = 0

                def can_accept(self):
                    return True

                async def enqueue(self, job):
                    seen_jobs.append(job)
                    return True

            async def _next_dispatch_cycle() -> int:
                return 99

            loop = SimpleNamespace(
                _task_store=store,
                _tick_dispatcher=_Dispatcher(),
                _run_driver=None,
                _auto_tick_due=True,
                _next_dispatch_cycle=_next_dispatch_cycle,
                _resolve_tick_chain_key=lambda **kwargs: "default",
            )

            cycle = await _run_cycle_impl(loop, 7)

            assert cycle == 7
            assert seen_jobs == []
            assert loop._auto_tick_due is False
        finally:
            await store.close()


async def _run_cycle_skips_direct_tick_when_waiting_focus_exists():
    from core.loop.cycle.driver import _run_cycle_impl
    from core.loop.cycle.focus import claim_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "run-cycle-waiting-direct.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "等待用户消息的任务",
                goal="waiting focus 不应触发 direct auto tick",
                status="waiting",
                wait_kind="external",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            loop_ref = SimpleNamespace(_task_store=store)
            await claim_focus_task(loop_ref, task, clear_current=True)

            ticked: list[int] = []

            async def _tick(cycle: int) -> None:
                ticked.append(cycle)

            loop = SimpleNamespace(
                _task_store=store,
                _tick_dispatcher=None,
                _run_driver=None,
                _tick=_tick,
            )

            cycle = await _run_cycle_impl(loop, 7)

            assert cycle == 7
            assert ticked == []
        finally:
            await store.close()


def test_try_dispatch_tick_job_graceful_skip_when_queue_full():
    asyncio.run(_try_dispatch_tick_job_graceful_skip_when_queue_full())


async def _try_dispatch_tick_job_graceful_skip_when_queue_full():
    from core.loop.cycle.focus import try_dispatch_tick_job

    class _Dispatcher:
        enabled = True

        async def enqueue(self, job):
            return False

    async def _next_dispatch_cycle() -> int:
        return 12

    loop = SimpleNamespace(
        _task_store=SimpleNamespace(),
        _next_dispatch_cycle=_next_dispatch_cycle,
        _resolve_tick_chain_key=lambda **kwargs: f"chat:{kwargs.get('chat_id')}",
        _tick_dispatcher=_Dispatcher(),
    )

    result = await try_dispatch_tick_job(
        loop,
        5,
        source="chat",
        chat_id="chat:test",
    )

    assert result.accepted is False
    assert result.can_retry is True
    assert result.context.dispatch_cycle == 12
    assert result.context.chain_key == "chat:chat:test"


def test_try_dispatch_tick_job_skips_when_dispatcher_disabled():
    asyncio.run(_try_dispatch_tick_job_skips_when_dispatcher_disabled())


async def _try_dispatch_tick_job_skips_when_dispatcher_disabled():
    from core.loop.cycle.focus import try_dispatch_tick_job

    async def _next_dispatch_cycle() -> int:
        return 33

    loop = SimpleNamespace(
        _task_store=SimpleNamespace(),
        _next_dispatch_cycle=_next_dispatch_cycle,
        _resolve_tick_chain_key=lambda **kwargs: f"chat:{kwargs.get('chat_id')}",
        _tick_dispatcher=SimpleNamespace(enabled=False),
    )

    result = await try_dispatch_tick_job(
        loop,
        1,
        source="auto",
    )

    assert result.accepted is False
    assert result.can_retry is False
    assert result.context.dispatch_cycle == 33


def test_resolve_focus_task_prefers_chat_bound_task_over_unrelated_current_focus():
    asyncio.run(_resolve_focus_task_prefers_chat_bound_task_over_unrelated_current_focus())


async def _resolve_focus_task_prefers_chat_bound_task_over_unrelated_current_focus():
    from core.loop.cycle.focus import claim_focus_task, resolve_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-resolve.db")
        await store.open()
        unrelated_id = await store.add_task(
            "后台整理任务",
            goal="继续处理后台任务",
            status="in_progress",
            next_step="继续后台整理",
        )
        waiting_chat_id = await store.add_task(
            "会话等待任务",
            goal="等待同一会话用户反馈",
            status="waiting",
            wait_kind="external",
            wait_key="chat:test",
            next_step="等用户补充截图",
        )
        unrelated = await store.get_task_by_id(unrelated_id)
        waiting_chat = await store.get_task_by_id(waiting_chat_id)
        assert unrelated is not None
        assert waiting_chat is not None

        loop = SimpleNamespace(_task_store=store)
        await store.set_fact(f"task:{waiting_chat.id}:chat_id", "chat:test", scope="task")
        await claim_focus_task(loop, unrelated, clear_current=True)
        await claim_focus_task(loop, waiting_chat, chat_id="chat:test", clear_current=False)

        resolved_chat = await resolve_focus_task(
            loop,
            chat_id="chat:test",
            include_waiting=True,
        )
        resolved_current = await resolve_focus_task(loop)

        assert resolved_chat is not None
        assert resolved_chat.id == waiting_chat_id
        assert resolved_current is not None
        assert resolved_current.id == unrelated_id
        await store.close()


def test_claim_focus_task_records_life_ledger():
    asyncio.run(_claim_focus_task_records_life_ledger())


async def _claim_focus_task_records_life_ledger():
    from core.loop.cycle.focus import claim_focus_task
    from core.metabolic import MetabolicEngine
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-ledger.db")
        await store.open()
        task_id = await store.add_task(
            "焦点账本任务",
            goal="focus changes should be metabolic events",
            status="in_progress",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        loop = SimpleNamespace(_task_store=store, _metabolic=MetabolicEngine(store))
        await claim_focus_task(loop, task, chat_id="chat:test", clear_current=True)

        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        chat_focus, chat_exists = await store.get_fact("focus:chat:chat:test")
        assert current_exists is True and current_focus == str(task_id)
        assert chat_exists is True and chat_focus == str(task_id)

        rows = await store.ledger_recent(limit=5)
        focus_rows = [row for row in rows if row["source"] == "loop/focus"]
        assert len(focus_rows) >= 2
        assert {row["key"] for row in focus_rows} >= {
            "focus:current_task_id",
            "focus:chat:chat:test",
        }
        assert all(row["op"] == "set_fact" for row in focus_rows)
        await store.close()


def test_resolve_focus_task_with_chat_id_is_fail_closed_by_default():
    asyncio.run(_resolve_focus_task_with_chat_id_is_fail_closed_by_default())


async def _resolve_focus_task_with_chat_id_is_fail_closed_by_default():
    from core.loop.cycle.focus import resolve_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-fail-closed.db")
        await store.open()
        for index in range(55):
            await store.add_task(
                f"普通任务{index}",
                goal=f"占满 open task 扫描窗口 {index}",
                status="in_progress",
            )
        matched_id = await store.add_task(
            "窗口外会话任务",
            goal="等待同会话继续反馈",
            status="in_progress",
        )
        await store.set_fact(f"task:{matched_id}:chat_id", "chat:test", scope="task")
        await store.set_fact("focus:current_task_id", str(matched_id), scope="system")

        loop = SimpleNamespace(_task_store=store)
        resolved = await resolve_focus_task(loop, chat_id="chat:test")

        assert resolved is None
        await store.close()


def test_prepare_focus_task_resumes_waiting_chat_task():
    asyncio.run(_prepare_focus_task_resumes_waiting_chat_task())


async def _prepare_focus_task_resumes_waiting_chat_task():
    from core.loop.cycle.focus import prepare_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-resume.db")
        await store.open()
        task_id = await store.add_task(
            "等待确认的任务",
            goal="等待用户继续指示",
            status="waiting",
            wait_kind="external",
            wait_key="chat:test",
            next_step="收到新消息后继续",
        )
        await store.set_fact(f"task:{task_id}:chat_id", "chat:test", scope="task")

        loop = SimpleNamespace(_task_store=store)
        task = await prepare_focus_task(loop, user_message="继续处理", chat_id="chat:test")
        assert task is not None
        assert task.status == "resumed"

        refreshed = await store.get_task_by_id(task_id)
        assert refreshed is not None
        assert refreshed.status == "resumed"
        assert refreshed.result_json.get("resumed_via") == "focus.chat"
        await store.close()


def test_finalize_focus_task_parks_user_facing_pause_into_waiting():
    asyncio.run(_finalize_focus_task_parks_user_facing_pause_into_waiting())


def test_finalize_focus_task_keeps_pause_with_actionable_next_step_runnable():
    asyncio.run(_finalize_focus_task_keeps_pause_with_actionable_next_step_runnable())


def test_finalize_focus_task_keeps_wait_with_next_step_runnable():
    asyncio.run(_finalize_focus_task_keeps_wait_with_next_step_runnable())


def test_finalize_focus_task_clears_terminal_self_drive_attention():
    asyncio.run(_finalize_focus_task_clears_terminal_self_drive_attention())


def test_terminal_task_result_cleanup_clears_attention_without_active_task():
    asyncio.run(_terminal_task_result_cleanup_clears_attention_without_active_task())


def test_terminal_task_fail_result_cleanup_clears_attention():
    asyncio.run(_terminal_task_fail_result_cleanup_clears_attention())


def test_finalize_focus_task_keeps_waiting_task_attention():
    asyncio.run(_finalize_focus_task_keeps_waiting_task_attention())


async def _finalize_focus_task_parks_user_facing_pause_into_waiting():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task, finalize_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-finalize.db")
        await store.open()
        task_id = await store.add_task(
            "网页核验任务",
            goal="等待用户明天反馈网页截图",
            status="in_progress",
            next_step="等用户发完整 URL",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        await store.set_fact(f"task:{task.id}:chat_id", "chat:test", scope="task")

        loop = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop, task, chat_id="chat:test", clear_current=True)
        action = JudgmentOutput(
            decision="pause",
            reply_to_user="等您明天给我完整 URL。",
            next_step="收到 URL 后核验并修复",
        )

        finalized = await finalize_focus_task(
            loop,
            action=action,
            active_task=task,
            chat_id=None,
            user_message="",
        )
        assert finalized is not None
        assert finalized.status == "waiting"

        refreshed = await store.get_task_by_id(task_id)
        assert refreshed is not None
        assert refreshed.status == "waiting"
        assert refreshed.wait_kind == "external"
        assert refreshed.wait_key == "chat:test"

        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        chat_focus, chat_exists = await store.get_fact("focus:chat:chat:test")
        assert current_exists is True
        assert current_focus == str(task_id)
        assert chat_exists is True
        assert chat_focus == str(task_id)
        await store.close()


async def _finalize_focus_task_keeps_pause_with_actionable_next_step_runnable():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task, finalize_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-finalize-pause-runnable.db")
        await store.open()
        task_id = await store.add_task(
            "OpenClaw 记忆找回",
            goal="继续用真实工具找回 OpenClaw 记忆",
            status="in_progress",
            next_step="先用 memory.search 和 shell.run 核对 OpenClaw 相关记忆",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        await store.set_fact(f"task:{task.id}:chat_id", "chat:test", scope="task")

        loop = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop, task, chat_id="chat:test", clear_current=True)
        action = JudgmentOutput(
            decision="pause",
            reply_to_user="本轮已经进入最终回复阶段，下一轮我会继续用真实工具核对。",
            next_step="使用 memory.search 召回 OpenClaw，并用 shell.run 核对导入文件。",
        )

        finalized = await finalize_focus_task(
            loop,
            action=action,
            active_task=task,
            chat_id=None,
            user_message="",
        )

        assert finalized is not None
        assert finalized.status == "in_progress"
        refreshed = await store.get_task_by_id(task_id)
        assert refreshed is not None
        assert refreshed.status == "in_progress"
        assert refreshed.wait_kind == ""
        await store.close()


async def _finalize_focus_task_clears_terminal_self_drive_attention():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task, finalize_focus_task
    from memory.working import WMItem, WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-finalize-terminal-cleanup.db")
        await store.open()
        task_id = await store.add_task(
            "已完成自驱任务",
            goal="完成后不再占用注意力",
            status="done",
            source="self_drive",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        wm = WorkingMemory(capacity=10)
        wm.add(WMItem(kind="task_anchor", content="旧任务锚点", priority=0.95))
        wm.add(WMItem(kind="self_drive", content="旧自驱信号", priority=0.9))
        wm.add(WMItem(kind="bootstrap_identity", content="身份锚点", priority=1.0))
        loop = SimpleNamespace(_task_store=store, _wm=wm)
        await claim_focus_task(loop, task, clear_current=True)

        finalized = await finalize_focus_task(
            loop,
            action=JudgmentOutput(decision="act", chosen_action_id="task.complete"),
            active_task=task,
            chat_id=None,
            user_message="",
        )

        assert finalized is not None
        remaining = {item["kind"] for item in wm.get_top(10)}
        assert "task_anchor" not in remaining
        assert "self_drive" not in remaining
        assert "bootstrap_identity" in remaining

        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        assert current_exists is False or current_focus == ""
        await store.close()


async def _terminal_task_result_cleanup_clears_attention_without_active_task():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task
    from core.loop.tick.exec import _cleanup_terminal_result_attention
    from memory.working import WMItem, WorkingMemory
    from store.task import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-terminal-result-cleanup.db")
        await store.open()
        task_id = await store.add_task(
            "刚完成的显式任务",
            goal="即使 active_task 已丢失，也要清理旧锚点",
            status="done",
            source="self_drive",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        wm = WorkingMemory(capacity=10)
        wm.add(WMItem(kind="task_anchor", content="旧任务锚点", priority=0.95))
        wm.add(WMItem(kind="self_drive", content="旧自驱信号", priority=0.9))
        loop = SimpleNamespace(_task_store=store, _wm=wm)
        await claim_focus_task(loop, task, clear_current=True)

        cleaned = await _cleanup_terminal_result_attention(
            loop,
            JudgmentOutput(decision="act", chosen_action_id="task.complete"),
            ToolResult(
                summary="任务已完成",
                resource_key=str(task_id),
                state_delta={"task_id": task_id, "task_status": "done"},
                metadata={"task_id": task_id},
            ),
            None,
            None,
        )

        assert cleaned is None
        remaining = {item["kind"] for item in wm.get_top(10)}
        assert "task_anchor" not in remaining
        assert "self_drive" not in remaining
        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        assert current_exists is False or current_focus == ""
        await store.close()


async def _terminal_task_fail_result_cleanup_clears_attention():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task
    from core.loop.tick.exec import _cleanup_terminal_result_attention
    from memory.working import WMItem, WorkingMemory
    from store.task import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-terminal-fail-cleanup.db")
        await store.open()
        task_id = await store.add_task(
            "失败的显式任务",
            goal="失败也是终结状态，需要清理旧锚点",
            status="failed",
            source="self_drive",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        wm = WorkingMemory(capacity=10)
        wm.add(WMItem(kind="task_anchor", content="旧任务锚点", priority=0.95))
        wm.add(WMItem(kind="self_drive", content="旧自驱信号", priority=0.9))
        loop = SimpleNamespace(_task_store=store, _wm=wm)
        await claim_focus_task(loop, task, clear_current=True)

        cleaned = await _cleanup_terminal_result_attention(
            loop,
            JudgmentOutput(decision="act", chosen_action_id="task.fail"),
            ToolResult(
                summary="任务已失败",
                resource_key=str(task_id),
                state_delta={"task_id": task_id, "task_status": "failed"},
                metadata={"task_id": task_id},
            ),
            None,
            None,
        )

        assert cleaned is None
        remaining = {item["kind"] for item in wm.get_top(10)}
        assert "task_anchor" not in remaining
        assert "self_drive" not in remaining
        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        assert current_exists is False or current_focus == ""
        await store.close()


async def _finalize_focus_task_keeps_waiting_task_attention():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import finalize_focus_task
    from memory.working import WMItem, WorkingMemory
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-finalize-waiting-keep-wm.db")
        await store.open()
        task_id = await store.add_task(
            "等待用户任务",
            goal="等待外部输入时仍需要注意力锚点",
            status="waiting",
            wait_kind="external",
            wait_key="chat:test",
            source="self_drive",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None

        wm = WorkingMemory(capacity=10)
        wm.add(WMItem(kind="task_anchor", content="等待任务锚点", priority=0.95))
        wm.add(WMItem(kind="self_drive", content="等待自驱信号", priority=0.9))
        loop = SimpleNamespace(_task_store=store, _wm=wm)

        finalized = await finalize_focus_task(
            loop,
            action=JudgmentOutput(decision="wait"),
            active_task=task,
            chat_id="chat:test",
            user_message="",
        )

        assert finalized is not None
        assert finalized.status == "waiting"
        remaining = {item["kind"] for item in wm.get_top(10)}
        assert "task_anchor" in remaining
        assert "self_drive" in remaining
        await store.close()


async def _finalize_focus_task_keeps_wait_with_next_step_runnable():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import claim_focus_task, finalize_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-finalize-runnable.db")
        await store.open()
        task_id = await store.add_task(
            "自我升级任务",
            goal="回复用户后继续内部推进",
            status="in_progress",
            next_step="查询 open tasks 与最近 runs",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        await store.set_fact(f"task:{task.id}:chat_id", "chat:test", scope="task")

        loop = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop, task, chat_id="chat:test", clear_current=True)
        action = JudgmentOutput(
            decision="wait",
            reply_to_user="我会继续查询 open tasks 与最近 runs。",
            next_step="查询 open tasks 与最近 10 条 runs，并写入 task.workbench",
        )

        finalized = await finalize_focus_task(
            loop,
            action=action,
            active_task=task,
            chat_id=None,
            user_message="",
        )

        assert finalized is not None
        assert finalized.status == "in_progress"
        refreshed = await store.get_task_by_id(task_id)
        assert refreshed is not None
        assert refreshed.status == "in_progress"
        assert refreshed.wait_kind == ""

        current_focus, current_exists = await store.get_fact("focus:current_task_id")
        chat_focus, chat_exists = await store.get_fact("focus:chat:chat:test")
        assert current_exists is True
        assert current_focus == str(task_id)
        assert chat_exists is True
        assert chat_focus == str(task_id)
        await store.close()


def test_process_pending_chat_turn_routes_bound_chat_to_task_chain():
    asyncio.run(_process_pending_chat_turn_routes_bound_chat_to_task_chain())


async def _process_pending_chat_turn_routes_bound_chat_to_task_chain():
    from core.loop.cycle.chat import _process_pending_chat_turn
    from core.loop.cycle.focus import claim_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-chat-route.db")
        await store.open()
        task_id = await store.add_task(
            "当前会话任务",
            goal="应由同一 task 链承接后续 chat 消息",
            status="in_progress",
            chain_id="chain-chat-focus",
            next_step="继续承接用户回合",
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        await store.set_fact(f"task:{task_id}:chat_id", "chat:test", scope="task")

        loop_ref = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop_ref, task, chat_id="chat:test", clear_current=True)
        await store.add_chat_message("user", "继续", chat_id="chat:test")

        seen_jobs = []

        class _Dispatcher:
            enabled = True

            def can_accept(self) -> bool:
                return True

            async def enqueue(self, job):
                seen_jobs.append(job)
                return True

        async def _next_dispatch_cycle() -> int:
            return 8

        loop = SimpleNamespace(
            _task_store=store,
            _tick_dispatcher=_Dispatcher(),
            _cfg=SimpleNamespace(loop=SimpleNamespace(wechat_coalesce_delay=0)),
            _next_dispatch_cycle=_next_dispatch_cycle,
            _resolve_tick_chain_key=lambda **kwargs: (
                f"task:{kwargs['active_task'].id}" if kwargs.get("active_task") is not None else f"chat:{kwargs.get('chat_id')}"
            ),
        )

        cycle, handled = await _process_pending_chat_turn(loop, 7)

        assert handled is True
        assert cycle == 8
        assert len(seen_jobs) == 1
        assert seen_jobs[0].chain_key == f"task:{task_id}"
        assert seen_jobs[0].chat_id == "chat:test"
        await store.close()


def test_adopt_result_task_picks_task_created_by_task_add():
    asyncio.run(_adopt_result_task_picks_task_created_by_task_add())


def test_wait_after_cycle_uses_focus_task_instead_of_global_active():
    asyncio.run(_wait_after_cycle_uses_focus_task_instead_of_global_active())


def test_task_change_signature_ignores_same_task_entering_waiting():
    asyncio.run(_task_change_signature_ignores_same_task_entering_waiting())


def test_wait_after_cycle_dispatcher_uses_max_idle_gap_without_focus_task():
    asyncio.run(_wait_after_cycle_dispatcher_uses_max_idle_gap_without_focus_task())


def test_prepare_active_task_ignores_self_drive_for_user_message():
    asyncio.run(_prepare_active_task_ignores_self_drive_for_user_message())


def test_prepare_active_task_creates_action_first_task_for_executable_user_message():
    asyncio.run(_prepare_active_task_creates_action_first_task_for_executable_user_message())


async def _adopt_result_task_picks_task_created_by_task_add():
    from core.judgment import JudgmentOutput
    from core.loop.cycle.focus import adopt_result_task
    from store.task import TaskStore
    from tools.registry import ToolResult

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-adopt.db")
        await store.open()
        task_id = await store.add_task("新建任务", goal="应在本轮收尾时成为焦点")

        loop = SimpleNamespace(_task_store=store)
        task = await adopt_result_task(
            loop,
            None,
            JudgmentOutput(decision="act", chosen_action_id="task.add"),
            ToolResult(summary="任务已创建", resource_key=str(task_id), metadata={"task_id": task_id}),
        )

        assert task is not None
        assert task.id == task_id
        await store.close()


async def _wait_after_cycle_uses_focus_task_instead_of_global_active():
    from core.loop.cycle.driver import _wait_after_cycle_impl
    from core.loop.cycle.focus import claim_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "focus-wait-after-cycle.db")
        await store.open()
        unrelated_id = await store.add_task(
            "全局活跃任务",
            goal="旧 get_active 会误命中这里",
            status="in_progress",
        )
        focus_id = await store.add_task(
            "当前焦点任务",
            goal="wait 路径应跟随这里",
            status="pending",
        )
        unrelated = await store.get_task_by_id(unrelated_id)
        focus_task = await store.get_task_by_id(focus_id)
        assert unrelated is not None
        assert focus_task is not None

        loop_ref = SimpleNamespace(_task_store=store)
        await claim_focus_task(loop_ref, focus_task, clear_current=True)

        seen_before_task: list[int | None] = []

        async def _capture_wait(loop: object, max_wait: float, before_task: object) -> None:
            seen_before_task.append(getattr(before_task, "id", None))

        async def _noop_reload() -> None:
            return None

        from core.loop.cycle import driver as driver_module

        original_wait = driver_module._wait_for_event_impl
        driver_module._wait_for_event_impl = _capture_wait  # type: ignore[assignment]
        try:
            loop = SimpleNamespace(
                _cfg=SimpleNamespace(
                    loop=SimpleNamespace(
                        arousal_min_factor=0.8,
                        arousal_sensitivity=0.0,
                        arousal_neutral=0.5,
                        active_idle_gap=500,
                        min_act_gap=100,
                        idle_with_task_bounds=[100, 30000],
                        max_idle_gap=60000,
                        wake_poll_interval=100,
                        wake_on_task_change=True,
                    )
                ),
                _emotion=SimpleNamespace(arousal=0.5),
                _tick_dispatcher=None,
                _last_decision="wait",
                _pending_idle_gap=None,
                _bootstrap_mode="none",
                _task_store=store,
                _maybe_hot_reload_provider=_noop_reload,
            )
            await _wait_after_cycle_impl(loop)
        finally:
            driver_module._wait_for_event_impl = original_wait  # type: ignore[assignment]
            await store.close()

        assert seen_before_task == [focus_id]


async def _task_change_signature_ignores_same_task_entering_waiting():
    from core.loop.cycle.driver import _task_change_signature
    from core.loop.cycle.focus import claim_focus_task
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "task-change-waiting.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "刚回复后进入等待的任务",
                goal="进入 waiting 不应立刻唤醒下一轮空闲 judgment",
                status="waiting",
                wait_kind="external",
            )
            task = await store.get_task_by_id(task_id)
            assert task is not None
            loop = SimpleNamespace(_task_store=store)
            await claim_focus_task(loop, task, clear_current=True)

            sig = await _task_change_signature(loop, (task_id, "resumed"))

            assert sig == (task_id, "resumed")
        finally:
            await store.close()


async def _wait_after_cycle_dispatcher_uses_max_idle_gap_without_focus_task():
    from core.loop.cycle.driver import _wait_after_cycle_impl
    from store.task import TaskStore

    class _Dispatcher:
        enabled = True

        def has_running(self) -> bool:
            return False

        def has_pending(self) -> bool:
            return False

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "dispatcher-idle-gap.db")
        await store.open()
        seen: list[tuple[float, int | None]] = []

        async def _capture_wait(loop: object, max_wait: float, before_task: object) -> None:
            seen.append((max_wait, getattr(before_task, "id", None)))

        from core.loop.cycle import driver as driver_module

        original_wait = driver_module._wait_for_event_impl
        driver_module._wait_for_event_impl = _capture_wait  # type: ignore[assignment]
        try:
            loop = SimpleNamespace(
                _cfg=SimpleNamespace(
                    loop=SimpleNamespace(
                        arousal_min_factor=0.8,
                        arousal_sensitivity=0.0,
                        arousal_neutral=0.5,
                        active_idle_gap=500,
                        min_act_gap=100,
                        idle_with_task_bounds=[100, 30000],
                        max_idle_gap=60000,
                        wake_poll_interval=100,
                        wake_on_task_change=True,
                    )
                ),
                _emotion=SimpleNamespace(arousal=0.5),
                _tick_dispatcher=_Dispatcher(),
                _task_store=store,
            )
            await _wait_after_cycle_impl(loop)
        finally:
            driver_module._wait_for_event_impl = original_wait  # type: ignore[assignment]
            await store.close()

        assert seen == [(60.0, None)]


async def _prepare_active_task_ignores_self_drive_for_user_message():
    from core.loop.tick import prep as prep_module

    self_drive_task = SimpleNamespace(
        id=42,
        source="self_drive",
        extras={},
    )
    seen_claims: list[object | None] = []

    async def _prepare_focus_task(loop, *, user_message, chat_id):
        return self_drive_task

    async def _task_matches_chat(loop, task, chat_id):
        return False

    async def _noop_ingest(*args, **kwargs):
        return None

    async def _consume_hints(task_store, active_task, wm, metabolic=None):
        return active_task

    async def _bind(loop, active_task, chat_id):
        return None

    async def _claim(loop, active_task, *, chat_id=None, clear_current=True):
        seen_claims.append(active_task)

    original_prepare = prep_module.prepare_focus_task
    original_matches = prep_module.task_matches_chat
    original_ingest = prep_module._ingest_actionable_meta_reflections
    original_hints = prep_module._consume_task_runtime_hints
    original_bind = prep_module._bind_chat_id
    original_claim = prep_module.claim_focus_task
    try:
        prep_module.prepare_focus_task = _prepare_focus_task  # type: ignore[assignment]
        prep_module.task_matches_chat = _task_matches_chat  # type: ignore[assignment]
        prep_module._ingest_actionable_meta_reflections = _noop_ingest  # type: ignore[assignment]
        prep_module._consume_task_runtime_hints = _consume_hints  # type: ignore[assignment]
        prep_module._bind_chat_id = _bind  # type: ignore[assignment]
        prep_module.claim_focus_task = _claim  # type: ignore[assignment]

        loop = SimpleNamespace(
            _task_store=SimpleNamespace(),
            _wm=SimpleNamespace(),
            _metabolic=None,
        )
        active = await prep_module._prepare_active_task_for_tick(
            loop,
            user_message="用户新请求",
            chat_id="wechat:chat-1",
        )
    finally:
        prep_module.prepare_focus_task = original_prepare  # type: ignore[assignment]
        prep_module.task_matches_chat = original_matches  # type: ignore[assignment]
        prep_module._ingest_actionable_meta_reflections = original_ingest  # type: ignore[assignment]
        prep_module._consume_task_runtime_hints = original_hints  # type: ignore[assignment]
        prep_module._bind_chat_id = original_bind  # type: ignore[assignment]
        prep_module.claim_focus_task = original_claim  # type: ignore[assignment]

    assert active is None
    assert seen_claims == [None]
    assert self_drive_task.extras == {}


async def _prepare_active_task_creates_action_first_task_for_executable_user_message():
    from core.loop.tick import prep as prep_module
    from memory.working import WorkingMemory
    from store.task import TaskStore

    async def _prepare_focus_task(loop, *, user_message, chat_id):
        return None

    async def _noop_ingest(*args, **kwargs):
        return None

    async def _consume_hints(task_store, active_task, wm, metabolic=None):
        return active_task

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "action-first-focus.db")
        await store.open()
        original_prepare = prep_module.prepare_focus_task
        original_ingest = prep_module._ingest_actionable_meta_reflections
        original_hints = prep_module._consume_task_runtime_hints
        try:
            prep_module.prepare_focus_task = _prepare_focus_task  # type: ignore[assignment]
            prep_module._ingest_actionable_meta_reflections = _noop_ingest  # type: ignore[assignment]
            prep_module._consume_task_runtime_hints = _consume_hints  # type: ignore[assignment]

            loop = SimpleNamespace(
                _task_store=store,
                _wm=WorkingMemory(),
                _metabolic=None,
                _recent_action_feedback=["tool=file.read | key=/tmp/a.py | status=ok | progressful=False"],
            )
            active = await prep_module._prepare_active_task_for_tick(
                loop,
                user_message="请你下载 https://example.com/config.yaml 并验证",
                chat_id="chat:action-first",
            )
        finally:
            prep_module.prepare_focus_task = original_prepare  # type: ignore[assignment]
            prep_module._ingest_actionable_meta_reflections = original_ingest  # type: ignore[assignment]
            prep_module._consume_task_runtime_hints = original_hints  # type: ignore[assignment]
            await store.close()

    assert active is not None
    assert active.source == "external"
    assert active.next_step
    anchors = [item for item in loop._wm.get_top(10) if item["kind"] == "task_anchor"]
    assert len(anchors) == 1
    assert active.title in anchors[0]["content"]
    assert active.next_step in anchors[0]["content"]
    assert "上一动作反馈: tool=file.read | key=/tmp/a.py | status=ok | progressful=False" in anchors[0]["content"]
    cortex = active.result_json["cortex"]
    assert cortex["action_first"]["must_act"] is True
    assert {"kind": "url", "value": "https://example.com/config.yaml"} in cortex["captured_inputs"]
