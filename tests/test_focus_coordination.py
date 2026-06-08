import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace


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


def test_prepare_active_task_ignores_self_drive_for_user_message():
    asyncio.run(_prepare_active_task_ignores_self_drive_for_user_message())


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
