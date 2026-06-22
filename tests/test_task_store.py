"""TaskStore 持久化测试"""
import asyncio
import sqlite3
import tempfile
from pathlib import Path

import aiosqlite
from conftest import (
    _tool_ctx,
)

# ══════════════════════════════════════════════════════════════════════════════
# TaskStore — JSON-first
# ══════════════════════════════════════════════════════════════════════════════

def test_task_store_basic():
    asyncio.run(_task_store_basic())

async def _task_store_basic():
    from store.task import TaskStore
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


def test_task_store_compacts_oversized_runtime_payloads():
    asyncio.run(_task_store_compacts_oversized_runtime_payloads())


def test_task_store_find_similar_open_tasks():
    asyncio.run(_task_store_find_similar_open_tasks())


async def _task_store_find_similar_open_tasks():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "similar.db")
        await store.open()

        open_id = await store.add_task(
            "排查远程运行重启循环",
            goal="分析 crash.log 并修复远程服务重启问题",
            source="external",
        )
        await store.add_task(
            "排查远程运行重启循环",
            goal="历史已完成任务",
            status="done",
            source="external",
        )

        hits = await store.find_similar_open_tasks("解决远程运行重启循环", limit=3)

        assert hits
        assert hits[0][0].id == open_id
        assert all(task.status != "done" for task, _ in hits)

        self_drive_id = await store.add_task(
            "解决远程运行重启循环",
            goal="self drive 中的相似诊断任务",
            source="self_drive",
        )
        filtered_hits = await store.find_similar_open_tasks(
            "解决远程运行重启循环",
            limit=5,
            excluded_sources=("self_drive",),
        )

        assert filtered_hits
        assert all(task.id != self_drive_id for task, _ in filtered_hits)

        await store.close()


def test_task_add_reuses_similar_open_task():
    asyncio.run(_task_add_reuses_similar_open_task())


async def _task_add_reuses_similar_open_task():
    from store.task import TaskStore
    from tools.task import task_add

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "task-add-reuse.db")
        await store.open()

        existing_id = await store.add_task(
            "排查远程运行重启循环",
            goal="分析 crash.log 并修复远程运行重启循环",
            source="external",
        )
        ctx = _tool_ctx(task_store=store)

        result = await task_add(
            {
                "title": "解决远程运行一直重启",
                "goal": "分析 crash.log 并修复远程运行重启循环",
            },
            ctx,
        )

        assert result.skipped is True
        assert result.metadata.get("task_id") == existing_id
        assert result.metadata.get("reused_existing_task") is True
        tasks = await store.list_tasks(limit=10)
        assert len(tasks) == 1

        await store.close()


def test_task_add_does_not_reuse_self_drive_task_for_external_request():
    asyncio.run(_task_add_does_not_reuse_self_drive_task_for_external_request())


async def _task_add_does_not_reuse_self_drive_task_for_external_request():
    from store.task import TaskStore
    from tools.task import task_add

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "task-add-no-self-drive-reuse.db")
        await store.open()

        await store.add_task(
            "排查远程运行重启循环",
            goal="自驱诊断远程运行重启问题",
            source="self_drive",
        )
        ctx = _tool_ctx(task_store=store)

        result = await task_add(
            {
                "title": "解决远程运行一直重启",
                "goal": "分析 crash.log 并修复远程运行重启循环",
            },
            ctx,
        )

        assert result.skipped is False
        assert result.metadata.get("reused_existing_task") is not True
        tasks = await store.list_tasks(limit=10)
        assert len(tasks) == 2
        ledger = await store.ledger_recent(limit=3)
        assert ledger[0]["op"] == "create_task"
        assert ledger[0]["source"] == "tools/task.add"
        assert ledger[0]["key"] == f"task:{result.metadata['task_id']}"

        await store.close()


async def _task_store_run_lifecycle():
    from store.task import TaskStore

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


async def _task_store_compacts_oversized_runtime_payloads():
    from store.task import TaskStore

    huge = "A" * 18_000 + "TAIL"
    huge_json = {"payload": huge, "items": [{"index": idx, "value": huge} for idx in range(90)]}

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "compact.db")
        await store.open()

        task_id = await store.add_task(
            "T" * 2_000,
            goal=huge,
            next_step=huge,
            state_json=huge_json,
            extras={"raw_trace": huge},
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert len(task.title) <= 1000
        assert "task title truncated chars=" in task.title
        assert "runtime store truncated chars=" in task.goal
        assert task.goal.endswith("TAIL")
        assert "runtime store truncated chars=" in task.state_json["payload"]
        assert task.state_json["items"][39]["_persistent_omitted_items"] == 11
        assert task.state_json["items"][-1]["index"] == 89

        await store.update_task_result(task_id, {"raw_result": huge})
        updated_task = await store.get_task_by_id(task_id)
        assert updated_task is not None
        assert "runtime store truncated chars=" in updated_task.result_json["raw_result"]

        run_id = await store.add_run(
            task_id=task_id,
            input_json=huge_json,
            log_text=huge,
            extras={"raw_extra": huge},
        )
        await store.update_run(
            run_id,
            status="failed",
            output_json={"raw_output": huge},
            error_text=huge,
        )
        run = await store.get_run_by_id(run_id)
        assert run is not None
        assert "runtime store truncated chars=" in run.input_json["payload"]
        assert "run log truncated chars=" in run.log_text
        assert run.log_text.endswith("TAIL")
        assert "runtime store truncated chars=" in run.output_json["raw_output"]
        assert "run error truncated chars=" in run.error_text
        assert "runtime store truncated chars=" in run.extras["raw_extra"]

        await store.record_failure("tool_error", huge, context=huge, task_id=str(task_id))
        failure = (await store.list_failures(limit=1))[0]
        assert "failure summary truncated chars=" in failure.summary
        assert "failure context truncated chars=" in failure.context

        await store.set_fact("runtime:large_json", __import__("json").dumps(huge_json, ensure_ascii=False))
        fact_value, found = await store.get_fact("runtime:large_json")
        assert found
        parsed_fact = __import__("json").loads(fact_value)
        assert "runtime store truncated chars=" in parsed_fact["payload"]
        assert parsed_fact["items"][39]["_persistent_omitted_items"] == 11
        assert parsed_fact["items"][-1]["index"] == 89

        await store.ledger_append(
            "set_fact",
            "runtime:large",
            huge,
            reason=huge,
            decision_basis=huge,
        )
        ledger = await store.ledger_recent(limit=1)
        assert "life_ledger value truncated chars=" in ledger[0]["value"]
        assert "life_ledger reason truncated chars=" in ledger[0]["reason"]
        assert "life_ledger decision_basis truncated chars=" in ledger[0]["decision_basis"]

        await store.add_meta_reflection(
            reflection_id="large-reflection",
            target_kind="tool",
            trigger="large",
            loop_level="task",
            diagnosis=huge,
            proposal=huge,
            verification_plan=huge,
            extras={"raw_extra": huge},
        )
        reflection = (await store.list_meta_reflections(limit=1))[0]
        assert "meta_reflection diagnosis truncated chars=" in reflection.diagnosis
        assert "meta_reflection proposal truncated chars=" in reflection.proposal
        assert "meta_reflection verification_plan truncated chars=" in reflection.verification_plan
        assert "runtime store truncated chars=" in reflection.extras["raw_extra"]

        chat_huge = "C" * 40_000 + "TAIL"
        message_id = await store.add_chat_message("user", chat_huge, chat_id="chat-1")
        messages = await store.get_chat_messages_since(message_id - 1, chat_id="chat-1")
        assert "chat message truncated chars=" in messages[0]["content"]
        assert messages[0]["content"].endswith("TAIL")

        await store.close()


def test_task_update_can_clear_runtime_fields():
    asyncio.run(_task_update_can_clear_runtime_fields())


async def _task_update_can_clear_runtime_fields():
    from store.task import TaskStore
    from tools.task import task_update

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
        ledger = await store.ledger_recent(limit=3)
        assert ledger[0]["op"] == "update_task_status"
        assert ledger[0]["source"] == "tools/task.update"
        assert ledger[0]["key"] == str(task_id)

        await store.close()


def test_update_status_can_patch_runtime_fields_in_one_call():
    asyncio.run(_update_status_can_patch_runtime_fields_in_one_call())


async def _update_status_can_patch_runtime_fields_in_one_call():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "status-patch.db")
        await store.open()
        task_id = await store.add_task(
            "状态补丁任务",
            goal="验证单次 update_status 可同步 runtime 字段",
            next_step="旧下一步",
            current_step="旧当前步骤",
            model_tier="reader",
        )

        await store.update_status(
            task_id,
            "in_progress",
            "",
            current_step="",
            model_tier="",
        )

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
    from store.task import TaskStore
    from tools.task import task_resume, task_wait

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
        ledger = await store.ledger_recent(limit=3)
        assert [row["op"] for row in ledger[:2]] == ["resume_task", "mark_task_waiting"]
        assert [row["source"] for row in ledger[:2]] == ["tools/task.resume", "tools/task.wait"]

        await store.close()


def test_status_transitions_can_merge_result_json_in_one_call():
    asyncio.run(_status_transitions_can_merge_result_json_in_one_call())


async def _status_transitions_can_merge_result_json_in_one_call():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "status-result.db")
        await store.open()
        task_id = await store.add_task(
            "状态结果合并",
            goal="验证 status/waiting 可顺带合并 result_json",
            result_json={"seed": "present"},
        )

        await store.update_status(task_id, "done", result_json={"summary": "ok"})
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "done"
        assert task.result_json == {"seed": "present", "summary": "ok"}

        await store.mark_waiting(
            task_id,
            wait_kind="task",
            wait_key="parent-1",
            wait_json={"terminal_decision": "wait"},
            result_json={"terminal_decision": "wait"},
        )
        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "waiting"
        assert task.wait_key == "parent-1"
        assert task.result_json == {
            "seed": "present",
            "summary": "ok",
            "terminal_decision": "wait",
        }

        await store.close()


def test_chat_messages_are_sanitized_on_write():
    asyncio.run(_chat_messages_are_sanitized_on_write())


def test_chat_pending_messages_are_recoverable_until_processed():
    asyncio.run(_chat_pending_messages_are_recoverable_until_processed())


def test_task_wait_rejects_external_wait_without_wait_key_or_next_step():
    asyncio.run(_task_wait_rejects_external_wait_without_wait_key_or_next_step())


async def _task_wait_rejects_external_wait_without_wait_key_or_next_step():
    from store.task import TaskStore
    from tools.task import task_wait

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

        assert wait_res.skipped is True
        assert wait_res.error == "WaitConditionAmbiguous"
        assert wait_res.state_delta["tool_input_invalid"] is True

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "pending"

        await store.close()


def test_task_wait_allows_external_wait_without_wait_key_when_next_step_is_clear():
    asyncio.run(_task_wait_allows_external_wait_without_wait_key_when_next_step_is_clear())


async def _task_wait_allows_external_wait_without_wait_key_when_next_step_is_clear():
    from store.task import TaskStore
    from tools.task import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "wait-guard-next-step.db")
        await store.open()
        task_id = await store.add_task("等待外部路径", goal="验证 task.wait 可用 next_step 作为恢复锚点")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "external",
                "next_step": "收到用户新日志后读取最新 runs 并写入 task.workbench",
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
        assert task.next_step == "收到用户新日志后读取最新 runs 并写入 task.workbench"

        await store.close()


def test_task_wait_rejects_unknown_wait_kind():
    asyncio.run(_task_wait_rejects_unknown_wait_kind())


def test_task_wait_rejects_blank_wait_kind_with_recovery_hint():
    asyncio.run(_task_wait_rejects_blank_wait_kind_with_recovery_hint())


def test_task_wait_blocks_self_drive_evidence_task_without_dependency():
    asyncio.run(_task_wait_blocks_self_drive_evidence_task_without_dependency())


def test_task_wait_allows_self_drive_evidence_task_with_process_dependency():
    asyncio.run(_task_wait_allows_self_drive_evidence_task_with_process_dependency())


async def _task_wait_blocks_self_drive_evidence_task_without_dependency():
    from store.task import TaskStore
    from tools.task import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "self-drive-evidence-wait.db")
        await store.open()
        task_id = await store.add_task(
            "自驱取证任务",
            goal="不能把可执行取证任务直接挂起",
            source="self_drive",
            status="in_progress",
            next_step="读取最近 daemon-stdout.log 并确认是否仍有重复 wait_no_progress。",
        )

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "external",
                "next_step": "读取最近 daemon-stdout.log 并确认是否仍有重复 wait_no_progress。",
            },
            ctx,
        )

        assert wait_res.skipped is True
        assert wait_res.error == "SelfDriveEvidenceRequiredBeforeWait"
        assert "file.read" in wait_res.state_delta["suggested_tools"]

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "in_progress"

        await store.close()


async def _task_wait_allows_self_drive_evidence_task_with_process_dependency():
    from store.task import TaskStore
    from tools.task import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "self-drive-evidence-process-wait.db")
        await store.open()
        task_id = await store.add_task(
            "自驱等待进程",
            goal="明确进程依赖时允许等待",
            source="self_drive",
            status="in_progress",
            next_step="读取进程输出并确认测试结果。",
        )

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait(
            {
                "task_id": task_id,
                "wait_kind": "process",
                "wait_key": "exec-123",
                "next_step": "读取进程输出并确认测试结果。",
            },
            ctx,
        )

        assert wait_res.skipped is False
        assert wait_res.error is None

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "waiting"
        assert task.wait_kind == "process"
        assert task.wait_key == "exec-123"

        await store.close()


async def _task_wait_rejects_blank_wait_kind_with_recovery_hint():
    from store.task import TaskStore
    from tools.task import task_wait

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "blank-wait-kind.db")
        await store.open()
        task_id = await store.add_task("空等待类型", goal="验证 task.wait 空 wait_kind 可恢复")

        ctx = _tool_ctx(task_store=store)
        wait_res = await task_wait({"task_id": task_id, "wait_kind": "   "}, ctx)

        assert wait_res.skipped is True
        assert wait_res.error == "ToolInputInvalid"
        assert wait_res.state_delta["tool_input_invalid"] is True
        assert "wait_kind 不能为空" in wait_res.state_delta["recovery_next_step"]

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "pending"

        await store.close()


async def _task_wait_rejects_unknown_wait_kind():
    from store.task import TaskStore
    from tools.task import task_wait

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
        assert wait_res.error == "ToolInputInvalid"
        assert wait_res.state_delta["tool_input_invalid"] is True
        assert "process/task/signal/time/external" in wait_res.state_delta["recovery_next_step"]
        assert "不支持的 wait_kind" in wait_res.summary

        task = await store.get_task_by_id(task_id)
        assert task is not None
        assert task.status == "pending"

        await store.close()


def test_task_steer_inbox_is_consumed_once():
    asyncio.run(_task_steer_inbox_is_consumed_once())


async def _task_steer_inbox_is_consumed_once():
    from store.task import TaskStore
    from tools.task import task_steer

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "steer.db")
        await store.open()
        task_id = await store.add_task("用户消息 inbox 任务", goal="验证用户消息 inbox 只消费一次")

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
        ledger = await store.ledger_recent(limit=3)
        assert ledger[0]["op"] == "update_task_data"
        assert ledger[0]["source"] == "tools/task.steer"

        inbox_again = await store.pop_task_inbox(task_id)
        assert inbox_again == []

        await store.close()


async def _chat_messages_are_sanitized_on_write():
    from store.task import TaskStore

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


async def _chat_pending_messages_are_recoverable_until_processed():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "chat-processing.db"
        store = TaskStore(db_path)
        await store.open()
        await store.add_chat_message("user", "hello")

        reserved = await store.pop_pending_chat_message()
        assert reserved is not None
        assert reserved["content"] == "hello"
        assert await store.has_pending_chat_message() is False

        await store.release_chat_messages([reserved["id"]])
        retried = await store.pop_pending_chat_message()
        assert retried is not None
        assert retried["id"] == reserved["id"]

        await store.close()

        reopened = TaskStore(db_path)
        await reopened.open()
        recovered = await reopened.pop_pending_chat_message()
        assert recovered is not None
        assert recovered["id"] == reserved["id"]

        await reopened.mark_chat_messages_processed([recovered["id"]])
        assert await reopened.has_pending_chat_message() is False
        await reopened.close()


async def _ingress_store_unifies_external_chat_and_task_writes():
    from store.task import TaskStore
    from store.task.ingress import IngressStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "ingress.db"
        store = TaskStore(db_path)
        await store.open()
        await store.close()

        ingress = IngressStore(db_path)
        msg_id = ingress.ingest_user_message(
            "\x1b[31mhi\x1b[0m\ufeff\u200b\ufffdthere",
            chat_id="wechat:user-1",
            facts={
                "wechat:ctx:user-1": "ctx-1",
                "wechat:last_user": "user-1",
            },
        )
        task_id = ingress.add_task(
            " webhook: inbound ",
            goal="来自 webhook 的外部任务",
            priority="high",
            source="gateway:webhook",
        )

        verify = TaskStore(db_path)
        await verify.open()
        messages = await verify.get_chat_messages_since(0, chat_id="wechat:user-1")
        assert len(messages) == 1
        assert messages[0]["id"] == msg_id
        assert messages[0]["content"] == "hithere"
        ctx_value, ctx_found = await verify.get_fact("wechat:ctx:user-1")
        assert ctx_found is True
        assert ctx_value == "ctx-1"
        task = await verify.get_task_by_id(task_id)
        assert task is not None
        assert task.title == "webhook: inbound"
        assert task.goal == "来自 webhook 的外部任务"
        assert task.source == "gateway:webhook"
        assistant_id = await verify.add_chat_message("assistant", "已收到", chat_id="wechat:user-1")
        await verify.close()

        outbound = ingress.list_pending_assistant_messages(chat_prefix="wechat:", limit=10)
        assert len(outbound) == 1
        assert outbound[0]["id"] == assistant_id
        assert outbound[0]["chat_id"] == "wechat:user-1"
        ingress.mark_chat_message_delivered(assistant_id)
        assert ingress.list_pending_assistant_messages(chat_prefix="wechat:", limit=10) == []


def test_task_store_migration():
    asyncio.run(_task_store_migration())


def test_ingress_store_unifies_external_chat_and_task_writes():
    asyncio.run(_ingress_store_unifies_external_chat_and_task_writes())


def test_wechat_reply_continues_when_local_poll_is_disabled(monkeypatch):
    asyncio.run(_wechat_reply_continues_when_local_poll_is_disabled(monkeypatch))


async def _wechat_reply_continues_when_local_poll_is_disabled(monkeypatch):
    from channels import wechat as wechat_mod
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "wechat.db"
        store = TaskStore(db_path)
        await store.open()
        assistant_id = await store.add_chat_message("assistant", "已收到", chat_id="wechat:user-1")
        await store.set_fact("wechat:ctx:user-1", "ctx-1")
        await store.close()

        sent: list[tuple[str, str, str, str, str | None]] = []
        monkeypatch.setattr(
            wechat_mod,
            "send_text",
            lambda base_url, token, to_user, text, ctx=None: sent.append((base_url, token, to_user, text, ctx)) or {"ok": True},
        )

        channel = wechat_mod.WechatChannel(
            wechat_mod.WechatConfig(
                base_url="https://send.example",
                poll_base_url="",
                token="token-1",
                reply_poll_sec=1,
            ),
            db_path,
        )

        channel.run_poll()
        assert not channel._stop.is_set()

        channel._check_and_reply()

        assert sent == [("https://send.example", "token-1", "user-1", "已收到", "ctx-1")]
        assert assistant_id in channel._replied
        assert channel._ingress.list_pending_assistant_messages(chat_prefix="wechat:", limit=10) == []


def test_ingress_store_retries_when_database_is_locked():
    from store.task.ingress import IngressStore

    class _Cursor:
        def __init__(self, rowid: int) -> None:
            self.lastrowid = rowid

    class _FlakyConn:
        attempts = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            return _Cursor(77)

    with tempfile.TemporaryDirectory() as d:
        ingress = IngressStore(Path(d) / "locked.db")
        ingress._connect = lambda: _FlakyConn()  # type: ignore[method-assign]

        msg_id = ingress.add_task("locked inbound", goal="retry on lock", source="gateway:webhook")

    assert msg_id == 77
    assert _FlakyConn.attempts == 2

async def _task_store_migration():
    """旧列式 schema → JSON-first 自动迁移。"""
    from store.task import TaskStore

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


def test_task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope():
    asyncio.run(_task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope())


async def _task_store_migrates_legacy_person_profile_facts_to_interlocutor_scope():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "legacy-facts.db"
        async with aiosqlite.connect(str(db_path)) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'general',
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)
            await db.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, ?, datetime('now'))",
                ("chat:wechat:chat-1:person_profile_id", "person-bat", "profile"),
            )
            await db.execute(
                "INSERT INTO facts (key, value, scope, updated_at) VALUES (?, ?, ?, datetime('now'))",
                ("user:person-bat:display_name", "bat", "profile"),
            )
            await db.commit()

        store = TaskStore(db_path)
        await store.open()
        try:
            new_chat_key = await store.get_fact("chat:wechat:chat-1:interlocutor_profile_id")
            new_display_key = await store.get_fact("interlocutor:person-bat:display_name")
            old_chat_key = await store.get_fact("chat:wechat:chat-1:person_profile_id")
            old_display_key = await store.get_fact("user:person-bat:display_name")

            assert new_chat_key == ("person-bat", True)
            assert new_display_key == ("bat", True)
            assert old_chat_key == ("", False)
            assert old_display_key == ("", False)
        finally:
            await store.close()


# ══════════════════════════════════════════════════════════════════════════════
# task.amend — 任务意图纠正
# ══════════════════════════════════════════════════════════════════════════════

def test_amend_task_updates_title_and_goal():
    asyncio.run(_amend_task_updates_title_and_goal())


async def _amend_task_updates_title_and_goal():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend.db")
        await store.open()

        tid = await store.add_task("发邮件给 Alice", goal="把会议纪要发给 Alice", source="external")

        ok = await store.amend_task(
            tid,
            title="发邮件给 Bob",
            goal="把会议纪要发给 Bob",
            amendment_reason="用户澄清：收件人是 Bob，不是 Alice",
        )
        assert ok is True

        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.title == "发邮件给 Bob"
        assert t.goal == "把会议纪要发给 Bob"

        # 修正历史已记录
        amendments = t.extras.get("amendments") or []
        assert len(amendments) == 1
        entry = amendments[0]
        assert entry["prev_title"] == "发邮件给 Alice"
        assert entry["prev_goal"] == "把会议纪要发给 Alice"
        assert "Bob" in entry["reason"]
        assert "ts" in entry

        await store.close()


def test_amend_task_partial_update_preserves_unchanged_fields():
    asyncio.run(_amend_task_partial_update_preserves_unchanged_fields())


async def _amend_task_partial_update_preserves_unchanged_fields():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend-partial.db")
        await store.open()

        tid = await store.add_task("分析日志", goal="找出崩溃原因", priority="normal", source="external")
        await store.update_status(tid, "in_progress", "读取日志文件")

        # 只改 goal，不改 title
        ok = await store.amend_task(
            tid,
            goal="找出崩溃原因并给出修复建议",
            amendment_reason="用户补充：需要给出修复方案",
        )
        assert ok is True

        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.title == "分析日志"           # title 未变
        assert t.goal == "找出崩溃原因并给出修复建议"
        assert t.status == "in_progress"      # status 未变
        assert t.next_step == "读取日志文件"  # next_step 未变

        await store.close()


def test_amend_task_records_multiple_amendment_history():
    asyncio.run(_amend_task_records_multiple_amendment_history())


async def _amend_task_records_multiple_amendment_history():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend-multi.db")
        await store.open()

        tid = await store.add_task("任务初始版本", goal="目标A", source="external")

        await store.amend_task(tid, goal="目标B", amendment_reason="第一次纠正")
        await store.amend_task(tid, goal="目标C", amendment_reason="第二次纠正")

        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.goal == "目标C"
        amendments = t.extras.get("amendments") or []
        assert len(amendments) == 2
        assert amendments[0]["prev_goal"] == "目标A"
        assert amendments[1]["prev_goal"] == "目标B"
        assert amendments[0]["reason"] == "第一次纠正"
        assert amendments[1]["reason"] == "第二次纠正"

        await store.close()


def test_amend_task_returns_false_for_nonexistent_task():
    asyncio.run(_amend_task_returns_false_for_nonexistent_task())


async def _amend_task_returns_false_for_nonexistent_task():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend-miss.db")
        await store.open()

        ok = await store.amend_task(
            99999,
            title="不存在的任务",
            amendment_reason="应该返回 False",
        )
        assert ok is False

        await store.close()


def test_task_amend_tool_end_to_end():
    asyncio.run(_task_amend_tool_end_to_end())


async def _task_amend_tool_end_to_end():
    from store.task import TaskStore
    from tools.task import task_amend

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend-tool.db")
        await store.open()

        tid = await store.add_task("帮用户订票", goal="订 6 月 5 日上海到北京的高铁票", source="external")

        ctx = _tool_ctx(task_store=store)

        # 正常修正
        res = await task_amend(
            {
                "task_id": tid,
                "title": "帮用户订票（已修正）",
                "goal": "订 6 月 6 日上海到北京的高铁票",
                "reason": "用户说出发日期是 6 日，不是 5 日",
            },
            ctx,
        )
        assert res.error is None
        assert res.skipped is not True

        t = await store.get_task_by_id(tid)
        assert t is not None
        assert "6 日" in t.goal
        assert t.title == "帮用户订票（已修正）"

        # reason 缺失 → skipped
        res2 = await task_amend(
            {"task_id": tid, "goal": "不应该生效"},
            ctx,
        )
        assert res2.skipped is True

        # title/goal/priority 均缺失 → skipped
        res3 = await task_amend(
            {"task_id": tid, "reason": "没有给任何字段"},
            ctx,
        )
        assert res3.skipped is True

        # 不存在的任务 → skipped
        res4 = await task_amend(
            {"task_id": 99999, "goal": "不存在", "reason": "测试"},
            ctx,
        )
        assert res4.skipped is True

        await store.close()


def test_task_amend_tool_uses_active_task_when_no_task_id():
    asyncio.run(_task_amend_tool_uses_active_task_when_no_task_id())


async def _task_amend_tool_uses_active_task_when_no_task_id():
    from store.task import TaskStore
    from tools.task import task_amend

    with tempfile.TemporaryDirectory() as d:
        store = TaskStore(Path(d) / "amend-active.db")
        await store.open()

        tid = await store.add_task("当前活跃任务", goal="原始目标", source="external")
        await store.update_status(tid, "in_progress")

        ctx = _tool_ctx(task_store=store)

        res = await task_amend(
            {
                "goal": "修正后目标（无需传 task_id）",
                "reason": "活跃任务自动解析测试",
            },
            ctx,
        )
        assert res.error is None
        assert res.skipped is not True

        t = await store.get_task_by_id(tid)
        assert t is not None
        assert t.goal == "修正后目标（无需传 task_id）"

        await store.close()
