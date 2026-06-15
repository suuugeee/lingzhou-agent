"""文件、进程、shell 工具测试"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import (
    _proj_root,
    _test_config,
    _tool_ctx,
)


def _test_soul_cfg() -> Any:
    from core.config_models import EthosBaseline, EthosConfig

    return SimpleNamespace(
        ethos=EthosConfig(
            baseline=EthosBaseline(
                truth=0.91,
                caution=0.81,
                continuity=0.71,
                curiosity=0.61,
                care=0.51,
            )
        )
    )


class _NoToolRegistry:
    def get(self, name: str):
        return None


def _task_tool_ctx(store: Any):
    ctx = _tool_ctx(task_store=store, episodic=SimpleNamespace(load_for_context=lambda *args, **kwargs: ""))
    ctx.registry = _NoToolRegistry()
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# 新增工具测试（file.edit / skill_ops / exec 覆盖）
# ══════════════════════════════════════════════════════════════════════════════


def test_worker_validates_required_tool_params_before_handler():
    asyncio.run(_worker_validates_required_tool_params_before_handler())


async def _worker_validates_required_tool_params_before_handler():
    from core.execution import WorkerLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry, ToolManifest, ToolParam

    async def _handler(params, ctx):
        raise AssertionError("缺少 required 参数时不应进入 handler")

    entry = ToolEntry(
        manifest=ToolManifest(
            name="demo.required",
            description="demo",
            params=[ToolParam("path", "string", "path", required=True)],
        ),
        handler=_handler,
    )

    result = await WorkerLayer(_test_config()).dispatch(
        "tool-chain-worker",
        entry,
        JudgmentOutput(decision="act", chosen_action_id="demo.required", params={"path": "   "}),
        _tool_ctx(),
    )

    assert result.skipped is True
    assert result.error == "ToolInputInvalid"
    assert result.metadata["missing_params"] == ["path"]
    assert result.state_delta["tool_input_invalid"] is True
    assert result.state_delta["missing_params"] == ["path"]
    assert result.state_delta["retry_params_template"] == {"path": "<string>"}
    assert result.metadata["retry_params_template"] == {"path": "<string>"}
    assert result.state_delta["expected_params"][0]["name"] == "path"
    assert "补齐必填参数 path" in result.state_delta["recovery_next_step"]


def test_worker_rewrites_task_workbench_flat_fields_into_workbench_arg():
    asyncio.run(_worker_rewrites_task_workbench_flat_fields_into_workbench_arg())


def test_worker_rewrites_task_workbench_recovery_aliases():
    asyncio.run(_worker_rewrites_task_workbench_recovery_aliases())


def test_task_complete_refreshes_active_task_before_growth_guard():
    asyncio.run(_task_complete_refreshes_active_task_before_growth_guard())


async def _worker_rewrites_task_workbench_flat_fields_into_workbench_arg():
    from core.execution import WorkerLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry, ToolManifest, ToolParam, ToolResult

    async def _handler(params, ctx):
        assert isinstance(params.get("workbench"), dict)
        assert params["workbench"]["domain"] == "runtime"
        assert params["workbench"]["intent"] == "bootstrap"
        assert params["workbench"]["next_verification"] == "完成一次验证"
        assert params["workbench"]["evidence"] == ["已重跑", "已归档"]
        return ToolResult(summary="ok", state_delta={"seen": True})

    entry = ToolEntry(
        manifest=ToolManifest(
            name="task.workbench",
            description="demo",
            params=[ToolParam("workbench", "object", "workbench patch", required=True)],
        ),
        handler=_handler,
    )

    result = await WorkerLayer(_test_config()).dispatch(
        "tool-chain-worker",
        entry,
        JudgmentOutput(
            decision="act",
            chosen_action_id="task.workbench",
            params={
                "domain": "runtime",
                "intent": "bootstrap",
                "next_verification": "完成一次验证",
                "evidence": ["已重跑", "已归档"],
                "unknown": "ignored",
            },
        ),
        _tool_ctx(),
    )

    assert result.error is None
    assert result.state_delta["seen"] is True


async def _worker_rewrites_task_workbench_recovery_aliases():
    from core.execution import WorkerLayer
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry, ToolManifest, ToolParam, ToolResult

    async def _handler(params, ctx):
        assert params["task_id"] == 3788
        assert params["workbench"]["progress"] == ["已读取关键文件"]
        assert params["workbench"]["evidence"] == ["形成最小成长证据"]
        return ToolResult(summary="ok", state_delta={"seen": True})

    entry = ToolEntry(
        manifest=ToolManifest(
            name="task.workbench",
            description="demo",
            params=[
                ToolParam("workbench", "object", "workbench patch", required=True),
                ToolParam("task_id", "number", "task id", required=False),
            ],
        ),
        handler=_handler,
    )

    result = await WorkerLayer(_test_config()).dispatch(
        "tool-chain-worker",
        entry,
        JudgmentOutput(
            decision="act",
            chosen_action_id="task.workbench",
            params={
                "id": 3788,
                "workbench": {},
                "current_step": "已读取关键文件",
                "result_summary": "形成最小成长证据",
            },
        ),
        _tool_ctx(),
    )

    assert result.error is None
    assert result.state_delta["seen"] is True


def test_task_store_get_active_prefers_started_task_over_pending():
    asyncio.run(_task_store_get_active_prefers_started_task_over_pending())


async def _task_store_get_active_prefers_started_task_over_pending():
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            pending_id = await store.add_task(
                "pending critical",
                goal="should not outrank running task",
                priority="critical",
                status="pending",
            )
            running_id = await store.add_task(
                "running low",
                goal="must stay active",
                priority="low",
                status="in_progress",
            )

            active = await store.get_active()

            assert active is not None
            assert active.id == running_id
            assert active.id != pending_id
            assert active.status == "in_progress"
        finally:
            await store.close()


def test_task_tools_do_not_reenter_terminal_tasks():
    asyncio.run(_task_tools_do_not_reenter_terminal_tasks())


async def _task_tools_do_not_reenter_terminal_tasks():
    from store.task import TaskStore
    from tools.task import task_advance, task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "runtime.db")
        await store.open()
        try:
            done_id = await store.add_task("done task", goal="already finished", status="done")
            cancelled_id = await store.add_task("cancelled task", goal="already cancelled", status="cancelled")
            ctx = _tool_ctx(task_store=store)

            done_advance = await task_advance({"task_id": done_id}, ctx)
            cancelled_advance = await task_advance({"task_id": cancelled_id}, ctx)
            done_complete = await task_complete({"task_id": done_id}, ctx)
            cancelled_complete = await task_complete({"task_id": cancelled_id}, ctx)

            assert done_advance.skipped is True
            assert done_advance.summary == f"任务 [{done_id}] 已完成，不能再次推进"
            assert done_advance.metadata["task_id"] == done_id

            assert cancelled_advance.skipped is True
            assert cancelled_advance.summary == f"任务 [{cancelled_id}] 已取消，不能再次推进"
            assert cancelled_advance.metadata["task_id"] == cancelled_id

            assert done_complete.skipped is True
            assert done_complete.summary == f"任务 [{done_id}] 已完成"
            assert done_complete.metadata["task_id"] == done_id

            assert cancelled_complete.skipped is True
            assert cancelled_complete.summary == f"任务 [{cancelled_id}] 已取消，不能完成"
            assert cancelled_complete.metadata["task_id"] == cancelled_id

            done_task = await store.get_task_by_id(done_id)
            cancelled_task = await store.get_task_by_id(cancelled_id)
            assert done_task is not None and done_task.status == "done"
            assert cancelled_task is not None and cancelled_task.status == "cancelled"
        finally:
            await store.close()


def test_task_complete_blocks_action_first_tasks_until_verifiable_success():
    asyncio.run(_task_complete_blocks_action_first_tasks_until_verifiable_success())


def test_task_complete_blocks_self_drive_growth_without_evidence():
    asyncio.run(_task_complete_blocks_self_drive_growth_without_evidence())


def test_task_complete_blocks_unresolved_workbench_next_verification():
    asyncio.run(_task_complete_blocks_unresolved_workbench_next_verification())


def test_task_complete_blocks_real_workbench_state_delta_next_verification():
    asyncio.run(_task_complete_blocks_real_workbench_state_delta_next_verification())


def test_task_complete_blocks_external_workbench_next_verification():
    asyncio.run(_task_complete_blocks_external_workbench_next_verification())


def test_task_complete_blocks_english_workbench_next_verification():
    asyncio.run(_task_complete_blocks_english_workbench_next_verification())


def test_task_next_verification_completed_word_does_not_hide_actionable_check():
    from tools.task import _has_actionable_next_verification

    assert _has_actionable_next_verification("确认已完成测试并读取最新 pytest 结果") is True
    assert _has_actionable_next_verification("已完成，无需继续验证") is False
    assert _has_actionable_next_verification("already done; no need to rerun") is False


def test_task_complete_blocks_semantic_memory_next_verification_until_memory_written():
    asyncio.run(_task_complete_blocks_semantic_memory_next_verification_until_memory_written())


def test_task_complete_blocks_self_drive_deferred_implementation_candidate():
    asyncio.run(_task_complete_blocks_self_drive_deferred_implementation_candidate())


async def _task_complete_blocks_action_first_tasks_until_verifiable_success():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "action-first-complete.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "下载并应用配置",
                goal="用户给了 URL，需要下载、验证并应用",
                status="in_progress",
                result_json={
                    "cortex": {
                        "action_first": {"intent": "execute", "must_act": True},
                        "captured_inputs": [{"kind": "url", "value": "https://example.com/sub"}],
                    }
                },
            )
            ctx = _task_tool_ctx(store)

            no_evidence = await task_complete({"task_id": task_id}, ctx)
            assert no_evidence.skipped is True
            assert no_evidence.error == "ActionFirstCompletionBlocked"
            assert "缺少执行型证据" in no_evidence.summary

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="failed",
                tool_name="web.fetch",
                error_text="Timeout",
            )
            failed_latest = await task_complete({"task_id": task_id}, ctx)
            assert failed_latest.skipped is True
            assert failed_latest.error == "ActionFirstCompletionBlocked"
            assert "失败" in failed_latest.summary

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="web.fetch",
                log_text="fetched ok",
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_unresolved_workbench_next_verification():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "workbench-verification-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "self_drive_growth",
                "domain": "self_evolution",
                "evidence": ["recent logs show route drift probe returns too much context"],
                "next_verification": "优先定位 reasoner_route_drift_watch 探针定义，确认是否能压缩正常路径输出。",
                "completion_checks": [
                    "已形成一个明确的一行级改进候选。",
                    "当前不直接修改核心代码，因为还未定位具体实现文件。",
                ],
            }
            task_id = await store.add_task(
                "自我进化日志优化",
                goal="定位并减少无效探针日志注入",
                source="self_drive",
                status="in_progress",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="counted recent logs",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={"cortex": cortex},
                log_text="wrote next verification",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="memory.add_semantic",
                log_text="stored observation",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "WorkbenchVerificationPending"
            assert "未验证的下一步" in blocked.summary
            assert "reasoner_route_drift_watch" in blocked.summary
            assert blocked.state_delta["completion_blocked"] is True
            assert blocked.state_delta["completion_blocker"] == "WorkbenchVerificationPending"
            assert "reasoner_route_drift_watch" in blocked.state_delta["next_verification"]
            assert "file.read" in blocked.state_delta["suggested_tools"]
            assert "shell.run" in blocked.state_delta["suggested_tools"]
            assert blocked.metadata["completion_blocked"] is True
            assert blocked.metadata["recovery_next_step"] == blocked.state_delta["recovery_next_step"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="file.read",
                log_text="read probe definition",
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_real_workbench_state_delta_next_verification():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "real-workbench-state-delta-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "debug_runtime_issue",
                "domain": "runtime",
                "evidence": ["真实 tool result 会把 workbench 皮层写在 state_delta.cortex。"],
                "next_verification": "执行 shell.run 读取最近 runs，确认 workbench 后有真实验证工具。",
                "completion_checks": ["已确认完成门能读取真实 ToolResult 持久化形态。"],
            }
            task_id = await store.add_task(
                "验证真实 workbench 输出形态",
                goal="完成门必须识别 output_json.state_delta.cortex",
                source="external",
                status="in_progress",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={
                    "summary": "工作台已更新",
                    "state_delta": {"cortex": cortex, "run_id": 1},
                    "metadata": {},
                },
                log_text="wrote real ToolResult-shaped next verification",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "WorkbenchVerificationPending"
            assert "shell.run" in blocked.state_delta["next_verification"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="runs verified",
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_external_workbench_next_verification():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "external-workbench-verification-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "debug_runtime_issue",
                "domain": "user_problem",
                "evidence": ["用户反馈同模型下问题解决能力弱，当前已定位到完成门可能过早放行。"],
                "next_verification": "运行 pytest 验证 task.complete 未执行 workbench 下一步时会被拦截。",
                "completion_checks": [
                    "已确认普通外部任务也受 workbench 验证门约束。",
                    "已确认验证工具成功后才允许完成。",
                ],
            }
            task_id = await store.add_task(
                "修复普通问题任务过早完成",
                goal="确保普通用户问题也不能把 workbench 当完成证据",
                source="external",
                status="in_progress",
                current_step="已写入 workbench，等待执行 next_verification",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={"cortex": cortex},
                log_text="wrote next verification",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="memory.add_semantic",
                log_text="stored observation",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "WorkbenchVerificationPending"
            assert "未验证的下一步" in blocked.summary
            assert "pytest" in blocked.state_delta["next_verification"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="pytest passed",
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_english_workbench_next_verification():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "english-workbench-verification-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "verify remote",
                "domain": "git",
                "evidence": ["remote push failed once"],
                "next_verification": "run git ls-remote to verify remote connectivity.",
                "completion_checks": ["remote connectivity verified"],
            }
            task_id = await store.add_task(
                "验证远程连接",
                goal="English next_verification should still block completion",
                source="external",
                status="in_progress",
                current_step="已写入 workbench，等待执行英文 next_verification",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={"cortex": cortex},
                log_text="wrote english next verification",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "WorkbenchVerificationPending"
            assert "git ls-remote" in blocked.state_delta["next_verification"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="git ls-remote ok",
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_semantic_memory_next_verification_until_memory_written():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "semantic-memory-workbench-verification-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "self_drive_growth",
                "domain": "self_evolution",
                "evidence": ["已完成一次日志取证并形成可复用经验。"],
                "next_verification": "沉淀一条语义记忆：自驱取证任务不应 wait；随后完成本次轻量探索任务。",
                "completion_checks": ["经验已进入长期语义记忆。"],
            }
            task_id = await store.add_task(
                "沉淀自驱经验",
                goal="语义记忆沉淀必须先执行",
                source="self_drive",
                status="in_progress",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="collected log evidence",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={"cortex": cortex},
                log_text="wrote semantic memory next verification",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "WorkbenchVerificationPending"
            assert "沉淀一条语义记忆" in blocked.state_delta["next_verification"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="memory.add_semantic",
                log_text="stored self-drive lesson",
            )

            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_blocks_self_drive_deferred_implementation_candidate():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "self-drive-deferred-implementation-complete.db")
        await store.open()
        try:
            cortex = {
                "intent": "self_drive_growth",
                "domain": "self_evolution",
                "evidence": [
                    "日志统计发现 wait_no_progress=7。",
                    "脚本给出的改进候选：加强守卫，当 self_drive 任务 next_step 要求取证时优先低成本取证而不是 wait。",
                ],
                "hypothesis": "最低成本的一行级改进方向是强化守卫。",
                "next_verification": "沉淀一条语义记忆：自驱取证任务不应 wait；随后完成本次轻量探索任务。",
                "open_questions": [
                    "这个守卫应落在 judgment 层，还是通过现有 anti-loop/task-continuity skill 强化即可？",
                    "是否值得直接改代码，还是先沉淀为经验并观察下一次自驱任务表现？",
                ],
                "completion_checks": [
                    "已提出一条具体、低风险、可实现的一行级改进候选。",
                    "下一步无需立即改核心代码；先固化经验并在后续重复出现时再改守卫代码。",
                ],
            }
            task_id = await store.add_task(
                "自驱取证任务不应空转",
                goal="从日志中定位可落地的自我进化修复点",
                source="self_drive",
                status="in_progress",
                result_json={"cortex": cortex},
            )
            ctx = _task_tool_ctx(store)

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="counted wait_no_progress and repeated reads",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="task.workbench",
                output_json={"cortex": cortex},
                log_text="wrote implementation candidate but deferred code change",
            )
            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="memory.add_semantic",
                log_text="stored self-drive lesson",
            )

            blocked = await task_complete({"task_id": task_id}, ctx)
            assert blocked.skipped is True
            assert blocked.error == "SelfDriveImplementationDecisionPending"
            assert "是否修改" in blocked.summary
            assert blocked.state_delta["completion_blocked"] is True
            assert blocked.state_delta["completion_blocker"] == "SelfDriveImplementationDecisionPending"
            assert "定位文件" in blocked.state_delta["recovery_next_step"]
            assert blocked.state_delta["suggested_tools"] == ["file.read", "shell.run", "task.workbench"]
        finally:
            await store.close()


async def _task_complete_blocks_self_drive_growth_without_evidence():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "self-drive-growth-complete.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "自驱成长探索",
                goal="持续推进低成本成长闭环",
                source="self_drive",
                status="in_progress",
                result_json={
                    "cortex": {
                        "intent": "self_drive_growth",
                        "domain": "self_evolution",
                        "evidence": [],
                    }
                },
            )
            ctx = _task_tool_ctx(store)

            no_probe = await task_complete({"task_id": task_id}, ctx)
            assert no_probe.skipped is True
            assert no_probe.error == "SelfDriveGrowthIncomplete"
            assert "非 task 工具取证" in no_probe.summary
            assert no_probe.state_delta["completion_blocked"] is True
            assert no_probe.state_delta["completion_blocker"] == "SelfDriveGrowthIncomplete"
            assert "memory.search" in no_probe.state_delta["suggested_tools"]
            assert "task.workbench" in no_probe.state_delta["suggested_tools"]

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="memory.search",
                log_text="searched ok",
            )
            no_evidence = await task_complete({"task_id": task_id}, ctx)
            assert no_evidence.skipped is True
            assert no_evidence.error == "SelfDriveGrowthIncomplete"
            assert "成长证据" in no_evidence.summary
            assert no_evidence.state_delta["completion_blocked"] is True
            assert "证据" in no_evidence.state_delta["recovery_next_step"]

            await store.update_status(
                task_id,
                "in_progress",
                result_json={
                    "cortex": {
                        "intent": "self_drive_growth",
                        "domain": "self_evolution",
                        "evidence": ["memory.search confirmed one repeated wait pattern"],
                    }
                },
            )
            completed = await task_complete({"task_id": task_id}, ctx)
            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


async def _task_complete_refreshes_active_task_before_growth_guard():
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "self-drive-growth-stale-active.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "自驱成长探索",
                goal="同一 tick 内 workbench 写入后应能完成",
                source="self_drive",
                status="in_progress",
                result_json={
                    "cortex": {
                        "intent": "self_drive_growth",
                        "domain": "self_evolution",
                        "evidence": [],
                    }
                },
            )
            stale_task = await store.get_task_by_id(task_id)
            assert stale_task is not None
            ctx = _task_tool_ctx(store)
            ctx.active_task = stale_task

            await store.add_run(
                task_id=task_id,
                run_type="tool_chain",
                worker_type="reasoner",
                status="succeeded",
                tool_name="shell.run",
                log_text="verified ok",
            )
            await store.update_task_result(
                task_id,
                {
                    "cortex": {
                        "intent": "self_drive_growth",
                        "domain": "self_evolution",
                        "evidence": ["shell.run verified ok"],
                    }
                },
            )

            completed = await task_complete({}, ctx)

            assert completed.error is None
            assert completed.skipped is False
            assert completed.state_delta["task_id"] == task_id
            assert completed.state_delta["task_status"] == "done"
        finally:
            await store.close()


def test_task_tools_prefer_ctx_focus_task_over_global_active():
    asyncio.run(_task_tools_prefer_ctx_focus_task_over_global_active())


async def _task_tools_prefer_ctx_focus_task_over_global_active():
    from store.task import TaskStore
    from tools.task import task_advance

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "focus-tool-default.db")
        await store.open()
        try:
            global_active_id = await store.add_task(
                "全局活跃任务",
                goal="不应被这次默认命中",
                status="in_progress",
            )
            focus_task_id = await store.add_task(
                "当前焦点任务",
                goal="工具默认应命中这里",
                status="pending",
            )
            focus_task = await store.get_task_by_id(focus_task_id)
            assert focus_task is not None

            ctx = _tool_ctx(task_store=store, active_task=focus_task)
            result = await task_advance({}, ctx)

            refreshed_global = await store.get_task_by_id(global_active_id)
            refreshed_focus = await store.get_task_by_id(focus_task_id)

            assert result.skipped is False
            assert result.metadata["task_id"] == focus_task_id
            assert refreshed_focus is not None and refreshed_focus.status == "in_progress"
            assert refreshed_global is not None and refreshed_global.status == "in_progress"
        finally:
            await store.close()


def test_memory_set_fact_goes_through_metabolic_ledger():
    asyncio.run(_memory_set_fact_goes_through_metabolic_ledger())


async def _memory_set_fact_goes_through_metabolic_ledger():
    from store.task import TaskStore
    from tools.memory import memory_get_fact, memory_set_fact

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "memory-tool-metabolic.db")
        await store.open()
        try:
            ctx = _tool_ctx(task_store=store)
            result = await memory_set_fact(
                {"key": "life:test:set_fact", "value": {"ok": True}, "scope": "system"},
                ctx,
            )

            assert result.skipped is False
            got = await memory_get_fact({"key": "life:test:set_fact"}, ctx)
            assert got.skipped is False
            assert "life:test:set_fact" in got.summary

            recent = await store.ledger_recent(limit=3)
            assert recent
            top = recent[0]
            assert top["op"] == "set_fact"
            assert top["key"] == "life:test:set_fact"
            assert top["scope"] == "system"
            assert top["source"] == "tools/memory.set_fact"
            assert top["accepted"] is True
        finally:
            await store.close()


def test_task_plan_goes_through_metabolic_ledger():
    asyncio.run(_task_plan_goes_through_metabolic_ledger())


def test_memory_add_semantic_goes_through_metabolic_ledger():
    asyncio.run(_memory_add_semantic_goes_through_metabolic_ledger())


def test_task_complete_compiles_skill_through_metabolic_ledger():
    asyncio.run(_task_complete_compiles_skill_through_metabolic_ledger())


async def _task_complete_compiles_skill_through_metabolic_ledger():
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.task import task_complete

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "task-complete-semantic.db")
        await store.open()
        try:
            task_id = await store.add_task("compile skill task", goal="compile narrative")
            semantic = cast("Any", SemanticMemory(root / "semantic"))
            episodic = cast(
                "Any",
                SimpleNamespace(
                    load_for_context=lambda task_id, n_recent=5: "user did x\nassistant did y"
                ),
            )
            ctx = _tool_ctx(task_store=store, semantic=semantic, episodic=episodic)

            result = await task_complete({"task_id": task_id, "force": True}, ctx)

            assert result.skipped is False
            skill_id = result.state_delta["compiled_skill"]
            node = semantic.get(skill_id)
            assert node is not None
            assert node.kind == "learned_skill"
            assert node.source == "tools/task.complete"

            recent = await store.ledger_recent(limit=5)
            semantic_rows = [row for row in recent if row["op"] == "add_semantic_memory"]
            assert semantic_rows
            assert semantic_rows[0]["key"] == f"semantic:{skill_id}"
            assert semantic_rows[0]["source"] == "tools/task.complete"
        finally:
            await store.close()


async def _memory_add_semantic_goes_through_metabolic_ledger():
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.memory import memory_add_semantic

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "semantic-tool-metabolic.db")
        await store.open()
        try:
            semantic = cast("Any", SemanticMemory(root / "semantic"))
            ctx = _tool_ctx(task_store=store, semantic=semantic)

            result = await memory_add_semantic(
                {"title": "semantic ledger note", "body": "long term memory entry", "kind": "fact"},
                ctx,
            )

            node_id = result.evidence.split("node_id=", 1)[1]
            node = semantic.get(node_id)
            assert node is not None
            assert node.title == "semantic ledger note"

            recent = await store.ledger_recent(limit=3)
            top = recent[0]
            assert top["op"] == "add_semantic_memory"
            assert top["key"] == f"semantic:{node_id}"
            assert top["scope"] == "semantic"
            assert top["source"] == "tools/memory.add_semantic"
            assert top["accepted"] is True
            assert top["decision_basis"].startswith("memory.add_semantic")
        finally:
            await store.close()


async def _task_plan_goes_through_metabolic_ledger():
    from store.task import TaskStore
    from tools.plan import task_plan

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "task-plan-metabolic.db")
        await store.open()
        try:
            task_id = await store.add_task("plan ledger task", goal="record plan writes")
            ctx = _tool_ctx(task_store=store)

            result = await task_plan(
                {"task_id": task_id, "plan": [{"step": "read state", "status": "in_progress"}]},
                ctx,
            )

            assert result.skipped is False
            task = await store.get_task_by_id(task_id)
            assert task.extras["plan"][0]["step"] == "read state"
            recent = await store.ledger_recent(limit=3)
            assert recent[0]["op"] == "update_task_data"
            assert recent[0]["key"] == str(task_id)
            assert recent[0]["source"] == "tools/task.plan"
        finally:
            await store.close()


def test_file_edit_single_replace():
    """file.edit 单处替换成功。"""
    asyncio.run(_file_edit_single_replace())

async def _file_edit_single_replace():
    from tools.file import file_edit, file_read, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "test.py"
        await file_write({"path": str(fpath), "content": "x = 1\ny = 2\nz = 3\n"}, ctx)

        # 单处替换
        res = await file_edit({"path": str(fpath), "edits": [{"oldText": "y = 2", "newText": "y = 20"}]}, ctx)
        assert res.error is None
        assert "1 处替换" in res.summary

        # 验证内容
        content = await file_read({"path": str(fpath)}, ctx)
        assert content.summary == "x = 1\ny = 20\nz = 3\n"


def test_file_edit_multiple_replace():
    """file.edit 多处替换成功。"""
    asyncio.run(_file_edit_multiple_replace())

async def _file_edit_multiple_replace():
    from tools.file import file_edit, file_read, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "multi.py"
        await file_write({"path": str(fpath), "content": "a = 1\nb = 2\nc = 3\n"}, ctx)

        res = await file_edit({"path": str(fpath), "edits": [
            {"oldText": "a = 1", "newText": "a = 10"},
            {"oldText": "c = 3", "newText": "c = 30"},
        ]}, ctx)
        assert res.error is None
        assert "2 处替换" in res.summary

        content = await file_read({"path": str(fpath)}, ctx)
        assert "a = 10" in content.summary
        assert "c = 30" in content.summary


def test_file_edit_errors():
    """file.edit 错误处理：oldText 不唯一 / 不存在 / 空 edits / 文件不存在。"""
    asyncio.run(_file_edit_errors())

async def _file_edit_errors():
    from tools.file import file_edit, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "err.py"
        await file_write({"path": str(fpath), "content": "x = 1\nx = 1\ny = 2\n"}, ctx)

        # 文件不存在
        r = await file_edit({"path": str(root / "nonexistent.py"), "edits": [{"oldText": "a", "newText": "b"}]}, ctx)
        assert r.error == "FileNotFound"

        # 空 edits
        r2 = await file_edit({"path": str(fpath), "edits": []}, ctx)
        assert r2.skipped is True
        assert r2.error == "EmptyEdits"

        # oldText 不存在
        r3 = await file_edit({"path": str(fpath), "edits": [{"oldText": "ZZZ", "newText": "b"}]}, ctx)
        assert r3.skipped is True
        assert r3.error == "OldTextNotFound"

        # oldText 不唯一
        r4 = await file_edit({"path": str(fpath), "edits": [{"oldText": "x = 1", "newText": "x = 10"}]}, ctx)
        assert r4.skipped is True
        assert r4.error == "NonUniqueOldText"


def test_file_edit_fuzzy_respects_blank_lines():
    """file.edit 模糊匹配应保留空行约束，避免跨空行误命中。"""
    asyncio.run(_file_edit_fuzzy_respects_blank_lines())


async def _file_edit_fuzzy_respects_blank_lines():
    from tools.file import file_edit, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "blank.py"
        await file_write({"path": str(fpath), "content": "a = 1\n# note\nb = 2\n"}, ctx)

        # oldText 中包含空行，目标文件无空行时不应误匹配。
        res = await file_edit(
            {
                "path": str(fpath),
                "edits": [{"oldText": "a = 1\n\nb = 2", "newText": "a = 10\n\nb = 20"}],
            },
            ctx,
        )

        assert res.skipped is True
        assert res.error == "OldTextNotFound"


def test_skill_list_and_search():
    """skill.list 和 skill.search 工具正常返回。"""
    asyncio.run(_skill_list_and_search())

async def _skill_list_and_search():
    from tools.skill import skill_list, skill_search

    ws = _proj_root() / "workspace"
    ctx = _tool_ctx(workspace_dir=str(ws))

    r = await skill_list({"scope": "seed"}, ctx)
    assert r.error is None
    # 至少有 seed skills
    assert "runtime-bootstrap [seed]" in r.summary

    r2 = await skill_search({"query": "失败"}, ctx)
    assert r2.error is None
    # 搜索 "失败" 应匹配 failure-reflection
    assert "failure-reflection" in r2.summary

    # 搜索不存在的词 → 返回"未找到"，不是 skipped
    r3 = await skill_search({"query": "zxcvbnm_nonexistent_skill_query"}, ctx)
    assert r3.error is None
    assert "没有找到" in r3.summary


def test_skill_activate_reads_skill_markdown_and_resources():
    asyncio.run(_skill_activate_reads_skill_markdown_and_resources())


async def _skill_activate_reads_skill_markdown_and_resources():
    from tools.skill import skill_activate

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        skill_dir = root / "skills" / "sample-skill"
        (skill_dir / "references").mkdir(parents=True)
        (skill_dir / "references" / "REFERENCE.md").write_text("reference body", encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(
            """---
name: sample-skill
description: Use when testing activation.
---
先读取 references/REFERENCE.md，再执行下一步。
""",
            encoding="utf-8",
        )

        ctx = _tool_ctx(workspace_dir=d)
        res = await skill_activate({"name": "sample-skill"}, ctx)

        assert res.error is None
        assert "<skill_content name=\"sample-skill\">" in res.summary
        assert "references/REFERENCE.md" in res.summary
        assert res.metadata["skill"] == "sample-skill"


def test_skill_evolve_uses_judgment_provider_and_registry(monkeypatch):
    asyncio.run(_skill_evolve_uses_judgment_provider_and_registry(monkeypatch))


async def _skill_evolve_uses_judgment_provider_and_registry(monkeypatch):
    import core.evolution as evolution_mod
    from tools.registry import ToolContext, ToolRegistry
    from tools.skill import skill_evolve

    provider = object()
    registry = ToolRegistry()
    observed: dict[str, Any] = {}

    class _FakeEngine:
        def __init__(self, cfg: Any, provider_arg: Any, registry_arg: Any) -> None:
            observed["cfg"] = cfg
            observed["provider"] = provider_arg
            observed["registry"] = registry_arg

        async def evolve_skill(self, name: str, feedback: str, ctx: Any = None) -> Any:
            observed["name"] = name
            observed["feedback"] = feedback
            observed["ctx"] = ctx
            return SimpleNamespace(success=True, target="sample-skill", new_code="# updated")

    monkeypatch.setattr(evolution_mod, "EvolutionEngine", _FakeEngine)

    ctx = ToolContext(
        config=_test_config(),
        wm=cast("Any", None),
        task_store=cast("Any", None),
        episodic=cast("Any", None),
        semantic=cast("Any", None),
        emotion=cast("Any", None),
        judgment=SimpleNamespace(_provider=provider, _registry=registry),
    )

    res = await skill_evolve({"name": "sample-skill", "feedback": "tighten guardrails"}, ctx)

    assert res.error is None
    assert observed["provider"] is provider
    assert observed["registry"] is registry
    assert observed["name"] == "sample-skill"
    assert observed["feedback"] == "tighten guardrails"


def test_config_set_rejects_unknown_interval_key(monkeypatch):
    asyncio.run(_config_set_rejects_unknown_interval_key(monkeypatch))


async def _config_set_rejects_unknown_interval_key(monkeypatch):
    import tools.config as config_mod
    from tools.config import config_set

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "lingzhou.json"
        cfg_path.write_text((_proj_root() / "lingzhou.json.example").read_text(encoding="utf-8"), encoding="utf-8")
        before = cfg_path.read_text(encoding="utf-8")

        monkeypatch.setattr(config_mod, "_resolve_config_path", lambda ctx=None: cfg_path)

        res = await config_set({"key": "loop.interval", "value": "100"}, _tool_ctx())

        assert res.error == "UnknownConfigKey"
        assert "固定 tick interval 已废弃" in res.summary
        assert cfg_path.read_text(encoding="utf-8") == before


def test_config_set_accepts_duration_string_for_millisecond_fields(monkeypatch):
    asyncio.run(_config_set_accepts_duration_string_for_millisecond_fields(monkeypatch))


async def _config_set_accepts_duration_string_for_millisecond_fields(monkeypatch):
    import tools.config as config_mod
    from tools.config import config_set

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "lingzhou.json"
        cfg_path.write_text((_proj_root() / "lingzhou.json.example").read_text(encoding="utf-8"), encoding="utf-8")

        monkeypatch.setattr(config_mod, "_resolve_config_path", lambda ctx=None: cfg_path)

        res = await config_set({"key": "loop.wake_poll_interval", "value": "100ms"}, _tool_ctx())

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        assert res.error is None
        assert "✅ loop.wake_poll_interval" in res.summary
        assert cfg["loop"]["wake_poll_interval"] == 100


def test_subagent_filtered_registry_blocks_parent_mutations():
    from core.subagent import _DEFAULT_BLOCKED_TOOLS, _FilteredRegistry
    from tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.discover(_proj_root() / "tools")
    filtered = _FilteredRegistry(registry, None, set(_DEFAULT_BLOCKED_TOOLS))

    assert filtered.get("memory.set_fact") is None
    assert filtered.get("schedule.add") is None
    assert filtered.get("task.plan") is None
    assert filtered.get("memory.search") is not None
    assert filtered.get("memory.add_wm") is not None
    assert filtered.get("task.ask") is not None


def test_tool_registry_discover_skips_hidden_smoke_failed_modules():
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"visible_tool_{time.time_ns()}"
        manifest_name = f"probe.hidden_skip.{time.time_ns()}"
        (root / f"{stem}.py").write_text(
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='visible probe'))\n"
            "async def _visible_probe(params, ctx):\n"
            "    return ToolResult(summary='ok')\n",
            encoding="utf-8",
        )
        (root / f".{stem}.smoke-failed.py").write_text(
            "raise RuntimeError('hidden smoke-failed artifact must not be imported')\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)

        assert registry.get(manifest_name) is not None


def test_tool_registry_alias_available_during_tool_discovery():
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"alias_import_tool_{time.time_ns()}"
        manifest_name = f"probe.alias_import.{time.time_ns()}"
        (root / f"{stem}.py").write_text(
            "from tools.registry import ToolManifest, ToolResult, tool, tool_registry\n"
            "assert tool_registry is not None\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='alias import probe'))\n"
            "async def _alias_import_probe(params, ctx):\n"
            "    return ToolResult(summary='ok')\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)

        assert registry.get(manifest_name) is not None


def test_tool_registry_discover_rejects_legacy_manifest_kwargs():
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"legacy_tool_{time.time_ns()}"
        manifest_name = f"probe.legacy_compat.{time.time_ns()}"
        (root / f"{stem}.py").write_text(
            "from tools.registry import ToolManifest, ToolParam, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='legacy probe', parameters=[ToolParam(name='path', type='string', description='目录路径', default='.')], required_caps=('plan_bootstrap_exempt',)))\n"
            "async def _legacy_probe(params, ctx):\n"
            "    return ToolResult(summary='ok')\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)

        assert registry.get(manifest_name) is None


def test_tool_registry_discover_cleans_partial_module_after_import_failure():
    import sys

    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        broken_stem = f"broken_tool_{time.time_ns()}"
        dependent_stem = f"dependent_tool_{time.time_ns()}"
        broken_module_name = f"tools.{broken_stem}"
        dependent_manifest = f"probe.dependent_recover.{time.time_ns()}"

        (root / f"{broken_stem}.py").write_text(
            "PARTIAL = True\n"
            "raise RuntimeError('boom during import')\n"
            "def resolve_read_path():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        (root / f"{dependent_stem}.py").write_text(
            f"from tools.{broken_stem} import resolve_read_path\n"
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={dependent_manifest!r}, description='dependent probe'))\n"
            "async def _dependent_probe(params, ctx):\n"
            "    return ToolResult(summary=resolve_read_path())\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        # discover() 应隔离失败：不抛异常，但 broken 模块不留在 sys.modules
        registry.discover(root)

        assert broken_module_name not in sys.modules

        (root / f"{broken_stem}.py").write_text(
            "def resolve_read_path():\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )

        registry.discover(root)

        assert registry.get(dependent_manifest) is not None
        sys.modules.pop(broken_module_name, None)
        sys.modules.pop(f"tools.{dependent_stem}", None)


def test_tool_registry_reload_restores_previous_module_after_failure():
    import sys

    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = f"reloadable_tool_{time.time_ns()}"
        module_name = f"tools.{stem}"
        manifest_name = f"probe.reload_restore.{time.time_ns()}"

        (root / f"{stem}.py").write_text(
            "def resolve_read_path():\n"
            "    return 'ok'\n"
            "from tools.registry import ToolManifest, ToolResult, tool\n"
            f"@tool(ToolManifest(name={manifest_name!r}, description='reload probe'))\n"
            "async def _reload_probe(params, ctx):\n"
            "    return ToolResult(summary=resolve_read_path())\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        registry.discover(root)
        baseline = sys.modules[module_name]
        assert baseline.resolve_read_path() == "ok"

        (root / f"{stem}.py").write_text(
            "PARTIAL = True\n"
            "raise RuntimeError('boom during reload')\n"
            "def resolve_read_path():\n"
            "    return 'broken'\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="boom during reload"):
            registry.reload_tool(stem, root)

        restored = sys.modules[module_name]
        assert restored is baseline
        assert restored.resolve_read_path() == "ok"
        sys.modules.pop(module_name, None)


def test_subagent_runner_restores_parent_registry_after_child_exception():
    asyncio.run(_subagent_runner_restores_parent_registry_after_child_exception())


async def _subagent_runner_restores_parent_registry_after_child_exception():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, tool

    @tool(ToolManifest(
        name="probe.raise_registry",
        description="测试子灵异常后 registry 是否恢复",
        progress_category="info",
    ))
    async def _probe_raise_registry(params: dict[str, Any], ctx: Any) -> Any:
        raise RuntimeError("registry restore probe")

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.raise_registry",
                    params={},
                    rationale="probe registry restore",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="恢复 registry", max_ticks=2, allowed_tools=["probe.raise_registry"]),
                parent_ctx,
                cast("Any", _FakeJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "工具执行异常: registry restore probe" in result.last_summary
            assert execution._registry is registry  # type: ignore[attr-defined]
            assert execution._registry.get("task.ask") is not None  # type: ignore[attr-defined]
            assert execution._registry.get("probe.raise_registry") is not None  # type: ignore[attr-defined]
        finally:
            await store.close()


def test_subagent_runner_passes_filtered_registry_to_judgment():
    asyncio.run(_subagent_runner_passes_filtered_registry_to_judgment())


async def _subagent_runner_passes_filtered_registry_to_judgment():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolRegistry

    captured_visible_tools: list[str] = []

    class _RecordingJudgment:
        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            registry_override = kwargs.get("registry_override")
            assert registry_override is not None
            captured_visible_tools.extend(sorted(m.name for m in registry_override.list_manifests()))
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="检查子灵可见工具", max_ticks=1, allowed_tools=["task.list"]),
                parent_ctx,
                cast("Any", _RecordingJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "task.list" in captured_visible_tools
            assert "shell.run" not in captured_visible_tools
            assert "subagent.run" not in captured_visible_tools
        finally:
            await store.close()


def test_subagent_task_store_view_exposes_local_state_to_subsequent_ticks():
    asyncio.run(_subagent_task_store_view_exposes_local_state_to_subsequent_ticks())


async def _subagent_task_store_view_exposes_local_state_to_subsequent_ticks():
    from core.subagent import _SubagentTaskStoreView
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "child active task",
                goal="verify local task result overlay",
                status="in_progress",
                result_json={"summary": "before"},
            )
            active_task = await store.get_task_by_id(task_id)
            assert active_task is not None
            view = _SubagentTaskStoreView(store, active_task=active_task)

            await view.set_fact("control:durable_failure_policy", json.dumps({"threshold": 5}), scope="system")
            fact, found = await view.get_fact("control:durable_failure_policy")
            assert found is True
            assert json.loads(fact)["threshold"] == 5
            facts = await view.list_facts(prefix="control:", limit=5)
            assert any(key == "control:durable_failure_policy" for key, _ in facts)

            await view.update_task_result(task_id, {"last_run_status": "failed", "summary": "child summary"})
            active_task = await view.get_active()
            assert active_task is not None
            assert active_task.id == task_id
            assert active_task.result_json["last_run_status"] == "failed"
            assert active_task.result_json["summary"] == "child summary"
            fetched_task = await view.get_task_by_id(task_id)
            assert fetched_task is not None
            assert fetched_task.result_json["last_run_status"] == "failed"
            listed_tasks = await view.list_tasks(status="in_progress", limit=5)
            assert any(item.id == task_id and item.result_json.get("last_run_status") == "failed" for item in listed_tasks)

            run_id = await view.add_run(
                task_id=task_id,
                run_type="llm",
                worker_type="llm-worker",
                status="running",
                tool_name="probe.local",
                input_json={"query": "child"},
            )
            await view.update_run(run_id, status="failed", progress="phase-2", error_text="boom")

            run = await view.get_run_by_id(run_id)
            assert run is not None
            assert run.status == "failed"
            assert run.progress == "phase-2"
            runs = await view.list_runs(task_id=task_id, limit=5)
            assert any(item.id == run_id and item.error_text == "boom" for item in runs)

            await view.add_meta_reflection(
                reflection_id="local-r1",
                target_kind="threshold",
                trigger="failure_pattern",
                loop_level="single",
                diagnosis="child diagnosis",
                proposal="child proposal",
                verification_plan="rerun once",
                decision="apply",
                task_id=task_id,
                run_id=run_id,
                tool_name="probe.local",
            )

            reflections = await view.list_meta_reflections(limit=5)
            assert any(item.id == "local-r1" and item.run_id == run_id for item in reflections)
            filtered = await view.list_meta_reflections(limit=5, loop_level="single")
            assert any(item.id == "local-r1" for item in filtered)
        finally:
            await store.close()


def test_task_workbench_merges_general_problem_solving_state():
    asyncio.run(_task_workbench_merges_general_problem_solving_state())


def test_task_workbench_accepts_flattened_workbench_fields():
    asyncio.run(_task_workbench_accepts_flattened_workbench_fields())


async def _task_workbench_accepts_flattened_workbench_fields():
    from store.task import TaskStore
    from tools.workbench import task_workbench

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "workbench-flat.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "general problem solving",
                goal="flattened workbench support",
                status="in_progress",
                result_json={"cortex": {}},
            )
            task = await store.get_task_by_id(task_id)
            ctx = _tool_ctx(task_store=store, active_task=task)

            result = await task_workbench(
                {
                    "domain": "runtime",
                    "intent": "verify recovery",
                    "next_verification": "检查结果是否收敛",
                    "completion_checks": ["step done"],
                },
                ctx,
            )

            assert result.error is None
            refreshed = await store.get_task_by_id(task_id)
            assert refreshed is not None
            cortex = refreshed.result_json["cortex"]
            assert cortex["domain"] == "runtime"
            assert cortex["intent"] == "verify recovery"
            assert cortex["next_verification"] == "检查结果是否收敛"
            assert cortex["completion_checks"] == ["step done"]
            assert "已自动从顶层字段组装 workbench" in result.summary
        finally:
            await store.close()


async def _task_workbench_merges_general_problem_solving_state():
    from store.task import TaskStore
    from tools.workbench import task_workbench

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "workbench.db")
        await store.open()
        try:
            task_id = await store.add_task(
                "general problem solving",
                goal="verify durable workbench",
                status="in_progress",
                result_json={"cortex": {"domain": "git", "evidence": ["existing evidence"]}},
            )
            task = await store.get_task_by_id(task_id)
            ctx = _tool_ctx(task_store=store, active_task=task)

            result = await task_workbench(
                {
                    "workbench": {
                        "intent": "retry push after capability discovery",
                        "hypothesis": "transport failure is environmental",
                        "capabilities": [{"name": "git ls-remote", "status": "available"}],
                        "experiments": [{"target": "github", "status": "pending"}],
                        "next_verification": "run git ls-remote",
                        "unknown": "ignored",
                    }
                },
                ctx,
            )

            assert result.skipped is False
            assert result.metadata["tool_name"] == "task.workbench"
            refreshed = await store.get_task_by_id(task_id)
            assert refreshed is not None
            cortex = refreshed.result_json["cortex"]
            assert cortex["domain"] == "git"
            assert cortex["intent"] == "retry push after capability discovery"
            assert cortex["hypothesis"] == "transport failure is environmental"
            assert cortex["evidence"] == ["existing evidence"]
            assert cortex["capabilities"] == [{"name": "git ls-remote", "status": "available"}]
            assert cortex["experiments"] == [{"target": "github", "status": "pending"}]
            assert cortex["next_verification"] == "run git ls-remote"
            assert "unknown" not in cortex
        finally:
            await store.close()


def test_subagent_task_store_view_without_virtual_task_does_not_fallback_to_parent_active():
    asyncio.run(_subagent_task_store_view_without_virtual_task_does_not_fallback_to_parent_active())


async def _subagent_task_store_view_without_virtual_task_does_not_fallback_to_parent_active():
    from core.subagent import _SubagentTaskStoreView
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            await store.add_task(
                "parent active task",
                goal="unbound child view must not inherit this active task",
                status="in_progress",
            )
            view = _SubagentTaskStoreView(store)

            active_task = await view.get_active()

            assert active_task is None
        finally:
            await store.close()


def test_subagent_task_store_view_hides_parent_waiting_tasks_from_child_context():
    asyncio.run(_subagent_task_store_view_hides_parent_waiting_tasks_from_child_context())


async def _subagent_task_store_view_hides_parent_waiting_tasks_from_child_context():
    from core.subagent import _SubagentTaskStoreView
    from store.task import TaskStore

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            waiting_task_id = await store.add_task(
                "parent waiting task",
                goal="should not leak into child waiting context",
                status="waiting",
                wait_kind="external",
                wait_key="user-input",
                next_step="wait for parent input",
            )
            view = _SubagentTaskStoreView(store)

            waiting_tasks = await view.list_tasks(status="waiting", limit=5)

            assert waiting_tasks == []
            parent_waiting = await store.list_tasks(status="waiting", limit=5)
            assert any(item.id == waiting_task_id for item in parent_waiting)
        finally:
            await store.close()


def test_subagent_runner_uses_virtual_active_task_instead_of_parent_task():
    asyncio.run(_subagent_runner_uses_virtual_active_task_instead_of_parent_task())


async def _subagent_runner_uses_virtual_active_task_instead_of_parent_task():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    observed: dict[str, Any] = {}

    @tool(ToolManifest(
        name="probe.capture_active_task",
        description="测试子灵 active task 使用本地虚拟 task 而非父灵 task",
        progress_category="info",
    ))
    async def _probe_capture_active_task(params: dict[str, Any], ctx: Any) -> Any:
        task = await ctx.task_store.get_active()
        observed["tool_task"] = task
        assert task is not None
        await ctx.task_store.update_task_result(task.id, {"probe_marker": "child-local"})
        return ToolResult(summary=f"active-task={task.id}:{task.title}", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0
            self._seen_tasks: list[Any] = []

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            task = await args[2].get_active()
            self._seen_tasks.append(task)
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.capture_active_task",
                    params={},
                    rationale="probe virtual child task",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent active task",
                goal="should stay in parent only",
                status="in_progress",
                current_step="parent-step",
                next_step="parent-next",
                result_json={"summary": "parent-summary"},
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            judgment = _FakeJudgment()
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="隔离子灵 active task", max_ticks=2, allowed_tools=["probe.capture_active_task"]),
                parent_ctx,
                cast("Any", judgment),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert len(judgment._seen_tasks) == 2
            first_task = judgment._seen_tasks[0]
            second_task = judgment._seen_tasks[1]
            assert first_task is not None
            assert first_task.id < 0
            assert first_task.id != parent_task_id
            assert first_task.title.startswith("子灵任务: ")
            assert first_task.goal == "隔离子灵 active task"
            assert first_task.current_step == ""
            assert first_task.next_step == ""
            assert first_task.result_json == {}
            assert second_task is not None
            assert second_task.id == first_task.id
            assert second_task.result_json["last_run_status"] == "succeeded"
            assert second_task.result_json["probe_marker"] == "child-local"

            tool_task = observed["tool_task"]
            assert tool_task is not None
            assert tool_task.id == first_task.id
            assert tool_task.title == first_task.title
            assert result.last_summary == f"active-task={first_task.id}:{first_task.title}"

            parent_active = await store.get_active()
            assert parent_active is not None
            assert parent_active.id == parent_task_id
            assert parent_active.title == "parent active task"
            assert parent_active.result_json["summary"] == "parent-summary"
        finally:
            await store.close()


def test_tool_context_active_task_does_not_fallback_to_global_active():
    asyncio.run(_tool_context_active_task_does_not_fallback_to_global_active())


async def _tool_context_active_task_does_not_fallback_to_global_active():
    from core.config import Config
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext

    cfg = Config.model_validate({
        "providers": {
            "copilot": {
                "type": "openai_compat",
                "mode": "copilot",
                "base_url": "https://api.githubcopilot.com",
                "api_key_env": "GITHUB_TOKEN",
            },
        },
        "model": "copilot/gpt-5.4",
        "thinking": "low",
        "temperature": 0.7,
        "timeout": 60.0,
    })

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "tool-context.db")
        await store.open()
        try:
            await store.add_task(
                "global active task",
                goal="ctx.get_active_task must not inherit this task implicitly",
                status="in_progress",
            )
            ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(capacity=20),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                active_task=None,
            )

            active_task = await ctx.get_active_task()

            assert active_task is None
        finally:
            await store.close()


def test_subagent_task_list_does_not_expose_parent_tasks():
    asyncio.run(_subagent_task_list_does_not_expose_parent_tasks())


async def _subagent_task_list_does_not_expose_parent_tasks():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolRegistry

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="task.list",
                    params={"limit": 5},
                    rationale="probe task list isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            await store.add_task(
                "parent active task",
                goal="should not show in child task.list",
                status="in_progress",
                current_step="parent-step",
            )
            await store.add_task(
                "parent pending task",
                goal="should also stay hidden",
                status="pending",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="列出子灵可见任务", max_ticks=2, allowed_tools=["task.list"]),
                parent_ctx,
                cast("Any", _FakeJudgment()),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert "子灵任务: 列出子灵可见任务" in result.last_summary
            assert "parent active task" not in result.last_summary
            assert "parent pending task" not in result.last_summary
        finally:
            await store.close()


def test_subagent_explicit_task_id_does_not_expose_parent_task():
    asyncio.run(_subagent_explicit_task_id_does_not_expose_parent_task())


async def _subagent_explicit_task_id_does_not_expose_parent_task():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_task_by_id",
        description="测试子灵不能通过显式 task_id 读取父灵任务",
        progress_category="info",
    ))
    async def _probe_read_task_by_id(params: dict[str, Any], ctx: Any) -> Any:
        task = await ctx.task_store.get_task_by_id(int(params.get("task_id") or 0))
        if task is None:
            return ToolResult(summary="task-by-id=not-found", kind="execute_result", priority=0.5)
        return ToolResult(summary=f"task-by-id={task.id}:{task.title}", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self, parent_task_id: int) -> None:
            self._calls = 0
            self._parent_task_id = parent_task_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_task_by_id",
                    params={"task_id": self._parent_task_id},
                    rationale="probe explicit task id isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent secret task",
                goal="should not be readable by child via explicit task_id",
                status="in_progress",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断显式 task_id 泄漏", max_ticks=2, allowed_tools=["probe.read_task_by_id"]),
                parent_ctx,
                cast("Any", _FakeJudgment(parent_task_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "task-by-id=not-found"
        finally:
            await store.close()


def test_subagent_run_history_does_not_expose_parent_runs():
    asyncio.run(_subagent_run_history_does_not_expose_parent_runs())


async def _subagent_run_history_does_not_expose_parent_runs():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_runs",
        description="测试子灵不能读取父灵 runs 历史",
        progress_category="info",
    ))
    async def _probe_read_runs(params: dict[str, Any], ctx: Any) -> Any:
        runs = await ctx.task_store.list_runs(limit=5)
        parent_run = await ctx.task_store.get_run_by_id(int(params.get("run_id") or 0))
        first_tool = runs[0].tool_name if runs else "-"
        parent_state = "hit" if parent_run is not None else "miss"
        return ToolResult(
            summary=f"runs={len(runs)} first={first_tool} parent-run={parent_state}",
            kind="execute_result",
            priority=0.5,
        )

    class _FakeJudgment:
        def __init__(self, parent_run_id: int) -> None:
            self._calls = 0
            self._parent_run_id = parent_run_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_runs",
                    params={"run_id": self._parent_run_id},
                    rationale="probe run history isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent task with run",
                goal="should not leak runs into child",
                status="in_progress",
            )
            parent_run_id = await store.add_run(
                task_id=parent_task_id,
                tool_name="probe.parent",
                status="succeeded",
                output_json={"summary": "parent run"},
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断 parent run 泄漏", max_ticks=2, allowed_tools=["probe.read_runs"]),
                parent_ctx,
                cast("Any", _FakeJudgment(parent_run_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "runs=1 first=probe.read_runs parent-run=miss"
        finally:
            await store.close()


def test_subagent_failure_and_reflection_history_do_not_expose_parent_state():
    asyncio.run(_subagent_failure_and_reflection_history_do_not_expose_parent_state())


async def _subagent_failure_and_reflection_history_do_not_expose_parent_state():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.read_failure_state",
        description="测试子灵不能读取父灵 failures/meta reflections",
        progress_category="info",
    ))
    async def _probe_read_failure_state(params: dict[str, Any], ctx: Any) -> Any:
        failures = await ctx.task_store.list_failures(limit=5)
        task_failures = await ctx.task_store.list_failures_for_task(str(params.get("task_id") or ""), limit=5)
        reflections = await ctx.task_store.list_meta_reflections(limit=5)
        return ToolResult(
            summary=(
                f"failures={len(failures)} "
                f"task-failures={len(task_failures)} "
                f"reflections={len(reflections)}"
            ),
            kind="execute_result",
            priority=0.5,
        )

    class _FakeJudgment:
        def __init__(self, parent_task_id: int) -> None:
            self._calls = 0
            self._parent_task_id = parent_task_id

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.read_failure_state",
                    params={"task_id": self._parent_task_id},
                    rationale="probe failure/reflection isolation",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            parent_task_id = await store.add_task(
                "parent task with failures",
                goal="should not leak failure state into child",
                status="in_progress",
            )
            await store.record_failure(
                kind="probe.parent_failure",
                summary="parent failure",
                context="parent failure context",
                task_id=str(parent_task_id),
            )
            await store.add_meta_reflection(
                reflection_id="parent-r1",
                target_kind="tool",
                trigger="failure_pattern",
                loop_level="single",
                diagnosis="parent diagnosis",
                proposal="parent proposal",
                verification_plan="parent rerun",
                decision="apply",
                task_id=parent_task_id,
                run_id=1,
                tool_name="probe.parent_failure",
            )
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            result = await make_subagent_runner(
                SubagentConfig(goal="阻断 parent failure 泄漏", max_ticks=2, allowed_tools=["probe.read_failure_state"]),
                parent_ctx,
                cast("Any", _FakeJudgment(parent_task_id)),
                execution,
                registry,
            ).run()

            assert result.completed is True
            assert result.last_summary == "failures=0 task-failures=0 reflections=0"
        finally:
            await store.close()


def test_subagent_runner_does_not_pollute_parent_store():
    asyncio.run(_subagent_runner_does_not_pollute_parent_store())


async def _subagent_runner_does_not_pollute_parent_store():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.perception import EmotionState
    from core.perception.ethos import EthosState
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolRegistry

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0
            self._last_emotion: EmotionState | None = None
            self._last_ethos: EthosState | None = None

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            self._last_emotion = cast("EmotionState", args[5])
            self._last_ethos = cast("EthosState | None", kwargs.get("ethos_state"))
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="task.ask",
                    params={"question": "请确认只读子灵是否生效？"},
                    rationale="ask once",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.23, baseline_arousal=0.34),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            execution = ExecutionLayer(registry, cfg)
            judgment = _FakeJudgment()
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=EpisodicMemory(root / "episodic"),
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            runner = make_subagent_runner(
                SubagentConfig(goal="只读检查", max_ticks=2),
                parent_ctx,
                cast("Any", judgment),
                execution,
                registry,
            )

            result = await runner.run()

            assert result.completed is True
            assert "已登记用户澄清请求" in result.last_summary
            assert await store.list_runs(limit=10) == []
            assert await store.list_failures(limit=10) == []
            assert await store.list_facts(prefix="durable_failure:", limit=10) == []
            assert await store.list_meta_reflections(limit=10) == []
            assert judgment._last_emotion is not None
            assert judgment._last_emotion.valence == pytest.approx(0.23)
            assert judgment._last_emotion.arousal == pytest.approx(0.34)
            assert judgment._last_ethos is not None
            assert judgment._last_ethos.values.truth == pytest.approx(0.91)
            assert judgment._last_ethos.values.caution == pytest.approx(0.81)
        finally:
            await store.close()


def test_subagent_runner_shared_memory_does_not_write_parent_episodic():
    asyncio.run(_subagent_runner_shared_memory_does_not_write_parent_episodic())


def test_subagent_episodic_view_accepts_parent_memory_signatures():
    from core.subagent import _SubagentEpisodicView

    class _Parent:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def load_for_context(self, task_id: str | None, n_recent: int = 20) -> str:
            self.calls.append(("load_for_context", task_id, n_recent))
            return "task narrative"

        def load_for_chat_context(
            self,
            chat_id: str | None,
            n_recent: int = 20,
            *,
            max_chars: int | None = None,
        ) -> str:
            self.calls.append(("load_for_chat_context", chat_id, n_recent, max_chars))
            return "chat narrative"

        def load_for_interlocutor_context(
            self,
            interlocutor_id: str | None,
            n_recent: int = 20,
            *,
            max_chars: int | None = None,
        ) -> str:
            self.calls.append(("load_for_interlocutor_context", interlocutor_id, n_recent, max_chars))
            return "speaker narrative"

        def load_for_task_narrative(self, task_id: str | None, n_recent: int = 20) -> str:
            self.calls.append(("load_for_task_narrative", task_id, n_recent))
            return "task-only narrative"

        def load_recent_daily_context(self, days: int = 2, max_chars: int = 1200) -> str:
            self.calls.append(("load_recent_daily_context", days, max_chars))
            return "daily narrative"

        def search(self, query: str, max_chars: int = 2000, exclude_task_id: str | None = None) -> str:
            self.calls.append(("search", query, max_chars, exclude_task_id))
            return "search result"

    parent = _Parent()
    view = _SubagentEpisodicView(parent)

    assert view.load_for_context("task-1", 6) == "task narrative"
    assert view.load_for_chat_context("chat-1", 7, max_chars=800) == "chat narrative"
    assert view.load_for_interlocutor_context("speaker-1", 8, max_chars=900) == "speaker narrative"
    assert view.load_for_task_narrative("task-2", 9) == "task-only narrative"
    assert view.load_recent_daily_context(3, 1000) == "daily narrative"
    assert parent.calls == [
        ("load_for_context", "task-1", 6),
        ("load_for_chat_context", "chat-1", 7, 800),
        ("load_for_interlocutor_context", "speaker-1", 8, 900),
        ("load_for_task_narrative", "task-2", 9),
        ("load_recent_daily_context", 3, 1000),
    ]


async def _subagent_runner_shared_memory_does_not_write_parent_episodic():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from core.subagent import SubagentConfig, make_subagent_runner
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool

    @tool(ToolManifest(
        name="probe.ep_write",
        description="测试子灵 shared-memory 是否会污染父灵 episodic",
        progress_category="info",
    ))
    async def _probe_ep_write(params: dict[str, Any], ctx: Any) -> Any:
        ctx.episodic.record(role="reflection", content="shared-memory-write")
        return ToolResult(summary="wrote episodic", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.ep_write",
                    params={},
                    rationale="probe episodic write",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            episodic = EpisodicMemory(root / "episodic")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12),
                emotion=SimpleNamespace(baseline_valence=0.5, baseline_arousal=0.5),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            execution = ExecutionLayer(registry, cfg)
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=episodic,
                semantic=SemanticMemory(root / "semantic"),
                emotion=cast("Any", SimpleNamespace()),
                judgment=None,
                execution=execution,
                registry=registry,
            )

            runner = make_subagent_runner(
                SubagentConfig(goal="检查 shared memory episodic 只读", max_ticks=2, allowed_tools=["probe.ep_write"]),
                parent_ctx,
                cast("Any", _FakeJudgment()),
                execution,
                registry,
            )

            result = await runner.run()

            assert result.completed is True
            assert result.last_summary == "wrote episodic"
            assert episodic.load_for_context(None, n_recent=4000) == ""
            assert episodic.get_recent_turns(task_id=None, limit=5) == []
        finally:
            await store.close()


def test_subagent_absorb_persists_parent_semantic_node_with_provenance():
    asyncio.run(_subagent_absorb_persists_parent_semantic_node_with_provenance())


async def _subagent_absorb_persists_parent_semantic_node_with_provenance():
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.subagent import subagent_absorb

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent-absorb.db")
        await store.open()
        try:
            semantic = SemanticMemory(root / "semantic")
            ctx = _tool_ctx(task_store=store, semantic=semantic)

            res = await subagent_absorb(
                {
                    "subagent_id": "sub-a1",
                    "memories_json": json.dumps([
                        {
                            "id": "note-1",
                            "kind": "learned_insight",
                            "title": "定位异常根因",
                            "body": "execution 与 semantic 接口未对齐。",
                            "activation": 0.76,
                            "valence": 0.61,
                            "importance": 0.88,
                            "tags": ["reflection", "execution"],
                            "created_at": "2026-05-22T10:00:00+00:00",
                        }
                    ], ensure_ascii=False),
                },
                ctx,
            )

            assert res.error is None
            assert res.metadata["absorbed"] == 1
            assert res.metadata["requested_total"] == 1
            node = semantic.get("absorbed-sub-a1-note-1")
            assert node is not None
            assert node.title == "定位异常根因"
            assert node.body == "execution 与 semantic 接口未对齐。"
            assert node.importance == pytest.approx(0.88)
            assert node.source == "subagent:sub-a1"
            assert "reflection" in node.tags
            assert "subagent:sub-a1" in node.tags

            recent = await store.ledger_recent(limit=3)
            assert recent[0]["op"] == "add_semantic_memory"
            assert recent[0]["source"] == "subagent:sub-a1"
        finally:
            await store.close()


def test_subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution():
    asyncio.run(_subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution())


def test_subagent_run_accepts_allowed_tools_list(monkeypatch):
    asyncio.run(_subagent_run_accepts_allowed_tools_list(monkeypatch))


async def _subagent_run_accepts_allowed_tools_list(monkeypatch):
    import core.subagent as subagent_core
    from core.subagent import SubagentResult
    from tools.subagent import subagent_run

    captured: dict[str, Any] = {}

    class _FakeRunner:
        async def run(self) -> SubagentResult:
            return SubagentResult(
                subagent_id="sub-list",
                goal="list allowed tools",
                ticks_run=1,
                completed=True,
                error=None,
                last_summary="ok",
                observations=[],
            )

    def _fake_make_subagent_runner(sub_cfg, *args, **kwargs):
        captured["allowed_tools"] = sub_cfg.allowed_tools
        return _FakeRunner()

    monkeypatch.setattr(subagent_core, "make_subagent_runner", _fake_make_subagent_runner)

    ctx = _tool_ctx(wm=SimpleNamespace(add=lambda *args, **kwargs: None))
    ctx.judgment = object()
    ctx.execution = object()
    ctx.registry = object()

    res = await subagent_run(
        {
            "goal": "list allowed tools",
            "allowed_tools": ["task.list", "memory.search"],
        },
        ctx,
    )

    assert res.error is None
    assert captured["allowed_tools"] == ["task.list", "memory.search"]


async def _subagent_run_isolated_memory_returns_absorbable_memories_without_parent_semantic_pollution():
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentOutput
    from memory.working import WorkingMemory
    from store.semantic import MemoryNode, SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolContext, ToolManifest, ToolRegistry, ToolResult, tool
    from tools.subagent import subagent_run

    @tool(ToolManifest(
        name="probe.semantic_note",
        description="测试 isolated-memory 子灵写入独立 semantic",
        progress_category="info",
    ))
    async def _probe_semantic_note(params: dict[str, Any], ctx: Any) -> Any:
        ctx.semantic.upsert(MemoryNode(
            id="sub-note-1",
            kind="learned_insight",
            title="隔离语义吸收测试",
            body="isolated-memory 子灵应返回 absorbable memories 且不污染父灵。",
            activation=0.73,
            valence=0.58,
            importance=0.81,
            tags=["subagent", "isolated"],
            source="child-runtime",
        ))
        return ToolResult(summary="wrote isolated semantic", kind="execute_result", priority=0.5)

    class _FakeJudgment:
        def __init__(self) -> None:
            self._calls = 0

        async def decide(self, *args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == 1:
                return JudgmentOutput(
                    decision="act",
                    chosen_action_id="probe.semantic_note",
                    params={},
                    rationale="probe isolated semantic write",
                )
            return JudgmentOutput.wait(reason="done")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent.db")
        await store.open()
        try:
            registry = ToolRegistry()
            registry.discover(_proj_root() / "tools")
            cfg = cast("Any", SimpleNamespace(
                loop=SimpleNamespace(act=True, debug=False, workspace_dir=str(root)),
                memory=SimpleNamespace(working_capacity=12, max_events=20),
                memory_dir=root / "memory",
                emotion=SimpleNamespace(baseline_valence=0.4, baseline_arousal=0.3),
                soul=_test_soul_cfg(),
                thresholds=SimpleNamespace(
                    durable_failure_threshold=3,
                    durable_failure_ttl_sec=7200,
                ),
            ))
            execution = ExecutionLayer(registry, cfg)
            parent_semantic = SemanticMemory(root / "semantic")
            parent_ctx = ToolContext(
                config=cfg,
                wm=WorkingMemory(12),
                task_store=store,
                episodic=cast("Any", SimpleNamespace(record=lambda *args, **kwargs: None, record_event=lambda *args, **kwargs: None)),
                semantic=parent_semantic,
                emotion=cast("Any", SimpleNamespace()),
                judgment=cast("Any", _FakeJudgment()),
                execution=execution,
                registry=registry,
            )

            res = await subagent_run(
                {
                    "goal": "隔离语义吸收",
                    "max_ticks": 2,
                    "allowed_tools": "probe.semantic_note",
                    "isolated_memory": True,
                },
                parent_ctx,
            )

            assert res.error is None
            assert res.metadata["absorbed_memories_count"] == 1
            assert res.metadata["memory_dir"]
            assert parent_semantic.get("sub-note-1") is None
            absorbed = res.metadata["absorbed_memories"]
            assert len(absorbed) == 1
            assert absorbed[0]["title"] == "隔离语义吸收测试"
            assert absorbed[0]["body"] == "isolated-memory 子灵应返回 absorbable memories 且不污染父灵。"
            assert absorbed[0]["source"] == "child-runtime"
            assert absorbed[0]["importance"] == pytest.approx(0.81)
        finally:
            await store.close()


def test_subagent_absorb_surfaces_truncation_and_invalid_nodes():
    asyncio.run(_subagent_absorb_surfaces_truncation_and_invalid_nodes())


async def _subagent_absorb_surfaces_truncation_and_invalid_nodes():
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.subagent import subagent_absorb

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        store = TaskStore(root / "subagent-absorb-truncate.db")
        await store.open()
        try:
            semantic = SemanticMemory(root / "semantic")
            ctx = _tool_ctx(task_store=store, semantic=semantic)

            nodes = [
                {"id": "good-1", "title": "A", "body": "alpha"},
                {"id": "good-2", "title": "B", "body": "beta"},
                {"id": "bad-3", "title": "", "body": "missing title"},
                {"id": "good-4", "title": "D", "body": "delta"},
                {"id": "good-5", "title": "E", "body": "epsilon"},
                {"id": "good-6", "title": "F", "body": "zeta"},
            ]

            res = await subagent_absorb(
                {
                    "subagent_id": "sub-b2",
                    "memories_json": json.dumps(nodes, ensure_ascii=False),
                    "max_absorb": 5,
                },
                ctx,
            )

            assert res.error is None
            assert res.metadata["requested_total"] == 6
            assert res.metadata["selected_total"] == 5
            assert res.metadata["truncated"] == 1
            assert res.metadata["invalid"] == 1
            assert "另有 1 条因 max_absorb 未吸收" in res.summary
            assert "1 条缺少标题或正文已跳过" in res.summary
            assert semantic.get("absorbed-sub-b2-good-6") is None
        finally:
            await store.close()


def test_browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch):
    asyncio.run(_browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch))


async def _browser_navigate_failure_uses_stdout_when_stderr_empty(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 7, "blocked by upstream gateway", ""

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateTargetBlocked"
    assert "exit=7" in res.summary
    assert "blocked by upstream gateway" in res.summary


def test_browser_navigate_timeout_classified(monkeypatch):
    asyncio.run(_browser_navigate_timeout_classified(monkeypatch))


async def _browser_navigate_timeout_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return -1, "", "操作超时"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateTimeout"
    assert "操作超时" in res.summary


def test_browser_navigate_network_unreachable_classified(monkeypatch):
    asyncio.run(_browser_navigate_network_unreachable_classified(monkeypatch))


async def _browser_navigate_network_unreachable_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 2, "", "net::ERR_NAME_NOT_RESOLVED"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateNetworkUnreachable"
    assert "网络不可达" in res.summary


def test_browser_navigate_dependency_missing_classified(monkeypatch):
    asyncio.run(_browser_navigate_dependency_missing_classified(monkeypatch))


async def _browser_navigate_dependency_missing_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 3, "", "Failed to launch browser process! libnss3.so: cannot open shared object file"

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateDependencyMissing"
    assert "浏览器依赖缺失" in res.summary


def test_browser_navigate_blank_page_classified(monkeypatch):
    asyncio.run(_browser_navigate_blank_page_classified(monkeypatch))


async def _browser_navigate_blank_page_classified(monkeypatch):
    import tools.browser as browser_mod

    monkeypatch.setattr(browser_mod, "_find_browser", lambda: "agent-browser")

    async def _fake_browser_run(*args: str, timeout: int = 30):
        return 0, "   \n   ", ""

    monkeypatch.setattr(browser_mod, "_browser_run", _fake_browser_run)
    res = await browser_mod.browser_navigate({"url": "https://example.com"}, _tool_ctx())

    assert res.error == "NavigateBlankPage"
    assert "页面空白" in res.summary


def test_web_fetch_recreates_closed_shared_client(monkeypatch):
    asyncio.run(_web_fetch_recreates_closed_shared_client(monkeypatch))


async def _web_fetch_recreates_closed_shared_client(monkeypatch):
    import tools.web as web_mod

    class _ClosedClient:
        is_closed = True

    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.text = "<html><body>hello web</body></html>"
            self.headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class _FreshClient:
        is_closed = False

        async def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert url == "https://example.com"
            return _FakeResponse()

    created: dict[str, Any] = {}

    def _factory(**kwargs: Any):
        created["kwargs"] = kwargs
        return _FreshClient()

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(web_mod, "_http_client", _ClosedClient())

    res = await web_mod.web_fetch({"url": "https://example.com"}, _tool_ctx())

    assert created["kwargs"]["follow_redirects"] is True
    assert res.error is None
    assert res.skipped is False
    assert "获取成功" in res.summary


def test_exec_empty_command():
    """exec 空命令应被拒绝。"""
    asyncio.run(_exec_empty_command())


def test_web_fetch_uses_https_proxy_env(monkeypatch):
    asyncio.run(_web_fetch_uses_https_proxy_env(monkeypatch))


async def _web_fetch_uses_https_proxy_env(monkeypatch):
    import tools.web as web_mod

    class _FakeResponse:
        status_code = 200
        text = "<html><body>proxied</body></html>"
        headers = {"content-type": "text/html; charset=utf-8"}

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        is_closed = False

        async def request(self, method: str, url: str, **kwargs):
            return _FakeResponse()

    created: dict[str, Any] = {}

    def _factory(**kwargs: Any):
        created["kwargs"] = kwargs
        return _FakeClient()

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(web_mod, "_http_client", None)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://https-proxy.internal:8443")

    res = await web_mod.web_fetch({"url": "https://example.com"}, _tool_ctx())

    assert res.error is None
    assert created["kwargs"]["proxy"] == "http://https-proxy.internal:8443"
    assert created["kwargs"]["trust_env"] is False


async def _exec_empty_command():
    from tools.exec import exec_run

    ctx = _tool_ctx()
    res = await exec_run({"command": ""}, ctx)
    assert res.skipped is True
    assert res.error == "EmptyCommand"


def test_resource_guard_detects_embedding_oom_preflight():
    from core.resource_guard import (
        local_embedding_memory_preflight,
        looks_like_local_embedding_command,
        parse_mem_available_mib,
    )

    assert parse_mem_available_mib("MemAvailable:    1048576 kB\n") == 1024
    assert looks_like_local_embedding_command("python build_embeddings.py --model BAAI/bge-m3")

    blocked = local_embedding_memory_preflight(
        command="python build_embeddings.py --model BAAI/bge-m3",
        min_available_mib=12288,
        available_mib=2048,
    )

    assert blocked.matched is True
    assert blocked.ok is False
    assert blocked.reason == "insufficient_available_memory_for_local_embedding"


def test_exec_limits_local_embedding_rebuild_when_memory_low(monkeypatch):
    asyncio.run(_exec_limits_local_embedding_rebuild_when_memory_low(monkeypatch))


async def _exec_limits_local_embedding_rebuild_when_memory_low(monkeypatch):
    import core.resource_guard as guard_mod
    from tools.exec import exec_run

    monkeypatch.setattr(guard_mod, "available_memory_mib", lambda: 1024)
    ctx = _tool_ctx()

    res = await exec_run(
        {"command": "printf ok # /root/.lingzhou/workspace/build_embeddings.py"},
        ctx,
    )

    assert res.skipped is False
    assert res.error is None
    assert res.metadata["resource_guard"]["available_mib"] == 1024
    assert res.metadata["resource_guard"]["limit_mib"] == 768


def test_shell_limits_local_embedding_rebuild_when_memory_low(monkeypatch):
    asyncio.run(_shell_limits_local_embedding_rebuild_when_memory_low(monkeypatch))


async def _shell_limits_local_embedding_rebuild_when_memory_low(monkeypatch):
    import core.resource_guard as guard_mod
    from tools.shell import shell_run

    monkeypatch.setattr(guard_mod, "available_memory_mib", lambda: 1024)
    ctx = _tool_ctx()

    res = await shell_run(
        {"command": "printf ok # sentence_transformers BAAI/bge-m3"},
        ctx,
    )

    assert res.skipped is False
    assert res.error is None
    assert res.metadata["resource_guard"]["reason"] == "insufficient_available_memory_for_local_embedding"
    assert res.metadata["resource_guard"]["limit_mib"] == 768


def test_memory_embed_backfill_runs_in_small_batches():
    asyncio.run(_memory_embed_backfill_runs_in_small_batches())


async def _memory_embed_backfill_runs_in_small_batches():
    from store.semantic import MemoryNode, SemanticMemory
    from tools.memory import memory_embed_backfill

    calls: list[str] = []

    def _embed(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        semantic = SemanticMemory(Path(d), decay_lambda=0.0, embed_fn=_embed)
        semantic.upsert(MemoryNode(id="n1", kind="fact", title="one", body="alpha"))
        semantic.upsert(MemoryNode(id="n2", kind="fact", title="two", body="beta"))
        ctx = _tool_ctx(semantic=semantic)

        first = await memory_embed_backfill(
            {"batch_size": 1, "sleep_seconds": 0, "model": "test-model", "max_text_chars": 16},
            ctx,
        )
        second = await memory_embed_backfill(
            {"batch_size": 10, "sleep_seconds": 0, "model": "test-model", "max_text_chars": 16},
            ctx,
        )

        assert first.error is None
        assert first.metadata["processed"] == 1
        assert second.error is None
        assert second.metadata["processed"] == 1
        assert len(calls) == 2
        assert semantic.get_unembedded(modality="text", model="test-model", limit=10) == []


def test_memory_embed_backfill_can_target_node_id():
    asyncio.run(_memory_embed_backfill_can_target_node_id())


async def _memory_embed_backfill_can_target_node_id():
    from store.semantic import MemoryNode, SemanticMemory
    from tools.memory import memory_embed_backfill

    calls: list[str] = []

    def _embed(text: str) -> list[float]:
        calls.append(text)
        return [0.0, 1.0]

    with tempfile.TemporaryDirectory() as d:
        semantic = SemanticMemory(Path(d), decay_lambda=0.0, embed_fn=_embed)
        semantic.upsert(MemoryNode(id="target", kind="fact", title="target title", body="target body"))
        semantic.upsert(MemoryNode(id="other", kind="fact", title="other title", body="other body"))
        ctx = _tool_ctx(semantic=semantic)

        result = await memory_embed_backfill(
            {"node_id": "target", "batch_size": 10, "sleep_seconds": 0, "model": "test-model", "max_text_chars": 64},
            ctx,
        )

        assert result.error is None
        assert result.metadata["processed"] == 1
        assert result.metadata["processed_ids"] == ["target"]
        assert result.metadata["node_id"] == "target"
        assert calls == ["target title target body"]
        assert ("target", "target title target body") not in semantic.get_unembedded(modality="text", model="test-model", limit=10)
        assert ("other", "other title other body") in semantic.get_unembedded(modality="text", model="test-model", limit=10)


def test_memory_embed_backfill_target_node_id_not_found():
    asyncio.run(_memory_embed_backfill_target_node_id_not_found())


async def _memory_embed_backfill_target_node_id_not_found():
    from store.semantic import SemanticMemory
    from tools.memory import memory_embed_backfill

    with tempfile.TemporaryDirectory() as d:
        semantic = SemanticMemory(Path(d), decay_lambda=0.0, embed_fn=lambda text: [1.0])
        ctx = _tool_ctx(semantic=semantic)

        result = await memory_embed_backfill(
            {"node_id": "missing", "sleep_seconds": 0, "model": "test-model"},
            ctx,
        )

        assert result.skipped is True
        assert result.error == "SemanticNodeNotFound"
        assert result.metadata["node_id"] == "missing"
        assert result.metadata["processed"] == 0


def test_process_kill():
    """process.kill 可以终止后台进程。"""
    asyncio.run(_process_kill())

async def _process_kill():
    import json

    from tools.exec import _MANAGER, exec_run, process_kill, process_poll

    _MANAGER.clear()
    ctx = _tool_ctx()
    try:
        res = await exec_run({"command": "sleep 60", "background": True, "timeout": 60}, ctx)
        sid = json.loads(res.evidence)["process_id"]

        # 确认进程存在
        poll1 = await process_poll({"session_id": sid}, ctx)
        status = json.loads(poll1.summary)
        assert status["status"] == "running"

        # kill
        kill_res = await process_kill({"session_id": sid}, ctx)
        assert kill_res.error is None
        assert "已终止" in kill_res.summary

        # 确认已终止
        poll2 = await process_poll({"session_id": sid}, ctx)
        status2 = json.loads(poll2.summary)
        assert status2["status"] == "finished"
    finally:
        _MANAGER.clear()


def test_process_list():
    """process.list 返回通过 exec 启动的进程。"""
    asyncio.run(_process_list())

async def _process_list():
    import json

    from tools.exec import _MANAGER, exec_run, process_list

    _MANAGER.clear()
    ctx = _tool_ctx()
    try:
        # 空列表
        r = await process_list({"state": "all"}, ctx)
        assert "无进程" in r.summary

        # 启动一个后台进程
        res = await exec_run({"command": "sleep 5", "background": True, "timeout": 10}, ctx)
        sid = json.loads(res.evidence)["process_id"]

        r2 = await process_list({"state": "running"}, ctx)
        assert sid in r2.summary
    finally:
        _MANAGER.clear()


def test_process_write_to_finished():
    """向已结束的进程写入应被拒绝。"""
    asyncio.run(_process_write_to_finished())

async def _process_write_to_finished():
    import time

    from tools.exec import _MANAGER, ProcessInfo, process_write

    _MANAGER.clear()
    ctx = _tool_ctx()

    sid = "finished-1"
    _MANAGER.register(ProcessInfo(
        session_id=sid,
        command="echo hi",
        started_at=time.time() - 1,
        finished=True,
        finished_at=time.time(),
        return_code=0,
        background=True,
    ))

    # 写入已结束进程
    w = await process_write({"session_id": sid, "data": "hello"}, ctx)
    assert w.skipped is True
    assert w.error == "ProcessFinished"


def test_exec_foreground_success_summary_is_exit_code_neutral():
    asyncio.run(_exec_foreground_success_summary_is_exit_code_neutral())


async def _exec_foreground_success_summary_is_exit_code_neutral():
    from tools.exec import exec_run

    ctx = _tool_ctx()
    res = await exec_run({"command": "printf 'payload-with-error-word'"}, ctx)

    assert res.error is None
    assert res.skipped is False
    assert res.summary.startswith("命令完成 (exit=0):")
    assert "payload-with-error-word" in res.summary


def test_process_poll_exposes_handle_lost_interaction_state():
    asyncio.run(_process_poll_exposes_handle_lost_interaction_state())


async def _process_poll_exposes_handle_lost_interaction_state():
    import json
    import os
    import time

    from tools.exec import _MANAGER, ProcessInfo, process_poll, process_write

    _MANAGER.clear()
    info = ProcessInfo(
        session_id="restored-1",
        command="python -i",
        pid=os.getpid(),
        started_at=time.time() - 5,
        background=True,
        restored=True,
        handle_lost=True,
    )
    _MANAGER.register(info)

    ctx = _tool_ctx()
    poll = await process_poll({"session_id": "restored-1"}, ctx)
    status = json.loads(poll.summary)
    assert status["restored"] is True
    assert status["handle_lost"] is True
    assert status["interaction_available"] is False

    write = await process_write({"session_id": "restored-1", "data": "hello"}, ctx)
    assert write.error == "ProcessHandleLost"
    assert write.metadata["handle_lost"] is True


def test_file_edit_json_string_edits():
    """file.edit 支持 edits 为 JSON 字符串。"""
    asyncio.run(_file_edit_json_string_edits())

async def _file_edit_json_string_edits():
    import json as _json

    from tools.file import file_edit, file_read, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "jsontest.py"
        await file_write({"path": str(fpath), "content": "v = 1\n"}, ctx)

        edits_str = _json.dumps([{"oldText": "v = 1", "newText": "v = 2"}])
        res = await file_edit({"path": str(fpath), "edits": edits_str}, ctx)
        assert res.error is None

        content = await file_read({"path": str(fpath)}, ctx)
        assert content.summary == "v = 2\n"


def test_file_edit_resolves_workspace_logical_path_for_existing_file():
    asyncio.run(_file_edit_resolves_workspace_logical_path_for_existing_file())


async def _file_edit_resolves_workspace_logical_path_for_existing_file():
    from tools.file import file_edit

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        workspace = root / ".lingzhou" / "workspace"
        workspace.mkdir(parents=True)
        target = workspace / "MEMORY.md"
        target.write_text("hello\nworld\n", encoding="utf-8")

        wrong_path = root / "root" / "lingzhou" / "MEMORY.md"
        ctx = _tool_ctx(workspace_dir=str(workspace))

        res = await file_edit(
            {"path": str(wrong_path), "edits": [{"oldText": "world", "newText": "dad"}]},
            ctx,
        )

        assert res.error is None
        assert target.read_text(encoding="utf-8") == "hello\ndad\n"
        assert not wrong_path.exists()


def test_file_write_resolves_workspace_logical_path_for_existing_file():
    asyncio.run(_file_write_resolves_workspace_logical_path_for_existing_file())


async def _file_write_resolves_workspace_logical_path_for_existing_file():
    from tools.file import file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        workspace = root / ".lingzhou" / "workspace"
        workspace.mkdir(parents=True)
        target = workspace / "MEMORY.md"
        target.write_text("old\n", encoding="utf-8")

        wrong_path = root / "root" / "lingzhou" / "MEMORY.md"
        ctx = _tool_ctx(workspace_dir=str(workspace))

        res = await file_write({"path": str(wrong_path), "content": "new\n"}, ctx)

        assert res.error is None
        assert target.read_text(encoding="utf-8") == "new\n"
        assert not wrong_path.exists()


def test_file_read_max_chars():
    """file.read max_chars 参数正确截断。"""
    asyncio.run(_file_read_max_chars())


def test_file_read_missing_path_returns_recovery_state():
    asyncio.run(_file_read_missing_path_returns_recovery_state())


async def _file_read_missing_path_returns_recovery_state():
    from tools.file import file_read

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        existing_parent = root / "core"
        existing_parent.mkdir()
        candidate = existing_parent / "assemble_context.py"
        candidate.write_text("# candidate\n", encoding="utf-8")
        ctx = _tool_ctx(workspace_dir=d)

        result = await file_read({"path": str(existing_parent / "assembler.py")}, ctx)

        assert result.error == "FileNotFound"
        assert result.state_delta["tool_input_invalid"] is True
        assert result.state_delta["recovery_reason"] == "path_not_found"
        assert result.state_delta["nearest_existing_parent"] == str(existing_parent)
        assert result.state_delta["candidate_paths"] == [str(candidate)]
        assert result.state_delta["suggested_tools"] == ["file.list", "shell.run"]
        assert "file.list" in result.state_delta["recovery_next_step"]


def test_file_list_missing_path_returns_recovery_state():
    asyncio.run(_file_list_missing_path_returns_recovery_state())


async def _file_list_missing_path_returns_recovery_state():
    from tools.file import file_list

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)

        result = await file_list({"path": str(root / "missing" / "child")}, ctx)

        assert result.error == "FileNotFound"
        assert result.state_delta["tool_input_invalid"] is True
        assert result.state_delta["expected"] == "directory"
        assert result.state_delta["nearest_existing_parent"] == str(root)
        assert "确认真实路径" in result.summary


async def _file_read_max_chars():
    from tools.file import file_read, file_write

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ctx = _tool_ctx(workspace_dir=d)
        fpath = root / "big.txt"
        await file_write({"path": str(fpath), "content": "abcdefghij" * 100}, ctx)  # 1000 chars

        r = await file_read({"path": str(fpath), "max_chars": 20}, ctx)
        assert len(r.summary) == 20
        assert r.state_delta["has_more"] is True
        assert r.state_delta["truncated"] is True
        assert r.state_delta["truncation_reason"] == "max_chars"
        assert r.state_delta["next_start"] == 20
        assert r.state_delta["next_params"] == {"path": str(fpath), "start": 20, "max_chars": 20}
        assert "继续读取同一文件剩余内容" in r.state_delta["recovery_next_step"]
        assert r.metadata["truncated"] is True
        assert r.metadata["next_params"] == {"path": str(fpath), "start": 20, "max_chars": 20}

        ranged = await file_read({"path": str(fpath), "start": 10, "end": 80, "max_chars": 15}, ctx)
        assert len(ranged.summary) == 15
        assert ranged.state_delta["next_start"] == 25
        assert ranged.state_delta["next_params"] == {"path": str(fpath), "start": 25, "end": 80, "max_chars": 15}

        lines = root / "lines.txt"
        await file_write({"path": str(lines), "content": "\n".join(f"line-{i}-" + ("x" * 20) for i in range(1, 8))}, ctx)
        line_window = await file_read({"path": str(lines), "offset": 2, "limit": 3, "max_chars": 25}, ctx)
        assert len(line_window.summary) == 25
        assert line_window.state_delta["truncated"] is True
        assert line_window.state_delta["next_params"]["offset"] == 2
        assert line_window.state_delta["next_params"]["limit"] == 3
        assert line_window.state_delta["next_params"]["max_chars"] > 25
        assert "扩大 max_chars" in line_window.state_delta["recovery_next_step"]
