"""core.loop.task.parallel — 任务并行执行器（delegate_tasks → asyncio.gather）。

主 tick（reasoner）通过 JudgmentOutput.delegate_tasks 将独立子目标委派为
真实 Task（经代谢器官持久化）。run_tasks_parallel() 用 asyncio.gather
并发执行多个 Task，每个 Task 独立调用 LLM（reader tier），结果经代谢器官写回。
全部完成后主 tick（reasoner）做统一审查决策。

架构：
  Main Tick (reasoner)
    └── decide() → delegate_tasks: [{id, goal, tools, max_rounds, params}]
          ↓ run_tasks_parallel()
    ┌────────────────────────────────────────────────────────────┐
    │  Task A  (metabolic create_task → Task)                    │
    │    decide_continue(active_task=A, reader) × max_rounds     │  asyncio.gather
    │    → metabolic update_task_status / mark_task_waiting      │
    │  Task B  ...                                               │
    └────────────────────────────────────────────────────────────┘
          ↓ history_entries → main tick decide_continue(reasoner)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any

from core.metabolic.lifecycle_utils import _decision_basis_from_parts as _decision_basis
from core.metabolic import (
    amend_task,
    create_task,
    mark_task_waiting,
    resume_task,
    update_task_data,
    update_task_result,
    update_task_status,
)
from store.task import build_task_similarity_query

_log = logging.getLogger("lingzhou.loop.task.parallel")

if TYPE_CHECKING:
    from core.loop.runtime import CognitionLoop
    from store.task import Task
    from tools.registry import ToolContext


class _ScopedTaskStore:
    """显式暴露并行任务所需的 TaskStore 表面，并固定 get_active()。

    并行执行时多个 _run_one_task 协程共享同一个 ctx，
    而 _dispatch_act 内部调用 ctx.task_store.get_active() 确定当前活跃任务。
    注入 pin 后确保每个子任务的 dispatch 只操作自己的行，
    避免 run 记录、update_task_result、record_failure 写入错误任务。
    新增 TaskStore 方法时，只有并行路径确实需要时才应显式加入这里。
    """

    def __init__(self, inner: Any, pinned: Task) -> None:
        self._inner = inner
        self._pinned = pinned

    async def _call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, method)(*args, **kwargs)

    async def get_active(self) -> Task:
        return self._pinned

    async def get_task_by_id(self, task_id: int) -> Task | None:
        return await self._call("get_task_by_id", task_id)

    async def add_task(self, title: str, goal: str = "", priority: str = "normal", **kwargs: Any) -> int:
        return await self._call("add_task", title, goal, priority=priority, **kwargs)

    async def list_tasks(self, *, status: str | None = None, limit: int = 20) -> list[Any]:
        return await self._call("list_tasks", status=status, limit=limit)

    async def list_runnable_tasks(self, limit: int = 20) -> list[Any]:
        return await self._call("list_runnable_tasks", limit)

    async def list_open_tasks(
        self,
        limit: int = 50,
        *,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> list[Any]:
        return await self._call("list_open_tasks", limit=limit, statuses=statuses)

    async def find_similar_open_tasks(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.45,
        exclude_task_ids: list[int] | tuple[int, ...] | set[int] | None = None,
        allowed_sources: list[str] | tuple[str, ...] | set[str] | None = None,
        excluded_sources: list[str] | tuple[str, ...] | set[str] | None = None,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> list[Any]:
        return await self._call(
            "find_similar_open_tasks",
            query,
            limit=limit,
            min_score=min_score,
            exclude_task_ids=exclude_task_ids,
            allowed_sources=allowed_sources,
            excluded_sources=excluded_sources,
            statuses=statuses,
        )

    async def update_status(
        self,
        task_id: int,
        status: str,
        next_step: str | None = None,
        *,
        current_step: str | None = None,
        model_tier: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "current_step": current_step,
            "model_tier": model_tier,
        }
        if result_json is not None:
            kwargs["result_json"] = result_json
        try:
            await update_task_status(
                self._inner,
                task_id,
                status=status,
                source="loop/task/parallel/update_status",
                next_step=next_step,
                current_step=current_step,
                model_tier=model_tier,
                result_json=result_json,
            )
            return
        except RuntimeError:
            pass
        await self._call("update_status", task_id, status, next_step, **kwargs)

    async def mark_waiting(self, task_id: int, **kwargs: Any) -> None:
        try:
            await mark_task_waiting(
                self._inner,
                task_id,
                wait_kind=kwargs.get("wait_kind") or "",
                source="loop/task/parallel/mark_waiting",
                wait_key=kwargs.get("wait_key", ""),
                wait_json=kwargs.get("wait_json"),
                current_step=kwargs.get("current_step"),
                next_step=kwargs.get("next_step"),
                result_json=kwargs.get("result_json"),
            )
            return
        except RuntimeError:
            pass
        await self._call("mark_waiting", task_id, **kwargs)

    async def resume_task(self, task_id: int, status: str = "in_progress", **kwargs: Any) -> None:
        try:
            await resume_task(
                self._inner,
                task_id,
                source="loop/task/parallel/resume_task",
                status=status,
                current_step=kwargs.get("current_step"),
                next_step=kwargs.get("next_step"),
                result_json=kwargs.get("result_json"),
            )
            return
        except RuntimeError:
            pass
        await self._call("resume_task", task_id, status=status, **kwargs)

    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None:
        try:
            await update_task_data(
                self._inner,
                task_id,
                extra_dict,
                source="loop/task/parallel/update_task_data",
            )
            return
        except RuntimeError:
            pass
        await self._call("update_task_data", task_id, extra_dict)

    async def amend_task(self, task_id: int, **kwargs: Any) -> bool:
        try:
            return await amend_task(
                self._inner,
                task_id,
                source="loop/task/parallel/amend_task",
                **kwargs,
            )
        except RuntimeError:
            pass
        return await self._call("amend_task", task_id, **kwargs)

    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None:
        try:
            await update_task_result(
                self._inner,
                task_id,
                result_json,
                source="loop/task/parallel/update_task_result",
            )
            return
        except RuntimeError:
            pass
        await self._call("update_task_result", task_id, result_json)

    async def list_runs(self, **kwargs: Any) -> list[Any]:
        return await self._call("list_runs", **kwargs)

    async def add_run(self, **kwargs: Any) -> int:
        return await self._call("add_run", **kwargs)

    async def update_run(self, run_id: int, **kwargs: Any) -> None:
        await self._call("update_run", run_id, **kwargs)

    async def add_meta_reflection(self, **kwargs: Any) -> int:
        return await self._call("add_meta_reflection", **kwargs)

    async def list_failures(self, limit: int = 20) -> list[Any]:
        return await self._call("list_failures", limit=limit)

    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Any]:
        return await self._call("list_failures_for_task", task_id, limit=limit)

    async def record_failure(self, **kwargs: Any) -> int:
        return await self._call("record_failure", **kwargs)

    async def dismiss_failure(self, failure_id: int) -> None:
        await self._call("dismiss_failure", failure_id)

    async def ledger_append(self, *args: Any, **kwargs: Any) -> None:
        await self._call("ledger_append", *args, **kwargs)

    async def get_fact(self, key: str) -> tuple[str, bool]:
        return await self._call("get_fact", key)

    async def set_fact(self, key: str, value: str, *, scope: str = "general") -> None:
        await self._call("set_fact", key, value, scope=scope)

    async def delete_fact(self, key: str) -> None:
        await self._call("delete_fact", key)

    async def list_facts(self, prefix: str = "", limit: int = 20) -> list[tuple[str, str]]:
        return await self._call("list_facts", prefix=prefix, limit=limit)

    async def add_signal(
        self,
        title: str,
        run_at: str,
        repeat_secs: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> int:
        return await self._call("add_signal", title, run_at, repeat_secs, payload)

    async def list_signals(self, limit: int = 50, include_done: bool = False) -> list[dict[str, Any]]:
        return await self._call("list_signals", limit=limit, include_done=include_done)

    async def get_signal(self, signal_id: int) -> dict[str, Any] | None:
        return await self._call("get_signal", signal_id)

    async def ack_signal(self, signal_id: int) -> None:
        await self._call("ack_signal", signal_id)

    async def cancel_signal(self, signal_id: int) -> None:
        await self._call("cancel_signal", signal_id)


def _parallel_history_entry(
    *,
    spec_id: str,
    task: Task,
    result: str,
    summary: str,
    error: str = "",
    status: str = "ok",
    goal: str | None = None,
) -> dict[str, Any]:
    return {
        "tool": f"task.parallel.{spec_id}",
        "params": {"goal": goal or task.goal, "task_id": task.id},
        "result": result,
        "summary": summary,
        "error": error,
        "status": status,
    }


async def _run_one_task(
    task: Task,
    spec: dict[str, Any],
    ctx: ToolContext,
    loop: CognitionLoop,
) -> dict[str, Any]:
    """运行单个 Task 的 judgment 循环（reader tier），结果写回 task_store。

    返回 tool_history 格式的 entry，供主 tick decide_continue() 审查。
    """
    tools: list[str] = spec.get("tools") or []
    max_rounds: int = int(spec.get("max_rounds") or 10)
    spec_id: str = str(spec.get("id") or task.id)

    # 并发安全：为本子任务创建隔离的 ToolContext，确保 _dispatch_act 内的
    # get_active() 始终返回本子任务，而不是兄弟任务或父任务。
    scoped_ctx = dataclasses.replace(ctx, task_store=_ScopedTaskStore(ctx.task_store, task))

    # 首条注入任务目标（让 decide_continue 知道当前目标）
    init_parts = [f"[任务 {spec_id}] 目标: {task.goal}"]
    if spec.get("params"):
        init_parts.append(f"上下文: {json.dumps(spec['params'], ensure_ascii=False)}")
    if tools:
        init_parts.append(f"可用工具限定: {tools}")

    tool_history: list[dict[str, Any]] = [{
        "tool": "task.init",
        "params": {},
        "result": "\n".join(init_parts),
        "summary": f"任务目标: {task.goal}",
        "error": "",
        "status": "ok",
    }]
    final_rationale = ""
    terminal_decision = "wait"
    error = ""

    for round_i in range(max_rounds):
        try:
            output = await loop._judgment.decide_continue(
                tool_history=tool_history,
                user_message=task.goal if round_i == 0 else "",
                active_task=task,
            )
        except Exception as exc:
            error = str(exc)
            _log.warning("[task_parallel:%s] decide_continue 失败(round=%d): %s", spec_id, round_i, exc)
            break

        terminal_decision = output.decision or terminal_decision
        final_rationale = output.rationale or ""

        if output.decision != "act":
            _log.info("[task_parallel:%s] 结束 round=%d decision=%s", spec_id, round_i, output.decision)
            break

        tool_name = output.chosen_action_id or ""
        if tools and tool_name not in tools:
            _log.info("[task_parallel:%s] 工具 %r 不在白名单，停止", spec_id, tool_name)
            break

        result = await loop._execution.dispatch(output, scoped_ctx)
        tool_history.append({
            "tool": tool_name,
            "params": output.params,
            "result": result.summary or "",
            "summary": result.summary or "",
            "error": result.error or "",
            "status": "error" if result.error else "ok",
        })
        _log.info("[task_parallel:%s] round=%d tool=%s ok=%s", spec_id, round_i, tool_name, not result.error)

        if result.error:
            error = result.error
            break

    # 将结果写回 task_store（持久化）
    ok_steps = sum(1 for h in tool_history[1:] if not h.get("error"))
    result_data = {
        "summary": final_rationale,
        "error": error,
        "rounds": len(tool_history) - 1,
        "ok_steps": ok_steps,
        "terminal_decision": terminal_decision,
    }
    if error:
        await update_task_status(
            loop,
            task.id,
            status="failed",
            result_json=result_data,
            source="loop/task.parallel",
            decision_basis=_decision_basis(error, final_rationale, "parallel child task failed"),
        )
    elif terminal_decision in {"wait", "pause"}:
        wait_key = str(getattr(task, "parent_task_id", "") or "")
        next_step = (str(getattr(task, "next_step", "") or "").strip() or (final_rationale or "").strip() or None)
        await mark_task_waiting(
            loop,
            task.id,
            wait_kind="task",
            wait_key=wait_key,
            wait_json={
                "wait_kind": "task",
                "wait_key": wait_key,
                "terminal_decision": terminal_decision,
            },
            next_step=next_step,
            result_json=result_data,
            source="loop/task.parallel",
            decision_basis=_decision_basis(
                final_rationale,
                f"parallel child task ended with {terminal_decision}",
            ),
        )
    else:
        await update_task_status(
            loop,
            task.id,
            status="done",
            result_json=result_data,
            source="loop/task.parallel",
            decision_basis=_decision_basis(final_rationale, "parallel child task completed"),
        )

    # 构建返回给主 tick 的 history entry
    steps_text = "\n".join(
        f"  [{h['tool']}] {h.get('result', '')}"
        for h in tool_history[1:]
    )
    result_text = (
        f"目标: {task.goal}\n"
        f"最终决策: {terminal_decision}\n"
        f"结论: {final_rationale or '(无结论)'}\n"
        f"步骤:\n{steps_text or '  (无步骤)'}"
    )
    if error:
        result_text += f"\n错误: {error}"

    summary_prefix = f"[{spec_id}/task:{task.id}]"
    if error:
        summary = f"{summary_prefix} error: {(error or final_rationale or '(无结论)')}"
    elif terminal_decision in {"wait", "pause"}:
        summary = f"{summary_prefix} {terminal_decision}: {(final_rationale or '(无结论)')}"
    else:
        summary = f"{summary_prefix} {ok_steps} 步完成: {(final_rationale or '(无结论)')}"

    entry_status = "error" if error else (terminal_decision if terminal_decision in {"wait", "pause"} else "ok")
    return _parallel_history_entry(
        spec_id=spec_id,
        task=task,
        result=result_text,
        summary=summary,
        error=error,
        status=entry_status,
    )


async def run_tasks_parallel(
    specs: list[dict[str, Any]],
    ctx: ToolContext,
    loop: CognitionLoop,
    parent_task_id: int | None = None,
) -> list[dict[str, Any]]:
    """创建真实 Task 并并行执行，返回 tool_history 格式的 entry 列表。

    每个 spec 对应一个新 Task（持久化到 task_store），
    所有 Task 通过 asyncio.gather 并发执行（各自用 reader tier 调 LLM），
    结果写入 task.result_json 后汇总返回，供主 tick gate decision 审查。
    """
    valid_specs = [s for s in specs if isinstance(s, dict) and s.get("id") and s.get("goal")]
    if not valid_specs:
        return []

    _log.info("[task_parallel] 并行启动 %d 个任务: %s", len(valid_specs), [s["id"] for s in valid_specs])

    parent_task = await loop._task_store.get_task_by_id(parent_task_id) if parent_task_id else None
    parent_source = str(getattr(parent_task, "source", "") or "").strip()
    allowed_sources = ("self_drive",) if parent_source == "self_drive" else None
    excluded_sources = None if parent_source == "self_drive" else ("self_drive",)

    def _reused_entry(spec: dict[str, Any], task: Task, score: float) -> dict[str, Any]:
        spec_id = str(spec.get("id") or task.id)
        result_text = (
            f"目标: {spec.get('goal') or task.goal}\n"
            f"复用已有任务: task#{task.id} [{task.status}] {task.title}\n"
            f"相似度: {score:.2f}\n"
            f"下一步: {task.next_step or '（未指定）'}"
        )
        return _parallel_history_entry(
            spec_id=spec_id,
            task=task,
            result=result_text,
            summary=f"[{spec_id}/task:{task.id}] reused existing {task.status}: {task.title}",
            goal=str(spec.get("goal") or task.goal),
        )

    # 先顺序创建所有 Task（写 DB 不适合并发）
    scheduled: list[dict[str, Any] | tuple[Task, dict]] = []
    finder: Any = getattr(loop._task_store, "find_similar_open_tasks", None)
    for spec in valid_specs:
        if finder is not None:
            similar_tasks = await finder(
                build_task_similarity_query(spec.get("goal")),
                limit=1,
                min_score=loop._cfg.thresholds.task_duplicate_reuse_score,
                exclude_task_ids=[parent_task_id] if parent_task_id else None,
                allowed_sources=allowed_sources,
                excluded_sources=excluded_sources,
            )
            if similar_tasks:
                existing, score = similar_tasks[0]
                scheduled.append(_reused_entry(spec, existing, score))
                continue
        title = f"[并行:{spec['id']}] {spec['goal']}"
        task_id = await create_task(
            loop,
            proposal_source="loop/task.parallel",
            title=title,
            goal=str(spec["goal"]),
            priority="normal",
            source="internal",
            parent_task_id=str(parent_task_id) if parent_task_id else "",
            next_step=str(spec["goal"]),
            decision_basis=_decision_basis(
                "parallel decomposition scheduled child task for",
                spec.get("goal"),
            ),
        )
        task = await loop._task_store.get_task_by_id(task_id)
        if task:
            scheduled.append((task, spec))

    # 并发执行所有任务
    entries: list[dict[str, Any] | None] = [None] * len(scheduled)
    pending_slots: list[int] = []
    pending_coros: list[Any] = []
    for index, item in enumerate(scheduled):
        if isinstance(item, dict):
            entries[index] = item
            continue
        task, spec = item
        pending_slots.append(index)
        pending_coros.append(_run_one_task(task, spec, ctx, loop))

    if pending_coros:
        results = await asyncio.gather(*pending_coros)
        for slot, result in zip(pending_slots, results, strict=False):
            entries[slot] = result

    return [entry for entry in entries if entry is not None]
