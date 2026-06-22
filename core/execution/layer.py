"""core/execution/layer.py — 执行层入口 (ExecutionLayer)。

职责：
- 接收 JudgmentOutput，dispatch 到具体工具
- 处理 act / pause / wait 三种决策
- 失败时写入 failures 表（绑定当前任务 ID，P2-B 原则）
- 对稳定重复失败的确定性动作做持久降噪（durable failure sensing）
- 返回 ToolResult 给 loop 层整合
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from core.execution.helpers import (
    _classify_durable_failure,
    _failure_fact_key,
    _load_durable_failure_policy,
    _normalize_tool_result_text_fields,
    _planned_run_task_id,
    _record_run_started,
    _resolve_execution_active_task,
    _resolved_run_task_id,
    _run_status_from_result,
    _tool_result_log_fields,
    _worker_limit_for_type,
    _worker_log_fields,
    action_key_param,
    build_meta_reflection,
    finalize_run,
    record_meta_reflection_memory,
    record_run_outcome_memory,
)
from core.execution.run_profile import (
    ExecutionProfile,
    RUN_TYPE_TOOL_CHAIN,
    WORKER_TOOL_CHAIN,
    resolve_execution_dispatch,
)
from core.execution.routing import resolve_run_type_routing
from core.execution.workers import WorkerLayer
from core.judgment.decision.helpers import _decision_basis
from core.log_fields import execution_scope_fields
from core.metabolic import add_run, submit_fact
from tools.registry import ToolContext, ToolResult, tool_metadata

_log = logging.getLogger("lingzhou.execution")
_TERMINAL_TASK_ACTIONS = frozenset({"task.complete", "task.fail"})
_DURABLE_FAILURE_SUMMARY_MAX_CHARS = 1200


@dataclass(frozen=True)
class _ExecutionDispatch:
    run_id: int | None
    run_task_id: int
    run_type: str
    tool_name: str
    worker_type: str
    task_tier: str
    effective_tier: str

    def scope_fields(self, *, dispatch_ms: int | None = None) -> str:
        return execution_scope_fields(
            run_id=self.run_id,
            task_id=self.run_task_id,
            tool=self.tool_name,
            worker=self.worker_type,
            tier=self.effective_tier or self.task_tier or None,
            dispatch_ms=dispatch_ms,
        )

    def with_run_id(self, run_id: int | None) -> "_ExecutionDispatch":
        return replace(self, run_id=run_id)

    def with_profile(self, profile: ExecutionProfile) -> "_ExecutionDispatch":
        return replace(
            self,
            run_type=profile.run_type,
            worker_type=profile.worker_type,
            effective_tier=profile.model_tier,
        )

    def stamp_metadata(self, tool_result: ToolResult) -> ToolResult:
        tool_result.metadata.setdefault("tool_name", self.tool_name)
        tool_result.metadata.setdefault("worker_type", self.worker_type)
        if self.run_id is not None:
            tool_result.metadata.setdefault("run_id", self.run_id)
        return tool_result


def _durable_failure_summary_fields(summary: str) -> dict[str, Any]:
    text = str(summary or "")
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    if len(text) <= _DURABLE_FAILURE_SUMMARY_MAX_CHARS:
        preview = text
    else:
        marker = f"\n...[durable failure summary truncated chars={len(text)} sha256={digest}]...\n"
        budget = max(0, _DURABLE_FAILURE_SUMMARY_MAX_CHARS - len(marker))
        head = max(160, budget // 2)
        tail = max(0, budget - head)
        preview = text[:head] + marker + (text[-tail:] if tail else "")
    return {
        "last_summary_preview": preview,
        "last_summary_chars": len(text),
        "last_summary_sha256": digest,
    }


if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentOutput
    from tools.registry import ToolRegistry


__all__ = [
    "ExecutionLayer",
    "_load_durable_failure_policy",
    "_run_status_from_result",
    "_tool_result_log_fields",
    "action_key_param",
    "build_meta_reflection",
    "record_meta_reflection_memory",
    "record_run_outcome_memory",
]


class ExecutionLayer:
    def __init__(self, registry: ToolRegistry, cfg: Config) -> None:
        self._registry = registry
        self._cfg = cfg
        self._workers = WorkerLayer(cfg)
        self._run_type_routing: dict[str, str] = resolve_run_type_routing(cfg)

    async def dispatch(self, action: JudgmentOutput, ctx: ToolContext) -> ToolResult:
        """根据 decision 类型分发执行。"""
        match action.decision:
            case "wait":
                return ToolResult(
                    summary=f"wait: {action.rationale}",
                    skipped=True,
                    kind="wait",
                    priority=0.3,
                )
            case "pause":
                from memory.working import WMItem

                ctx.wm.add(WMItem(
                    kind="caution",
                    content=f"pause: {action.rationale}",
                    priority=0.9,
                ))
                return ToolResult(
                    summary=f"pause: {action.rationale}",
                    skipped=True,
                    kind="pause",
                    priority=0.9,
                )
            case "act":
                if action.parallel_actions:
                    return await self._dispatch_parallel(action, ctx)
                return await self._dispatch_act(action, ctx)
            case _:
                return ToolResult(
                    summary=f"未知决策类型: {action.decision!r}",
                    skipped=True,
                    kind="error",
                )

    async def _dispatch_parallel(self, action: JudgmentOutput, ctx: ToolContext) -> ToolResult:
        """gather 并行执行 parallel_actions 列表中的多个工具，合并结果返回。"""
        import asyncio

        from core.judgment import JudgmentOutput as _JO

        sub_actions = [
            _JO(
                decision="act",
                chosen_action_id=item["action_id"],
                params=dict(item.get("params") or {}),
                rationale=action.rationale,
            )
            for item in action.parallel_actions
            if isinstance(item, dict) and isinstance(item.get("action_id"), str) and item["action_id"]
        ]
        if not sub_actions:
            return ToolResult(summary="parallel_actions 为空，退化为 wait", skipped=True, kind="wait")

        terminal_actions = [a for a in sub_actions if a.chosen_action_id in _TERMINAL_TASK_ACTIONS]
        if terminal_actions and len(sub_actions) > 1:
            executable_actions = [a for a in sub_actions if a.chosen_action_id not in _TERMINAL_TASK_ACTIONS]
            deferred_names = [str(a.chosen_action_id or "") for a in terminal_actions]
            if not executable_actions:
                return ToolResult(
                    summary="终结任务动作不能并发执行，请只保留一个 task.complete 或 task.fail。",
                    skipped=True,
                    kind="execute_result",
                    error="TerminalActionParallelInvalid",
                    state_delta={
                        "deferred_terminal_actions": deferred_names,
                        "recovery_next_step": "将 task.complete/task.fail 作为单独动作重试，不能放在 parallel_actions 同批执行。",
                    },
                    metadata=tool_metadata(
                        "exec.parallel",
                        "exec.parallel terminal_actions_invalid",
                        parallel_count=len(sub_actions),
                        deferred_terminal_actions=deferred_names,
                    ),
                )
            _log.info(
                "[exec.parallel] deferring terminal task actions until evidence batch completes: %s",
                deferred_names,
            )
            sub_actions = executable_actions

        _log.info(
            "[exec.parallel] launching %d tools: %s",
            len(sub_actions), [a.chosen_action_id for a in sub_actions],
        )
        results: list[ToolResult] = list(await asyncio.gather(
            *[self._dispatch_act(a, ctx) for a in sub_actions]
        ))
        merged_summary = "\n".join(
            f"[{a.chosen_action_id}] {r.summary}"
            for a, r in zip(sub_actions, results, strict=False)
            if r.summary
        )
        errors = [r.error for r in results if r.error]
        combined_error = "; ".join(errors) if errors else None
        if terminal_actions:
            deferred_names = [str(a.chosen_action_id or "") for a in terminal_actions]
            deferred_line = (
                f"[{', '.join(deferred_names)}] 已延后：终结任务前需要先让本轮证据/工作台写入落地，"
                "下一轮再单独执行完成动作。"
            )
            merged_summary = "\n".join(part for part in (merged_summary, deferred_line) if part)
            state_delta = {
                "deferred_terminal_actions": deferred_names,
                "recovery_next_step": "先根据本轮工具结果更新任务证据；若 completion_checks 已满足，再单独执行 task.complete。",
            }
        else:
            state_delta = {}
        return ToolResult(
            summary=merged_summary,
            error=combined_error,
            kind="execute_result",
            state_delta=state_delta,
            priority=max((r.priority for r in results), default=0.9),
            metadata=tool_metadata(
                "exec.parallel",
                f"exec.parallel count={len(sub_actions)} errors={len(errors)}",
                parallel_count=len(sub_actions),
                errors=errors,
                deferred_terminal_actions=[str(a.chosen_action_id or "") for a in terminal_actions],
            ),
        )

    def _make_dispatch_ctx(self, action: JudgmentOutput, active_task: Any) -> _ExecutionDispatch:
        active_task_id = active_task.id if active_task is not None else 0
        return _ExecutionDispatch(
            run_id=None,
            run_task_id=_planned_run_task_id(action, active_task_id),
            run_type=RUN_TYPE_TOOL_CHAIN,
            tool_name=action.chosen_action_id or "",
            worker_type=WORKER_TOOL_CHAIN,
            task_tier=(active_task.model_tier or "").strip() if active_task is not None else "",
            effective_tier="",
        )

    def _resolve_execution_profile(
        self,
        dispatch_ctx: _ExecutionDispatch,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ExecutionProfile:
        effective_registry = ctx.registry or self._registry
        return resolve_execution_dispatch(
            task_tier=dispatch_ctx.task_tier,
            tool_name=dispatch_ctx.tool_name,
            run_type_routing=self._run_type_routing,
            params=action.params,
            registry=effective_registry,
        )

    async def _start_run_record(
        self,
        dispatch_ctx: _ExecutionDispatch,
        action: JudgmentOutput,
        ctx: ToolContext,
        *,
        decision_basis: str,
    ) -> _ExecutionDispatch:
        if ctx.task_store is None:
            return dispatch_ctx

        run_id = await add_run(
            ctx,
            task_id=dispatch_ctx.run_task_id,
            run_type=dispatch_ctx.run_type,
            worker_type=dispatch_ctx.worker_type,
            status="running",
            input_json={
                "decision": action.decision,
                "tool": dispatch_ctx.tool_name,
                "params": action.params or {},
            },
            tool_name=dispatch_ctx.tool_name,
            model_tier=dispatch_ctx.effective_tier,
            source="execution/layer/_dispatch_act",
            decision_basis=decision_basis,
        )
        dispatch_ctx = dispatch_ctx.with_run_id(run_id)
        if run_id is not None:
            _record_run_started(
                ctx,
                run_id=dispatch_ctx.run_id,
                task_id=dispatch_ctx.run_task_id,
                tool_name=dispatch_ctx.tool_name,
                run_type=dispatch_ctx.run_type,
                worker_type=dispatch_ctx.worker_type,
                model_tier=dispatch_ctx.effective_tier,
            )
            _log.info(
                "[run-start] %s limit=%s",
                dispatch_ctx.scope_fields(),
                _worker_limit_for_type(self._cfg, dispatch_ctx.worker_type),
            )
        return dispatch_ctx

    async def _preflight_tool_result(
        self,
        action: JudgmentOutput,
        dispatch_ctx: _ExecutionDispatch,
        ctx: ToolContext,
        active_task: Any,
    ) -> ToolResult | None:
        failure_key = _failure_fact_key(action)
        if ctx.task_store is not None:
            raw_mute, mute_found = await ctx.task_store.get_fact(failure_key)
            if mute_found and raw_mute:
                try:
                    mute_data = json.loads(raw_mute)
                    muted_until = float(mute_data.get("muted_until") or 0)
                    if muted_until > time.time():
                        _log.info(
                            "[exec-mute] %s reason=%s count=%s muted_until=%s",
                            dispatch_ctx.scope_fields(),
                            mute_data.get("reason") or "-",
                            mute_data.get("count") or 0,
                            int(muted_until),
                        )
                        return dispatch_ctx.stamp_metadata(ToolResult(
                            summary=(
                                f"跳过已知稳定失败动作 {dispatch_ctx.tool_name!r}："
                                f" {mute_data.get('last_summary_preview') or mute_data.get('last_summary', '')}"
                            ),
                            error="KnownStableFailure",
                            skipped=True,
                            kind="error",
                        ))
                except Exception:
                    pass

        is_mutation = not dispatch_ctx.tool_name.startswith("task.") and dispatch_ctx.tool_name not in {
            "memory.get_fact", "file.read", "file.list", "file.search",
        }
        if is_mutation and active_task is not None and ctx.task_store is not None:
            plan = active_task.extras.get("plan") if active_task.extras else None
            if plan:
                in_progress_steps = [s.get("step", "") for s in plan if s.get("status") == "in_progress"]
                cur_step = (active_task.current_step or "").strip()
                if in_progress_steps and not cur_step:
                    step_name = in_progress_steps[0]
                    _log.info(
                        "[exec-gate] %s blocked_step=%s",
                        dispatch_ctx.scope_fields(),
                        step_name,
                    )
                    return dispatch_ctx.stamp_metadata(ToolResult(
                        summary=f"当前步骤未对齐，请先完成「{step_name}」再执行变更操作",
                        error="PlanStepMismatch",
                        skipped=True,
                        kind="error",
                    ))

        return None

    async def _finalize_and_return(
        self,
        dispatch_ctx: _ExecutionDispatch,
        ctx: ToolContext,
        result: ToolResult,
    ) -> ToolResult:
        await finalize_run(
            cfg=self._cfg,
            run_id=dispatch_ctx.run_id,
            result=result,
            ctx=ctx,
            active_task_id=dispatch_ctx.run_task_id or None,
        )
        return result

    async def _sync_durable_failure(
        self,
        action: JudgmentOutput,
        dispatch_ctx: _ExecutionDispatch,
        ctx: ToolContext,
        result: ToolResult,
        *,
        durable_threshold: int,
        durable_ttl_sec: int,
    ) -> None:
        if ctx.task_store is None:
            return

        failure_key = _failure_fact_key(action)
        if result.error:
            raw, found = await ctx.task_store.get_fact(failure_key)
            reason = _classify_durable_failure(result)
            if not reason:
                return
            prev: dict[str, Any] = {}
            if found:
                try:
                    prev = json.loads(raw)
                except Exception:
                    prev = {}
            count = int(prev.get("count") or 0) + 1 if prev.get("reason") == reason else 1
            payload = {
                "tool": dispatch_ctx.tool_name,
                "key": action_key_param(action.params),
                "reason": reason,
                "count": count,
                "last_seen": time.time(),
                "muted_until": time.time() + durable_ttl_sec if count >= durable_threshold else 0,
                "policy_threshold": durable_threshold,
                "policy_ttl_sec": durable_ttl_sec,
            }
            payload.update(_durable_failure_summary_fields(result.summary))
            await submit_fact(
                ctx,
                key=failure_key,
                value=json.dumps(payload, ensure_ascii=False),
                scope="system",
                source="execution/failure_track",
                decision_basis=_decision_basis(
                    action,
                    f"durable failure tracked for {dispatch_ctx.tool_name}",
                ),
            )
            return

        await submit_fact(
            ctx,
            key=failure_key,
            value=json.dumps({
                "tool": dispatch_ctx.tool_name,
                "key": action_key_param(action.params),
                "reason": "",
                "count": 0,
                **_durable_failure_summary_fields(result.summary),
                "last_seen": time.time(),
                "muted_until": 0,
            }, ensure_ascii=False),
            scope="system",
            source="execution/failure_clear",
            decision_basis=_decision_basis(
                action,
                f"durable failure cleared for {dispatch_ctx.tool_name}",
            ),
        )

    async def _dispatch_act(self, action: JudgmentOutput, ctx: ToolContext) -> ToolResult:
        active_task = await _resolve_execution_active_task(ctx)
        dispatch_ctx = self._make_dispatch_ctx(action, active_task)
        dispatch_ctx = dispatch_ctx.with_profile(
            self._resolve_execution_profile(
                dispatch_ctx=dispatch_ctx,
                action=action,
                ctx=ctx,
            )
        )

        durable_policy = await _load_durable_failure_policy(ctx.task_store)
        durable_threshold = int(durable_policy.get("threshold") or self._cfg.thresholds.durable_failure_threshold)
        durable_ttl_sec = int(durable_policy.get("ttl_sec") or self._cfg.thresholds.durable_failure_ttl_sec)

        decision_basis = _decision_basis(action)
        dispatch_ctx = await self._start_run_record(
            dispatch_ctx,
            action,
            ctx,
            decision_basis=decision_basis,
        )

        entry = (ctx.registry or self._registry).get(dispatch_ctx.tool_name)
        if not entry:
            _log.warning(
                "[exec-miss] %s not_registered=true",
                dispatch_ctx.scope_fields(),
            )
            return await self._finalize_and_return(
                dispatch_ctx,
                ctx,
                dispatch_ctx.stamp_metadata(ToolResult(
                    summary=f"工具不存在: {dispatch_ctx.tool_name!r}",
                    error="ToolNotFound",
                    skipped=True,
                    kind="error",
                )),
            )

        if self._cfg.loop.debug:
            _log.debug("[exec] %s params=%s", dispatch_ctx.tool_name, action.params)
        _log.info("[exec] %s", dispatch_ctx.tool_name)

        preflight_result = await self._preflight_tool_result(
            action,
            dispatch_ctx,
            ctx,
            active_task,
        )
        if preflight_result is not None:
            return await self._finalize_and_return(
                dispatch_ctx,
                ctx,
                preflight_result,
            )

        dispatch_started = time.monotonic()
        try:
            result = await self._workers.dispatch(dispatch_ctx.worker_type, entry, action, ctx)
        except Exception as exc:
            _log.exception(
                "[exec-error] %s dispatch=raised",
                dispatch_ctx.scope_fields(),
            )
            result = ToolResult(
                summary=f"工具执行异常: {exc}",
                evidence=str(exc),
                error=str(exc),
                kind="execute_result",
            )
        result = dispatch_ctx.stamp_metadata(result)
        result = _normalize_tool_result_text_fields(result)
        result.metadata.setdefault("dispatch_ms", int((time.monotonic() - dispatch_started) * 1000))

        summary_log, error_log, state_log = _tool_result_log_fields(result)
        worker_log = _worker_log_fields(result)
        _log.info(
            "[tool-result] %s worker_meta=%s skipped=%s error=%s summary=%s state=%s",
            dispatch_ctx.scope_fields(dispatch_ms=result.metadata.get("dispatch_ms")),
            worker_log,
            result.skipped,
            error_log or "-",
            summary_log or "-",
            state_log or "-",
        )

        if result.error and not result.skipped and ctx.task_store is not None:
            task_id = str(_resolved_run_task_id(result, dispatch_ctx.run_task_id) or "")
            await ctx.task_store.record_failure(
                kind=dispatch_ctx.tool_name,
                summary=_durable_failure_summary_fields(result.summary)["last_summary_preview"],
                context=_durable_failure_summary_fields(result.evidence)["last_summary_preview"],
                task_id=task_id,
            )

        await self._sync_durable_failure(
            action,
            dispatch_ctx,
            ctx,
            result,
            durable_threshold=durable_threshold,
            durable_ttl_sec=durable_ttl_sec,
        )

        return await self._finalize_and_return(
            dispatch_ctx,
            ctx,
            result,
        )
