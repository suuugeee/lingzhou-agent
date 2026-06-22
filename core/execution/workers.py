"""core/execution/workers.py — Run worker 抽象 (WorkerLayer)。

这一层现在承担真正的 worker 语义边界：
- tool-chain-worker：普通工具链调用
- exec-worker：后台/前台进程执行与监控元数据规范化
- multimodal-worker：多模态输入归一化
- llm-worker：LLM 驱动工具的监控协议归一化
并发语义：
- 每类 worker 拥有独立 semaphore，形成独立并发域
- dispatch 会记录等待时长 / inflight / limit，便于判断 worker 是否真的并发
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.execution.run_profile import (
    WORKER_EVOLVE,
    WORKER_EXEC,
    WORKER_LIMITS_CONFIG_KEY,
    WORKER_LLM,
    WORKER_MULTIMODAL,
    WORKER_TOOL_CHAIN,
)
from tools.registry import ToolContext, ToolResult

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry


WorkerHandler = Callable[["ToolEntry", "JudgmentOutput", ToolContext], Awaitable[ToolResult]]

_WORKBENCH_PREFERENCE_FIELDS = {
    "domain",
    "intent",
    "hypothesis",
    "working_hypothesis",
    "recovery_state",
    "next_verification",
    "capabilities",
    "experiments",
    "evidence",
    "open_questions",
    "completion_checks",
    "progress",
    "failures",
}

_TOOL_PARAM_ALIASES = {
    "shell.run": {
        "command": ("cmd", "payload.command", "params.command"),
        "timeout": ("timeout_sec", "timeoutSec", "payload.timeout", "params.timeout"),
        "workdir": ("work_dir", "dir", "payload.workdir", "params.workdir"),
        "sandbox": (
            "sandbox_enabled",
            "is_sandbox",
            "payload.sandbox",
            "params.sandbox",
        ),
    },
    "task.resume": {
        "task_id": ("id", "payload.task_id", "params.task_id", "payload.id", "params.id"),
    },
    "task.workbench": {
        "task_id": ("id", "payload.task_id", "params.task_id", "payload.id", "params.id"),
    },
}

_DEFAULT_WORKER_TYPE = WORKER_TOOL_CHAIN
_WORKER_SPECS = (
    (_DEFAULT_WORKER_TYPE, "_execute_tool_chain", WORKER_LIMITS_CONFIG_KEY[WORKER_TOOL_CHAIN]),
    (WORKER_EVOLVE, "_execute_tool_chain", WORKER_LIMITS_CONFIG_KEY[WORKER_EVOLVE]),
    (WORKER_EXEC, "_execute_exec", WORKER_LIMITS_CONFIG_KEY[WORKER_EXEC]),
    (WORKER_MULTIMODAL, "_execute_multimodal", WORKER_LIMITS_CONFIG_KEY[WORKER_MULTIMODAL]),
    (WORKER_LLM, "_execute_llm", WORKER_LIMITS_CONFIG_KEY[WORKER_LLM]),
)


def _resolve_nested_alias(params: dict[str, Any], alias: str) -> Any:
    if "." not in alias:
        return params.get(alias)
    if alias.count(".") != 1:
        return None
    root_key, nested_key = alias.split(".", 1)
    nested = params.get(root_key)
    if isinstance(nested, dict):
        return nested.get(nested_key)
    return None


def _is_missing_param_value(param_name: str, value: Any) -> bool:
    if value is None:
        return True
    if param_name == "command" and isinstance(value, str) and not value.strip():
        return True
    if param_name == "task_id":
        try:
            return int(value or 0) <= 0
        except (TypeError, ValueError):
            return True
    return False


def _image_input_count(params: dict[str, Any]) -> int:
    """统计工具参数里的图片输入数量。"""
    image_count = 0
    for key in ("path", "paths", "image", "images"):
        value = params.get(key)
        if not value:
            continue
        if isinstance(value, list):
            image_count += len(value)
        else:
            image_count += 1
    return image_count


def _is_background_result(result: ToolResult) -> bool:
    return bool(isinstance(result.state_delta, dict) and result.state_delta.get("background"))


def _background_monitor_payload(session_id: str | None) -> dict[str, str]:
    return {"kind": "process", "session_id": session_id or ""}


def _repair_task_workbench_params(entry: ToolEntry, params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        return params
    if entry.manifest.name != "task.workbench":
        return params
    existing_workbench = params.get("workbench")
    if isinstance(existing_workbench, dict) and existing_workbench:
        return params

    workbench_payload: dict[str, Any] = {
        key: value for key, value in params.items() if key in _WORKBENCH_PREFERENCE_FIELDS
    }
    progress_items: list[str] = []
    for alias in ("current_step", "summary"):
        value = str(params.get(alias) or "").strip()
        if value:
            progress_items.append(value)
    result_summary = str(params.get("result_summary") or "").strip()
    if result_summary:
        workbench_payload.setdefault("evidence", [])
        if isinstance(workbench_payload["evidence"], list):
            workbench_payload["evidence"].append(result_summary)
    if progress_items:
        workbench_payload.setdefault("progress", [])
        if isinstance(workbench_payload["progress"], list):
            workbench_payload["progress"].extend(progress_items)
    if not workbench_payload:
        return params

    repaired = dict(params)
    if "task_id" not in repaired and "id" in repaired:
        repaired["task_id"] = repaired["id"]
    repaired["workbench"] = workbench_payload
    return repaired


def _param_template_value(param_type: str) -> Any:
    if param_type == "string":
        return "<string>"
    if param_type == "number":
        return 0
    if param_type == "boolean":
        return False
    if param_type == "array":
        return []
    if param_type == "object":
        return {}
    return None


def _expected_param_specs(entry: ToolEntry) -> list[dict[str, Any]]:
    return [
        {
            "name": param.name,
            "type": param.type,
            "required": bool(param.required),
            "description": param.description,
        }
        for param in entry.manifest.params
    ]


def _retry_params_template(entry: ToolEntry, params: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    retry = dict(params or {})
    for param in entry.manifest.params:
        if param.name not in missing:
            continue
        retry[param.name] = _param_template_value(param.type)
    return retry


def _repair_tool_param_aliases(entry: ToolEntry, params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        return params
    alias_map = _TOOL_PARAM_ALIASES.get(entry.manifest.name)
    if not alias_map:
        return params

    repaired = dict(params)
    for canonical, aliases in alias_map.items():
        value = repaired.get(canonical)
        if not _is_missing_param_value(canonical, value):
            continue
        for alias in aliases:
            aliased_value = _resolve_nested_alias(repaired, alias)
            if aliased_value is None:
                continue
            if _is_missing_param_value(canonical, aliased_value):
                continue
            repaired[canonical] = aliased_value
            break
    return repaired


def _validate_required_params(entry: ToolEntry, params: dict[str, Any]) -> ToolResult | None:
    missing: list[str] = []
    for param in entry.manifest.params:
        if not param.required:
            continue
        value = params.get(param.name)
        if _is_missing_param_value(param.name, value):
            missing.append(param.name)
            continue
        if param.type == "string" and not str(value).strip():
            # 仅将显式空串 path 交给工具层返回 EmptyPath；缺省/纯空白仍在 worker 拦截
            if param.name == "path" and str(value) == "":
                continue
            missing.append(param.name)
    if not missing:
        return None
    expected_params = _expected_param_specs(entry)
    retry_template = _retry_params_template(entry, params, missing)
    recovery_next_step = (
        f"按 {entry.manifest.name} 的 manifest 重新调用工具；"
        f"补齐必填参数 {', '.join(missing)}（优先使用标准字段 {', '.join(missing)}，不要把参数嵌套在 payload/params 这类外层字段）。"
    )
    state_delta = {
        "tool_input_invalid": True,
        "tool_name": entry.manifest.name,
        "missing_params": missing,
        "expected_params": expected_params,
        "retry_params_template": retry_template,
        "recovery_next_step": recovery_next_step,
    }
    return ToolResult(
        summary=f"工具参数缺失: {entry.manifest.name} requires {', '.join(missing)}",
        error="ToolInputInvalid",
        skipped=True,
        kind="error",
        state_delta=state_delta,
        metadata={
            "tool_name": entry.manifest.name,
            "log_summary": f"{entry.manifest.name} missing_params={','.join(missing)}",
            "missing_params": missing,
            "expected_params": expected_params,
            "retry_params_template": retry_template,
            "recovery_next_step": recovery_next_step,
        },
    )


async def _repair_task_id_from_active_task(entry: ToolEntry, params: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if not isinstance(params, dict):
        return params
    if entry.manifest.name not in {"task.resume", "task.workbench"}:
        return params
    if not _is_missing_param_value("task_id", params.get("task_id")):
        return params
    get_active = getattr(ctx, "get_active_task", None)
    if not callable(get_active):
        return params
    try:
        active_task = await get_active()
    except Exception:
        active_task = None
    task_id = getattr(active_task, "id", None)
    if _is_missing_param_value("task_id", task_id):
        return params
    repaired = dict(params)
    repaired["task_id"] = task_id
    return repaired


async def _call_handler(entry: ToolEntry, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """调用工具 handler，兼容同步函数与 dict 返回值（进化工具可能产生这两种情况）。"""
    params = _repair_tool_param_aliases(entry, _repair_task_workbench_params(entry, params))
    params = await _repair_task_id_from_active_task(entry, params, ctx)
    validation_error = _validate_required_params(entry, params)
    if validation_error is not None:
        return validation_error
    raw = entry.handler(params, ctx)
    if asyncio.iscoroutine(raw):
        raw = await raw
    if isinstance(raw, dict):
        # 防御性清洗：过滤非 ToolResult 字段（如旧工具遗留的 success），并补全必填 summary
        valid = ToolResult.__dataclass_fields__.keys()
        raw = {k: v for k, v in raw.items() if k in valid}
        raw.setdefault('summary', '执行完成')
        raw = ToolResult(**raw)
    if not isinstance(raw, ToolResult):
        raw = ToolResult(summary=str(raw))
    return raw  # type: ignore[return-value]


@dataclass
class _WorkerPool:
    name: str
    limit: int
    semaphore: asyncio.Semaphore
    inflight: int = 0
    waiting: int = 0
    peak_inflight: int = 0


def _worker_limit(cfg: Any | None, attr_name: str) -> int:
    loop_cfg = getattr(cfg, "loop", None)
    raw = getattr(loop_cfg, attr_name, None)
    try:
        limit = int(raw or 1)
    except (TypeError, ValueError):
        limit = 1
    return max(1, limit)


class WorkerLayer:
    def __init__(self, cfg: Any | None = None) -> None:
        self._handlers = {
            worker_type: getattr(self, handler_name)
            for worker_type, handler_name, _ in _WORKER_SPECS
        }
        self._pools: dict[str, _WorkerPool] = {}
        for worker_type, _, attr_name in _WORKER_SPECS:
            limit = _worker_limit(cfg, attr_name)
            self._pools[worker_type] = _WorkerPool(
                name=worker_type,
                limit=limit,
                semaphore=asyncio.Semaphore(limit),
            )

    async def dispatch(
        self,
        worker_type: str,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        handler = self._handlers.get(worker_type, self._execute_tool_chain)
        pool = self._pools.get(worker_type) or self._pools[_DEFAULT_WORKER_TYPE]
        wait_ms, waiting_before = await self._acquire_slot(pool)
        inflight_now = self._enter_flight(pool)
        try:
            result = await handler(entry, action, ctx)
        finally:
            self._leave_flight(pool)
            pool.semaphore.release()
        self._attach_common_metadata(
            result=result,
            worker_type=worker_type,
            tool_name=action.chosen_action_id or "",
            pool=pool,
            wait_ms=wait_ms,
            waiting=max(0, waiting_before),
            inflight=inflight_now,
        )
        return result

    async def _acquire_slot(self, pool: _WorkerPool) -> tuple[int, int]:
        started = time.monotonic()
        waiting_before = pool.waiting
        pool.waiting += 1
        try:
            await pool.semaphore.acquire()
        finally:
            wait_ms = int((time.monotonic() - started) * 1000)
            pool.waiting = max(0, pool.waiting - 1)
        return wait_ms, waiting_before

    def _enter_flight(self, pool: _WorkerPool) -> int:
        pool.inflight += 1
        pool.peak_inflight = max(pool.peak_inflight, pool.inflight)
        return pool.inflight

    @staticmethod
    def _leave_flight(pool: _WorkerPool) -> None:
        pool.inflight = max(0, pool.inflight - 1)

    def _attach_common_metadata(
        self,
        *,
        result: ToolResult,
        worker_type: str,
        tool_name: str,
        pool: _WorkerPool,
        wait_ms: int,
        waiting: int,
        inflight: int,
    ) -> None:
        """补齐 worker 统一执行元信息（保持默认优先级）。"""
        result.metadata.setdefault("worker_type", worker_type)
        result.metadata.setdefault("tool_name", tool_name)
        result.metadata.setdefault("worker_limit", pool.limit)
        result.metadata.setdefault("worker_wait_ms", wait_ms)
        result.metadata.setdefault("worker_inflight", inflight)
        result.metadata.setdefault("worker_waiting", waiting)
        result.metadata.setdefault("worker_peak_inflight", pool.peak_inflight)

    async def _execute_tool_chain(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        return await self._call_tool_and_tag_path(
            entry=entry,
            action=action,
            ctx=ctx,
            worker_path="tool-chain",
        )

    async def _execute_exec(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await self._call_tool_and_tag_path(
            entry=entry,
            action=action,
            ctx=ctx,
            worker_path="exec",
        )
        background = _is_background_result(result)
        result.metadata.setdefault("execution_mode", "background" if background else "foreground")
        if background and not result.metadata.get("session_id") and result.resource_key:
            result.metadata["session_id"] = result.resource_key
        if background and result.metadata.get("session_id"):
            result.metadata.setdefault(
                "run_monitor",
                _background_monitor_payload(str(result.metadata.get("session_id") or "")),
            )
        return result

    async def _execute_multimodal(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await self._call_tool_and_tag_path(
            entry=entry,
            action=action,
            ctx=ctx,
            worker_path="multimodal",
        )
        image_count = _image_input_count(action.params)
        result.metadata.setdefault("modality", "image")
        result.metadata.setdefault("input_count", max(1, image_count) if image_count else 1)
        return result

    async def _execute_llm(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await self._call_tool_and_tag_path(
            entry=entry,
            action=action,
            ctx=ctx,
            worker_path="llm",
        )
        result.metadata.setdefault("reasoning_mode", "tool-mediated-llm")
        monitor_key = str(
            action.params.get("monitor_fact_key")
            or action.params.get("status_fact_key")
            or ""
        ).strip()
        if monitor_key:
            result.metadata.setdefault(
                "run_monitor",
                {
                    "kind": "fact",
                    "key": monitor_key,
                    "status_field": str(action.params.get("monitor_status_field") or "status"),
                    "progress_field": str(action.params.get("monitor_progress_field") or "progress"),
                },
            )
        return result

    async def _call_tool_and_tag_path(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
        worker_path: str,
    ) -> ToolResult:
        result = await _call_handler(entry, action.params, ctx)
        result.metadata.setdefault("worker_path", worker_path)
        return result
