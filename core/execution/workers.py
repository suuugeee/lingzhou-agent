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

from tools.registry import ToolContext, ToolResult

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from tools.registry import ToolEntry


WorkerHandler = Callable[["ToolEntry", "JudgmentOutput", ToolContext], Awaitable[ToolResult]]


def _validate_required_params(entry: ToolEntry, params: dict[str, Any]) -> ToolResult | None:
    missing: list[str] = []
    for param in entry.manifest.params:
        if not param.required:
            continue
        value = params.get(param.name)
        if value is None:
            missing.append(param.name)
            continue
        if param.type == "string" and not str(value).strip():
            # 仅将显式空串 path 交给工具层返回 EmptyPath；缺省/纯空白仍在 worker 拦截
            if param.name == "path" and str(value) == "":
                continue
            missing.append(param.name)
    if not missing:
        return None
    return ToolResult(
        summary=f"工具参数缺失: {entry.manifest.name} requires {', '.join(missing)}",
        error="ToolInputInvalid",
        skipped=True,
        kind="error",
        metadata={
            "tool_name": entry.manifest.name,
            "log_summary": f"{entry.manifest.name} missing_params={','.join(missing)}",
            "missing_params": missing,
        },
    )


async def _call_handler(entry: ToolEntry, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """调用工具 handler，兼容同步函数与 dict 返回值（进化工具可能产生这两种情况）。"""
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
        self._handlers: dict[str, WorkerHandler] = {
            "tool-chain-worker": self._execute_tool_chain,
            "exec-worker": self._execute_exec,
            "multimodal-worker": self._execute_multimodal,
            "llm-worker": self._execute_llm,
        }
        self._pools: dict[str, _WorkerPool] = {}
        for worker_type, attr_name in (
            ("tool-chain-worker", "max_tool_chain_workers"),
            ("exec-worker", "max_exec_workers"),
            ("multimodal-worker", "max_multimodal_workers"),
            ("llm-worker", "max_llm_workers"),
        ):
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
        pool = self._pools.get(worker_type) or self._pools["tool-chain-worker"]
        wait_started = time.monotonic()
        queued_before = pool.waiting
        pool.waiting += 1
        await pool.semaphore.acquire()
        wait_ms = int((time.monotonic() - wait_started) * 1000)
        pool.waiting = max(0, pool.waiting - 1)
        pool.inflight += 1
        pool.peak_inflight = max(pool.peak_inflight, pool.inflight)
        inflight_now = pool.inflight
        try:
            result = await handler(entry, action, ctx)
        finally:
            pool.inflight = max(0, pool.inflight - 1)
            pool.semaphore.release()
        result.metadata.setdefault("worker_type", worker_type)
        result.metadata.setdefault("tool_name", action.chosen_action_id or "")
        result.metadata.setdefault("worker_limit", pool.limit)
        result.metadata.setdefault("worker_wait_ms", wait_ms)
        result.metadata.setdefault("worker_inflight", inflight_now)
        result.metadata.setdefault("worker_waiting", max(0, queued_before))
        result.metadata.setdefault("worker_peak_inflight", pool.peak_inflight)
        return result

    async def _execute_tool_chain(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await _call_handler(entry, action.params, ctx)
        result.metadata.setdefault("worker_path", "tool-chain")
        return result

    async def _execute_exec(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await _call_handler(entry, action.params, ctx)
        result.metadata.setdefault("worker_path", "exec")
        background = bool(isinstance(result.state_delta, dict) and result.state_delta.get("background"))
        result.metadata.setdefault("execution_mode", "background" if background else "foreground")
        if background and not result.metadata.get("session_id") and result.resource_key:
            result.metadata["session_id"] = result.resource_key
        if background and result.metadata.get("session_id"):
            result.metadata.setdefault(
                "run_monitor",
                {"kind": "process", "session_id": str(result.metadata.get("session_id") or "")},
            )
        return result

    async def _execute_multimodal(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await _call_handler(entry, action.params, ctx)
        result.metadata.setdefault("worker_path", "multimodal")
        image_count = 0
        for key in ("path", "paths", "image", "images"):
            value = action.params.get(key)
            if not value:
                continue
            if isinstance(value, list):
                image_count += len(value)
            else:
                image_count += 1
        result.metadata.setdefault("modality", "image")
        result.metadata.setdefault("input_count", max(1, image_count) if image_count else 1)
        return result

    async def _execute_llm(
        self,
        entry: ToolEntry,
        action: JudgmentOutput,
        ctx: ToolContext,
    ) -> ToolResult:
        result = await _call_handler(entry, action.params, ctx)
        result.metadata.setdefault("worker_path", "llm")
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
