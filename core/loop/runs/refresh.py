from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from core.cortex import build_auto_cortex_result_patch
from core.execution import (
    WORKER_EXEC,
    build_meta_reflection,
    record_meta_reflection_memory,
    record_run_outcome_memory,
)
from core.metabolic import resolve_metabolic, submit_fact, update_run, update_task_result
from store.compact import compact_runtime_text
from store.task import Run, TaskStore, build_task_run_result_patch
from tools.registry import ToolResult

if TYPE_CHECKING:
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory

_log = logging.getLogger("lingzhou.loop")

_RUN_PROGRESS_CRYSTAL_CHARS = 120
_RUN_REFRESH_TEXT_PREVIEW_CHARS = 4000


def _refresh_text_preview(value: Any, *, marker_label: str = "run monitor output") -> str:
    return compact_runtime_text(
        value,
        limit=_RUN_REFRESH_TEXT_PREVIEW_CHARS,
        marker_label=marker_label,
    )


def _first_nonempty_preview(*values: Any, fallback: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return _refresh_text_preview(text)
    return fallback


def _compact_monitor_value(value: Any, *, label: str = "run monitor value", depth: int = 0) -> Any:
    if isinstance(value, str):
        return _refresh_text_preview(value, marker_label=label)
    if isinstance(value, dict):
        if depth >= 4:
            return _refresh_text_preview(value, marker_label=label)
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            compacted[text_key] = _compact_monitor_value(
                item,
                label=f"run monitor {text_key}",
                depth=depth + 1,
            )
            if isinstance(item, str) and len(item) > _RUN_REFRESH_TEXT_PREVIEW_CHARS:
                compacted[f"{text_key}_chars"] = len(item)
        return compacted
    if isinstance(value, list):
        if depth >= 4:
            return _refresh_text_preview(value, marker_label=label)
        if len(value) <= 80:
            return [
                _compact_monitor_value(item, label=label, depth=depth + 1)
                for item in value
            ]
        head_count = 39
        tail_count = 40
        omitted = len(value) - head_count - tail_count
        return [
            *[
                _compact_monitor_value(item, label=label, depth=depth + 1)
                for item in value[:head_count]
            ],
            {"_persistent_omitted_items": omitted},
            *[
                _compact_monitor_value(item, label=label, depth=depth + 1)
                for item in value[-tail_count:]
            ],
        ]
    return value


def _compact_monitor_snapshot(payload: Any) -> Any:
    return _compact_monitor_value(payload, label="run monitor snapshot")


def _run_monitor_config(run: Run) -> dict[str, Any] | None:
    candidates = [
        run.extras.get("run_monitor"),
        (run.output_json.get("state_delta") or {}).get("run_monitor") if isinstance(run.output_json, dict) else None,
        (run.output_json.get("metadata") or {}).get("run_monitor") if isinstance(run.output_json, dict) else None,
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("kind") or "").strip()
        if kind == "fact" and str(candidate.get("key") or "").strip():
            return candidate
        if kind == "process" and str(candidate.get("session_id") or "").strip():
            return candidate
    if run.session_id:
        return {"kind": "process", "session_id": run.session_id}
    return None


def _run_update_summary(run: Run, status: str | None = None, **extra: Any) -> dict[str, Any]:
    summary = {"run_id": run.id, "task_id": run.task_id, "status": status or run.status}
    summary.update(extra)
    return summary


async def _upsert_task_progress_fact(
    task_store_or_metabolic: Any,
    task_id: int | None,
    value: str,
    *,
    source: str,
) -> None:
    if not task_id or not value:
        return
    await submit_fact(
        task_store_or_metabolic,
        key=f"task:{task_id}:progress",
        value=value,
        scope="task",
        source=source,
    )


def _parse_run_monitor_snapshot(raw: str, monitor: dict[str, Any]) -> tuple[str, str, Any]:
    payload: Any = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = raw

    status_field = str(monitor.get("status_field") or "status")
    progress_field = str(monitor.get("progress_field") or "progress")
    success_values = {str(v).strip().lower() for v in (monitor.get("success_values") or ["succeeded", "success", "done", "completed"])}
    failed_values = {str(v).strip().lower() for v in (monitor.get("failed_values") or ["failed", "error"])}
    cancelled_values = {str(v).strip().lower() for v in (monitor.get("cancelled_values") or ["cancelled", "canceled"])}

    status_value: Any = raw
    progress_value: Any = ""
    if isinstance(payload, dict):
        status_value = payload.get(status_field, payload.get("status", raw))
        progress_value = payload.get(progress_field, payload.get("progress", ""))

    normalized = str(status_value or "").strip().lower()
    if normalized in success_values:
        status = "succeeded"
    elif normalized in failed_values:
        status = "failed"
    elif normalized in cancelled_values:
        status = "cancelled"
    else:
        status = "running"
    progress = str(progress_value or "").strip()
    return status, progress, payload


async def _finalize_refreshed_run_learning(
    task_store: TaskStore,
    *,
    run: Run,
    status: str,
    summary: str,
    error: str,
    evidence: str,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    metabolic: Any | None = None,
) -> None:
    if error:
        _log.debug("[run-refresh] record failure for run=%s tool=%s error=%s", run.id, run.tool_name, error)
        await task_store.record_failure(
            kind=run.tool_name or run.worker_type or "run",
            summary=summary,
            context=evidence,
            task_id=str(run.task_id) if run.task_id else "",
        )

    result = ToolResult(
        summary=summary,
        evidence=evidence,
        error=error,
        skipped=(status == "cancelled"),
        kind="execute_result",
        metadata={
            "tool_name": run.tool_name,
            "worker_type": run.worker_type,
            "session_id": run.session_id,
        },
    )
    meta = build_meta_reflection(
        run_id=run.id,
        task_id=run.task_id,
        tool_name=run.tool_name,
        result=result,
    )
    if not meta:
        return
    _log.debug(
        "[run-refresh] meta reflection run=%s target=%s decision=%s",
        run.id,
        meta.get("target_kind"),
        meta.get("decision"),
    )
    await task_store.add_meta_reflection(
        reflection_id=str(meta["reflection_id"]),
        target_kind=str(meta["target_kind"]),
        trigger=str(meta["trigger"]),
        loop_level=str(meta["loop_level"]),
        diagnosis=str(meta["diagnosis"]),
        proposal=str(meta["proposal"]),
        verification_plan=str(meta["verification_plan"]),
        decision=str(meta["decision"]),
        task_id=int(meta["task_id"]),
        run_id=int(meta["run_id"]),
        tool_name=str(meta["tool_name"]),
    )
    if episodic is None and semantic is None:
        return
    await record_meta_reflection_memory(
        episodic,
        semantic,
        meta,
        task_store=task_store,
        metabolic=metabolic,
    )


async def _refresh_run_via_fact_monitor(
    task_store: TaskStore,
    run: Run,
    monitor: dict[str, Any],
    *,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    memory_cfg: Any | None = None,
    metabolic: Any | None = None,
) -> dict[str, Any]:
    key = str(monitor.get("key") or "").strip()
    raw, found = await task_store.get_fact(key)
    if not found:
        _log.debug("[run-monitor] fact key=%s run=%s snapshot missing", key, run.id)
        return _run_update_summary(run)

    status, progress, payload = _parse_run_monitor_snapshot(raw, monitor)
    compact_payload = _compact_monitor_snapshot(payload)
    progress_preview = _refresh_text_preview(progress, marker_label="run monitor progress")
    crystal = progress_preview if progress and progress_preview != run.progress else ""
    if crystal:
        _log.debug("[run-monitor] fact key=%s run=%s progress=%s", key, run.id, progress_preview)
    output_json = dict(run.output_json)
    output_json["monitor_snapshot"] = compact_payload
    output_json["monitor_key"] = key
    output_json["monitor_raw_chars"] = len(raw)
    await update_run(
        metabolic or task_store,
        run.id,
        status=status,
        output_json=output_json,
        log_text=(progress_preview or _refresh_text_preview(payload or raw)),
        error_text=(
            _refresh_text_preview(payload.get("error") or "", marker_label="run monitor error")
            if isinstance(payload, dict)
            else (_refresh_text_preview(raw, marker_label="run monitor error") if status == "failed" else "")
        ),
        progress=progress_preview,
        source="loop/runs/refresh/monitor_fact",
        proposal_run_id=run.id,
        decision_basis="run_monitor_fact_snapshot",
    )
    await _upsert_task_progress_fact(
        metabolic or task_store,
        task_id=run.task_id,
        value=crystal,
        source="run_refresh/monitor",
    )
    if status in {"succeeded", "failed", "cancelled"}:
        summary_raw = progress or (str(payload.get("summary") or "") if isinstance(payload, dict) else raw) or f"run {status}"
        error_raw = str(payload.get("error") or "") if isinstance(payload, dict) else (raw if status == "failed" else "")
        summary = _refresh_text_preview(summary_raw, marker_label="run monitor summary")
        error = _refresh_text_preview(error_raw, marker_label="run monitor error")
        evidence = _refresh_text_preview(
            payload if isinstance(payload, dict) else raw,
            marker_label="run monitor evidence",
        )
        _log.info(
            "[run-monitor] fact key=%s run=%s status=%s progress=%s error=%s",
            key,
            run.id,
            status,
            (progress_preview or "-"),
            (error or "-"),
        )
        if run.task_id:
            task_result_patch = build_task_run_result_patch(
                run_id=run.id,
                status=status,
                worker_type=run.worker_type,
                tool_name=run.tool_name,
                session_id=run.session_id,
                summary=summary,
                error=error or None,
            )
            task_result_patch.update(await build_auto_cortex_result_patch(
                task_store=task_store,
                task_id=run.task_id,
                run_id=run.id,
                tool_name=run.tool_name,
                status=status,
                summary=summary,
                error=error or None,
                evidence=evidence,
                progress=progress_preview,
                state_delta=compact_payload if isinstance(compact_payload, dict) else {},
                artifact_paths=[],
            ))
            await update_task_result(
                task_store,
                run.task_id,
                task_result_patch,
                source="loop/runs/refresh/monitor_fact",
            )
        await record_run_outcome_memory(
            episodic,
            semantic,
            memory_cfg=memory_cfg,
            run_id=run.id,
            task_id=run.task_id,
            tool_name=run.tool_name,
            worker_type=run.worker_type,
            status=status,
            progress=progress_preview,
            summary=summary,
            error=error,
            task_store=task_store,
            metabolic=metabolic,
        )
        await _finalize_refreshed_run_learning(
            task_store,
            run=run,
            status=status,
            summary=summary,
            error=error,
            evidence=evidence,
            episodic=episodic,
            semantic=semantic,
            metabolic=metabolic,
        )
    return _run_update_summary(run, status, session_id=run.session_id, crystal=crystal)


async def _refresh_run_via_process_monitor(
    task_store: TaskStore,
    run: Run,
    monitor: dict[str, Any],
    *,
    manager: Any,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    memory_cfg: Any | None = None,
    metabolic: Any | None = None,
) -> dict[str, Any]:
    session_id = str(monitor.get("session_id") or run.session_id or "").strip()
    if not session_id or manager is None:
        return _run_update_summary(run)

    info = manager.get(session_id)
    if info is None:
        return _run_update_summary(run)
    if not info.finished:
        _log.debug("[run-monitor] process session=%s run=%s still running", session_id, run.id)
        stdout_text = info.stdout or ""
        stdout_preview = _refresh_text_preview(stdout_text, marker_label="run monitor stdout")
        last_crystal_chars = int(run.extras.get("last_crystal_chars", 0) or 0)
        crystal_excerpt = ""
        if len(stdout_text) - last_crystal_chars >= _RUN_PROGRESS_CRYSTAL_CHARS:
            crystal_excerpt = stdout_text[last_crystal_chars:][-400:].strip()
            await update_run(
                metabolic or task_store,
                run.id,
                status="running",
                output_json={
                    **run.output_json,
                    "progress_excerpt": crystal_excerpt,
                    "stdout_preview": stdout_preview,
                    "stdout_chars": len(stdout_text),
                },
                log_text=stdout_preview,
                session_id=session_id,
                progress=crystal_excerpt,
                extras={**run.extras, "last_crystal_chars": len(stdout_text), "run_monitor": monitor},
                source="loop/runs/refresh/process_progress",
                proposal_run_id=run.id,
                decision_basis="run_monitor_progress_update",
            )
            await _upsert_task_progress_fact(
                metabolic or task_store,
                task_id=run.task_id,
                value=crystal_excerpt,
                source="run_refresh/progress",
            )
        return _run_update_summary(run, "running", session_id=session_id, crystal=crystal_excerpt)

    status = "succeeded" if (info.return_code in (0, None) and not info.timed_out and not info.error) else "failed"
    _log.info("[run-monitor] process session=%s run=%s finished status=%s", session_id, run.id, status)
    stdout_preview = _refresh_text_preview(info.stdout or "", marker_label="run monitor stdout")
    stderr_preview = _refresh_text_preview(info.stderr or "", marker_label="run monitor stderr")
    error_preview = _refresh_text_preview(info.error or "", marker_label="run monitor error")
    summary_preview = _first_nonempty_preview(
        info.stdout,
        info.stderr,
        info.error,
        fallback=f"process {status}",
    )
    progress_preview = _first_nonempty_preview(
        info.stdout,
        info.stderr,
        info.error,
        fallback=status,
    )
    output_json = dict(run.output_json)
    output_json.update({
        "session_id": session_id,
        "return_code": info.return_code,
        "timed_out": info.timed_out,
        "stdout": stdout_preview,
        "stderr": stderr_preview,
        "error": error_preview,
        "stdout_chars": len(info.stdout or ""),
        "stderr_chars": len(info.stderr or ""),
        "error_chars": len(info.error or ""),
    })
    await update_run(
        metabolic or task_store,
        run.id,
        status=status,
        output_json=output_json,
        log_text=stdout_preview,
        error_text=error_preview or ("timed_out" if info.timed_out else (f"exit_code={info.return_code}" if status == "failed" else "")),
        session_id=session_id,
        progress=progress_preview,
        extras={
            **run.extras,
            "return_code": info.return_code,
            "timed_out": info.timed_out,
            "background": info.background,
            "run_monitor": monitor,
        },
        source="loop/runs/refresh/process_finished",
        proposal_run_id=run.id,
        decision_basis="run_monitor_process_finished",
    )
    if run.task_id:
        task_result_patch = build_task_run_result_patch(
            run_id=run.id,
            status=status,
            worker_type=run.worker_type,
            tool_name=run.tool_name,
            session_id=session_id,
            summary=summary_preview,
            error=error_preview or None,
        )
        task_result_patch.update(await build_auto_cortex_result_patch(
            task_store=task_store,
            task_id=run.task_id,
            run_id=run.id,
            tool_name=run.tool_name,
            status=status,
            summary=summary_preview,
            error=error_preview,
            evidence="\n".join(part for part in [stderr_preview, error_preview] if part).strip(),
            progress=progress_preview,
            state_delta=output_json,
            artifact_paths=[],
        ))
        await update_task_result(
            task_store,
            run.task_id,
            task_result_patch,
            source="loop/runs/refresh/monitor_process",
        )
    await record_run_outcome_memory(
        episodic,
        semantic,
        memory_cfg=memory_cfg,
        run_id=run.id,
        task_id=run.task_id,
        tool_name=run.tool_name,
        worker_type=run.worker_type,
        status=status,
        progress=progress_preview,
        summary=summary_preview,
        error=error_preview,
        task_store=task_store,
        metabolic=metabolic,
    )
    await _finalize_refreshed_run_learning(
        task_store,
        run=run,
        status=status,
        summary=summary_preview,
        error=error_preview,
        evidence="\n".join(part for part in [stderr_preview, error_preview] if part).strip(),
        episodic=episodic,
        semantic=semantic,
        metabolic=metabolic,
    )
    return _run_update_summary(run, status, session_id=session_id)


async def refresh_running_runs(
    task_store: TaskStore,
    *,
    episodic: EpisodicMemory | None = None,
    semantic: SemanticMemory | None = None,
    memory_cfg: Any | None = None,
    metabolic: Any | None = None,
) -> list[dict[str, Any]]:
    """刷新所有 running runs，优先走内建 exec 监控，其次走通用 fact-backed run_monitor 协议。"""
    try:
        from tools.exec import _MANAGER
    except Exception:
        _MANAGER = None

    if metabolic is None:
        metabolic = resolve_metabolic(task_store)

    runs = await task_store.list_runs(status="running", limit=20)
    if runs:
        _log.debug("[run-refresh] scanning running runs count=%d", len(runs))

    updates: list[dict[str, Any]] = []
    for run in runs:
        monitor = _run_monitor_config(run)
        if monitor is not None:
            if str(monitor.get("kind") or "") == "fact":
                updates.append(await _refresh_run_via_fact_monitor(task_store, run, monitor, episodic=episodic, semantic=semantic, memory_cfg=memory_cfg, metabolic=metabolic))
            elif str(monitor.get("kind") or "") == "process":
                updates.append(await _refresh_run_via_process_monitor(task_store, run, monitor, manager=_MANAGER, episodic=episodic, semantic=semantic, memory_cfg=memory_cfg, metabolic=metabolic))
            else:
                updates.append(_run_update_summary(run))
            continue
        if run.worker_type != WORKER_EXEC or not run.session_id or _MANAGER is None:
            updates.append(_run_update_summary(run))
            continue
        updates.append(await _refresh_run_via_process_monitor(task_store, run, {"kind": "process", "session_id": run.session_id}, manager=_MANAGER, episodic=episodic, semantic=semantic, memory_cfg=memory_cfg, metabolic=metabolic))
    if updates:
        terminal = sum(1 for item in updates if str(item.get("status") or "") in {"succeeded", "failed", "cancelled"})
        running = sum(1 for item in updates if str(item.get("status") or "") == "running")
        _log.debug(
            "[run-refresh] scanned=%d terminal=%d running=%d",
            len(updates),
            terminal,
            running,
        )
    return updates
