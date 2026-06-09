"""Automatic cortex workbench updates from run outcomes."""

from __future__ import annotations

import json
from typing import Any

_MAX_ITEMS = {
    "capabilities": 8,
    "experiments": 10,
    "evidence": 10,
    "failures": 8,
    "progress": 8,
}
_SKIP_AUTO_TOOLS = {"task.workbench"}


def _clip_text(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_json(value: Any, *, limit: int = 220) -> str:
    if value in (None, "", {}, []):
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return _clip_text(text, limit=limit)


def _item_key(item: Any) -> str:
    if isinstance(item, dict):
        try:
            return json.dumps(item, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(sorted(item.items()))
    return str(item)


def _merge_list(existing: Any, incoming: list[Any], *, limit: int) -> list[Any]:
    base = existing if isinstance(existing, list) else []
    merged: list[Any] = []
    seen: set[str] = set()
    for item in [*incoming, *base]:
        key = _item_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _meaningful_detail(*values: Any) -> str:
    for value in values:
        text = _clip_text(value)
        if text:
            return text
    return ""


def _should_auto_record(tool_name: str, status: str, *, summary: str, error: str, evidence: str, state_delta: dict[str, Any], artifact_paths: list[str]) -> bool:
    if tool_name in _SKIP_AUTO_TOOLS:
        return False
    if status in {"failed", "cancelled"} or error:
        return True
    if evidence or state_delta or artifact_paths:
        return True
    return bool(summary and tool_name)


def build_auto_cortex_patch(
    *,
    existing_cortex: dict[str, Any] | None,
    run_id: int,
    task_id: int,
    tool_name: str,
    status: str,
    summary: str = "",
    error: str | None = None,
    evidence: str = "",
    progress: str = "",
    state_delta: dict[str, Any] | None = None,
    artifact_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Return ``{"cortex": merged}`` for a run outcome, or ``{}`` when irrelevant.

    The patch is intentionally compact and append-only for list fields. It does
    not infer domain/intent/hypothesis, because those belong to the judgment
    layer or explicit ``task.workbench`` updates.
    """
    state = state_delta if isinstance(state_delta, dict) else {}
    artifacts = list(artifact_paths or [])
    tool = _clip_text(tool_name or "unknown", limit=80)
    run_status = _clip_text(status or "unknown", limit=40)
    err = _clip_text(error or "", limit=220)
    detail = _meaningful_detail(progress, evidence, summary, err, _compact_json(state), ", ".join(artifacts))
    if not _should_auto_record(tool, run_status, summary=summary, error=err, evidence=evidence, state_delta=state, artifact_paths=artifacts):
        return {}

    cortex = dict(existing_cortex or {})
    experiment: dict[str, str] = {
        "run_id": str(int(run_id)),
        "tool": tool,
        "status": run_status,
    }
    if detail:
        experiment["summary"] = detail
    cortex["experiments"] = _merge_list(
        cortex.get("experiments"),
        [experiment],
        limit=_MAX_ITEMS["experiments"],
    )
    runtime = dict(cortex.get("problem_runtime") if isinstance(cortex.get("problem_runtime"), dict) else {})
    runtime["last_run_id"] = str(int(run_id))
    runtime["last_tool"] = tool
    runtime["last_status"] = run_status

    if run_status == "succeeded":
        if not tool.startswith("task."):
            runtime["phase"] = "verification_collected"
            runtime["last_success_run_id"] = str(int(run_id))
            runtime["failure_streak"] = 0
            action_first = dict(cortex.get("action_first") if isinstance(cortex.get("action_first"), dict) else {})
            if action_first:
                action_first["last_verifiable_action_run_id"] = str(int(run_id))
                action_first["last_verifiable_action_tool"] = tool
                cortex["action_first"] = action_first
        capability = {"name": f"{tool} 可用", "status": "available"}
        cortex["capabilities"] = _merge_list(
            cortex.get("capabilities"),
            [capability],
            limit=_MAX_ITEMS["capabilities"],
        )
        if detail:
            cortex["evidence"] = _merge_list(
                cortex.get("evidence"),
                [f"run#{int(run_id)} {tool} succeeded: {detail}"],
                limit=_MAX_ITEMS["evidence"],
            )
        progress_line = f"run#{int(run_id)} [{run_status}] {tool}"
        if detail:
            progress_line += f": {detail}"
        cortex["progress"] = _merge_list(
            cortex.get("progress"),
            [progress_line],
            limit=_MAX_ITEMS["progress"],
        )
    elif run_status in {"failed", "cancelled"} or err:
        runtime["phase"] = "recovering"
        runtime["failure_streak"] = int(runtime.get("failure_streak") or 0) + 1
        runtime["last_failure_run_id"] = str(int(run_id))
        failure_detail = _meaningful_detail(err, summary, evidence, progress)
        failure_line = f"run#{int(run_id)} {tool} {run_status}"
        if failure_detail:
            failure_line += f": {failure_detail}"
        cortex["failures"] = _merge_list(
            cortex.get("failures"),
            [failure_line],
            limit=_MAX_ITEMS["failures"],
        )
        cortex["recovery_state"] = "recovering_from_run_failure"
        if not str(cortex.get("next_verification") or "").strip():
            cortex["next_verification"] = f"修正 {tool} 的失败原因后，用不同证据路径验证任务 #{int(task_id)} 是否推进。"

    if runtime:
        cortex["problem_runtime"] = runtime
    return {"cortex": cortex}


async def build_auto_cortex_result_patch(
    *,
    task_store: Any,
    task_id: int,
    run_id: int,
    tool_name: str,
    status: str,
    summary: str = "",
    error: str | None = None,
    evidence: str = "",
    progress: str = "",
    state_delta: dict[str, Any] | None = None,
    artifact_paths: list[str] | None = None,
) -> dict[str, Any]:
    if not task_store or not task_id:
        return {}
    getter = getattr(task_store, "get_task_by_id", None)
    if getter is None:
        return {}
    try:
        task = await getter(int(task_id))
    except Exception:
        return {}
    if task is None:
        return {}
    result_json = getattr(task, "result_json", {}) or {}
    existing_cortex = result_json.get("cortex") if isinstance(result_json, dict) else None
    return build_auto_cortex_patch(
        existing_cortex=existing_cortex if isinstance(existing_cortex, dict) else {},
        run_id=run_id,
        task_id=task_id,
        tool_name=tool_name,
        status=status,
        summary=summary,
        error=error,
        evidence=evidence,
        progress=progress,
        state_delta=state_delta,
        artifact_paths=artifact_paths,
    )
