"""tools/workbench.py - generic task problem-solving workbench."""

from __future__ import annotations

from typing import Any

from tools.registry import (
    CAPS_EXEMPT,
    ToolContext,
    ToolManifest,
    ToolParam,
    ToolResult,
    tool,
    tool_metadata,
)

_TEXT_FIELDS = {
    "domain",
    "intent",
    "hypothesis",
    "working_hypothesis",
    "recovery_state",
    "next_verification",
}
_LIST_FIELDS = {
    "capabilities",
    "experiments",
    "evidence",
    "open_questions",
    "completion_checks",
    "progress",
    "failures",
}
_ALLOWED_FIELDS = _TEXT_FIELDS | _LIST_FIELDS


async def _resolve_task(task_id: Any, ctx: ToolContext) -> Any | None:
    if not ctx.task_store:
        return None
    if task_id:
        try:
            return await ctx.task_store.get_task_by_id(int(task_id))
        except Exception:
            return None
    if hasattr(ctx, "get_active_task"):
        try:
            return await ctx.get_active_task()
        except Exception:
            pass
    return getattr(ctx, "active_task", None)


def _clean_text(value: Any, *, max_chars: int = 300) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _clean_list(value: Any, *, max_items: int = 12) -> list[Any]:
    if not isinstance(value, list):
        return []
    cleaned: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            entry: dict[str, str] = {}
            for key, raw in item.items():
                text = _clean_text(raw)
                if text:
                    entry[str(key)] = text
            if entry:
                cleaned.append(entry)
        else:
            text = _clean_text(item)
            if text:
                cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _clean_workbench_patch(raw: Any) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        return {}, ["workbench 必须是对象"]
    patch: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in raw.items():
        key = str(key or "").strip()
        if key not in _ALLOWED_FIELDS:
            warnings.append(f"忽略未知字段: {key}")
            continue
        if key in _TEXT_FIELDS:
            text = _clean_text(value)
            if text:
                patch[key] = text
        else:
            items = _clean_list(value)
            if items:
                patch[key] = items
    return patch, warnings


def _extract_workbench_input(raw_params: Any) -> tuple[Any, list[str]]:
    if not isinstance(raw_params, dict):
        return raw_params, []
    workbench_value = raw_params.get("workbench")
    if workbench_value is not None:
        return workbench_value, []
    candidate = {
        key: value for key, value in raw_params.items() if key in _ALLOWED_FIELDS
    }
    if candidate:
        return candidate, ["已自动从顶层字段组装 workbench"]
    return None, []


@tool(ToolManifest(
    name="task.workbench",
    description=(
        "维护当前任务的通用问题解决工作台，写入 task.result_json.cortex。"
        "用于记录领域/意图、工作假设、能力发现、实验记录、证据、开放问题、恢复状态和完成检查。"
        "适合所有非平凡排查/实现任务，不局限于某个领域。"
    ),
    prefer_tier="reasoner",
    progress_category="mutation",
    capabilities=CAPS_EXEMPT,
    params=[
        ToolParam("workbench", "object", "工作台 patch，可含 domain/intent/hypothesis/capabilities/experiments/evidence/open_questions/recovery_state/next_verification/completion_checks", required=True),
        ToolParam("task_id", "number", "任务 ID（可选，默认当前活跃任务）", required=False),
    ],
))
async def task_workbench(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    task = await _resolve_task(params.get("task_id"), ctx)
    if not task:
        return ToolResult(summary="未找到任务。请先创建任务，或指定 task_id。", error="NoTask", skipped=True)
    workbench_input, auto_warnings = _extract_workbench_input(params)
    patch, warnings = _clean_workbench_patch(workbench_input)
    warnings = auto_warnings + warnings
    if not patch:
        reason = "; ".join(warnings) if warnings else "workbench 为空"
        return ToolResult(summary=f"工作台未更新: {reason}", error="InvalidWorkbench", skipped=True)

    result_json = dict(getattr(task, "result_json", {}) or {})
    cortex = result_json.get("cortex")
    if not isinstance(cortex, dict):
        cortex = {}
    merged_cortex = dict(cortex)
    merged_cortex.update(patch)
    await ctx.task_store.update_task_result(int(task.id), {"cortex": merged_cortex})

    field_names = ", ".join(sorted(patch))
    warning_text = f"；{'; '.join(warnings)}" if warnings else ""
    return ToolResult(
        summary=f"任务 #{task.id} 工作台已更新: {field_names}{warning_text}",
        evidence=f"task_id={task.id} fields={field_names}",
        resource_key=str(task.id),
        state_delta={"cortex": patch},
        metadata=tool_metadata(
            "task.workbench",
            f"task.workbench id={task.id} fields={field_names}",
            task_id=int(task.id),
            fields=sorted(patch),
        ),
    )
