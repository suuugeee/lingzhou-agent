"""Judgment context formatters focused on task and fact related sections."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .utils import (
    _cache_put,
    _clip_text,
    _clip_for_context,
    _context_fmt_cache,
    _format_fact_value,
    _run_summary,
)

if TYPE_CHECKING:
    from store.task import Failure, Run, Task


def _task_narrative(task: Task | None) -> str:
    """从任务状态构建叙事线：目标 → 当前步骤 → 下一步。"""
    if not task:
        return "无"
    parts = []
    if task.goal:
        parts.append(f"目标: {task.goal}")
    if task.current_step:
        parts.append(f"进展: {task.current_step}")
    if task.next_step:
        parts.append(f"下一步: {task.next_step}")
    return " → ".join(parts) if parts else f"执行中 ({task.status})"


def _fmt_task(task: Task | None) -> str:
    if not task:
        return "（无活跃任务，可自主探索或等待）"
    age_str = ""
    if task.created_at:
        try:
            created = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            elapsed = datetime.now(UTC) - created
            total_secs = int(elapsed.total_seconds())
            if total_secs < 60:
                age_str = f"（已进行 {total_secs}s）"
            elif total_secs < 3600:
                age_str = f"（已进行 {total_secs // 60}m）"
            elif total_secs < 86400:
                hours, minutes = divmod(total_secs // 60, 60)
                age_str = f"（已进行 {hours}h {minutes}m）"
            else:
                days, rem = divmod(total_secs, 86400)
                age_str = f"（已进行 {days}d {rem // 3600}h）"
        except Exception:
            pass
    last_run_status = str((task.result_json or {}).get("last_run_status") or "").strip()
    lines = [
        f"ID: {task.id}",
        f"标题: {task.title}{age_str}",
        f"状态: {task.status}",
        f"目标: {task.goal or '（未指定）'}",
        f"优先级: {task.priority}",
        f"模型层级: {task.model_tier or '（未指定）'}",
        f"当前步骤: {task.current_step or '（未指定）'}",
        f"下一步: {task.next_step or '（未指定）'}",
        f"叙事线: {_task_narrative(task)}",
    ]
    raw_plan = task.extras.get("plan") if isinstance(task.extras, dict) else None
    if isinstance(raw_plan, list) and raw_plan:
        status_icons = {"completed": "✅", "in_progress": "🔄", "pending": "⏳"}
        plan_lines: list[str] = []
        for index, item in enumerate(raw_plan, 1):
            if not isinstance(item, dict):
                continue
            step = str(item.get("step") or "").strip()
            if not step:
                continue
            status = str(item.get("status") or "pending").strip()
            icon = status_icons.get(status, "•")
            plan_lines.append(f"  [{index}] {icon} {_clip_text(step, 80)}")
        if plan_lines:
            lines.append("当前计划:")
            lines.extend(plan_lines)
            in_progress_step = next(
                (
                    str(step_info.get("step") or "").strip()
                    for step_info in raw_plan
                    if isinstance(step_info, dict) and step_info.get("status") == "in_progress"
                ),
                None,
            )
            if in_progress_step:
                lines.append(
                    f"⚠️ 计划信号：步骤 [{in_progress_step}] 当前处于 in_progress。"
                    "若没有更强的新证据或 inbox 新消息，优先直接推进这一步，而不是重新 plan。"
                )
    inbox: list = task.extras.get("inbox_messages") or [] if isinstance(task.extras, dict) else []
    if isinstance(inbox, list) and inbox:
        lines.append(f"⚠️ 新增用户消息（inbox {len(inbox)} 条，先评估这些新消息是否改变当前方向）:")
        for index, msg in enumerate(inbox, 1):
            lines.append(f"  [{index}] {str(msg)}")
    if last_run_status:
        lines.append(f"最近运行状态: {last_run_status}")
    return "\n".join(lines)


def _fmt_recent_runs(runs: list[Run]) -> str:
    cache_key = f"_fmt_recent_runs:{hash(tuple(run.id for run in runs)) if runs else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not runs:
        result = "（暂无近期运行记录）"
        _cache_put(cache_key, result)
        return result
    lines: list[str] = []
    for run in runs:
        summary = _clip_for_context(_run_summary(run), 120)
        tool = run.tool_name or run.run_type or "-"
        progress = _clip_for_context(run.progress.strip(), 80) if run.progress else ""
        line = f"- run#{run.id} [{run.status}] tool={tool} tier={run.model_tier or '-'}"
        if progress:
            line += f" progress={progress}"
        if summary:
            line += f" summary={summary}"
        lines.append(line)
    result = "\n".join(lines)
    _cache_put(cache_key, result)
    return result


def _fmt_context_facts(facts: list[tuple[str, str]]) -> str:
    cache_key = f"_fmt_context_facts:{hash(tuple(facts)) if facts else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not facts:
        result = "（暂无近期关键事实）"
        _cache_put(cache_key, result)
        return result
    result = "\n".join(
        f"- {key} = {_format_fact_value(value)}"
        for key, value in facts
    )
    _cache_put(cache_key, result)
    return result


def _fmt_evolution_breakers(facts: list[tuple[str, str]]) -> str:
    """格式化 evolution breaker 运行时状态，供 LLM 感知当前熔断真相。"""
    if not facts:
        return "（无 active breaker）"

    now = datetime.now(UTC).timestamp()
    lines: list[str] = []
    for key, value in facts:
        try:
            payload = json.loads(value)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        cooldown_until = float(payload.get("cooldown_until", 0.0) or 0.0)
        remain = max(0, int(cooldown_until - now))
        streak = int(payload.get("failure_streak", 0) or 0)
        reason = str(payload.get("reason") or "").strip()
        target = str(payload.get("target") or payload.get("failure_target") or key.replace("evolution:breaker:", "")).strip()
        if remain > 0:
            line = f"- {target}: breaker=OPEN remain={remain}s streak={streak}"
        else:
            line = f"- {target}: breaker=CLOSED streak={streak}"
        if reason:
            line += f" reason={_clip_text(reason, 120)}"
        lines.append(line)

    return "\n".join(lines) if lines else "（无 active breaker）"


def _fmt_waiting_tasks(tasks: list[Task]) -> str:
    cache_key = f"_fmt_waiting_tasks:{hash(tuple(task.id for task in tasks)) if tasks else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not tasks:
        result = "（无 waiting 任务）"
        _cache_put(cache_key, result)
        return result
    lines: list[str] = []
    for task in tasks:
        wait_desc = task.wait_kind or "unknown"
        if task.wait_key:
            wait_desc += f"/{task.wait_key}"
        line = f"- task#{task.id} [{task.status}] {task.title} wait={wait_desc}"
        if task.next_step:
            line += f" next={_clip_text(task.next_step, 80)}"
        lines.append(line)
    result = "\n".join(lines)
    _cache_put(cache_key, result)
    return result


def _fmt_runnable_tasks(tasks: list[Task], active_task_id: int | None = None) -> str:
    cache_key = (
        f"_fmt_runnable_tasks:{hash(tuple((task.id, task.status) for task in tasks)) if tasks else 'none'}:"
        f"{active_task_id or 0}"
    )
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    visible_tasks = [task for task in tasks if active_task_id is None or task.id != active_task_id]
    if not visible_tasks:
        result = "（无其他 runnable 任务）"
        _cache_put(cache_key, result)
        return result
    lines: list[str] = []
    for task in visible_tasks:
        line = f"- task#{task.id} [{task.status}/{task.priority}] {task.title}"
        if task.next_step:
            line += f" next={_clip_text(task.next_step, 80)}"
        elif task.goal:
            line += f" goal={_clip_text(task.goal, 80)}"
        lines.append(line)
    result = "\n".join(lines)
    _cache_put(cache_key, result)
    return result


def _fmt_similar_tasks(items: list[tuple[Task, float]]) -> str:
    cache_key = (
        f"_fmt_similar_tasks:{hash(tuple((task.id, round(score, 3)) for task, score in items)) if items else 'none'}"
    )
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not items:
        result = "（未发现相似开放任务）"
        _cache_put(cache_key, result)
        return result
    lines: list[str] = []
    for task, score in items:
        line = f"- {round(score * 100)}% task#{task.id} [{task.status}] {task.title}"
        if task.next_step:
            line += f" next={_clip_text(task.next_step, 80)}"
        elif task.goal:
            line += f" goal={_clip_text(task.goal, 80)}"
        lines.append(line)
    result = "\n".join(lines)
    _cache_put(cache_key, result)
    return result


def _fmt_failures(failures: list[Failure]) -> str:
    if not failures:
        return "（无近期失败）"
    lines = [f"- [#{failure.id}][{failure.kind}] {failure.summary}" for failure in failures]
    return "\n".join(lines)


def _fmt_durable_failures(snapshot: dict[str, Any]) -> str:
    threshold = int(snapshot.get("threshold") or 0)
    ttl_sec = int(snapshot.get("ttl_sec") or 0)
    lines = [f"policy: threshold={threshold} ttl_sec={ttl_sec}"]
    muted_actions = snapshot.get("muted_actions") or []
    if not muted_actions:
        lines.append("- 当前无稳定失败静默中的动作")
        return "\n".join(lines)
    for item in muted_actions:
        tool = item.get("tool") or "-"
        key = item.get("key") or ""
        reason = item.get("reason") or "stable_failure"
        count = int(item.get("count") or 0)
        remaining_sec = int(item.get("remaining_sec") or 0)
        line = f"- {tool}"
        if key:
            line += f" {key}"
        line += f" reason={reason} failures={count} remaining={remaining_sec}s"
        lines.append(line)
    return "\n".join(lines)


__all__ = [
    "_task_narrative",
    "_fmt_task",
    "_fmt_recent_runs",
    "_fmt_context_facts",
    "_fmt_evolution_breakers",
    "_fmt_waiting_tasks",
    "_fmt_runnable_tasks",
    "_fmt_similar_tasks",
    "_fmt_failures",
    "_fmt_durable_failures",
]
