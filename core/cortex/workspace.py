"""Task-level cortex workspace.

The short WM is intentionally aggressive; this module builds a durable task
workspace from task state, plan, recent runs, facts and failures so judgment
does not rely on a cramped recency-only context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CortexWorkspace:
    task_id: int = 0
    title: str = ""
    goal: str = ""
    status: str = ""
    current_step: str = ""
    next_step: str = ""
    domain: str = ""
    intent: str = ""
    hypothesis: str = ""
    recovery_state: str = ""
    next_verification: str = ""
    plan: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    completion_checks: list[str] = field(default_factory=list)
    action_first_intent: str = ""
    action_first_must_act: bool = False
    action_first_markers: list[str] = field(default_factory=list)
    minimum_next_action: str = ""
    captured_inputs: list[str] = field(default_factory=list)
    runtime_phase: str = ""
    runtime_last_status: str = ""
    runtime_failure_streak: int = 0


def _clip_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def _clip_for_context(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split())
    return _clip_text(value, max_chars)


def _as_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item.get("summary") or item.get("step") or "").strip()
            status = str(item.get("status") or "").strip()
            if text and status:
                text = f"[{status}] {text}"
        else:
            text = str(item or "").strip()
        if text:
            result.append(_clip_for_context(text, 180))
        if len(result) >= limit:
            break
    return result


def _text_from_mapping(item: dict[str, Any]) -> str:
    text = str(item.get("text") or item.get("content") or item.get("summary") or item.get("step") or item.get("name") or "").strip()
    if not text:
        ordered_keys = ("target", "action", "command", "tool", "result", "error")
        parts = [f"{key}={item[key]}" for key in ordered_keys if str(item.get(key) or "").strip()]
        text = " ".join(parts)
    status = str(item.get("status") or "").strip()
    if text and status:
        text = f"[{status}] {text}"
    return text


def _structured_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = _text_from_mapping(item)
        else:
            text = str(item or "").strip()
        if text:
            result.append(_clip_for_context(text, 220))
        if len(result) >= limit:
            break
    return result


def _text_field(data: dict[str, Any], *names: str) -> str:
    for name in names:
        value = str(data.get(name) or "").strip()
        if value:
            return _clip_for_context(value, 240)
    return ""


def _captured_inputs(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            kind = str(item.get("kind") or "input").strip()
            raw_value = str(item.get("value") or "").strip()
            text = f"{kind}={raw_value}" if raw_value else ""
        else:
            text = str(item or "").strip()
        if text:
            result.append(_clip_for_context(text, 260))
        if len(result) >= limit:
            break
    return result


def _plan_from_task(task: Any) -> list[str]:
    raw_plan = task.extras.get("plan") if isinstance(getattr(task, "extras", None), dict) else None
    if not isinstance(raw_plan, list):
        return []
    result: list[str] = []
    for index, item in enumerate(raw_plan, 1):
        if not isinstance(item, dict):
            continue
        step = str(item.get("step") or "").strip()
        if not step:
            continue
        status = str(item.get("status") or "pending").strip()
        result.append(f"{index}. [{status}] {_clip_for_context(step, 140)}")
        if len(result) >= 8:
            break
    return result


def _progress_from_runs(recent_runs: list[Any], *, limit: int = 5) -> list[str]:
    result: list[str] = []
    for run in recent_runs[:limit]:
        status = str(getattr(run, "status", "") or "").strip()
        tool = str(getattr(run, "tool_name", "") or getattr(run, "run_type", "") or "").strip()
        progress = str(getattr(run, "progress", "") or "").strip()
        summary = ""
        output_json = getattr(run, "output_json", {}) or {}
        if isinstance(output_json, dict):
            summary = str(output_json.get("summary") or output_json.get("result") or "").strip()
        if not summary:
            summary = str(getattr(run, "log_text", "") or "").strip()
        text = f"run#{getattr(run, 'id', '?')} [{status}] {tool or '-'}"
        detail = progress or summary
        if detail:
            text += f": {_clip_for_context(detail, 160)}"
        result.append(text)
    return result


def _facts_as_evidence(context_facts: list[Any], *, limit: int = 6) -> list[str]:
    result: list[str] = []
    for item in context_facts[:limit]:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        key, value = item[0], item[1]
        result.append(f"{key}: {_clip_for_context(str(value or ''), 160)}")
    return result


def _failure_lines(failures: list[Any], *, limit: int = 4) -> list[str]:
    result: list[str] = []
    for failure in failures[:limit]:
        kind = str(getattr(failure, "kind", "") or "").strip()
        summary = str(getattr(failure, "summary", "") or "").strip()
        context = str(getattr(failure, "context", "") or "").strip()
        text = kind or "failure"
        if summary or context:
            text += f": {_clip_for_context(summary or context, 160)}"
        result.append(text)
    return result


def build_cortex_workspace(
    *,
    task: Any | None,
    recent_runs: list[Any] | None = None,
    context_facts: list[Any] | None = None,
    failures: list[Any] | None = None,
) -> CortexWorkspace:
    if task is None:
        return CortexWorkspace()
    result_json = getattr(task, "result_json", {}) or {}
    cortex = result_json.get("cortex") if isinstance(result_json, dict) else None
    if not isinstance(cortex, dict):
        cortex = {}
    problem = result_json.get("problem_solving") if isinstance(result_json, dict) else None
    if isinstance(problem, dict):
        merged = dict(problem)
        merged.update(cortex)
        cortex = merged
    evidence = _as_list(cortex.get("evidence"), limit=8)
    evidence.extend(_facts_as_evidence(context_facts or [], limit=max(0, 8 - len(evidence))))
    action_first = cortex.get("action_first") if isinstance(cortex.get("action_first"), dict) else {}
    problem_runtime = cortex.get("problem_runtime") if isinstance(cortex.get("problem_runtime"), dict) else {}
    return CortexWorkspace(
        task_id=int(getattr(task, "id", 0) or 0),
        title=str(getattr(task, "title", "") or "").strip(),
        goal=str(getattr(task, "goal", "") or "").strip(),
        status=str(getattr(task, "status", "") or "").strip(),
        current_step=str(getattr(task, "current_step", "") or "").strip(),
        next_step=str(getattr(task, "next_step", "") or "").strip(),
        domain=_text_field(cortex, "domain", "active_domain"),
        intent=_text_field(cortex, "intent"),
        hypothesis=_text_field(cortex, "hypothesis", "working_hypothesis"),
        recovery_state=_text_field(cortex, "recovery_state", "state"),
        next_verification=_text_field(cortex, "next_verification", "next_experiment", "verification"),
        plan=_as_list(cortex.get("plan"), limit=8) or _plan_from_task(task),
        capabilities=_structured_list(cortex.get("capabilities"), limit=6),
        experiments=_structured_list(cortex.get("experiments"), limit=8),
        evidence=evidence[:8],
        progress=_as_list(cortex.get("progress"), limit=6) or _progress_from_runs(recent_runs or []),
        failures=_as_list(cortex.get("failures"), limit=4) or _failure_lines(failures or []),
        open_questions=_as_list(cortex.get("open_questions"), limit=5),
        completion_checks=_structured_list(cortex.get("completion_checks"), limit=6),
        action_first_intent=_text_field(action_first, "intent"),
        action_first_must_act=bool(action_first.get("must_act")),
        action_first_markers=_as_list(action_first.get("markers"), limit=6),
        minimum_next_action=_text_field(action_first, "minimum_next_action"),
        captured_inputs=_captured_inputs(cortex.get("captured_inputs"), limit=8),
        runtime_phase=_text_field(problem_runtime, "phase"),
        runtime_last_status=_text_field(problem_runtime, "last_status"),
        runtime_failure_streak=int(problem_runtime.get("failure_streak") or 0),
    )


def _section(title: str, lines: list[str]) -> list[str]:
    if not lines:
        return []
    return [title, *[f"- {line}" for line in lines]]


def format_cortex_workspace(workspace: CortexWorkspace) -> str:
    if workspace.task_id <= 0:
        return "（无活跃任务级皮层工作区）"
    lines = [
        f"task_id={workspace.task_id} status={workspace.status or 'unknown'}",
        f"title={_clip_text(workspace.title, 160) or '（未命名）'}",
        f"goal={_clip_text(workspace.goal, 220) or '（未指定）'}",
        f"current_step={_clip_text(workspace.current_step, 180) or '（未指定）'}",
        f"next_step={_clip_text(workspace.next_step, 180) or '（未指定）'}",
    ]
    if workspace.domain or workspace.intent or workspace.hypothesis or workspace.recovery_state or workspace.next_verification:
        lines.extend([
            "problem_solving:",
            f"- domain={workspace.domain or '（未识别）'}",
            f"- intent={workspace.intent or '（未识别）'}",
            f"- hypothesis={workspace.hypothesis or '（未建立）'}",
            f"- recovery_state={workspace.recovery_state or '（未进入恢复状态）'}",
            f"- next_verification={workspace.next_verification or '（未指定）'}",
        ])
    if workspace.runtime_phase or workspace.runtime_last_status or workspace.runtime_failure_streak:
        lines.extend([
            "problem_runtime:",
            f"- phase={workspace.runtime_phase or 'unknown'}",
            f"- last_status={workspace.runtime_last_status or 'unknown'}",
            f"- failure_streak={workspace.runtime_failure_streak}",
        ])
    if (
        workspace.action_first_intent
        or workspace.action_first_must_act
        or workspace.minimum_next_action
        or workspace.captured_inputs
    ):
        lines.extend([
            "action_first:",
            f"- intent={workspace.action_first_intent or 'unknown'}",
            f"- must_act={'yes' if workspace.action_first_must_act else 'no'}",
        ])
        if workspace.action_first_markers:
            lines.append(f"- markers={', '.join(workspace.action_first_markers)}")
        if workspace.minimum_next_action:
            lines.append(f"- minimum_next_action={workspace.minimum_next_action}")
    lines.extend(_section("plan_state:", workspace.plan))
    lines.extend(_section("captured_inputs:", workspace.captured_inputs))
    lines.extend(_section("capability_map:", workspace.capabilities))
    lines.extend(_section("experiment_log:", workspace.experiments))
    lines.extend(_section("evidence_board:", workspace.evidence))
    lines.extend(_section("recent_progress:", workspace.progress))
    lines.extend(_section("known_failures:", workspace.failures))
    lines.extend(_section("open_questions:", workspace.open_questions))
    lines.extend(_section("completion_checks:", workspace.completion_checks))
    return "\n".join(lines)
