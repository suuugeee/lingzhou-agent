"""Generic problem-solving guard for judgment context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workspace import CortexWorkspace

_CORRECTION_MARKERS = (
    "不是",
    "不对",
    "错了",
    "搞错",
    "我指的是",
    "你误解",
    "不是这个意思",
    "我说的是",
)
_DIAGNOSTIC_MARKERS = (
    "为什么",
    "排查",
    "诊断",
    "失败",
    "报错",
    "不行",
    "不能",
    "解决",
    "修复",
    "验证",
    "测试",
    "继续",
)


@dataclass(frozen=True)
class ProblemSolvingGuard:
    active: bool = False
    signals: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    required_next_action: str = ""
    rationale: str = ""


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _recent_run_failed(recent_runs: list[Any]) -> bool:
    for run in recent_runs[:4]:
        status = str(getattr(run, "status", "") or "").lower()
        error_text = str(getattr(run, "error_text", "") or "").strip()
        if status in {"failed", "error"} or error_text:
            return True
    return False


def _missing_workbench_fields(workspace: CortexWorkspace) -> list[str]:
    missing: list[str] = []
    required_text_fields = {
        "domain": workspace.domain,
        "intent": workspace.intent,
        "hypothesis": workspace.hypothesis,
        "next_verification": workspace.next_verification,
    }
    for name, value in required_text_fields.items():
        if not str(value or "").strip():
            missing.append(name)
    if not workspace.capabilities:
        missing.append("capabilities")
    if not workspace.experiments and not workspace.evidence:
        missing.append("experiments_or_evidence")
    if not workspace.completion_checks:
        missing.append("completion_checks")
    return missing


def _blocking_missing_workbench_fields(workspace: CortexWorkspace) -> list[str]:
    """Fields that are missing enough to pause action and rebuild the workbench."""
    missing = _missing_workbench_fields(workspace)
    blockers = [field for field in missing if field in {"domain", "intent", "hypothesis", "next_verification"}]
    has_evidence = bool(workspace.experiments or workspace.evidence or workspace.progress or workspace.failures)
    if "experiments_or_evidence" in missing and not has_evidence:
        blockers.append("experiments_or_evidence")
    return blockers


def build_problem_solving_guard(
    *,
    task: Any | None,
    workspace: CortexWorkspace,
    user_message: str = "",
    failures: list[Any] | None = None,
    recent_runs: list[Any] | None = None,
) -> ProblemSolvingGuard:
    if task is None or workspace.task_id <= 0:
        return ProblemSolvingGuard(rationale="无活跃任务")

    message = str(user_message or "")
    signals: list[str] = []
    if _has_any_marker(message, _CORRECTION_MARKERS):
        signals.append("user_correction")
    if _has_any_marker(message, _DIAGNOSTIC_MARKERS):
        signals.append("diagnostic_or_repair_intent")
    if failures:
        signals.append("visible_failures")
    if _recent_run_failed(recent_runs or []):
        signals.append("recent_run_failed")
    if workspace.action_first_must_act:
        signals.append("action_first_required")

    missing = _missing_workbench_fields(workspace)
    blocking_missing = _blocking_missing_workbench_fields(workspace)
    has_existing_workbench = bool(
        workspace.domain
        or workspace.intent
        or workspace.hypothesis
        or workspace.capabilities
        or workspace.experiments
        or workspace.completion_checks
    )
    complex_task = bool(
        str(getattr(task, "current_step", "") or "").strip()
        or str(getattr(task, "next_step", "") or "").strip()
        or workspace.plan
        or has_existing_workbench
    )
    if complex_task and blocking_missing:
        signals.append("workbench_incomplete")

    active = bool(signals and blocking_missing)
    if not active:
        return ProblemSolvingGuard(
            active=False,
            signals=signals,
            missing_fields=missing,
            rationale="工作台足以支撑当前问题解决循环" if signals else "未触发通用问题解决守卫",
        )

    if "user_correction" in signals:
        required = (
            "若用户纠正改变了任务定义，先调用 task.amend；随后调用 task.workbench "
            "重写阻断字段 domain/intent/hypothesis/next_verification，并补入当前证据。"
        )
    else:
        required = (
            "优先调用 task.workbench 补齐阻断字段 domain/intent/hypothesis/"
            "experiments_or_evidence/next_verification，再继续执行或回复；"
            "capabilities/completion_checks 可在收尾前补强。"
        )
    return ProblemSolvingGuard(
        active=True,
        signals=signals,
        missing_fields=missing,
        required_next_action=required,
        rationale="非平凡问题解决缺少结构化工作台，继续推进容易误解领域、重复失败或跨轮丢失承诺。",
    )


def format_problem_solving_guard(guard: ProblemSolvingGuard) -> str:
    if not guard.active:
        signal_text = ", ".join(guard.signals) if guard.signals else "none"
        missing_text = ", ".join(guard.missing_fields) if guard.missing_fields else "none"
        return f"guard=idle\nsignals={signal_text}\nmissing_fields={missing_text}\nrationale={guard.rationale}"
    return "\n".join([
        "guard=active",
        f"signals={', '.join(guard.signals)}",
        f"missing_fields={', '.join(guard.missing_fields)}",
        f"required_next_action={guard.required_next_action}",
        f"rationale={guard.rationale}",
    ])
