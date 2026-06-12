from __future__ import annotations

from typing import Any

from core.judgment.output import JudgmentOutput

_MISSING = object()


def build_workbench_action(
    *,
    workbench: dict[str, Any],
    rationale: str,
    source_action: Any | None = None,
    next_step: Any = _MISSING,
    reflection: str | None = None,
    model_strategy: dict[str, Any] | None = None,
    applied_skills: list[str] | None = None,
) -> JudgmentOutput:
    if source_action is not None:
        reflection = getattr(source_action, "reflection", reflection)
        model_strategy = model_strategy or getattr(source_action, "model_strategy", None)
        applied_skills = applied_skills or getattr(source_action, "applied_skills", None)
    return JudgmentOutput(
        decision="act",
        chosen_action_id="task.workbench",
        params={"workbench": workbench},
        rationale=rationale,
        reflection=reflection,
        next_step=str(workbench.get("next_verification") or "") if next_step is _MISSING else next_step,
        model_strategy=dict(model_strategy or {}),
        applied_skills=list(applied_skills or []),
    )
