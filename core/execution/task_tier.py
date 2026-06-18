"""core.execution.task_tier — 与 run_type/task_tier 决策相关的纯函数逻辑。"""
from __future__ import annotations

from collections.abc import Callable

from core.judgment.tiers import normalize_tier

TASK_DEFAULT_TIER = "task_default"


def resolve_task_model_tier(
    task_tier: str | None,
    run_type: str,
    run_type_routing: dict[str, str],
    resolve_default_tier_for_run_type: Callable[[str, dict[str, str] | None], str],
) -> str:
    """按规则返回最终模型档位。

    规则：
    1. 任务层已有有效显式档位时优先沿用。
    2. 任务层为空或 task_default 时，按 run_type 映射覆盖。
    3. 覆盖仍为 task_default 时，保留原值（为空或 task_default）。
    """
    normalized_task_tier = normalize_tier(task_tier, fallback="")
    if normalized_task_tier and normalized_task_tier != TASK_DEFAULT_TIER:
        return normalized_task_tier

    mapped = normalize_tier(
        resolve_default_tier_for_run_type(run_type, run_type_routing),
        fallback=TASK_DEFAULT_TIER,
    )
    if mapped != TASK_DEFAULT_TIER:
        return mapped
    return TASK_DEFAULT_TIER
