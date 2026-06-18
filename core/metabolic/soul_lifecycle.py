"""灵魂/人格 fact 提案提交器。

将人格层（``soul:*`` 命名空间）变更统一转到代谢器官，避免在业务逻辑层直接调用
``set_fact``，确保写入一致经过免疫检查 + 生命史账本。
"""
from __future__ import annotations

from typing import Any

from core.metabolic.fact_lifecycle import resolve_metabolic
from core.metabolic.lifecycle_utils import build_proposal


def _validate_soul_key(key: str) -> str:
    normalized = (key or "").strip()
    if not normalized:
        raise ValueError("soul fact key cannot be empty")
    if not normalized.startswith("soul:"):
        raise ValueError(f"soul fact key must start with 'soul:', got {normalized!r}")
    return normalized


async def set_soul_fact(
    owner: Any,
    *,
    key: str,
    value: Any,
    scope: str = "system",
    source: str,
    run_id: int = 0,
    decision_basis: str = "",
    task_store: Any | None = None,
) -> bool:
    """提交 soul fact 变更提案。owner 无法构建 metabolic 时返回 False。"""
    fact_key = _validate_soul_key(key)
    metabolic = resolve_metabolic(owner, task_store=task_store)
    if metabolic is None:
        return False

    await metabolic.submit(
        build_proposal(
            op="soul_change",
            key=fact_key,
            value=value,
            scope=scope,
            source=source,
            run_id=run_id,
            decision_basis=decision_basis,
        )
    )
    return True
