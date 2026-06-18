"""代谢生命周期工具：统一重复的提案构造与提交行为。"""
from __future__ import annotations

from typing import Any

from core.metabolic.proposal import StateProposal

_DECISION_BASIS_DEFAULT_LIMIT = 240


def decision_extras(decision_basis: str) -> dict[str, str]:
    """为提案统一写入决策依据字段。"""
    return {"decision_basis": decision_basis} if decision_basis else {}


def decision_basis_from_parts(*parts: Any, fallback: str = "", limit: int = _DECISION_BASIS_DEFAULT_LIMIT) -> str:
    """将多个片段压平并归一化为单行审计摘要。

    统一规则：
    - 过滤空白片段；
    - 用 `|` 拼接关键语义段；
    - fallback 兜底；
    - 最后压成单空格并按 limit 截断。
    """
    text = " | ".join(str(part).strip() for part in parts if str(part or "").strip())
    if not text and fallback:
        text = str(fallback)
    return " ".join(text.split())[:limit]


def _decision_basis_from_parts(*parts: Any, fallback: str = "", limit: int = _DECISION_BASIS_DEFAULT_LIMIT) -> str:
    """兼容旧入口的内部实现别名（用于上下文/路由等跨层复用）。"""
    return decision_basis_from_parts(*parts, fallback=fallback, limit=limit)


def require_metabolic(owner: Any, action: str) -> Any:
    """解析代谢器官，不存在时抛出统一错误。"""
    from core.metabolic.fact_lifecycle import resolve_metabolic

    metabolic = resolve_metabolic(owner)
    if metabolic is None:
        raise RuntimeError(f"metabolic {action} requires a task store")
    return metabolic


def build_proposal(
    *,
    op: str,
    key: str,
    value: Any,
    scope: str,
    source: str,
    run_id: int = 0,
    decision_basis: str = "",
) -> StateProposal:
    """构造标准 StateProposal。"""
    return StateProposal(
        op=op,
        key=key,
        value=value,
        scope=scope,
        source=source,
        run_id=run_id,
        extras=decision_extras(decision_basis),
    )


async def submit_proposal(owner: Any, action: str, proposal: StateProposal) -> Any:
    metabolic = require_metabolic(owner, action)
    return await metabolic.submit(proposal)
