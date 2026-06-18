"""语义记忆生命周期提交器：让长期语义记忆写入经过代谢器官。"""
from __future__ import annotations

from typing import Any

from core.metabolic.fact_lifecycle import resolve_metabolic
from core.metabolic.lifecycle_utils import build_proposal


def _semantic_memory_value(
    *,
    node_id: str,
    kind: str,
    title: str,
    body: str,
    activation: float,
    valence: float,
    importance: float,
    tags: list[str] | None,
    created_at: str,
    source: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": kind,
        "title": title,
        "body": body,
        "activation": activation,
        "valence": valence,
        "importance": importance,
        "tags": tags or [],
        "created_at": created_at,
        "source": source,
    }


async def add_semantic_memory(
    owner: Any | None = None,
    *,
    node_id: str,
    kind: str,
    title: str,
    body: str,
    activation: float = 0.5,
    valence: float = 0.5,
    importance: float = 0.0,
    tags: list[str] | None = None,
    created_at: str = "",
    source: str,
    run_id: int = 0,
    decision_basis: str = "",
    task_store: Any | None = None,
    semantic_memory: Any | None = None,
) -> str:
    """经代谢器官写入长期语义记忆，并返回 node_id。

    owner、task_store 与 semantic_memory 任选其一可提供上下文：
    - 直接提供 owner：要求其可解析出 MetabolicEngine
    - 仅提供 task_store：同时需提供 semantic_memory 才能创建代谢器官
    """
    metabolic = resolve_metabolic(owner, task_store=task_store, semantic_memory=semantic_memory)
    if metabolic is None:
        raise RuntimeError("metabolic semantic memory write requires task_store and semantic memory")
    result = await metabolic.submit(
        build_proposal(
            op="add_semantic_memory",
            key=node_id,
            value=_semantic_memory_value(
                node_id=node_id,
                kind=kind,
                title=title,
                body=body,
                activation=activation,
                valence=valence,
                importance=importance,
                tags=tags,
                created_at=created_at,
                source=source,
            ),
            scope="semantic",
            source=source,
            run_id=run_id,
            decision_basis=decision_basis,
        )
    )
    return str(result or node_id)
