"""任务生命周期提交器：让任务创建、等待、恢复、修正都经过代谢器官。"""
from __future__ import annotations

from typing import Any

from core.metabolic.lifecycle_utils import build_proposal, submit_proposal


async def create_task(
    owner: Any,
    *,
    proposal_source: str,
    decision_basis: str = "",
    **data: Any,
) -> int:
    """经代谢器官创建任务，并返回新任务 id。"""
    task_id = await submit_proposal(
        owner=owner,
        action="creation",
        proposal=build_proposal(
            op="create_task",
            key="task:new",
            value=data,
            scope="task",
            source=proposal_source,
            decision_basis=decision_basis,
        ),
    )
    return int(task_id)


async def update_task_status(
    owner: Any,
    task_id: int,
    *,
    status: str,
    source: str,
    next_step: str | None = None,
    current_step: str | None = None,
    model_tier: str | None = None,
    result_json: dict[str, Any] | None = None,
    decision_basis: str = "",
) -> None:
    """经代谢器官更新任务状态和步骤。"""
    await submit_proposal(
        owner=owner,
        action="update",
        proposal=build_proposal(
            op="update_task_status",
            key=str(task_id),
            value={
                "status": status,
                "next_step": next_step,
                "current_step": current_step,
                "model_tier": model_tier,
                "result_json": result_json,
            },
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )


async def mark_task_waiting(
    owner: Any,
    task_id: int,
    *,
    wait_kind: str,
    source: str,
    wait_key: str = "",
    wait_json: dict[str, Any] | None = None,
    current_step: str | None = None,
    next_step: str | None = None,
    result_json: dict[str, Any] | None = None,
    decision_basis: str = "",
) -> None:
    """经代谢器官将任务切入 waiting。"""
    await submit_proposal(
        owner=owner,
        action="wait",
        proposal=build_proposal(
            op="mark_task_waiting",
            key=str(task_id),
            value={
                "wait_kind": wait_kind,
                "wait_key": wait_key,
                "wait_json": wait_json,
                "current_step": current_step,
                "next_step": next_step,
                "result_json": result_json,
            },
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )


async def resume_task(
    owner: Any,
    task_id: int,
    *,
    source: str,
    status: str = "resumed",
    current_step: str | None = None,
    next_step: str | None = None,
    result_json: dict[str, Any] | None = None,
    decision_basis: str = "",
) -> None:
    """经代谢器官恢复 waiting/blocked 任务。"""
    await submit_proposal(
        owner=owner,
        action="resume",
        proposal=build_proposal(
            op="resume_task",
            key=str(task_id),
            value={
                "status": status,
                "current_step": current_step,
                "next_step": next_step,
                "result_json": result_json,
            },
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )


async def update_task_data(
    owner: Any,
    task_id: int,
    data: dict[str, Any],
    *,
    source: str,
    decision_basis: str = "",
) -> None:
    await submit_proposal(
        owner=owner,
        action="data update",
        proposal=build_proposal(
            op="update_task_data",
            key=str(task_id),
            value=data,
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )


async def update_task_result(
    owner: Any,
    task_id: int,
    result_json: dict[str, Any],
    *,
    source: str,
    decision_basis: str = "",
) -> None:
    await submit_proposal(
        owner=owner,
        action="result update",
        proposal=build_proposal(
            op="update_task_result",
            key=str(task_id),
            value=result_json if isinstance(result_json, dict) else {"value": result_json},
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )


async def amend_task(
    owner: Any,
    task_id: int,
    *,
    source: str,
    title: str | None = None,
    goal: str | None = None,
    priority: str | None = None,
    amendment_reason: str,
    decision_basis: str = "",
) -> bool:
    result = await submit_proposal(
        owner=owner,
        action="amendment",
        proposal=build_proposal(
            op="amend_task",
            key=str(task_id),
            value={
                "title": title,
                "goal": goal,
                "priority": priority,
                "amendment_reason": amendment_reason,
            },
            scope="task",
            source=source,
            decision_basis=decision_basis,
        ),
    )
    return bool(result)
