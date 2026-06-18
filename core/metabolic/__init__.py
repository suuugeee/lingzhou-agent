"""core.metabolic — metabolic public API facade."""
from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "MetabolicEngine": ("core.metabolic.engine", "MetabolicEngine"),
    "StateProposal": ("core.metabolic.proposal", "StateProposal"),
    "StateWriteResult": ("core.metabolic.state_writer", "StateWriteResult"),
    "add_run": ("core.metabolic.run_lifecycle", "add_run"),
    "add_semantic_memory": ("core.metabolic.semantic_lifecycle", "add_semantic_memory"),
    "amend_task": ("core.metabolic.task_lifecycle", "amend_task"),
    "apply_state_write": ("core.metabolic.state_writer", "apply_state_write"),
    "create_task": ("core.metabolic.task_lifecycle", "create_task"),
    "delete_fact": ("core.metabolic.fact_lifecycle", "delete_fact"),
    "mark_task_waiting": ("core.metabolic.task_lifecycle", "mark_task_waiting"),
    "resolve_metabolic": ("core.metabolic.fact_lifecycle", "resolve_metabolic"),
    "resume_task": ("core.metabolic.task_lifecycle", "resume_task"),
    "set_soul_fact": ("core.metabolic.soul_lifecycle", "set_soul_fact"),
    "submit_fact": ("core.metabolic.fact_lifecycle", "submit_fact"),
    "update_run": ("core.metabolic.run_lifecycle", "update_run"),
    "update_task_data": ("core.metabolic.task_lifecycle", "update_task_data"),
    "update_task_result": ("core.metabolic.task_lifecycle", "update_task_result"),
    "update_task_status": ("core.metabolic.task_lifecycle", "update_task_status"),
}

__all__ = [
    "MetabolicEngine",
    "StateProposal",
    "StateWriteResult",
    "add_semantic_memory",
    "amend_task",
    "add_run",
    "apply_state_write",
    "create_task",
    "delete_fact",
    "mark_task_waiting",
    "resolve_metabolic",
    "resume_task",
    "set_soul_fact",
    "submit_fact",
    "update_run",
    "update_task_data",
    "update_task_result",
    "update_task_status",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
