"""core.judgment - 稳定 façade，统一导出 judgment 包的公开 API。"""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "CognitionFrame": ("core.judgment.frame", "CognitionFrame"),
    "JudgmentContextAssembler": ("core.judgment.assembler", "JudgmentContextAssembler"),
    "JudgmentExecutor": ("core.judgment.executor", "JudgmentExecutor"),
    "JudgmentLayer": ("core.judgment.runtime", "JudgmentLayer"),
    "JudgmentOutput": ("core.judgment.output", "JudgmentOutput"),
    "ModelHealth": ("core.judgment.output", "ModelHealth"),
    "ModelSelection": ("core.judgment.output", "ModelSelection"),
    "apply_context_budget": ("core.judgment.context.budget", "apply_context_budget"),
    "tool_tier": ("core.judgment.output", "tool_tier"),
}

__all__ = [
    "CognitionFrame",
    "JudgmentContextAssembler",
    "JudgmentExecutor",
    "JudgmentLayer",
    "JudgmentOutput",
    "ModelHealth",
    "ModelSelection",
    "apply_context_budget",
    "tool_tier",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
