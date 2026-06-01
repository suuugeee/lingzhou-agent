"""core/contracts — 跨层稳定契约（类型与纯函数，无运行时编排逻辑）。"""
from core.contracts.execution import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_SUCCEEDED,
    action_key_param,
)
from core.contracts.probe import (
    PROBE_COVERAGE_HINTS,
    ProbeConfig,
    ProbeDataBack,
    ProbeKind,
    ProbeResult,
    normalize_probe_coverage_tags,
)
from core.contracts.tools import ToolMetadataContract, tool_metadata_contract

__all__ = [
    "PROBE_COVERAGE_HINTS",
    "ProbeConfig",
    "ProbeDataBack",
    "ProbeKind",
    "ProbeResult",
    "RUN_STATUS_CANCELLED",
    "RUN_STATUS_FAILED",
    "RUN_STATUS_RUNNING",
    "RUN_STATUS_SUCCEEDED",
    "ToolMetadataContract",
    "action_key_param",
    "normalize_probe_coverage_tags",
    "tool_metadata_contract",
]
