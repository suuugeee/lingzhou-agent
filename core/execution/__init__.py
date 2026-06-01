"""core/execution — 执行层（dispatch、finalize、worker 池）。"""
from __future__ import annotations

from core.contracts.execution import action_key_param
from core.execution.helpers import (
    _classify_durable_failure,
    _infer_run_profile,
    _load_durable_failure_policy,
    _normalize_tool_result_text_fields,
    _run_progress_text,
    _run_status_from_result,
    _tool_result_log_fields,
    build_meta_reflection,
    finalize_run,
    record_meta_reflection_memory,
    record_run_outcome_memory,
)
from core.execution.layer import ExecutionLayer
from core.execution.workers import WorkerLayer

__all__ = [
    "ExecutionLayer",
    "WorkerLayer",
    "_classify_durable_failure",
    "_infer_run_profile",
    "_load_durable_failure_policy",
    "_normalize_tool_result_text_fields",
    "_run_progress_text",
    "_run_status_from_result",
    "_tool_result_log_fields",
    "action_key_param",
    "build_meta_reflection",
    "finalize_run",
    "record_meta_reflection_memory",
    "record_run_outcome_memory",
]
