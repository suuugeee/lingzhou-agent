"""执行层跨层契约 — 纯函数，无 loop/judgment 编排依赖。"""
from __future__ import annotations

from typing import Any

# Run 收尾状态（与 execution.helpers._run_status_from_result 对齐）
RUN_STATUS_RUNNING = "running"
RUN_STATUS_SUCCEEDED = "succeeded"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"

_ACTION_KEY_PRIMARY_FIELDS = ("path", "name", "title", "key")


def action_key_param(params: dict[str, Any] | None) -> str:
    """从工具 params 提取用于失败降噪 / 动作指纹的资源键片段。"""
    p = params or {}
    for field in _ACTION_KEY_PRIMARY_FIELDS:
        value = p.get(field)
        if value:
            return value
    return (
        str(p.get("id") or "")
        or p.get("command")
        or p.get("query")
        or ""
    )
