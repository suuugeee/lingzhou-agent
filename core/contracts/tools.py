"""工具结果 metadata 契约（与 tools.registry.tool_metadata 对齐）。"""
from __future__ import annotations

from typing import Any, TypedDict


class ToolMetadataContract(TypedDict, total=False):
    """ToolResult.metadata 推荐字段；tools 经 tool_metadata() 构造。"""

    tool_name: str
    log_summary: str
    run_id: int
    worker_type: str
    session_id: str


def tool_metadata_contract(
    tool_name: str,
    log_summary: str,
    **extra: Any,
) -> dict[str, Any]:
    """与 tools.registry.tool_metadata 同形；供 core 侧文档化与测试断言。"""
    meta: dict[str, Any] = {"tool_name": tool_name, "log_summary": log_summary}
    meta.update(extra)
    return meta
