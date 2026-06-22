"""执行层跨层契约 — 纯函数，无 loop/judgment 编排依赖。"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Run 收尾状态（与 execution.helpers._run_status_from_result 对齐）
RUN_STATUS_RUNNING = "running"
RUN_STATUS_SUCCEEDED = "succeeded"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"

_ACTION_KEY_PRIMARY_FIELDS = (
    "path",
    "url",
    "name",
    "title",
    "key",
    "id",
    "task_id",
    "process_id",
    "query",
    "command",
    "status",
)
_ACTION_KEY_EXCLUDED_FIELDS = frozenset({
    "content",
    "old_text",
    "oldText",
    "new_text",
    "newText",
    "workbench",
    "edits",
    "nodes",
    "messages",
    "headers",
    "api_key",
    "token",
    "password",
    "secret",
})
_ACTION_KEY_MAX_VALUE_CHARS = 120
_ACTION_KEY_MAX_MODIFIERS = 6
_ACTION_KEY_MODIFIER_ORDER = (
    "offset",
    "limit",
    "start",
    "end",
    "max_chars",
    "top_k",
    "recursive",
    "include_hidden",
    "timeout",
)


def _stable_param_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(value)
    if len(text) <= _ACTION_KEY_MAX_VALUE_CHARS:
        return text
    digest = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{text[:_ACTION_KEY_MAX_VALUE_CHARS]}...#{digest}"


def _stable_modifier_parts(params: dict[str, Any], *, primary_field: str) -> list[str]:
    parts: list[str] = []
    ordered_keys = [
        key for key in _ACTION_KEY_MODIFIER_ORDER if key in params
    ] + [
        key for key in sorted(params) if key not in _ACTION_KEY_MODIFIER_ORDER
    ]
    for key in ordered_keys:
        if key == primary_field or key in _ACTION_KEY_EXCLUDED_FIELDS:
            continue
        value = _stable_param_value(params.get(key))
        if not value:
            continue
        parts.append(f"{key}={value}")
        if len(parts) >= _ACTION_KEY_MAX_MODIFIERS:
            break
    return parts


def action_key_param(params: dict[str, Any] | None) -> str:
    """从工具 params 提取用于失败降噪 / 动作指纹的资源键片段。"""
    p = params or {}
    for field in _ACTION_KEY_PRIMARY_FIELDS:
        value = p.get(field)
        if value:
            base = _stable_param_value(value)
            modifiers = _stable_modifier_parts(p, primary_field=field)
            return " ".join([base, *modifiers]) if modifiers else base
    modifiers = _stable_modifier_parts(p, primary_field="")
    return " ".join(modifiers)
