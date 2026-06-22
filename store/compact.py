from __future__ import annotations

import hashlib
import json
from typing import Any

RUNTIME_TEXT_MAX_CHARS = 12_000
RUNTIME_COLLECTION_MAX_ITEMS = 80
RUNTIME_MAX_DEPTH = 8


def compact_runtime_text(
    value: Any,
    *,
    limit: int = RUNTIME_TEXT_MAX_CHARS,
    marker_label: str = "runtime store",
) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    marker = f"\n...[{marker_label} truncated chars={len(text)} sha256={digest}]...\n"
    remaining = max(0, limit - len(marker))
    if remaining <= 0:
        return marker[:limit]
    head = max(1, remaining // 2)
    tail = max(1, remaining - head)
    return text[:head] + marker + text[-tail:]


def compact_runtime_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return compact_runtime_text(value)
    if isinstance(value, dict):
        if depth >= RUNTIME_MAX_DEPTH:
            return compact_runtime_text(_json_preview(value))
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:RUNTIME_COLLECTION_MAX_ITEMS]:
            compacted[str(key)] = compact_runtime_value(item, depth=depth + 1)
        omitted = len(items) - len(compacted)
        if omitted > 0:
            compacted["_persistent_omitted_items"] = omitted
        return compacted
    if isinstance(value, (list, tuple, set)):
        if depth >= RUNTIME_MAX_DEPTH:
            return compact_runtime_text(_json_preview(list(value)))
        items = list(value)
        if len(items) <= RUNTIME_COLLECTION_MAX_ITEMS:
            return [
                compact_runtime_value(item, depth=depth + 1)
                for item in items
            ]
        retained_items = max(2, RUNTIME_COLLECTION_MAX_ITEMS - 1)
        head_count = max(1, retained_items // 2)
        tail_count = max(1, retained_items - head_count)
        omitted = len(items) - head_count - tail_count
        return [
            *[
                compact_runtime_value(item, depth=depth + 1)
                for item in items[:head_count]
            ],
            {"_persistent_omitted_items": omitted},
            *[
                compact_runtime_value(item, depth=depth + 1)
                for item in items[-tail_count:]
            ],
        ]
    return value


def compact_runtime_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    compacted = compact_runtime_value(value or {})
    return compacted if isinstance(compacted, dict) else {}


def compact_runtime_json_text(
    value: Any,
    *,
    marker_label: str = "runtime store",
) -> str:
    text = "" if value is None else str(value)
    try:
        payload = json.loads(text)
    except Exception:
        return compact_runtime_text(text, marker_label=marker_label)
    if not isinstance(payload, (dict, list)):
        return compact_runtime_text(text, marker_label=marker_label)
    compacted = compact_runtime_value(payload)
    try:
        return json.dumps(compacted, ensure_ascii=False, sort_keys=True)
    except Exception:
        return compact_runtime_text(text, marker_label=marker_label)


def _json_preview(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(value)
