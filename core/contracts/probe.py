"""探针数据契约 — tools 与 core.probe 共享的稳定类型层。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

ProbeKind = Literal["shell", "http", "python", "builtin"]
ProbeDataBack = Literal["none", "wm"]

PROBE_COVERAGE_HINTS: dict[str, str] = {
    "ops:channel_health": "关键外部通道/代理/API 网关健康",
    "ops:api_quota": "API 配额、额度或速率限制",
    "workspace:git_state": "git 变更与工作区状态",
}


def normalize_probe_coverage_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except Exception:
                decoded = text
            if decoded is not text:
                return normalize_probe_coverage_tags(decoded)
        items = text.split(",")
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []

    normalized: list[str] = []
    for item in items:
        tag = str(item or "").strip().lower()
        if not tag or tag in normalized:
            continue
        normalized.append(tag)
    return normalized


@dataclass
class ProbeConfig:
    name: str
    kind: ProbeKind
    spec: str
    trigger: str
    purpose: str = ""
    data_back: ProbeDataBack = "wm"
    coverage_tags: list[str] = field(default_factory=list)
    alert_expr: str | None = None
    alert_message: str | None = None
    enabled: bool = True
    id: int = 0
    created_at: str = ""
    last_run_at: str | None = None
    last_result: str | None = None
    last_error: str | None = None
    last_confidence: float | None = None
    last_confidence_reason: str | None = None
    last_suspect: bool = False
    last_alerted: bool = False
    last_alert_detail: str | None = None


@dataclass
class ProbeResult:
    probe_name: str
    output: str
    error: str | None
    triggered_at: str
    duration_ms: int
    alerted: bool = False
    alert_detail: str | None = None
    confidence: float = 0.5
    confidence_reason: str = ""
    deployment_suspect: bool = False
