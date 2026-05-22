"""core/probe/types.py — 探针系统核心数据类型。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


# 探针执行方式
ProbeKind = Literal["shell", "http", "python"]

# 数据回传路径（LLM 自行决定如何处置 probe.run 的返回值；interval 探针后台推送到 wm）
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
    """一个已安装探针的完整配置。

    Attributes
    ----------
    name:
        探针唯一名称（人类可读）。
    purpose:
        部署目的/原因（由安装探针的 LLM 填写），如"监控服务器磁盘使用率，防止磁盘满导致任务失败"。
        在 judgment context 中始终可见，帮助下一轮 LLM 理解读数含义。
    kind:
        执行方式：
        - "shell"  — 执行 shell 命令，stdout 作为结果
        - "http"   — GET 请求指定 URL，响应体作为结果
        - "python" — 执行 Python 代码片段，stdout 作为结果
    spec:
        对应 kind 的内容：命令字符串 / URL / Python 代码。
    trigger:
        调度方式：
        - "interval:<seconds>" — 每隔 N 秒执行一次，如 "interval:60"
        - "manual"             — 仅手动触发（probe.run 工具）
    data_back:
        interval 探针周期结果的自动回传路径（manual 探针结果直接通过工具返回值获取）。
    coverage_tags:
        显式声明该探针覆盖的感知维度。blind spots 只读取此字段，不再从 purpose/spec 猜测。
    alert_expr:
        Python 布尔表达式，变量 ``output`` 为结果字符串。
        表达式为 True 时触发告警，如 ``float(output.strip()) > 35.0``
    alert_message:
        告警时注入 WM 的人类可读消息。支持 ``{output}`` 占位符。
    enabled:
        False 时探针被暂停，不自动执行。
    """

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

    # 以下字段由 store 填充，不由用户设置
    id: int = 0
    created_at: str = ""
    last_run_at: str | None = None
    last_result: str | None = None
    last_error: str | None = None
    last_confidence: float | None = None
    last_confidence_reason: str | None = None
    last_suspect: bool = False


@dataclass
class ProbeResult:
    """单次探针执行结果。"""

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
