"""Shared lightweight intent classifiers for runtime guards."""

from __future__ import annotations

from typing import Any

EVIDENCE_MARKERS = (
    "读取", "查询", "自检", "检查", "核验", "验证", "查看", "确认", "核对", "核实",
    "排查", "定位", "列出", "分析", "梳理", "追踪", "对比", "运行", "执行", "复现",
    "修复", "实现", "测试", "调试", "调研", "日志", "schema", "read", "query",
    "check", "verify", "diagnose", "inspect", "list", "analyze", "run", "execute",
    "fix", "implement", "test", "debug", "log",
)
WAIT_DEPENDENCY_MARKERS = (
    "等待", "外部输入", "外部信号", "下一次", "到期", "用户输入",
    "wait ", "external", "signal", "inbox",
)
NON_ACTIONABLE_VERIFICATION_MARKERS = (
    "无需", "不需要", "进入低频观察", "等待新用户", "no need", "not needed", "wait for user",
)
COMPLETED_VERIFICATION_PREFIXES = ("已完成", "完成该", "already done")
SEMANTIC_MEMORY_VERIFICATION_MARKERS = (
    "memory.add_semantic", "add_semantic", "语义记忆", "写入语义",
    "沉淀", "固化经验", "semantic memory",
)
ACTIONABLE_VERIFICATION_MARKERS = EVIDENCE_MARKERS + SEMANTIC_MEMORY_VERIFICATION_MARKERS + (
    "修改", "改进", "编辑", "推送", "提交", "rerun", "fetch", "search", "open", "pytest",
    "git ", "curl",
)
_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in "　\t\n（）【】()[]，。,.;；：:"})


def normalize_intent_text(value: Any) -> str:
    return " ".join(str(value or "").translate(_PUNCT_TRANSLATION).split()).lower()


def contains_marker(text: str, lowered: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text or marker in lowered for marker in markers)


def contains_evidence_intent(text: str) -> bool:
    normalized = normalize_intent_text(text)
    return any(marker in normalized for marker in EVIDENCE_MARKERS)


def contains_wait_dependency(text: str) -> bool:
    normalized = normalize_intent_text(text)
    return any(marker in normalized for marker in WAIT_DEPENDENCY_MARKERS)


def has_actionable_next_verification(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if any(marker in lowered for marker in NON_ACTIONABLE_VERIFICATION_MARKERS):
        return False
    if any(lowered.startswith(marker) for marker in COMPLETED_VERIFICATION_PREFIXES):
        return False
    return contains_marker(value, lowered, ACTIONABLE_VERIFICATION_MARKERS)


def requests_semantic_memory(next_verification: str) -> bool:
    lowered = str(next_verification or "").lower()
    return contains_marker(next_verification, lowered, SEMANTIC_MEMORY_VERIFICATION_MARKERS)
