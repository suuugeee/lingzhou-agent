"""Shared lightweight intent classifiers for runtime guards."""

from __future__ import annotations

from enum import Enum
from typing import Any


class VerificationIntent(Enum):
    ACTIONABLE = "actionable"
    NON_ACTIONABLE = "non_actionable"
    UNKNOWN = "unknown"


VERIFICATION_STATE_KEY = "verification_state"
VERIFICATION_PENDING = "pending"
VERIFICATION_RESOLVED_STATES = {
    "resolved",
    "satisfied",
    "complete",
    "completed",
    "skipped",
    "suppressed",
}
EVIDENCE_MARKERS = (
    "读取", "查询", "自检", "检查", "核验", "验证", "查看", "确认", "核对", "核实",
    "排查", "定位", "列出", "分析", "梳理", "追踪", "对比", "运行", "执行", "复现",
    "修复", "实现", "测试", "调试", "调研", "日志", "schema", "read", "query",
    "check", "verify", "diagnose", "inspect", "list", "analyze", "run", "execute",
    "fix", "implement", "test", "debug", "log",
)
NON_ACTIONABLE_VERIFICATION_PREFIXES = (
    "none",
    "已完成",
    "completed",
    "already done",
    "resolved",
    "satisfied",
    "完成",
    "已达成",
    "已满足",
    "已验证",
)
NON_ACTIONABLE_COMPLETION_HINTS = (
    "none",
    "done",
    "已足够",
    "已经足够",
    "无需",
    "不需要",
    "无须",
    "无需继续",
    "无需重复",
    "无需再",
    "无需再做",
    "no need",
    "not needed",
)
WAIT_DEPENDENCY_MARKERS = (
    "等待", "外部输入", "外部信号", "下一次", "到期", "用户输入",
    "wait ", "external", "signal", "inbox",
)
NON_ACTIONABLE_VERIFICATION_MARKERS = (
    "无需", "不需要", "进入低频观察", "等待新用户", "no need", "not needed", "wait for user",
)
SEMANTIC_MEMORY_VERIFICATION_MARKERS = (
    "memory.add_semantic", "add_semantic", "语义记忆", "写入语义",
    "沉淀", "固化经验", "semantic memory",
)
ACTIONABLE_VERIFICATION_MARKERS = EVIDENCE_MARKERS + SEMANTIC_MEMORY_VERIFICATION_MARKERS + (
    "修改", "改进", "编辑", "推送", "提交", "rerun", "fetch", "search", "open", "pytest",
    "git ", "curl",
)
NON_VERIFICATION_TOOLS = {
    "memory.add_semantic",
    "memory.add_wm",
    "memory.drop_wm",
    "task.workbench",
    "task.advance",
    "task.complete",
    "task.add",
    "task.update",
    "task.amend",
}
VERIFICATION_TOOL_PREFIXES = (
    "file.",
    "shell.",
    "exec.",
    "process.",
    "probe.",
    "config.",
    "browser.",
    "web.",
    "memory.search",
    "memory.embed_backfill",
    "subagent.run",
)
_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in "　\t\n（）【】()[]，。,.;；：:"})
_CONTROL_NEXT_VERIFICATION_PREFIX = "[[control-next-verification]]"


def normalize_intent_text(value: Any) -> str:
    return " ".join(str(value or "").translate(_PUNCT_TRANSLATION).split()).lower()


def strip_control_next_verification_prefix(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith(_CONTROL_NEXT_VERIFICATION_PREFIX):
        text = text[len(_CONTROL_NEXT_VERIFICATION_PREFIX):].strip()
    return text


def clean_next_verification_text(value: Any) -> str:
    return strip_control_next_verification_prefix(str(value or "").strip())


def _is_completion_preface(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(prefix) for prefix in NON_ACTIONABLE_VERIFICATION_PREFIXES)


def _is_resolved_completion(value: str, lowered: str) -> bool:
    if lowered.startswith("none"):
        return True
    if not _is_completion_preface(value):
        return False
    if any(prefix in lowered for prefix in NON_ACTIONABLE_COMPLETION_HINTS):
        return True
    return not contains_marker(value, lowered, ACTIONABLE_VERIFICATION_MARKERS)


def control_next_verification(value: str) -> str:
    cleaned = strip_control_next_verification_prefix(str(value or "").strip())
    return f"{_CONTROL_NEXT_VERIFICATION_PREFIX} {cleaned}" if cleaned else ""


def is_control_next_verification(text: str) -> bool:
    return str(text or "").strip().startswith(_CONTROL_NEXT_VERIFICATION_PREFIX)


def contains_marker(text: str, lowered: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text or marker in lowered for marker in markers)


def contains_evidence_intent(text: str) -> bool:
    normalized = normalize_intent_text(text)
    return any(marker in normalized for marker in EVIDENCE_MARKERS)


def contains_wait_dependency(text: str) -> bool:
    normalized = normalize_intent_text(text)
    return any(marker in normalized for marker in WAIT_DEPENDENCY_MARKERS)


def classify_verification_intent(text: str) -> VerificationIntent:
    """将 next_verification 文本按可执行性分类为可执行/非可执行/未知。"""
    raw = str(text or "").strip()
    if not raw:
        return VerificationIntent.NON_ACTIONABLE
    if is_control_next_verification(raw):
        return VerificationIntent.NON_ACTIONABLE
    value = strip_control_next_verification_prefix(raw)
    lowered = normalize_intent_text(value)
    if _is_resolved_completion(value, lowered):
        return VerificationIntent.NON_ACTIONABLE
    normalized = lowered
    if any(marker in normalized for marker in NON_ACTIONABLE_VERIFICATION_MARKERS):
        return VerificationIntent.NON_ACTIONABLE
    return (
        VerificationIntent.ACTIONABLE
        if contains_marker(value, lowered, ACTIONABLE_VERIFICATION_MARKERS)
        else VerificationIntent.UNKNOWN
    )


def has_actionable_next_verification(text: str) -> bool:
    return classify_verification_intent(text) == VerificationIntent.ACTIONABLE


def verification_state_from_cortex(cortex: Any) -> tuple[str, str]:
    """Return (status, goal) for cortex verification fields.

    status is one of "pending", "resolved", or "none". Unknown explicit statuses
    are treated as resolved so they do not create accidental completion blockers.
    """
    if not isinstance(cortex, dict):
        return "none", ""

    state = cortex.get(VERIFICATION_STATE_KEY)
    if isinstance(state, dict):
        status = str(state.get("status") or "").strip().lower()
        raw_goal = str(
            state.get("goal") or state.get("next_verification") or state.get("text") or ""
        ).strip()
        if is_control_next_verification(raw_goal):
            return "resolved", clean_next_verification_text(raw_goal)
        goal = clean_next_verification_text(raw_goal)
        if status == VERIFICATION_PENDING:
            goal = goal or clean_next_verification_text(cortex.get("next_verification"))
            if goal and has_actionable_next_verification(goal):
                return "pending", goal
            return "resolved", goal
        if status in VERIFICATION_RESOLVED_STATES:
            return "resolved", goal
        if status:
            return "resolved", goal

    if not state:
        raw_goal = str(cortex.get("next_verification") or "").strip()
        if is_control_next_verification(raw_goal):
            return "resolved", clean_next_verification_text(raw_goal)
        goal = clean_next_verification_text(raw_goal)
        if not goal:
            return "none", ""
        status = VERIFICATION_PENDING if has_actionable_next_verification(goal) else "resolved"
        return status, goal

    return "resolved", clean_next_verification_text(getattr(state, "goal", ""))


def build_verification_state(patch: dict[str, Any], *, source: str = "workbench") -> dict[str, Any] | None:
    if VERIFICATION_STATE_KEY in patch:
        raw_state = patch[VERIFICATION_STATE_KEY]
        if not isinstance(raw_state, dict):
            return None
        status = str(raw_state.get("status") or "").strip().lower() or "resolved"
        goal = clean_next_verification_text(
            raw_state.get("goal") or raw_state.get("next_verification") or ""
        )
        return {
            "status": status,
            "goal": goal,
            "source": source,
            **{k: v for k, v in raw_state.items() if k not in {"status", "goal"}},
        }

    if "next_verification" not in patch:
        return None
    next_verification = clean_next_verification_text(patch.get("next_verification"))
    status = VERIFICATION_PENDING if has_actionable_next_verification(next_verification) else "resolved"
    return {
        "status": status,
        "goal": next_verification,
        "source": source,
    }


def requests_semantic_memory(next_verification: str) -> bool:
    lowered = str(next_verification or "").lower()
    return contains_marker(next_verification, lowered, SEMANTIC_MEMORY_VERIFICATION_MARKERS)


def is_verification_tool(tool_name: str, *, next_verification: str = "") -> bool:
    name = str(tool_name or "").strip()
    if not name:
        return False
    if name == "memory.add_semantic" and requests_semantic_memory(next_verification):
        return True
    if name in NON_VERIFICATION_TOOLS or name.startswith("task."):
        return False
    return any(name.startswith(prefix) for prefix in VERIFICATION_TOOL_PREFIXES)


def is_successful_verification_run(run: Any, *, next_verification: str = "") -> bool:
    if str(getattr(run, "status", "") or "").strip() != "succeeded":
        return False
    return is_verification_tool(
        str(getattr(run, "tool_name", "") or ""),
        next_verification=next_verification,
    )
