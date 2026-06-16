"""判断输出边界流水线：解析修复 + 形态归一化。"""
from __future__ import annotations

import re
from typing import Any

from core.cortex.actions import build_workbench_action
from core.judgment.boundary.normalize import normalize_action_shape, normalize_reply_pseudo_tool
from core.judgment.output import JudgmentOutput

_PROBLEM_SOLVING_GUARD_ACTIVE = "### 通用问题解决守卫"
_PROBLEM_SOLVING_ALLOWED_ACTIONS = {"task.workbench", "task.amend"}
_RECOVERY_GATE_ACTIVE_MARKER = re.compile(
    r"### 任务级皮层工作区\n(?:(?:.|\n)*?)(?:^### |\Z)",
    re.MULTILINE,
)


_RECOVERY_PLACEHOLDER_TEXTS = {
    "未指定",
    "未进入恢复状态",
    "无",
    "none",
    "null",
    "n/a",
    "na",
    "",
}


def _coalesce_recovery_text(*candidates: str) -> str:
    """从候选文本中取第一个有效恢复文本。"""
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if (value.startswith("（") and value.endswith("）")) or (value.startswith("(") and value.endswith(")")):
            inner = value[1:-1].strip()
            if not inner:
                continue
            value = inner
        normalized = value.strip().lower().replace("　", "").replace("\u2002", "")
        normalized = re.sub(r"\s+", "", normalized)
        if normalized in _RECOVERY_PLACEHOLDER_TEXTS:
            continue
        return value
    return ""


def _extract_recovery_fields(context_text: str) -> tuple[str, str]:
    """从上下文提取 recovery_state 与 next_verification."""
    text = str(context_text or "")
    section_match = _RECOVERY_GATE_ACTIVE_MARKER.search(text)
    section_text = section_match.group(0) if section_match else text

    recovery_state = ""
    next_verification = ""
    recovery_match = re.search(r"-\s*recovery_state=([^\n]+)", section_text)
    if recovery_match:
        recovery_state = str(recovery_match.group(1) or "").strip()
    if not recovery_state:
        recovery_match = re.search(r"恢复状态[:：]\s*(.+)", section_text)
        if recovery_match:
            recovery_state = str(recovery_match.group(1) or "").strip()

    next_match = re.search(r"-\s*next_verification=([^\n]+)", section_text)
    if next_match:
        next_verification = str(next_match.group(1) or "").strip()
    if not next_match:
        next_match = re.search(r"下一步验证[:：]\s*(.+)", section_text)
        if next_match:
            next_verification = str(next_match.group(1) or "").strip()

    recovery_state = _coalesce_recovery_text(recovery_state)
    next_verification = _coalesce_recovery_text(next_verification)
    return recovery_state, next_verification


def _build_recovery_fallback_action(
    next_verification: str,
    registry: Any | None,
) -> tuple[str, dict[str, Any]] | None:
    lowered = str(next_verification or "").lower()
    getter = getattr(registry, "get", None)

    def _has_tool(name: str) -> bool:
        if getter is None:
            return False
        try:
            return getter(name) is not None
        except Exception:
            return False

    if "probe.run" in lowered and _has_tool("probe.run"):
        return "probe.run", {}
    if (
        any(
            marker in lowered
            for marker in ("memory.search", "查找", "搜索", "检索", "记录", "历史")
        )
        and _has_tool("memory.search")
    ):
        return "memory.search", {"query": next_verification[:420], "top_k": 5}
    if (
        any(marker in lowered for marker in ("task.list", "任务列表", "任务状态", "列出任务"))
        and _has_tool("task.list")
    ):
        return "task.list", {"status": "all", "limit": 8}
    return None


def _problem_solving_guard_active(context_text: str) -> bool:
    marker_index = context_text.find(_PROBLEM_SOLVING_GUARD_ACTIVE)
    if marker_index < 0:
        return False
    next_section = context_text.find("\n### ", marker_index + len(_PROBLEM_SOLVING_GUARD_ACTIVE))
    section = context_text[marker_index:] if next_section < 0 else context_text[marker_index:next_section]
    return "guard=active" in section


def _problem_solving_guard_values(context_text: str) -> dict[str, str]:
    marker_index = context_text.find(_PROBLEM_SOLVING_GUARD_ACTIVE)
    if marker_index < 0:
        return {}
    next_section = context_text.find("\n### ", marker_index + len(_PROBLEM_SOLVING_GUARD_ACTIVE))
    section = context_text[marker_index:] if next_section < 0 else context_text[marker_index:next_section]
    values: dict[str, str] = {}
    for line in section.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"guard", "signals", "missing_fields", "required_next_action", "rationale"}:
            values[key] = value.strip()
    return values


def _enforce_recovery_continuation(
    output: JudgmentOutput,
    *,
    context_text: str,
    registry: Any | None = None,
) -> JudgmentOutput:
    if output.decision != "wait":
        return output

    recovery_state, next_verification = _extract_recovery_fields(context_text)
    if not next_verification:
        next_verification = _coalesce_recovery_text(str(output.next_step or ""))
    if not recovery_state:
        return output

    fallback = _build_recovery_fallback_action(next_verification, registry)
    if not fallback:
        return output

    action_id, params = fallback
    return JudgmentOutput(
        decision="act",
        chosen_action_id=action_id,
        params=params,
        rationale=(
            f"任务仍处于恢复态 {recovery_state} 且 next_verification 未完成，"
            f"需要继续最小验证动作：{action_id}"
        ),
        reflection=output.reflection,
        next_step=output.next_step,
        model_strategy=dict(output.model_strategy or {}),
        applied_skills=list(output.applied_skills or []),
    )


def _action_first_must_act(context_text: str) -> bool:
    marker = "action_first:"
    marker_index = context_text.find(marker)
    if marker_index < 0:
        return False
    next_section = context_text.find("\n### ", marker_index + len(marker))
    section = context_text[marker_index:] if next_section < 0 else context_text[marker_index:next_section]
    return "must_act=yes" in section


def _registry_has(registry: Any | None, tool_name: str) -> bool:
    if registry is None:
        return False
    getter = getattr(registry, "get", None)
    if getter is None:
        return False
    try:
        return getter(tool_name) is not None
    except Exception:
        return False


def _captured_context_values(context_text: str, kind: str) -> list[str]:
    pattern = re.compile(rf"-\s*{re.escape(kind)}=([^\n]+)")
    values: list[str] = []
    for match in pattern.finditer(context_text):
        value = str(match.group(1) or "").strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= 4:
            break
    return values


def enforce_action_first_progress(
    output: JudgmentOutput,
    *,
    context_text: str,
    registry: Any | None = None,
) -> JudgmentOutput:
    """Convert empty waits under action-first pressure into safe evidence actions."""
    if not _action_first_must_act(context_text):
        return output
    if output.decision == "act":
        return output

    urls = _captured_context_values(context_text, "url")
    if urls and _registry_has(registry, "web.fetch"):
        return JudgmentOutput(
            decision="act",
            chosen_action_id="web.fetch",
            params={"url": urls[0], "max_chars": 20000},
            rationale="Action-first fallback: 用户给出 URL 且本轮不能空等，先抓取 URL 形成证据。",
            reflection=output.reflection,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            applied_skills=list(output.applied_skills or []),
        )

    paths = _captured_context_values(context_text, "path")
    if paths:
        path = paths[0]
        if path.endswith("/") and _registry_has(registry, "file.list"):
            return JudgmentOutput(
                decision="act",
                chosen_action_id="file.list",
                params={"path": path},
                rationale="Action-first fallback: 用户给出目录路径且本轮不能空等，先列目录形成证据。",
                reflection=output.reflection,
                next_step=output.next_step,
                model_strategy=dict(output.model_strategy or {}),
                applied_skills=list(output.applied_skills or []),
            )
        if _registry_has(registry, "file.read"):
            return JudgmentOutput(
                decision="act",
                chosen_action_id="file.read",
                params={"path": path, "max_chars": 12000},
                rationale="Action-first fallback: 用户给出文件路径且本轮不能空等，先读取路径形成证据。",
                reflection=output.reflection,
                next_step=output.next_step,
                model_strategy=dict(output.model_strategy or {}),
                applied_skills=list(output.applied_skills or []),
            )

    if _registry_has(registry, "task.list"):
        return JudgmentOutput(
            decision="act",
            chosen_action_id="task.list",
            params={"status": "all", "limit": 8},
            rationale="Action-first fallback: 本轮必须推进但缺少更具体安全输入，先读取任务状态形成证据。",
            reflection=output.reflection,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            applied_skills=list(output.applied_skills or []),
        )

    return JudgmentOutput(
        decision="wait",
        rationale="Action-first 要求本轮产生新证据，但 registry 中没有可用的安全取证工具。",
        reflection=output.reflection,
        next_step=output.next_step,
        model_strategy=dict(output.model_strategy or {}),
        applied_skills=list(output.applied_skills or []),
    )


def _action_allowed_by_problem_solving_guard(output: JudgmentOutput) -> bool:
    if output.decision != "act":
        return False
    if output.chosen_action_id:
        return output.chosen_action_id in _PROBLEM_SOLVING_ALLOWED_ACTIONS
    if output.parallel_actions:
        action_ids = {
            str(item.get("action_id") or "").strip()
            for item in output.parallel_actions
            if str(item.get("action_id") or "").strip()
        }
        return bool(action_ids) and action_ids.issubset(_PROBLEM_SOLVING_ALLOWED_ACTIONS)
    return False


def enforce_problem_solving_guard(
    output: JudgmentOutput,
    *,
    context_text: str,
    registry: Any | None = None,
) -> JudgmentOutput:
    """Prevent non-workbench actions while the generic problem-solving guard is active."""
    if not _problem_solving_guard_active(context_text):
        return output
    if _action_first_must_act(context_text) and output.decision == "act":
        return output
    if _action_allowed_by_problem_solving_guard(output):
        return output
    guard_values = _problem_solving_guard_values(context_text)
    if not _registry_has(registry, "task.workbench"):
        return JudgmentOutput(
            decision="wait",
            rationale=(
                "通用问题解决守卫已触发，但 registry 中没有 task.workbench；"
                "无法安全固化问题解决工作台。"
            ),
            reflection=output.reflection,
            next_step=output.next_step,
            model_strategy=dict(output.model_strategy or {}),
            applied_skills=list(output.applied_skills or []),
        )
    return build_workbench_action(
        workbench={
            "domain": "problem-solving",
            "intent": "补齐问题解决工作台，恢复可验证闭环",
            "hypothesis": "当前问题解决缺少结构化工作台，继续旧动作或直接回复会放大误解、重复失败或跨轮丢失承诺。",
            "evidence": [
                f"guard signals: {guard_values.get('signals') or 'unknown'}",
                f"missing fields: {guard_values.get('missing_fields') or 'unknown'}",
            ],
            "next_verification": (
                guard_values.get("required_next_action")
                or "补齐工作台后，基于 domain/intent/hypothesis 选择一个不同证据源执行最小验证。"
            ),
            "completion_checks": [
                "domain/intent/hypothesis 已明确。",
                "experiments_or_evidence 已包含当前证据。",
                "next_verification 指向一个可执行验证动作。",
            ],
        },
        rationale=(
            "通用问题解决守卫已触发：本轮不能继续旧动作或直接回复，"
            "先用 task.workbench 固化问题定义、证据、假设和下一步验证。"
        ),
        source_action=output,
        next_step=output.next_step,
    )


async def normalize_judgment_output(
    executor: Any,
    output: JudgmentOutput,
    *,
    context_text: str,
    raw: str,
    record_parse_failure: Any | None = None,
    registry: Any | None = None,
    allow_delegate_tasks: bool = False,
) -> JudgmentOutput:
    """在输出进入执行层前完成边界校验与归一化。"""
    if output.rationale.startswith("LLM 输出解析失败"):
        repaired = await executor._repair_output(context_text, raw)
        if repaired is not None:
            output = repaired
        elif record_parse_failure is not None:
            await record_parse_failure("judgment_parse", output.rationale)

    output = normalize_reply_pseudo_tool(output)
    output = enforce_action_first_progress(output, context_text=context_text, registry=registry)
    output = enforce_problem_solving_guard(output, context_text=context_text, registry=registry)
    output = _enforce_recovery_continuation(output, context_text=context_text, registry=registry)
    return normalize_action_shape(
        output,
        registry=registry,
        allow_delegate_tasks=allow_delegate_tasks,
    )
