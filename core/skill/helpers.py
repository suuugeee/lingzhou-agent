from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("core.skill")


@dataclass
class SkillStateCondition:
    signal_name: str
    operator: str
    threshold: float


@dataclass
class SkillStateRule:
    signal_name: str = ""
    weight: float = 0.0
    conditions: list[SkillStateCondition] = field(default_factory=list)
    inhibit: bool = False


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content.strip()
    lines = content.splitlines()
    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end <= 0:
        return {}, content.strip()

    raw = lines[1:end]
    body = "\n".join(lines[end + 1:]).strip()
    meta: dict[str, str] = {}
    i = 0
    while i < len(raw):
        line = raw[i]
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not match:
            i += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if value in {"|", ">"} or (
            value == ""
            and i + 1 < len(raw)
            and bool(raw[i + 1])
            and raw[i + 1].startswith((" ", "\t"))
        ):
            i += 1
            block: list[str] = []
            while i < len(raw):
                nxt = raw[i]
                if nxt and not nxt.startswith((" ", "\t")) and re.match(r"^[A-Za-z_][\w-]*:\s*", nxt):
                    break
                block.append(nxt.lstrip())
                i += 1
            meta[key] = "\n".join(block).strip()
            continue
        meta[key] = value.strip().strip('"\'')
        i += 1
    return meta, body


def _parse_listish(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    raw = raw.strip("[]")
    parts = re.split(r"[,，、;；/／\n|]+", raw)
    return [part.strip().strip('"\'') for part in parts if part.strip()]


def _parse_state_bias(raw: str) -> dict[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    bias: dict[str, float] = {}
    for part in re.split(r"[,;；\n]+", raw):
        chunk = part.strip()
        if not chunk:
            continue
        match = re.match(r"^([A-Za-z_][\w-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)$", chunk)
        if not match:
            continue
        bias[match.group(1)] = float(match.group(2))
    return bias


_STATE_RULE_RE = re.compile(
    r"^([A-Za-z_][\w-]*)(?:\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?))?\s*(?:=>|:=)\s*(-?\d+(?:\.\d+)?)(?:\s+if\s+(.+))?$",
    re.IGNORECASE,
)
_STATE_CONDITION_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?)$")
_STATE_BARE_CONDITION_RE = re.compile(r"^([A-Za-z_][\w-]*)$")
_STATE_CONDITION_SPLIT_RE = re.compile(r"\s+(?:and|&&)\s+", re.IGNORECASE)


def _parse_state_condition(raw: str) -> SkillStateCondition | None:
    chunk = (raw or "").strip()
    if not chunk:
        return None
    match = _STATE_CONDITION_RE.match(chunk)
    if match:
        return SkillStateCondition(
            signal_name=match.group(1),
            operator="==" if match.group(2) == "=" else match.group(2),
            threshold=float(match.group(3)),
        )
    bare = _STATE_BARE_CONDITION_RE.match(chunk)
    if bare:
        return SkillStateCondition(signal_name=bare.group(1), operator=">=", threshold=0.5)
    return None


def _parse_state_conditions(raw: str) -> list[SkillStateCondition]:
    text = (raw or "").strip()
    if not text:
        return []
    conditions: list[SkillStateCondition] = []
    for part in _STATE_CONDITION_SPLIT_RE.split(text):
        condition = _parse_state_condition(part)
        if condition is not None:
            conditions.append(condition)
    return conditions


def _parse_state_rules(raw: str) -> list[SkillStateRule]:
    text = (raw or "").strip()
    if not text:
        return []
    rules: list[SkillStateRule] = []
    for part in re.split(r"[,;；\n]+", text):
        chunk = part.strip()
        if not chunk:
            continue
        inhibit_match = re.match(r"^inhibit(?:\s+if)?\s+(.+)$", chunk, flags=re.IGNORECASE)
        if inhibit_match:
            conditions = _parse_state_conditions(inhibit_match.group(1))
            if conditions:
                rules.append(SkillStateRule(inhibit=True, conditions=conditions))
            continue
        legacy_match = re.match(r"^([A-Za-z_][\w-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)$", chunk)
        if legacy_match:
            rules.append(SkillStateRule(
                signal_name=legacy_match.group(1),
                weight=float(legacy_match.group(2)),
            ))
            continue
        match = _STATE_RULE_RE.match(chunk)
        if not match:
            continue
        signal_name = match.group(1)
        conditions: list[SkillStateCondition] = []
        if match.group(2) and match.group(3):
            conditions.append(SkillStateCondition(
                signal_name=signal_name,
                operator="==" if match.group(2) == "=" else match.group(2),
                threshold=float(match.group(3)),
            ))
        if match.group(5):
            conditions.extend(_parse_state_conditions(match.group(5)))
        rules.append(SkillStateRule(
            signal_name=signal_name,
            weight=float(match.group(4)),
            conditions=conditions,
        ))
    return rules


def _parse_metadata_map(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    data: dict[str, str] = {}
    for line in raw.splitlines():
        chunk = line.strip()
        if not chunk or chunk.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][\w.-]*):\s*(.*)$", chunk)
        if not match:
            continue
        data[match.group(1)] = match.group(2).strip().strip('"\'')
    return data


def _parse_allowed_tools(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split() if item.strip()]


_STANDARD_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _warn_skill_shape(name: str, description: str, md_file: Path) -> None:
    if len(name) > 64 or not _STANDARD_SKILL_NAME_RE.fullmatch(name):
        _log.warning("[skill] %s name=%r 不符合 Agent Skills 推荐约束，已按宽松模式加载", md_file, name)
    if md_file.name == "SKILL.md" and md_file.parent.name != name:
        _log.warning("[skill] %s 的 name=%r 与父目录 %r 不一致，已按宽松模式加载", md_file, name, md_file.parent.name)
    if not description.strip():
        _log.warning("[skill] %s description 为空；该 skill 将难以被 catalog 触发", md_file)


def _extract_trigger_text(description: str, meta: dict[str, str]) -> list[str]:
    triggers: list[str] = []
    for key in ("trigger", "triggers"):
        if key in meta:
            triggers.extend(_parse_listish(meta[key]))
    match = re.search(r"(?:Triggers?|触发(?:词|器|条件)?)[：:]\s*(.+)$", description, flags=re.IGNORECASE | re.DOTALL)
    if match:
        triggers.extend(_parse_listish(match.group(1)))
    return list(dict.fromkeys(t.strip() for t in triggers if t.strip()))


def _trim_guidance(text: str, limit: int = 1600) -> str:
    text = text.strip()
    return text


def _state_signal_values(
    *,
    has_active_task: bool,
    has_next_step: bool,
    failure_count: int,
    high_error_streak: int,
    wm_pressure: float,
    failure_threshold: int,
    wm_pressure_threshold: float,
) -> dict[str, float]:
    failure_ratio = min(1.0, failure_count / max(1, failure_threshold))
    error_denominator = max(1, failure_threshold - 1)
    error_ratio = min(1.0, high_error_streak / error_denominator)
    failure_signal_ratio = max(failure_ratio, error_ratio)
    wm_floor = max(0.2, wm_pressure_threshold)
    wm_pressure_ratio = min(1.0, wm_pressure / wm_floor) if wm_floor > 0 else 0.0
    return {
        "idle_only": 1.0 if not has_active_task and not has_next_step else 0.0,
        "has_active_task": 1.0 if has_active_task else 0.0,
        "has_next_step": 1.0 if has_next_step else 0.0,
        "failure_signal": 1.0 if failure_signal_ratio > 0 else 0.0,
        "failure_signal_ratio": failure_signal_ratio,
        "failure_count_ratio": failure_ratio,
        "high_error_ratio": error_ratio,
        "wm_pressure": 1.0 if wm_pressure >= wm_floor else 0.0,
        "wm_pressure_ratio": wm_pressure_ratio,
    }


def _state_score(
    skill: Any,
    *,
    state_values: dict[str, float],
) -> float:
    rules = list(skill.state_rules or [])
    if not rules:
        rules = [
            SkillStateRule(signal_name=signal_name, weight=weight)
            for signal_name, weight in (skill.state_bias or {}).items()
        ]
    total = 0.0
    for rule in rules:
        if not all(_state_condition_matches(condition, state_values) for condition in (rule.conditions or [])):
            continue
        if rule.inhibit:
            return float("-inf")
        signal_value = state_values.get(rule.signal_name, 0.0) if rule.signal_name else 1.0
        total += signal_value * rule.weight
    return total


def _state_condition_matches(condition: SkillStateCondition, state_values: dict[str, float]) -> bool:
    current = state_values.get(condition.signal_name, 0.0)
    expected = condition.threshold
    if condition.operator == ">=":
        return current >= expected
    if condition.operator == "<=":
        return current <= expected
    if condition.operator == ">":
        return current > expected
    if condition.operator == "<":
        return current < expected
    return abs(current - expected) <= 1e-9


def _skill_activation_text(skill: Any, *, include_frontmatter: bool = False, guidance_limit: int | None = None) -> str:
    content = skill.load_markdown()
    _meta, body = _split_frontmatter(content)
    guidance = (content if include_frontmatter else (body or content)).strip()
    if guidance_limit is not None and guidance_limit > 0:
        guidance = _trim_guidance(guidance, limit=guidance_limit)
    resources = skill.list_resources()
    lines = [
        f"<skill_content name=\"{skill.name}\">",
        guidance or "（该 skill 目前只有 metadata，没有额外 instructions）",
        "",
        f"Skill directory: {skill.skill_dir}",
        f"Skill source: {skill.source_path}",
    ]
    if skill.compatibility:
        lines.append(f"Compatibility: {skill.compatibility}")
    if skill.allowed_tools:
        lines.append(f"Allowed tools: {' '.join(skill.allowed_tools)}")
    if resources:
        lines.append("<skill_resources>")
        lines.extend(f"- {rel}" for rel in resources)
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)
