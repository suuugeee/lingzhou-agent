"""core/judgment/context/budget.py — 判断上下文 token/字符预算裁剪。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.config.budget import adaptive_judgment_input_budget, context_window_input_hard_budget

from .utils import _cache_put, _clip_for_context, _context_fmt_cache, _estimate_tokens

_CONTEXT_PROMPT_RESERVE_RATIO = 0.93
_MESSAGE_TRIM_TARGET_RATIO = 0.9
_MESSAGE_TRIM_MIN_TOKENS = 1024

_CRITICAL_CONTEXT_SECTIONS = frozenset({
    "tools_section",
    "task_section",
    "chat_continuity_section",
    "memories_section",
    "episodic_section",
    "cross_task_episodic_section",
    "current_interlocutor_continuity_section",
    "current_interlocutor_profile_section",
    "chat_memory_section",
})
_CRITICAL_SECTION_MIN_TOKENS = 180


def _effective_context_budget(token_budget: int) -> int:
    """Leave prompt headroom for template/system text before LLM-level trimming."""
    if token_budget <= 0:
        return token_budget
    return max(1, int(token_budget * _CONTEXT_PROMPT_RESERVE_RATIO))


def target_prompt_budget(prompt_limit: int, *, min_tokens: int = _MESSAGE_TRIM_MIN_TOKENS) -> int:
    if prompt_limit <= 0:
        return 0
    return max(min_tokens, int(prompt_limit * _MESSAGE_TRIM_TARGET_RATIO))


def _unique_ordered(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _compact_tools_section(text: str) -> str:
    names = _unique_ordered(re.findall(r"`([^`]+)`", text or ""), limit=96)
    if not names:
        return _clip_for_context(text, 1600)
    required_lines = []
    critical_tools = {"task.workbench", "memory.add_semantic", "memory.search", "task.complete"}
    for name, params in re.findall(r"\s*-\s*`([^`]+)`:.*?\s+参数:\s*\[(.*)\]\s*$", text, flags=re.MULTILINE):
        if name.strip() not in critical_tools:
            continue
        required = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(\*\)", params)
        if required:
            required_lines.append(f"- `{name.strip()}` required: " + ", ".join(required[:8]))
    lines = [
        "**TOOL CATALOG COMPACTED** — 上下文超预算，保留工具名索引；参数细节以 manifest/schema 校验为准。",
        f"available_tools({len(names)} shown): " + ", ".join(f"`{name}`" for name in names),
    ]
    if required_lines:
        lines.append("critical_required_params:")
        lines.extend(required_lines)
    lines.append(
        "策略: 需要精确参数时优先选择最相关工具；执行层会拦截缺参并给出恢复模板。",
    )
    return "\n".join(lines)


def _compact_skill_catalog_section(text: str) -> str:
    names = _unique_ordered(re.findall(r"`([^`]+)`", text or ""), limit=64)
    if not names:
        return _clip_for_context(text, 1400)
    return "\n".join([
        "**SKILL CATALOG COMPACTED** — 上下文超预算，保留技能名索引。",
        "candidate_skills: " + ", ".join(f"`{name}`" for name in names),
        "策略: 只有任务明显匹配时再用 skill.activate 读取完整规则。",
    ])


def _compact_probe_sensors_section(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    headers = [
        line.strip()
        for line in lines
        if re.match(r"^[✓⊘]\s+\[[^\]]+\]", line.strip()) or re.match(r"^[\-\*]\s+\[[^\]]+\]", line.strip())
    ]
    if not headers:
        headers = [line.strip() for line in lines if "[" in line and "]" in line][:32]
    headers = headers[:32]
    summary = [
        "**PROBE SENSORS COMPACTED** — 上下文超预算，保留探针状态摘要；异常/告警优先相信具体 probe.run 结果。",
    ]
    if headers:
        summary.extend(f"- {line}" for line in headers)
    else:
        summary.append(_clip_for_context(text, 1200))
    return "\n".join(summary)


def _compact_model_routing_section(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return _clip_for_context(text, 1800)
    if not isinstance(payload, dict):
        return _clip_for_context(text, 1800)
    compact = {
        key: payload[key]
        for key in (
            "active_overrides",
            "available_models",
            "current_action_capabilities",
            "continue_phase_policy",
            "budget_state",
            "routing_hint",
            "reference_resolution",
            "primary_provider",
        )
        if key in payload
    }
    for key, out_key in (
        ("tool_tier_mapping", "tool_tier_mapping_keys"),
        ("tool_capability_mapping", "tool_capability_mapping_keys"),
    ):
        mapping = payload.get(key)
        if isinstance(mapping, dict):
            compact[out_key] = sorted(str(item) for item in mapping)[:96]
    return "**MODEL ROUTING COMPACTED**\n" + json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=2)


def _compact_wm_section(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    header = lines[0]
    warnings = [line for line in lines if "大条目" in line or "⚠" in line][:3]
    items = [line for line in lines[1:] if line.startswith("- ")][:12]
    compact_items = [_clip_for_context(line, 260) for line in items]
    out = [f"{header} **WM COMPACTED** — 上下文超预算，仅保留最高优先级条目摘要。"]
    out.extend(compact_items)
    out.extend(_clip_for_context(line, 260) for line in warnings)
    return "\n".join(out)


def _compact_static_sections(budgeted: dict[str, str]) -> dict[str, str]:
    compacted = dict(budgeted)
    compactors = {
        "tools_section": _compact_tools_section,
        "skills_catalog_section": _compact_skill_catalog_section,
        "probe_sensors_section": _compact_probe_sensors_section,
        "model_routing_section": _compact_model_routing_section,
        "wm_section": _compact_wm_section,
    }
    for key, compactor in compactors.items():
        original = str(compacted.get(key) or "")
        if _estimate_tokens(original) < 1000:
            continue
        replacement = compactor(original)
        if replacement and _estimate_tokens(replacement) < _estimate_tokens(original):
            compacted[key] = replacement
    return compacted


def _clip_to_token_budget(text: str, token_budget: int) -> str:
    raw = str(text or "")
    if token_budget <= 0 or not raw:
        return ""
    if _estimate_tokens(raw) <= token_budget:
        return raw

    low = 0
    high = len(raw)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = _clip_for_context(raw, mid)
        if _estimate_tokens(candidate) <= token_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def apply_context_budget(
    ctx: dict[str, str],
    token_budget: int | None = None,
    max_chars: int | None = None,
    skill_min_tokens: int = 0,
) -> dict[str, str]:
    if token_budget is None:
        token_budget = max_chars
    if token_budget is None:
        raise TypeError("apply_context_budget() missing required argument: 'token_budget'")
    if token_budget <= 0:
        return ctx

    ctx_hash = hashlib.md5("".join(sorted(ctx.values())).encode()).hexdigest()
    target_budget = _effective_context_budget(token_budget)
    cache_key = f"budget:{ctx_hash}:{token_budget}:{target_budget}:{skill_min_tokens}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]

    budgeted = dict(ctx)
    # Low-value/reconstructable sections are dropped before memory-bearing sections.
    # This keeps recall usable under pressure instead of preserving tool/probe catalogs.
    drop_priority = [
        "tools_section",
        "probe_sensors_section",
        "skills_catalog_section",
        "model_routing_section",
        "shell_capabilities_section",
        "wm_proposal_sections",
        "wm_section",
        "failures_section",
        "durable_failure_section",
        "cognitive_signals_section",
        "signals_section",
        "perception_replay_section",
        "perception_section",
        "blind_spot_section",
        "risk_sections",
        "uncertainty_sections",
        "skills_section",
        "daily_continuity_section",
        "episodic_section",
        "cross_task_episodic_section",
        "chat_continuity_section",
        "memories_section",
        "chat_memory_section",
        "current_interlocutor_continuity_section",
        "current_interlocutor_profile_section",
        "task_section",
    ]

    def total_tokens(items: dict[str, str]) -> int:
        return sum(_estimate_tokens(value) for value in items.values())

    current_total = total_tokens(budgeted)
    if current_total <= target_budget:
        return budgeted

    budgeted = _compact_static_sections(budgeted)
    current_total = total_tokens(budgeted)
    if current_total <= target_budget:
        _cache_put(cache_key, budgeted)
        return budgeted

    drop_order = list(drop_priority)
    drop_order.extend(key for key in budgeted if key not in drop_priority and key != "user_message")

    def _drop_keys(keys: list[str], allow_critical: bool) -> None:
        nonlocal current_total
        for key in keys:
            if current_total <= target_budget:
                break
            if not allow_critical and key in _CRITICAL_CONTEXT_SECTIONS:
                continue
            original = budgeted.get(key, "")
            if not original:
                continue
            budgeted[key] = ""
            current_total -= _estimate_tokens(original)

    _drop_keys([key for key in drop_order if key not in _CRITICAL_CONTEXT_SECTIONS], allow_critical=False)

    def _trim_critical_keys(keys: list[str]) -> None:
        nonlocal current_total
        for key in keys:
            if current_total <= target_budget:
                break
            if key not in _CRITICAL_CONTEXT_SECTIONS:
                continue
            original = budgeted.get(key, "")
            original_tokens = _estimate_tokens(original)
            if original_tokens <= _CRITICAL_SECTION_MIN_TOKENS:
                continue
            desired_tokens = max(
                _CRITICAL_SECTION_MIN_TOKENS,
                original_tokens - (current_total - target_budget),
            )
            replacement = _clip_to_token_budget(original, desired_tokens)
            if not replacement or replacement == original:
                continue
            budgeted[key] = replacement
            current_total += _estimate_tokens(replacement) - original_tokens

    _trim_critical_keys(drop_order)
    if current_total > target_budget:
        _drop_keys(drop_order, allow_critical=True)

    _cache_put(cache_key, budgeted)
    return budgeted


def resolve_judgment_prompt_budget(cfg: Any, model_ref: str, *, catalog_path: Path | None = None) -> int:
    """计算单次 judgment LLM 调用的有效输入预算（token）。

    优先级：
    1. 模型上下文窗口（当前模型若有 catalog 记录，使用动态窗口）
    2. `max_judgment_input_tokens`（显式上限）
    3. 默认安全兜底（避免高 context window 模型一次性喂入超长上下文）
    """
    model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
    from provider.catalog import resolve_context_window

    active_model = getattr(cfg, "model", None)
    context_window = resolve_context_window(
        model_id,
        cfg.context_window_tokens if model_ref == active_model else None,
        catalog_path=Path(catalog_path) if catalog_path is not None else None,
    )
    fallback_budget = 16_000

    max_limit = getattr(cfg, "max_judgment_input_tokens", None)
    if max_limit is not None:
        max_limit = int(max_limit)

    if context_window is None or context_window <= 0:
        if max_limit and max_limit > 0:
            return max_limit
        return fallback_budget

    hard_budget = context_window_input_hard_budget(context_window)
    if max_limit is not None and max_limit > 0:
        return min(hard_budget, max_limit)
    return adaptive_judgment_input_budget(context_window)
