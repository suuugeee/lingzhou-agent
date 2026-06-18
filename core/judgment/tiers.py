"""Shared judgment routing tiers."""
from __future__ import annotations

READER_TIER = "reader"
REASONER_TIER = "reasoner"
REPAIR_TIER = "repair"

INITIAL_PHASE = "initial"
CONTINUE_PHASE = "continue"
REPLY_PHASE = "reply"
FINAL_PHASE = "final"
REPAIR_PHASE = "repair"

JUDGMENT_TIERS = (READER_TIER, REASONER_TIER, REPAIR_TIER)
JUDGMENT_TIER_SET = frozenset(JUDGMENT_TIERS)
JUDGMENT_TIER_ROLE_DESCRIPTIONS = {
    READER_TIER: "工具执行层 — 快速/低成本，由系统自动调度执行轻量工具",
    REASONER_TIER: "思考层 — 你本人，负责所有判断、规划、推理与用户交互",
    REPAIR_TIER: "修复层 — 专用于解析失败、格式错误、小修小补",
}
JUDGMENT_TIER_ROUTING_DESCRIPTIONS = {
    READER_TIER: "轻量感知层：适合常规状态查询、读文件、检查计划、无复杂推理的心跳 tick",
    REASONER_TIER: "深度推理层：适合用户交互、要求判断、处理复杂状态、制定或调整计划",
    REPAIR_TIER: "修复层：专用于解析失败、格式错误、小修小补",
}
JUDGMENT_TIER_DISPLAY_LABELS = {
    READER_TIER: "Reader 层",
    REASONER_TIER: "思考层",
    REPAIR_TIER: "Repair 层",
}
DEFAULT_TIER = JUDGMENT_TIERS[1]
REPLY_ONLY_FALLBACK_TIER = DEFAULT_TIER
REASONER_ONLY_PHASES = frozenset({INITIAL_PHASE, CONTINUE_PHASE, REPLY_PHASE, FINAL_PHASE})


def tier_role_description(tier: str) -> str:
    return JUDGMENT_TIER_ROLE_DESCRIPTIONS.get(_normalize_tier_text(tier), "")


def tier_routing_description(tier: str) -> str:
    return JUDGMENT_TIER_ROUTING_DESCRIPTIONS.get(_normalize_tier_text(tier), "")


def tier_display_label(tier: str) -> str:
    text = _normalize_tier_text(tier)
    return JUDGMENT_TIER_DISPLAY_LABELS.get(text, text)


def _normalize_tier_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def is_judgment_tier(value: str | None) -> bool:
    return _normalize_tier_text(value) in JUDGMENT_TIER_SET


def normalize_tier(value: str | None, *, fallback: str = DEFAULT_TIER) -> str:
    text = _normalize_tier_text(value)
    return text if is_judgment_tier(text) else fallback


def fallback_tiers(primary_tier: str, *, exclude_reader: bool = False) -> tuple[str, ...]:
    """返回 fallback 顺序；在未知层时回退到标准顺序。"""
    if not is_judgment_tier(primary_tier):
        return (REPAIR_TIER, REASONER_TIER) if exclude_reader else (REPAIR_TIER, REASONER_TIER, READER_TIER)
    normalized_primary = normalize_tier(primary_tier)
    ordered = tuple(t for t in JUDGMENT_TIERS if t != normalized_primary)
    if exclude_reader:
        ordered = tuple(t for t in ordered if t != READER_TIER)
    return ordered


def is_reasoner_only_phase(phase: str) -> bool:
    """返回 phase 是否强制不走 reader 的路径。"""
    return str(phase or "").strip().lower() in REASONER_ONLY_PHASES


def select_tier_for_phase(phase: str, prefer_tier: str | None = None) -> str:
    """在不考虑可用性时，按 phase 与 prefer_tier 给出层级候选。"""
    if is_judgment_tier(prefer_tier):
        return normalize_tier(prefer_tier)
    return REPAIR_TIER if str(phase or "").strip().lower() == REPAIR_PHASE else REASONER_TIER


def should_exclude_reader_for_phase(phase: str, prefer_tier: str | None = None) -> bool:
    """若无 prefer_tier 且当前 phase 强制走 reasoner/repair，则排除 reader。"""
    return not is_judgment_tier(prefer_tier) and is_reasoner_only_phase(phase)


def retry_fallback_tier(
    current_tier: str,
    phase: str,
    *,
    fallback_prefer_tier: str | None,
) -> str:
    """按 retry 流程的语义返回下一候选 tier。"""
    if fallback_prefer_tier:
        return normalize_tier(fallback_prefer_tier, fallback=DEFAULT_TIER)

    exclude_reader = should_exclude_reader_for_phase(phase)
    candidates = fallback_tiers(current_tier, exclude_reader=exclude_reader)
    if candidates:
        return candidates[0]
    return normalize_tier(current_tier, fallback=DEFAULT_TIER)
