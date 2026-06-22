from __future__ import annotations

from core.judgment.tiers import (
    DEFAULT_TIER,
    INITIAL_PHASE as PHASE_INITIAL,
    JUDGMENT_TIERS,
    READER_TIER,
    REPAIR_TIER,
    REASONER_TIER,
    retry_fallback_tier,
    select_tier_for_phase,
    should_exclude_reader_for_phase,
)


def test_retry_fallback_tier_prefers_explicit_prefer_tier_when_given():
    assert retry_fallback_tier(
        current_tier=READER_TIER,
        phase=PHASE_INITIAL,
        fallback_prefer_tier=REPAIR_TIER,
    ) == REPAIR_TIER


def test_retry_fallback_tier_excludes_reader_on_initial_like_phases():
    # initial / reply / final / continue 都应走 reasoner-only 流程
    assert retry_fallback_tier(
        current_tier=READER_TIER,
        phase=PHASE_INITIAL,
        fallback_prefer_tier=None,
    ) == REASONER_TIER
    assert retry_fallback_tier(
        current_tier=READER_TIER,
        phase="reply",
        fallback_prefer_tier=None,
    ) == REASONER_TIER
    assert retry_fallback_tier(
        current_tier=READER_TIER,
        phase="continue",
        fallback_prefer_tier=None,
    ) == REASONER_TIER
    assert retry_fallback_tier(
        current_tier=READER_TIER,
        phase="final",
        fallback_prefer_tier=None,
    ) == REASONER_TIER


def test_retry_fallback_tier_uses_default_phase_fallback_order():
    # 非 reasoner-only 阶段，reader -> fallback 到 reasoner（按 JUDGMENT_TIERS 顺序去掉当前 tier）
    fallback = retry_fallback_tier(
        current_tier=READER_TIER,
        phase="summary",  # 非阶段约束
        fallback_prefer_tier=None,
    )
    assert fallback == REASONER_TIER
    tiers = list(JUDGMENT_TIERS)
    assert tiers[0] == READER_TIER
    assert tiers[1] == REASONER_TIER
    assert tiers[2] == REPAIR_TIER


def test_retry_fallback_tier_normalizes_unknown_prefer_tier_to_default():
    assert retry_fallback_tier(
        current_tier="invalid-tier",
        phase="summary",
        fallback_prefer_tier="not_real",
    ) == DEFAULT_TIER


def test_retry_fallback_tier_invalid_current_tier_falls_back_to_repair_first():
    assert retry_fallback_tier(
        current_tier="invalid-tier",
        phase="summary",
        fallback_prefer_tier=None,
    ) == REPAIR_TIER


def test_tier_routing_normalizes_case_and_space():
    assert select_tier_for_phase(" INITIAL ", prefer_tier=" Reader ") == READER_TIER
    assert should_exclude_reader_for_phase(" INITIAL ", prefer_tier=" not-real ") is True
    assert retry_fallback_tier(
        current_tier=" Reader ",
        phase="summary",
        fallback_prefer_tier=None,
    ) == REASONER_TIER
