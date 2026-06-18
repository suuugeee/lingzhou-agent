"""core/judgment/decision/routing_mixin.py — tier 路由与 provider 解析。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.judgment.tiers import (
    fallback_tiers,
    select_tier_for_phase,
    should_exclude_reader_for_phase,
)

if TYPE_CHECKING:
    from core.config import Config
    from provider.base import Provider

_log = logging.getLogger("lingzhou.judgment")


class ExecutorRoutingMixin:
    _provider: Provider
    _cfg: Config
    _routing_providers: dict[str, Provider]
    _override_providers: dict[str, Provider]
    _model_health: dict[str, Any]

    def _resolve_tier_model(self, tier: str) -> tuple[str, str]:
        model_ref = self._cfg.routing.get(tier)
        if model_ref:
            return tier, model_ref
        return "default", self._cfg.model

    def _tier_fallback_models(self, tier: str) -> list[str]:
        out: list[str] = []
        for m in self._cfg.model_fallbacks.get(tier, []):
            if m and m not in out:
                out.append(m)
        return out

    def _tier_model_candidates(
        self,
        tier: str,
        routing_overrides: dict[str, str] | None = None,
        *,
        excluded_model_refs: set[str] | None = None,
        excluded_provider_names: set[str] | None = None,
    ) -> list[str]:
        excluded_model_refs = set(excluded_model_refs or set())
        excluded_provider_names = set(excluded_provider_names or set())
        candidates: list[str] = []
        override_model = (routing_overrides or {}).get(tier)
        candidate_sources = (
            override_model,
            self._resolve_tier_model(tier)[1],
            *self._tier_fallback_models(tier),
            self._cfg.model,
        )
        for model_ref in candidate_sources:
            if not model_ref:
                continue
            if model_ref in candidates:
                continue
            if model_ref in excluded_model_refs:
                continue
            if model_ref.partition("/")[0] in excluded_provider_names:
                continue
            candidates.append(model_ref)
        return candidates

    def _find_or_create_provider(self, model_ref: str) -> Provider:
        if model_ref == self._cfg.model:
            return self._provider
        for p in self._routing_providers.values():
            p_ref = (
                getattr(p, "model_ref", None)
                or getattr(p, "_model_ref", None)
                or getattr(p, "_model", None)
            )
            if p_ref == model_ref:
                return p
        if model_ref not in self._override_providers:
            from provider import create_provider_with_model

            self._override_providers[model_ref] = create_provider_with_model(self._cfg, model_ref)
        return self._override_providers[model_ref]

    def _fallback_tiers(self, tier: str, *, exclude_reader: bool = False) -> tuple[str, ...]:
        return fallback_tiers(tier, exclude_reader=exclude_reader)

    def _least_bad_model(
        self,
        tier: str,
        routing_overrides: dict[str, str] | None,
        *,
        exclude_reader: bool = False,
        excluded_model_refs: set[str] | None = None,
        excluded_provider_names: set[str] | None = None,
    ) -> str | None:
        best_model: str | None = None
        best_until = float("inf")
        tiers = (tier, *self._fallback_tiers(tier, exclude_reader=exclude_reader))
        excluded_model_refs = set(excluded_model_refs or set())
        excluded_provider_names = set(excluded_provider_names or set())
        for cand_tier in tiers:
            for model_ref in self._tier_model_candidates(
                cand_tier,
                routing_overrides=routing_overrides,
                excluded_model_refs=excluded_model_refs,
                excluded_provider_names=excluded_provider_names,
            ):
                until = self._get_health(model_ref).cooldown_until
                if until < best_until:
                    best_until = until
                    best_model = model_ref
        return best_model

    def _select_tier(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        prefer_tier: str | None = None,
    ) -> str:
        del user_message, current_action, tool_history
        return select_tier_for_phase(phase, prefer_tier=prefer_tier)

    def _should_exclude_reader(self, phase: str, *, prefer_tier: str | None = None) -> bool:
        return should_exclude_reader_for_phase(phase, prefer_tier=prefer_tier)

    def _cost_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        name = model_ref.lower()
        if "gpt-5" in name or "o3" in name or "qwen3-max" in name:
            return "high"
        if reasoning or "mini" in name or "qwen3.5" in name:
            return "medium"
        return "low"

    def _latency_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        name = model_ref.lower()
        if "gpt-5" in name or "o3" in name:
            return "high"
        if reasoning or "max" in name:
            return "medium"
        return "low"
