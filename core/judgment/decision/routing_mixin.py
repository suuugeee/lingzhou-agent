"""core/judgment/decision/routing_mixin.py — tier 路由与 provider 解析。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

    _REASONER_ONLY_PHASES = frozenset({"initial", "continue", "reply", "final"})

    def _routing_aliases(self, tier: str) -> tuple[str, ...]:
        return {
            "reader": ("reader", "simple"),
            "reasoner": ("reasoner", "complex"),
            "repair": ("repair", "reader", "simple"),
        }.get(tier, (tier,))

    def _resolve_tier_model(self, tier: str) -> tuple[str, str]:
        for alias in self._routing_aliases(tier):
            model_ref = self._cfg.routing.get(alias)
            if model_ref:
                return alias, model_ref
        return "default", self._cfg.model

    def _tier_fallback_models(self, tier: str) -> list[str]:
        out: list[str] = []
        for key in (tier, *self._routing_aliases(tier)):
            for m in self._cfg.model_fallbacks.get(key, []):
                if m and m not in out:
                    out.append(m)
        return out

    def _tier_model_candidates(
        self,
        tier: str,
        routing_overrides: dict[str, str] | None = None,
    ) -> list[str]:
        candidates: list[str] = []
        override_model = (routing_overrides or {}).get(tier)
        if override_model:
            candidates.append(override_model)
        _, primary = self._resolve_tier_model(tier)
        if primary and primary not in candidates:
            candidates.append(primary)
        for m in self._tier_fallback_models(tier):
            if m not in candidates:
                candidates.append(m)
        if self._cfg.model not in candidates:
            candidates.append(self._cfg.model)
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
        if tier == "reasoner":
            return ("repair",) if exclude_reader else ("reader", "repair")
        if tier == "reader":
            return ("reasoner", "repair")
        if tier == "repair":
            return ("reasoner",) if exclude_reader else ("reader", "reasoner")
        return ("reasoner", "repair") if exclude_reader else ("reader", "reasoner", "repair")

    def _least_bad_model(
        self,
        tier: str,
        routing_overrides: dict[str, str] | None,
        *,
        exclude_reader: bool = False,
    ) -> str | None:
        best_model: str | None = None
        best_until = float("inf")
        tiers = (tier, *self._fallback_tiers(tier, exclude_reader=exclude_reader))
        for cand_tier in tiers:
            for model_ref in self._tier_model_candidates(cand_tier, routing_overrides=routing_overrides):
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
        if phase == "repair":
            return "repair"
        if prefer_tier in {"reader", "reasoner", "repair"}:
            return prefer_tier
        if phase == "continue":
            return "reasoner"
        if phase in {"reply", "final"}:
            return "reasoner"
        return "reasoner"

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
