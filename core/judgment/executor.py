"""core/judgment/executor.py — LLM provider 管理与调用层（JudgmentExecutor）。"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.judgment.decision.health_mixin import ExecutorHealthMixin
from core.judgment.decision.helpers import (
    _chat_with_retry_impl,
    _repair_output_impl,
    _select_provider_impl,
)
from core.judgment.decision.prompt_mixin import ExecutorPromptMixin
from core.judgment.decision.routing_mixin import ExecutorRoutingMixin
from core.persona.self_model import SelfModel

from .output import JudgmentOutput, ModelHealth, ModelSelection

if TYPE_CHECKING:
    from core.config import Config
    from provider.base import Provider

_log = logging.getLogger("lingzhou.judgment")


class JudgmentExecutor(
    ExecutorRoutingMixin,
    ExecutorHealthMixin,
    ExecutorPromptMixin,
):
    """LLM provider 管理与调用层。由 JudgmentLayer 创建并持有。"""

    def __init__(self, provider: Provider, cfg: Config) -> None:
        self._provider = provider
        self._cfg = cfg
        self._routing_providers: dict[str, Provider] = {}
        self._override_providers: dict[str, Provider] = {}
        self._model_health: dict[str, ModelHealth] = {}
        self._provider_errors: dict[str, str] = {}
        self._last_call_meta: dict[str, Any] = {
            "phase": "",
            "tier": "default",
            "model_ref": cfg.model,
            "thinking": cfg.thinking,
            "skills": "",
        }
        self._last_prompt_capsule: str = ""
        self._last_prompt_capsule_source_tokens: int = 0
        self.self_model = SelfModel()

    def set_routing_providers(self, providers: dict[str, Provider]) -> None:
        provider_names = list(providers)
        changed = set(provider_names) != set(self._routing_providers)
        self._routing_providers = providers
        if providers:
            if changed:
                _log.info("[judgment] 路由 providers 已设置: %s", provider_names)
            else:
                _log.debug("[judgment] 路由 providers 刷新（无变化）: %s", provider_names)

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return dict(self._last_call_meta)

    def _select_provider(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        prefer_tier: str | None = None,
        thinking_override: str | None = None,
        routing_overrides: dict[str, str] | None = None,
        excluded_model_refs: set[str] | None = None,
        excluded_provider_names: set[str] | None = None,
    ) -> tuple[Provider, ModelSelection]:
        return _select_provider_impl(
            self,
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
            excluded_model_refs=excluded_model_refs,
            excluded_provider_names=excluded_provider_names,
        )

    def _set_last_call_meta(
        self,
        selection: ModelSelection,
        *,
        thinking_override: str | None,
        skills: str,
        primary_skill_name: str | None = None,
        primary_skill_guidance: bool | None = None,
    ) -> None:
        meta: dict[str, Any] = {
            "phase": selection.phase,
            "tier": selection.tier,
            "model_ref": selection.model_ref,
            "thinking": thinking_override or selection.thinking,
            "skills": skills,
        }
        if primary_skill_name is not None or primary_skill_guidance is not None:
            meta["primary_skill"] = primary_skill_name
            meta["primary_skill_guidance"] = bool(primary_skill_guidance)
        self._last_call_meta = meta

    def _track_token_usage(self, provider: Provider) -> None:
        usage = getattr(provider, "last_usage", None)
        if isinstance(usage, dict):
            self.self_model.record_token_usage(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )
            self._last_call_meta["usage_source"] = str(usage.get("usage_source") or "missing")

    async def _chat_with_retry(
        self,
        *,
        selected_provider: Provider,
        selection: ModelSelection,
        messages: list[Any],
        phase: str,
        user_message: str,
        thinking_override: str | None,
        routing_overrides: dict[str, str] | None,
        log_prefix: str,
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        fallback_prefer_tier: str | None = None,
        skills: str = "none",
        primary_skill_name: str | None = None,
        primary_skill_guidance: bool | None = None,
    ) -> tuple[str | None, ModelSelection, Exception | None]:
        return await _chat_with_retry_impl(
            self,
            selected_provider=selected_provider,
            selection=selection,
            messages=messages,
            phase=phase,
            user_message=user_message,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
            log_prefix=log_prefix,
            current_action=current_action,
            tool_history=tool_history,
            fallback_prefer_tier=fallback_prefer_tier,
            skills=skills,
            primary_skill_name=primary_skill_name,
            primary_skill_guidance=primary_skill_guidance,
        )

    async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
        return await _repair_output_impl(self, context_text, raw)
