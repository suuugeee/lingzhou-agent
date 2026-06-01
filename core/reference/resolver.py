from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.config_models import ThresholdsConfig

from .entities import format_section, format_speaker_section, resolve_entities
from .extraction import extract_identity_cues, extract_signals, extract_source_traits
from .models import ExtractedSignals, ResolvedEntity, ResolvedSpeaker
from .reasoning import (
    categorize_llm_error_code,
    reason_about_candidates_with_llm,
    reason_about_speaker_with_llm,
)
from .retrieval import retrieve_candidates, retrieve_speaker_candidates
from .speaker import (
    build_provisional_speaker,
    remember_speaker,
    resolve_current_speaker,
    resolve_speaker_locally,
)

if TYPE_CHECKING:
    from core.metabolic import MetabolicEngine
    from provider.base import Provider
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore


class ReferenceResolver:
    """实体共指消解器：本地候选收窄 + LLM 推理判断。

    JudgmentLayer 持有单例，跨 tick 复用。
    Provider 不可用时自动降级为纯本地评分。
    """

    _log = logging.getLogger("lingzhou.reference")

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        thresholds: ThresholdsConfig | None = None,
        reason_temperature: float | None = None,
    ) -> None:
        self._provider = provider
        self._last_llm_error: str = ""
        self._last_llm_error_code: str = ""
        self._thresholds = thresholds or ThresholdsConfig()
        self._reason_temperature = reason_temperature

    @property
    def last_llm_error(self) -> str:
        return self._last_llm_error

    @property
    def last_llm_error_code(self) -> str:
        return self._last_llm_error_code

    @property
    def llm_available(self) -> bool:
        return self._provider is not None and not self._last_llm_error

    def _categorize_llm_error_code(self, err_text: str) -> str:
        return categorize_llm_error_code(err_text)

    def extract_signals(self, message: str) -> ExtractedSignals:
        return extract_signals(self, message)

    def _retrieve_candidates(
        self,
        message: str,
        sigs: ExtractedSignals,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        source: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        return retrieve_candidates(self, message, sigs, semantic, episodic, source=source)

    def _extract_source_traits(self, message: str, *, chat_id: str = "", source_hint: str = "") -> list[str]:
        return extract_source_traits(message, chat_id=chat_id, source_hint=source_hint)

    def _extract_identity_cues(self, message: str, *, chat_id: str = "", source_hint: str = "") -> dict[str, list[str]]:
        return extract_identity_cues(message, chat_id=chat_id, source_hint=source_hint)

    def _retrieve_speaker_candidates(
        self,
        message: str,
        semantic: SemanticMemory,
        *,
        chat_id: str = "",
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        cached_profile_id: str = "",
        source_hint: str = "",
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        return retrieve_speaker_candidates(
            self,
            message,
            semantic,
            chat_id=chat_id,
            recent_turns=recent_turns,
            chat_continuity=chat_continuity,
            cached_profile_id=cached_profile_id,
            source_hint=source_hint,
        )

    async def _reason_about_candidates_with_llm(
        self,
        message: str,
        candidates: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return await reason_about_candidates_with_llm(self, message, candidates)

    async def _reason_about_speaker_with_llm(
        self,
        message: str,
        *,
        candidates: dict[str, dict[str, Any]],
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        interlocutor_continuity: str = "",
        chat_id: str = "",
        source_hint: str = "",
        cues: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        return await reason_about_speaker_with_llm(
            self,
            message,
            candidates=candidates,
            recent_turns=recent_turns,
            chat_continuity=chat_continuity,
            interlocutor_continuity=interlocutor_continuity,
            chat_id=chat_id,
            source_hint=source_hint,
            cues=cues,
        )

    def _resolve_speaker_locally(
        self,
        candidates: dict[str, dict[str, Any]],
        *,
        cues: dict[str, list[str]],
        chat_id: str = "",
        cached_profile_id: str = "",
    ) -> ResolvedSpeaker | None:
        return resolve_speaker_locally(
            self,
            candidates,
            cues=cues,
            chat_id=chat_id,
            cached_profile_id=cached_profile_id,
        )

    def _build_provisional_speaker(
        self,
        *,
        message: str,
        cues: dict[str, list[str]],
        chat_id: str = "",
        hint_title: str = "",
    ) -> ResolvedSpeaker | None:
        return build_provisional_speaker(message, cues=cues, chat_id=chat_id, hint_title=hint_title)

    async def resolve_current_speaker(
        self,
        message: str,
        semantic: SemanticMemory,
        *,
        chat_id: str = "",
        recent_turns: list[dict[str, Any]] | None = None,
        chat_continuity: str = "",
        interlocutor_continuity: str = "",
        cached_profile_id: str = "",
        source_hint: str = "",
    ) -> ResolvedSpeaker | None:
        return await resolve_current_speaker(
            self,
            message,
            semantic,
            chat_id=chat_id,
            recent_turns=recent_turns,
            chat_continuity=chat_continuity,
            interlocutor_continuity=interlocutor_continuity,
            cached_profile_id=cached_profile_id,
            source_hint=source_hint,
        )

    async def remember_speaker(
        self,
        speaker: ResolvedSpeaker,
        semantic: SemanticMemory,
        task_store: TaskStore | None,
        *,
        message: str,
        chat_id: str = "",
        task_id: str | int | None = None,
        source_hint: str = "",
        metabolic: MetabolicEngine | None = None,
    ) -> None:
        await remember_speaker(
            speaker,
            self,
            semantic,
            task_store,
            message=message,
            chat_id=chat_id,
            task_id=task_id,
            source_hint=source_hint,
            metabolic=metabolic,
        )

    async def resolve(
        self,
        message: str,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
    ) -> list[ResolvedEntity]:
        return await resolve_entities(self, message, semantic, episodic)

    @staticmethod
    def format_section(entities: list[ResolvedEntity]) -> str:
        return format_section(entities)

    @staticmethod
    def format_speaker_section(speaker: ResolvedSpeaker | None) -> str:
        return format_speaker_section(speaker)
