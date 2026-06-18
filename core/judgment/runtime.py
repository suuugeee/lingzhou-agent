"""core/judgment/runtime.py — JudgmentLayer 稳定入口（编排委托 decision.rounds）。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.judgment.boundary import normalize_judgment_output as _normalize_judgment_output_fn
from core.judgment.tiers import INITIAL_PHASE

from .assembler import JudgmentContextAssembler
from .decision.rounds import (
    JudgmentRoundDeps,
    decide_initial,
)
from .decision.rounds import (
    decide_continue as decide_continue_round,
)
from .executor import JudgmentExecutor
from .frame import CognitionFrame
from .output import JudgmentOutput, ModelSelection

if TYPE_CHECKING:
    from core.config import Config
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from memory.working import WorkingMemory
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry


class JudgmentLayer:
    def __init__(
        self,
        provider: Any,
        registry: ToolRegistry,
        cfg: Config,
    ) -> None:
        self._cfg = cfg
        self._executor = JudgmentExecutor(provider, cfg)
        self._assembler = JudgmentContextAssembler(provider, registry, cfg, executor=self._executor)

    def _round_deps(self) -> JudgmentRoundDeps:
        return JudgmentRoundDeps(self._assembler, self._executor, self._cfg)

    def reload_skills(self) -> None:
        self._assembler.reload_skills()

    def set_identity_prefix(self, prefix: str) -> None:
        self._assembler.set_identity_prefix(prefix)

    def reload_prompt(self, key: str) -> None:
        self._assembler.reload_prompt(key)

    def set_routing_providers(self, providers: dict[str, Any]) -> None:
        """注入分层路由 providers（由 CognitionLoop.open() 调用）。"""
        self._executor.set_routing_providers(providers)

    @property
    def self_model(self) -> Any:
        return self._executor.self_model

    @self_model.setter
    def self_model(self, v: Any) -> None:
        self._executor.self_model = v

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return self._executor.last_call_meta

    @property
    def _last_call_meta(self) -> dict[str, Any]:
        return self._executor._last_call_meta

    @_last_call_meta.setter
    def _last_call_meta(self, v: dict[str, Any]) -> None:
        self._executor._last_call_meta = v

    async def decide(
        self,
        frame_or_percept: CognitionFrame | Percept,
        wm: WorkingMemory | None = None,
        task_store: TaskStore | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        emotion: EmotionState | None = None,
        active_task: Any | None = None,
        user_message: str = "",
        chat_id: str | None = None,
        ethos_state: EthosState | None = None,
        judgment_signals: JudgmentSignals | None = None,
        hard_boundaries: list[str] | None = None,
        perception_replay: PerceptionReplaySummary | None = None,
        cognitive_signals: CognitiveSignals | None = None,
        thinking_override: str | None = None,
        prefer_tier: str | None = None,
        routing_overrides: dict[str, str] | None = None,
        phase: str = INITIAL_PHASE,
        registry_override: Any | None = None,
        runtime_life_snapshot: dict[str, Any] | None = None,
    ) -> JudgmentOutput:
        return await decide_initial(
            self._round_deps(),
            frame_or_percept,
            wm,
            task_store,
            episodic,
            semantic,
            emotion,
            active_task=active_task,
            user_message=user_message,
            chat_id=chat_id,
            ethos_state=ethos_state,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
            thinking_override=thinking_override,
            prefer_tier=prefer_tier,
            routing_overrides=routing_overrides,
            phase=phase,
            registry_override=registry_override,
            runtime_life_snapshot=runtime_life_snapshot,
        )

    async def decide_continue(
        self,
        tool_history: list[dict],
        user_message: str = "",
        active_task: Any | None = None,
        prefer_tier: str | None = None,
        thinking_override: str | None = None,
        routing_overrides: dict[str, str] | None = None,
        reply_only: bool = False,
        wm_delta: list[dict[str, Any]] | None = None,
        speech_intent: str = "",
        action_result: Any | None = None,
        emotion_state: dict[str, Any] | None = None,
    ) -> JudgmentOutput:
        return await decide_continue_round(
            self._round_deps(),
            tool_history,
            user_message=user_message,
            active_task=active_task,
            prefer_tier=prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
            reply_only=reply_only,
            wm_delta=wm_delta,
            speech_intent=speech_intent,
            action_result=action_result,
            emotion_state=emotion_state,
        )

    async def _normalize_output(
        self,
        output: JudgmentOutput,
        *,
        context_text: str,
        raw: str,
        record_parse_failure: Any | None = None,
        registry: Any | None = None,
        allow_delegate_tasks: bool = False,
    ) -> JudgmentOutput:
        return await _normalize_judgment_output_fn(
            self._executor,
            output,
            context_text=context_text,
            raw=raw,
            record_parse_failure=record_parse_failure,
            registry=registry,
            allow_delegate_tasks=allow_delegate_tasks,
        )


__all__ = ["CognitionFrame", "JudgmentLayer", "ModelSelection"]
