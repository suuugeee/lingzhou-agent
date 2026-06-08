"""core/judgment/assembler — 判断层上下文组装器。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from .assemble_context import _assemble_context as _assemble_context_impl
from .continue_context import _build_continue_context as _build_continue_context_impl
from .model_routing import _build_model_routing_section

_build_model_routing_section_impl = _build_model_routing_section

_log = logging.getLogger("lingzhou.judgment")


if TYPE_CHECKING:
    from core.config import Config
    from core.judgment.frame import CognitionFrame
    from core.perception import (
        CognitiveSignals,
        EmotionState,
        EthosState,
        JudgmentSignals,
        Percept,
        PerceptionReplaySummary,
    )
    from core.skill import Skill
    from memory.working import WorkingMemory
    from provider.base import Provider
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from store.task import TaskStore
    from tools.registry import ToolRegistry

    from ..executor import JudgmentExecutor


class JudgmentContextAssembler:
    """组装判断层的上下文：skills、prompts、感知状态 → LLM 消息。"""

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        cfg: Config,
        executor: JudgmentExecutor,
    ) -> None:
        from core.reference import ReferenceResolver
        from core.skill import SkillRegistry

        self._registry = registry
        self._cfg = cfg
        self._executor = executor
        self._system_prompt = cfg.load_prompt("system")
        self._identity_prefix: str = ""
        self._judgment_template = cfg.load_prompt("judgment")
        # 统一使用 cfg.workspace_dir（~ 展开 + 相对路径按配置文件目录解析），
        # 避免不同运行环境中 workspace 路径漂移（例如 "~" 未展开）。
        _skills_dir = cfg.workspace_dir / "skills"
        self._skills = SkillRegistry(skills_dir=_skills_dir)
        self._ref_resolver = ReferenceResolver(provider=provider, thresholds=cfg.thresholds, reason_temperature=cfg.temperature)
        self._last_context_text: str = ""
        self._last_context_sections: dict[str, Any] = {}
        self._last_context_budget: int = 0
        self._context_cache: dict[str, str] = {}
        self._probe_manager: Any = None
        self._last_selected_skills: list[Skill] = []
        self._last_applied_skill_names: list[str] = []

    def reload_skills(self) -> None:
        from core.skill import SkillRegistry

        skills_dir = self._cfg.workspace_dir / "skills"
        self._skills = SkillRegistry(skills_dir=skills_dir)
        _log.info("[judgment] 已从 %s 重新加载 skills", skills_dir)

    def _coerce_frame_args(
        self,
        frame_or_percept: CognitionFrame | Percept,
        wm: WorkingMemory | None,
        task_store: TaskStore | None,
        episodic: EpisodicMemory | None,
        semantic: SemanticMemory | None,
        emotion: EmotionState | None,
    ) -> tuple[Percept, WorkingMemory, TaskStore, EpisodicMemory, SemanticMemory, EmotionState]:
        from ..frame import CognitionFrame as _CognitionFrame

        if isinstance(frame_or_percept, _CognitionFrame):
            return (
                frame_or_percept.percept,
                frame_or_percept.wm,
                frame_or_percept.task_store,
                frame_or_percept.episodic,
                frame_or_percept.semantic,
                frame_or_percept.emotion,
            )
        if None in (wm, task_store, episodic, semantic, emotion):
            raise TypeError("decide/_assemble_context 缺少认知基底参数")
        return (
            frame_or_percept,
            cast("WorkingMemory", wm),
            cast("TaskStore", task_store),
            cast("EpisodicMemory", episodic),
            cast("SemanticMemory", semantic),
            cast("EmotionState", emotion),
        )

    def set_identity_prefix(self, prefix: str) -> None:
        self._identity_prefix = prefix
        _log.debug("[judgment] identity_prefix 已设置（%d 字符）", len(prefix))

    def reload_prompt(self, key: str) -> None:
        if key == "judgment":
            self._judgment_template = self._cfg.load_prompt("judgment")
        elif key == "system":
            self._system_prompt = self._cfg.load_prompt("system")

    def _build_messages(self, user_content: str) -> list[Any]:
        from provider.base import Message

        system_content = self._identity_prefix + "\n\n" + self._system_prompt if self._identity_prefix else self._system_prompt
        return [Message(role="system", content=system_content), Message(role="user", content=user_content)]

    def _build_continue_context(
        self,
        tool_history: list[dict[str, Any]],
        *,
        user_message: str,
        reply_only: bool,
        wm_delta: list[dict[str, Any]] | None,
        speech_intent: str = "",
        action_result: Any | None = None,
        emotion_state: dict[str, Any] | None = None,
    ) -> str:
        return _build_continue_context_impl(
            self,
            tool_history,
            user_message=user_message,
            reply_only=reply_only,
            wm_delta=wm_delta,
            speech_intent=speech_intent,
            action_result=action_result,
            emotion_state=emotion_state,
        )

    def _build_model_routing_section(
        self,
        *,
        phase: str,
        user_message: str,
        current_action: str,
        tool_history: list[dict[str, Any]] | None,
        effective_thinking: str,
        routing_overrides: dict[str, str] | None = None,
        registry: Any | None = None,
    ) -> str:
        impl = globals().get("_build_model_routing_section_impl")
        if impl is None:
            # 兜底恢复：运行时若出现半初始化/热更新漂移，按需重新绑定实现，避免 NameError 自旋。
            impl = globals().get("_build_model_routing_section")
            globals()["_build_model_routing_section_impl"] = impl
        return impl(
            self,
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            effective_thinking=effective_thinking,
            routing_overrides=routing_overrides,
            registry=registry,
        )

    async def _assemble_context(
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
        phase: str = "initial",
        current_action: str = "",
        tool_history: list[dict[str, Any]] | None = None,
        effective_thinking: str | None = None,
        routing_overrides: dict[str, str] | None = None,
        registry_override: Any | None = None,
        runtime_life_snapshot: dict[str, Any] | None = None,
    ) -> str:
        return await _assemble_context_impl(
            self,
            frame_or_percept,
            wm=wm,
            task_store=task_store,
            episodic=episodic,
            semantic=semantic,
            emotion=emotion,
            active_task=active_task,
            user_message=user_message,
            chat_id=chat_id,
            ethos_state=ethos_state,
            judgment_signals=judgment_signals,
            hard_boundaries=hard_boundaries,
            perception_replay=perception_replay,
            cognitive_signals=cognitive_signals,
            phase=phase,
            current_action=current_action,
            tool_history=tool_history,
            effective_thinking=effective_thinking,
            routing_overrides=routing_overrides,
            registry_override=registry_override,
            runtime_life_snapshot=runtime_life_snapshot,
        )
