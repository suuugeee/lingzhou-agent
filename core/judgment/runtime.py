"""core/judgment/runtime.py — 判断层（JudgmentLayer 核心类）。

职责：
1. 组装 bundle（运行时状态 → 结构化 context）
2. 填入 prompts/judgment.md 模板（{{variable}} 语法）
3. 调用 LLM provider
4. 解析 JSON 输出 → JudgmentOutput

数据模型 / 工具常量 / 前置改写函数 → output.py
解耦原则：此模块不知道工具如何执行，只负责"决定做什么"。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .executor import JudgmentExecutor
from .assembler import JudgmentContextAssembler
from .parser import (
    simulate_safe_output as _simulate_safe_output_fn,
    coerce_reply_only_output as _coerce_reply_only_output_fn,
    apply_memory_honesty_guard as _apply_memory_honesty_guard,
)
from .output import (
    JudgmentOutput,
    ModelHealth,
    ModelSelection,
    _rewrite_task_ask_to_evidence,
    _rewrite_complex_act_to_task_plan,
    tool_tier,
)
from .context import _clear_context_cache

_log = logging.getLogger("lingzhou.judgment")


if TYPE_CHECKING:
    from core.config import Config
    from core.perception import (
        Percept, EmotionState, EthosState, JudgmentSignals, PerceptionReplaySummary,
        CognitiveSignals,
    )
    from core.skill import Skill
    from memory.working import WorkingMemory
    from store.task import TaskStore
    from store.episodic import EpisodicMemory
    from store.semantic import SemanticMemory
    from tools.registry import ToolRegistry
    from provider.base import Provider


# ── 认知基底（传入 decide/assemble_context 的感知+记忆层快照） ────────────────

@dataclass(slots=True)
class CognitionFrame:
    """6 个认知基底字段的轻量容器，兼容旧调用点。"""

    percept: "Percept"
    wm: "WorkingMemory"
    task_store: "TaskStore"
    episodic: "EpisodicMemory"
    semantic: "SemanticMemory"
    emotion: "EmotionState"


# ── 判断层 ─────────────────────────────────────────────────────────────────────

class JudgmentLayer:
    def __init__(
        self,
        provider: "Provider",
        registry: "ToolRegistry",
        cfg: "Config",
    ) -> None:
        self._cfg = cfg
        self._executor = JudgmentExecutor(provider, cfg)
        self._assembler = JudgmentContextAssembler(provider, registry, cfg, executor=self._executor)

    def reload_skills(self) -> None:
        self._assembler.reload_skills()

    def set_identity_prefix(self, prefix: str) -> None:
        self._assembler.set_identity_prefix(prefix)

    def reload_prompt(self, key: str) -> None:
        self._assembler.reload_prompt(key)

    @property
    def _probe_manager(self) -> Any:
        return self._assembler._probe_manager

    @_probe_manager.setter
    def _probe_manager(self, v: Any) -> None:
        self._assembler._probe_manager = v
    def set_routing_providers(self, providers: dict[str, "Provider"]) -> None:
        """注入分层路由 providers（由 CognitionLoop.open() 调用）。"""
        self._executor.set_routing_providers(providers)

    @property
    def self_model(self) -> Any:
        return self._executor.self_model

    @self_model.setter
    def self_model(self, v: Any) -> None:
        self._executor.self_model = v

    @property
    def _model_health(self) -> Any:
        return self._executor._model_health

    @_model_health.setter
    def _model_health(self, v: Any) -> None:
        self._executor._model_health = v

    @property
    def _provider_errors(self) -> Any:
        return self._executor._provider_errors

    @_provider_errors.setter
    def _provider_errors(self, v: Any) -> None:
        self._executor._provider_errors = v

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return self._executor.last_call_meta

    def _finalize_continue_output(
        self,
        output: JudgmentOutput,
        *,
        reply_only: bool,
        user_message: str,
        active_task: Any | None,
        tool_history: list[dict[str, Any]],
        selection: ModelSelection,
    ) -> JudgmentOutput:
        if reply_only:
            output = _coerce_reply_only_output_fn(output)

        applied = self._assembler._record_applied_skills(output)

        _log.info(
            "[judgment.continue] round=%d phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s",
            len(tool_history), selection.phase, selection.tier, selection.model_ref,
            self._executor._last_call_meta["thinking"], applied,
            output.decision, output.action_label(),
        )
        return output

    async def decide(
        self,
        frame_or_percept: "CognitionFrame | Percept",
        wm: "WorkingMemory | None" = None,
        task_store: "TaskStore | None" = None,
        episodic: "EpisodicMemory | None" = None,
        semantic: "SemanticMemory | None" = None,
        emotion: "EmotionState | None" = None,
        active_task: Any | None = None,
        user_message: str = "",
        chat_id: str | None = None,
        ethos_state: "EthosState | None" = None,
        judgment_signals: "JudgmentSignals | None" = None,
        hard_boundaries: "list[str] | None" = None,
        perception_replay: "PerceptionReplaySummary | None" = None,
        cognitive_signals: "CognitiveSignals | None" = None,
        thinking_override: "str | None" = None,
        prefer_tier: "str | None" = None,
        routing_overrides: "dict[str, str] | None" = None,
        phase: str = "initial",
        registry_override: "Any | None" = None,
    ) -> JudgmentOutput:
        """组装上下文，调用 LLM，返回决策。
        
        thinking_override: 覆盖 cfg.thinking（如 chat 模式用 "low" 加速首轮判断）。
        routing_overrides: 临时覆盖 tier→model 映射（由 loop.py 从 model_strategy 读取）。
        registry_override: 临时覆盖本轮可见工具集（如子灵受限工具视图）。
        """
        percept, wm, task_store, episodic, semantic, emotion = self._assembler._coerce_frame_args(
            frame_or_percept,
            wm,
            task_store,
            episodic,
            semantic,
            emotion,
        )
        try:
            # per-tick 清空静态缓存（静态 section 仅在本 tick 复用）
            self._assembler._context_cache.clear()
            _clear_context_cache()
            context_text = await self._assembler._assemble_context(
                percept, wm, task_store, episodic, semantic, emotion,
                active_task=active_task,
                user_message=user_message,
                chat_id=chat_id,
                ethos_state=ethos_state,
                judgment_signals=judgment_signals,
                hard_boundaries=hard_boundaries,
                perception_replay=perception_replay,
                cognitive_signals=cognitive_signals,
                phase=phase,
                current_action="",
                tool_history=None,
                effective_thinking=thinking_override or self._cfg.thinking,
                routing_overrides=routing_overrides,
                registry_override=registry_override,
            )
        except Exception as _ctx_exc:
            _log.exception("[judgment] _assemble_context() 异常，返回 wait 兜底: %s", _ctx_exc)
            return _simulate_safe_output_fn(
                failure_count=0,
                signals=judgment_signals,
                hard_boundaries=hard_boundaries or [],
                reason=f"上下文组装异常: {_ctx_exc}",
            )
        # 缓存给内层工具循环的续判请求用
        self._assembler._last_context_text = context_text
        messages = self._assembler._build_messages(context_text)

        selected_provider, selection = self._executor._select_provider(
            phase=phase,
            user_message=user_message,
            prefer_tier=prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
        )
        _primary = self._assembler._last_selected_skills[0] if self._assembler._last_selected_skills else None
        raw, selection, llm_error = await self._executor._chat_with_retry(
            selected_provider=selected_provider,
            selection=selection,
            messages=messages,
            phase=phase,
            user_message=user_message,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
            log_prefix="[judgment]",
            skills=self._assembler._skills_for_log(self._assembler._last_selected_skills),
            primary_skill_name=_primary.name if _primary else None,
            primary_skill_guidance=bool(_primary and getattr(_primary, "guidance", None)),
        )
        if raw is None:
            _err = str(llm_error) or repr(llm_error) if llm_error is not None else "unknown error"
            return _simulate_safe_output_fn(
                failure_count=0,
                signals=judgment_signals,
                hard_boundaries=hard_boundaries or [],
                reason=_err,
            )

        output = JudgmentOutput.from_llm(raw)

        # 解析失败时尝试一次修复，避免因为截断/格式噪声直接进入空转
        output = await self._normalize_output(
            output,
            context_text=context_text,
            raw=raw,
            record_parse_failure=task_store.record_failure,
        )
        _applied = self._assembler._record_applied_skills(output)
        _log.info(
            "[judgment] phase=%s tier=%s model=%s thinking=%s applied_skills=%s decision=%s action=%s rationale=%s",
            selection.phase, selection.tier, selection.model_ref, selection.thinking,
            _applied,
            output.decision, output.action_label(), output.rationale or "",
        )

        return output

    async def decide_continue(
        self,
        tool_history: list[dict],
        user_message: str = "",
        active_task: Any | None = None,
        prefer_tier: str | None = None,
        thinking_override: str | None = None,
        routing_overrides: "dict[str, str] | None" = None,
        reply_only: bool = False,
        wm_delta: "list[dict[str, Any]] | None" = None,
    ) -> JudgmentOutput:
        """内层工具循环的续判请求。

        不重践 perception 链路，直接在上次 decide() 缓存的全量上下文后面追加工具历史续判。
        每次 HTTP 请求与普通请求相同，但输入 token 显著减少（不重发全量感知层）。

        Args:
            tool_history: [{"tool": str, "params": dict, "result": str}, ...]
            user_message:  原始用户消息（不再次向 LLM 重复，仅用于选择 provider tier）
        """
        if not self._assembler._last_context_text:
            return JudgmentOutput.wait(reason="[inner-loop] no cached context for continuation")
        continuation_context = self._assembler._build_continue_context(
            tool_history,
            user_message=user_message,
            reply_only=reply_only,
            wm_delta=wm_delta,
        )
        messages = self._assembler._build_messages(continuation_context)

        current_action = "" if reply_only else str(tool_history[-1].get("tool", "")) if tool_history else ""
        phase = "reply" if reply_only else "continue"
        forced_prefer_tier = "reasoner" if reply_only else prefer_tier
        selected_provider, selection = self._executor._select_provider(
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=forced_prefer_tier,
            thinking_override=thinking_override,
            routing_overrides=routing_overrides,
        )
        resolved_thinking = thinking_override
        if resolved_thinking is None and selection.tier == "reasoner" and user_message:
            resolved_thinking = "low"
        raw, selection, llm_error = await self._executor._chat_with_retry(
            selected_provider=selected_provider,
            selection=selection,
            messages=messages,
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            thinking_override=resolved_thinking,
            routing_overrides=routing_overrides,
            fallback_prefer_tier="reasoner" if reply_only else None,
            log_prefix="[judgment.continue]",
            skills=self._executor._last_call_meta.get("skills") or "none",
        )
        if raw is None:
            if llm_error is not None:
                return JudgmentOutput.wait(reason=f"[inner-loop] LLM 不可用: {llm_error!r}")
            return JudgmentOutput.wait(reason="[inner-loop] LLM returned None")

        output = JudgmentOutput.from_llm(raw)
        output = await self._normalize_output(
            output,
            context_text=continuation_context,
            raw=raw,
        )
        return self._finalize_continue_output(
            output,
            reply_only=reply_only,
            user_message=user_message,
            active_task=active_task,
            tool_history=tool_history,
            selection=selection,
        )


    async def _normalize_output(
        self,
        output: JudgmentOutput,
        *,
        context_text: str,
        raw: str,
        record_parse_failure: Any | None = None,
    ) -> JudgmentOutput:
        if output.rationale.startswith("LLM 输出解析失败"):
            repaired = await self._executor._repair_output(context_text, raw)
            if repaired is not None:
                output = repaired
            elif record_parse_failure is not None:
                await record_parse_failure("judgment_parse", output.rationale)

        if output.decision not in ("act", "pause", "wait"):
            return JudgmentOutput.wait(reason=f"无效 decision: {output.decision!r}")
        if output.decision == "act" and not output.chosen_action_id \
                and not output.parallel_actions and not output.delegate_tasks:
            return JudgmentOutput.wait(reason="act 决策缺少 chosen_action_id")
        output = _apply_memory_honesty_guard(output, context_text=context_text)
        return output

