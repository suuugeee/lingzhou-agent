"""core/judgment/executor.py — LLM provider 管理与调用层。

职责：
- 按 tier/phase 选择 provider（含 routing、fallback、override）
- 模型健康监控（冷却窗口、429/402/timeout 等错误分类）
- _chat_with_retry：带 fallback 的 LLM 调用
- token 使用量追踪

与 JudgmentLayer 解耦：不知道上下文如何组装，只负责"把 messages 送给哪个模型"。
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from core.self_model import SelfModel

from ._executor_helpers import (
    _chat_with_retry_impl,
    _repair_output_impl,
    _select_provider_impl,
    _trim_messages_for_prompt_limit_impl,
)
from .output import JudgmentOutput, ModelHealth, ModelSelection

if TYPE_CHECKING:
    from core.config import Config
    from provider.base import Provider

_log = logging.getLogger("lingzhou.judgment")
_PROMPT_LIMIT_RE = re.compile(r"prompt token count of\s*(\d+)\s*exceeds the limit of\s*(\d+)", re.IGNORECASE)


class JudgmentExecutor:
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
        self.self_model = SelfModel()

    # ── 公开 API ───────────────────────────────────────────────────────────────

    def set_routing_providers(self, providers: dict[str, Provider]) -> None:
        """注入分层路由 providers（由 CognitionLoop.open() 调用）。"""
        changed = set(providers.keys()) != set(self._routing_providers.keys())
        self._routing_providers = providers
        if providers:
            if changed:
                _log.info("[judgment] 路由 providers 已设置: %s", list(providers.keys()))
            else:
                _log.debug("[judgment] 路由 providers 刷新（无变化）: %s", list(providers.keys()))

    @property
    def last_call_meta(self) -> dict[str, Any]:
        return dict(self._last_call_meta)

    # ── 模型路由 ───────────────────────────────────────────────────────────────

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
        """返回某个 tier 的显式回退模型链（按配置顺序）。"""
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
        """按优先级构建 tier 的候选模型：override -> routing 主模型 -> 显式 fallback -> 顶层 model。"""
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

    # ── 模型健康 ───────────────────────────────────────────────────────────────

    def _get_health(self, model_ref: str) -> ModelHealth:
        health = self._model_health.get(model_ref)
        if health is None:
            health = ModelHealth()
            self._model_health[model_ref] = health
        return health

    def _classify_error_code(self, err_text: str) -> str:
        text = (err_text or "").lower()
        if " 429 " in f" {text} " or "too many requests" in text:
            if "quota" in text:
                return "quota"  # 配额耗尽（如 Copilot quota exceeded），需长时间冷却
            return "429"
        if " 402 " in f" {text} " or "payment required" in text or "insufficient balance" in text:
            return "402"
        if " 401 " in f" {text} " or "unauthorized" in text:
            return "401"
        if " 403 " in f" {text} " or "forbidden" in text:
            return "403"
        if " 400 " in f" {text} " or "bad request" in text:
            return "400"
        if "readtimeout" in text or "timeout" in text:
            return "timeout"
        return "other"

    def _cooldown_seconds(self, code: str, failure_streak: int) -> float:
        streak = max(1, failure_streak)
        if code == "quota":
            return 3600.0  # 配额耗尽，冷却 1 小时
        if code == "429":
            return min(180.0, 30.0 * streak)
        if code == "402":
            # 余额耗尽 — 不会自动恢复，本次会话屏蔽 24h
            return 86400.0
        if code in {"401", "403"}:
            return min(300.0, 120.0 + 30.0 * (streak - 1))
        if code == "400":
            return min(180.0, 45.0 * streak)
        if code == "timeout":
            return min(120.0, 20.0 * streak)
        return min(90.0, 15.0 * streak)

    def _mark_model_failure(self, model_ref: str, err_text: str) -> None:
        code = self._classify_error_code(err_text)
        health = self._get_health(model_ref)
        health.failure_streak += 1
        health.last_error = err_text  # 保留完整错误信息，不截断
        health.last_code = code
        health.cooldown_until = time.time() + self._cooldown_seconds(code, health.failure_streak)
        self._provider_errors[model_ref] = health.last_error

    def _mark_model_success(self, model_ref: str) -> None:
        health = self._get_health(model_ref)
        health.failure_streak = 0
        health.last_error = ""
        health.last_code = ""
        health.cooldown_until = 0.0
        self._provider_errors.pop(model_ref, None)

    def _is_model_available(self, model_ref: str) -> bool:
        return self._get_health(model_ref).cooldown_until <= time.time()

    def _find_or_create_provider(self, model_ref: str) -> Provider:
        """按 model_ref 找到或创建 provider（用于 routing_overrides 临时覆盖）。"""
        if model_ref == self._cfg.model:
            return self._provider
        # _routing_providers 按 tier 存储，用完整 model_ref 匹配
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

    # 必须使用 reasoner 级别的 phase（不允许降级到 reader）
    _REASONER_ONLY_PHASES = frozenset({"initial", "continue", "reply", "final"})

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
        """全部候选均在冷却时，选冷却最短（剩余等待时间最少）的模型作为兜底。"""
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

    # ── provider 选择 ──────────────────────────────────────────────────────────

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
        )

    # ── 成本/延迟信息（供 context 展示用）─────────────────────────────────────

    def _cost_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        _name = model_ref.lower()
        if "gpt-5" in _name or "o3" in _name or "qwen3-max" in _name:
            return "high"
        if reasoning or "mini" in _name or "qwen3.5" in _name:
            return "medium"
        return "low"

    def _latency_level_for_model(self, model_ref: str, reasoning: bool) -> str:
        _name = model_ref.lower()
        if "gpt-5" in _name or "o3" in _name:
            return "high"
        if reasoning or "max" in _name:
            return "medium"
        return "low"

    # ── LLM 调用 ───────────────────────────────────────────────────────────────

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
        """从 provider 读取 last_usage 并累积到 self_model。"""
        usage = getattr(provider, "last_usage", None)
        if isinstance(usage, dict):
            self.self_model.record_token_usage(
                prompt=usage.get("prompt_tokens", 0),
                completion=usage.get("completion_tokens", 0),
            )

    @staticmethod
    def _extract_model_id(model_ref: str) -> str:
        return model_ref.split("/", 1)[1] if "/" in model_ref else model_ref

    @staticmethod
    def _extract_prompt_limit(err_text: str) -> tuple[int | None, int | None]:
        match = _PROMPT_LIMIT_RE.search(err_text or "")
        if not match:
            return None, None
        try:
            prompt = int(match.group(1))
            limit = int(match.group(2))
            return prompt, limit
        except Exception:
            return None, None

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        other = len(text) - cjk - ascii_chars
        return max(1, int(cjk * 1.8 + ascii_chars * 0.3 + other * 1.0))

    def _compress_text_to_budget(self, text: str, keep_tokens: int) -> str:
        if keep_tokens <= 0:
            return ""
        if self._estimate_text_tokens(text) <= keep_tokens:
            return text

        instruction = (
            "[PROMPT_COMPRESSION_REQUIRED]\n"
            "输入过长。请先压缩提炼关键信息，再继续完成任务。\n"
            "保留：目标、约束、错误信息、关键实体、关键数字。\n"
            "删除：重复和冗余描述；不确定信息标记为 unknown。\n\n"
            "[SOURCE_HEAD_TAIL]\n"
        )

        estimated = max(1, self._estimate_text_tokens(text))
        instruction_tokens = self._estimate_text_tokens(instruction)
        if keep_tokens <= instruction_tokens + 16:
            # 极小预算时退化为最短提示，确保重试请求可发送。
            return instruction[: max(1, min(len(instruction), keep_tokens))]

        source_budget_tokens = max(32, int((keep_tokens - instruction_tokens) * 0.9))
        ratio = max(0.01, min(1.0, source_budget_tokens / float(estimated)))
        keep_chars = max(1, int(len(text) * ratio * 0.9))

        marker = "\n\n...[prompt 已压缩]...\n\n"
        if keep_chars <= len(marker) + 2:
            return instruction + text[: max(1, keep_chars)]

        head_chars = max(1, int((keep_chars - len(marker)) * 0.6))
        tail_chars = max(1, keep_chars - len(marker) - head_chars)
        if head_chars + tail_chars >= len(text):
            compact_source = text[: max(1, len(text) // 2)]
        else:
            compact_source = f"{text[:head_chars]}{marker}{text[-tail_chars:]}"

        return instruction + compact_source

    def _trim_messages_for_prompt_limit(
        self,
        messages: list[Any],
        prompt_limit: int,
        *,
        prompt_count: int | None = None,
    ) -> list[Any]:
        return _trim_messages_for_prompt_limit_impl(
            self,
            messages,
            prompt_limit,
            prompt_count=prompt_count,
        )

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

    # ── 输出修复（二次 LLM 调用）──────────────────────────────────────────────

    async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
        """对被截断或损坏的 JSON 做一次二次修复。"""
        return await _repair_output_impl(self, context_text, raw)
