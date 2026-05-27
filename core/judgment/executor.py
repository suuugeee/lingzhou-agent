"""core/judgment/executor.py — LLM provider 管理与调用层。

职责：
- 按 tier/phase 选择 provider（含 routing、fallback、override）
- 模型健康监控（冷却窗口、429/402/timeout 等错误分类）
- _chat_with_retry：带 fallback 的 LLM 调用
- token 使用量追踪

与 JudgmentLayer 解耦：不知道上下文如何组装，只负责"把 messages 送给哪个模型"。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from core.self_model import SelfModel
from .output import JudgmentOutput, ModelHealth, ModelSelection

if TYPE_CHECKING:
    from provider.base import Provider
    from core.config import Config

_log = logging.getLogger("lingzhou.judgment")


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
        h = self._model_health.get(model_ref)
        if h is None:
            h = ModelHealth()
            self._model_health[model_ref] = h
        return h

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

    def _fallback_tiers(self, tier: str) -> tuple[str, ...]:
        if tier == "reasoner":
            return ("reader", "repair")
        if tier == "reader":
            return ("reasoner", "repair")
        if tier == "repair":
            return ("reader", "reasoner")
        return ("reader", "reasoner", "repair")

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
        tier = self._select_tier(
            phase=phase,
            user_message=user_message,
            current_action=current_action,
            tool_history=tool_history,
            prefer_tier=prefer_tier,
        )
        chosen_tier = tier
        chosen_model = self._cfg.model
        provider: Provider = self._provider
        selected = False

        # 先试当前 tier，再按 tier fallback 试其他 tier。
        # 每个 tier 内按：override -> routing 主模型 -> model_fallbacks -> 顶层 model。
        for cand_tier in (tier, *self._fallback_tiers(tier)):
            for model_ref in self._tier_model_candidates(cand_tier, routing_overrides=routing_overrides):
                if not self._is_model_available(model_ref):
                    continue
                try:
                    provider = self._find_or_create_provider(model_ref)
                    chosen_tier = cand_tier
                    chosen_model = model_ref
                    selected = True
                    break
                except Exception as e:
                    _log.warning("[routing] tier=%s model=%s provider 构建失败，跳过: %s", cand_tier, model_ref, e)
                    continue
            if selected:
                break

        thinking = thinking_override if thinking_override is not None else self._cfg.thinking
        return provider, ModelSelection(phase=phase, tier=chosen_tier, model_ref=chosen_model, thinking=thinking)

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
        raw: str | None = None
        last_error: Exception | None = None
        for _attempt in range(2):
            self._set_last_call_meta(
                selection,
                thinking_override=thinking_override,
                skills=skills,
                primary_skill_name=primary_skill_name,
                primary_skill_guidance=primary_skill_guidance,
            )
            try:
                raw = await selected_provider.chat(messages, thinking_override=thinking_override)
                self._mark_model_success(selection.model_ref)
                self._track_token_usage(selected_provider)
                return raw, selection, None
            except Exception as exc:
                last_error = exc
                _err = str(exc) or repr(exc)
                self._mark_model_failure(selection.model_ref, _err)
                if _attempt == 0:
                    _fallback_tier = fallback_prefer_tier or self._fallback_tiers(selection.tier)[0]
                    fb_provider, fb_selection = self._select_provider(
                        phase=phase,
                        user_message=user_message,
                        current_action=current_action,
                        tool_history=tool_history,
                        prefer_tier=_fallback_tier,
                        thinking_override=thinking_override,
                        routing_overrides=routing_overrides,
                    )
                    if fb_selection.model_ref != selection.model_ref:
                        _log.warning(
                            "%s LLM 调用失败，切换模型重试: from=%s(%s) to=%s(%s) err=%s",
                            log_prefix,
                            selection.model_ref,
                            selection.tier,
                            fb_selection.model_ref,
                            fb_selection.tier,
                            _err,
                        )
                        selected_provider, selection = fb_provider, fb_selection
                        continue
                    _log.warning("%s LLM 调用失败，1s 后重试: %s", log_prefix, _err)
                    await asyncio.sleep(1.0)
                    continue
                _log.warning("%s LLM 调用失败: %s", log_prefix, _err)
        return raw, selection, last_error

    # ── 输出修复（二次 LLM 调用）──────────────────────────────────────────────

    async def _repair_output(self, context_text: str, raw: str) -> JudgmentOutput | None:
        """对被截断或损坏的 JSON 做一次二次修复。"""
        from provider.base import Message

        repair_messages = [
            Message(
                role="system",
                content=(
                    "你是一个严格的 JSON 修复器。"
                    "只输出合法 JSON，不要解释，不要使用 markdown 代码块。"
                    "必须遵循这个 schema: {decision, chosen_action_id, params, parallel_actions, delegate_tasks, rationale, reflection, reply_to_user, next_step, model_strategy}."
                    "如果原输出被截断，请根据上下文重新生成一个完整、简短的 JSON。"
                    "如果 broken_output 是裸代码（bash/python 脚本等），将代码原文放入 reply_to_user 字段，decision 设为 pause，rationale 说明代码已封装。"
                ),
            ),
            Message(
                role="user",
                content=(
                    "下面是原始判断上下文和一段损坏/截断的模型输出，请修复为合法 JSON。\n\n"
                    f"[context]\n{context_text}\n\n"
                    f"[broken_output]\n{raw[:4000]}\n\n"
                    "只返回 JSON，不要用 markdown 代码块包裹。"
                ),
            ),
        ]

        try:
            repaired_raw = await self._provider.chat(
                repair_messages,
                temperature=0.0,
            )
        except Exception as exc:
            _log.warning("[judgment] repair request failed: %s", exc)
            return None

        repaired = JudgmentOutput.from_llm(repaired_raw)
        if repaired.rationale.startswith("LLM 输出解析失败"):
            _log.warning("[judgment] repair failed: %s", repaired.rationale)
            return None

        _log.info("[judgment] malformed JSON repaired via second pass")
        return repaired
