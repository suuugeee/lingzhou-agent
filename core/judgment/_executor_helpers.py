"""core/judgment/_executor_helpers.py — JudgmentExecutor 内部实现函数。"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .output import JudgmentOutput, ModelSelection

if TYPE_CHECKING:
    from provider.base import Provider

    from .executor import JudgmentExecutor

_log = logging.getLogger("lingzhou.judgment")


def _message_log_stats(executor: JudgmentExecutor, messages: list[Any]) -> tuple[int, int, int]:
    message_count = len(messages)
    char_count = 0
    est_tokens = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            char_count += len(content)
            est_tokens += executor._estimate_text_tokens(content)
    return message_count, char_count, est_tokens


def _select_provider_impl(
    executor: JudgmentExecutor,
    *,
    phase: str,
    user_message: str,
    current_action: str = "",
    tool_history: list[dict[str, Any]] | None = None,
    prefer_tier: str | None = None,
    thinking_override: str | None = None,
    routing_overrides: dict[str, str] | None = None,
) -> tuple[Provider, ModelSelection]:
    _effective_prefer_tier = prefer_tier
    tier = executor._select_tier(
        phase=phase,
        user_message=user_message,
        current_action=current_action,
        tool_history=tool_history,
        prefer_tier=_effective_prefer_tier,
    )
    chosen_tier = tier
    chosen_model = executor._cfg.model
    provider: Provider = executor._provider
    selected = False

    exclude_reader = phase in executor._REASONER_ONLY_PHASES and _effective_prefer_tier is None

    for cand_tier in (tier, *executor._fallback_tiers(tier, exclude_reader=exclude_reader)):
        for model_ref in executor._tier_model_candidates(cand_tier, routing_overrides=routing_overrides):
            if not executor._is_model_available(model_ref):
                continue
            try:
                provider = executor._find_or_create_provider(model_ref)
                chosen_tier = cand_tier
                chosen_model = model_ref
                selected = True
                break
            except Exception as e:
                _log.warning("[routing] tier=%s model=%s provider 构建失败，跳过: %s", cand_tier, model_ref, e)
                continue
        if selected:
            break

    if not selected and exclude_reader:
        fallback_ref = executor._least_bad_model(tier, routing_overrides, exclude_reader=True)
        if fallback_ref:
            try:
                provider = executor._find_or_create_provider(fallback_ref)
                chosen_tier = tier
                chosen_model = fallback_ref
                _log.info("[routing] 全部 reasoner 冷却，强制使用冷却最短模型: %s", fallback_ref)
            except Exception as e:
                _log.warning("[routing] least-bad model %s 构建失败: %s", fallback_ref, e)

    thinking = thinking_override if thinking_override is not None else executor._cfg.thinking
    return provider, ModelSelection(phase=phase, tier=chosen_tier, model_ref=chosen_model, thinking=thinking)


def _trim_messages_for_prompt_limit_impl(
    executor: JudgmentExecutor,
    messages: list[Any],
    prompt_limit: int,
    *,
    prompt_count: int | None = None,
) -> list[Any]:
    try:
        from provider.base import Message
    except Exception:
        Message = None  # type: ignore[assignment]

    if prompt_limit <= 0:
        return messages

    content_slots: list[tuple[int, str, int, str]] = []
    approx_total = 0
    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            content_tokens = executor._estimate_text_tokens(content)
            approx_total += content_tokens
            content_slots.append((idx, str(role or ""), content_tokens, content))

    if not content_slots:
        return messages

    target_prompt_budget = max(1024, int(prompt_limit * 0.82))
    current_total = prompt_count if prompt_count and prompt_count > 0 else approx_total
    if current_total <= target_prompt_budget and approx_total <= target_prompt_budget:
        return messages

    new_messages = list(messages)
    changed = False
    keep_ratio: float | None = None
    if prompt_count and prompt_count > target_prompt_budget:
        keep_ratio = max(0.12, min(0.95, target_prompt_budget / float(prompt_count)))

    compressed_indices: set[int] = set()
    while True:
        current_total = 0
        for msg in new_messages:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                current_total += executor._estimate_text_tokens(content)
        if current_total <= target_prompt_budget:
            break

        candidates: list[tuple[int, int, int, str]] = []
        for idx, role, tokens, content in content_slots:
            if idx in compressed_indices or not content:
                continue
            role_priority = 0 if role != "system" else 1
            candidates.append((tokens, role_priority, idx, content))
        if not candidates:
            break

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, idx, content = candidates[0]
        other_tokens = max(0, current_total - executor._estimate_text_tokens(content))
        message_budget = max(256, target_prompt_budget - other_tokens)
        if keep_ratio is not None:
            message_budget = max(256, int(message_budget * keep_ratio))
        trimmed_content = executor._compress_text_to_budget(content, message_budget)
        if trimmed_content == content:
            compressed_indices.add(idx)
            continue

        role = getattr(new_messages[idx], "role", "user")
        if Message is not None:
            new_messages[idx] = Message(role=role, content=trimmed_content)
        else:
            new_messages[idx] = type(new_messages[idx])(role=role, content=trimmed_content)
        compressed_indices.add(idx)
        changed = True

    return new_messages if changed else messages


async def _chat_with_retry_impl(
    executor: JudgmentExecutor,
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
        executor._set_last_call_meta(
            selection,
            thinking_override=thinking_override,
            skills=skills,
            primary_skill_name=primary_skill_name,
            primary_skill_guidance=primary_skill_guidance,
        )
        message_count, char_count, est_tokens = _message_log_stats(executor, messages)
        try:
            raw = await selected_provider.chat(messages, thinking_override=thinking_override)
            executor._mark_model_success(selection.model_ref)
            executor._track_token_usage(selected_provider)
            usage = getattr(selected_provider, "last_usage", None)
            prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
            completion_tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)) if isinstance(usage, dict) else 0
            _log.info(
                "%s [llm] ok model=%s tier=%s phase=%s thinking=%s attempt=%s messages=%s chars=%s est_tokens=%s usage_prompt=%s usage_completion=%s usage_total=%s skills=%s",
                log_prefix,
                selection.model_ref,
                selection.tier,
                selection.phase,
                thinking_override if thinking_override is not None else selection.thinking,
                _attempt + 1,
                message_count,
                char_count,
                est_tokens,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                skills,
            )
            return raw, selection, None
        except Exception as exc:
            last_error = exc
            _err = str(exc) or repr(exc)
            prompt_count, prompt_limit = executor._extract_prompt_limit(_err)
            if prompt_limit:
                try:
                    from provider.catalog import set_context_window_hint

                    set_context_window_hint(executor._extract_model_id(selection.model_ref), prompt_limit)
                except Exception:
                    pass
                trimmed_messages = executor._trim_messages_for_prompt_limit(
                    messages,
                    prompt_limit,
                    prompt_count=prompt_count,
                )
                if trimmed_messages is not messages:
                    _log.warning(
                        "%s LLM 提示词超限，自适应压缩后同模型重试: model=%s prompt=%s limit=%s attempt=%s messages=%s est_tokens=%s",
                        log_prefix,
                        selection.model_ref,
                        prompt_count,
                        prompt_limit,
                        _attempt + 1,
                        message_count,
                        est_tokens,
                    )
                    messages = trimmed_messages
                    continue

            executor._mark_model_failure(selection.model_ref, _err)
            if _attempt == 0:
                _fallback_tier = fallback_prefer_tier or executor._fallback_tiers(selection.tier)[0]
                fb_provider, fb_selection = executor._select_provider(
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


async def _repair_output_impl(
    executor: JudgmentExecutor,
    context_text: str,
    raw: str,
) -> JudgmentOutput | None:
    from provider.base import Message

    _ = context_text  # 保持签名兼容；repair 仅依赖 broken_output。
    compact_raw = raw
    if len(compact_raw) > 50000:
        compact_raw = compact_raw[:25000] + "\n...\n" + compact_raw[-25000:]

    repair_messages = [
        Message(
            role="system",
            content=(
                "你是一个严格的 JSON 修复器。"
                "只输出合法 JSON，不要解释，不要使用 markdown 代码块。"
                "必须遵循这个 schema: {decision, chosen_action_id, params, parallel_actions, delegate_tasks, rationale, reflection, reply_to_user, next_step, model_strategy}."
                "只根据 broken_output 修复 JSON，不要依赖原始判断上下文。"
                "如果原输出被截断，请尽量保留已经可见的字段并补全成合法 JSON。"
                "如果 broken_output 是裸代码（bash/python 脚本等），将代码原文放入 reply_to_user 字段，decision 设为 pause，rationale 说明代码已封装。"
            ),
        ),
        Message(
            role="user",
            content=(
                "下面是一段损坏/截断的模型输出，请修复为合法 JSON。\n\n"
                f"[broken_output]\n{compact_raw}\n\n"
                "只返回 JSON，不要用 markdown 代码块包裹。"
            ),
        ),
    ]

    try:
        repaired_raw = await executor._provider.chat(
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
