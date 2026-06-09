"""core/judgment/decision/helpers.py — 判断层 LLM 调用与路由实现。"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from core.judgment.output import JudgmentOutput, ModelSelection
from core.log_fields import format_log_fields, llm_call_fields
from core.judgment.context.budget import resolve_judgment_prompt_budget

if TYPE_CHECKING:
    from core.judgment.executor import JudgmentExecutor
    from provider.base import Provider

_log = logging.getLogger("lingzhou.judgment")

# 整消息省略说明（进模型 prompt）；不做正文头尾切片——见 ADR 0015。
_PROMPT_OVERFLOW_OMIT_STUB = (
    "[该条消息因模型上下文窗口未纳入本轮请求；"
    "完整证据见本轮 WM、工具历史或 session 记录。]"
)
_PROMPT_OVERFLOW_TIGHT_STUB = "[省略]"


def _role_drop_priority(role: str, *, is_last_message: bool) -> int:
    """数值越小越先用 stub 替换整条消息（不切片正文）。tool/assistant 优先，system 最后。"""
    if role == "tool":
        return 0
    if role == "assistant":
        return 1
    if role == "user":
        return 3 if is_last_message else 2
    if role == "system":
        return 4
    return 50


def _llm_scope(selection: ModelSelection, **extra: Any) -> str:
    return llm_call_fields(
        model_ref=selection.model_ref,
        tier=selection.tier,
        phase=selection.phase,
        **extra,
    )


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

    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for cand_tier in (tier, *executor._fallback_tiers(tier, exclude_reader=exclude_reader)):
        for model_ref in executor._tier_model_candidates(cand_tier, routing_overrides=routing_overrides):
            duplicate_key = (cand_tier, model_ref)
            if duplicate_key in seen:
                continue
            seen.add(duplicate_key)
            candidates.append((cand_tier, model_ref))

    for cand_tier, model_ref in candidates:
        if not executor._is_model_available(model_ref):
            continue
        try:
            provider = executor._find_or_create_provider(model_ref)
            chosen_tier = cand_tier
            chosen_model = model_ref
            selected = True
            break
        except Exception as e:
            executor._mark_model_failure(model_ref, str(e) or repr(e))
            code = executor._get_health(model_ref).last_code or "other"
            _log.warning(
                "[routing] %s provider_build_failed code=%s err=%s",
                format_log_fields(tier=cand_tier, model_ref=model_ref),
                code,
                e,
            )
        if selected:
            break

    if not selected:
        fallback_ref = executor._least_bad_model(tier, routing_overrides, exclude_reader=exclude_reader)
        if fallback_ref:
            try:
                provider = executor._find_or_create_provider(fallback_ref)
                chosen_tier = tier
                chosen_model = fallback_ref
                if exclude_reader:
                    _log.info("[routing] 全部可用模型被冷却，强制使用冷却最短模型: %s", fallback_ref)
                else:
                    _log.info("[routing] 全部 reasoner/reader/repair 冷却，强制使用冷却最短模型: %s", fallback_ref)
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
    tight: bool = False,
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

    def _estimate_messages_total(msgs: list[Any]) -> int:
        total = 0
        for msg in msgs:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                total += executor._estimate_text_tokens(content)
        return total

    new_messages = list(messages)
    last_index = len(new_messages) - 1
    changed = False
    omitted_indices: set[int] = set()

    if tight:
        for idx, msg in enumerate(new_messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str) or content.strip() != _PROMPT_OVERFLOW_OMIT_STUB:
                continue
            role = str(getattr(msg, "role", "") or "user")
            if Message is not None:
                new_messages[idx] = Message(role=role, content=_PROMPT_OVERFLOW_TIGHT_STUB)
            else:
                new_messages[idx] = type(msg)(role=role, content=_PROMPT_OVERFLOW_TIGHT_STUB)
            changed = True

    while _estimate_messages_total(new_messages) > target_prompt_budget:
        droppable: list[tuple[int, int, int]] = []
        for idx, role, tokens, content in content_slots:
            if idx in omitted_indices or not content:
                continue
            if content.strip() in {_PROMPT_OVERFLOW_OMIT_STUB, _PROMPT_OVERFLOW_TIGHT_STUB}:
                continue
            droppable.append(
                (_role_drop_priority(role, is_last_message=(idx == last_index)), idx, tokens)
            )
        if not droppable:
            break
        droppable.sort(key=lambda item: (item[0], item[1]))
        _, drop_idx, _ = droppable[0]
        role = str(getattr(new_messages[drop_idx], "role", "") or "user")
        replacement = _PROMPT_OVERFLOW_TIGHT_STUB if tight else _PROMPT_OVERFLOW_OMIT_STUB
        if Message is not None:
            new_messages[drop_idx] = Message(role=role, content=replacement)
        else:
            new_messages[drop_idx] = type(new_messages[drop_idx])(role=role, content=replacement)
        omitted_indices.add(drop_idx)
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
    max_attempts = 3
    call_timeout = _configured_llm_timeout(executor._cfg)
    for _attempt in range(max_attempts):
        executor._set_last_call_meta(
            selection,
            thinking_override=thinking_override,
            skills=skills,
            primary_skill_name=primary_skill_name,
            primary_skill_guidance=primary_skill_guidance,
        )
        prompt_budget = resolve_judgment_prompt_budget(
            executor._cfg,
            selection.model_ref,
            catalog_path=executor._cfg.workspace_dir / "models.json",
        )
        message_count, char_count, est_tokens = _message_log_stats(executor, messages)
        if est_tokens > prompt_budget:
            trimmed_messages = executor._trim_messages_for_prompt_limit(
                messages,
                prompt_budget,
                prompt_count=est_tokens,
            )
            if trimmed_messages is not messages:
                _log.warning(
                    "%s [llm] proactive_prompt_trim %s messages=%s chars=%s est_tokens=%s limit=%s",
                    log_prefix,
                    _llm_scope(selection, attempt=_attempt + 1, proactive=True),
                    message_count,
                    char_count,
                    est_tokens,
                    prompt_budget,
                )
                messages = trimmed_messages
                message_count, char_count, est_tokens = _message_log_stats(executor, messages)
        try:
            chat_coro = selected_provider.chat(messages, thinking_override=thinking_override)
            raw = await asyncio.wait_for(chat_coro, timeout=call_timeout) if call_timeout is not None else await chat_coro
            executor._mark_model_success(selection.model_ref)
            executor._track_token_usage(selected_provider)
            usage = getattr(selected_provider, "last_usage", None)
            prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
            completion_tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)) if isinstance(usage, dict) else 0
            usage_source = str(usage.get("usage_source") or "missing") if isinstance(usage, dict) else "missing"
            scope = _llm_scope(
                selection,
                usage_source=usage_source,
                thinking=thinking_override if thinking_override is not None else selection.thinking,
                attempt=_attempt + 1,
            )
            _log.info(
                "%s [llm] ok %s messages=%s chars=%s est_tokens=%s usage_prompt=%s usage_completion=%s usage_total=%s skills=%s",
                log_prefix,
                scope,
                message_count,
                char_count,
                est_tokens,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                skills,
            )
            return raw, selection, None
        except TimeoutError as exc:
            _err = f"llm call timeout {call_timeout:.1f}s" if call_timeout is not None else "llm call timeout"
            last_error = TimeoutError(_err)
            _log.warning(
                "%s LLM timeout %s attempt=%s/%s err=%s",
                log_prefix,
                _llm_scope(selection, attempt=_attempt + 1),
                _attempt + 1,
                max_attempts,
                _err,
            )
            executor._mark_model_failure(selection.model_ref, _err)
            if _attempt < max_attempts - 1:
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
                        "%s LLM failover %s -> %s overflow_kind=timeout attempt=%s/%s err=%s",
                        log_prefix,
                        _llm_scope(selection),
                        _llm_scope(fb_selection),
                        _attempt + 1,
                        max_attempts,
                        _err,
                    )
                    selected_provider, selection = fb_provider, fb_selection
                    continue
                retry_after = executor._extract_retry_after_seconds(_err, exc)
                delay = executor._retry_delay_seconds(
                    _attempt + 1,
                    retry_after_seconds=retry_after,
                )
                _log.warning(
                    "%s LLM backoff %s delay_sec=%.2f retry_after=%s err=%s",
                    log_prefix,
                    _llm_scope(
                        selection,
                        overflow_kind="timeout",
                        attempt=f"{_attempt + 1}/{max_attempts}",
                    ),
                    delay,
                    f"{retry_after:.2f}s" if retry_after is not None else "none",
                    _err,
                )
                await asyncio.sleep(delay)
                continue
            continue
        except Exception as exc:
            last_error = exc
            _err = str(exc) or repr(exc)
            prompt_count, prompt_limit = executor._extract_prompt_limit(_err)
            is_output_overflow = executor._is_output_overflow_error(_err)
            overflow_kind = "output" if is_output_overflow else ("prompt" if prompt_limit else "none")
            # Copilot / 网关偶发 413 Request Entity Too Large：服务端不一定返回可解析的 prompt_limit。
            # 该错误本质仍是“请求体过大”，优先尝试用既有的 prompt_limit 修剪逻辑来压缩 messages，
            # 以避免简单失败后直接切模型/空转。
            if not prompt_limit and not is_output_overflow:
                lowered = _err.lower()
                if "413" in lowered or "request entity too large" in lowered or "payload too large" in lowered:
                    cfg_limit = int(getattr(executor._cfg, "max_judgment_input_tokens", 0) or 0)
                    prompt_limit = cfg_limit if cfg_limit > 0 else 8000
                    overflow_kind = "request_entity_too_large"
            if prompt_limit and not is_output_overflow:
                try:
                    from provider.catalog import set_context_window_hint

                    set_context_window_hint(executor._extract_model_id(selection.model_ref), prompt_limit)
                except Exception:
                    pass
                trimmed_messages = executor._trim_messages_for_prompt_limit(
                    messages,
                    prompt_limit,
                    prompt_count=prompt_count,
                    tight=True,
                )
                if trimmed_messages is not messages:
                    _log.warning(
                        "%s LLM prompt_overflow %s messages_omitted=true prompt=%s limit=%s messages=%s est_tokens=%s",
                        log_prefix,
                        _llm_scope(selection, overflow_kind=overflow_kind, attempt=_attempt + 1),
                        prompt_count,
                        prompt_limit,
                        message_count,
                        est_tokens,
                    )
                    messages = trimmed_messages
                    continue

            if is_output_overflow:
                available_output = executor._extract_available_output_tokens(_err)
                _log.warning(
                    "%s LLM output_overflow %s messages_omitted=false available_output=%s max_attempts=%s err=%s",
                    log_prefix,
                    _llm_scope(
                        selection,
                        overflow_kind=overflow_kind,
                        attempt=f"{_attempt + 1}/{max_attempts}",
                    ),
                    available_output,
                    max_attempts,
                    _err,
                )

            executor._mark_model_failure(selection.model_ref, _err)
            if _attempt < max_attempts - 1:
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
                        "%s LLM failover %s -> %s overflow_kind=%s attempt=%s/%s err=%s",
                        log_prefix,
                        _llm_scope(selection, overflow_kind=overflow_kind),
                        _llm_scope(fb_selection),
                        overflow_kind,
                        _attempt + 1,
                        max_attempts,
                        _err,
                    )
                    selected_provider, selection = fb_provider, fb_selection
                    continue
                retry_after = executor._extract_retry_after_seconds(_err, exc)
                delay = executor._retry_delay_seconds(
                    _attempt + 1,
                    retry_after_seconds=retry_after,
                )
                _log.warning(
                    "%s LLM backoff %s delay_sec=%.2f retry_after=%s err=%s",
                    log_prefix,
                    _llm_scope(
                        selection,
                        overflow_kind=overflow_kind,
                        attempt=f"{_attempt + 1}/{max_attempts}",
                    ),
                    delay,
                    f"{retry_after:.2f}s" if retry_after is not None else "none",
                    _err,
                )
                await asyncio.sleep(delay)
                continue
            _log.warning("%s LLM failed %s err=%s", log_prefix, _llm_scope(selection), _err)
    return raw, selection, last_error


async def _repair_output_impl(
    executor: JudgmentExecutor,
    context_text: str,
    raw: str,
) -> JudgmentOutput | None:
    from provider.base import Message

    _ = context_text  # repair 仅依赖 broken_output，完整上下文由调用层保留。
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
        _, repair_model_ref = executor._resolve_tier_model("repair")
        repair_provider = executor._find_or_create_provider(repair_model_ref)
        _log.info("[judgment] repair %s", format_log_fields(tier="repair", model_ref=repair_model_ref))
        repair_coro = repair_provider.chat(repair_messages, temperature=0.0)
        repair_timeout = _configured_llm_timeout(executor._cfg)
        repaired_raw = await asyncio.wait_for(repair_coro, timeout=repair_timeout) if repair_timeout is not None else await repair_coro
    except Exception as exc:
        _log.warning("[judgment] repair request failed: %s", exc)
        return None

    repaired = JudgmentOutput.from_llm(repaired_raw)
    if repaired.rationale.startswith("LLM 输出解析失败"):
        _log.warning("[judgment] repair failed: %s", repaired.rationale)
        return None

    _log.info(
        "[judgment] repair_ok %s",
        format_log_fields(tier="repair", model_ref=repair_model_ref),
    )
    return repaired


def _configured_llm_timeout(cfg: Any) -> float | None:
    raw = getattr(cfg, "timeout", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if value > 0 else None
