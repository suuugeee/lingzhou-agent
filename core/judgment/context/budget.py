"""core/judgment/context/budget.py — 判断上下文 token/字符预算裁剪。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .utils import _cache_put, _context_fmt_cache, _estimate_tokens
from core.config.budget import adaptive_judgment_input_budget, context_window_input_hard_budget
from provider.catalog import resolve_context_window


def apply_context_budget(
    ctx: dict[str, str],
    token_budget: int | None = None,
    max_chars: int | None = None,
    skill_min_tokens: int = 0,
) -> dict[str, str]:
    if token_budget is None:
        token_budget = max_chars
    if token_budget is None:
        raise TypeError("apply_context_budget() missing required argument: 'token_budget'")
    if token_budget <= 0:
        return ctx

    ctx_hash = hashlib.md5("".join(sorted(ctx.values())).encode()).hexdigest()
    cache_key = f"budget:{ctx_hash}:{token_budget}:{skill_min_tokens}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]

    budgeted = dict(ctx)
    priority = [
        "skills_catalog_section",
        "current_interlocutor_profile_section",
        "current_interlocutor_continuity_section",
        "chat_memory_section",
        "memories_section",
        "cross_task_episodic_section",
        "chat_continuity_section",
        "daily_continuity_section",
        "episodic_section",
        "skills_section",
        "wm_proposal_sections",
        "wm_section",
        "tools_section",
    ]

    def total_tokens(items: dict[str, str]) -> int:
        return sum(_estimate_tokens(value) for value in items.values())

    current_total = total_tokens(budgeted)
    if current_total <= token_budget:
        return budgeted

    drop_order = list(reversed(priority))
    drop_order.extend(key for key in budgeted if key not in priority and key != "user_message")

    for key in drop_order:
        if current_total <= token_budget:
            break
        original = budgeted.get(key, "")
        if not original:
            continue
        budgeted[key] = ""
        current_total -= _estimate_tokens(original)

    _cache_put(cache_key, budgeted)
    return budgeted


def resolve_judgment_prompt_budget(cfg: Any, model_ref: str, *, catalog_path: Path | None = None) -> int:
    """计算单次 judgment LLM 调用的有效输入预算（token）。

    优先级：
    1. 模型上下文窗口（当前模型若有 catalog 记录，使用动态窗口）
    2. `max_judgment_input_tokens`（显式上限）
    3. 默认安全兜底（避免高 context window 模型一次性喂入超长上下文）
    """
    model_id = model_ref.split("/", 1)[1] if "/" in model_ref else model_ref
    context_window = resolve_context_window(
        model_id,
        cfg.context_window_tokens if model_ref == cfg.model else None,
        catalog_path=Path(catalog_path) if catalog_path is not None else None,
    )
    fallback_budget = 16_000

    max_limit = getattr(cfg, "max_judgment_input_tokens", None)
    if max_limit is not None:
        max_limit = int(max_limit)

    if context_window is None or context_window <= 0:
        if max_limit and max_limit > 0:
            return max_limit
        return fallback_budget

    hard_budget = context_window_input_hard_budget(context_window)
    if max_limit is not None and max_limit > 0:
        return min(hard_budget, max_limit)
    return adaptive_judgment_input_budget(context_window)
