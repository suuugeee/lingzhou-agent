"""core/judgment/context/budget.py — 判断上下文 token/字符预算裁剪。"""
from __future__ import annotations

import hashlib

from .utils import _cache_put, _context_fmt_cache, _estimate_tokens


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
