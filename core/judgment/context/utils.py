"""core/judgment/context/utils.py — 缓存、模板、schema、token 估算（判断上下文共享工具）。"""
from __future__ import annotations

import functools
import json
import logging
import re
from collections import OrderedDict
from typing import Any

_log = logging.getLogger("lingzhou.judgment")

_context_fmt_cache: OrderedDict[str, Any] = OrderedDict()
_MAX_CONTEXT_CACHE_SIZE = 512


def _cache_put(key: str, value: Any) -> None:
    """写入缓存；超过上限时 LRU 驱逐最旧半数（而非全清），保留热数据。"""
    if len(_context_fmt_cache) >= _MAX_CONTEXT_CACHE_SIZE:
        for _ in range(_MAX_CONTEXT_CACHE_SIZE // 2):
            _context_fmt_cache.popitem(last=False)
    _context_fmt_cache[key] = value


def _clear_context_cache() -> None:
    """在每 tick 开头调用，清除上一 tick 的所有缓存。"""
    _context_fmt_cache.clear()


def _run_summary(run: Any) -> str:
    if run.error_text:
        return f"error: {run.error_text.strip()}"
    for key in ("summary", "result", "message", "reply_to_user"):
        value = run.output_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if run.log_text.strip():
        return run.log_text.strip()
    return ""


def _clip_text(text: str, limit: int = 0) -> str:
    """仅归一化空白；不按 limit 截断正文（ADR 0015）。limit 保留供调用方签名兼容。"""
    _ = limit
    return " ".join((text or "").split())


def _clip_for_context(text: str, limit: int = 160) -> str:
    """为模型上下文增加文本上限，保留开头/结尾与省略提示。"""
    value = " ".join((text or "").split())
    if limit <= 0 or len(value) <= limit:
        return value
    half = max(16, (limit - 9) // 2)
    omitted = max(0, len(value) - half * 2)
    return f"{value[:half]} ...({omitted} chars omitted)... {value[-half:]}"


def _format_fact_value(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "（空）"
    try:
        payload = json.loads(text)
    except Exception:
        return _clip_text(text, 180)
    if isinstance(payload, dict):
        parts = [f"{key}={payload[key]}" for key in sorted(payload)]
        return _clip_text("; ".join(parts), 180)
    if isinstance(payload, list):
        return _clip_text(", ".join(str(item) for item in payload), 180)
    return _clip_text(str(payload), 180)


def _fill_template(template: str, ctx: dict[str, Any]) -> str:
    missing = sorted({
        match.group(1).strip()
        for match in re.finditer(r"\{\{([^}]+)\}\}", template)
        if match.group(1).strip() not in ctx
    })
    if missing:
        msg = "[judgment] 模板变量缺失: " + ", ".join(missing)
        _log.error("%s（judgment.md 与 context 组装已失配）", msg)
        raise ValueError(msg)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return str(ctx[key])

    return re.sub(r"\{\{([^}]+)\}\}", replace, template)


@functools.lru_cache(maxsize=8192)
def _estimate_tokens(text: str) -> int:
    if not text:
        return 0

    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other = len(text) - cjk - ascii_chars

    return max(1, int(cjk * 1.8 + ascii_chars * 0.3 + other * 1.0))


def _split_segments(text: str) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    segments: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n", part):
            if buffer:
                segments.append(buffer)
                buffer = ""
            segments.append(part)
        else:
            buffer += part
    if buffer:
        segments.append(buffer)
    return segments


def _compress_single_segment(text: str, keep_tokens: int) -> str:
    """测试/工具用；不用于模型可见 judgment 组装（ADR 0015）。"""
    lines = text.splitlines(keepends=True)
    if len(lines) <= 1:
        return text[: max(1, min(len(text), keep_tokens * 4))]

    kept: list[str] = []
    token_count = 0
    for line in lines:
        line_tokens = _estimate_tokens(line)
        if token_count + line_tokens > keep_tokens:
            break
        kept.append(line)
        token_count += line_tokens

    if kept:
        return "".join(kept) + ("\n[...省略...]" if len(kept) < len(lines) else "")
    return text[: max(1, min(len(text), keep_tokens * 4))]


def _compress_text_segments(text: str, keep_tokens: int) -> str:
    """测试/工具用；不用于模型可见 judgment 组装（ADR 0015）。"""
    if keep_tokens <= 0:
        return ""
    if _estimate_tokens(text) <= keep_tokens:
        return text

    segments = _split_segments(text)
    if not segments:
        return ""

    keep_head: list[str] = []
    keep_tail: list[str] = []
    head_tokens = 0
    tail_tokens = 0
    head_idx = 0
    tail_idx = len(segments) - 1
    turn = 0

    while head_idx <= tail_idx:
        if turn % 2 == 0:
            candidate = segments[head_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_head.append(candidate)
                head_tokens += candidate_tokens
                head_idx += 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_head.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        else:
            candidate = segments[tail_idx]
            candidate_tokens = _estimate_tokens(candidate)
            if head_tokens + tail_tokens + candidate_tokens <= keep_tokens:
                keep_tail.append(candidate)
                tail_tokens += candidate_tokens
                tail_idx -= 1
            elif tail_idx == head_idx and not keep_head and not keep_tail:
                keep_tail.append(_compress_single_segment(candidate, keep_tokens))
                break
            else:
                break
        turn += 1

    if not keep_head and not keep_tail:
        return _compress_single_segment(text, keep_tokens)

    body = keep_head + (["\n[...省略...]\n"] if head_idx <= tail_idx else []) + keep_tail[::-1]
    result = "".join(body)
    open_chars = "([{"
    close_chars = ")] }".replace(" ", "")
    stack = []
    for ch in result:
        if ch in open_chars:
            stack.append(close_chars[open_chars.index(ch)])
        elif ch in close_chars and stack and stack[-1] == ch:
            stack.pop()
    return result + "".join(reversed(stack))


_CONTEXT_SCHEMA_KEYS = ["identity", "tasks", "memory", "perception"]


def _validate_context_schema(ctx: dict) -> tuple[bool, str]:
    """严格校验上下文结构，防止畸形数据注入。"""
    missing = [k for k in _CONTEXT_SCHEMA_KEYS if k not in ctx]
    if missing:
        return False, f"缺少必需字段: {', '.join(missing)}"
    return True, "ok"
