"""core/judgment/decision/prompt_mixin.py — prompt 超限检测与整消息省略（见 ADR 0015）。"""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.judgment.executor import JudgmentExecutor

_PROMPT_LIMIT_RE = re.compile(r"prompt token count of\s*(\d+)\s*exceeds the limit of\s*(\d+)", re.IGNORECASE)
_PROMPT_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    _PROMPT_LIMIT_RE,
    re.compile(r"maximum context length is\s*(\d+)\s*tokens", re.IGNORECASE),
    re.compile(r"context_length_exceeded\s*[:=]\s*(\d+)", re.IGNORECASE),
    re.compile(r"model\W*s max context length is\s*(\d+)", re.IGNORECASE),
)
_OUTPUT_AVAILABLE_RE = re.compile(
    r"available[_\s-]?tokens\s*[:=]\s*(\d+)",
    re.IGNORECASE,
)
_OUTPUT_WINDOW_RE = re.compile(
    r"max[_\s-]?tokens\s*[:=]?\s*(\d+)\s*>\s*context[_\s-]?window\s*[:=]?\s*(\d+)\s*-\s*input[_\s-]?tokens\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(r"retry[-_ ]?after\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_RETRY_IN_SECONDS_RE = re.compile(r"retry(?:ing)?[^\n]*?in\s*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class ExecutorPromptMixin:
    @staticmethod
    def _extract_model_id(model_ref: str) -> str:
        return model_ref.split("/", 1)[1] if "/" in model_ref else model_ref

    @staticmethod
    def _extract_prompt_limit(err_text: str) -> tuple[int | None, int | None]:
        text = err_text or ""
        match = _PROMPT_LIMIT_RE.search(text)
        if match:
            try:
                prompt = int(match.group(1))
                limit = int(match.group(2))
                return prompt, limit
            except Exception:
                return None, None
        for pattern in _PROMPT_LIMIT_PATTERNS[1:]:
            m = pattern.search(text)
            if not m:
                continue
            try:
                return None, int(m.group(1))
            except Exception:
                continue
        return None, None

    @staticmethod
    def _extract_available_output_tokens(err_text: str) -> int | None:
        text = err_text or ""
        match = _OUTPUT_AVAILABLE_RE.search(text)
        if match:
            try:
                value = int(match.group(1))
                return value if value > 0 else None
            except Exception:
                return None
        match = _OUTPUT_WINDOW_RE.search(text)
        if not match:
            return None
        try:
            _, context_window, input_tokens = match.groups()
            value = int(context_window) - int(input_tokens)
            return value if value > 0 else None
        except Exception:
            return None

    @staticmethod
    def _is_output_overflow_error(err_text: str) -> bool:
        text = (err_text or "").lower()
        if "max_tokens" in text and "available_tokens" in text:
            return True
        return (
            "max_tokens" in text
            and "context_window" in text
            and "input_tokens" in text
        )

    @staticmethod
    def _extract_retry_after_seconds(err_text: str, exc: Exception | None = None) -> float | None:
        if exc is not None:
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                value = headers.get("retry-after") if hasattr(headers, "get") else None
                if isinstance(value, str):
                    try:
                        sec = float(value.strip())
                        if sec >= 0:
                            return sec
                    except Exception:
                        pass
        text = err_text or ""
        for pattern in (_RETRY_AFTER_RE, _RETRY_IN_SECONDS_RE):
            m = pattern.search(text)
            if not m:
                continue
            try:
                sec = float(m.group(1))
                if sec >= 0:
                    return sec
            except Exception:
                continue
        return None

    @staticmethod
    def _retry_delay_seconds(
        attempt: int,
        *,
        base_delay: float = 1.2,
        max_delay: float = 30.0,
        retry_after_seconds: float | None = None,
    ) -> float:
        exp = max(0, attempt - 1)
        delay = min(max_delay, base_delay * (2**exp))
        if retry_after_seconds is not None:
            delay = max(delay, retry_after_seconds)
            jitter = random.uniform(0.0, min(0.2 * delay, 1.5))
            return min(max_delay, delay + jitter)
        jitter = random.uniform(-0.2 * delay, 0.2 * delay)
        return max(0.1, min(max_delay, delay + jitter))

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        other = len(text) - cjk - ascii_chars
        return max(1, int(cjk * 1.8 + ascii_chars * 0.3 + other * 1.0))

    def _trim_messages_for_prompt_limit(
        self: JudgmentExecutor,
        messages: list[Any],
        prompt_limit: int,
        *,
        prompt_count: int | None = None,
        tight: bool = False,
    ) -> list[Any]:
        from core.judgment.decision.helpers import _trim_messages_for_prompt_limit_impl

        return _trim_messages_for_prompt_limit_impl(
            self,
            messages,
            prompt_limit,
            prompt_count=prompt_count,
            tight=tight,
        )
