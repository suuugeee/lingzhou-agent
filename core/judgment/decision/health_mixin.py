"""core/judgment/decision/health_mixin.py — 模型健康与冷却。"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from core.judgment.output import ModelHealth

if TYPE_CHECKING:
    pass


class ExecutorHealthMixin:
    _model_health: dict[str, ModelHealth]
    _provider_errors: dict[str, str]

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
                return "quota"
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
            return 3600.0
        if code == "429":
            return min(180.0, 30.0 * streak)
        if code == "402":
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
        health.last_error = err_text
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
