"""provider/openai_compat.py — OpenAI 兼容接口实现（百炼/qwen/openai/copilot 等）。"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import httpx

from provider.base import Message
from provider.catalog import lookup_model
from provider.openai_compat_helpers import (
    _EMBED_MAX_CHARS,
    _LEVEL_FRACS,
    _MAX_COMPLETION_TOKENS_DEFAULT,
    COPILOT_TOKEN_URL,
    DEFAULT_COPILOT_API_BASE_URL,
    _build_copilot_ide_headers,
    _copilot_reasoning_effort,
    _derive_copilot_api_base_url_from_token,
    _extract_responses_text,
    _normalize_copilot_api_base_url,
    _normalize_responses_message_content,
    _raise_for_status_with_body,
)
from store.auth import (
    load_copilot_token_cache,
    resolve_copilot_token,
    save_copilot_token_cache,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from core.config import Config

_log = logging.getLogger("lingzhou.provider.openai_compat")


def _token_state(token: str) -> str:
    return "nonempty" if str(token or "").strip() else "empty"


def _request_timeout_override(client: Any, level: str | None) -> float | None:
    """Return per-request timeout only when caller explicitly configured one.

    With cfg.timeout=None, httpx clients are constructed without a local timeout and
    requests pass timeout=None, leaving timeout behavior to the provider/gateway.
    """
    if level in (None, "off"):
        return None
    try:
        timeout = getattr(getattr(client, "timeout", None), "read", None)
        return float(timeout) if timeout is not None else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 模式适配器：用数据封装 openai / copilot 的差异，消除 if/elif 堆砌
# ═══════════════════════════════════════════════════════════════════════════════

class _ModeAdapter:
    """模式差异的抽象基类。每个具体模式只需覆写差异方法。"""

    def __init__(self, base_url: str, api_key: str, timeout: float | None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def build_sync_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )

    def build_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=20),
        )

    def resolve_url(self, path: str) -> str:
        return path  # openai 模式：base_url 已设在 client 上

    async def request_headers(self) -> dict[str, str]:
        return {}  # openai 模式：Authorization 已在 client headers

    def embedding_url(self) -> str:
        return "/embeddings"


class _OpenAIMode(_ModeAdapter):
    """标准 OpenAI 兼容模式：百炼、DeepSeek 等。"""


class _CopilotMode(_ModeAdapter):
    """GitHub Copilot 模式：token exchange + IDE headers + responses API。"""

    def __init__(self, base_url: str, api_key: str, timeout: float | None):
        super().__init__(base_url, api_key, timeout)
        self._copilot_api_base_url = _normalize_copilot_api_base_url(base_url)
        self._copilot_gh_token: str = api_key
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0.0

    def build_sync_client(self) -> httpx.Client:
        return httpx.Client(headers={"Content-Type": "application/json"}, timeout=30.0)

    def build_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=20),
        )

    def resolve_url(self, path: str) -> str:
        return f"{self._copilot_api_base_url}{path}"

    async def request_headers(self) -> dict[str, str]:
        token = await self._ensure_copilot_token()
        return _build_copilot_ide_headers() | {"Authorization": f"Bearer {token}"}

    # ── Copilot 内部方法 ─────────────────────────────────────────────────

    def _copilot_url(self, path: str) -> str:
        return f"{self._copilot_api_base_url}{path}"

    async def _ensure_copilot_token(self, *, force_refresh: bool = False) -> str:
        if (not force_refresh) and self._copilot_token and time.time() < self._copilot_token_expires - 300:
            return self._copilot_token

        cache = load_copilot_token_cache()
        if (not force_refresh) and cache and (time.time() * 1000) < cache.expires_at_ms - 300_000:
            cached_token = str(getattr(cache, "token", "") or "").strip()
            # 缓存里若是空 token，会导致 Authorization: Bearer 触发 httpx Illegal header value。
            # 将其视为无效缓存，继续走 token exchange 刷新。
            if cached_token:
                self._copilot_token = cached_token
                self._copilot_token_expires = cache.expires_at_ms / 1000
                self._copilot_api_base_url = (
                    _derive_copilot_api_base_url_from_token(cached_token)
                    or DEFAULT_COPILOT_API_BASE_URL
                )
                return self._copilot_token

        try:
            async with httpx.AsyncClient(timeout=15.0) as tmp:
                resp = await tmp.get(
                    COPILOT_TOKEN_URL,
                    headers={
                        "Authorization": f"token {self._copilot_gh_token}",
                        "Accept": "application/json",
                        **_build_copilot_ide_headers(include_api_version=False),
                    },
                )
            _raise_for_status_with_body(resp)
            data = resp.json()
            token = str(data.get("token", "")).strip()
            if not token:
                raise RuntimeError("Copilot token exchange succeeded but returned empty token")
            expires_str = str(data.get("expires_at", "")).strip()
            if expires_str:
                if expires_str.isdigit():
                    expires_at_ms = int(expires_str) * 1000
                else:
                    dt = datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
                    expires_at_ms = int(dt.timestamp() * 1000)
            else:
                expires_at_ms = int((time.time() + 1800) * 1000)
            self._copilot_token = token
            self._copilot_token_expires = expires_at_ms / 1000
            self._copilot_api_base_url = (
                _derive_copilot_api_base_url_from_token(token)
                or DEFAULT_COPILOT_API_BASE_URL
            )
            save_copilot_token_cache(token, expires_at_ms=expires_at_ms)
            return self._copilot_token
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403, 404):
                raise RuntimeError(
                    "GitHub token 无法完成 Copilot token exchange。\n"
                    "请重新执行 `lingzhou auth login-copilot`。"
                ) from exc
            raise


def _build_mode_adapter(
    *,
    mode: str,
    base_url: str,
    api_key: str = "",
    api_key_env: str | None = None,
    timeout: float | None,
) -> _ModeAdapter:
    """根据 provider.mode 创建对应的适配器。

    接收已解析的 api_key（调用方负责凭证解析），
    这是唯一需要 if 的地方——模式选择在构造时完成一次，
    之后所有模式差异通过多态分发。
    """
    resolved_api_key = api_key
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.environ.get(api_key_env, "").strip()

    if mode == "copilot":
        if not resolved_api_key:
            raise OSError(
                "未找到 Copilot 的 GitHub token。\n"
                "Lingzhou 使用：GitHub token → Copilot token exchange → Copilot API\n"
                "请执行以下任一操作：\n"
                "  lingzhou auth login-copilot\n"
                "  export COPILOT_GITHUB_TOKEN=your_token\n"
                "  export GH_TOKEN=your_token\n"
                "  export GITHUB_TOKEN=your_token"
            )
        return _CopilotMode(base_url, resolved_api_key, timeout)

    # openai 模式（百炼、DeepSeek 等标准 OpenAI 兼容）
    if not resolved_api_key:
        missing_hint = f"{api_key_env!r}"
        raise OSError(
            f"OpenAI 兼容 provider 的 API key 为空（{missing_hint}）。"
            "请执行 `lingzhou auth bailian` 或设置对应环境变量。"
        )
    return _OpenAIMode(base_url, resolved_api_key, timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider 主体
# ═══════════════════════════════════════════════════════════════════════════════


class OpenAICompatProvider:
    """OpenAI 兼容接口。

    模式差异通过 _ModeAdapter 封装，避免 if/elif 堆砌：
    - mode=openai: 标准 OpenAI 兼容（百炼/DeepSeek），Bearer token + base_url
    - mode=copilot: GitHub Copilot，token exchange + IDE headers + responses API
    """

    def __init__(self, cfg: Config) -> None:
        provider = cfg.active_provider
        self.model_ref = cfg.model
        self._model = cfg.active_model_id
        self._catalog_path = cfg.workspace_dir / "models.json"
        self._temperature = cfg.temperature
        self._thinking_level = cfg.thinking
        self._extra_body: dict[str, Any] = dict(provider.extra_body)
        self._base_url = provider.base_url.rstrip("/")
        self._embed_model: str | None = cfg.memory.embedding_model
        self._provider_mode = provider.mode

        # 凭证解析：openai 用 provider.api_key（env→auth-profile）
        #           copilot 用 resolve_copilot_token（COPILOT_ENV_ORDER + profile）
        if provider.mode == "copilot":
            _res = resolve_copilot_token(provider.api_key_env)
            _resolved_key = _res.token if _res else ""
        else:
            _resolved_key = provider.api_key

        # 模式适配器：封装 openai / copilot 的差异行为
        self._mode = _build_mode_adapter(
            mode=provider.mode,
            base_url=self._base_url,
            api_key=_resolved_key,
            timeout=cfg.timeout,
        )

        self._sync_client = self._mode.build_sync_client()
        self._client = self._mode.build_async_client()
        self.last_usage: dict[str, int] = {}  # 最近一次 API 调用的 token 用量

    def _resolve_url(self, path: str) -> str:
        _m = self.__dict__.get("_mode")
        if _m is not None:
            return _m.resolve_url(path)
        # Fallback for test injection: _copilot_url overridden as instance attr
        _cu = self.__dict__.get("_copilot_url")
        if callable(_cu):
            return str(cast("Callable[[str], object]", _cu)(path))
        base = self._copilot_api_base_url or self.__dict__.get("_base_url", "")
        return f"{str(base).rstrip('/')}{path}"

    async def _request_headers(self) -> dict[str, str]:
        _m = self.__dict__.get("_mode")
        if _m is not None:
            return await _m.request_headers()
        # Fallback for test injection
        _ensure_token = self.__dict__.get("_ensure_copilot_token")
        _request_headers = self.__dict__.get("_copilot_request_headers")
        if callable(_ensure_token) and callable(_request_headers):
            token = str(await cast("Callable[..., Awaitable[str]]", _ensure_token)()).strip()
            mode = str(getattr(self, "_provider_mode", "unknown"))
            model_ref = str(getattr(self, "model_ref", getattr(self, "_model", "unknown")))
            _log.debug(
                "[copilot.auth] source=fallback mode=%s model_ref=%s token_state=%s",
                mode,
                model_ref,
                _token_state(token),
            )
            if not token:
                raise RuntimeError("Copilot token 为空，拒绝构造 Authorization header")
            return cast("Callable[[str], dict[str, str]]", _request_headers)(token)
        return {}

    async def _copilot_refreshed_headers(self) -> dict[str, str]:
        """Force-refresh Copilot token and return new request headers."""
        _m = self.__dict__.get("_mode")
        if _m is not None and hasattr(_m, "_ensure_copilot_token"):
            token = str(await _m._ensure_copilot_token(force_refresh=True)).strip()  # type: ignore[union-attr]
            mode = str(getattr(self, "_provider_mode", "unknown"))
            model_ref = str(getattr(self, "model_ref", getattr(self, "_model", "unknown")))
            _log.debug(
                "[copilot.auth] source=refresh mode=%s model_ref=%s token_state=%s",
                mode,
                model_ref,
                _token_state(token),
            )
            if not token:
                raise RuntimeError("Copilot token refresh 返回空 token")
            return _build_copilot_ide_headers() | {"Authorization": f"Bearer {token}"}
        _ensure_token = self.__dict__.get("_ensure_copilot_token")
        _request_headers = self.__dict__.get("_copilot_request_headers")
        if callable(_ensure_token) and callable(_request_headers):
            token = str(await cast("Callable[..., Awaitable[str]]", _ensure_token)(force_refresh=True)).strip()
            mode = str(getattr(self, "_provider_mode", "unknown"))
            model_ref = str(getattr(self, "model_ref", getattr(self, "_model", "unknown")))
            _log.debug(
                "[copilot.auth] source=fallback-refresh mode=%s model_ref=%s token_state=%s",
                mode,
                model_ref,
                _token_state(token),
            )
            if not token:
                raise RuntimeError("Copilot token refresh 返回空 token")
            return cast("Callable[[str], dict[str, str]]", _request_headers)(token)
        return await self._request_headers()

    # ── 向后兼容：copilot 方法透传（测试用）───────────────────────────

    @property
    def _copilot_api_base_url(self) -> str:
        backing = self.__dict__.get("_copilot_api_base_url_backing")
        if backing is not None:
            return backing
        return getattr(self.__dict__.get("_mode"), "_copilot_api_base_url", "")

    @_copilot_api_base_url.setter
    def _copilot_api_base_url(self, value: str) -> None:
        self.__dict__["_copilot_api_base_url_backing"] = value
        _m = self.__dict__.get("_mode")
        if _m is not None:
            _m._copilot_api_base_url = value

    async def _ensure_copilot_token(self, *, force_refresh: bool = False) -> str:
        if hasattr(self._mode, "_ensure_copilot_token"):
            return await self._mode._ensure_copilot_token(force_refresh=force_refresh)  # type: ignore[union-attr]
        raise RuntimeError("Not in copilot mode")

    def _uses_responses_api(self) -> bool:
        return self._model_api() == "responses"

    def _inject_completion_limits(self, payload: dict[str, Any]) -> None:
        """Inject completion limit param based on model spec (no _mode dependency)."""
        if "max_completion_tokens" in payload or "max_tokens" in payload:
            return
        spec = self._model_spec()
        req_params = spec.get("request_params") if spec else {}
        if not isinstance(req_params, dict):
            req_params = {}
        param_name = req_params.get("completion_limit_param")
        if not param_name:
            return
        max_tokens = spec.get("max_tokens")
        payload[param_name] = int(max_tokens) if max_tokens else _MAX_COMPLETION_TOKENS_DEFAULT

    def _build_responses_payload(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        thinking_override: str | None = None,
    ) -> dict[str, Any]:
        """Build responses API payload (no _mode dependency)."""
        spec = self._model_spec()
        level = thinking_override if thinking_override is not None else self._thinking_level
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str) and m.content.strip():
                    instructions_parts.append(m.content)
                continue
            input_items.append({
                "role": m.role,
                "content": _normalize_responses_message_content(m.content),
            })
        payload: dict[str, Any] = {
            "model": self._model,
            "input": input_items or [{"role": "user", "content": ""}],
        }
        t = temperature if temperature is not None else self._temperature
        unsupported = self._unsupported_request_params()
        if "temperature" not in unsupported:
            payload["temperature"] = t
        if instructions_parts:
            payload["instructions"] = "\n\n".join(instructions_parts)
        if spec.get("reasoning") and level != "off":
            payload["reasoning"] = {"effort": _copilot_reasoning_effort(level)}
        if self._extra_body:
            payload.update(self._extra_body)
        return payload

    def _copilot_compat_fallback_payload(self, *, base_payload: dict[str, Any], payload: dict[str, Any]) -> dict | None:
        # 去除 reasoning 字段：返回干净的 base_payload（无思考/无限制参数）
        if "reasoning_effort" in payload or "max_completion_tokens" in payload:
            return dict(base_payload)
        return None

    # ── thinking 注入 ──────────────────────────────────────────────────────

    def _inject_thinking(self, payload: dict[str, Any], level_override: str | None = None) -> None:
        """按 provider.mode 和 cfg.thinking 向 payload 注入 thinking 参数。"""
        level = level_override if level_override is not None else self._thinking_level
        spec = self._model_spec()

        if self._provider_mode == "openai":
            thinking_spec = spec.get("thinking") if spec else None
            if thinking_spec is None or level == "off":
                if level == "off":
                    payload["enable_thinking"] = False
                return
            budget = self._compute_budget(thinking_spec, level)
            payload["enable_thinking"] = True
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        elif self._provider_mode == "copilot":
            is_reasoning = bool(spec and spec.get("reasoning")) if spec else False
            if is_reasoning and level != "off":
                effort = "low" if level == "minimal" else level
                if self._model_api() == "chat_completions":
                    payload["reasoning_effort"] = effort
                    if "temperature" not in self._unsupported_request_params():
                        payload["temperature"] = 1

    @staticmethod
    def _compute_budget(thinking_spec: dict[str, Any], level: str) -> int:
        frac = _LEVEL_FRACS.get(level, 0.0)
        budget_max = thinking_spec.get("budget_max", 4096)
        budget_min = thinking_spec.get("budget_min", 1024)
        return max(int(budget_max * frac), budget_min)

    def _model_spec(self) -> dict[str, Any]:
        spec = lookup_model(self._model, catalog_path=getattr(self, "_catalog_path", None))
        return spec if isinstance(spec, dict) else {}

    def _record_usage(self, usage: dict | None) -> None:
        if not isinstance(usage, dict):
            return
        # 不同 OpenAI-compat / Copilot / 第三方网关对 usage 字段命名不一致。
        # 统一归一化到 prompt/completion/total，避免日志里 usage_prompt=0 误导预算与健康度判断。
        def _as_int(v: Any) -> int:
            try:
                return int(v or 0)
            except Exception:
                return 0

        native_present = any(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
        alias_present = any(key in usage for key in ("input_tokens", "output_tokens", "tokens"))
        usage_source = "native" if native_present else ("alias" if alias_present else "missing")

        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        total = _as_int(usage.get("total_tokens"))

        # 常见别名（部分代理/网关用 input/output）
        if prompt <= 0:
            prompt = _as_int(usage.get("input_tokens"))
        if completion <= 0:
            completion = _as_int(usage.get("output_tokens"))
        if total <= 0:
            total = _as_int(usage.get("tokens")) or (prompt + completion)

        self.last_usage = {
            "prompt_tokens": max(0, prompt),
            "completion_tokens": max(0, completion),
            "total_tokens": max(0, total),
            "usage_source": usage_source,
        }

    def _model_api(self) -> str:
        api = self._model_spec().get("api")
        return str(api) if isinstance(api, str) and api else "chat_completions"

    def _request_params_spec(self) -> dict[str, Any]:
        params = self._model_spec().get("request_params")
        return params if isinstance(params, dict) else {}

    def _unsupported_request_params(self) -> set[str]:
        raw = self._request_params_spec().get("unsupported")
        return {str(item) for item in raw} if isinstance(raw, list) else set()

    # ── chat ───────────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        """若 AsyncClient 已关闭（如因取消/网络中断），自动重建。"""
        if self._client.is_closed:
            self._client = self._mode.build_async_client()

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        thinking_override: str | None = None,
    ) -> str:
        self._ensure_client()
        temp = temperature if temperature is not None else self._temperature
        level = thinking_override if thinking_override is not None else self._thinking_level

        # responses API 路径（copilot 专属）
        if self._uses_responses_api():
            payload = self._build_responses_payload(
                messages, temperature=temp, thinking_override=thinking_override,
            )
            req_timeout = _request_timeout_override(self._client, level)
            target = self._resolve_url("/responses")

            headers = await self._request_headers()
            resp = await self._client.post(target, content=json.dumps(payload),
                                           headers=headers or None, timeout=req_timeout)
            # Copilot responses: on 400, refresh token and retry
            if resp.status_code == 400 and self._provider_mode == "copilot":
                headers = await self._copilot_refreshed_headers()
                resp = await self._client.post(target, content=json.dumps(payload),
                                               headers=headers or None, timeout=req_timeout)
            _raise_for_status_with_body(resp)
            data = resp.json()
            self._record_usage(data.get("usage"))
            return _extract_responses_text(data)

        # chat/completions 路径（openai + copilot 通用）
        base_payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temp,
        }
        payload = dict(base_payload)
        self._inject_thinking(payload, level_override=thinking_override)
        self._inject_completion_limits(payload)
        if self._extra_body:
            payload.update(self._extra_body)

        req_timeout = _request_timeout_override(self._client, level)
        target = self._resolve_url("/chat/completions")

        headers = await self._request_headers()
        resp = await self._client.post(target, content=json.dumps(payload),
                                       headers=headers or None, timeout=req_timeout)

        # Copilot: on 400, refresh token and retry; still 400 → fallback without reasoning
        if resp.status_code == 400 and self._provider_mode == "copilot":
            headers = await self._copilot_refreshed_headers()
            resp = await self._client.post(target, content=json.dumps(payload),
                                           headers=headers or None, timeout=req_timeout)
            if resp.status_code == 400:
                fallback = self._copilot_compat_fallback_payload(base_payload=base_payload, payload=payload)
                if fallback is not None:
                    resp = await self._client.post(target, content=json.dumps(fallback),
                                                   headers=headers or None, timeout=req_timeout)

        _raise_for_status_with_body(resp)
        data = resp.json()
        self._record_usage(data.get("usage"))
        choices = data.get("choices") or []
        if not choices:
            finish = (data.get("choices") or [{}])[0].get("finish_reason") if data.get("choices") else None
            raise RuntimeError(
                "API 返回空 choices（可能触发内容过滤或限流）"
                + (f"，finish_reason={finish}" if finish else "")
            )
        msg = choices[0]["message"]
        content: str = msg.get("content") or ""
        # 不在 provider 层机械剥离 <think>...</think>。
        # 若下游只需要 JSON，会在解析层做“仅用于解析”的局部清洗；保留原文有利于感知与取证。
        return content

    async def close(self) -> None:
        await self._client.aclose()
        self._sync_client.close()

    async def ping(self, timeout: float = 8.0) -> tuple[bool, int, str | None]:  # noqa: ASYNC109
        """连通性探测：根据模型 spec 选择正确端点，返回 (success, latency_ms, error_or_None)。"""
        import time as _time
        _t0 = _time.monotonic()
        try:
            headers = await self._request_headers()
            if self._uses_responses_api():
                target = self._resolve_url("/responses")
                payload: dict[str, Any] = {
                    "model": self._model,
                    "input": "ping",
                    "max_output_tokens": 1,
                }
            else:
                target = self._resolve_url("/chat/completions")
                payload = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
            resp = await self._client.post(
                target,
                content=json.dumps(payload),
                headers=headers or None,
                timeout=timeout,
            )
            _ms = int((_time.monotonic() - _t0) * 1000)
            if resp.status_code in (200, 201):
                return True, _ms, None
            if resp.status_code in (401, 403):
                return False, _ms, f"认证失败 (HTTP {resp.status_code})"
            return False, _ms, f"HTTP {resp.status_code}"
        except Exception as _e:
            _ms = int((_time.monotonic() - _t0) * 1000)
            return False, _ms, str(_e)

    def embed(self, text: str) -> list[float]:
        if not self._embed_model:
            raise RuntimeError("embedding_model not configured")

        headers = {}
        if self._provider_mode == "copilot":
            cache = load_copilot_token_cache()
            if not cache or (time.time() * 1000) >= cache.expires_at_ms - 300_000:
                raise RuntimeError(
                    "Copilot embeddings 需要先完成 GitHub token → Copilot token exchange。\n"
                    "请先执行一次 chat 请求，或关闭 embedding_model。"
                )
            token = str(cache.token or "").strip()
            mode = str(getattr(self, "_provider_mode", "unknown"))
            model_ref = str(getattr(self, "model_ref", getattr(self, "_model", "unknown")))
            _log.debug(
                "[copilot.auth] source=embed-cache mode=%s model_ref=%s token_state=%s",
                mode,
                model_ref,
                _token_state(token),
            )
            if not token:
                raise RuntimeError("Copilot token 缓存为空，无法调用 embeddings")
            headers = _build_copilot_ide_headers() | {"Authorization": f"Bearer {token}"}

        target = self._resolve_url(self._mode.embedding_url())
        resp = self._sync_client.post(
            target,
            content=json.dumps({
                "model": self._embed_model,
                "input": [text[:_EMBED_MAX_CHARS]],
            }),
            headers=headers or None,
        )
        _raise_for_status_with_body(resp)
        return resp.json()["data"][0]["embedding"]
