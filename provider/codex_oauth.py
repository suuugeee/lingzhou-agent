"""OpenAI Codex OAuth helpers.

This module owns the provider-specific OpenAI Codex OAuth protocol:
device authorization, token refresh, runtime token resolution, and
ChatGPT/Codex backend headers. The generic auth store remains in store.auth.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from store.auth import TokenResolution, get_auth_profile, set_oauth_profile

CODEX_AUTH_BASE_URL = "https://auth.openai.com"
CODEX_PROFILE_ID = "openai-codex:default"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = f"{CODEX_AUTH_BASE_URL}/oauth/token"
CODEX_DEVICE_CALLBACK_URL = f"{CODEX_AUTH_BASE_URL}/deviceauth/callback"
CODEX_DEVICE_VERIFICATION_URL = f"{CODEX_AUTH_BASE_URL}/codex/device"
DEFAULT_CODEX_API_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
CODEX_DEVICE_AUTH_TIMEOUT_SECONDS = 15 * 60


@dataclass(frozen=True)
class CodexDeviceCode:
    user_code: str
    device_auth_id: str
    verification_url: str
    interval_s: int


@dataclass(frozen=True)
class CodexDeviceAuthorization:
    authorization_code: str
    code_verifier: str


def _auth_headers(content_type: str) -> dict[str, str]:
    return {
        "Content-Type": content_type,
        "originator": "lingzhou",
        "User-Agent": "lingzhou",
    }


def request_codex_device_code(timeout: float = 15.0) -> CodexDeviceCode:
    resp = httpx.post(
        f"{CODEX_AUTH_BASE_URL}/api/accounts/deviceauth/usercode",
        json={"client_id": CODEX_OAUTH_CLIENT_ID},
        headers=_auth_headers("application/json"),
        timeout=timeout,
    )
    if resp.status_code != 200:
        detail = (resp.text or "").strip()
        raise RuntimeError(f"OpenAI device code request failed: HTTP {resp.status_code}" + (f" {detail}" if detail else ""))
    data = resp.json()
    user_code = str(data.get("user_code") or data.get("usercode") or "").strip()
    device_auth_id = str(data.get("device_auth_id") or "").strip()
    interval_s = max(1, int(data.get("interval") or 5))
    if not user_code or not device_auth_id:
        raise RuntimeError("OpenAI device code response missing user_code/device_auth_id")
    return CodexDeviceCode(
        user_code=user_code,
        device_auth_id=device_auth_id,
        verification_url=CODEX_DEVICE_VERIFICATION_URL,
        interval_s=interval_s,
    )


def poll_codex_device_authorization(
    device: CodexDeviceCode,
    *,
    timeout_seconds: int = CODEX_DEVICE_AUTH_TIMEOUT_SECONDS,
    request_timeout: float = 15.0,
    on_waiting: Any = None,
) -> CodexDeviceAuthorization:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(device.interval_s)
        poll = httpx.post(
            f"{CODEX_AUTH_BASE_URL}/api/accounts/deviceauth/token",
            json={"device_auth_id": device.device_auth_id, "user_code": device.user_code},
            headers=_auth_headers("application/json"),
            timeout=request_timeout,
        )
        if poll.status_code == 200:
            data = poll.json()
            authorization_code = str(data.get("authorization_code") or "").strip()
            code_verifier = str(data.get("code_verifier") or "").strip()
            if not authorization_code or not code_verifier:
                raise RuntimeError("OpenAI device authorization response missing authorization_code/code_verifier")
            return CodexDeviceAuthorization(
                authorization_code=authorization_code,
                code_verifier=code_verifier,
            )
        if poll.status_code in (403, 404):
            if callable(on_waiting):
                on_waiting()
            continue
        detail = (poll.text or "").strip()
        raise RuntimeError(
            f"OpenAI device authorization failed: HTTP {poll.status_code}"
            + (f" {detail}" if detail else "")
        )
    raise TimeoutError("OpenAI Codex device authorization timed out")


def exchange_codex_device_authorization(
    authorization: CodexDeviceAuthorization,
    *,
    timeout: float = 15.0,
) -> dict[str, Any]:
    resp = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": authorization.authorization_code,
            "redirect_uri": CODEX_DEVICE_CALLBACK_URL,
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": authorization.code_verifier,
        },
        headers=_auth_headers("application/x-www-form-urlencoded"),
        timeout=timeout,
    )
    if resp.status_code != 200:
        detail = (resp.text or "").strip()
        raise RuntimeError(f"OpenAI token exchange failed: HTTP {resp.status_code}" + (f" {detail}" if detail else ""))
    tokens = resp.json()
    access_token = str(tokens.get("access_token") or "").strip()
    refresh_token = str(tokens.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        raise RuntimeError("OpenAI token exchange succeeded but did not return complete OAuth tokens")
    return tokens


def save_codex_oauth_tokens(tokens: dict[str, Any], *, path: Path | None = None) -> None:
    set_oauth_profile(
        profile_id=CODEX_PROFILE_ID,
        provider="openai-codex",
        tokens=tokens,
        auth_mode="device_code",
        path=path,
    )


def _jwt_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not access_token or "." not in access_token:
        return False
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= time.time() + max(0, int(skew_seconds))
    except Exception:
        return False


def _codex_env_token() -> TokenResolution | None:
    env_token = os.environ.get("OPENAI_CODEX_ACCESS_TOKEN", "").strip()
    return TokenResolution(token=env_token, source="env:OPENAI_CODEX_ACCESS_TOKEN") if env_token else None


def refresh_codex_oauth_tokens(
    tokens: dict[str, Any],
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise OSError("OpenAI Codex OAuth 缺少 refresh_token，请重新执行 `lingzhou auth login-codex`。")

    with httpx.Client(timeout=max(5.0, float(timeout_seconds)), headers={"Accept": "application/json"}) as client:
        resp = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )
    if resp.status_code != 200:
        detail = (resp.text or "").strip().replace("\n", " ")
        raise OSError(
            f"OpenAI Codex token refresh 失败 (HTTP {resp.status_code})"
            + (f": {detail}" if detail else "")
            + "。请重新执行 `lingzhou auth login-codex`。"
        )

    payload = resp.json()
    access_token = str(payload.get("access_token", "") or "").strip()
    if not access_token:
        raise OSError("OpenAI Codex token refresh 成功但未返回 access_token。")
    updated = dict(tokens)
    updated["access_token"] = access_token
    refreshed = str(payload.get("refresh_token", "") or "").strip()
    if refreshed:
        updated["refresh_token"] = refreshed
    if payload.get("expires_in") is not None:
        updated["expires_in"] = payload.get("expires_in")
    return updated


def resolve_codex_oauth_token(
    *,
    profile_id: str = CODEX_PROFILE_ID,
    refresh_if_expiring: bool = True,
    force_refresh: bool = False,
    path: Path | None = None,
) -> TokenResolution | None:
    profile = get_auth_profile(profile_id, path)
    if not isinstance(profile, dict):
        return _codex_env_token()

    tokens = profile.get("tokens")
    if not isinstance(tokens, dict):
        return _codex_env_token()
    access_token = str(tokens.get("access_token", "") or "").strip()
    if not access_token:
        return _codex_env_token()

    if force_refresh or (refresh_if_expiring and _jwt_expiring(access_token, CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS)):
        updated = refresh_codex_oauth_tokens(tokens)
        set_oauth_profile(
            profile_id=profile_id,
            provider="openai-codex",
            tokens=updated,
            auth_mode=str(profile.get("auth_mode") or "oauth"),
            path=path,
        )
        access_token = str(updated.get("access_token", "") or "").strip()

    return TokenResolution(token=access_token, source="auth-profile", profile_id=profile_id)


def build_codex_headers(access_token: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "codex_cli_rs/0.0.0 (Lingzhou)",
        "originator": "codex_cli_rs",
    }
    try:
        parts = access_token.split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
            account_id = str(payload.get("chatgpt_account_id") or "").strip()
            if account_id:
                headers["ChatGPT-Account-ID"] = account_id
    except Exception:
        pass
    return headers
